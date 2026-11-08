"""Deterministic event-time schedule for replay-accepted Fusion measurements."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.integration.contracts import (
    DeliveryDisposition,
    EnvelopeIdentity,
    TransportReplaySummary,
)
from roboqa.integration.measurement_adapter import AdaptedMeasurement
from roboqa.integration.replay import TransportReplayResult


class FusionSchedulingError(ValueError):
    """Raised when replay and adapted measurements cannot form a safe schedule."""


class FusionSchedulingPolicy(str, Enum):
    """Supported execution order for the current monotonic EKF."""

    ACQUISITION_TIME = "acquisition_time_then_sensor_identity"


def _identity_for(envelope: RawMeasurementEnvelope) -> EnvelopeIdentity:
    return EnvelopeIdentity(
        sensor_id=envelope.sensor_id,
        source_id=envelope.source_id,
        sequence_number=envelope.sequence_number,
    )


def _identity_key(identity: EnvelopeIdentity) -> tuple[str, str, int]:
    return (identity.sensor_id, identity.source_id or "", identity.sequence_number)


def _entry_key(item: AdaptedMeasurement) -> tuple[float, str, str, int]:
    identity = item.identity
    return (
        float(item.measurement.t),
        item.measurement.sensor.strip().lower(),
        identity.source_id or "",
        identity.sequence_number,
    )


def _schedule_document(summary, entries, policy) -> dict:
    return {
        "run_id": summary.run_id,
        "policy": policy.value,
        "source_hash": summary.source_hash,
        "delivered_hash": summary.delivered_hash,
        "replay_configuration_hash": summary.configuration_hash,
        "entries": [
            {
                "sensor_id": item.identity.sensor_id,
                "source_id": item.identity.source_id,
                "sequence_number": item.identity.sequence_number,
                "acquisition_time": float(item.measurement.t),
                "arrival_time": float(item.arrival_time),
                "processing_time": (
                    None if item.processing_time is None else float(item.processing_time)
                ),
                "z": item.measurement.z.tolist(),
                "R": item.measurement.R.tolist(),
            }
            for item in entries
        ],
    }


def _schedule_digest(summary, entries, policy) -> str:
    encoded = json.dumps(
        _schedule_document(summary, entries, policy),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FusionSchedule:
    """Immutable, provenance-linked sequence for monotonic Fusion execution."""

    replay_summary: TransportReplaySummary
    entries: tuple[AdaptedMeasurement, ...]
    schedule_hash: str
    policy: FusionSchedulingPolicy = FusionSchedulingPolicy.ACQUISITION_TIME

    def __post_init__(self) -> None:
        if not isinstance(self.replay_summary, TransportReplaySummary):
            raise FusionSchedulingError("replay_summary must be a TransportReplaySummary")
        if self.policy is not FusionSchedulingPolicy.ACQUISITION_TIME:
            raise FusionSchedulingError("unsupported Fusion scheduling policy")
        entries = tuple(self.entries)
        if any(not isinstance(item, AdaptedMeasurement) for item in entries):
            raise FusionSchedulingError("entries must contain AdaptedMeasurement instances")
        keys = [_identity_key(item.identity) for item in entries]
        if len(set(keys)) != len(keys):
            raise FusionSchedulingError("schedule identities must be unique")
        accepted_keys = {
            _identity_key(decision.identity)
            for decision in self.replay_summary.decisions
            if decision.disposition is DeliveryDisposition.ACCEPTED
        }
        if set(keys) != accepted_keys:
            raise FusionSchedulingError(
                "schedule identities must match replay-accepted identities"
            )
        entry_order = tuple(_entry_key(item) for item in entries)
        if entry_order != tuple(sorted(entry_order)):
            raise FusionSchedulingError("entries must be sorted by acquisition time and identity")
        if len(entries) != self.replay_summary.accepted_count:
            raise FusionSchedulingError("schedule entry count must match replay accepted_count")
        if not isinstance(self.schedule_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.schedule_hash
        ) is None:
            raise FusionSchedulingError("schedule_hash must be a SHA-256 hex digest")
        expected = _schedule_digest(self.replay_summary, entries, self.policy)
        if self.schedule_hash != expected:
            raise FusionSchedulingError("schedule_hash does not match schedule evidence")
        object.__setattr__(self, "entries", entries)

    @property
    def measurements(self) -> tuple:
        """Normalized Measurement objects in deterministic Fusion execution order."""
        return tuple(item.measurement for item in self.entries)

    @property
    def run_id(self) -> str:
        return self.replay_summary.run_id


def build_fusion_schedule(
    replay: TransportReplayResult,
    adapted_measurements: Sequence[AdaptedMeasurement],
) -> FusionSchedule:
    """Validate adapter output and create a deterministic event-time schedule.

    Replay may have produced accepted envelopes in ARRIVAL_ORDER. This function
    reorders them by acquisition_time, normalized sensor ID, source_id, and
    sequence_number. The current EKF requires monotonic event time and has no
    rewind/reprocessing capability, so arrival-order execution is not offered.

    Every accepted envelope must have exactly one matching adapter result.
    Identity, acquisition/arrival/processing clocks, sensor, and payload are
    checked before scheduling. Explicit R remains caller-supplied adapter data
    and is included in the schedule hash.
    """
    if not isinstance(replay, TransportReplayResult):
        raise FusionSchedulingError("replay must be a TransportReplayResult")
    if not isinstance(adapted_measurements, Sequence) or isinstance(
        adapted_measurements, (str, bytes)
    ):
        raise FusionSchedulingError("adapted_measurements must be a sequence")

    adapted = tuple(adapted_measurements)
    if any(not isinstance(item, AdaptedMeasurement) for item in adapted):
        raise FusionSchedulingError(
            "adapted_measurements must contain AdaptedMeasurement instances"
        )

    by_identity: dict[tuple[str, str, int], AdaptedMeasurement] = {}
    for item in adapted:
        key = _identity_key(item.identity)
        if key in by_identity:
            raise FusionSchedulingError("adapted measurement identities must be unique")
        by_identity[key] = item

    accepted = replay.accepted_envelopes
    expected_keys = {_identity_key(_identity_for(item)) for item in accepted}
    if set(by_identity) != expected_keys or len(adapted) != len(accepted):
        raise FusionSchedulingError(
            "adapter output must contain exactly one record for every replay-accepted envelope"
        )

    validated: list[AdaptedMeasurement] = []
    for envelope in accepted:
        identity = _identity_for(envelope)
        item = by_identity[_identity_key(identity)]
        measurement = item.measurement
        if measurement.sensor.strip().lower() != identity.sensor_id:
            raise FusionSchedulingError("adapted sensor does not match replay envelope")
        if not math.isclose(
            float(measurement.t), float(envelope.acquisition_time), rel_tol=0.0, abs_tol=0.0
        ):
            raise FusionSchedulingError("Measurement.t must equal envelope acquisition_time")
        if not math.isclose(
            float(item.arrival_time), float(envelope.arrival_time), rel_tol=0.0, abs_tol=0.0
        ):
            raise FusionSchedulingError("adapted arrival_time does not match replay envelope")
        if item.processing_time != envelope.processing_time:
            raise FusionSchedulingError("adapted processing_time does not match replay envelope")
        try:
            expected_payload = np.asarray(envelope.payload, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FusionSchedulingError("accepted replay payload cannot be normalized") from exc
        if not np.array_equal(measurement.z, expected_payload):
            raise FusionSchedulingError("Measurement.z does not match replay payload")
        validated.append(item)

    entries = tuple(sorted(validated, key=_entry_key))
    policy = FusionSchedulingPolicy.ACQUISITION_TIME
    digest = _schedule_digest(replay.summary, entries, policy)
    return FusionSchedule(
        replay_summary=replay.summary,
        entries=entries,
        schedule_hash=digest,
        policy=policy,
    )
