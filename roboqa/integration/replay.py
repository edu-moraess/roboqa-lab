"""Deterministic offline transport replay (Phase 4.5-B).

Replay is deliberately separate from Fusion execution. It classifies deliveries,
suppresses repeated source identities, accounts for drops, and emits accepted
envelopes in the selected transport/event-time order.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.integration.contracts import (
    DeliveryDecision,
    DeliveryDisposition,
    EnvelopeIdentity,
    IntegrationContractError,
    TransportOrderingPolicy,
    TransportReplaySummary,
)


@dataclass(frozen=True)
class TransportReplayResult:
    """Immutable replay output; only ACCEPTED envelopes are emitted downstream."""

    summary: TransportReplaySummary
    accepted_envelopes: tuple[RawMeasurementEnvelope, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.summary, TransportReplaySummary):
            raise IntegrationContractError("summary must be a TransportReplaySummary")
        values = tuple(self.accepted_envelopes)
        if any(not isinstance(item, RawMeasurementEnvelope) for item in values):
            raise IntegrationContractError("accepted_envelopes must contain raw envelopes")
        if len(values) != self.summary.accepted_count:
            raise IntegrationContractError("accepted envelope count must match summary")
        object.__setattr__(self, "accepted_envelopes", values)


def _identity(envelope: RawMeasurementEnvelope) -> EnvelopeIdentity:
    return EnvelopeIdentity(
        sensor_id=envelope.sensor_id,
        source_id=envelope.source_id,
        sequence_number=envelope.sequence_number,
    )


def _portable(value):
    """Encode raw NaN/Inf and immutable containers into canonical JSON values."""
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "positive_inf" if value > 0 else "negative_inf"}
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return value


def _envelope_document(item: RawMeasurementEnvelope) -> dict:
    return {
        "sensor_id": item.sensor_id.strip().lower(),
        "source_id": item.source_id,
        "sequence_number": item.sequence_number,
        "acquisition_time": _portable(item.acquisition_time),
        "arrival_time": _portable(item.arrival_time),
        "processing_time": _portable(item.processing_time),
        "payload": _portable(item.payload),
    }


def _identity_sort_key(item: RawMeasurementEnvelope) -> tuple[str, str, int]:
    identity = _identity(item)
    return (identity.sensor_id, identity.source_id or "", identity.sequence_number)


def _fingerprint(item: RawMeasurementEnvelope) -> str:
    encoded = json.dumps(
        _envelope_document(item), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _collection_hash(items: Sequence[RawMeasurementEnvelope]) -> str:
    documents = [_envelope_document(item) for item in items]
    encoded = json.dumps(
        documents, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_candidate(item: RawMeasurementEnvelope) -> tuple[bool, str | None]:
    if not item.timestamps_finite:
        return False, "non-finite acquisition_time or arrival_time"
    if not item.processing_timestamp_finite:
        return False, "non-finite processing_time"
    if not item.payload_finite:
        return False, "non-finite payload"
    if item.arrival_time < item.acquisition_time:
        return False, "arrival_time precedes acquisition_time"
    if item.processing_time is not None and item.processing_time < item.arrival_time:
        return False, "processing_time precedes arrival_time"
    return True, None


def replay_transport(
    source_envelopes: Sequence[RawMeasurementEnvelope],
    delivered_envelopes: Sequence[RawMeasurementEnvelope],
    *,
    ordering_policy: TransportOrderingPolicy = TransportOrderingPolicy.ARRIVAL_ORDER,
    allowed_lateness_s: float = 0.0,
    run_id: str | None = None,
) -> TransportReplayResult:
    """Replay a delivered batch against its original source snapshot.

    ARRIVAL_ORDER sorts by arrival clock and applies an acquisition-time
    watermark: watermark = max_acquisition_seen - allowed_lateness_s. A first
    delivery strictly older than the watermark is LATE. Equality is accepted.
    ACQUISITION_ORDER sorts by acquisition clock and therefore has no late
    classification in this offline batch mode.

    The first delivery for an identity wins; every later copy is DUPLICATE,
    even if the first copy was invalid or late. Invalid payloads/clocks are
    rejected before watermark classification. Missing source identities are
    recorded as DROPPED. Unknown delivered identities and repeated identities
    in the source snapshot are rejected as ambiguous evidence.

    This function does not call run_filter or claim that Fusion consumes the
    returned order. Fusion's existing runner independently sorts Measurement.t.
    """
    if not isinstance(ordering_policy, TransportOrderingPolicy):
        raise IntegrationContractError("ordering_policy must be a TransportOrderingPolicy")
    if isinstance(allowed_lateness_s, bool):
        raise IntegrationContractError("allowed_lateness_s must be finite and >= 0")
    try:
        lateness = float(allowed_lateness_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrationContractError("allowed_lateness_s must be finite and >= 0") from exc
    if not math.isfinite(lateness) or lateness < 0.0:
        raise IntegrationContractError("allowed_lateness_s must be finite and >= 0")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise IntegrationContractError("run_id must be a non-empty string or None")
    if not isinstance(source_envelopes, Sequence) or isinstance(source_envelopes, (str, bytes)):
        raise IntegrationContractError("source_envelopes must be a sequence")
    if not isinstance(delivered_envelopes, Sequence) or isinstance(delivered_envelopes, (str, bytes)):
        raise IntegrationContractError("delivered_envelopes must be a sequence")
    source = tuple(source_envelopes)
    delivered = tuple(delivered_envelopes)
    if any(not isinstance(item, RawMeasurementEnvelope) for item in source + delivered):
        raise IntegrationContractError("all replay inputs must be RawMeasurementEnvelope instances")

    source_by_key: dict[tuple[str, str | None, int], RawMeasurementEnvelope] = {}
    for item in source:
        key = _identity(item).key
        if key in source_by_key:
            raise IntegrationContractError("source snapshot must have unique envelope identities")
        source_by_key[key] = item
    for item in delivered:
        if _identity(item).key not in source_by_key:
            raise IntegrationContractError("delivered envelope identity is absent from source snapshot")

    clock = "arrival_time" if ordering_policy is TransportOrderingPolicy.ARRIVAL_ORDER else "acquisition_time"
    ordered = sorted(
        delivered,
        key=lambda item: (
            getattr(item, clock) if math.isfinite(getattr(item, clock)) else math.inf,
            item.sensor_id.strip().lower(),
            item.source_id or "",
            item.sequence_number,
            _fingerprint(item),
        ),
    )

    seen: set[tuple[str, str | None, int]] = set()
    decisions: list[DeliveryDecision] = []
    accepted: list[RawMeasurementEnvelope] = []
    counts = {disposition: 0 for disposition in DeliveryDisposition}
    max_acquisition_seen = -math.inf

    for item in ordered:
        identity = _identity(item)
        key = identity.key
        if key in seen:
            disposition = DeliveryDisposition.DUPLICATE
            reason = "source identity already observed in this replay"
        else:
            seen.add(key)
            valid, invalid_reason = _valid_candidate(item)
            if not valid:
                disposition = DeliveryDisposition.INVALID
                reason = invalid_reason
            else:
                max_acquisition_seen = max(max_acquisition_seen, item.acquisition_time)
                watermark = max_acquisition_seen - lateness
                if (
                    ordering_policy is TransportOrderingPolicy.ARRIVAL_ORDER
                    and item.acquisition_time < watermark
                ):
                    disposition = DeliveryDisposition.LATE
                    reason = (
                        f"acquisition_time {item.acquisition_time} is older than "
                        f"watermark {watermark}"
                    )
                else:
                    disposition = DeliveryDisposition.ACCEPTED
                    reason = None
                    accepted.append(item)

        counts[disposition] += 1
        decisions.append(DeliveryDecision(
            identity=identity,
            disposition=disposition,
            acquisition_time=item.acquisition_time,
            arrival_time=item.arrival_time,
            processing_time=item.processing_time,
            reason=reason,
        ))

    dropped = [
        (key, item) for key, item in source_by_key.items() if key not in seen
    ]
    for _, item in sorted(dropped, key=lambda pair: (
        pair[0][0], pair[0][1] or "", pair[0][2]
    )):
        decisions.append(DeliveryDecision(
            identity=_identity(item),
            disposition=DeliveryDisposition.DROPPED,
            acquisition_time=item.acquisition_time,
            arrival_time=None,
            processing_time=None,
            reason="source identity absent from delivered batch",
        ))
    counts[DeliveryDisposition.DROPPED] = len(dropped)

    source_hash = _collection_hash(sorted(source, key=lambda item: (
        _identity_sort_key(item), _fingerprint(item)
    )))
    delivered_hash = _collection_hash(ordered)
    configuration = {
        "ordering_policy": ordering_policy.value,
        "allowed_lateness_s": lateness,
        "duplicate_policy": "first_delivery_wins",
        "late_policy": "arrival_order_acquisition_watermark",
        "invalid_policy": "reject_nonfinite_or_nonmonotonic_clocks_and_payload",
    }
    configuration_hash = hashlib.sha256(json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    resolved_run_id = run_id.strip() if run_id is not None else (
        f"replay-{source_hash[:12]}-{delivered_hash[:12]}-{configuration_hash[:8]}"
    )

    summary = TransportReplaySummary(
        run_id=resolved_run_id,
        ordering_policy=ordering_policy,
        source_count=len(source),
        delivered_count=len(delivered),
        accepted_count=counts[DeliveryDisposition.ACCEPTED],
        duplicate_count=counts[DeliveryDisposition.DUPLICATE],
        late_count=counts[DeliveryDisposition.LATE],
        invalid_count=counts[DeliveryDisposition.INVALID],
        dropped_count=counts[DeliveryDisposition.DROPPED],
        source_hash=source_hash,
        delivered_hash=delivered_hash,
        configuration_hash=configuration_hash,
        decisions=tuple(decisions),
    )
    return TransportReplayResult(summary=summary, accepted_envelopes=tuple(accepted))
