"""Immutable contracts for transport-to-fusion QA integration (Phase 4.5-A).

These contracts describe delivery decisions and evidence. They do not schedule
packets, adapt measurements, or execute the Fusion Engine.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum


class IntegrationContractError(ValueError):
    """Raised when a Fault-to-QA integration contract is invalid."""


class TransportOrderingPolicy(str, Enum):
    ARRIVAL_ORDER = "arrival_order"
    ACQUISITION_ORDER = "acquisition_order"


class DeliveryDisposition(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    LATE = "late"
    INVALID = "invalid"
    DROPPED = "dropped"


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationContractError(f"{name} must be a non-empty string")
    return value.strip()


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegrationContractError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: str, name: str) -> str:
    value = _text(value, name).lower()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise IntegrationContractError(f"{name} must be a 64-character SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class EnvelopeIdentity:
    """Stable source identity; sequence numbers are scoped to a sensor/source."""

    sensor_id: str
    sequence_number: int
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_id", _text(self.sensor_id, "sensor_id").lower())
        object.__setattr__(self, "sequence_number", _count(self.sequence_number, "sequence_number"))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))

    @property
    def key(self) -> tuple[str, str | None, int]:
        return (self.sensor_id, self.source_id, self.sequence_number)


@dataclass(frozen=True)
class DeliveryDecision:
    """Disposition of one delivered packet or one source packet proven dropped.

    Non-finite raw timestamps are retained only for INVALID/DROPPED evidence.
    A packet marked ACCEPTED must have finite acquisition/arrival clocks and,
    when present, a finite processing clock not earlier than arrival.
    """

    identity: EnvelopeIdentity
    disposition: DeliveryDisposition
    acquisition_time: float | None
    arrival_time: float | None
    processing_time: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EnvelopeIdentity):
            raise IntegrationContractError("identity must be an EnvelopeIdentity")
        if not isinstance(self.disposition, DeliveryDisposition):
            raise IntegrationContractError("disposition must be a DeliveryDisposition")
        for name in ("acquisition_time", "arrival_time", "processing_time"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise IntegrationContractError(f"{name} must be numeric or None")
                object.__setattr__(self, name, float(value))
        if self.reason is not None:
            object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if self.disposition is DeliveryDisposition.ACCEPTED:
            if self.acquisition_time is None or self.arrival_time is None:
                raise IntegrationContractError("ACCEPTED requires acquisition and arrival timestamps")
            if not math.isfinite(self.acquisition_time) or not math.isfinite(self.arrival_time):
                raise IntegrationContractError("ACCEPTED timestamps must be finite")
            if self.processing_time is not None:
                if not math.isfinite(self.processing_time):
                    raise IntegrationContractError("ACCEPTED processing_time must be finite")
                if self.processing_time < self.arrival_time:
                    raise IntegrationContractError("ACCEPTED processing_time cannot precede arrival_time")
        if self.disposition is not DeliveryDisposition.ACCEPTED and self.reason is None:
            raise IntegrationContractError("non-accepted decisions require an explicit reason")
        if self.disposition is DeliveryDisposition.DROPPED:
            if self.arrival_time is not None or self.processing_time is not None:
                raise IntegrationContractError("DROPPED evidence cannot claim arrival or processing")


@dataclass(frozen=True)
class TransportReplaySummary:
    """Auditable accounting for one transport replay.

    Decisions contain one record per delivered packet (including duplicates)
    plus one DROPPED record per source packet that never arrived.
    """

    run_id: str
    ordering_policy: TransportOrderingPolicy
    source_count: int
    delivered_count: int
    accepted_count: int
    duplicate_count: int
    late_count: int
    invalid_count: int
    dropped_count: int
    source_hash: str
    delivered_hash: str
    configuration_hash: str
    decisions: tuple[DeliveryDecision, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        if not isinstance(self.ordering_policy, TransportOrderingPolicy):
            raise IntegrationContractError("ordering_policy must be a TransportOrderingPolicy")
        for name in (
            "source_count", "delivered_count", "accepted_count", "duplicate_count",
            "late_count", "invalid_count", "dropped_count",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        for name in ("source_hash", "delivered_hash", "configuration_hash"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        decisions = tuple(self.decisions)
        if any(not isinstance(item, DeliveryDecision) for item in decisions):
            raise IntegrationContractError("decisions must contain DeliveryDecision instances")
        object.__setattr__(self, "decisions", decisions)

        delivered_accounted = (
            self.accepted_count + self.duplicate_count
            + self.late_count + self.invalid_count
        )
        if delivered_accounted != self.delivered_count:
            raise IntegrationContractError("delivered_count must equal accepted + duplicate + late + invalid")
        if self.dropped_count > self.source_count:
            raise IntegrationContractError("dropped_count cannot exceed source_count")
        if len(decisions) != self.delivered_count + self.dropped_count:
            raise IntegrationContractError("decisions must account for every delivery and dropped source packet")

        observed = {item: 0 for item in DeliveryDisposition}
        for item in decisions:
            observed[item.disposition] += 1
        expected = {
            DeliveryDisposition.ACCEPTED: self.accepted_count,
            DeliveryDisposition.DUPLICATE: self.duplicate_count,
            DeliveryDisposition.LATE: self.late_count,
            DeliveryDisposition.INVALID: self.invalid_count,
            DeliveryDisposition.DROPPED: self.dropped_count,
        }
        if observed != expected:
            raise IntegrationContractError("decision dispositions do not match summary counts")
