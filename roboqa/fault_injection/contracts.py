"""Immutable contracts for deterministic fault-injection experiments.

Phase 3.1 intentionally defines contracts only; it does not mutate measurements
or execute fault operators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


class ContractError(ValueError):
    """Raised when a fault-injection contract is invalid."""


class FaultType(str, Enum):
    SENSOR_DROPOUT = "sensor_dropout"
    INTERMITTENT_OUTAGE = "intermittent_outage"
    PACKET_LOSS = "packet_loss"
    OUTLIER = "outlier"
    STUCK_AT = "stuck_at"
    BIAS = "bias"
    BIAS_DRIFT = "bias_drift"
    INVALID_PAYLOAD = "invalid_payload"
    EXCESSIVE_NOISE = "excessive_noise"
    DELAY = "delay"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    TIMESTAMP_DRIFT = "timestamp_drift"
    IRREGULAR_ARRIVAL = "irregular_arrival"
    PROGRESSIVE_NOISE = "progressive_noise"
    PROGRESSIVE_DROPOUT = "progressive_dropout"
    INCREASING_BIAS = "increasing_bias"
    REDUCED_UPDATE_FREQUENCY = "reduced_update_frequency"


class FaultStatus(str, Enum):
    APPLIED = "APPLIED"
    PARTIALLY_APPLIED = "PARTIALLY_APPLIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REJECTED = "REJECTED"


class RecoveryStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RECOVERING = "RECOVERING"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _freeze_json(value: Any, path: str = "value") -> Any:
    """Freeze JSON-like values and reject non-portable or non-finite parameters."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{path} must not contain NaN or Inf")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise ContractError(f"{path} contains unsupported value type: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return a JSON-serializable copy of a recursively frozen value."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class FaultScenario:
    scenario_id: str
    fault_type: FaultType
    target_sensors: tuple[str, ...]
    start_time: float
    end_time: float
    seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _non_empty(self.scenario_id, "scenario_id"))
        if not isinstance(self.fault_type, FaultType):
            raise ContractError("fault_type must be a FaultType")
        sensors = tuple(_non_empty(sensor, "target_sensors item") for sensor in self.target_sensors)
        if not sensors or len(set(sensors)) != len(sensors):
            raise ContractError("target_sensors must be non-empty and unique")
        object.__setattr__(self, "target_sensors", sensors)
        start = _finite(self.start_time, "start_time")
        end = _finite(self.end_time, "end_time")
        if start >= end:
            raise ContractError("scenario interval must satisfy start_time < end_time")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError("seed must be an integer")
        if not 0 <= self.seed <= (2**64 - 1):
            raise ContractError("seed must be in [0, 2**64 - 1]")
        if not isinstance(self.parameters, Mapping):
            raise ContractError("parameters must be a mapping")
        object.__setattr__(self, "parameters", _freeze_json(self.parameters, "parameters"))

    def contains(self, acquisition_time: float) -> bool:
        """The scenario interval is start-inclusive and end-exclusive."""
        timestamp = _finite(acquisition_time, "acquisition_time")
        return self.start_time <= timestamp < self.end_time


@dataclass(frozen=True)
class FaultEvent:
    event_id: str
    scenario_id: str
    operator_id: str
    sensor_id: str
    acquisition_time: float
    arrival_time: float
    sequence_number: int
    fault_type: FaultType
    severity: float = 1.0
    affected: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    processing_time: float | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "scenario_id", "operator_id", "sensor_id"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        object.__setattr__(self, "acquisition_time", _finite(self.acquisition_time, "acquisition_time"))
        object.__setattr__(self, "arrival_time", _finite(self.arrival_time, "arrival_time"))
        if self.processing_time is not None:
            object.__setattr__(self, "processing_time", _finite(self.processing_time, "processing_time"))
        if isinstance(self.sequence_number, bool) or not isinstance(self.sequence_number, int):
            raise ContractError("sequence_number must be an integer")
        if self.sequence_number < 0:
            raise ContractError("sequence_number must be non-negative")
        if not isinstance(self.fault_type, FaultType):
            raise ContractError("fault_type must be a FaultType")
        severity = _finite(self.severity, "severity")
        if not 0.0 <= severity <= 1.0:
            raise ContractError("severity must be in [0, 1]")
        object.__setattr__(self, "severity", severity)
        if not isinstance(self.affected, bool):
            raise ContractError("affected must be a bool")
        if not isinstance(self.metadata, Mapping):
            raise ContractError("metadata must be a mapping")
        # Event metadata may contain descriptive values only; non-finite values
        # belong in the raw envelope, not in the normalized event contract.
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))


@dataclass(frozen=True)
class FaultResult:
    scenario_id: str
    input_reference: str
    output_reference: str
    events: tuple[FaultEvent, ...]
    applied_count: int
    skipped_count: int
    status: FaultStatus
    seed: int

    def __post_init__(self) -> None:
        for name in ("scenario_id", "input_reference", "output_reference"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        events = tuple(self.events)
        if any(not isinstance(event, FaultEvent) for event in events):
            raise ContractError("events must contain only FaultEvent instances")
        if any(event.scenario_id != self.scenario_id for event in events):
            raise ContractError("all events must belong to the result scenario")
        object.__setattr__(self, "events", events)
        for name in ("applied_count", "skipped_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")
        if self.applied_count + self.skipped_count < len(events):
            raise ContractError("counts cannot be smaller than the recorded event count")
        if not isinstance(self.status, FaultStatus):
            raise ContractError("status must be a FaultStatus")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError("seed must be an integer")
        if not 0 <= self.seed <= (2**64 - 1):
            raise ContractError("seed must be in [0, 2**64 - 1]")
        if self.status == FaultStatus.APPLIED and self.applied_count == 0:
            raise ContractError("APPLIED requires applied_count > 0")
        if self.status == FaultStatus.NOT_APPLICABLE and self.applied_count != 0:
            raise ContractError("NOT_APPLICABLE requires applied_count == 0")
        if self.status == FaultStatus.REJECTED and self.applied_count != 0:
            raise ContractError("REJECTED requires applied_count == 0")


@dataclass(frozen=True)
class RecoveryRecord:
    recovery_id: str
    scenario_id: str
    sensor_id: str
    fault_end_time: float
    sensor_return_time: float | None
    recovery_time: float | None
    baseline_error: float
    recovery_error: float | None
    recovery_status: RecoveryStatus
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("recovery_id", "scenario_id", "sensor_id"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        fault_end = _finite(self.fault_end_time, "fault_end_time")
        object.__setattr__(self, "fault_end_time", fault_end)
        for name in ("sensor_return_time", "recovery_time", "recovery_error"):
            value = getattr(self, name)
            if value is not None:
                value = _finite(value, name)
                if name in ("recovery_time", "recovery_error") and value < 0:
                    raise ContractError(f"{name} must be non-negative")
                object.__setattr__(self, name, value)
        baseline = _finite(self.baseline_error, "baseline_error")
        if baseline < 0:
            raise ContractError("baseline_error must be non-negative")
        object.__setattr__(self, "baseline_error", baseline)
        if self.sensor_return_time is not None and self.sensor_return_time < fault_end:
            raise ContractError("sensor_return_time cannot precede fault_end_time")
        if self.sensor_return_time is None and self.recovery_time is not None:
            raise ContractError("recovery_time requires sensor_return_time")
        if not isinstance(self.recovery_status, RecoveryStatus):
            raise ContractError("recovery_status must be a RecoveryStatus")
        if not isinstance(self.evidence, Mapping):
            raise ContractError("evidence must be a mapping")
        object.__setattr__(self, "evidence", _freeze_json(self.evidence, "evidence"))


@runtime_checkable
class FaultOperator(Protocol):
    """Stateless operator interface; implementations own their validation."""

    operator_id: str
    version: str

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Raise ContractError (or a documented subtype) for invalid parameters."""

    def apply(self, input_data: Any, scenario: FaultScenario, rng: Any) -> FaultResult:
        """Apply a scenario without mutating input_data; use the supplied local RNG."""

    def describe(self) -> Mapping[str, Any]:
        """Return a JSON-compatible description of the operator."""
