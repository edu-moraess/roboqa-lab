"""Raw event envelopes at the boundary before Measurement validation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from roboqa.fault_injection.contracts import ContractError, _non_empty


def _freeze_raw(value: Any, path: str = "payload") -> Any:
    """Freeze JSON-like and NumPy raw values while preserving NaN and Inf."""
    if isinstance(value, np.ndarray):
        return _freeze_raw(value.tolist(), path)
    if isinstance(value, np.generic):
        return _freeze_raw(value.item(), path)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} keys must be strings")
            frozen[key] = _freeze_raw(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_raw(item, f"{path}[]") for item in value)
    raise ContractError(f"{path} contains unsupported type {type(value).__name__}")


@dataclass(frozen=True)
class RawMeasurementEnvelope:
    """Transport-level record; timestamps/payload may be invalid by design.

    This type is not a Measurement and must pass the validation boundary before
    entering the Fusion Engine. NaN/Inf are preserved for diagnostics rather
    than silently normalized or rejected at envelope construction.
    """

    sensor_id: str
    acquisition_time: float
    arrival_time: float
    sequence_number: int
    payload: Any
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_id", _non_empty(self.sensor_id, "sensor_id"))
        for name in ("acquisition_time", "arrival_time"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"{name} must be numeric at the raw boundary")
            object.__setattr__(self, name, float(value))
        if isinstance(self.sequence_number, bool) or not isinstance(self.sequence_number, int):
            raise ContractError("sequence_number must be an integer")
        if self.sequence_number < 0:
            raise ContractError("sequence_number must be non-negative")
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _non_empty(self.source_id, "source_id"))
        object.__setattr__(self, "payload", _freeze_raw(self.payload))

    @property
    def timestamps_finite(self) -> bool:
        return math.isfinite(self.acquisition_time) and math.isfinite(self.arrival_time)

    @property
    def payload_finite(self) -> bool:
        def finite(value: Any) -> bool:
            if isinstance(value, float):
                return math.isfinite(value)
            if isinstance(value, Mapping):
                return all(finite(item) for item in value.values())
            if isinstance(value, tuple):
                return all(finite(item) for item in value)
            return True
        return finite(self.payload)

    @property
    def is_valid_candidate(self) -> bool:
        return self.timestamps_finite and self.payload_finite
