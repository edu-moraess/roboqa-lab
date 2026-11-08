"""
Filter result structures, status codes, and immutable innovation telemetry.
Fase 2 / Phase 4.1 — Fusion Engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FilterStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    INVALID_INPUT = "invalid_input"
    NUMERICAL_FAILURE = "numerical_failure"
    INIT_INSUFFICIENT = "init_insufficient"
    TEMPORAL_UNSUPPORTED = "temporal_unsupported"


@dataclass(frozen=True)
class InnovationTelemetry:
    """Immutable evidence produced by one successfully accepted EKF update."""

    timestamp: float
    sensor_id: str
    measurement_dof: int
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    nis: float

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        nis = float(self.nis)
        if not np.isfinite(timestamp):
            raise ValueError("telemetry timestamp must be finite")
        if not isinstance(self.sensor_id, str) or not self.sensor_id.strip():
            raise ValueError("sensor_id must be a non-empty string")
        if isinstance(self.measurement_dof, bool) or not isinstance(self.measurement_dof, int) or self.measurement_dof < 1:
            raise ValueError("measurement_dof must be a positive integer")
        innovation = np.array(self.innovation, dtype=np.float64, copy=True)
        covariance = np.array(self.innovation_covariance, dtype=np.float64, copy=True)
        if innovation.shape != (self.measurement_dof,):
            raise ValueError("innovation shape must match measurement_dof")
        if covariance.shape != (self.measurement_dof, self.measurement_dof):
            raise ValueError("innovation_covariance shape must match measurement_dof")
        if not np.all(np.isfinite(innovation)) or not np.all(np.isfinite(covariance)):
            raise ValueError("innovation telemetry arrays must be finite")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
            raise ValueError("innovation_covariance must be symmetric")
        if not np.isfinite(nis) or nis < 0.0:
            raise ValueError("nis must be finite and non-negative")
        innovation.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "sensor_id", self.sensor_id.strip().lower())
        object.__setattr__(self, "innovation", innovation)
        object.__setattr__(self, "innovation_covariance", covariance)
        object.__setattr__(self, "nis", nis)

    @property
    def degrees_of_freedom(self) -> int:
        """Alias used by statistical consistency evaluators."""
        return self.measurement_dof


@dataclass
class FilterDiagnostics:
    n_predict: int = 0
    n_update: int = 0
    n_rejected: int = 0
    sensors_seen: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_predict": self.n_predict,
            "n_update": self.n_update,
            "n_rejected": self.n_rejected,
            "sensors_seen": list(self.sensors_seen),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass
class FilterResult:
    """Structured output of a fusion run, including per-update telemetry."""

    t: np.ndarray
    x: np.ndarray
    P: np.ndarray
    status: FilterStatus
    diagnostics: FilterDiagnostics
    provenance: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    telemetry: tuple[InnovationTelemetry, ...] = ()

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.x = np.asarray(self.x, dtype=np.float64)
        self.P = np.asarray(self.P, dtype=np.float64)
        self.telemetry = tuple(self.telemetry)
        if any(not isinstance(item, InnovationTelemetry) for item in self.telemetry):
            raise TypeError("telemetry must contain InnovationTelemetry instances")

    @property
    def n(self) -> int:
        return int(self.t.shape[0])

    def position(self) -> np.ndarray:
        return self.x[:, :2]
