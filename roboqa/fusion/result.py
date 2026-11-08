"""
Filter result structures and status codes.
Fase 2 — Fusion Engine
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
    """Structured output of a fusion run."""

    t: np.ndarray
    x: np.ndarray
    P: np.ndarray
    status: FilterStatus
    diagnostics: FilterDiagnostics
    provenance: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.x = np.asarray(self.x, dtype=np.float64)
        self.P = np.asarray(self.P, dtype=np.float64)

    @property
    def n(self) -> int:
        return int(self.t.shape[0])

    def position(self) -> np.ndarray:
        return self.x[:, :2]
