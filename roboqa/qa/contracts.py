"""Immutable contracts for RoboQA quality metrics and evaluation reports (Phase 4.0)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class QualityContractError(ValueError):
    """Raised when a quality metric or report violates its public contract."""


class MetricAvailability(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    INVALID = "invalid"


class MetricScope(str, Enum):
    RUN = "run"
    SENSOR = "sensor"
    WINDOW = "window"


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    FAIL = "fail"
    BLOCK = "block"


class QualityDecision(str, Enum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"
    BLOCKED = "blocked"


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualityContractError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_optional(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise QualityContractError(f"{name} must be finite or None")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QualityContractError(f"{name} must be finite or None") from exc
    if not math.isfinite(result):
        raise QualityContractError(f"{name} must be finite or None")
    return result


@dataclass(frozen=True)
class MetricObservation:
    """One scalar metric with explicit availability, scope, and sample count.

    A missing/invalid metric carries no numeric value. Availability is never
    inferred from a numeric sentinel such as zero or NaN.
    """

    name: str
    value: float | None
    unit: str
    scope: MetricScope
    sample_count: int
    availability: MetricAvailability = MetricAvailability.AVAILABLE
    sensor_id: str | None = None
    window_start: float | None = None
    window_end: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "name"))
        if not isinstance(self.unit, str):
            raise QualityContractError("unit must be a string")
        if not isinstance(self.scope, MetricScope):
            raise QualityContractError("scope must be a MetricScope")
        if not isinstance(self.availability, MetricAvailability):
            raise QualityContractError("availability must be a MetricAvailability")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise QualityContractError("sample_count must be a non-negative integer")
        value = _finite_optional(self.value, "value")
        object.__setattr__(self, "value", value)
        if self.availability is MetricAvailability.AVAILABLE and value is None:
            raise QualityContractError("available metrics require a finite value")
        if self.availability is not MetricAvailability.AVAILABLE and value is not None:
            raise QualityContractError("missing/invalid metrics must not carry a numeric value")
        if self.scope is MetricScope.SENSOR:
            object.__setattr__(self, "sensor_id", _non_empty(self.sensor_id, "sensor_id"))
        elif self.sensor_id is not None:
            raise QualityContractError("sensor_id is only valid for sensor-scoped metrics")
        start = _finite_optional(self.window_start, "window_start")
        end = _finite_optional(self.window_end, "window_end")
        if self.scope is MetricScope.WINDOW:
            if start is None or end is None or start >= end:
                raise QualityContractError("window metrics require finite window_start < window_end")
        elif start is not None or end is not None:
            raise QualityContractError("window bounds are only valid for window-scoped metrics")


@dataclass(frozen=True)
class QualityFinding:
    """A human-readable finding linked to a metric and optional evidence."""

    rule_id: str
    severity: FindingSeverity
    message: str
    metric_name: str | None = None
    observed_value: float | None = None
    expected_range: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _non_empty(self.rule_id, "rule_id"))
        object.__setattr__(self, "message", _non_empty(self.message, "message"))
        if not isinstance(self.severity, FindingSeverity):
            raise QualityContractError("severity must be a FindingSeverity")
        if self.metric_name is not None:
            object.__setattr__(self, "metric_name", _non_empty(self.metric_name, "metric_name"))
        object.__setattr__(self, "observed_value", _finite_optional(self.observed_value, "observed_value"))
        if self.expected_range is not None:
            object.__setattr__(self, "expected_range", _non_empty(self.expected_range, "expected_range"))
        refs = tuple(self.evidence_refs)
        if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
            raise QualityContractError("evidence_refs must contain non-empty strings")
        object.__setattr__(self, "evidence_refs", tuple(ref.strip() for ref in refs))


@dataclass(frozen=True)
class QualityReport:
    """Immutable, provenance-linked evaluation output.

    The decision is supplied by the evaluator; this contract deliberately does
    not implement policy or derive PASS/FAIL from findings.
    """

    run_id: str
    decision: QualityDecision
    metrics: tuple[MetricObservation, ...]
    findings: tuple[QualityFinding, ...]
    generated_at: str
    source_hash: str
    output_hash: str
    configuration_hash: str
    policy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _non_empty(self.run_id, "run_id"))
        object.__setattr__(self, "policy_version", _non_empty(self.policy_version, "policy_version"))
        if not isinstance(self.decision, QualityDecision):
            raise QualityContractError("decision must be a QualityDecision")
        metrics = tuple(self.metrics)
        findings = tuple(self.findings)
        if any(not isinstance(item, MetricObservation) for item in metrics):
            raise QualityContractError("metrics must contain MetricObservation instances")
        if any(not isinstance(item, QualityFinding) for item in findings):
            raise QualityContractError("findings must contain QualityFinding instances")
        keys = [(m.name, m.scope, m.sensor_id, m.window_start, m.window_end) for m in metrics]
        if len(set(keys)) != len(keys):
            raise QualityContractError("metric observations must have unique scope identity")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "findings", findings)
        timestamp = _non_empty(self.generated_at, "generated_at")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise QualityContractError("generated_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise QualityContractError("generated_at must include UTC timezone")
        object.__setattr__(self, "generated_at", timestamp)
        for name in ("source_hash", "output_hash", "configuration_hash"):
            digest = _non_empty(getattr(self, name), name).lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise QualityContractError(f"{name} must be a 64-character SHA-256 hex digest")
            object.__setattr__(self, name, digest)
