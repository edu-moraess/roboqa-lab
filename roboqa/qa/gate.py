"""Deterministic quality policy evaluation (Phase 4.3)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from roboqa.qa.contracts import (
    FindingSeverity,
    MetricAvailability,
    MetricObservation,
    MetricScope,
    QualityContractError,
    QualityDecision,
    QualityFinding,
    QualityReport,
)


class QualityGateError(ValueError):
    """Raised when a quality-gate policy or its metric input is invalid."""


def _finite_bound(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise QualityGateError(f"{name} must be a finite number or None")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QualityGateError(f"{name} must be a finite number or None") from exc
    if not math.isfinite(result):
        raise QualityGateError(f"{name} must be a finite number or None")
    return result


@dataclass(frozen=True)
class MetricRule:
    """One exact metric identity and its hard/warning bounds.

    Warning bounds must be stricter than hard bounds. Values outside a hard
    bound produce FAIL; values outside only a warning bound produce a warning.
    """

    rule_id: str
    metric_name: str
    scope: MetricScope
    required: bool = True
    sensor_id: str | None = None
    window_start: float | None = None
    window_end: float | None = None
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    warning_minimum: float | None = None
    warning_maximum: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise QualityGateError("rule_id must be a non-empty string")
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise QualityGateError("metric_name must be a non-empty string")
        object.__setattr__(self, "rule_id", self.rule_id.strip())
        object.__setattr__(self, "metric_name", self.metric_name.strip())
        if not isinstance(self.scope, MetricScope):
            raise QualityGateError("scope must be a MetricScope")
        if not isinstance(self.required, bool):
            raise QualityGateError("required must be a boolean")
        if self.scope is MetricScope.SENSOR:
            if not isinstance(self.sensor_id, str) or not self.sensor_id.strip():
                raise QualityGateError("sensor-scoped rules require sensor_id")
            object.__setattr__(self, "sensor_id", self.sensor_id.strip().lower())
        elif self.sensor_id is not None:
            raise QualityGateError("sensor_id is only valid for sensor-scoped rules")

        start = _finite_bound(self.window_start, "window_start")
        end = _finite_bound(self.window_end, "window_end")
        if self.scope is MetricScope.WINDOW:
            if start is None or end is None or start >= end:
                raise QualityGateError("window rules require finite window_start < window_end")
        elif start is not None or end is not None:
            raise QualityGateError("window bounds are only valid for window-scoped rules")
        object.__setattr__(self, "window_start", start)
        object.__setattr__(self, "window_end", end)

        if self.unit is not None:
            if not isinstance(self.unit, str):
                raise QualityGateError("unit must be a string or None")
            object.__setattr__(self, "unit", self.unit.strip())
            if not self.unit:
                raise QualityGateError("unit must not be empty")

        for field in ("minimum", "maximum", "warning_minimum", "warning_maximum"):
            object.__setattr__(self, field, _finite_bound(getattr(self, field), field))
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise QualityGateError("minimum must be <= maximum")
        if self.warning_minimum is not None:
            if self.minimum is None or self.warning_minimum < self.minimum:
                raise QualityGateError("warning_minimum requires minimum and must be >= minimum")
        if self.warning_maximum is not None:
            if self.maximum is None or self.warning_maximum > self.maximum:
                raise QualityGateError("warning_maximum requires maximum and must be <= maximum")

    @property
    def identity(self) -> tuple[str, MetricScope, str | None, float | None, float | None]:
        return (self.metric_name, self.scope, self.sensor_id, self.window_start, self.window_end)


@dataclass(frozen=True)
class QualityGatePolicy:
    """Versioned deterministic policy; rules are evaluated in declared order."""

    policy_version: str
    rules: tuple[MetricRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise QualityGateError("policy_version must be a non-empty string")
        object.__setattr__(self, "policy_version", self.policy_version.strip())
        try:
            rules = tuple(self.rules)
        except TypeError as exc:
            raise QualityGateError("rules must be an iterable of MetricRule instances") from exc
        if not rules or any(not isinstance(rule, MetricRule) for rule in rules):
            raise QualityGateError("rules must contain at least one MetricRule")
        rule_ids = [rule.rule_id for rule in rules]
        identities = [rule.identity for rule in rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise QualityGateError("rule_id values must be unique")
        if len(set(identities)) != len(identities):
            raise QualityGateError("a metric identity may be governed by only one rule")
        object.__setattr__(self, "rules", rules)


def _metric_identity(item: MetricObservation):
    sensor_id = item.sensor_id.strip().lower() if item.sensor_id is not None else None
    return (item.name, item.scope, sensor_id, item.window_start, item.window_end)


def _expected_range(rule: MetricRule) -> str | None:
    parts = []
    if rule.minimum is not None:
        parts.append(f">= {rule.minimum:g}")
    if rule.maximum is not None:
        parts.append(f"<= {rule.maximum:g}")
    if rule.warning_minimum is not None:
        parts.append(f"warning >= {rule.warning_minimum:g}")
    if rule.warning_maximum is not None:
        parts.append(f"warning <= {rule.warning_maximum:g}")
    return "; ".join(parts) or None


def _finding(rule: MetricRule, severity: FindingSeverity, message: str,
             metric: MetricObservation | None = None) -> QualityFinding:
    return QualityFinding(
        rule_id=rule.rule_id,
        severity=severity,
        message=message,
        metric_name=rule.metric_name,
        observed_value=None if metric is None else metric.value,
        expected_range=_expected_range(rule),
    )


def _decision(findings: Iterable[QualityFinding]) -> QualityDecision:
    severities = {item.severity for item in findings}
    if FindingSeverity.BLOCK in severities:
        return QualityDecision.BLOCKED
    if FindingSeverity.FAIL in severities:
        return QualityDecision.FAIL
    if FindingSeverity.WARNING in severities:
        return QualityDecision.PASS_WITH_WARNINGS
    return QualityDecision.PASS


def evaluate_quality_gate(
    metrics: Iterable[MetricObservation],
    policy: QualityGatePolicy,
    *,
    run_id: str,
    generated_at: str,
    source_hash: str,
    output_hash: str,
    configuration_hash: str,
) -> QualityReport:
    """Evaluate metrics against an explicit policy and return an immutable report.

    Required missing/invalid observations fail closed as BLOCKED. Optional
    missing/invalid observations create warnings. Hard-bound breaches are FAIL;
    warning-band breaches are warnings. Unconfigured metrics are retained only
    as input evidence and are not implicitly assigned thresholds.
    """
    if not isinstance(policy, QualityGatePolicy):
        raise QualityGateError("policy must be a QualityGatePolicy")
    try:
        observations = tuple(metrics)
    except TypeError as exc:
        raise QualityGateError("metrics must be iterable") from exc
    if any(not isinstance(item, MetricObservation) for item in observations):
        raise QualityGateError("metrics must contain MetricObservation instances")

    by_identity = {}
    for item in observations:
        identity = _metric_identity(item)
        if identity in by_identity:
            raise QualityGateError("duplicate metric observation identity")
        by_identity[identity] = item

    findings = []
    for rule in policy.rules:
        metric = by_identity.get(rule.identity)
        if metric is None:
            if rule.required:
                findings.append(_finding(
                    rule, FindingSeverity.BLOCK,
                    "Required metric observation is missing.",
                ))
            else:
                findings.append(_finding(
                    rule, FindingSeverity.WARNING,
                    "Optional metric observation is missing.",
                ))
            continue

        if metric.availability is not MetricAvailability.AVAILABLE:
            state = metric.availability.value
            severity = FindingSeverity.BLOCK if rule.required else FindingSeverity.WARNING
            findings.append(_finding(
                rule, severity,
                f"Metric evidence is {state}; no numeric value is accepted.",
                metric,
            ))
            continue

        if rule.unit is not None and metric.unit != rule.unit:
            severity = FindingSeverity.BLOCK if rule.required else FindingSeverity.WARNING
            findings.append(_finding(
                rule, severity,
                f"Metric unit mismatch: expected {rule.unit!r}, received {metric.unit!r}.",
                metric,
            ))
            continue

        value = metric.value
        if rule.minimum is not None and value < rule.minimum:
            findings.append(_finding(
                rule, FindingSeverity.FAIL,
                f"Metric value {value:g} is below hard minimum {rule.minimum:g}.",
                metric,
            ))
            continue
        if rule.maximum is not None and value > rule.maximum:
            findings.append(_finding(
                rule, FindingSeverity.FAIL,
                f"Metric value {value:g} is above hard maximum {rule.maximum:g}.",
                metric,
            ))
            continue
        if rule.warning_minimum is not None and value < rule.warning_minimum:
            findings.append(_finding(
                rule, FindingSeverity.WARNING,
                f"Metric value {value:g} is below warning threshold {rule.warning_minimum:g}.",
                metric,
            ))
            continue
        if rule.warning_maximum is not None and value > rule.warning_maximum:
            findings.append(_finding(
                rule, FindingSeverity.WARNING,
                f"Metric value {value:g} is above warning threshold {rule.warning_maximum:g}.",
                metric,
            ))

    decision = _decision(findings)
    try:
        return QualityReport(
            run_id=run_id,
            decision=decision,
            metrics=observations,
            findings=tuple(findings),
            generated_at=generated_at,
            source_hash=source_hash,
            output_hash=output_hash,
            configuration_hash=configuration_hash,
            policy_version=policy.policy_version,
        )
    except QualityContractError as exc:
        raise QualityGateError(f"cannot construct quality report: {exc}") from exc
