"""Unit tests for the deterministic Phase 4.3 Quality Gate."""
import pytest

from roboqa.qa import (
    FindingSeverity,
    MetricAvailability,
    MetricObservation,
    MetricScope,
    QualityDecision,
)
from roboqa.qa.gate import (
    MetricRule,
    QualityGateError,
    QualityGatePolicy,
    evaluate_quality_gate,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def metric(name="position_rmse", value=0.5, *, availability=MetricAvailability.AVAILABLE,
           unit="m", scope=MetricScope.RUN, sensor_id=None):
    return MetricObservation(
        name=name, value=value, unit=unit, scope=scope, sensor_id=sensor_id,
        sample_count=3, availability=availability,
    )


def provenance(**overrides):
    values = {
        "run_id": "run-001",
        "generated_at": "2026-11-08T05:00:00Z",
        "source_hash": HASH_A,
        "output_hash": HASH_B,
        "configuration_hash": HASH_C,
    }
    values.update(overrides)
    return values


def evaluate(observations, rules, **metadata):
    return evaluate_quality_gate(
        observations, QualityGatePolicy("policy-v1", tuple(rules)),
        **(provenance() | metadata),
    )


def test_happy_path_returns_pass_with_provenance():
    report = evaluate(
        [metric(value=0.5)],
        [MetricRule("rmse", "position_rmse", MetricScope.RUN,
                    minimum=0.0, maximum=1.0, unit="m")],
    )
    assert report.decision is QualityDecision.PASS
    assert report.findings == ()
    assert report.policy_version == "policy-v1"
    assert report.source_hash == HASH_A


def test_warning_band_returns_pass_with_warnings():
    report = evaluate(
        [metric(value=0.9)],
        [MetricRule("rmse", "position_rmse", MetricScope.RUN,
                    minimum=0.0, maximum=1.0, warning_maximum=0.8, unit="m")],
    )
    assert report.decision is QualityDecision.PASS_WITH_WARNINGS
    assert report.findings[0].severity is FindingSeverity.WARNING


def test_hard_threshold_breach_returns_fail():
    report = evaluate(
        [metric(value=1.1)],
        [MetricRule("rmse", "position_rmse", MetricScope.RUN,
                    maximum=1.0, unit="m")],
    )
    assert report.decision is QualityDecision.FAIL
    assert report.findings[0].severity is FindingSeverity.FAIL
    assert report.findings[0].observed_value == 1.1


def test_required_missing_metric_fails_closed_as_blocked():
    report = evaluate(
        [],
        [MetricRule("rmse", "position_rmse", MetricScope.RUN)],
    )
    assert report.decision is QualityDecision.BLOCKED
    assert report.findings[0].severity is FindingSeverity.BLOCK


def test_required_invalid_metric_fails_closed_as_blocked():
    report = evaluate(
        [metric(value=None, availability=MetricAvailability.INVALID)],
        [MetricRule("rmse", "position_rmse", MetricScope.RUN)],
    )
    assert report.decision is QualityDecision.BLOCKED


def test_optional_missing_metric_is_warning_not_pass():
    report = evaluate(
        [],
        [MetricRule("optional-rmse", "position_rmse", MetricScope.RUN, required=False)],
    )
    assert report.decision is QualityDecision.PASS_WITH_WARNINGS


def test_optional_invalid_metric_is_warning():
    report = evaluate(
        [metric(value=None, availability=MetricAvailability.INVALID)],
        [MetricRule("optional-rmse", "position_rmse", MetricScope.RUN, required=False)],
    )
    assert report.decision is QualityDecision.PASS_WITH_WARNINGS


def test_decision_precedence_blocked_then_fail_then_warning():
    report = evaluate(
        [metric("position_rmse", 2.0)],
        [
            MetricRule("rmse", "position_rmse", MetricScope.RUN, maximum=1.0),
            MetricRule("cov", "covariance_trace_mean", MetricScope.RUN, required=False),
            MetricRule("nis", "nis_mean", MetricScope.RUN),
        ],
    )
    assert report.decision is QualityDecision.BLOCKED
    assert [finding.severity for finding in report.findings] == [
        FindingSeverity.FAIL, FindingSeverity.WARNING, FindingSeverity.BLOCK
    ]


def test_sensor_identity_is_exact_and_normalized():
    report = evaluate(
        [metric("nis_mean", 3.0, scope=MetricScope.SENSOR, sensor_id="GPS",
                unit="dimensionless")],
        [MetricRule("gps-nis", "nis_mean", MetricScope.SENSOR,
                    sensor_id="gps", maximum=5.0, unit="dimensionless")],
    )
    assert report.decision is QualityDecision.PASS


def test_window_scope_rule_matches_exact_window():
    observation = MetricObservation(
        name="latency", value=0.1, unit="s", scope=MetricScope.WINDOW,
        sample_count=2, window_start=1.0, window_end=2.0,
    )
    rule = MetricRule(
        "latency-window", "latency", MetricScope.WINDOW,
        window_start=1.0, window_end=2.0, maximum=0.5, unit="s",
    )
    assert evaluate([observation], [rule]).decision is QualityDecision.PASS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum": 2.0, "maximum": 1.0},
        {"warning_minimum": 1.0},
        {"minimum": 2.0, "warning_minimum": 1.0},
        {"maximum": 2.0, "warning_maximum": 3.0},
        {"minimum": float("nan")},
        {"maximum": float("inf")},
        {"minimum": True},
    ],
)
def test_invalid_metric_rule_bounds_are_rejected(kwargs):
    with pytest.raises(QualityGateError):
        MetricRule("r", "m", MetricScope.RUN, **kwargs)


def test_policy_rejects_empty_or_duplicate_rules():
    with pytest.raises(QualityGateError):
        QualityGatePolicy("v1", ())
    rule = MetricRule("same", "metric", MetricScope.RUN)
    with pytest.raises(QualityGateError):
        QualityGatePolicy("v1", (rule, rule))


def test_policy_rejects_duplicate_rule_ids():
    with pytest.raises(QualityGateError):
        QualityGatePolicy("v1", (
            MetricRule("same", "metric-a", MetricScope.RUN),
            MetricRule("same", "metric-b", MetricScope.RUN),
        ))


def test_gate_rejects_non_metric_and_duplicate_observations():
    rule = MetricRule("r", "position_rmse", MetricScope.RUN)
    with pytest.raises(QualityGateError):
        evaluate([object()], [rule])
    item = metric()
    with pytest.raises(QualityGateError):
        evaluate([item, item], [rule])


def test_gate_rejects_invalid_policy_and_metadata():
    with pytest.raises(QualityGateError):
        evaluate_quality_gate([], object(), **provenance())
    with pytest.raises(QualityGateError):
        evaluate([], [MetricRule("r", "position_rmse", MetricScope.RUN)],
                 generated_at="not-a-timestamp")


def test_unit_mismatch_blocks_required_evidence():
    report = evaluate(
        [metric(value=0.5, unit="km")],
        [MetricRule("rmse", "position_rmse", MetricScope.RUN, unit="m")],
    )
    assert report.decision is QualityDecision.BLOCKED


def test_unconfigured_metrics_are_preserved_but_not_thresholded():
    report = evaluate(
        [metric(value=0.5), metric("nis_mean", 99.0)],
        [MetricRule("rmse", "position_rmse", MetricScope.RUN, maximum=1.0)],
    )
    assert report.decision is QualityDecision.PASS
    assert len(report.metrics) == 2
