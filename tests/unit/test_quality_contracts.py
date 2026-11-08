import pytest

from roboqa.qa import (
    FindingSeverity,
    MetricAvailability,
    MetricObservation,
    MetricScope,
    QualityContractError,
    QualityDecision,
    QualityFinding,
    QualityReport,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def metric(**overrides):
    values = dict(
        name="nis_mean",
        value=2.1,
        unit="dimensionless",
        scope=MetricScope.RUN,
        sample_count=100,
    )
    values.update(overrides)
    return MetricObservation(**values)


def report(**overrides):
    values = dict(
        run_id="run-001",
        decision=QualityDecision.PASS,
        metrics=(metric(),),
        findings=(),
        generated_at="2026-11-08T00:00:00Z",
        source_hash=HASH_A,
        output_hash=HASH_B,
        configuration_hash=HASH_C,
        policy_version="quality-policy-v1",
    )
    values.update(overrides)
    return QualityReport(**values)


def test_metric_observation_is_immutable_and_explicitly_available():
    item = metric()
    assert item.value == 2.1
    assert item.availability is MetricAvailability.AVAILABLE


@pytest.mark.parametrize("overrides", [
    {"value": float("nan")},
    {"value": float("inf")},
    {"value": None},
    {"sample_count": -1},
    {"sample_count": True},
    {"scope": MetricScope.SENSOR},
    {"sensor_id": "gps"},
])
def test_invalid_run_metric_is_rejected(overrides):
    with pytest.raises(QualityContractError):
        metric(**overrides)


def test_sensor_scoped_metric_requires_sensor_identity():
    item = metric(scope=MetricScope.SENSOR, sensor_id="gps")
    assert item.sensor_id == "gps"
    with pytest.raises(QualityContractError):
        metric(scope=MetricScope.SENSOR)


def test_window_scoped_metric_requires_valid_half_open_bounds():
    item = metric(scope=MetricScope.WINDOW, window_start=1.0, window_end=2.0)
    assert item.window_start == 1.0
    with pytest.raises(QualityContractError):
        metric(scope=MetricScope.WINDOW, window_start=2.0, window_end=2.0)


@pytest.mark.parametrize("availability", [MetricAvailability.MISSING, MetricAvailability.INVALID])
def test_unavailable_metric_has_no_numeric_sentinel(availability):
    item = metric(value=None, availability=availability, sample_count=0)
    assert item.value is None
    assert item.availability is availability
    with pytest.raises(QualityContractError):
        metric(value=0.0, availability=availability)


def test_finding_preserves_evidence_references():
    finding = QualityFinding(
        rule_id="nis-upper-bound",
        severity=FindingSeverity.WARNING,
        message="NIS exceeded warning threshold",
        metric_name="nis_mean",
        observed_value=8.2,
        expected_range="[0, 6]",
        evidence_refs=("sha256:" + HASH_A, "campaign://run-001"),
    )
    assert len(finding.evidence_refs) == 2


def test_report_is_immutable_and_provenance_linked():
    item = report()
    assert item.decision is QualityDecision.PASS
    assert item.source_hash == HASH_A
    assert item.metrics[0].name == "nis_mean"


@pytest.mark.parametrize("overrides", [
    {"generated_at": "2026-11-08T00:00:00"},
    {"generated_at": "2026-11-08T03:00:00+03:00"},
    {"source_hash": "not-a-hash"},
    {"output_hash": "g" * 64},
    {"configuration_hash": ""},
    {"metrics": (object(),)},
    {"findings": (object(),)},
])
def test_invalid_report_is_rejected(overrides):
    with pytest.raises(QualityContractError):
        report(**overrides)


def test_report_rejects_duplicate_metric_scope_identity():
    with pytest.raises(QualityContractError):
        report(metrics=(metric(), metric()))


def test_report_does_not_derive_decision_from_findings():
    finding = QualityFinding(
        rule_id="hard-fail",
        severity=FindingSeverity.FAIL,
        message="Deliberate policy conflict for contract test",
    )
    item = report(decision=QualityDecision.PASS, findings=(finding,))
    assert item.decision is QualityDecision.PASS
