"""Phase 4.5-D3 integration tests for replay/execution evidence propagation."""
import numpy as np
import pytest

from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.fusion.ekf import EKFNumericalError
from roboqa.integration import (
    TransportOrderingPolicy,
    adapt_accepted_envelopes,
    build_fusion_schedule,
    execute_fusion_schedule,
    replay_transport,
)
from roboqa.integration.execution import ExecutionDisposition
from roboqa.qa import (
    MetricAvailability,
    MetricRule,
    MetricScope,
    QualityDecision,
    QualityGatePolicy,
    compute_integrated_quality_metrics,
    evaluate_integrated_quality,
)
from roboqa.qa.integrated import IntegratedQualityError


def packet(sequence, sensor, acquisition, arrival, payload, *, source="d3-run",
           processing=None):
    return RawMeasurementEnvelope(
        sensor_id=sensor,
        source_id=source,
        sequence_number=sequence,
        acquisition_time=acquisition,
        arrival_time=arrival,
        processing_time=processing,
        payload=payload,
    )


def make_pipeline():
    gps_1 = packet(1, "gps", 0.0, 0.1, (10.0, 20.0), processing=0.12)
    gps_2 = packet(2, "gps", 1.0, 1.1, (11.0, 20.5), processing=1.12)
    speed_1 = packet(1, "speed", 1.5, 1.6, (4.0,), processing=1.62)
    gps_late = packet(3, "gps", 0.5, 2.1, (10.5, 20.2), processing=2.12)
    gps_dropped = packet(4, "gps", 3.0, 3.1, (12.0, 21.0), processing=3.12)
    gps_invalid = packet(5, "gps", 4.0, float("nan"), (13.0, 22.0))
    source = (gps_1, gps_2, speed_1, gps_late, gps_dropped, gps_invalid)
    delivered = (gps_1, gps_2, gps_2, speed_1, gps_late, gps_invalid)
    replay = replay_transport(
        source,
        delivered,
        ordering_policy=TransportOrderingPolicy.ARRIVAL_ORDER,
    )
    adapted = adapt_accepted_envelopes(
        replay,
        sensor_covariances={
            "gps": np.eye(2),
            "speed": np.array([[0.25]]),
        },
    )
    schedule = build_fusion_schedule(replay, adapted)
    execution = execute_fusion_schedule(schedule)
    return source, delivered, replay, schedule, execution


def metric_map(metrics):
    return {(item.name, item.scope, item.sensor_id): item for item in metrics}


def test_integrated_metrics_conserve_transport_and_fusion_evidence():
    _, delivered, replay, _, execution = make_pipeline()
    metrics = compute_integrated_quality_metrics(
        execution.filter_result,
        replay_summary=replay.summary,
        execution_record=execution.record,
        envelopes=delivered,
    )
    by_key = metric_map(metrics)

    assert replay.summary.source_count == 6
    assert replay.summary.delivered_count == 6
    assert replay.summary.accepted_count == 3
    assert replay.summary.duplicate_count == 1
    assert replay.summary.late_count == 1
    assert replay.summary.invalid_count == 1
    assert replay.summary.dropped_count == 1

    assert by_key[("transport_source_count", MetricScope.RUN, None)].value == 6
    assert by_key[("transport_delivered_count", MetricScope.RUN, None)].value == 6
    assert by_key[("transport_accepted_count", MetricScope.RUN, None)].value == 3
    assert by_key[("transport_duplicate_count", MetricScope.RUN, None)].value == 1
    assert by_key[("transport_late_count", MetricScope.RUN, None)].value == 1
    assert by_key[("transport_invalid_count", MetricScope.RUN, None)].value == 1
    assert by_key[("transport_dropped_count", MetricScope.RUN, None)].value == 1

    assert by_key[("fusion_updated_count", MetricScope.RUN, None)].value == 3
    assert by_key[("fusion_failed_count", MetricScope.RUN, None)].value == 0
    assert by_key[("fusion_not_executed_count", MetricScope.RUN, None)].value == 0
    assert by_key[("transport_acceptance_ratio", MetricScope.RUN, None)].value == pytest.approx(0.5)
    assert by_key[("fusion_update_ratio", MetricScope.RUN, None)].value == pytest.approx(0.5)
    assert by_key[("fusion_execution_success_ratio", MetricScope.RUN, None)].value == pytest.approx(1.0)

    gps_source = by_key[("transport_source_count", MetricScope.SENSOR, "gps")]
    gps_delivered = by_key[("transport_delivered_count", MetricScope.SENSOR, "gps")]
    gps_duplicate = by_key[("transport_duplicate_count", MetricScope.SENSOR, "gps")]
    assert gps_source.value == 5
    assert gps_delivered.value == 5
    assert gps_duplicate.value == 1
    assert by_key[("fusion_updated_count", MetricScope.SENSOR, "gps")].value == 2
    assert by_key[("transport_acceptance_ratio", MetricScope.SENSOR, "gps")].value == pytest.approx(0.4)


def test_execution_failure_and_not_executed_are_separate_qa_metrics(monkeypatch):
    _, delivered, replay, schedule, _ = make_pipeline()
    original = __import__("roboqa.fusion.ekf", fromlist=["EKF"]).EKF.update
    calls = {"count": 0}

    def fail_second(self, measurement):
        calls["count"] += 1
        if calls["count"] == 2:
            raise EKFNumericalError("D3 controlled failure")
        return original(self, measurement)

    monkeypatch.setattr("roboqa.fusion.ekf.EKF.update", fail_second)
    execution = execute_fusion_schedule(schedule)
    metrics = compute_integrated_quality_metrics(
        execution.filter_result,
        replay_summary=replay.summary,
        execution_record=execution.record,
        envelopes=delivered,
    )
    by_key = metric_map(metrics)

    assert execution.record.updated_count == 1
    assert execution.record.failed_count == 1
    assert execution.record.not_executed_count == 1
    assert by_key[("fusion_updated_count", MetricScope.RUN, None)].value == 1
    assert by_key[("fusion_failed_count", MetricScope.RUN, None)].value == 1
    assert by_key[("fusion_not_executed_count", MetricScope.RUN, None)].value == 1
    assert by_key[("fusion_execution_success_ratio", MetricScope.RUN, None)].value == pytest.approx(1 / 3)


def test_integrated_quality_blocks_missing_denominator_and_preserves_zero_counts():
    source = packet(1, "gps", 0.0, 0.1, (1.0, 2.0), processing=0.12)
    replay = replay_transport((source,), ())
    schedule = build_fusion_schedule(replay, ())
    execution = execute_fusion_schedule(schedule)
    metrics = compute_integrated_quality_metrics(
        execution.filter_result,
        replay_summary=replay.summary,
        execution_record=execution.record,
    )
    by_key = metric_map(metrics)

    assert by_key[("transport_source_count", MetricScope.RUN, None)].value == 1
    assert by_key[("transport_dropped_count", MetricScope.RUN, None)].value == 1
    ratio = by_key[("fusion_execution_success_ratio", MetricScope.RUN, None)]
    assert ratio.availability is MetricAvailability.MISSING
    assert ratio.value is None

    policy = QualityGatePolicy(
        "d3-required-evidence-v1",
        (MetricRule(
            "execution-success",
            "fusion_execution_success_ratio",
            MetricScope.RUN,
            minimum=1.0,
            unit="ratio",
        ),),
    )
    report = evaluate_integrated_quality(
        execution.filter_result,
        replay_summary=replay.summary,
        execution_record=execution.record,
        policy=policy,
        generated_at="2026-11-08T08:00:00Z",
        configuration_hash="c" * 64,
    )
    assert report.decision is QualityDecision.BLOCKED
    assert report.source_hash == replay.summary.source_hash
    assert report.output_hash == execution.record.execution_hash


def test_quality_gate_can_fail_on_explicit_transport_thresholds():
    _, delivered, replay, _, execution = make_pipeline()
    policy = QualityGatePolicy(
        "d3-transport-policy-v1",
        (
            MetricRule(
                "no-dropped-source-packets",
                "transport_dropped_count",
                MetricScope.RUN,
                maximum=0.0,
                unit="packets",
            ),
            MetricRule(
                "all-accepted-measurements-executed",
                "fusion_execution_success_ratio",
                MetricScope.RUN,
                minimum=1.0,
                unit="ratio",
            ),
        ),
    )
    report = evaluate_integrated_quality(
        execution.filter_result,
        replay_summary=replay.summary,
        execution_record=execution.record,
        policy=policy,
        generated_at="2026-11-08T08:00:00Z",
        configuration_hash="d" * 64,
        envelopes=delivered,
    )
    assert report.decision is QualityDecision.FAIL
    assert {item.rule_id for item in report.findings} == {"no-dropped-source-packets"}


def test_rejects_replay_summary_not_bound_to_execution_record():
    _, _, replay, _, execution = make_pipeline()
    other = replay_transport((), ()).summary
    with pytest.raises(IntegratedQualityError, match="exactly match"):
        compute_integrated_quality_metrics(
            execution.filter_result,
            replay_summary=other,
            execution_record=execution.record,
        )


def test_rejects_filter_result_telemetry_not_bound_to_execution_record():
    _, _, replay, _, execution = make_pipeline()
    altered = type(execution.filter_result)(
        t=execution.filter_result.t.copy(),
        x=execution.filter_result.x.copy(),
        P=execution.filter_result.P.copy(),
        status=execution.filter_result.status,
        diagnostics=execution.filter_result.diagnostics,
        provenance=execution.filter_result.provenance.copy(),
        config=execution.filter_result.config.copy(),
        telemetry=(),
    )
    with pytest.raises(IntegratedQualityError, match="telemetry"):
        compute_integrated_quality_metrics(
            altered,
            replay_summary=replay.summary,
            execution_record=execution.record,
        )


def test_rejects_filter_status_not_bound_to_execution_record():
    from roboqa.fusion.result import FilterStatus

    _, _, replay, _, execution = make_pipeline()
    altered = type(execution.filter_result)(
        t=execution.filter_result.t.copy(),
        x=execution.filter_result.x.copy(),
        P=execution.filter_result.P.copy(),
        status=FilterStatus.WARNING,
        diagnostics=execution.filter_result.diagnostics,
        provenance=execution.filter_result.provenance.copy(),
        config=execution.filter_result.config.copy(),
        telemetry=execution.filter_result.telemetry,
    )
    with pytest.raises(IntegratedQualityError, match="status"):
        compute_integrated_quality_metrics(
            altered,
            replay_summary=replay.summary,
            execution_record=execution.record,
        )
