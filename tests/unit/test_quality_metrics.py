"""Unit tests for Phase 4.2 quality metric calculations."""

import numpy as np
import pytest

from roboqa.data.schemas import GroundTruth
from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.fusion.result import FilterDiagnostics, FilterResult, FilterStatus, InnovationTelemetry
from roboqa.qa import MetricAvailability
from roboqa.qa.metrics import QualityMetricError, compute_quality_metrics


def _record(t, sensor, innovation, nis):
    dof = len(innovation)
    return InnovationTelemetry(t, sensor, dof, np.asarray(innovation, dtype=float), np.eye(dof), nis)


def _result():
    telemetry = (_record(0.0, "gps", [3.0, 4.0], 5.0),
                 _record(0.0, "speed", [2.0], 2.0),
                 _record(1.0, "gps", [0.0, 2.0], 1.0))
    return FilterResult(
        t=np.array([0.0, 0.0, 0.0, 1.0]),
        x=np.array([[0, 0, 0, 0, 0], [1, 1, 0, 0, 0],
                    [2, 2, 0, 0, 0], [3, 3, 0, 0, 0]], dtype=float),
        P=np.repeat(np.eye(5)[None, :, :], 4, axis=0),
        status=FilterStatus.OK,
        diagnostics=FilterDiagnostics(n_update=3, n_rejected=1),
        telemetry=telemetry,
    )


def _ground_truth():
    return GroundTruth(
        t=np.array([0.0, 1.0, 2.0]),
        pos=np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]),
        yaw=np.zeros(3), speed=np.ones(3), yaw_rate=np.zeros(3), source="synthetic",
    )


def test_run_and_sensor_metrics_are_calculated_with_explicit_sample_counts():
    metrics = compute_quality_metrics(
        _result(), ground_truth=_ground_truth(),
        expected_measurements={"gps": 4, "speed": 2, "yaw_rate": 1},
    )
    by_key = {(m.name, m.sensor_id): m for m in metrics}
    assert by_key[("position_rmse", None)].value == pytest.approx(np.sqrt(5.0))
    assert by_key[("covariance_trace_mean", None)].value == pytest.approx(5.0)
    assert by_key[("innovation_norm_mean", "gps")].value == pytest.approx(4.0)
    assert by_key[("nis_mean", "gps")].value == pytest.approx(3.0)
    assert by_key[("sensor_availability", "gps")].value == pytest.approx(0.5)
    assert by_key[("sensor_availability", "speed")].value == pytest.approx(0.5)
    assert by_key[("temporal_rejection_rate", None)].value == pytest.approx(0.25)


def test_duplicate_filter_timestamps_are_not_overweighted_in_rmse():
    metrics = compute_quality_metrics(_result(), ground_truth=_ground_truth())
    rmse = next(m for m in metrics if m.name == "position_rmse")
    assert rmse.sample_count == 2


def test_missing_ground_truth_and_expected_counts_remain_missing_not_zero():
    metrics = compute_quality_metrics(_result())
    rmse = next(m for m in metrics if m.name == "position_rmse")
    availability = next(m for m in metrics if m.name == "sensor_availability")
    assert rmse.availability is MetricAvailability.MISSING and rmse.value is None
    assert availability.availability is MetricAvailability.MISSING and availability.value is None


def test_rmse_outside_alignment_tolerance_is_missing():
    shifted = GroundTruth(
        t=np.array([0.1, 1.1, 2.1]),
        pos=np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]),
        yaw=np.zeros(3), speed=np.ones(3), yaw_rate=np.zeros(3), source="synthetic",
    )
    metrics = compute_quality_metrics(_result(), ground_truth=shifted,
                                      alignment_tolerance_s=0.001)
    rmse = next(m for m in metrics if m.name == "position_rmse")
    assert rmse.availability is MetricAvailability.MISSING


def test_latency_metrics_use_three_clocks_and_preserve_invalid_evidence():
    envelopes = [
        RawMeasurementEnvelope("gps", 1.0, 1.2, 0, [1.0, 2.0], processing_time=1.3),
        RawMeasurementEnvelope("gps", 2.0, 2.4, 1, [2.0, 3.0], processing_time=2.7),
    ]
    metrics = compute_quality_metrics(_result(), envelopes=envelopes)
    by_name = {m.name: m for m in metrics if m.sensor_id == "gps"}
    assert by_name["transport_latency_mean"].value == pytest.approx(0.3)
    assert by_name["processing_wait_mean"].value == pytest.approx(0.2)
    assert by_name["end_to_end_latency_mean"].value == pytest.approx(0.5)


def test_latency_is_invalid_when_temporal_values_are_negative():
    envelopes = [RawMeasurementEnvelope("gps", 2.0, 1.0, 0, [1.0, 2.0], processing_time=3.0)]
    metrics = compute_quality_metrics(_result(), envelopes=envelopes)
    latency = next(m for m in metrics if m.name == "transport_latency_mean" and m.sensor_id == "gps")
    assert latency.availability is MetricAvailability.INVALID and latency.value is None


def test_processing_latency_is_missing_when_processing_clock_is_absent():
    envelopes = [RawMeasurementEnvelope("gps", 1.0, 1.1, 0, [1.0, 2.0])]
    metrics = compute_quality_metrics(_result(), envelopes=envelopes)
    processing = next(m for m in metrics if m.name == "processing_wait_mean" and m.sensor_id == "gps")
    assert processing.availability is MetricAvailability.MISSING


def test_availability_rejects_more_accepted_updates_than_expected():
    metrics = compute_quality_metrics(_result(), expected_measurements={"gps": 1})
    availability = next(m for m in metrics if m.name == "sensor_availability" and m.sensor_id == "gps")
    assert availability.availability is MetricAvailability.INVALID and availability.value is None


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf, True])
def test_alignment_tolerance_must_be_finite_and_non_negative(tolerance):
    with pytest.raises(QualityMetricError):
        compute_quality_metrics(_result(), alignment_tolerance_s=tolerance)


def test_invalid_result_shape_fails_closed():
    result = _result()
    result.P = np.zeros((1, 5, 5))
    with pytest.raises(QualityMetricError):
        compute_quality_metrics(result)
