"""Deterministic quality metric calculations (Phase 4.2).

Only metrics with explicit denominators/alignment are reported as available.
Missing inputs remain missing; malformed evidence is marked invalid.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite, sqrt

import numpy as np

from roboqa.data.schemas import GroundTruth
from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.fusion.result import FilterResult
from roboqa.qa.contracts import MetricAvailability, MetricObservation, MetricScope


class QualityMetricError(ValueError):
    """Raised when metric inputs violate the evaluator API contract."""


def _metric(name, value, unit, scope, sample_count, availability=MetricAvailability.AVAILABLE, *, sensor_id=None):
    return MetricObservation(
        name=name, value=value, unit=unit, scope=scope, sample_count=sample_count,
        availability=availability, sensor_id=sensor_id,
    )


def _validate_result(result: FilterResult) -> None:
    if not isinstance(result, FilterResult):
        raise QualityMetricError("result must be a FilterResult")
    if result.t.ndim != 1 or result.x.ndim != 2 or result.P.ndim != 3:
        raise QualityMetricError("FilterResult arrays have invalid dimensions")
    if result.x.shape[0] != result.t.size or result.P.shape[0] != result.t.size:
        raise QualityMetricError("FilterResult t, x, and P lengths must match")
    if result.x.shape[1:] != (5,) or result.P.shape[1:] != (5, 5):
        raise QualityMetricError("FilterResult state/covariance shapes are invalid")
    if not (np.all(np.isfinite(result.t)) and np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.P))):
        raise QualityMetricError("FilterResult contains non-finite state evidence")
    if result.t.size and np.any(np.diff(result.t) < -1e-12):
        raise QualityMetricError("FilterResult timestamps must be non-decreasing")


def _position_rmse(result, ground_truth, tolerance):
    if ground_truth is None or result.t.size == 0:
        return _metric("position_rmse", None, "m", MetricScope.RUN, 0, MetricAvailability.MISSING)
    if not isinstance(ground_truth, GroundTruth):
        raise QualityMetricError("ground_truth must be a GroundTruth instance")

    # Multiple sensor updates may share a timestamp. Keep the final posterior.
    unique_t, reverse_indices = np.unique(result.t[::-1], return_index=True)
    final_indices = result.t.size - 1 - reverse_indices
    order = np.argsort(unique_t)
    eval_t = unique_t[order]
    eval_pos = result.x[final_indices[order], :2]
    inside = (eval_t >= ground_truth.t[0]) & (eval_t <= ground_truth.t[-1])
    eval_t, eval_pos = eval_t[inside], eval_pos[inside]
    if eval_t.size == 0:
        return _metric("position_rmse", None, "m", MetricScope.RUN, 0, MetricAvailability.MISSING)

    right = np.searchsorted(ground_truth.t, eval_t, side="left")
    left = np.clip(right - 1, 0, ground_truth.t.size - 1)
    right = np.clip(right, 0, ground_truth.t.size - 1)
    nearest = np.minimum(np.abs(eval_t - ground_truth.t[left]), np.abs(eval_t - ground_truth.t[right]))
    aligned = nearest <= tolerance + 1e-12
    eval_t, eval_pos = eval_t[aligned], eval_pos[aligned]
    if eval_t.size == 0:
        return _metric("position_rmse", None, "m", MetricScope.RUN, 0, MetricAvailability.MISSING)

    reference = np.column_stack([
        np.interp(eval_t, ground_truth.t, ground_truth.pos[:, axis]) for axis in range(2)
    ])
    errors = np.linalg.norm(eval_pos - reference, axis=1)
    return _metric("position_rmse", float(sqrt(float(np.mean(np.square(errors))))),
                   "m", MetricScope.RUN, int(errors.size))


def _latency_metric(name, envelopes, sensor_id, attribute):
    selected = [item for item in envelopes if item.sensor_id.strip().lower() == sensor_id]
    if not selected:
        return _metric(name, None, "s", MetricScope.SENSOR, 0,
                       MetricAvailability.MISSING, sensor_id=sensor_id)
    clock_fields = {
        "transport_latency_s": ("acquisition_time", "arrival_time"),
        "processing_wait_s": ("arrival_time", "processing_time"),
        "end_to_end_latency_s": ("acquisition_time", "processing_time"),
    }[attribute]
    if any(
        getattr(item, field) is not None
        and not isfinite(float(getattr(item, field)))
        for item in selected
        for field in clock_fields
    ):
        return _metric(name, None, "s", MetricScope.SENSOR, 0,
                       MetricAvailability.INVALID, sensor_id=sensor_id)
    raw_values = [getattr(item, attribute) for item in selected]
    present = [value for value in raw_values if value is not None]
    if not present:
        return _metric(name, None, "s", MetricScope.SENSOR, 0,
                       MetricAvailability.MISSING, sensor_id=sensor_id)
    if len(present) != len(selected) or any(
        not isfinite(float(value)) or float(value) < 0.0 for value in present
    ):
        return _metric(name, None, "s", MetricScope.SENSOR, len(present),
                       MetricAvailability.INVALID, sensor_id=sensor_id)
    return _metric(name, float(np.mean(np.asarray(present, dtype=np.float64))),
                   "s", MetricScope.SENSOR, len(present), sensor_id=sensor_id)


def compute_quality_metrics(
    result: FilterResult,
    *,
    ground_truth: GroundTruth | None = None,
    envelopes: Sequence[RawMeasurementEnvelope] = (),
    expected_measurements: Mapping[str, int] | None = None,
    alignment_tolerance_s: float = 0.05,
) -> tuple[MetricObservation, ...]:
    """Calculate supported metrics from explicit evidence.

    expected_measurements maps normalized sensor IDs to expected packet counts.
    Availability is never inferred from observed packets. Latency metrics use
    raw envelopes and require complete, finite, non-negative clock deltas.
    RMSE uses interpolated ground truth only within the overlap and alignment
    tolerance. temporal_rejection_rate counts only runner temporal rejections;
    it is not a general update rejection rate.
    """
    _validate_result(result)
    if isinstance(alignment_tolerance_s, bool):
        raise QualityMetricError("alignment_tolerance_s must be finite and >= 0")
    try:
        tolerance = float(alignment_tolerance_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QualityMetricError("alignment_tolerance_s must be finite and >= 0") from exc
    if not isfinite(tolerance) or tolerance < 0.0:
        raise QualityMetricError("alignment_tolerance_s must be finite and >= 0")

    if not isinstance(envelopes, Sequence) or isinstance(envelopes, (str, bytes)):
        raise QualityMetricError("envelopes must be a sequence")
    if any(not isinstance(item, RawMeasurementEnvelope) for item in envelopes):
        raise QualityMetricError("envelopes must contain RawMeasurementEnvelope instances")
    envelope_items = tuple(envelopes)

    expected = None
    if expected_measurements is not None:
        if not isinstance(expected_measurements, Mapping):
            raise QualityMetricError("expected_measurements must be a mapping")
        expected = {}
        for sensor, count in expected_measurements.items():
            if not isinstance(sensor, str) or not sensor.strip():
                raise QualityMetricError("expected sensor IDs must be non-empty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise QualityMetricError("expected counts must be non-negative integers")
            key = sensor.strip().lower()
            if key in expected:
                raise QualityMetricError("expected sensor IDs must be unique after normalization")
            expected[key] = count

    observations = [_position_rmse(result, ground_truth, tolerance)]
    if result.P.shape[0] == 0:
        observations.append(_metric("covariance_trace_mean", None, "state_covariance_units",
                                     MetricScope.RUN, 0, MetricAvailability.MISSING))
    else:
        traces = np.trace(result.P, axis1=1, axis2=2)
        if not np.all(np.isfinite(traces)) or np.any(traces < -1e-10):
            observations.append(_metric("covariance_trace_mean", None, "state_covariance_units",
                                         MetricScope.RUN, int(traces.size), MetricAvailability.INVALID))
        else:
            observations.append(_metric("covariance_trace_mean",
                                         float(np.mean(np.maximum(traces, 0.0))),
                                         "state_covariance_units", MetricScope.RUN, int(traces.size)))

    by_sensor = defaultdict(list)
    for record in result.telemetry:
        by_sensor[record.sensor_id].append(record)
    sensor_ids = set(by_sensor)
    sensor_ids.update(item.sensor_id.strip().lower() for item in envelope_items)
    if expected is not None:
        sensor_ids.update(expected)

    for sensor_id in sorted(sensor_ids):
        records = by_sensor.get(sensor_id, [])
        if records:
            norms = [float(np.linalg.norm(item.innovation)) for item in records]
            nis_values = [item.nis for item in records]
            observations.append(_metric("innovation_norm_mean", float(np.mean(norms)),
                                         "sensor_units", MetricScope.SENSOR, len(norms), sensor_id=sensor_id))
            observations.append(_metric("nis_mean", float(np.mean(nis_values)),
                                         "dimensionless", MetricScope.SENSOR, len(nis_values), sensor_id=sensor_id))
        else:
            for name, unit in (("innovation_norm_mean", "sensor_units"), ("nis_mean", "dimensionless")):
                observations.append(_metric(name, None, unit, MetricScope.SENSOR, 0,
                                             MetricAvailability.MISSING, sensor_id=sensor_id))

        if expected is None or sensor_id not in expected or expected[sensor_id] == 0:
            observations.append(_metric("sensor_availability", None, "ratio", MetricScope.SENSOR,
                                         len(records), MetricAvailability.MISSING, sensor_id=sensor_id))
        elif len(records) > expected[sensor_id]:
            observations.append(_metric("sensor_availability", None, "ratio", MetricScope.SENSOR,
                                         len(records), MetricAvailability.INVALID, sensor_id=sensor_id))
        else:
            observations.append(_metric("sensor_availability", len(records) / expected[sensor_id],
                                         "ratio", MetricScope.SENSOR, expected[sensor_id], sensor_id=sensor_id))

        for metric_name, attr in (
            ("transport_latency_mean", "transport_latency_s"),
            ("processing_wait_mean", "processing_wait_s"),
            ("end_to_end_latency_mean", "end_to_end_latency_s"),
        ):
            observations.append(_latency_metric(metric_name, envelope_items, sensor_id, attr))

    accepted = result.diagnostics.n_update
    temporal_rejected = result.diagnostics.n_rejected
    denominator = accepted + temporal_rejected
    if denominator == 0:
        observations.append(_metric("temporal_rejection_rate", None, "ratio",
                                     MetricScope.RUN, 0, MetricAvailability.MISSING))
    else:
        observations.append(_metric("temporal_rejection_rate", temporal_rejected / denominator,
                                     "ratio", MetricScope.RUN, denominator))
    return tuple(observations)
