"""Leakage-aware temporal feature extraction from immutable Fusion telemetry (Phase 4.4-B)."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from roboqa.fusion.result import InnovationTelemetry
from roboqa.ml.contracts import (
    AnomalyReference,
    DatasetSplit,
    FeatureAvailability,
    FeatureDataset,
    FeatureDatasetError,
    FeatureSample,
    FeatureValue,
)


class FeatureEngineeringError(ValueError):
    """Raised when telemetry cannot be converted into a valid feature dataset."""


_FEATURE_UNITS = {
    "update_count": "count",
    "innovation_norm_mean": "sensor_units",
    "innovation_norm_std": "sensor_units",
    "innovation_norm_max": "sensor_units",
    "nis_mean": "dimensionless",
    "nis_std": "dimensionless",
    "nis_max": "dimensionless",
    "innovation_covariance_trace_mean": "measurement_covariance_units",
    "inter_update_gap_mean": "s",
    "inter_update_gap_max": "s",
}


def _digest(document: object) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _telemetry_document(item: InnovationTelemetry) -> dict[str, object]:
    return {
        "timestamp": item.timestamp,
        "sensor_id": item.sensor_id,
        "measurement_dof": item.measurement_dof,
        "innovation": item.innovation.tolist(),
        "innovation_covariance": item.innovation_covariance.tolist(),
        "nis": item.nis,
    }


def _available(name: str, value: float, count: int) -> FeatureValue:
    return FeatureValue(
        name=name, value=float(value), unit=_FEATURE_UNITS[name],
        availability=FeatureAvailability.AVAILABLE, sample_count=count,
    )


def _missing(name: str) -> FeatureValue:
    return FeatureValue(
        name=name, value=None, unit=_FEATURE_UNITS[name],
        availability=FeatureAvailability.MISSING, sample_count=0,
    )


def extract_temporal_features(
    telemetry: Sequence[InnovationTelemetry],
    *,
    run_id: str,
    split: DatasetSplit,
    dataset_id: str,
    schema_version: str,
    created_at: str,
    window_duration_s: float,
    scenario_id: str | None = None,
    reference_label: AnomalyReference = AnomalyReference.UNKNOWN,
) -> FeatureDataset:
    """Create per-sensor, fixed-duration window features from Fusion telemetry.

    Windows are anchored at the earliest telemetry timestamp. Every observed
    sensor receives one sample per window, including explicit MISSING feature
    values for windows without updates. No interpolation, imputation, scaling,
    clipping, or model fitting is performed.
    """
    if not isinstance(telemetry, Sequence) or isinstance(telemetry, (str, bytes)):
        raise FeatureEngineeringError("telemetry must be a sequence")
    records = tuple(telemetry)
    if not records:
        raise FeatureEngineeringError("telemetry must contain at least one record")
    if any(not isinstance(item, InnovationTelemetry) for item in records):
        raise FeatureEngineeringError("telemetry must contain InnovationTelemetry records")
    if not isinstance(split, DatasetSplit):
        raise FeatureEngineeringError("split must be a DatasetSplit")
    if not isinstance(reference_label, AnomalyReference):
        raise FeatureEngineeringError("reference_label must be an AnomalyReference")
    if isinstance(window_duration_s, bool):
        raise FeatureEngineeringError("window_duration_s must be finite and > 0")
    try:
        duration = float(window_duration_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureEngineeringError("window_duration_s must be finite and > 0") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise FeatureEngineeringError("window_duration_s must be finite and > 0")

    ordered = tuple(sorted(records, key=lambda item: (item.timestamp, item.sensor_id)))
    origin = ordered[0].timestamp
    final_time = ordered[-1].timestamp
    window_count = int(math.floor((final_time - origin) / duration)) + 1
    by_sensor_window: dict[tuple[str, int], list[InnovationTelemetry]] = defaultdict(list)
    sensors = sorted({item.sensor_id for item in ordered})

    for item in ordered:
        index = int(math.floor((item.timestamp - origin) / duration))
        by_sensor_window[(item.sensor_id, index)].append(item)

    samples = []
    for sensor_id in sensors:
        for window_index in range(window_count):
            start = origin + window_index * duration
            end = start + duration
            window_records = by_sensor_window.get((sensor_id, window_index), [])
            if not window_records:
                features = (
                    _available("update_count", 0.0, 0),
                    *(_missing(name) for name in tuple(_FEATURE_UNITS)[1:]),
                )
            else:
                norms = np.asarray(
                    [float(np.linalg.norm(item.innovation)) for item in window_records],
                    dtype=np.float64,
                )
                nis = np.asarray([item.nis for item in window_records], dtype=np.float64)
                traces = np.asarray(
                    [float(np.trace(item.innovation_covariance)) for item in window_records],
                    dtype=np.float64,
                )
                timestamps = np.asarray(
                    [item.timestamp for item in window_records], dtype=np.float64,
                )
                gaps = np.diff(timestamps)
                values = {
                    "update_count": float(len(window_records)),
                    "innovation_norm_mean": float(np.mean(norms)),
                    "innovation_norm_std": float(np.std(norms, ddof=0)),
                    "innovation_norm_max": float(np.max(norms)),
                    "nis_mean": float(np.mean(nis)),
                    "nis_std": float(np.std(nis, ddof=0)),
                    "nis_max": float(np.max(nis)),
                    "innovation_covariance_trace_mean": float(np.mean(traces)),
                }
                features = tuple(
                    _available(name, values[name], len(window_records))
                    for name in tuple(_FEATURE_UNITS)[:8]
                )
                if gaps.size:
                    features += (
                        _available("inter_update_gap_mean", float(np.mean(gaps)), int(gaps.size)),
                        _available("inter_update_gap_max", float(np.max(gaps)), int(gaps.size)),
                    )
                else:
                    features += (
                        _missing("inter_update_gap_mean"),
                        _missing("inter_update_gap_max"),
                    )

            sample_id = f"{run_id}:{sensor_id}:window-{window_index:06d}"
            samples.append(FeatureSample(
                sample_id=sample_id,
                run_id=run_id,
                split=split,
                features=tuple(features),
                scenario_id=scenario_id,
                window_start=start,
                window_end=end,
                reference_label=reference_label,
            ))

    source_hash = _digest([_telemetry_document(item) for item in ordered])
    configuration_hash = _digest({
        "feature_schema": "roboqa-temporal-telemetry-v1",
        "window_duration_s": duration,
        "window_anchor": "earliest_telemetry_timestamp",
        "grouping": "sensor_id",
        "statistics": "population_std",
        "split": split.value,
        "scenario_id": scenario_id,
        "reference_label": reference_label.value,
    })
    try:
        return FeatureDataset(
            dataset_id=dataset_id,
            schema_version=schema_version,
            created_at=created_at,
            samples=tuple(samples),
            source_hash=source_hash,
            configuration_hash=configuration_hash,
        )
    except FeatureDatasetError as exc:
        raise FeatureEngineeringError(f"generated feature dataset is invalid: {exc}") from exc
