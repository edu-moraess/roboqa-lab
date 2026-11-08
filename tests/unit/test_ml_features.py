"""Phase 4.4-B tests for deterministic, sensor-scoped temporal feature extraction."""
import numpy as np
import pytest

from roboqa.fusion.result import InnovationTelemetry
from roboqa.ml import (
    AnomalyReference,
    DatasetSplit,
    FeatureAvailability,
)
from roboqa.ml.features import FeatureEngineeringError, extract_temporal_features


def record(timestamp, sensor, innovation, nis, covariance):
    return InnovationTelemetry(
        timestamp=timestamp,
        sensor_id=sensor,
        measurement_dof=len(innovation),
        innovation=np.asarray(innovation, dtype=float),
        innovation_covariance=np.asarray(covariance, dtype=float),
        nis=nis,
    )


def sample_map(dataset):
    return {sample.sample_id: {item.name: item for item in sample.features}
            for sample in dataset.samples}


def build(records, **overrides):
    options = {
        "run_id": "run-001",
        "split": DatasetSplit.TRAIN,
        "dataset_id": "dataset-001",
        "schema_version": "1.0",
        "created_at": "2026-11-08T09:00:00Z",
        "window_duration_s": 1.0,
        "scenario_id": "nominal",
    }
    options.update(overrides)
    return extract_temporal_features(records, **options)


def test_extracts_sensor_scoped_window_statistics_and_explicit_missing_values():
    telemetry = (
        record(0.0, "gps", (3.0, 4.0), 2.0, ((2.0, 0.0), (0.0, 2.0))),
        record(0.5, "gps", (0.0, 2.0), 4.0, ((1.0, 0.0), (0.0, 1.0))),
        record(1.2, "speed", (2.0,), 1.5, ((0.5,),)),
    )
    dataset = build(telemetry)
    by_sample = sample_map(dataset)

    gps_0 = by_sample["run-001:gps:window-000000"]
    assert gps_0["update_count"].value == 2.0
    assert gps_0["innovation_norm_mean"].value == pytest.approx(3.5)
    assert gps_0["innovation_norm_std"].value == pytest.approx(1.5)
    assert gps_0["innovation_norm_max"].value == 5.0
    assert gps_0["nis_mean"].value == 3.0
    assert gps_0["innovation_covariance_trace_mean"].value == 3.0
    assert gps_0["inter_update_gap_mean"].value == 0.5

    gps_1 = by_sample["run-001:gps:window-000001"]
    assert gps_1["update_count"].value == 0.0
    assert gps_1["innovation_norm_mean"].availability is FeatureAvailability.MISSING
    assert gps_1["innovation_norm_mean"].value is None

    speed_1 = by_sample["run-001:speed:window-000001"]
    assert speed_1["update_count"].value == 1.0
    assert speed_1["inter_update_gap_mean"].availability is FeatureAvailability.MISSING


def test_feature_dataset_hashes_are_deterministic_and_bind_configuration():
    telemetry = (
        record(0.0, "gps", (1.0, 2.0), 0.5, ((1.0, 0.0), (0.0, 1.0))),
        record(0.2, "gps", (2.0, 1.0), 1.0, ((1.0, 0.0), (0.0, 1.0))),
    )
    first = build(telemetry)
    second = build(tuple(reversed(telemetry)))
    changed = build(telemetry, window_duration_s=2.0)

    assert first.source_hash == second.source_hash
    assert first.configuration_hash == second.configuration_hash
    assert first.samples == second.samples
    assert first.source_hash != changed.source_hash or first.configuration_hash != changed.configuration_hash
    assert first.configuration_hash != changed.configuration_hash


def test_reference_label_is_metadata_not_a_feature():
    telemetry = (
        record(0.0, "gps", (1.0, 0.0), 0.5, ((1.0, 0.0), (0.0, 1.0))),
    )
    dataset = build(
        telemetry,
        reference_label=AnomalyReference.ANOMALOUS,
    )
    sample = dataset.samples[0]
    assert sample.reference_label is AnomalyReference.ANOMALOUS
    assert "reference_label" not in {item.name for item in sample.features}


def test_each_run_is_assigned_to_explicit_split_and_windows_are_sensor_scoped():
    telemetry = (
        record(0.0, "gps", (1.0, 0.0), 0.5, ((1.0, 0.0), (0.0, 1.0))),
        record(0.2, "speed", (1.0,), 0.5, ((1.0,),)),
    )
    dataset = build(telemetry, split=DatasetSplit.TEST)
    assert {sample.split for sample in dataset.samples} == {DatasetSplit.TEST}
    assert {sample.run_id for sample in dataset.samples} == {"run-001"}
    assert {sample.scenario_id for sample in dataset.samples} == {"nominal"}


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, "bad"])
def test_rejects_invalid_window_duration(duration):
    telemetry = (
        record(0.0, "gps", (1.0, 0.0), 0.5, ((1.0, 0.0), (0.0, 1.0))),
    )
    with pytest.raises(FeatureEngineeringError, match="window_duration_s"):
        build(telemetry, window_duration_s=duration)


def test_rejects_empty_or_invalid_telemetry():
    with pytest.raises(FeatureEngineeringError, match="at least one"):
        build(())
    with pytest.raises(FeatureEngineeringError, match="InnovationTelemetry"):
        build((object(),))


def test_rejects_invalid_split_and_does_not_mutate_telemetry():
    original = np.array([3.0, 4.0])
    telemetry = (
        record(0.0, "gps", original, 2.0, ((2.0, 0.0), (0.0, 2.0))),
    )
    before = telemetry[0].innovation.copy()
    with pytest.raises(FeatureEngineeringError, match="DatasetSplit"):
        build(telemetry, split="train")
    build(telemetry)
    np.testing.assert_array_equal(telemetry[0].innovation, before)
