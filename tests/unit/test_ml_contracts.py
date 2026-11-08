"""Contract tests for Phase 4.4-A ML feature datasets."""
import pytest

from roboqa.ml.contracts import (
    AnomalyReference,
    DatasetSplit,
    FeatureAvailability,
    FeatureDataset,
    FeatureDatasetError,
    FeatureSample,
    FeatureValue,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def feature(name="nis_mean", value=2.0, *, unit="dimensionless",
            availability=FeatureAvailability.AVAILABLE, sample_count=4):
    return FeatureValue(name, value, unit, availability, sample_count)


def sample(sample_id="s1", run_id="run-1", split=DatasetSplit.TRAIN, *,
           features=None, scenario_id="nominal", label=AnomalyReference.UNKNOWN,
           window_start=None, window_end=None):
    return FeatureSample(
        sample_id, run_id, split,
        tuple(features if features is not None else [feature()]),
        scenario_id, window_start, window_end, label,
    )


def dataset(samples):
    return FeatureDataset(
        "dataset-v1", "1.0", "2026-11-08T05:00:00Z",
        tuple(samples), HASH_A, HASH_B,
    )


def test_valid_dataset_preserves_immutable_rows_and_evaluation_label():
    item = sample(label=AnomalyReference.ANOMALOUS)
    result = dataset([item])
    assert result.samples == (item,)
    assert result.samples[0].reference_label is AnomalyReference.ANOMALOUS
    assert result.samples[0].features[0].value == 2.0


@pytest.mark.parametrize("availability,value", [
    (FeatureAvailability.AVAILABLE, None),
    (FeatureAvailability.MISSING, 0.0),
    (FeatureAvailability.INVALID, 1.0),
])
def test_feature_availability_must_match_value(availability, value):
    with pytest.raises(FeatureDatasetError):
        feature(value=value, availability=availability)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_feature_values_must_be_finite_non_boolean_scalars(value):
    with pytest.raises(FeatureDatasetError):
        feature(value=value)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_feature_sample_count_must_be_non_negative_integer(count):
    with pytest.raises(FeatureDatasetError):
        feature(sample_count=count)


def test_missing_and_invalid_features_remain_explicit():
    item = sample(features=[
        feature("nis_mean", None, availability=FeatureAvailability.MISSING),
        feature("latency_mean", None, unit="s", availability=FeatureAvailability.INVALID),
    ])
    result = dataset([item])
    assert [x.availability for x in result.samples[0].features] == [
        FeatureAvailability.MISSING, FeatureAvailability.INVALID
    ]
    assert [x.value for x in result.samples[0].features] == [None, None]


def test_sample_rejects_empty_or_duplicate_features():
    with pytest.raises(FeatureDatasetError):
        sample(features=[])
    with pytest.raises(FeatureDatasetError):
        sample(features=[feature(), feature()])


def test_sample_rejects_invalid_identity_split_and_label():
    with pytest.raises(FeatureDatasetError):
        FeatureSample("", "run", DatasetSplit.TRAIN, (feature(),))
    with pytest.raises(FeatureDatasetError):
        FeatureSample("s", "run", "train", (feature(),))
    with pytest.raises(FeatureDatasetError):
        FeatureSample("s", "run", DatasetSplit.TRAIN, (feature(),),
                      reference_label="anomalous")


def test_sample_window_bounds_must_be_paired_and_ordered():
    with pytest.raises(FeatureDatasetError):
        sample(window_start=1.0)
    with pytest.raises(FeatureDatasetError):
        sample(window_start=2.0, window_end=1.0)
    with pytest.raises(FeatureDatasetError):
        sample(window_start=float("nan"), window_end=2.0)


def test_dataset_rejects_empty_samples_and_non_sample_items():
    with pytest.raises(FeatureDatasetError):
        dataset([])
    with pytest.raises(FeatureDatasetError):
        dataset([object()])


def test_dataset_rejects_duplicate_sample_ids():
    with pytest.raises(FeatureDatasetError):
        dataset([sample("same", "run-a"), sample("same", "run-b")])


def test_dataset_prevents_run_level_split_leakage():
    with pytest.raises(FeatureDatasetError, match="multiple dataset splits"):
        dataset([
            sample("train-row", "same-run", DatasetSplit.TRAIN),
            sample("test-row", "same-run", DatasetSplit.TEST),
        ])


def test_dataset_allows_multiple_windows_from_one_run_in_same_split():
    result = dataset([
        sample("window-1", "run-a", window_start=0.0, window_end=1.0),
        sample("window-2", "run-a", window_start=1.0, window_end=2.0),
    ])
    assert len(result.samples) == 2


def test_dataset_rejects_inconsistent_units_for_same_feature():
    with pytest.raises(FeatureDatasetError, match="inconsistent units"):
        dataset([
            sample("s1", "run-a", features=[feature(unit="dimensionless")]),
            sample("s2", "run-b", split=DatasetSplit.TEST,
                   features=[feature(unit="s")]),
        ])


@pytest.mark.parametrize("timestamp", [
    "not-a-date", "2026-11-08T05:00:00", "2026-11-08T05:00:00+03:00",
])
def test_dataset_timestamp_must_be_valid_utc(timestamp):
    with pytest.raises(FeatureDatasetError):
        FeatureDataset("d", "1", timestamp, (sample(),), HASH_A, HASH_B)


@pytest.mark.parametrize("digest", ["", "xyz", "a" * 63, "g" * 64])
def test_dataset_hashes_must_be_sha256(digest):
    with pytest.raises(FeatureDatasetError):
        FeatureDataset("d", "1", "2026-11-08T05:00:00Z",
                       (sample(),), digest, HASH_B)


def test_dataset_normalizes_hashes_to_lowercase():
    result = FeatureDataset("d", "1", "2026-11-08T05:00:00Z",
                            (sample(),), HASH_A.upper(), HASH_B.upper())
    assert result.source_hash == HASH_A
    assert result.configuration_hash == HASH_B


def test_dataset_rejects_non_iterable_samples():
    with pytest.raises(FeatureDatasetError):
        FeatureDataset("d", "1", "2026-11-08T05:00:00Z",
                       None, HASH_A, HASH_B)

def test_dataset_rejects_omitted_feature_instead_of_explicit_missing_value():
    with pytest.raises(FeatureDatasetError, match="same feature names"):
        dataset([
            sample("s1", "run-a", features=[
                feature("nis_mean"),
                feature("latency_mean", 0.2, unit="s"),
            ]),
            sample("s2", "run-b", split=DatasetSplit.TEST,
                   features=[feature("nis_mean")]),
        ])
