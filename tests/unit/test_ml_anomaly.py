"""Phase 4.4-C tests for leakage-aware advisory anomaly baselines."""
import pytest

from roboqa.ml.anomaly import (
    AnomalyDecision,
    AnomalyDetectionError,
    fit_isolation_forest,
    fit_statistical_baseline,
    score_isolation_forest,
    score_statistical_baseline,
)
from roboqa.ml.contracts import (
    AnomalyReference,
    DatasetSplit,
    FeatureAvailability,
    FeatureDataset,
    FeatureSample,
    FeatureValue,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def row(sample_id, split, signal, *, label=AnomalyReference.UNKNOWN, missing=False):
    feature = FeatureValue(
        "signal", None if missing else signal, "unit",
        FeatureAvailability.MISSING if missing else FeatureAvailability.AVAILABLE,
        0 if missing else 1,
    )
    constant = FeatureValue("constant", 7.0, "unit")
    return FeatureSample(
        sample_id, f"run-{sample_id}", split, (feature, constant),
        reference_label=label,
    )


def dataset(*, validation_value=100.0, label=AnomalyReference.UNKNOWN,
            missing_validation=False):
    train = [
        row(f"train-{index}", DatasetSplit.TRAIN, float(index))
        for index in range(12)
    ]
    validation = row(
        "validation-outlier", DatasetSplit.VALIDATION, validation_value,
        label=label, missing=missing_validation,
    )
    test = row("test-normal", DatasetSplit.TEST, 5.5)
    return FeatureDataset(
        "anomaly-fixture", "1.0", "2026-11-08T09:00:00Z",
        tuple(train + [validation, test]), HASH_A, HASH_B,
    )


def test_statistical_baseline_fits_train_only_and_flags_extreme_validation_sample():
    data = dataset(validation_value=100.0, label=AnomalyReference.NORMAL)
    model = fit_statistical_baseline(data, contamination=0.1)
    predictions = score_statistical_baseline(model, data)
    by_id = {item.sample_id: item for item in predictions}

    assert model.transform.training_sample_ids == tuple(
        f"train-{index}" for index in range(12)
    )
    assert model.transform.feature_names == ("signal",)
    assert by_id["validation-outlier"].decision is AnomalyDecision.ANOMALOUS
    assert by_id["test-normal"].decision is AnomalyDecision.NORMAL
    assert by_id["validation-outlier"].method == "robust_max_z_v1"


def test_statistical_model_hash_is_independent_of_validation_values_and_labels():
    first = fit_statistical_baseline(dataset(validation_value=20.0))
    second = fit_statistical_baseline(
        dataset(validation_value=2000.0, label=AnomalyReference.ANOMALOUS)
    )
    assert first.model_hash == second.model_hash
    assert first.transform.training_data_hash == second.transform.training_data_hash


def test_statistical_scoring_does_not_impute_missing_features():
    data = dataset(missing_validation=True)
    model = fit_statistical_baseline(data)
    prediction = {
        item.sample_id: item for item in score_statistical_baseline(model, data)
    }["validation-outlier"]
    assert prediction.decision is AnomalyDecision.UNSCORABLE
    assert prediction.score is None
    assert "unavailable" in prediction.reason


def test_unavailable_training_feature_is_excluded_and_constant_feature_dropped():
    data = dataset()
    model = fit_statistical_baseline(data)
    assert model.transform.feature_names == ("signal",)


@pytest.mark.parametrize("contamination", [0.0, -0.1, 0.51, float("nan"), True])
def test_rejects_invalid_contamination(contamination):
    with pytest.raises(AnomalyDetectionError, match="contamination"):
        fit_statistical_baseline(dataset(), contamination=contamination)


def test_requires_at_least_three_training_samples():
    data = dataset()
    reduced = FeatureDataset(
        data.dataset_id, data.schema_version, data.created_at,
        data.samples[:2] + data.samples[-2:], data.source_hash, data.configuration_hash,
    )
    with pytest.raises(AnomalyDetectionError, match="three TRAIN"):
        fit_statistical_baseline(reduced)


def test_rejects_training_data_without_variable_available_features():
    samples = tuple(
        FeatureSample(
            f"constant-{index}", f"run-{index}", DatasetSplit.TRAIN,
            (FeatureValue("constant", 1.0, "unit"),),
        )
        for index in range(3)
    )
    data = FeatureDataset(
        "constant", "1", "2026-11-08T09:00:00Z",
        samples, HASH_A, HASH_B,
    )
    with pytest.raises(AnomalyDetectionError, match="constant"):
        fit_statistical_baseline(data)


def test_model_does_not_consume_reference_labels():
    normal = dataset(label=AnomalyReference.NORMAL)
    anomalous = dataset(label=AnomalyReference.ANOMALOUS)
    assert fit_statistical_baseline(normal).model_hash == fit_statistical_baseline(
        anomalous
    ).model_hash


def test_isolation_forest_is_reproducible_and_uses_train_only():
    pytest.importorskip("sklearn")
    data = dataset(validation_value=100.0)
    first = fit_isolation_forest(data, contamination=0.1, random_state=42, n_estimators=50)
    second = fit_isolation_forest(data, contamination=0.1, random_state=42, n_estimators=50)
    assert first.model_hash == second.model_hash
    assert first.transform.training_sample_ids == second.transform.training_sample_ids
    predictions = score_isolation_forest(first, data)
    assert len(predictions) == len(data.samples)
    assert all(item.method == "isolation_forest_v1" for item in predictions)


def test_isolation_forest_rejects_invalid_parameters():
    pytest.importorskip("sklearn")
    with pytest.raises(AnomalyDetectionError, match="random_state"):
        fit_isolation_forest(dataset(), random_state=True)
    with pytest.raises(AnomalyDetectionError, match="n_estimators"):
        fit_isolation_forest(dataset(), n_estimators=0)
