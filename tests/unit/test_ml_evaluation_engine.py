"""Phase 4.4-D1 deterministic anomaly evaluation engine tests."""
import pytest

from roboqa.ml.anomaly import AnomalyDecision, AnomalyPrediction
from roboqa.ml.contracts import (
    AnomalyReference, DatasetSplit, FeatureDataset, FeatureSample, FeatureValue,
)
from roboqa.ml.evaluation import AnomalyEvaluationError, EvaluationScope
from roboqa.ml.evaluation_engine import evaluate_anomaly_predictions


HASH_A = "a" * 64
HASH_B = "b" * 64
GENERATED = "2026-11-08T20:00:00Z"


def make_dataset():
    rows = [
        ("t1", "train-1", DatasetSplit.TRAIN, AnomalyReference.NORMAL, "base"),
        ("t2", "train-1", DatasetSplit.TRAIN, AnomalyReference.ANOMALOUS, "base"),
        ("v1", "run-a", DatasetSplit.VALIDATION, AnomalyReference.NORMAL, "nominal"),
        ("v2", "run-a", DatasetSplit.VALIDATION, AnomalyReference.NORMAL, "bias"),
        ("v3", "run-b", DatasetSplit.VALIDATION, AnomalyReference.ANOMALOUS, "bias"),
        ("v4", "run-b", DatasetSplit.VALIDATION, AnomalyReference.ANOMALOUS, "dropout"),
        ("v5", "run-b", DatasetSplit.VALIDATION, AnomalyReference.UNKNOWN, "dropout"),
        ("v6", "run-b", DatasetSplit.VALIDATION, AnomalyReference.NORMAL, "dropout"),
        ("x1", "test-1", DatasetSplit.TEST, AnomalyReference.NORMAL, "nominal"),
    ]
    samples = tuple(
        FeatureSample(
            sample_id=sample_id, run_id=run_id, split=split,
            scenario_id=scenario,
            features=(FeatureValue("nis_mean", 1.0, "dimensionless"),),
            reference_label=label,
        )
        for sample_id, run_id, split, label, scenario in rows
    )
    return FeatureDataset(
        "dataset-v1", "1", GENERATED, samples, HASH_A, HASH_B,
    )


def make_predictions(dataset):
    decisions = {
        "t1": AnomalyDecision.NORMAL,
        "t2": AnomalyDecision.ANOMALOUS,
        "v1": AnomalyDecision.NORMAL,
        "v2": AnomalyDecision.ANOMALOUS,
        "v3": AnomalyDecision.ANOMALOUS,
        "v4": AnomalyDecision.NORMAL,
        "v5": AnomalyDecision.NORMAL,
        "v6": AnomalyDecision.UNSCORABLE,
        "x1": AnomalyDecision.NORMAL,
    }
    return tuple(
        AnomalyPrediction(
            sample.sample_id, sample.run_id, sample.split, "test_detector_v1",
            None if decisions[sample.sample_id] is AnomalyDecision.UNSCORABLE else 0.5,
            0.4, decisions[sample.sample_id], ("nis_mean",),
            "missing feature" if decisions[sample.sample_id] is AnomalyDecision.UNSCORABLE else None,
        )
        for sample in dataset.samples
    )


def test_evaluation_builds_overall_run_and_scenario_slices():
    dataset = make_dataset()
    report = evaluate_anomaly_predictions(
        dataset, make_predictions(dataset),
        evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
    )
    assert report.dataset_id == "dataset-v1"
    assert report.evaluation_split is DatasetSplit.VALIDATION
    assert report.method == "test_detector_v1"
    assert [item.scope for item in report.slices].count(EvaluationScope.OVERALL) == 1
    assert [item.scope for item in report.slices].count(EvaluationScope.RUN) == 2
    assert [item.scope for item in report.slices].count(EvaluationScope.SCENARIO) == 3
    overall = report.slices[0]
    assert overall.counts.total_samples == 6
    assert overall.counts.labeled_samples == 5
    assert overall.counts.unknown_samples == 1
    assert overall.counts.scored_labeled_samples == 4
    assert overall.counts.unscorable_labeled_samples == 1
    assert overall.counts.scored_unknown_samples == 1
    assert overall.counts.unscorable_unknown_samples == 0
    assert overall.confusion_matrix.true_positive == 1
    assert overall.confusion_matrix.false_positive == 1
    assert overall.confusion_matrix.true_negative == 1
    assert overall.confusion_matrix.false_negative == 1
    assert overall.metrics.precision == pytest.approx(0.5)
    assert overall.metrics.recall == pytest.approx(0.5)
    assert overall.metrics.f1 == pytest.approx(0.5)
    assert overall.metrics.balanced_accuracy == pytest.approx(0.5)


def test_test_split_evaluation_excludes_validation_rows():
    dataset = make_dataset()
    report = evaluate_anomaly_predictions(
        dataset, make_predictions(dataset),
        evaluation_split=DatasetSplit.TEST, generated_at=GENERATED,
    )
    assert report.slices[0].counts.total_samples == 1
    assert report.slices[0].confusion_matrix.true_negative == 1


def test_hashes_are_deterministic():
    dataset = make_dataset()
    first = evaluate_anomaly_predictions(
        dataset, make_predictions(dataset),
        evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
    )
    second = evaluate_anomaly_predictions(
        dataset, make_predictions(dataset),
        evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
    )
    assert first.dataset_hash == second.dataset_hash
    assert first.prediction_hash == second.prediction_hash


@pytest.mark.parametrize("split", [DatasetSplit.TRAIN, "validation"])
def test_rejects_invalid_evaluation_split(split):
    dataset = make_dataset()
    with pytest.raises(AnomalyEvaluationError):
        evaluate_anomaly_predictions(
            dataset, make_predictions(dataset),
            evaluation_split=split, generated_at=GENERATED,
        )


def test_rejects_missing_prediction():
    dataset = make_dataset()
    with pytest.raises(AnomalyEvaluationError, match="missing predictions"):
        evaluate_anomaly_predictions(
            dataset, make_predictions(dataset)[:-1],
            evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
        )


def test_rejects_duplicate_prediction_identity():
    dataset = make_dataset()
    predictions = make_predictions(dataset)
    with pytest.raises(AnomalyEvaluationError, match="duplicate prediction"):
        evaluate_anomaly_predictions(
            dataset, predictions + (predictions[0],),
            evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
        )


def test_rejects_prediction_for_unknown_sample():
    dataset = make_dataset()
    predictions = list(make_predictions(dataset))
    predictions[-1] = AnomalyPrediction(
        "not-in-dataset", "test-1", DatasetSplit.TEST, "test_detector_v1",
        0.1, 0.4, AnomalyDecision.NORMAL, ("nis_mean",),
    )
    with pytest.raises(AnomalyEvaluationError, match="not present"):
        evaluate_anomaly_predictions(
            dataset, predictions,
            evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
        )


def test_rejects_run_or_split_identity_mismatch():
    dataset = make_dataset()
    predictions = list(make_predictions(dataset))
    original = predictions[2]
    predictions[2] = AnomalyPrediction(
        original.sample_id, "wrong-run", original.split, original.method,
        original.score, original.threshold, original.decision, original.feature_names,
    )
    with pytest.raises(AnomalyEvaluationError, match="identity mismatch"):
        evaluate_anomaly_predictions(
            dataset, predictions,
            evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
        )


def test_rejects_mixed_detector_methods():
    dataset = make_dataset()
    predictions = list(make_predictions(dataset))
    original = predictions[-1]
    predictions[-1] = AnomalyPrediction(
        original.sample_id, original.run_id, original.split, "other_detector",
        original.score, original.threshold, original.decision, original.feature_names,
    )
    with pytest.raises(AnomalyEvaluationError, match="exactly one"):
        evaluate_anomaly_predictions(
            dataset, predictions,
            evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
        )


def test_rejects_empty_evaluation_split():
    dataset = make_dataset()
    with pytest.raises(AnomalyEvaluationError, match="no samples"):
        evaluate_anomaly_predictions(
            dataset, make_predictions(dataset),
            evaluation_split=DatasetSplit.VALIDATION, generated_at=GENERATED,
        ) if not any(s.split is DatasetSplit.VALIDATION for s in dataset.samples) else None
