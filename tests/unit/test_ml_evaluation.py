"""Phase 4.4-D0 tests for immutable anomaly evaluation contracts."""
import pytest

from roboqa.ml.contracts import DatasetSplit
from roboqa.ml.evaluation import (
    AnomalyClassificationMetrics,
    AnomalyConfusionMatrix,
    AnomalyEvaluationCounts,
    AnomalyEvaluationError,
    AnomalyEvaluationReport,
    AnomalyEvaluationSlice,
    EvaluationScope,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def counts():
    return AnomalyEvaluationCounts(
        total_samples=12,
        labeled_samples=10,
        unknown_samples=2,
        scored_labeled_samples=8,
        unscorable_labeled_samples=2,
        scored_unknown_samples=1,
        unscorable_unknown_samples=1,
    )


def overall():
    return AnomalyEvaluationSlice(
        EvaluationScope.OVERALL,
        None,
        counts(),
        AnomalyConfusionMatrix(2, 1, 4, 1),
        AnomalyClassificationMetrics(
            precision=2 / 3,
            recall=2 / 3,
            f1=2 / 3,
            specificity=4 / 5,
            false_positive_rate=1 / 5,
            false_negative_rate=1 / 3,
            balanced_accuracy=11 / 15,
        ),
    )


def report(*, split=DatasetSplit.TEST, slices=None):
    return AnomalyEvaluationReport(
        "dataset-v1",
        split,
        "robust_max_z_v1",
        "2026-11-08T09:00:00Z",
        HASH_A,
        HASH_B,
        (overall(),) if slices is None else slices,
    )


def test_report_accepts_test_split_and_run_scenario_slices():
    value = report(slices=(
        overall(),
        AnomalyEvaluationSlice(
            EvaluationScope.RUN, "run-1", counts(),
            AnomalyConfusionMatrix(2, 1, 4, 1),
            AnomalyClassificationMetrics(),
        ),
        AnomalyEvaluationSlice(
            EvaluationScope.SCENARIO, "packet-loss", counts(),
            AnomalyConfusionMatrix(2, 1, 4, 1),
            AnomalyClassificationMetrics(),
        ),
    ))
    assert value.evaluation_split is DatasetSplit.TEST
    assert len(value.slices) == 3


def test_validation_split_is_allowed():
    assert report(split=DatasetSplit.VALIDATION).evaluation_split is DatasetSplit.VALIDATION


def test_train_split_is_prohibited():
    with pytest.raises(AnomalyEvaluationError, match="VALIDATION or TEST"):
        report(split=DatasetSplit.TRAIN)


@pytest.mark.parametrize("field", [
    "total_samples", "labeled_samples", "unknown_samples",
    "scored_labeled_samples", "unscorable_labeled_samples",
    "scored_unknown_samples", "unscorable_unknown_samples",
])
def test_counts_reject_boolean_and_negative_values(field):
    values = dict(
        total_samples=12, labeled_samples=10, unknown_samples=2,
        scored_labeled_samples=8, unscorable_labeled_samples=2,
        scored_unknown_samples=1, unscorable_unknown_samples=1,
    )
    values[field] = True
    with pytest.raises(AnomalyEvaluationError):
        AnomalyEvaluationCounts(**values)
    values[field] = -1
    with pytest.raises(AnomalyEvaluationError):
        AnomalyEvaluationCounts(**values)


def test_counts_enforce_total_and_labeled_accounting():
    with pytest.raises(AnomalyEvaluationError, match="total_samples"):
        AnomalyEvaluationCounts(11, 10, 2, 8, 2, 1, 1)
    with pytest.raises(AnomalyEvaluationError, match="scored"):
        AnomalyEvaluationCounts(12, 10, 2, 7, 2, 1, 1)


def test_counts_preserve_unknown_and_unscorable_rows_separately():
    value = counts()
    assert value.unknown_samples == 2
    assert value.unscorable_labeled_samples == 2
    assert value.unscorable_unknown_samples == 1


def test_confusion_matrix_rejects_invalid_counts_and_exposes_total():
    with pytest.raises(AnomalyEvaluationError):
        AnomalyConfusionMatrix(true_positive=True)
    with pytest.raises(AnomalyEvaluationError):
        AnomalyConfusionMatrix(false_negative=-1)
    assert AnomalyConfusionMatrix(2, 1, 4, 1).total == 8


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_metrics_reject_invalid_ratios(value):
    with pytest.raises(AnomalyEvaluationError):
        AnomalyClassificationMetrics(precision=value)


def test_metrics_allow_none_for_undefined_denominators():
    value = AnomalyClassificationMetrics(precision=None, recall=None)
    assert value.precision is None
    assert value.recall is None


def test_slice_requires_confusion_total_to_match_scored_labeled():
    with pytest.raises(AnomalyEvaluationError, match="confusion matrix total"):
        AnomalyEvaluationSlice(
            EvaluationScope.OVERALL, None, counts(),
            AnomalyConfusionMatrix(1, 1, 4, 1),
            AnomalyClassificationMetrics(),
        )


def test_overall_scope_requires_no_scope_id():
    with pytest.raises(AnomalyEvaluationError, match="overall scope_id"):
        AnomalyEvaluationSlice(
            EvaluationScope.OVERALL, "all", counts(),
            AnomalyConfusionMatrix(2, 1, 4, 1),
            AnomalyClassificationMetrics(),
        )


@pytest.mark.parametrize("scope,scope_id", [
    (EvaluationScope.RUN, None),
    (EvaluationScope.SCENARIO, " "),
])
def test_grouped_scope_requires_non_empty_scope_id(scope, scope_id):
    with pytest.raises(AnomalyEvaluationError):
        AnomalyEvaluationSlice(
            scope, scope_id, counts(), AnomalyConfusionMatrix(2, 1, 4, 1),
            AnomalyClassificationMetrics(),
        )


def test_report_requires_exactly_one_overall_slice():
    with pytest.raises(AnomalyEvaluationError, match="exactly one overall"):
        report(slices=())
    with pytest.raises(AnomalyEvaluationError, match="exactly one overall"):
        report(slices=(overall(), overall()))


def test_report_rejects_duplicate_group_slice_identity():
    grouped = AnomalyEvaluationSlice(
        EvaluationScope.RUN, "run-1", counts(),
        AnomalyConfusionMatrix(2, 1, 4, 1),
        AnomalyClassificationMetrics(),
    )
    with pytest.raises(AnomalyEvaluationError, match="identities"):
        report(slices=(overall(), grouped, grouped))


@pytest.mark.parametrize("timestamp", [
    "2026-11-08T09:00:00",
    "2026-11-08T09:00:00+03:00",
    "not-a-date",
])
def test_report_requires_utc_timestamp(timestamp):
    with pytest.raises(AnomalyEvaluationError, match="generated_at"):
        AnomalyEvaluationReport(
            "dataset-v1", DatasetSplit.TEST, "method", timestamp,
            HASH_A, HASH_B, (overall(),),
        )


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64, "a" * 63])
def test_report_requires_sha256_provenance(digest):
    with pytest.raises(AnomalyEvaluationError, match="SHA-256"):
        AnomalyEvaluationReport(
            "dataset-v1", DatasetSplit.TEST, "method",
            "2026-11-08T09:00:00Z", digest, HASH_B, (overall(),),
        )


def test_report_is_immutable():
    value = report()
    with pytest.raises((AttributeError, TypeError)):
        value.method = "changed"
