"""Deterministic anomaly evaluation engine for Phase 4.4-D1.

Evaluation joins predictions to dataset rows by exact sample identity. Labels are
read only here, after detector scoring; they are never used for fitting/scoring.
This module produces descriptive evidence and never makes Quality Gate decisions.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from roboqa.ml.anomaly import AnomalyDecision, AnomalyPrediction
from roboqa.ml.contracts import AnomalyReference, DatasetSplit, FeatureDataset
from roboqa.ml.evaluation import (
    AnomalyClassificationMetrics,
    AnomalyConfusionMatrix,
    AnomalyEvaluationCounts,
    AnomalyEvaluationError,
    AnomalyEvaluationReport,
    AnomalyEvaluationSlice,
    EvaluationScope,
)


def _digest(document: object) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metrics(matrix: AnomalyConfusionMatrix) -> AnomalyClassificationMetrics:
    tp, fp, tn, fn = (
        matrix.true_positive, matrix.false_positive,
        matrix.true_negative, matrix.false_negative,
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None else None
    )
    return AnomalyClassificationMetrics(
        precision=precision, recall=recall, f1=f1,
        specificity=specificity, false_positive_rate=fpr,
        false_negative_rate=fnr, balanced_accuracy=balanced,
    )


def _build_slice(rows: Sequence[tuple[object, AnomalyPrediction]], scope: EvaluationScope,
                 scope_id: str | None) -> AnomalyEvaluationSlice:
    labeled = unknown = scored_labeled = unscorable_labeled = 0
    scored_unknown = unscorable_unknown = 0
    tp = fp = tn = fn = 0
    for sample, prediction in rows:
        is_unknown = sample.reference_label is AnomalyReference.UNKNOWN
        if is_unknown:
            unknown += 1
            if prediction.decision is AnomalyDecision.UNSCORABLE:
                unscorable_unknown += 1
            else:
                scored_unknown += 1
            continue
        labeled += 1
        if prediction.decision is AnomalyDecision.UNSCORABLE:
            unscorable_labeled += 1
            continue
        scored_labeled += 1
        expected_anomaly = sample.reference_label is AnomalyReference.ANOMALOUS
        predicted_anomaly = prediction.decision is AnomalyDecision.ANOMALOUS
        if expected_anomaly and predicted_anomaly:
            tp += 1
        elif not expected_anomaly and predicted_anomaly:
            fp += 1
        elif not expected_anomaly and not predicted_anomaly:
            tn += 1
        else:
            fn += 1

    matrix = AnomalyConfusionMatrix(tp, fp, tn, fn)
    counts = AnomalyEvaluationCounts(
        total_samples=len(rows), labeled_samples=labeled, unknown_samples=unknown,
        scored_labeled_samples=scored_labeled,
        unscorable_labeled_samples=unscorable_labeled,
        scored_unknown_samples=scored_unknown,
        unscorable_unknown_samples=unscorable_unknown,
    )
    return AnomalyEvaluationSlice(scope, scope_id, counts, matrix, _metrics(matrix))


def evaluate_anomaly_predictions(
    dataset: FeatureDataset,
    predictions: Sequence[AnomalyPrediction],
    *,
    evaluation_split: DatasetSplit,
    generated_at: str,
) -> AnomalyEvaluationReport:
    """Evaluate predictions against exact dataset sample identities.

    predictions must contain exactly one prediction for every dataset sample,
    as returned by the Phase 4.4-C score APIs. Rows outside evaluation_split
    are identity-validated but never included in evaluation counts or hashes.
    """
    if not isinstance(dataset, FeatureDataset):
        raise AnomalyEvaluationError("dataset must be a FeatureDataset")
    if not isinstance(evaluation_split, DatasetSplit) or evaluation_split not in (\n        DatasetSplit.VALIDATION, DatasetSplit.TEST\n    ):
        raise AnomalyEvaluationError(
            "evaluation_split must be VALIDATION or TEST; TRAIN is prohibited"
        )
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise AnomalyEvaluationError("predictions must be a sequence")
    prediction_rows = tuple(predictions)
    if any(not isinstance(item, AnomalyPrediction) for item in prediction_rows):
        raise AnomalyEvaluationError("predictions must contain AnomalyPrediction values")

    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    prediction_by_id: dict[str, AnomalyPrediction] = {}
    for prediction in prediction_rows:
        if prediction.sample_id in prediction_by_id:
            raise AnomalyEvaluationError(
                f"duplicate prediction sample_id: {prediction.sample_id}"
            )
        sample = sample_by_id.get(prediction.sample_id)
        if sample is None:
            raise AnomalyEvaluationError(
                f"prediction sample_id is not present in dataset: {prediction.sample_id}"
            )
        if prediction.run_id != sample.run_id or prediction.split is not sample.split:
            raise AnomalyEvaluationError(
                f"prediction identity mismatch for sample_id: {prediction.sample_id}"
            )
        prediction_by_id[prediction.sample_id] = prediction

    missing = sorted(set(sample_by_id) - set(prediction_by_id))
    if missing:
        raise AnomalyEvaluationError(
            "missing predictions for dataset sample_id values: " + ", ".join(missing[:5])
        )
    methods = {item.method for item in prediction_rows}
    if len(methods) != 1:
        raise AnomalyEvaluationError("all predictions must use exactly one detector method")
    method = next(iter(methods))
    selected = tuple(
        sample for sample in dataset.samples if sample.split is evaluation_split
    )
    if not selected:
        raise AnomalyEvaluationError(
            f"dataset contains no samples in evaluation split {evaluation_split.value}"
        )
    selected_pairs = tuple(
        (sample, prediction_by_id[sample.sample_id]) for sample in selected
    )

    slices = [_build_slice(selected_pairs, EvaluationScope.OVERALL, None)]
    by_run: dict[str, list[tuple[object, AnomalyPrediction]]] = {}
    by_scenario: dict[str, list[tuple[object, AnomalyPrediction]]] = {}
    for sample, prediction in selected_pairs:
        by_run.setdefault(sample.run_id, []).append((sample, prediction))
        if sample.scenario_id is not None:
            by_scenario.setdefault(sample.scenario_id, []).append((sample, prediction))
    for run_id in sorted(by_run):
        slices.append(_build_slice(by_run[run_id], EvaluationScope.RUN, run_id))
    for scenario_id in sorted(by_scenario):
        slices.append(
            _build_slice(by_scenario[scenario_id], EvaluationScope.SCENARIO, scenario_id)
        )

    dataset_document = [
        {
            "sample_id": sample.sample_id,
            "run_id": sample.run_id,
            "scenario_id": sample.scenario_id,
            "reference_label": sample.reference_label.value,
            "features": [
                [item.name, item.value, item.unit, item.availability.value, item.sample_count]
                for item in sample.features
            ],
        }
        for sample in selected
    ]
    prediction_document = [
        {
            "sample_id": prediction.sample_id,
            "run_id": prediction.run_id,
            "split": prediction.split.value,
            "method": prediction.method,
            "score": prediction.score,
            "threshold": prediction.threshold,
            "decision": prediction.decision.value,
            "feature_names": prediction.feature_names,
            "reason": prediction.reason,
        }
        for _, prediction in selected_pairs
    ]
    return AnomalyEvaluationReport(
        dataset_id=dataset.dataset_id,
        evaluation_split=evaluation_split,
        method=method,
        generated_at=generated_at,
        dataset_hash=_digest({
            "dataset_id": dataset.dataset_id,
            "schema_version": dataset.schema_version,
            "source_hash": dataset.source_hash,
            "configuration_hash": dataset.configuration_hash,
            "samples": dataset_document,
        }),
        prediction_hash=_digest(prediction_document),
        slices=tuple(slices),
    )
