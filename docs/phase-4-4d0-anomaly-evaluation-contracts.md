# Phase 4.4-D0 — Anomaly Evaluation Contracts

**Status: VALIDATED — manual Google Colab.**

Validated production checkpoint: `60e866c7120578fa6a0aafe97126ce3971a9545e`.
- Targeted `tests/unit/test_ml_evaluation.py`: 33 passed.
- Full `ruff check roboqa tests`: PASS.
- Full `python -m compileall -q roboqa tests`: exit code 0.
- Full `python -m pytest -q`: exit code 0, reached 100%, no failures; aggregate count was not exposed and is not inferred.
- Evidence is manual Colab, not GitHub Actions. PR #22 remains Draft/open/unmerged.

The implementation scope below describes D0 contracts only; metric aggregation is implemented separately in D1.

## Purpose

Define immutable, auditable contracts for evaluating the Phase 4.4-C anomaly detectors. This checkpoint deliberately establishes the evaluation evidence model before implementing metric aggregation or changing detector thresholds.

## Evaluation boundaries

- Only VALIDATION and TEST are permitted as evaluation splits. TRAIN evaluation reports are rejected.
- Reference labels remain evaluation metadata. They are not passed to fit or score functions.
- NORMAL and ANOMALOUS labels contribute to binary classification counts only when a prediction is scorable.
- UNKNOWN labels are accounted for separately and never enter TP/FP/TN/FN.
- UNSCORABLE samples are not silently classified as normal or anomalous. They are counted separately from scorable samples.
- Undefined ratios are represented by None, never by zero.
- Overall, per-run, and per-scenario slices can be represented independently.
- Dataset and prediction evidence are bound to SHA-256 provenance hashes.
- These reports are descriptive evaluation artifacts, not Quality Gate decisions. The deterministic Quality Gate remains the sole quality authority.

## Contracts

AnomalyConfusionMatrix stores non-negative TP/FP/TN/FN counts.

AnomalyClassificationMetrics stores optional precision, recall, F1, specificity, false-positive rate, false-negative rate, and balanced accuracy. Values must be finite and in [0, 1]. None represents an undefined denominator.

AnomalyEvaluationCounts reconciles total, labeled, unknown, scored, and unscorable samples. Its accounting equations prevent missing evidence from disappearing.

AnomalyEvaluationSlice binds counts and metrics to one overall, run, or scenario scope. The confusion matrix total must equal the number of scored labeled samples.

AnomalyEvaluationReport binds the dataset, evaluation split, detector method, UTC generation time, dataset/prediction hashes, and unique slices. Exactly one overall slice is required.

## Explicitly not implemented in D0

- No metric computation or evaluation runner.
- No detector fitting, scoring, threshold selection, calibration, or hyperparameter search.
- No automatic label generation or inference of missing labels.
- No aggregation that mixes runs or scenarios without explicit scope.
- No Quality Gate integration or policy changes.
- No model training beyond the already validated Phase 4.4-C baseline implementations.
- No datasets, weights, or external artifacts are downloaded.

## Planned continuation

1. 4.4-D1 — Evaluation engine: implemented in `roboqa.ml.evaluation_engine`; manual Colab validation pending on its dedicated branch.
2. 4.4-D2 — Robustness evaluation: compare both detectors on fixed validation/test scenarios and report scorable coverage and false-positive/negative behavior.
3. 4.4-D3 — Calibration protocol: only after evaluation evidence is stable, define train/validation-only threshold calibration and freeze the final protocol before one-time TEST evaluation.

No threshold or hyperparameter is changed in D0.
