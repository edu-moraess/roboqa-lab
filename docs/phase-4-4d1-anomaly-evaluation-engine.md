# Phase 4.4-D1 — Anomaly Evaluation Engine

## Objective

Turn the immutable Phase 4.4-D0 evaluation contracts into a deterministic,
identity-safe evaluation engine for Phase 4.4-C advisory detector predictions.

## Public API

- \`evaluate_anomaly_predictions(dataset, predictions, evaluation_split, generated_at)\`
- Exported from \`roboqa.ml\`.

The input predictions are the complete output tuple from a Phase 4.4-C score API:
exactly one prediction for every dataset sample. This allows the engine to verify
the complete sample identity map before selecting VALIDATION or TEST rows.

## Evaluation semantics

- Only VALIDATION and TEST are accepted as evaluation splits. TRAIN is rejected.
- Each prediction must match exactly one dataset sample by \`sample_id\`, \`run_id\`,
  and split. Duplicate, missing, unknown, or identity-mismatched predictions fail closed.
- A report contains one OVERALL slice, one slice per represented run, and one
  slice per non-null scenario ID. Scenario-less rows remain in OVERALL and RUN.
- NORMAL and ANOMALOUS references contribute to TP/FP/TN/FN only when the
  detector prediction is scorable.
- UNKNOWN references are counted separately and never enter the confusion matrix.
- UNSCORABLE predictions are counted separately for labeled and UNKNOWN rows.
- Undefined metric denominators remain None; they are never coerced to zero.
- All predictions in one report must come from exactly one detector method.
- Dataset and prediction SHA-256 hashes are deterministic and bind only the
  selected evaluation split. TRAIN and the other split are identity-checked,
  but do not contribute to evaluation metrics or evaluation hashes.

## Metrics

Positive class: ANOMALOUS.

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = harmonic mean of precision and recall
- Specificity = TN / (TN + FP)
- False-positive rate = FP / (FP + TN)
- False-negative rate = FN / (FN + TP)
- Balanced accuracy = (recall + specificity) / 2

If a denominator is zero, the corresponding metric is None. F1 is None if
precision or recall is undefined; if both are defined and zero, F1 is zero.

## Leakage and authority boundaries

- Reference labels are read only by this evaluation module, after predictions exist.
- Detector fitting, transforms, thresholds, and scoring remain Phase 4.4-C responsibilities.
- No threshold calibration, hyperparameter search, training, inference, or downloads.
- Evaluation reports are descriptive evidence, not a policy decision.
- The deterministic Quality Gate remains the sole quality decision authority.
- TEST evaluation is explicit and separate; callers must not use TEST results to
  tune a detector or threshold.

## Validation protocol

Manual Google Colab validation is required before this phase can be marked VALIDATED:

\`\`\`bash
python -m pytest -q tests/unit/test_ml_evaluation_engine.py tests/unit/test_ml_evaluation.py
ruff check roboqa tests
python -m compileall -q roboqa tests
python -m pytest -q
\`\`\`

Review the actual command outputs and exact remote HEAD. Do not infer aggregate
test counts when pytest output does not expose them. No GitHub Actions workflow
is required or authorized for this project.

**Status: IMPLEMENTED — MANUAL COLAB VALIDATION PENDING.**
