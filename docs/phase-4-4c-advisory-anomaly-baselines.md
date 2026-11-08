# Phase 4.4-C — Advisory Anomaly Baselines

## Objective

Provide reproducible, leakage-aware anomaly scores from the validated temporal feature dataset. ML output is advisory only; it cannot change, override, or supply a decision to the deterministic Quality Gate.

## Implemented APIs

- `fit_statistical_baseline(dataset, contamination=0.05)`
- `score_statistical_baseline(model, dataset)`
- `fit_isolation_forest(dataset, contamination=0.05, random_state=42, n_estimators=100)`
- `score_isolation_forest(model, dataset)`

## Training and leakage controls

- Both models select **only** rows whose `split` is `TRAIN`.
- At least three TRAIN samples are required.
- Feature eligibility, medians, robust scales, and model thresholds are derived only from TRAIN values.
- A feature is eligible only when it is AVAILABLE in every TRAIN row. Constant features are excluded. No imputation or silent row deletion occurs.
- Validation/test values are used only for scoring. Missing required values produce an explicit `UNSCORABLE` prediction with no score.
- `reference_label` is not read by either fitting or scoring path and is never included in the model feature matrix.
- The training data hash binds the training sample/run identities and selected training feature values. Model hash binds that hash, transform, and parameters.

## Statistical baseline

The deterministic baseline standardizes each eligible feature using median and a robust scale (1.4826 × MAD; IQR and standard deviation are fallback scales when MAD is zero). Constant dimensions are removed. Per-sample anomaly score is the maximum absolute robust z-score across eligible features. The threshold is the `higher` empirical quantile of TRAIN scores at `1 - contamination`. A score strictly greater than the threshold is marked `ANOMALOUS`.

This is a transparent baseline, not a calibrated probability or universal safety threshold.

## Isolation Forest

Isolation Forest is optional and loaded lazily from scikit-learn. Install it with `pip install -e ".[ml]"`. It uses the same TRAIN-only feature selection and robust scaling, explicit `random_state`, `n_estimators`, and `contamination`. It is an independent advisory path; it does not replace the statistical baseline and does not control Quality Gate outcomes.

## Output semantics

Predictions retain sample/run identity, split, method, score, threshold, eligible feature names, and one of:

- `NORMAL`
- `ANOMALOUS`
- `UNSCORABLE`

The terms NORMAL/ANOMALOUS describe detector output only. They are not QualityDecision values. No automatic finding or gate policy is created here.

## Boundaries

- No model training uses VALIDATION or TEST rows.
- No labels are used for fitting, feature selection, threshold selection, or inference.
- No imputation, normalization fitted outside TRAIN, hyperparameter search, calibration claims, model persistence, external data download, or Gate integration.
- No training, inference, or model package is mandatory for core RoboQA imports; scikit-learn remains an optional dependency.
- No Quality Gate semantics are changed.

## Manual Colab validation

```bash
python -m pytest -q tests/unit/test_ml_anomaly.py
ruff check roboqa tests
python -m compileall -q roboqa tests
python -m pytest -q
```

Status: **PENDING MANUAL COLAB VALIDATION**.