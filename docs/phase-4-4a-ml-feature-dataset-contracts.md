# RoboQA Lab — Phase 4.4-A: ML Feature and Dataset Contracts

## Purpose

Phase 4.4-A establishes immutable, auditable contracts for feature rows and
experimental datasets. It does not implement feature extraction, preprocessing,
model fitting, inference, or quality decisions.

## Contracts

- `FeatureValue`: named scalar, explicit unit, sample count, and
  AVAILABLE/MISSING/INVALID state. Missing and invalid values carry `None`;
  numeric sentinels and implicit imputation are forbidden.
- `FeatureSample`: one run-level or time-window row, explicit TRAIN,
  VALIDATION, or TEST assignment, feature values, optional scenario identity,
  and optional reference annotation.
- `FeatureDataset`: versioned manifest with UTC creation time and SHA-256
  source/configuration provenance.

## Leakage and schema rules

1. A `run_id` may appear in only one split, even when it contributes multiple
   time windows. This prevents windows from one execution leaking across
   train/validation/test.
2. Every sample must contain the same feature names. A feature that is not
   available must be represented explicitly as MISSING or INVALID.
3. A feature name has one consistent unit throughout a dataset.
4. Sample IDs are unique. Window bounds, when present, are finite and ordered.
5. Dataset timestamps must be timezone-aware UTC. Provenance hashes must be
   64-character SHA-256 hex digests.

## Reference labels

`AnomalyReference` is evaluation metadata only. NORMAL, ANOMALOUS, and UNKNOWN
labels are not part of the feature vector and must never be passed to an
unsupervised model as input. The contract does not claim that labels are
complete or independently verified.

## Explicit non-goals

- No model training or automatic model loading.
- No automatic train/test split or random row-level splitting.
- No imputation, normalization, clipping, feature selection, or silent row
  dropping.
- No Isolation Forest or other detector implementation in this subphase.
- No ML score can change a deterministic Quality Gate decision.
- No dataset downloads or generated experimental results.

## API sketch

```python
from roboqa.ml import (
    DatasetSplit, FeatureAvailability, FeatureDataset,
    FeatureSample, FeatureValue,
)

sample = FeatureSample(
    sample_id="run-001-window-000",
    run_id="run-001",
    split=DatasetSplit.TRAIN,
    features=(
        FeatureValue("nis_mean", 2.1, "dimensionless", sample_count=40),
        FeatureValue(
            "end_to_end_latency_mean", None, "s",
            availability=FeatureAvailability.MISSING, sample_count=0,
        ),
    ),
)

dataset = FeatureDataset(
    dataset_id="roboqa-experiment-v1",
    schema_version="1.0",
    created_at="2026-11-08T05:00:00Z",
    samples=(sample,),
    source_hash="<64-character SHA-256>",
    configuration_hash="<64-character SHA-256>",
)
```

## Validation

Run manually in Colab:

```bash
python -m pytest -q tests/unit/test_ml_contracts.py
python -m pytest -q
ruff check roboqa tests --output-format=full
python -m compileall -q roboqa tests
```

Do not mark Phase 4.4-A validated until the actual outputs are reviewed.


## Validation evidence — 2026-11-08

Phase 4.4-A was manually validated in Google Colab on branch `feat/ml-intelligence-contracts-phase-4-4a`, HEAD `fa03643f18683b6317c4806e4d04709733ebb3d7`:

- `python -m pytest -q tests/unit/test_ml_contracts.py`: **30 passed**.
- Targeted Ruff: `All checks passed!`.
- Targeted compileall: completed without errors.
- Full `python -m pytest -q`: reached 100%, with no reported failures.
- Full `ruff check roboqa tests --output-format=full`: `All checks passed!`.
- Full `python -m compileall -q roboqa tests`: completed without errors.

The full-suite output did not expose an unambiguous aggregate test count. This is a contract and compatibility validation, not evidence of trained-model performance. No training, inference, or dataset download was performed.

**Status: VALIDATION PASS.** PR #13 remains Draft and unmerged.
