# Phase 4.4-B — Temporal Feature Engineering

## Objective

Extract deterministic, auditable time-window features from immutable Fusion
`InnovationTelemetry` for downstream advisory ML evaluation. Phase 4.4-A
contracts remain the output schema. This phase does not train or run models.

## Public API

`roboqa.ml.extract_temporal_features(...)`

Inputs:
- a non-empty sequence of `InnovationTelemetry`;
- explicit `run_id`, `DatasetSplit`, dataset identity/schema version;
- UTC `created_at`;
- finite positive `window_duration_s`;
- optional scenario and evaluation-only reference label.

## Feature schema

Each observed sensor receives one row per fixed-duration window. Windows are
anchored at the earliest telemetry timestamp in the run and use half-open
intervals `[start, end)`.

| Feature | Unit | Definition |
|---|---|---|
| `update_count` | count | Successful telemetry updates in the sensor/window |
| `innovation_norm_mean` | sensor_units | Mean Euclidean innovation norm |
| `innovation_norm_std` | sensor_units | Population standard deviation |
| `innovation_norm_max` | sensor_units | Maximum innovation norm |
| `nis_mean` | dimensionless | Mean normalized innovation squared |
| `nis_std` | dimensionless | Population standard deviation |
| `nis_max` | dimensionless | Maximum NIS |
| `innovation_covariance_trace_mean` | measurement_covariance_units | Mean trace of innovation covariance S |
| `inter_update_gap_mean` | s | Mean gap between consecutive updates in the window |
| `inter_update_gap_max` | s | Maximum gap between consecutive updates in the window |

Sensor grouping prevents combining innovation magnitudes or covariance traces
from sensors with different measurement dimensions/units. A window with no
updates retains an explicit available zero `update_count`; other features are
MISSING with no numeric value. Inter-update gaps are MISSING when fewer than
two updates are available.

## Provenance and determinism

- Source SHA-256 binds the canonically ordered telemetry records, including
  timestamp, sensor, degrees of freedom, innovation, covariance, and NIS.
- Configuration SHA-256 binds feature schema version, window duration/anchor,
  grouping, population-standard-deviation convention, split, scenario and
  reference label.
- Input telemetry is not mutated.
- Output is a validated `FeatureDataset`; all windows from a run share one
  explicit split, preserving the 4.4-A run-level leakage constraint.

## Explicit boundaries

- No imputation, normalization, clipping, interpolation, feature selection,
  training, inference, model downloads, or automatic split assignment.
- `AnomalyReference` remains metadata and is never emitted as a feature.
- This first extractor uses successful Fusion telemetry. Transport-only
  dispositions and missing/failed updates remain in D3 QA evidence; they are
  not silently represented as telemetry. A later fusion step may join those
  validated evidence streams explicitly.
- No Quality Gate policy or decision semantics are changed.

## Manual validation

Run in Colab:

```bash
python -m pytest -q tests/unit/test_ml_features.py
ruff check roboqa tests
python -m compileall -q roboqa tests
python -m pytest -q
```

Status: **PENDING MANUAL COLAB VALIDATION**.
