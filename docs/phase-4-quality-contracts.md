# RoboQA Lab — Phase 4.0: Quality Contracts

## Objective

Establish stable, immutable contracts for quality metrics, findings, and
provenance-linked reports before implementing metric calculation, policy
evaluation, machine learning, or user interfaces.

## Scope delivered

- `MetricObservation`: scalar value, unit, scope, sample count, and explicit
  availability (`available`, `missing`, `invalid`).
- `QualityFinding`: severity, rule identity, message, optional observed value,
  expected range, and evidence references.
- `QualityReport`: immutable aggregate with decision, metrics, findings,
  UTC generation timestamp, source/output/configuration SHA-256 hashes, and
  policy version.
- Contract validation rejects non-finite numeric values, invalid scopes,
  duplicate metric identities, malformed hashes, and non-UTC timestamps.

## Metric scopes

- `run`: one value for the complete execution.
- `sensor`: one value for a named sensor; `sensor_id` is mandatory.
- `window`: one value for a half-open interval `[window_start, window_end)`;
  both finite bounds are mandatory and start must be less than end.

Missing or invalid metrics carry `value=None`; zero remains a valid numeric
measurement. The contract never treats NaN or infinity as missing values.

## Planned metric catalog (calculation is not part of Phase 4.0)

| Metric | Scope | Unit / interpretation |
|---|---|---|
| position_rmse | run / window | m; requires aligned ground truth |
| velocity_rmse | run / window | m/s; requires aligned ground truth |
| innovation_norm | sensor / window | sensor measurement units |
| nis | sensor / window | dimensionless; (y^T S^{-1} y) |
| nis_mean | sensor / run | dimensionless mean over valid updates |
| sensor_availability | sensor / window | ratio in [0, 1] |
| transport_latency | sensor / window | seconds; arrival minus acquisition |
| processing_wait | sensor / window | seconds; processing minus arrival |
| end_to_end_latency | sensor / window | seconds; processing minus acquisition |
| update_rejection_rate | sensor / run | ratio in [0, 1] |
| covariance_trace | run / window | state covariance trace |

Metric names are identifiers, not proof that a metric has been calculated.
Each future evaluator must document alignment, denominator, invalid-sample
handling, and aggregation semantics.

## NIS contract

For innovation vector (y) and innovation covariance (S):

[
NIS = y^T S^{-1} y
]

The measurement degrees of freedom equal the measurement dimension. Any future
statistical consistency test must compare NIS against a documented chi-square
interval for its degrees of freedom and confidence level; a universal fixed
threshold is not valid across sensors with different dimensions.

## Quality decision semantics

- `PASS`: all required evidence exists and all blocking criteria pass.
- `PASS_WITH_WARNINGS`: required criteria pass, with one or more non-blocking
  warnings.
- `FAIL`: sufficient evidence exists and at least one failure criterion is
  violated.
- `BLOCKED`: required evidence is missing, invalid, or insufficient to make
  a defensible decision.

The Phase 4.0 report contract stores a decision but deliberately does not
calculate it. Deterministic policy evaluation belongs to a later phase.

## ML boundary

Machine-learning outputs are advisory evidence (for example anomaly scores or
labels). They do not independently approve a run, override hard constraints,
or turn missing evidence into a pass. Training is explicitly out of scope until
the metric pipeline, data provenance, and evaluation protocol are validated.

## Non-goals

No EKF changes, NIS computation, threshold evaluation, Quality Gate engine,
model training, model download, Streamlit UI, CI workflow, or merge to `main`.
