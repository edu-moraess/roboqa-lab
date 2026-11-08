# Phase 4.5-D3 — Integrated QA evidence propagation

## Objective

Propagate the validated transport replay summary and identity-preserving Fusion execution record into quality metrics and the existing deterministic Quality Gate. The original Phase 4.2 `compute_quality_metrics` API and its semantics remain unchanged.

## Public API

- `compute_integrated_quality_metrics(result, replay_summary=..., execution_record=..., ...)`
- `evaluate_integrated_quality(result, replay_summary=..., execution_record=..., policy=..., generated_at=..., configuration_hash=..., ...)`
- `IntegratedQualityError`

Both functions are exported from `roboqa.qa`.

## Evidence binding and fail-closed validation

The integrated evaluator verifies:
1. Replay summary and execution record are the same immutable evidence object by value, including decisions, counts, ordering policy, and provenance hashes.
2. Run IDs match.
3. Native `FilterResult` status, prediction/update/rejection diagnostics match the execution record.
4. Native telemetry exactly matches the ordered UPDATED telemetry embedded in the execution outcomes.
5. The existing Phase 4.2 metric evaluator accepts the supplied estimator, ground-truth, envelope, and expected-count evidence.

A mismatch raises `IntegratedQualityError`; evidence is not silently combined across runs.

## Added metrics

Run-scoped and sensor-scoped metrics:

| Metric | Meaning |
|---|---|
| `transport_source_count` | Unique source packets in the replay source snapshot |
| `transport_delivered_count` | Delivery attempts, including duplicate copies |
| `transport_accepted_count` | Unique packets accepted by replay |
| `transport_duplicate_count` | Duplicate delivery attempts suppressed by replay |
| `transport_late_count` | Unique source packets classified late |
| `transport_invalid_count` | Unique source packets classified invalid |
| `transport_dropped_count` | Source packets with no delivery |
| `fusion_updated_count` | Accepted packets with successful EKF update telemetry |
| `fusion_failed_count` | Accepted packet at which Fusion stopped on a measurement failure |
| `fusion_not_executed_count` | Accepted packets not reached by Fusion execution |
| `transport_acceptance_ratio` | accepted / unique source packets |
| `fusion_update_ratio` | updated / unique source packets |
| `fusion_execution_success_ratio` | updated / replay-accepted packets |

Counts are explicit, finite, and may legitimately be zero. Ratios with a zero denominator are `MISSING` with no numeric value. Duplicate delivery attempts do not inflate the unique source denominator. Per-sensor counts use normalized sensor identity and the same source/delivery distinction.

The existing Phase 4.2 `sensor_availability` remains unchanged and retains its accepted-update/explicit-expected-count semantics. The new ratios are separate transport-to-Fusion measures, not physical sensor uptime.

## Quality Gate integration

`evaluate_integrated_quality` builds the combined metric set and passes it to the existing `evaluate_quality_gate`. No implicit policy or universal threshold is introduced. Callers must supply a versioned `QualityGatePolicy`.

Report provenance is bound as follows:
- `run_id`: replay run ID;
- `source_hash`: replay source snapshot hash;
- `output_hash`: identity-preserving Fusion execution hash;
- `configuration_hash`: explicit caller-supplied configuration hash;
- `generated_at`: explicit UTC timestamp supplied by caller.

The Gate remains the sole authority for PASS, PASS_WITH_WARNINGS, FAIL, and BLOCKED.

## Scope boundaries

- No change to Phase 4.2 metric calculations or Phase 4.3 policy semantics.
- No inferred thresholds, probabilities, or physical sensor uptime.
- No ML training/inference, model dependency, data download, Streamlit, CI/GitHub Actions, or merge.
- No changes to `main`.

## Manual validation

Status: **PENDING MANUAL COLAB VALIDATION**.

Required:
- targeted integrated QA tests;
- existing QA metrics and Quality Gate regression tests;
- integration/replay/scheduling/adapter/execution tests;
- full pytest;
- full Ruff;
- full compileall.

Do not mark this phase validated until the actual Colab outputs are reviewed.
