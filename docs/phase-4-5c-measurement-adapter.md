# Phase 4.5-C — Strict Measurement Adapter

**Status:** implemented and **manual Colab validation PASS** (2026-11-08).

## Objective

Create a controlled boundary from replay-accepted raw transport envelopes into the existing strict Fusion Measurement contract. This phase converts data; it does not schedule or execute the filter.

## API

`adapt_accepted_envelopes(replay, sensor_covariances=...)`

Returns an immutable tuple of `AdaptedMeasurement` records. Each record retains:
- `EnvelopeIdentity(sensor_id, source_id, sequence_number)`;
- normalized `Measurement`;
- original `arrival_time`;
- optional `processing_time`.

## Conversion rules

1. Input must be a `TransportReplayResult`.
2. Only `replay.accepted_envelopes` are adapted. Duplicate, late, invalid, and dropped records remain represented in the replay summary and are never converted.
3. Payload is interpreted as measurement vector `z`.
4. `Measurement.t` is acquisition time, not arrival or processing time.
5. Sensor identifier is normalized to lowercase and checked against the existing Fusion sensor registry and dimensions.
6. Covariance `R` must be supplied explicitly in `sensor_covariances`, keyed by normalized sensor ID. No covariance is inferred, estimated, defaulted, or learned by this adapter.
7. The existing `Measurement` constructor remains the authority for finite values, shape, symmetry, positive-semidefiniteness, and defensive read-only copies.
8. Unknown sensors, absent covariance, malformed payloads, invalid covariance, and incompatible dimensions fail closed with `MeasurementAdapterError`.

## Temporal and execution boundary

The adapter preserves all three clocks, but only acquisition time maps to `Measurement.t`. The returned tuple follows the replay's accepted-envelope order.

**Important:** the existing `run_filter` sorts Measurements by `(t, sensor)` before EKF execution. Therefore, this adapter does not make Fusion execute in arrival order. An arrival-order replay may still produce an accepted set whose acquisition timestamps are out of order; Phase 4.5-D must specify the scheduling/replay policy for Fusion explicitly rather than claiming the adapter changes it.

## Validation scope and non-goals

- No change to `Measurement`, EKF, `run_filter`, sensor models, Quality Metrics, or Quality Gate.
- No automatic covariance selection or sensor calibration.
- No timestamp substitution or repair.
- No live queue, async processing, buffering, or late-event correction.
- No model training/inference, Streamlit, CI/GitHub Actions, or merge.

Manual Colab validation is required: targeted adapter tests, full pytest, Ruff, and compileall. Keep the PR Draft until the actual outputs are reviewed.


## Manual validation evidence — 2026-11-08

Validated on branch `feat/measurement-adapter-phase-4-5c`, code checkpoint `77f0e6e267f08f86ddc7f1468da0b9d474d348ef`, with two validation-only corrections:
- Adapter unit tests: **10 passed**.
- Full `python -m pytest -q`: exit code **0**, no failures. The captured output did not expose an aggregate test count; none is inferred.
- `ruff check roboqa tests`: **All checks passed**.
- `python -m compileall -q roboqa tests`: exit code **0**.

The only changes needed were the case-insensitive dimension-error assertion and import ordering in `roboqa/integration/__init__.py`; production adapter behavior was unchanged. The validation covers adapter contracts and repository compatibility, not arrival-order Fusion execution or end-to-end Fusion/QA integration.
