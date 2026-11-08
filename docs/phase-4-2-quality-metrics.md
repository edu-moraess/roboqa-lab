# RoboQA Lab — Phase 4.2: Quality Metrics Engine

## Audit findings carried into implementation

The existing pipeline supplies per-update innovation/NIS telemetry, state/covariance snapshots, runner diagnostics, raw three-clock envelopes, and optional ground truth as a separate object. It does not currently carry an expected packet schedule, a complete per-sensor rejection ledger, or a ground-truth alignment policy in FilterResult.

Consequently:
- Position RMSE requires explicitly supplied GroundTruth and a caller-selected alignment tolerance.
- Sensor availability requires explicit expected-measurement counts per sensor; it is never inferred from observed packets.
- Transport, processing-wait, and end-to-end latency are calculated from raw envelopes. Missing processing timestamps remain missing; negative/non-finite deltas are invalid.
- The current runner n_rejected counts temporal rejections, not every possible numerical/input rejection. The metric is therefore named temporal_rejection_rate, not generic update rejection rate.
- Covariance trace is aggregated over stored FilterResult covariance snapshots, including the initial prior and one posterior per accepted update.
- Repeated filter timestamps are collapsed to the final posterior before RMSE so simultaneous sensor updates do not overweight that instant.

## Implemented metrics

| Metric | Scope | Availability rule |
|---|---|---|
| position_rmse | run | Requires GroundTruth and aligned timestamps |
| covariance_trace_mean | run | Requires finite covariance snapshots |
| innovation_norm_mean | sensor | Mean Euclidean innovation norm over accepted updates |
| nis_mean | sensor | Mean NIS over accepted updates; no universal pass threshold |
| sensor_availability | sensor | Accepted update count / explicit expected count |
| transport_latency_mean | sensor | Mean non-negative arrival minus acquisition |
| processing_wait_mean | sensor | Mean non-negative processing minus arrival |
| end_to_end_latency_mean | sensor | Mean non-negative processing minus acquisition |
| temporal_rejection_rate | run | Runner temporal rejections / (accepted updates + temporal rejections) |

Missing evidence yields MetricAvailability.MISSING with value=None. Malformed, contradictory, or incomplete latency evidence yields INVALID with value=None. Zero remains a valid measured value.

## RMSE alignment

Ground-truth positions are linearly interpolated at unique filter timestamps. Evaluation is restricted to the overlapping time range and timestamps whose nearest ground-truth sample is within alignment_tolerance_s. No extrapolation is performed. The final posterior is used when multiple updates share a timestamp.

## Availability denominator

The caller supplies expected_measurements={sensor_id: expected_count}. The numerator is the number of accepted telemetry updates for that sensor. A zero expected count is undefined and remains missing. More accepted updates than expected is invalid rather than silently capped. This first implementation is update availability, not proof of physical sensor uptime.

## Explicit non-goals

No chi-square decision thresholds, findings policy, Quality Gate, automatic inference of expected packet rates, complete rejection ledger, ML training, fault-to-QA orchestration, Streamlit UI, physical hardware, dataset downloads, GitHub Actions, or merge to main.

## Manual Colab validation

\`\`\`bash
python -m compileall -q roboqa tests
ruff check roboqa tests
pytest -q tests/unit/test_quality_metrics.py
pytest -q tests/unit/test_quality_contracts.py
pytest -q
coverage run --branch --source=roboqa/qa -m pytest -q
coverage report -m
\`\`\`

Keep the phase unvalidated until actual Colab output is reviewed.
