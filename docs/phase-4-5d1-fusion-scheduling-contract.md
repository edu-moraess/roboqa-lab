# Phase 4.5-D1 — Deterministic Fusion Scheduling Contract

**Implementation status:** implemented; **manual Colab validation PASS**.

## Audit findings

The current transport replay is an offline batch operation. Its `ARRIVAL_ORDER` policy sorts by arrival time and applies an acquisition-time watermark. The strict Measurement adapter preserves that accepted order, but the existing `run_filter` independently sorts normalized measurements by `(Measurement.t, sensor)`.

The EKF is monotonic in time and has no rewind/reprocess mechanism. Consequently, directly forwarding arrival-ordered measurements is not a valid way to obtain arrival-order Fusion execution. With nonzero allowed lateness, replay may accept event-time disorder; the Fusion boundary must reorder those accepted measurements into event-time order before calling the existing runner.

## Scope of D1

This phase adds an immutable, deterministic scheduling boundary:

- `build_fusion_schedule(replay, adapted_measurements)` accepts the Phase 4.5-B replay result and Phase 4.5-C adapter output.
- It verifies a one-to-one identity match between replay-accepted envelopes and adapted records.
- It verifies sensor identity, acquisition time, arrival time, processing time, and measurement payload consistency.
- It orders entries by `(acquisition_time, normalized_sensor_id, source_id, sequence_number)`.
- It retains the complete `TransportReplaySummary`, including duplicate, late, invalid, and dropped decisions.
- It computes a schedule SHA-256 over replay provenance, event identity, all three clocks, normalized payload, and the explicit covariance matrix R.
- Empty accepted batches produce an empty schedule; they do not fabricate a measurement.

## Normative execution semantics

1. **Acquisition/event time controls EKF execution.** This is the only supported scheduling policy in D1.
2. Arrival and processing clocks remain provenance/latency evidence. They never replace `Measurement.t`.
3. Replay dispositions other than ACCEPTED never enter the schedule. They remain visible in the attached replay summary.
4. Accepted packets are sorted before Fusion execution, including packets accepted within the replay watermark's allowed-lateness window.
5. Ties are deterministic: normalized sensor ID, source ID (empty string for `None`), then sequence number.
6. The existing `run_filter` and EKF are not modified. The schedule's primary sort keys match the runner's stable `(t, sensor)` sort, so deterministic tie order is retained.
7. D1 is offline batch scheduling. It does not implement wall-clock buffering, per-source idle detection, streaming watermark finalization, event-time correction, or arrival-order EKF execution.

## Why arrival-order execution is not exposed

The EKF rejects negative prediction intervals and cannot rewind its state/covariance when an earlier acquisition-time event arrives after a later one. A true online arrival-order mode would require a separately specified fixed-lag buffer plus state checkpoint/replay semantics, bounded memory, watermark finalization, and a policy for packets arriving after finalization. D1 deliberately does not approximate that behavior.

## Validation — PASS

Manual Google Colab evidence:
- Targeted `tests/unit/test_fusion_schedule.py`: **15 passed**.
- Full `python -m pytest -q`: reached 100%, exit code 0, no failures reported. The captured output did not expose an aggregate test count; none is inferred.
- Full `ruff check roboqa tests`: **All checks passed!**
- Full `python -m compileall -q roboqa tests`: exit code 0.
- Result: **FULL VALIDATION PASS** for D1 and repository compatibility.

The full suite was run on branch `feat/fusion-scheduling-phase-4-5d1`, validated code checkpoint `43d0bd73f43e47b6dc0d3e0cc0f545e0127ba929`.

## Explicit non-goals

- No changes to `Measurement`, EKF, or `run_filter`.
- No live/async scheduler, fixed-lag reprocessing, or arrival-order execution.
- No change to Quality Metrics or Quality Gate in D1.
- No ML training/inference, dataset download, Streamlit, CI/GitHub Actions, or merge.

## Next subphase

Phase 4.5-D2 will execute the immutable schedule through the existing Fusion runner and produce an execution record that links each accepted source identity to successful telemetry or an explicit execution outcome. Phase 4.5-D3 will propagate replay and execution evidence into Quality Metrics and the deterministic Quality Gate, keeping the Gate as the sole decision authority.
