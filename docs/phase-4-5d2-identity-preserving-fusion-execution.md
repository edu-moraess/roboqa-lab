# Phase 4.5-D2 — Identity-preserving Fusion execution

## Objective

Execute a validated `FusionSchedule` with the existing monotonic EKF runner and preserve a deterministic association between each replay-accepted source identity and its Fusion execution outcome.

## Public API

- `execute_fusion_schedule(schedule, ...)`
- `FusionExecutionResult`: immutable wrapper containing the execution record and native `FilterResult`.
- `FusionExecutionRecord`: immutable, hash-addressed execution evidence.
- `MeasurementExecutionOutcome`: one terminal outcome for each schedule identity.
- `ExecutionDisposition`: `UPDATED`, `FAILED`, or `NOT_EXECUTED`.

## Normative behavior

1. Only entries already accepted by transport replay and validated by D1 can be executed.
2. The D1 acquisition-time order is passed unchanged to the existing `run_filter`.
3. Each successful runner telemetry item is associated with the corresponding schedule identity. The runner's stable `(measurement.t, measurement.sensor)` sort preserves D1's deterministic tie ordering.
4. The current runner stops on the first EKF numerical/schema update failure. The failed item is recorded as `FAILED`; remaining schedule entries are `NOT_EXECUTED`.
5. Invalid run configuration that prevents processing does not blame an individual measurement; all schedule entries remain `NOT_EXECUTED`.
6. A runner rejection count is treated as an integration error. D1 guarantees monotonic acquisition-time scheduling; this implementation will not guess which identity was rejected.
7. The execution hash covers run/schedule provenance, every identity and disposition, telemetry, diagnostics, configuration, and returned state/covariance/time traces.
8. The original `FilterResult` is retained for downstream metrics and analysis.

## Scope boundaries

- No changes to EKF mathematics or the existing runner's filtering policy.
- No online arrival-order execution, rewind, fixed-lag smoothing, or watermark scheduler.
- No changes to Quality Metrics or deterministic Quality Gate.
- No ML training/inference, dataset download, Streamlit UI, CI, or automatic training.
- Transport decisions for duplicate, late, invalid, and dropped packets remain in the replay summary; D2 outcomes cover replay-accepted schedule entries only.

## Validation

Status: **FULL VALIDATION PASS — MANUAL COLAB**.

Validation evidence:
- Targeted D2 + D1 scheduling + adapter tests: **32 passed**.
- Full `python -m pytest -q`: exit code **0**, reached 100%; the captured output did not expose an unambiguous aggregate test count, so none is inferred.
- Full `ruff check roboqa tests`: **All checks passed!**
- Full `python -m compileall -q roboqa tests`: exit code **0**.
- The initial Ruff run found only import ordering in `roboqa/integration/__init__.py`; corrected in commit `43de077cf01b16c3114d5f7dfeeeaf53a6d2057d`. Targeted Ruff and compileall then passed, followed by the full validation above.
- Validated branch HEAD: `43de077cf01b16c3114d5f7dfeeeaf53a6d2057d`.

## Next step

Phase 4.5-D3 will propagate replay and execution evidence into Quality Metrics and the deterministic Quality Gate. The Gate remains the sole decision authority.
