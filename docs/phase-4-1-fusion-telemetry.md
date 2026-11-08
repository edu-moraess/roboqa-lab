# RoboQA Lab — Phase 4.1: Fusion Telemetry

## Objective

Expose auditable mathematical evidence for each successful EKF measurement update without changing the filter state-estimation policy or weakening existing contracts.

## Delivered API

`InnovationTelemetry` is a frozen record containing timestamp, normalized sensor ID, measurement dimension / chi-square degrees of freedom, innovation vector y, innovation covariance S, and NIS.

Both arrays are copied defensively and marked read-only. The record validates finite values, dimensions, symmetric covariance, and a finite non-negative NIS.

## Numerical method

The EKF computes NIS = y.T @ solve(S, y) using `numpy.linalg.solve`; it never constructs an explicit matrix inverse. Singular solves raise `EKFNumericalError` before state mutation. Tiny negative NIS from floating-point roundoff is clamped only within a 1e-10 tolerance; materially negative values fail closed.

## Integration and compatibility

- `EKF.update(measurement)` returns `InnovationTelemetry`; callers that ignore the return value remain compatible.
- `run_filter` collects one record for every successful update in `FilterResult.telemetry`.
- Failed updates do not create telemetry records; earlier successful records remain available if a later update fails.
- `FilterResult` adds a trailing defaulted telemetry field to preserve existing constructor call sites.
- State, covariance, timestamp, and source measurements are not modified by telemetry collection.
- GPS uses 2 degrees of freedom; speed and yaw-rate use 1 each.

## Boundaries

No Quality Metrics aggregation, chi-square policy thresholds, PASS/FAIL decision engine, ML training, fault-to-QA adapter, Streamlit UI, physical sensor drivers, dataset downloads, GitHub Actions, or merge to main. Telemetry is evidence for later phases, not a quality verdict.

## Manual Colab validation

```bash
python -m compileall -q roboqa tests
ruff check roboqa tests
pytest -q tests/unit/test_fusion_telemetry.py
pytest -q tests/unit/test_fusion_ekf.py tests/unit/test_fusion_runner.py
pytest -q tests/integration/test_fusion_pipeline.py
pytest -q
coverage run --branch --source=roboqa/fusion -m pytest -q
coverage report -m
```

Do not mark this phase validated until actual Colab output confirms the checks.