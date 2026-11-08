# RoboQA Lab — Phase 4.3: Deterministic Quality Gate

## Purpose

The Quality Gate evaluates Phase 4.2 `MetricObservation` values against an explicit, versioned policy and creates the Phase 4.0 immutable `QualityReport`. It is deterministic and policy-driven. It does not train models, infer thresholds, or accept an ML score as authority.

## Decision semantics

Decision precedence is:

1. `BLOCKED`: a required metric is absent, marked MISSING/INVALID, or has a unit mismatch.
2. `FAIL`: an available metric violates a hard minimum or maximum.
3. `PASS_WITH_WARNINGS`: an optional metric is absent/invalid, or an available metric breaches a configured warning band without breaching a hard limit.
4. `PASS`: every configured rule has acceptable evidence and no finding was emitted.

A required metric missing or invalid always blocks. A hard threshold breach is FAIL unless a required evidence blocker is also present, in which case BLOCKED takes precedence. This preserves both findings in the report.

## Explicit policy configuration

Each `MetricRule` identifies exactly one metric by name, scope, and optional sensor/window identity. Optional unit matching prevents comparing values in incompatible units.

- `minimum` and `maximum` are hard bounds (inclusive at equality).
- `warning_minimum` and `warning_maximum` define a warning band inside the hard bounds.
- No thresholds are inferred or shipped as universal defaults.
- A policy must have a non-empty version and unique rule IDs/metric identities.
- Only metrics named by policy rules are evaluated; other observations remain preserved in the report but are not implicitly judged.

Sensor rules normalize sensor IDs to lowercase. Window rules require exact finite half-open window bounds. Missing/invalid optional metrics create warnings; they do not silently become zero or PASS.

## API

```python
from roboqa.qa import MetricScope
from roboqa.qa.gate import MetricRule, QualityGatePolicy, evaluate_quality_gate

policy = QualityGatePolicy(
    policy_version="vehicle-qa-v1",
    rules=(
        MetricRule(
            rule_id="position-rmse",
            metric_name="position_rmse",
            scope=MetricScope.RUN,
            required=True,
            unit="m",
            maximum=1.0,
            warning_maximum=0.8,
        ),
    ),
)

report = evaluate_quality_gate(
    metrics,
    policy,
    run_id="experiment-001",
    generated_at="2026-11-08T05:00:00Z",
    source_hash="<64-character SHA-256>",
    output_hash="<64-character SHA-256>",
    configuration_hash="<64-character SHA-256>",
)
```

The caller supplies the run identity, UTC generation timestamp, and SHA-256 provenance digests. The Gate validates them through the existing `QualityReport` contract; it does not fabricate provenance or silently hash partial inputs.

## Scope and non-goals

Phase 4.3 provides the deterministic policy evaluator and immutable report. It does not define universal NIS/chi-square limits, infer required sensors, invent expected packet schedules, aggregate time windows automatically, connect injected faults end-to-end, add ML authority, or implement Streamlit. Those require explicit domain policy or later phases.

## Manual Colab validation

```bash
python -m compileall -q roboqa tests
ruff check roboqa tests
pytest -q tests/unit/test_quality_gate.py
pytest -q tests/unit/test_quality_contracts.py
pytest -q
```

Keep the phase unvalidated until the actual manual Colab outputs are reviewed.
