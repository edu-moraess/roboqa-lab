# RoboQA Lab — Fault Injection Engine (Phase 3.2)

## Scope

Phase 3.2 adds deterministic operators and an execution layer over
`RawMeasurementEnvelope`. It does not alter Data Engine schemas, Fusion Engine,
or the Phase 3.1 immutable contracts. It does not implement recovery policies,
Quality Gate, or Streamlit.

## Supported operators

| Fault type | Operator | Parameters | Behavior |
|---|---|---|---|
| `SENSOR_DROPOUT` | `dropout` | none | Removes eligible envelopes |
| `PACKET_LOSS` | `packet-loss` | none | Removes eligible packets |
| `DELAY` | `delay` | `delay_s >= 0` | Increases arrival time; acquisition time is preserved |
| `OUT_OF_ORDER` | `arrival-shift` | finite `shift_s`, optional `sequence_numbers` | Shifts selected packet arrival times; negative values can reorder a selected subset |
| `BIAS` | `bias` | `offset` vector | Adds component-wise offset to payload `z` |
| `OUTLIER` | `outlier` | `sigma > 0`, optional `index` | Adds seeded Gaussian perturbation to one `z` component |
| `INVALID_PAYLOAD` | `invalid-payload` | optional `kind`, `index` | Injects NaN or +/-Inf into raw `z` |
| `DUPLICATE` | `duplicate` | optional `duplicate_delay_s >= 0` | Emits original plus duplicate with the same source sequence identity |

Other values in `FaultType` are deliberately unsupported in this phase and fail closed.

## Temporal and ordering semantics

A scenario targets only its declared sensors and the half-open acquisition
interval `[start_time, end_time)`. The acquisition timestamp is never changed
by a transport fault. Delay and arrival-shift affect only `arrival_time`.
The engine emits the resulting tuple sorted by arrival time, then source
sequence number, then original input index. The sequence number is retained as
source identity, not rewritten to match transport order.

Envelopes with non-finite acquisition or arrival timestamps are preserved
unchanged and counted as skipped because they cannot be safely selected or
represented in normalized `FaultEvent` timestamps. Non-finite payload values
are preserved and can be hashed using an explicit tagged representation.

## Determinism, atomicity, and provenance

Each run creates a local NumPy generator from `scenario.seed`; no global RNG is
used. Operators are stateless and cannot mutate frozen input envelopes. An
operator failure aborts the run before a `FaultExecution` is returned. The
source and output hashes use canonical JSON; raw NaN and infinities are encoded
as explicit tagged values before hashing. Provenance includes operator version,
configuration hash, seed, and UTC creation time. Run identifiers and data hashes
are stable for identical inputs/configuration; creation time is intentionally
execution-specific.

## API

```python
from roboqa.fault_injection import (
    FaultInjectionEngine, FaultScenario, FaultType, RawMeasurementEnvelope,
)

envelopes = (
    RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"z": [10.0, 20.0]}),
)
scenario = FaultScenario(
    scenario_id="gps-delay-01",
    fault_type=FaultType.DELAY,
    target_sensors=("gps",),
    start_time=0.0,
    end_time=5.0,
    seed=42,
    parameters={"delay_s": 0.25},
)
execution = FaultInjectionEngine().run(envelopes, scenario)
assert execution.output_envelopes[0].acquisition_time == 1.0
assert execution.output_envelopes[0].arrival_time == 1.35
```

## Non-goals

No automatic downloads, training, GPU, network access, GitHub Actions, changes
to normalized `Measurement`, recovery policies, Quality Gate, or UI.
