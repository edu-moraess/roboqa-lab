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
| `PACKET_LOSS` | `packet-loss` | `loss_probability` in [0, 1] | Independent Bernoulli loss per eligible packet using the scenario seed |
| `INTERMITTENT_OUTAGE` | `burst-loss` | `burst_intervals_s`: sorted, non-overlapping `[start, end)` intervals | Removes packets acquired inside explicitly declared burst windows |
| `DELAY` | `delay` | constant `delay_s`; uniform `min_delay_s`/`max_delay_s`; exponential `mean_delay_s` | Adds a non-negative sampled transport delay; acquisition time is preserved |
| `OUT_OF_ORDER` | `arrival-shift` | finite `shift_s`, optional `sequence_numbers` | Shifts selected packet arrival times; negative values can reorder a selected subset |
| `BIAS` | `bias` | `offset` vector | Adds component-wise offset to payload `z` |
| `OUTLIER` | `outlier` | `sigma > 0`, optional `index` | Adds seeded Gaussian perturbation to one `z` component |
| `INVALID_PAYLOAD` | `invalid-payload` | optional `kind`, `index` | Injects NaN or +/-Inf into raw `z` |
| `DUPLICATE` | `duplicate` | optional `duplicate_delay_s >= 0` | Emits original plus duplicate with the same source sequence identity |

Other values in `FaultType` are deliberately unsupported in this phase and fail closed.

## Transport fault semantics (Phase 3.3-C)

`PACKET_LOSS` performs an independent Bernoulli trial for each eligible envelope.
`loss_probability=0` preserves all packets; `loss_probability=1` removes all
eligible packets. Intermediate probabilities use the scenario-local NumPy RNG,
so the same input order, parameters, and seed reproduce the same decisions.
A packet retained by the stochastic operator is recorded with
`FaultEvent.affected=false` and counted as skipped; a run with no actual changes
is reported as `NOT_APPLICABLE`.

`INTERMITTENT_OUTAGE` models deterministic burst windows. Each interval is
start-inclusive and end-exclusive and is evaluated against acquisition time.
Intervals must be sorted and non-overlapping. Packets outside the declared
windows are not candidates for the operator and remain unchanged. This models
explicit outage bursts, not a learned or stateful Gilbert-Elliott channel.

`DELAY` preserves the legacy constant-delay parameter `delay_s`. It additionally
supports `distribution="uniform"` with non-negative `min_delay_s` and
`max_delay_s`, or `distribution="exponential"` with positive `mean_delay_s`.
The RNG is local to the scenario. All sampled delays are non-negative; only
arrival time changes. Arrival sorting in the engine can consequently expose
reordering when packets receive different delays. Delay, arrival-shift, and
duplicate operators reject arithmetic overflow that would produce a non-finite
arrival timestamp. The existing `OUT_OF_ORDER` arrival-shift operator remains
available for explicit selected-packet shifts.

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

## Three-clock temporal contract (Phase 3.3-A)

RawMeasurementEnvelope carries three distinct timestamps, expressed in seconds
on the same simulation-time coordinate:

- acquisition_time: when the sensor acquired the measurement.
- arrival_time: when the transport boundary received the envelope.
- processing_time: when downstream processing handled the envelope; optional
  until that processing stage has occurred.

The optional processing_time field is appended after the existing constructor
fields, preserving positional and keyword compatibility for Phase 3.1/3.2
callers. Existing operators preserve it unchanged. Delay and arrival-shift
operators modify only arrival_time; they do not fabricate a processing time.

When the required timestamps are finite, the envelope exposes:

- transport_latency_s = arrival_time - acquisition_time
- processing_wait_s = processing_time - arrival_time
- end_to_end_latency_s = processing_time - acquisition_time

Metrics that require a missing or non-finite timestamp return None. Negative
deltas are retained as evidence, not silently clamped: arrival_precedes_acquisition
and processing_precedes_arrival are exposed through temporal_diagnostics.
This is intentional at the raw boundary, where clock anomalies must remain
inspectable. temporal_order_valid currently checks that a present processing
timestamp does not precede arrival; it does not reject the envelope.

A non-finite processing timestamp is preserved for diagnostics and causes
processing_timestamp_finite / is_valid_candidate to report false; the existing
timestamps_finite property retains its Phase 3.2 meaning and checks acquisition
and arrival only. Such an envelope remains eligible for transport fault
injection when its acquisition and arrival timestamps are finite. The raw
processing timestamp participates in canonical envelope hashes, while the
normalized FaultEvent omits a non-finite processing timestamp. This phase
defines metadata and diagnostics only: it does not add a scheduler, buffering
policy, processing simulator, or clock-domain conversion.

## Campaign orchestration

`FaultInjectionEngine.run_campaign(envelopes, scenarios)` applies a non-empty
sequence of unique scenarios in the exact order supplied by the caller. Each
scenario consumes the previous scenario's output. The returned
`FaultCampaignExecution` retains the original input and every intermediate
`FaultExecution`. Provenance records are linked through `parent_run_id`.
Campaign order is explicit and is never inferred or silently reordered.

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
