# RoboQA Lab — Fault Injection Contracts (Phase 3.1)

## Scope

Phase 3.1 establishes immutable contracts and the raw-event boundary. It does
not implement fault operators, recovery policies, the Quality Gate, or UI.

## Temporal semantics

- `acquisition_time`: sensor-side measurement time.
- `arrival_time`: transport arrival time.
- `sequence_number`: receive-stream sequence identity.
- Processing time belongs to the future event-processing layer.

A scenario interval is `[start_time, end_time)`. A delayed event retains its
acquisition timestamp. The raw envelope deliberately permits non-finite
timestamps and payload numbers so that the validation boundary can diagnose
corrupted input. It is not a `Measurement` and must never be passed directly
to the Fusion Engine.

## Immutability and parameter model

Scenario parameters and normalized event metadata accept JSON-like values.
Mappings become read-only mappings and sequences become tuples. Non-finite
numbers are rejected from normalized contracts. Raw envelopes preserve
non-finite numeric values intentionally.

## Deterministic provenance

Canonical JSON uses sorted keys, compact separators, UTF-8 and rejects NaN/Inf.
SHA-256 digests are calculated from canonical JSON, not Python's built-in
`hash()`. Reproducibility additionally requires the same input data,
operator implementation/version, scenario configuration and local RNG seed.

## Operator interface

The `FaultOperator` protocol defines validation, application and description.
Implementations must be stateless across runs, must not mutate input data, and
must use the RNG supplied by the execution layer. No concrete operator is
included in this phase.

## Non-goals

- No changes to Data Engine schemas or Fusion Engine behavior.
- No automatic data downloads, training, GPU use or external services.
- No Streamlit integration.
- No GitHub Actions requirement.
