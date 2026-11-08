# Phase 4.5-A — Fault-to-QA Integration Contracts

**Status:** VALIDATION PASS — manual Google Colab.

## Objective

Formalize the evidence boundary between transport-level fault injection and downstream sensor fusion / quality evaluation before implementing a production adapter or replay scheduler.

The integrated Colab experiment exposed three concrete gaps:

1. The experimental adapter mapped `acquisition_time` to `Measurement.t`, so a delayed `arrival_time` changed latency metrics but did not change Fusion Engine scheduling.
2. Duplicate deliveries reused the source sequence number and were passed into fusion as independent updates; the existing availability metric correctly became INVALID when accepted updates exceeded the expected denominator.
3. A packet's identity and disposition were not represented in a first-class integration contract, making duplicate suppression, late-packet policy, and dropped-packet accounting implicit.

## Contracts introduced

- `EnvelopeIdentity`: sensor + optional source + sequence number.
- `DeliveryDecision`: explicit ACCEPTED / DUPLICATE / LATE / INVALID / DROPPED disposition with preserved three-clock evidence and a reason for non-accepted deliveries.
- `TransportOrderingPolicy`: explicit ARRIVAL_ORDER or ACQUISITION_ORDER policy.
- `TransportReplaySummary`: count reconciliation plus source, delivered, and configuration SHA-256 hashes.

The summary accounts for each delivered packet (including duplicates) and each source packet proven dropped. Non-finite raw timestamps may be preserved only in INVALID/DROPPED evidence; accepted deliveries require finite acquisition and arrival timestamps, and processing cannot precede arrival.

## Explicit non-goals

This checkpoint does **not** implement:
- packet scheduling, watermarks, bounded-lateness buffers, or replay;
- conversion from RawMeasurementEnvelope to Measurement;
- duplicate suppression or late-packet handling in the Fusion Engine;
- metric additions or changes to Quality Gate policy;
- model training, inference, dataset downloads, Streamlit UI, or CI.

These behaviors require an explicit policy decision and their own tests. In particular, ARRIVAL_ORDER is a declared policy value, not evidence that the current `run_filter` processes arrival order: the current Fusion runner sorts `Measurement.t` (acquisition time).

## Validation plan

Run in Google Colab after switching to `feat/fault-to-qa-contracts-phase-4-5a`:

```bash
python -m pytest -q tests/unit/test_fault_to_qa_contracts.py
ruff check roboqa/integration tests/unit/test_fault_to_qa_contracts.py
python -m compileall -q roboqa/integration tests/unit/test_fault_to_qa_contracts.py
python -m pytest -q
ruff check roboqa tests
python -m compileall -q roboqa tests
```

## Manual validation evidence

Validated on the Phase 4.5-A branch after the final test and Ruff fixes:

- `python -m pytest -q tests/unit/test_fault_to_qa_contracts.py`: **52 passed**.
- `python -m pytest -q`: completed at 100% with exit code 0; no aggregate test count is inferred from the captured output.
- `ruff check roboqa tests`: **All checks passed!**
- `python -m compileall -q roboqa tests`: completed without errors.
- Targeted compileall for `roboqa/integration` and its unit tests also completed without errors.

Result: **VALIDATION PASS** for the Phase 4.5-A contracts and repository compatibility. This does not validate transport replay, scheduling, adapter semantics, or end-to-end fault-to-fusion execution.
