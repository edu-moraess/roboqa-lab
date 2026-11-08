# Phase 4.5-B — Deterministic Transport Replay

**Implementation status:** implemented; **manual Colab validation pending**.

## Objective

Turn the Phase 4.5-A delivery contracts into a deterministic offline replay boundary. The replay reconciles a source snapshot with the delivered batch, applies explicit ordering and lateness semantics, suppresses duplicate identities, classifies invalid packets, and accounts for source packets that never arrived.

This is a batch replay primitive. It is not a live network queue, wall-clock scheduler, or Fusion Engine adapter.

## API

`roboqa.integration.replay_transport(source_envelopes, delivered_envelopes, *, ordering_policy, allowed_lateness_s, run_id)`

Returns an immutable `TransportReplayResult`:
- `summary`: existing `TransportReplaySummary`, with source/delivery hashes, configuration hash, counts, and one `DeliveryDecision` for every delivered copy plus every dropped source packet.
- `accepted_envelopes`: only first-seen, valid, non-late envelopes, in deterministic replay order.

The public symbols are exported from `roboqa.integration`.

## Normative semantics

### Source identity

Identity is the normalized tuple `(sensor_id, source_id, sequence_number)`. Sensor IDs are lowercased by `EnvelopeIdentity`; `source_id` remains case-sensitive. Sequence numbers are scoped to sensor and source.

- The source snapshot must contain unique identities.
- Every delivered identity must exist in the source snapshot.
- Unknown deliveries and ambiguous duplicate identities in the source snapshot fail closed with `IntegrationContractError`.

### Ordering and deterministic ties

- `ARRIVAL_ORDER`: ascending `arrival_time`.
- `ACQUISITION_ORDER`: ascending `acquisition_time`.
- Equal clock values are resolved by normalized sensor ID, source ID (empty string for `None`), sequence number, then canonical envelope SHA-256 fingerprint.
- The canonical ordering is independent of caller collection order, including same-identity copies with different payloads.

### Duplicate suppression

The first delivery for an identity in the selected deterministic order wins. Every later delivery for that identity is `DUPLICATE` and is excluded from `accepted_envelopes`. The first copy remains authoritative even if it is ultimately classified `INVALID` or `LATE`; a later copy does not silently replace it.

### Late packets and watermark

Only `ARRIVAL_ORDER` applies event-time lateness:
- Event time is `acquisition_time`.
- `max_acquisition_seen` is the greatest valid acquisition time observed so far in arrival order.
- `watermark = max_acquisition_seen - allowed_lateness_s`.
- A first-seen packet with `acquisition_time < watermark` is `LATE`; equality is accepted.
- `allowed_lateness_s` must be finite and non-negative.
- `ACQUISITION_ORDER` is an offline event-time sort and does not classify packets as late.

This is a deterministic batch policy, not a streaming watermark with idle-source detection, timer-based finalization, or post-finalization correction.

### Invalid temporal/data evidence

A first-seen packet is `INVALID` if:
- acquisition or arrival timestamp is non-finite;
- a present processing timestamp is non-finite;
- payload contains NaN/Inf;
- `arrival_time < acquisition_time`; or
- a present `processing_time < arrival_time`.

The raw envelope remains immutable and available to the caller for diagnostics. The replay does not coerce or repair it. A missing processing timestamp is permitted and remains missing.

### Dropped-packet accounting

A source identity absent from the delivered batch receives one `DROPPED` decision. Dropped decisions preserve source acquisition time but explicitly set arrival and processing times to `None`; they do not claim delivery.

The summary enforces:
- `delivered_count = accepted + duplicate + late + invalid`
- `source_count = accepted + late + invalid + dropped`
- `len(decisions) = delivered_count + dropped_count`

Duplicates are additional delivery copies, not additional unique source packets.

## Provenance and determinism

- Source hash uses canonical source-envelope order.
- Delivered hash uses deterministic replay order.
- Configuration hash records ordering, lateness, duplicate, late, and invalid policies.
- Default run ID is derived from source, delivered, and configuration hashes. A caller may provide an explicit non-empty run ID.

## Boundaries / non-goals

- No production conversion from `RawMeasurementEnvelope` to `Measurement`.
- No invocation or modification of `run_filter`.
- No claim that `ARRIVAL_ORDER` changes Fusion execution: the current runner still sorts normalized `Measurement.t`.
- No online buffering, network I/O, async workers, timers, or streaming watermark finalization.
- No Quality Metrics or Quality Gate policy changes.
- No ML training/inference, dataset downloads, Streamlit, CI/GitHub Actions, or merge.

The next integration phase must define a separate adapter/scheduling boundary and test its effect on Fusion telemetry and QA. The strict normalized `Measurement` contract remains unchanged.

## Manual Colab validation

After syncing and switching to `feat/temporal-replay-phase-4-5b`:

```bash
python -m pytest -q tests/unit/test_transport_replay.py
ruff check roboqa/integration tests/unit/test_transport_replay.py
python -m compileall -q roboqa/integration tests/unit/test_transport_replay.py
python -m pytest -q
ruff check roboqa tests
python -m compileall -q roboqa tests
```

Keep the PR Draft until actual Colab outputs are reviewed. Do not infer a total test count from a progress bar without an explicit aggregate summary.
