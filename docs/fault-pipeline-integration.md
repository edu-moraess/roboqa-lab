# RoboQA Lab — Fault Pipeline Integration (Phase 3.3-E)

## Purpose

Phase 3.3-E validates the integration boundary between immutable fault contracts,
raw sensor envelopes, the temporal scheduler, fault operators, overlapping
campaign orchestration, and provenance records.

This phase is an integration and adversarial validation layer. It does not add
new fault operators or change the normalized `Measurement` contract.

## End-to-end path

1. Define immutable `FaultScenario` templates.
2. Expand templates into deterministic occurrences with `TemporalScheduler`.
3. Select occurrences with overlapping half-open time windows.
4. Pass their scenarios to `FaultInjectionEngine.run_overlapping_campaign`.
5. Inspect each intermediate `FaultExecution`, event list, hashes, and
   `parent_run_id` provenance chain.
6. Compare final raw envelopes and evidence with expected behavior.

## Composition guarantees

- Schedule expansion is deterministic for identical templates and horizons.
- Occurrence seeds are derived from the template seed, template identity, and
  recurrence index.
- Overlapping campaign execution order is independent of caller scenario order:
  `(start_time, end_time, scenario_id)`.
- Scenario operators use local seeded random generators.
- The original input snapshot and every intermediate execution are retained.
- Each execution's output hash is the next execution's source hash.
- The provenance chain links each run to its immediate predecessor.
- Input envelopes and nested payloads are immutable; operators create new raw
  envelopes rather than mutating their inputs.
- Half-open scenario intervals preserve start-inclusive/end-exclusive behavior.
- Non-finite raw processing timestamps remain available in raw evidence but are
  omitted from normalized `FaultEvent.processing_time`.

## Conflict semantics

The current campaign API models logical concurrency with serial deterministic
composition; it does not start parallel threads or asynchronous workers.

- Removal operators prevent subsequent scenarios from observing removed
  envelopes.
- Duplication operators make each emitted copy visible to later scenarios.
- Later operators may create distinct events for copies that share the same
  source sequence number; event identity also includes the input position.
- A campaign requires at least one overlapping pair of scenario windows.
  Additional scenarios may be outside that overlapping pair and remain part of
  the deterministic campaign.
- Unsupported fault types fail closed. A failed campaign returns no partial
  campaign result; the original immutable inputs remain unchanged.

## Manual Colab validation

Run from the repository checkout:

```bash
python -m compileall -q roboqa tests
ruff check roboqa tests --output-format=full
pytest -q tests/integration/test_fault_pipeline_integration.py
pytest -q tests/adversarial/test_campaign_composition_adversarial.py
pytest -q tests/unit/test_overlapping_campaign.py
pytest -q
coverage erase
coverage run --branch --source=roboqa/fault_injection -m pytest -q
coverage report -m
```

The global coverage gate remains at least 90%. No GitHub Actions workflow is
required or introduced.

## Explicit non-goals

No ML training, model downloads, GPU dependency, CI workflow, parallel execution,
recovery policy, Quality Gate, Streamlit UI, Data Engine schema change, Fusion
Engine change, or merge into `main`.
