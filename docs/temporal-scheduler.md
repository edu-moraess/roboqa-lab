# RoboQA Lab — Temporal Scheduler (Phase 3.3-B)

## Purpose and boundary

The Temporal Scheduler expands declarative fault schedules into deterministic
FaultScenario occurrences. It is separate from FaultInjectionEngine: the
scheduler decides when a scenario is active; the engine decides how its
operator transforms eligible raw envelopes. Each generated scenario can be
passed directly to FaultInjectionEngine.run.

This phase does not add transport queues, real-time sleeping, asynchronous
workers, event-loop execution, recovery policies, concurrent mutation, or
clock-domain conversion.

## Contracts

- FaultSchedule wraps one immutable FaultScenario template.
- The template interval [start_time, end_time) defines the first occurrence
  and the duration of every recurrence.
- repeat_interval_s=None means one-shot execution.
- A positive repeat_interval_s enables fixed-interval recurrence. It must be
  greater than or equal to the scenario duration; recurrence within one
  schedule therefore cannot overlap.
- max_occurrences optionally bounds recurrence. Without it, the requested
  horizon bounds expansion.
- Different schedules may overlap. Their relative order is deterministic:
  occurrence start, source scenario ID, then occurrence index.
- The scheduler requires a finite half-open horizon
  [window_start, window_end). It selects occurrences that intersect the
  horizon but does not clip their fault intervals.

## Determinism and identity

Occurrence zero retains the template's initial interval. Later starts are
calculated as template.start_time + index * repeat_interval_s, avoiding
cumulative floating-point drift. Each occurrence gets a stable ID
<scenario_id>::occurrence:<zero-padded-index> and an independent 64-bit
seed derived from SHA-256 of the template seed, scenario ID, and occurrence
index. Identical schedule inputs produce identical IDs, seeds, and ordering.

The scheduler rejects duplicate template IDs and caps one expansion at
100,000 occurrences. It does not mutate the source scenario or schedule.

## Example

A caller creates a FaultScenario template, wraps it in FaultSchedule with
repeat_interval_s=30.0 and max_occurrences=4, then calls
TemporalScheduler.schedule(..., window_start=0.0, window_end=120.0).
Each returned occurrence contains a normal FaultScenario and can be executed
with FaultInjectionEngine.run(envelopes, occurrence.scenario).

For independent experiments, each occurrence receives the same baseline
envelope snapshot. Composition of outputs remains an explicit caller decision;
the existing Phase 3.2 run_campaign semantics are unchanged.

## Non-goals

No changes to Data Engine or Fusion Engine; no changes to Phase 3.2 sequential
campaign semantics; no probabilistic packet loss, burst transport, event queue,
wall-clock scheduler, processing simulator, Quality Gate, Streamlit UI,
training, downloads, or GitHub Actions.
