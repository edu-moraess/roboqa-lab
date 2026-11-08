# RoboQA Lab — Engineering Status Report

**Date:** 2026-11-08  
**Organization:** ArqTech Labs  
**Repository:** https://github.com/edu-moraess/roboqa-lab  
**Snapshot branch:** `feat/quality-contracts-phase-4-0`  
**Snapshot HEAD before this report:** `78e4725f2848b6968c3f4b27e5d8a29eb33355ef`

## 1. Executive summary

RoboQA Lab is being developed as a modular quality-assurance platform for sensor-fusion and state-estimation systems. Its intended scope goes beyond an EKF demonstration: the platform is being built around data contracts, sensor-fusion estimation, controlled fault injection, temporal and transport simulation, measurable quality evidence, deterministic quality decisions, and a Streamlit Control Center.

As of this report, the Data Engine, Fusion Engine hardening, Fault Injection pipeline, temporal scheduler, and Phase 4.0 quality contracts have been implemented and manually validated in Google Colab. The next planned implementation step is **Phase 4.1 — Fusion Telemetry**.

**Overall classification: IMPLEMENTATION IN PROGRESS — VALIDATED THROUGH PHASE 4.0.**  
This is not a claim that the full product or roadmap is complete.

## 2. Product architecture and intended roadmap

1. **Data Engine:** ground truth, measurements, synthetic data, dataset ingestion, and provenance.
2. **Fusion Engine:** CTRV motion model, EKF, sensor updates, and deterministic execution.
3. **Fault Injection:** immutable raw envelopes, fault operators, deterministic campaigns, and provenance.
4. **Quality Assurance:** telemetry, metrics, deterministic policies, and quality reports.
5. **ML Intelligence:** advisory anomaly detection after the evidence and metric pipeline is validated.
6. **Streamlit Control Center:** operational visualization, diagnostics, experiment control, and reports.
7. **Hardening and release:** reproducibility, documentation, packaging, security, and publication.

The Control Center remains a core product component in the roadmap; it has **not** been implemented yet. ML is intended to provide advisory evidence only and must never override deterministic safety or quality rules.

## 3. Delivery and validation history

Validation evidence below comes from manual Google Colab runs reported during the engineering workflow. No GitHub Actions workflow is used as the validation authority. A phase is marked validated only where the corresponding manual checks were reported as passing.

| Phase | Scope | Reported validation | Status |
|---|---|---|---|
| 1 | Data Engine contracts and ingestion foundation | Unit/adversarial tests, reported 92% coverage, Ruff clean | VALIDATED |
| 2.2 | CTRV + EKF hardening and process-noise model | Full test suite, reported 95% coverage, Ruff clean | VALIDATED |
| 3.1 | Fault contracts, raw envelopes, provenance | Full suite, 90% package coverage, Ruff clean | VALIDATED |
| 3.2 | Fault operators and deterministic campaign engine | Full suite, 94% package coverage, Ruff clean | VALIDATED |
| 3.3-A | Three-clock temporal contracts | Full suite, 95% package coverage, Ruff clean | VALIDATED |
| 3.3-B | Temporal scheduler and recurring occurrences | Full suite, 94% package coverage, Ruff clean | VALIDATED |
| 3.3-C | Probabilistic packet loss, burst loss, delay distributions | Full suite, 94% package coverage, Ruff clean | VALIDATED |
| 3.3-D | Overlapping campaigns and deterministic composition | Full suite, 94% package coverage, Ruff clean | VALIDATED |
| 3.3-E | Pipeline integration and adversarial validation | Integration/adversarial tests, full suite, 94% package coverage, Ruff clean | VALIDATED |
| 4.0 | Immutable quality metric/finding/report contracts | 23 targeted tests, full suite, 90% `roboqa/qa` coverage, Ruff clean | VALIDATED |

Exact aggregate test counts are intentionally omitted where the captured validation output did not expose an unambiguous final count.

### Phase 1 — Data Engine

- Immutable `GroundTruth` and `Measurement` contracts.
- Defensive copies and read-only arrays.
- Synthetic data and dataset-ingestion foundations.
- Data integrity and adversarial tests were reported as passing.

### Phase 2 — Fusion Engine

- CTRV state representation: `[x, y, psi, v, omega]`.
- Analytical CTRV transition/Jacobian and angle normalization.
- EKF with Joseph covariance update.
- GPS, speed, and yaw-rate measurement support.
- Deterministic event ordering and temporal validation.
- Discrete zero-order-hold process-noise model with documented approximation boundaries.
- Atomic predict/update behavior and covariance validation.

### Phase 3 — Fault Injection and temporal pipeline

- Immutable fault scenarios, events, execution records, and recovery/provenance contracts.
- Raw measurement envelopes preserve malformed/non-finite payloads without weakening normalized `Measurement` contracts.
- Separate acquisition, arrival, and processing clocks, including derived latency diagnostics.
- Deterministic seeded operators for bias, outliers, delay, arrival shifts/reordering, duplication, dropout, packet loss, and raw payload corruption.
- Probabilistic packet loss, burst-loss windows, and uniform/exponential delay distributions.
- Temporal scheduler for one-shot and recurring scenarios with deterministic occurrence IDs/seeds and half-open horizons.
- Overlapping campaign composition with deterministic serial semantics and provenance chaining.
- Integration and adversarial tests for the end-to-end fault pipeline.

**Important semantic boundary:** overlapping campaigns currently implement deterministic logical composition, not parallel threads or asynchronous execution. The legacy packet-loss behavior was all-or-nothing before Phase 3.3-C; probabilistic semantics were introduced in 3.3-C. The visual experiment from Phase 3.2 showed the behavior of that earlier implementation and should not be cited as evidence of the later stochastic model.

### Phase 4.0 — Quality contracts

- `MetricObservation`: explicit availability (`available`, `missing`, `invalid`), units, scope, and sample count.
- `QualityFinding`: severity, rule identity, explanatory message, expected/observed values, and evidence references.
- `QualityReport`: immutable aggregate, decision field, UTC timestamp, policy version, and source/output/configuration SHA-256 hashes.
- Validation for finite values, scope metadata, unique metric identities, hashes, and timestamps.
- Decision states: `PASS`, `PASS_WITH_WARNINGS`, `FAIL`, and `BLOCKED`.
- Planned metric catalog includes position/velocity RMSE, innovation norm, NIS, sensor availability, latency, update rejection rate, and covariance trace.

Phase 4.0 defines contracts only. It does **not** yet calculate NIS or the other metrics, evaluate gate policies, or produce operational quality decisions.

## 4. Current validated checkpoint

- **Branch:** `feat/quality-contracts-phase-4-0`
- **HEAD before this report:** `78e4725f2848b6968c3f4b27e5d8a29eb33355ef`
- **Commit:** `Apply Ruff import formatting to quality contract tests`
- **Commit URL:** https://github.com/edu-moraess/roboqa-lab/commit/78e4725f2848b6968c3f4b27e5d8a29eb33355ef
- **Latest reported manual checks:** Ruff clean; complete pytest suite passed; Phase 4.0 targeted tests 23/23; QA package coverage 90%.
- **Working environment:** Google Colab. Temporary local artifacts such as `.coverage`, `__pycache__`, and `*.egg-info` are generated by validation and are not source deliverables.

## 5. Pull-request and branch governance

The work has been organized into incremental branches and pull requests. At the time of this report, PRs #1 through #9 are intended to remain open and unmerged, preserving the staged review history. The current Phase 4.0 branch is the documentation/reporting checkpoint.

- PR #1 — Fusion Engine / Phase 2.2: https://github.com/edu-moraess/roboqa-lab/pull/1
- PR #2 — Fault Injection contracts: https://github.com/edu-moraess/roboqa-lab/pull/2
- PR #3 — Fault Injection engine: https://github.com/edu-moraess/roboqa-lab/pull/3
- PR #4 — Three-clock temporal contracts: https://github.com/edu-moraess/roboqa-lab/pull/4
- PR #5 — Temporal scheduler: https://github.com/edu-moraess/roboqa-lab/pull/5
- PR #6 — Transport fault model: https://github.com/edu-moraess/roboqa-lab/pull/6
- PR #7 — Overlapping campaigns: https://github.com/edu-moraess/roboqa-lab/pull/7
- PR #8 — Fault pipeline integration/adversarial validation: https://github.com/edu-moraess/roboqa-lab/pull/8
- PR #9 — Phase 4.0 quality contracts: https://github.com/edu-moraess/roboqa-lab/pull/9

Do not merge or retarget these PRs as part of documentation maintenance. The default branch is not represented as containing all work described here; the validated implementation is distributed across the staged branch/PR chain.

## 6. Outstanding work and next phases

### Immediate next step — Phase 4.1: Fusion Telemetry

Instrument each filter update with immutable, provenance-linked telemetry containing:

- innovation/residual vector (y);
- innovation covariance (S);
- normalized innovation squared (NIS), calculated using a stable linear solve rather than explicitly forming (S^{-1});
- sensor identity and measurement degrees of freedom;
- event/update timestamps and accepted/rejected outcome.

Preserve existing EKF result compatibility, validate singular/ill-conditioned covariance behavior, and add unit, integration, and adversarial tests. Do not implement the Quality Gate or ML training in this step.

### Subsequent roadmap

- **4.2 — Quality Metrics Engine:** formally calculate metrics, document alignment/denominators/aggregation, and handle missing/invalid samples.
- **4.3 — Deterministic Quality Gate:** evaluate required evidence and policy rules; missing or invalid required evidence must fail closed as `BLOCKED`.
- **4.4 — ML Intelligence:** only after metrics and datasets are validated; begin with an explainable baseline and rigorous evaluation. ML remains advisory.
- **4.5 — Fault-to-QA integration:** connect injected scenarios to telemetry, metrics, findings, and reports.
- **4.6 — Streamlit Control Center:** implement the central operational UI, visual diagnostics, experiment controls, and report inspection.
- **Later hardening/release:** documentation, reproducibility, packaging, security, and publication.

## 7. Engineering guardrails

- Audit first; inspect branch, HEAD, working tree, and scope before changes.
- Preserve commit history and existing branches. No destructive reset, force push, branch recreation, or silent fallback.
- No merge without explicit approval.
- Manual Colab validation is the project validation path; do not add GitHub Actions as a substitute.
- No model training, large builds, automatic dataset downloads, or weight downloads before the relevant audit and validation gates pass.
- Never fabricate metrics, test counts, coverage, runtime performance, or dataset evidence.
- Keep raw malformed sensor data at the raw-envelope boundary; do not weaken normalized domain contracts to accommodate injected faults.
- Keep deterministic quality policy authoritative; ML outputs are advisory only.
- Record exact branch, commit, commands, and outputs for each validation checkpoint.

## 8. Current conclusion

The project has a substantial validated foundation through Phase 4.0, but the end-to-end product is not complete. Fusion telemetry, metric calculation, deterministic quality evaluation, ML advisory analysis, and the Streamlit Control Center remain future work. The next work session should begin with a read-only inspection of the Phase 4.0 branch and then implement Phase 4.1 in a new branch based on the validated checkpoint.


---

## 9. Engineering addendum — Phase 4.1 and Phase 4.2 (2026-11-08)

This addendum supersedes the earlier "next step — Phase 4.1" wording in the historical snapshot above.

### Phase 4.1 — Fusion Telemetry

- Branch: `feat/fusion-telemetry-phase-4-1`
- Validated HEAD: `5640bf5404eebc0ab2c22445c9efb0ed1f877f8c`
- PR #10: https://github.com/edu-moraess/roboqa-lab/pull/10
- Manual Colab evidence supplied: compileall PASS; Ruff PASS; telemetry tests 14 passed; EKF/runner tests 51 passed; integration tests 3 passed; complete suite 378 passed; Fusion Engine branch coverage 94%.
- Result: **VALIDATION PASS**. PR remains separate and unmerged.

### Phase 4.2 — Quality Metrics Engine

- Branch: `feat/quality-metrics-phase-4-2`
- Base: validated Phase 4.1 HEAD above.
- PR #11: https://github.com/edu-moraess/roboqa-lab/pull/11
- Current implementation HEAD at time of addendum: `f722183436f3954a4733f9278cd497731f429b0b`
- Implemented calculations: position RMSE (explicit aligned GroundTruth), mean covariance trace, per-sensor innovation norm and NIS means, sensor update availability (explicit expected counts), three-clock latency means, and temporal rejection rate.
- Missing and invalid evidence are represented using existing MetricAvailability states; no absent evidence is silently converted to zero.
- Current runner does not expose a complete rejected-measurement ledger. The metric is therefore explicitly named temporal_rejection_rate and does not claim to be a general update rejection rate.
- Availability is accepted-update ratio, not physical sensor uptime.
- Validation: **PENDING MANUAL COLAB**. No test, Ruff, compile, or coverage result is claimed for Phase 4.2 yet.

### Next step

Run the documented Phase 4.2 Colab validation. Fix any evidence-backed failures before marking the phase validated. Phase 4.3 Quality Gate remains out of scope until Phase 4.2 passes.

---

## 10. Engineering addendum — Phase 4.2 targeted test evidence (2026-11-08)

- Branch: `feat/quality-metrics-phase-4-2`
- PR: https://github.com/edu-moraess/roboqa-lab/pull/11
- Corrected the GPS mean innovation norm test expectation from 4.0 to 3.5. The fixture contains innovation vectors `[3, 4]` and `[0, 2]`, whose Euclidean norms are 5 and 2; their arithmetic mean is 3.5.
- Added contract tests for invalid FilterResult types/shapes, non-finite and non-monotonic evidence, invalid GroundTruth, absent RMSE overlap/alignment, invalid tolerance and envelope inputs, expected-count validation, asymmetric covariance, eigendecomposition failure, empty results, and incomplete/non-finite clock evidence.
- Manual Colab targeted coverage report for `roboqa/qa/metrics.py`: **142 statements, 0 missed; 76 branches, 0 partial; 100% coverage**.
- The targeted test run associated with this coverage report completed successfully.
- **Overall Phase 4.2 validation remains PENDING** until the complete pytest suite, Ruff, and compileall outputs are reviewed. No claim is made here that those checks passed.
- PR #11 remains Draft and unmerged. Phase 4.3 remains gated on complete Phase 4.2 validation.

---

## 11. Engineering addendum — Phase 4.2 validation and Phase 4.3 implementation (2026-11-08)

This addendum supersedes the earlier Phase 4.2 pending-validation statements in Sections 9 and 10, which are retained as historical snapshots.

### Phase 4.2 — Quality Metrics Engine: VALIDATION PASS

- Branch: `feat/quality-metrics-phase-4-2`
- Final checkpoint inherited by Phase 4.3: `661167cc8569693dc24bf749063afed6dd540f4c`
- Manual Colab evidence supplied: complete `pytest -q` reached 100% with no reported failures; Ruff reported `All checks passed!`; `python -m compileall -q roboqa tests` completed without errors.
- Targeted coverage for `roboqa/qa/metrics.py`: 100% statements and branches (142 statements, 0 missed; 76 branches, 0 partial).
- Exact total test count is not recorded because the captured terminal output did not expose a final aggregate count.
- Result: **VALIDATION PASS**. PR #11 remains open, Draft, and unmerged.

### Phase 4.3 — Deterministic Quality Gate: IMPLEMENTED, VALIDATION PENDING

- Branch: `feat/deterministic-quality-gate-phase-4-3`
- Base: Phase 4.2 checkpoint `661167cc8569693dc24bf749063afed6dd540f4c`.
- PR #12: https://github.com/edu-moraess/roboqa-lab/pull/12
- Added `MetricRule`, `QualityGatePolicy`, `QualityGateError`, and `evaluate_quality_gate`.
- Rules match exact metric identity (name, scope, optional sensor/window) and can validate units.
- Explicit hard bounds produce FAIL; warning bands produce PASS_WITH_WARNINGS.
- Required missing/invalid evidence or required unit mismatch produces BLOCKED. Optional missing/invalid evidence produces a warning. Decision precedence is BLOCKED, FAIL, PASS_WITH_WARNINGS, PASS.
- The output is the existing immutable `QualityReport`, preserving all input observations, findings, policy version, UTC timestamp, and caller-supplied SHA-256 provenance.
- Unconfigured observations remain in the report but are not implicitly judged. No universal thresholds are invented; no ML authority is introduced.
- Added unit tests and policy/API documentation.
- Validation: **PENDING MANUAL COLAB**. Do not mark this phase complete until compileall, Ruff, targeted tests, and the full suite are reviewed.
- PR #12 remains Draft; no PR was merged and `main` was not changed.



---

## 12. Engineering addendum — Phase 4.3 validation and Phase 4.4-A implementation (2026-11-08)

### Phase 4.3 — Deterministic Quality Gate: VALIDATION PASS

- Validated branch: `feat/deterministic-quality-gate-phase-4-3`.
- Validated HEAD used for the Phase 4.4-A branch: `1390ddd45c0da9fc4a0dfc4fd1d897c783d46105`.
- Manual Colab evidence supplied: complete `python -m pytest -q` reached 100% with no reported failures; `ruff check roboqa tests --output-format=full` reported `All checks passed!`; `python -m compileall -q roboqa tests` completed without errors.
- Exact total test count was not exposed in the captured output and is intentionally not inferred.
- Result: **VALIDATION PASS**. PR #12 remains open, Draft, and unmerged; no changes were made to `main`.

### Phase 4.4-A — ML Feature and Dataset Contracts: IMPLEMENTED, VALIDATION PENDING

- Branch: `feat/ml-intelligence-contracts-phase-4-4a`.
- Base: validated Phase 4.3 HEAD `1390ddd45c0da9fc4a0dfc4fd1d897c783d46105`.
- Added `roboqa.ml` contracts: `FeatureValue`, `FeatureSample`, `FeatureDataset`, `FeatureAvailability`, `DatasetSplit`, and evaluation-only `AnomalyReference`.
- Feature availability is explicit (AVAILABLE/MISSING/INVALID); missing/invalid values cannot carry numeric sentinels. No imputation or silent dropping occurs.
- Dataset rows require a consistent feature-name schema and units. Missing feature slots must be represented explicitly.
- All windows belonging to one `run_id` must remain in one split, preventing run-level leakage across TRAIN/VALIDATION/TEST.
- Dataset identity, UTC creation timestamp, source/configuration SHA-256, unique sample IDs, and finite ordered windows are validated.
- Optional NORMAL/ANOMALOUS/UNKNOWN reference labels are evaluation metadata, not feature values or model inputs.
- Added unit tests and Phase 4.4-A contract documentation.
- No feature extraction, preprocessing, training, inference, model dependency, dataset download, or Quality Gate integration was added.
- Validation: **PENDING MANUAL COLAB**. Do not claim the phase validated until targeted tests, full pytest, Ruff, and compileall outputs are reviewed.
- PR #13 is intended to remain Draft and target the Phase 4.3 branch. No merge is authorized or performed.


---

## 13. Engineering addendum — Phase 4.4-A validation (2026-11-08)

### Phase 4.4-A — ML Feature and Dataset Contracts: VALIDATION PASS

- Branch: `feat/ml-intelligence-contracts-phase-4-4a`.
- Validated HEAD before documentation updates: `fa03643f18683b6317c4806e4d04709733ebb3d7`.
- Manual Colab targeted validation: `tests/unit/test_ml_contracts.py` — **30 tests passed**; Ruff on the ML package and targeted test file reported `All checks passed!`; targeted compileall completed without errors.
- Manual Colab full validation: `python -m pytest -q` reached 100% with no reported failures; `ruff check roboqa tests --output-format=full` reported `All checks passed!`; `python -m compileall -q roboqa tests` completed without errors.
- The captured full-suite output did not expose an unambiguous aggregate test count; no total is inferred.
- Result: **VALIDATION PASS**. This validates contracts and repository compatibility only; it does not validate a trained detector or anomaly-detection performance.
- No training, inference, feature extraction, model dependency, dataset download, or Quality Gate integration was introduced.
- PR #13 remains Draft, open, and unmerged. No changes were made to `main`.


---

## 14. Engineering addendum — Phase 4.5-A Fault-to-QA contracts (2026-11-08)

### Phase 4.5-A — VALIDATION PASS

- Branch: `feat/fault-to-qa-contracts-phase-4-5a`.
- Validated HEAD: `c7e5e3e04e575fab66267835c8a249bbe5273247`.
- Base: `feat/ml-intelligence-contracts-phase-4-4a`, with no divergence behind the base at branch creation.
- PR #14: https://github.com/edu-moraess/roboqa-lab/pull/14 — remains Draft, open, and unmerged.
- Added immutable `EnvelopeIdentity`, `DeliveryDecision`, `TransportReplaySummary`, explicit delivery dispositions, ordering policy declaration, three-clock evidence validation, source/delivery accounting, and SHA-256 provenance contracts.
- Manual Colab targeted tests: **52 passed**.
- Manual Colab full `pytest -q`: reached 100%, exit code 0. The captured output did not expose an aggregate test count; no total is inferred.
- `ruff check roboqa tests`: **All checks passed!**
- `python -m compileall -q roboqa tests`: completed without errors.
- A test-helper defect was corrected: the original helper masked an explicitly supplied arrival timestamp for DROPPED cases. The corrected tests independently verify that DROPPED cannot claim arrival or processing. A subsequent Ruff-only spacing correction was also applied.
- Result: **VALIDATION PASS** for contracts and repository compatibility. This is not evidence of a functioning transport replay/scheduler, Measurement adapter, duplicate suppression, arrival-order Fusion execution, or complete end-to-end integration.
- No training, model inference, UI, CI, merge, or changes to `main` were performed.

### Next engineering checkpoint

Proceed to Phase 4.5-B with explicit replay semantics: define ordering, late packet/watermark policy, duplicate identity and suppression, dropped-packet accounting, three-clock propagation, and a tested adapter boundary before wiring faulted deliveries into Fusion. Preserve the deterministic Quality Gate as the sole decision authority.


---

## 15. Session checkpoint — 2026-11-08 06:34 UTC

### Saved state

- Repository: `edu-moraess/roboqa-lab`.
- Working branch: `feat/fault-to-qa-contracts-phase-4-5a`.
- Last verified GitHub branch HEAD at checkpoint preparation: `744ae3de152d4fd3e07e278e4104f6cc4809fb88`.
- Branch comparison against `feat/ml-intelligence-contracts-phase-4-4a`: **11 commits ahead, 0 behind**.
- PR #14: https://github.com/edu-moraess/roboqa-lab/pull/14 — open, Draft, unmerged; target remains `feat/ml-intelligence-contracts-phase-4-4a`.
- PR remains deliberately unmerged. No changes to `main` were made.
- The manual validation was run in Colab at code checkpoint `c7e5e3e04e575fab66267835c8a249bbe5273247`. Subsequent commits through `744ae3de...` are documentation and PR-record updates; no production-code changes followed the validated code checkpoint.
- Colab's last reported local status contained only untracked generated artifacts (`.coverage`, `__pycache__`, and `roboqa_lab.egg-info`). They were not committed or deleted. On resumption, fast-forward the Colab branch from origin and verify HEAD before running more work.

### Validated scope at pause

Phase 4.5-A is **VALIDATION PASS** for immutable Fault-to-QA integration contracts and repository compatibility:
- 52 targeted unit tests passed.
- Full pytest suite completed at 100%, exit code 0 (aggregate count not inferred).
- Full Ruff check passed.
- Full compileall completed without errors.

This does not establish a working transport replay engine, effective arrival-order scheduling, late-packet/watermark behavior, duplicate suppression, Measurement adapter, or end-to-end fault-to-fusion integration.

### Exact continuation point — Phase 4.5-B

Resume with a design/specification and tests-first checkpoint for a deterministic transport replay boundary. Resolve and document, before production implementation:
1. Ordering semantics for ARRIVAL_ORDER versus ACQUISITION_ORDER and deterministic tie-breaking.
2. Late-packet policy and watermark/finalization behavior.
3. Duplicate identity scope and suppression semantics, including source_id/sequence collisions across sensors.
4. Accounting invariants for accepted, duplicate, late, invalid, and dropped deliveries.
5. Propagation and validation of acquisition_time, arrival_time, and processing_time.
6. A controlled adapter boundary into normalized `Measurement`, without weakening its strict validation contract.
7. Integration tests proving how the chosen replay policy affects Fusion telemetry, Quality Metrics, and the deterministic Quality Gate.

Keep the deterministic Quality Gate as the sole decision authority. Do not begin ML model training, add CI/GitHub Actions, alter `main`, or merge PR #14 without explicit authorization.


---

## 16. Engineering addendum — Phase 4.5-B deterministic transport replay (2026-11-08)

### Implementation checkpoint — manual validation pending

- Branch: `feat/temporal-replay-phase-4-5b`.
- Base: Phase 4.5-A checkpoint `d4010790a4654d304a443aa7f5603af92b251926`.
- PR #15: https://github.com/edu-moraess/roboqa-lab/pull/15 — open, Draft, unmerged; targets `feat/fault-to-qa-contracts-phase-4-5a`, not `main`.
- Added `roboqa.integration.replay_transport` and immutable `TransportReplayResult`.
- Implemented deterministic ARRIVAL_ORDER and ACQUISITION_ORDER batch ordering, canonical tie-breaking, source-identity validation, first-delivery-wins duplicate suppression, acquisition-time watermark lateness for ARRIVAL_ORDER, invalid three-clock/payload classification, dropped-source accounting, and SHA-256 provenance.
- Exported replay API from `roboqa.integration`.
- Added focused unit/adversarial tests in `tests/unit/test_transport_replay.py` and normative specification in `docs/phase-4-5b-deterministic-transport-replay.md`.
- README roadmap now distinguishes implemented-but-unvalidated Phase 4.5-B from the planned adapter and end-to-end phases.
- Current PR head at documentation checkpoint: `ebd21fad94b3e0c8b9f70519ff1e9a4ef5ef7ba4`.\n- Compare against Phase 4.5-A base: 7 commits ahead, 0 behind; six changed files.\n- **Validation status: PENDING MANUAL COLAB.** No pytest, Ruff, compileall, coverage, or runtime PASS is claimed for this branch yet.

### Normative semantics

- Identity: normalized `(sensor_id, source_id, sequence_number)`; duplicate source identities are rejected.
- ARRIVAL_ORDER sorts by arrival time and uses `watermark = max_acquisition_seen - allowed_lateness_s`; first-seen event time strictly below the watermark is LATE. Equality is accepted.
- ACQUISITION_ORDER is a deterministic offline event-time sort and does not classify late packets.
- Deterministic ties use normalized sensor, source ID, sequence, and canonical envelope fingerprint.
- First delivery wins; subsequent same-identity copies are DUPLICATE, even if the first copy was INVALID or LATE.
- Invalid non-finite clocks/payload, arrival before acquisition, or processing before arrival are classified INVALID without mutating the raw envelope.
- Source packets absent from delivered batch are DROPPED with arrival/processing set to None.
- Accounting: delivered = accepted + duplicate + late + invalid; source = accepted + late + invalid + dropped.
- This is offline batch replay only. It does not create a live streaming watermark or invoke Fusion.

### Important integration boundary

The existing `run_filter` still sorts normalized `Measurement.t`. The new ARRIVAL_ORDER replay output is not proof that Fusion processes in arrival order. The adapter and Fusion scheduling boundary remain a separate Phase 4.5-C/D task. The strict normalized `Measurement` schema and deterministic Quality Gate remain unchanged.

### Manual Colab gate

Run targeted replay tests, targeted Ruff and compileall, then full pytest, full Ruff, and full compileall. Keep PR #15 Draft until the actual outputs are reviewed. Do not infer test totals or coverage percentages.


---

## 17. Engineering addendum — Phase 4.5-B validation (2026-11-08)

### Phase 4.5-B — Deterministic Transport Replay: VALIDATION PASS

- Branch: `feat/temporal-replay-phase-4-5b`.
- Code checkpoint used for manual validation: `9e239725775657ca0ac5bb65e3d6dd4e72fd2d5f`.
- PR #15: https://github.com/edu-moraess/roboqa-lab/pull/15 — remains Draft, open, and unmerged; base remains `feat/fault-to-qa-contracts-phase-4-5a), not `main`.
- Targeted replay tests: **19 passed**.
- The first targeted run exposed three fixture errors: two packets had arrival timestamps earlier than acquisition timestamps. Production validation was retained; the three test fixtures were corrected to preserve causal acquisition/arrival order while maintaining out-of-order event time. The corrected replay suite passed all 19 tests.
- Targeted Ruff and targeted compileall: **PASS**.
- Full `pytest -q`: exit code 0, no failures reported. The captured output did not expose an unambiguous aggregate count; no total is inferred.
- Full `ruff check roboqa tests`: **PASS**.
- Full `python -m compileall -q roboqa tests`: **PASS**.
- README and phase specification updated to record the evidence and preserve the explicit scope boundary.
- Result: **VALIDATION PASS** for the deterministic offline transport replay and repository compatibility.
- This does not validate a production RawMeasurementEnvelope-to-Measurement adapter, arrival-order execution inside Fusion, or end-to-end Fusion Telemetry → Quality Metrics → Quality Gate behavior.
- No model training, inference, dataset downloads, CI/GitHub Actions, merge, or changes to `main` were performed.

### Next engineering step

Proceed to Phase 4.5-C/D with a separately specified adapter and Fusion scheduling boundary. Define how accepted envelopes map to normalized `Measurement`, whether event-time or arrival-time controls execution, and how late/duplicate/dropped decisions are carried into telemetry and QA. Add tests that prove the effect on Fusion telemetry, metrics, and deterministic Quality Gate before extending the product UI. Keep the Quality Gate as the sole decision authority.

---

## 18. Engineering addendum — Phase 4.5-C Measurement adapter (2026-11-08)

### Implementation checkpoint — manual validation pending

- Branch: `feat/measurement-adapter-phase-4-5c`.
- Base: validated Phase 4.5-B checkpoint `7fd0fd808b3732392775d0dc28e52127ec9a8924`.
- PR #16: https://github.com/edu-moraess/roboqa-lab/pull/16 — open, Draft, unmerged; targets `feat/temporal-replay-phase-4-5b`, not `main`.
- Added `roboqa.integration.measurement_adapter` with `AdaptedMeasurement`, `MeasurementAdapterError`, and `adapt_accepted_envelopes`.
- Only replay-accepted envelopes are converted. Duplicate, late, invalid, and dropped decisions remain in the replay summary.
- Payload maps to Measurement z; acquisition_time maps to Measurement.t. Arrival and processing clocks and source identity are retained alongside each normalized measurement.
- Covariance R must be supplied explicitly by the caller per normalized sensor ID. No noise model is inferred or defaulted.
- Existing Measurement schema and Fusion sensor registry enforce finite values, dimensions, covariance validity, and supported sensor types.
- Added targeted unit tests and adapter specification; README roadmap updated.
- Important limitation: the adapter preserves replay order but does not change Fusion execution order. Existing `run_filter` sorts by `(Measurement.t, sensor)`; arrival-order Fusion scheduling is a separate Phase 4.5-D design decision.
- **Validation status: PENDING MANUAL COLAB.** No pytest, Ruff, compileall, coverage, or runtime PASS is claimed yet.
- No changes to `main`, no merge, no CI/GitHub Actions, no model training/inference, and no Streamlit changes.

### Manual Colab gate

Sync branch `feat/measurement-adapter-phase-4-5c`, then run the targeted adapter tests, targeted Ruff/compileall, full pytest, full Ruff, and full compileall. Keep PR #16 Draft until outputs are reviewed.


---

## 19. Engineering addendum — Phase 4.5-C Measurement adapter validation (2026-11-08)

### Phase 4.5-C — Strict Measurement Adapter: VALIDATION PASS

- Branch: `feat/measurement-adapter-phase-4-5c`.
- Validated code checkpoint: `77f0e6e267f08f86ddc7f1468da0b9d474d348ef`; subsequent changes in this checkpoint are limited to a test assertion and import ordering.
- Base: Phase 4.5-B validated checkpoint `7fd0fd808b3732392775d0dc28e52127ec9a8924`.
- PR #16: https://github.com/edu-moraess/roboqa-lab/pull/16 — remains Draft, open, and unmerged; targets `feat/temporal-replay-phase-4-5b`, not `main`.
- Manual Colab targeted adapter tests: **10 passed**.
- Manual Colab full `python -m pytest -q`: exit code **0**, no failures reported. The captured output did not expose an unambiguous aggregate test count; no total is inferred.
- Full `ruff check roboqa tests`: **All checks passed!**
- Full `python -m compileall -q roboqa tests`: exit code **0**.
- The first validation exposed a test assertion regex case mismatch (`dim` versus the Portuguese message beginning with `Dimensão`), not an adapter behavior defect. The assertion was made case-insensitive. Ruff also identified import ordering in `roboqa/integration/__init__.py`; imports were reordered.
- The two corrections changed only `tests/unit/test_measurement_adapter.py` and `roboqa/integration/__init__.py`; the production adapter implementation was not changed. `git diff --check` passed.
- Result: **VALIDATION PASS** for the strict Measurement adapter contracts and repository compatibility.
- Scope limitation remains: the adapter maps acquisition time to `Measurement.t`, preserves arrival/processing clocks as metadata, and does not change `run_filter` sorting. This is not evidence of arrival-order Fusion execution or end-to-end Fusion Telemetry → Quality Metrics → Quality Gate integration.
- No training, inference, dataset downloads, CI/GitHub Actions, merge, or changes to `main` were performed.

### Next engineering step

Phase 4.5-D should specify and test the Fusion scheduling boundary explicitly: whether accepted measurements execute in acquisition/event-time order or arrival order, how late/duplicate/dropped decisions influence execution and telemetry, and how replay evidence propagates into Quality Metrics and the deterministic Quality Gate. Preserve the Quality Gate as the sole decision authority.


---

## 20. Engineering addendum — Phase 4.5-D1 deterministic Fusion scheduling validation (2026-11-08)

### Phase 4.5-D1 — VALIDATION PASS

- Branch: `feat/fusion-scheduling-phase-4-5d1`.
- Validated code checkpoint: `43d0bd73f43e47b6dc0d3e0cc0f545e0127ba929`.
- Base: Phase 4.5-C checkpoint `b5ba536425c3b66070f0340802b99745f3cdb053`.
- PR #17: https://github.com/edu-moraess/roboqa-lab/pull/17 — open, Draft, and unmerged; targets `feat/measurement-adapter-phase-4-5c`, not `main`.
- Manual Colab targeted tests: **15 passed**.
- Full `python -m pytest -q`: reached 100%, exit code **0**, no failures reported. The captured output did not expose an unambiguous aggregate test count; no total is inferred.
- Full `ruff check roboqa tests`: **All checks passed!**
- Full `python -m compileall -q roboqa tests`: exit code **0**.
- Result: **FULL VALIDATION PASS** for the offline deterministic Fusion scheduling boundary and repository compatibility.

### Implemented behavior

- `FusionSchedule` is immutable and retains the complete `TransportReplaySummary`.
- `build_fusion_schedule` requires an exact one-to-one match between replay-accepted envelope identities and adapter records.
- Sensor identity, acquisition time, arrival time, processing time, and normalized payload are checked before scheduling.
- Event-time order is deterministic: acquisition timestamp, normalized sensor ID, source ID, sequence number.
- Schedule SHA-256 includes replay provenance, three clocks, normalized payload, and explicit caller-supplied covariance R.
- Replay dispositions other than ACCEPTED remain in the attached replay summary and never enter the Fusion schedule.
- Empty accepted batches produce empty schedules without fabricating measurements.

### Explicit limitations

- D1 is an offline batch scheduling boundary. It does not implement a live queue, wall-clock buffering, watermark finalization, fixed-lag rewind/reprocessing, or arrival-order EKF execution.
- EKF and `run_filter` were not modified.
- Quality Metrics and Quality Gate were not modified in D1. Their integration is deferred to D3.
- No model training/inference, dataset downloads, Streamlit, CI/GitHub Actions, merge, or changes to `main` were performed.

### Next engineering checkpoints

- **D2 — Fusion execution record:** execute the immutable schedule through the existing Fusion runner and associate source identities with telemetry or explicit execution outcomes.
- **D3 — QA evidence propagation:** propagate replay and execution evidence into Quality Metrics and deterministic Quality Gate, preserving the Gate as the sole decision authority.


---

## 21. Engineering addendum — Phase 4.5-D2 identity-preserving Fusion execution validation (2026-11-08)

### Phase 4.5-D2 — FULL VALIDATION PASS

- Branch: `feat/fusion-execution-phase-4-5d2`.
- Validated production-code checkpoint: `43de077cf01b16c3114d5f7dfeeeaf53a6d2057d`.
- Base: Phase 4.5-D1 validated checkpoint `98aa80b857bb8fc9ee6061495ffad22c2929f4da`.
- PR #18: https://github.com/edu-moraess/roboqa-lab/pull/18 — remains Draft, open, and unmerged; targets `feat/fusion-scheduling-phase-4-5d1`, not `main`.
- Manual Colab targeted tests across D2 execution, D1 scheduling, and Measurement adapter: **32 passed**.
- Full `python -m pytest -q`: reached 100%, exit code **0**, no failures reported. The captured output did not expose an unambiguous aggregate test count; no total is inferred.
- Full `ruff check roboqa tests`: **All checks passed!**
- Full `python -m compileall -q roboqa tests`: exit code **0**.
- The initial targeted Ruff run identified only import ordering in `roboqa/integration/__init__.py`; it was corrected in commit `43de077cf01b16c3114d5f7dfeeeaf53a6d2057d`. Targeted Ruff and compileall then passed, followed by all full-repository checks.
- Result: **FULL VALIDATION PASS** for identity-preserving Fusion schedule execution and repository compatibility.

### Implemented behavior

- Executes the immutable D1 acquisition-time schedule through the existing monotonic Fusion runner.
- Produces exactly one terminal outcome per replay-accepted identity: `UPDATED`, `FAILED`, or `NOT_EXECUTED`.
- Associates successful `InnovationTelemetry` with the exact sensor/source/sequence identity and acquisition timestamp.
- Preserves the complete transport replay summary, runner diagnostics, configuration, state/covariance/time traces, and deterministic execution SHA-256.
- Attributes the first numerical update failure to the failing scheduled item and marks remaining entries `NOT_EXECUTED`. Invalid run configuration does not incorrectly blame an individual measurement.
- Retains the native `FilterResult` for downstream metrics.

### Explicit limitations

- D2 does not modify EKF mathematics, `run_filter` ordering policy, Quality Metrics, or Quality Gate.
- It does not implement online arrival-order execution, rewind, fixed-lag smoothing, or a live scheduler.
- Duplicate, late, invalid, and dropped packet dispositions remain in the attached replay summary; execution outcomes apply only to replay-accepted schedule entries.
- No ML training/inference, dataset downloads, Streamlit, CI/GitHub Actions, merge, or changes to `main` were performed.

### Next engineering checkpoint — Phase 4.5-D3

Propagate replay and execution evidence into Quality Metrics and the deterministic Quality Gate. Define how non-accepted transport dispositions, failed updates, and not-executed measurements affect availability, rejection, latency, and evidence completeness. Missing evidence must remain explicit; the Quality Gate remains the sole decision authority.


---

## 22. Engineering addendum — Phase 4.5-D3 integrated QA evidence propagation (2026-11-08)

### Implementation checkpoint — manual validation pending

- Branch: `feat/qa-evidence-propagation-phase-4-5d3`.
- Base: Phase 4.5-D2 validated documentation checkpoint `5ddd3a50d4e2de4666aaf65fb70f5c4bb6b5a54b`.
- Implementation introduces `roboqa/qa/integrated.py` and exports `compute_integrated_quality_metrics`, `evaluate_integrated_quality`, and `IntegratedQualityError` from `roboqa.qa`.
- Added focused tests in `tests/unit/test_integrated_quality.py` and specification `docs/phase-4-5d3-integrated-qa-evidence.md`.
- README now records D3 as implemented with manual validation pending.
- **Validation status: PENDING MANUAL COLAB.** No pytest, Ruff, compileall, coverage, or runtime PASS is claimed for this branch.

### Implemented behavior

- Preserves Phase 4.2 `compute_quality_metrics` unchanged and adds an explicit integrated API.
- Verifies replay summary equality with the D2 execution record, matching run IDs, FilterResult status and diagnostics, and exact ordered telemetry agreement before combining evidence.
- Adds run- and sensor-scoped packet accounting for unique source packets, delivery attempts, accepted, duplicate, late, invalid, and dropped dispositions.
- Adds Fusion updated, failed, and not-executed counts linked to replay-accepted identities.
- Adds transport acceptance, Fusion update, and execution-success ratios. Ratios with zero denominators are MISSING; explicit count metrics may legitimately be zero.
- Adds `evaluate_integrated_quality`, delegating all decisions to the existing versioned deterministic Quality Gate. Source hash comes from replay; output hash comes from the execution record; configuration hash and UTC generation timestamp remain caller-supplied.

### Scope and safeguards

- Existing Phase 4.2 metric semantics and Phase 4.3 Gate policy semantics are not changed.
- Existing `sensor_availability` retains its explicit expected-measurement denominator and is not redefined as physical sensor uptime.
- No universal thresholds, implicit imputation, ML authority, Streamlit, CI/GitHub Actions, training, merge, or changes to `main`.
- Manual Colab must run targeted D3 tests, existing QA metrics/Gate regression tests, integration replay/scheduler/adapter/execution tests, full pytest, full Ruff, and full compileall before validation status changes.
