"""Identity-preserving execution of deterministic Fusion schedules (Phase 4.5-D2)."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from roboqa.fusion.result import FilterResult, FilterStatus, InnovationTelemetry
from roboqa.fusion.runner import run_filter
from roboqa.integration.contracts import EnvelopeIdentity, TransportReplaySummary
from roboqa.integration.scheduler import FusionSchedule


class FusionExecutionError(ValueError):
    """Raised when a Fusion schedule cannot be executed or traced safely."""


class ExecutionDisposition(str, Enum):
    UPDATED = "updated"
    FAILED = "failed"
    NOT_EXECUTED = "not_executed"


@dataclass(frozen=True)
class MeasurementExecutionOutcome:
    """Terminal execution outcome for exactly one replay-accepted identity."""

    identity: EnvelopeIdentity
    acquisition_time: float
    disposition: ExecutionDisposition
    telemetry: InnovationTelemetry | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EnvelopeIdentity):
            raise FusionExecutionError("identity must be an EnvelopeIdentity")
        if not isinstance(self.disposition, ExecutionDisposition):
            raise FusionExecutionError("disposition must be an ExecutionDisposition")
        timestamp = float(self.acquisition_time)
        if not math.isfinite(timestamp):
            raise FusionExecutionError("acquisition_time must be finite")
        object.__setattr__(self, "acquisition_time", timestamp)
        if self.disposition is ExecutionDisposition.UPDATED:
            if not isinstance(self.telemetry, InnovationTelemetry):
                raise FusionExecutionError("UPDATED outcome requires InnovationTelemetry")
            if self.reason is not None:
                raise FusionExecutionError("UPDATED outcome cannot carry a failure reason")
            if self.telemetry.sensor_id != self.identity.sensor_id:
                raise FusionExecutionError("telemetry sensor does not match outcome identity")
            if not math.isclose(
                self.telemetry.timestamp, timestamp, rel_tol=0.0, abs_tol=1e-12
            ):
                raise FusionExecutionError("telemetry timestamp does not match acquisition_time")
        else:
            if self.telemetry is not None:
                raise FusionExecutionError("non-UPDATED outcome cannot carry telemetry")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise FusionExecutionError("non-UPDATED outcome requires an explicit reason")
            object.__setattr__(self, "reason", self.reason.strip())


def _telemetry_document(item: InnovationTelemetry) -> dict[str, Any]:
    return {
        "timestamp": item.timestamp,
        "sensor_id": item.sensor_id,
        "measurement_dof": item.measurement_dof,
        "innovation": item.innovation.tolist(),
        "innovation_covariance": item.innovation_covariance.tolist(),
        "nis": item.nis,
    }


def _record_document(
    *,
    run_id: str,
    schedule_hash: str,
    filter_status: FilterStatus,
    replay_summary: TransportReplaySummary,
    n_predict: int,
    n_update: int,
    n_rejected: int,
    outcomes: tuple[MeasurementExecutionOutcome, ...],
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    state_trace: tuple[tuple[float, ...], ...],
    covariance_trace: tuple[tuple[tuple[float, ...], ...], ...],
    time_trace: tuple[float, ...],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "schedule_hash": schedule_hash,
        "filter_status": filter_status.value,
        "replay_summary": {
            "ordering_policy": replay_summary.ordering_policy.value,
            "source_count": replay_summary.source_count,
            "delivered_count": replay_summary.delivered_count,
            "accepted_count": replay_summary.accepted_count,
            "duplicate_count": replay_summary.duplicate_count,
            "late_count": replay_summary.late_count,
            "invalid_count": replay_summary.invalid_count,
            "dropped_count": replay_summary.dropped_count,
            "source_hash": replay_summary.source_hash,
            "delivered_hash": replay_summary.delivered_hash,
            "configuration_hash": replay_summary.configuration_hash,
            "decisions": [
                {
                    "sensor_id": item.identity.sensor_id,
                    "source_id": item.identity.source_id,
                    "sequence_number": item.identity.sequence_number,
                    "disposition": item.disposition.value,
                    "acquisition_time": item.acquisition_time,
                    "arrival_time": item.arrival_time,
                    "processing_time": item.processing_time,
                    "reason": item.reason,
                }
                for item in replay_summary.decisions
            ],
        },
        "n_predict": n_predict,
        "n_update": n_update,
        "n_rejected": n_rejected,
        "outcomes": [
            {
                "sensor_id": item.identity.sensor_id,
                "source_id": item.identity.source_id,
                "sequence_number": item.identity.sequence_number,
                "acquisition_time": item.acquisition_time,
                "disposition": item.disposition.value,
                "telemetry": (
                    None if item.telemetry is None else _telemetry_document(item.telemetry)
                ),
                "reason": item.reason,
            }
            for item in outcomes
        ],
        "errors": list(errors),
        "warnings": list(warnings),
        "time_trace": list(time_trace),
        "state_trace": [list(row) for row in state_trace],
        "covariance_trace": [
            [list(row) for row in matrix] for matrix in covariance_trace
        ],
        "configuration": configuration,
    }


@dataclass(frozen=True)
class FusionExecutionRecord:
    """Immutable, hash-addressed execution evidence for one Fusion schedule."""

    run_id: str
    schedule_hash: str
    filter_status: FilterStatus
    replay_summary: TransportReplaySummary
    n_predict: int
    n_update: int
    n_rejected: int
    outcomes: tuple[MeasurementExecutionOutcome, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    time_trace: tuple[float, ...]
    state_trace: tuple[tuple[float, ...], ...]
    covariance_trace: tuple[tuple[tuple[float, ...], ...], ...]
    configuration: tuple[tuple[str, Any], ...]
    execution_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise FusionExecutionError("run_id must be non-empty")
        if not isinstance(self.schedule_hash, str) or len(self.schedule_hash) != 64:
            raise FusionExecutionError("schedule_hash must be a SHA-256 digest")
        if not isinstance(self.filter_status, FilterStatus):
            raise FusionExecutionError("filter_status must be a FilterStatus")
        if not isinstance(self.replay_summary, TransportReplaySummary):
            raise FusionExecutionError("replay_summary must be a TransportReplaySummary")
        if self.replay_summary.run_id != self.run_id:
            raise FusionExecutionError("replay summary run_id does not match execution run_id")
        for name in ("n_predict", "n_update", "n_rejected"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FusionExecutionError(f"{name} must be a non-negative integer")
        outcomes = tuple(self.outcomes)
        if any(not isinstance(item, MeasurementExecutionOutcome) for item in outcomes):
            raise FusionExecutionError("outcomes must contain MeasurementExecutionOutcome")
        identities = [item.identity.key for item in outcomes]
        if len(set(identities)) != len(identities):
            raise FusionExecutionError("execution outcome identities must be unique")
        accepted_keys = {
            item.identity.key
            for item in self.replay_summary.decisions
            if item.disposition.value == "accepted"
        }
        if set(identities) != accepted_keys:
            raise FusionExecutionError(
                "execution outcomes must match replay-accepted identities exactly"
            )
        dispositions = [item.disposition for item in outcomes]
        if dispositions != sorted(
            dispositions,
            key=lambda item: {
                ExecutionDisposition.UPDATED: 0,
                ExecutionDisposition.FAILED: 1,
                ExecutionDisposition.NOT_EXECUTED: 2,
            }[item],
        ):
            raise FusionExecutionError(
                "execution outcomes must preserve the runner's updated/failure/pending prefix"
            )
        if sum(item.disposition is ExecutionDisposition.FAILED for item in outcomes) > 1:
            raise FusionExecutionError("a stopped runner can have at most one failed measurement")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "time_trace", tuple(float(x) for x in self.time_trace))
        object.__setattr__(
            self, "state_trace",
            tuple(tuple(float(x) for x in row) for row in self.state_trace),
        )
        object.__setattr__(
            self, "covariance_trace",
            tuple(
                tuple(tuple(float(x) for x in row) for row in matrix)
                for matrix in self.covariance_trace
            ),
        )
        object.__setattr__(self, "configuration", tuple(self.configuration))
        if not (
            len(self.time_trace) == len(self.state_trace) == len(self.covariance_trace)
        ):
            raise FusionExecutionError("state, covariance, and time traces must align")
        if not all(math.isfinite(x) for x in self.time_trace):
            raise FusionExecutionError("time_trace must be finite")
        if not all(
            math.isfinite(x)
            for row in self.state_trace for x in row
        ):
            raise FusionExecutionError("state_trace must be finite")
        if not all(
            math.isfinite(x)
            for matrix in self.covariance_trace for row in matrix for x in row
        ):
            raise FusionExecutionError("covariance_trace must be finite")
        expected = sum(item.disposition is ExecutionDisposition.UPDATED for item in outcomes)
        failed = sum(item.disposition is ExecutionDisposition.FAILED for item in outcomes)
        pending = sum(item.disposition is ExecutionDisposition.NOT_EXECUTED for item in outcomes)
        if expected + failed + pending != len(outcomes):
            raise FusionExecutionError("execution outcome accounting is inconsistent")
        if len(outcomes) != self.replay_summary.accepted_count:
            raise FusionExecutionError("outcomes must account for every replay-accepted packet")
        if self.n_update != expected:
            raise FusionExecutionError("n_update must match UPDATED outcome count")
        if self.n_rejected:
            raise FusionExecutionError("scheduled execution cannot contain runner rejections")
        digest = self.compute_hash()
        if self.execution_hash != digest:
            raise FusionExecutionError("execution_hash does not match execution evidence")

    @property
    def updated_count(self) -> int:
        return sum(item.disposition is ExecutionDisposition.UPDATED for item in self.outcomes)

    @property
    def failed_count(self) -> int:
        return sum(item.disposition is ExecutionDisposition.FAILED for item in self.outcomes)

    @property
    def not_executed_count(self) -> int:
        return sum(item.disposition is ExecutionDisposition.NOT_EXECUTED for item in self.outcomes)

    def compute_hash(self) -> str:
        document = _record_document(
            run_id=self.run_id,
            schedule_hash=self.schedule_hash,
            filter_status=self.filter_status,
            replay_summary=self.replay_summary,
            n_predict=self.n_predict,
            n_update=self.n_update,
            n_rejected=self.n_rejected,
            outcomes=self.outcomes,
            errors=self.errors,
            warnings=self.warnings,
            state_trace=self.state_trace,
            covariance_trace=self.covariance_trace,
            time_trace=self.time_trace,
            configuration=dict(self.configuration),
        )
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FusionExecutionResult:
    """Execution record plus the native FilterResult for downstream analysis."""

    record: FusionExecutionRecord
    filter_result: FilterResult

    def __post_init__(self) -> None:
        if not isinstance(self.record, FusionExecutionRecord):
            raise FusionExecutionError("record must be a FusionExecutionRecord")
        if not isinstance(self.filter_result, FilterResult):
            raise FusionExecutionError("filter_result must be a FilterResult")
        if len(self.filter_result.telemetry) != self.record.updated_count:
            raise FusionExecutionError("FilterResult telemetry count differs from execution record")


def execute_fusion_schedule(
    schedule: FusionSchedule,
    *,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    sigma_a: float = 0.5,
    sigma_alpha: float = 0.1,
    t0: float | None = None,
) -> FusionExecutionResult:
    """Execute a validated schedule and bind each successful update to its identity.

    The current runner processes the schedule in stable (time, sensor) order.
    D1 has already imposed that order, including deterministic identity tie-breaks.
    The runner stops at the first numerical/schema update failure; successful
    telemetry therefore corresponds to a prefix of the schedule. A non-monotonic
    rejection is treated as an integration error rather than guessed attribution.
    """
    if not isinstance(schedule, FusionSchedule):
        raise FusionExecutionError("schedule must be a FusionSchedule")

    result = run_filter(
        schedule.measurements,
        x0=x0,
        P0=P0,
        sigma_a=sigma_a,
        sigma_alpha=sigma_alpha,
        t0=t0,
        provenance={
            "run_id": schedule.run_id,
            "schedule_hash": schedule.schedule_hash,
            "execution_policy": schedule.policy.value,
        },
    )
    telemetry = tuple(result.telemetry)
    if len(telemetry) != result.diagnostics.n_update:
        raise FusionExecutionError("runner update count and telemetry count diverged")
    if len(telemetry) > len(schedule.entries):
        raise FusionExecutionError("runner emitted more telemetry than scheduled measurements")
    if result.diagnostics.n_rejected:
        raise FusionExecutionError(
            "runner rejected scheduled measurements; exact identity attribution is unavailable"
        )

    outcomes: list[MeasurementExecutionOutcome] = []
    for index, entry in enumerate(schedule.entries):
        if index < len(telemetry):
            outcomes.append(MeasurementExecutionOutcome(
                identity=entry.identity,
                acquisition_time=entry.measurement.t,
                disposition=ExecutionDisposition.UPDATED,
                telemetry=telemetry[index],
            ))
            continue

        if (
            index == len(telemetry)
            and result.status in (FilterStatus.NUMERICAL_FAILURE, FilterStatus.INVALID_INPUT)
            and result.diagnostics.errors
            and (
                result.status is FilterStatus.NUMERICAL_FAILURE
                or len(telemetry) > 0
            )
        ):
            disposition = ExecutionDisposition.FAILED
            reason = result.diagnostics.errors[0]
        else:
            disposition = ExecutionDisposition.NOT_EXECUTED
            reason = (
                "Fusion runner did not reach this scheduled measurement; "
                "see run-level errors and status."
            )
        outcomes.append(MeasurementExecutionOutcome(
            identity=entry.identity,
            acquisition_time=entry.measurement.t,
            disposition=disposition,
            reason=reason,
        ))

    errors = tuple(result.diagnostics.errors)
    warnings = tuple(result.diagnostics.warnings)
    state_trace = tuple(tuple(float(x) for x in row) for row in result.x)
    covariance_trace = tuple(
        tuple(tuple(float(x) for x in row) for row in matrix) for matrix in result.P
    )
    time_trace = tuple(float(x) for x in result.t)
    configuration = tuple(sorted({
        **result.config,
        "schedule_policy": schedule.policy.value,
    }.items()))
    # Compute the canonical digest before constructing the self-validating record.
    document = _record_document(
        run_id=schedule.run_id,
        schedule_hash=schedule.schedule_hash,
        filter_status=result.status,
        replay_summary=schedule.replay_summary,
        n_predict=result.diagnostics.n_predict,
        n_update=result.diagnostics.n_update,
        n_rejected=result.diagnostics.n_rejected,
        outcomes=tuple(outcomes),
        errors=errors,
        warnings=warnings,
        state_trace=state_trace,
        covariance_trace=covariance_trace,
        time_trace=time_trace,
        configuration=dict(configuration),
    )
    digest = hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    record = FusionExecutionRecord(
        run_id=schedule.run_id,
        schedule_hash=schedule.schedule_hash,
        filter_status=result.status,
        replay_summary=schedule.replay_summary,
        n_predict=result.diagnostics.n_predict,
        n_update=result.diagnostics.n_update,
        n_rejected=result.diagnostics.n_rejected,
        outcomes=tuple(outcomes),
        errors=errors,
        warnings=warnings,
        time_trace=time_trace,
        state_trace=state_trace,
        covariance_trace=covariance_trace,
        configuration=configuration,
        execution_hash=digest,
    )
    return FusionExecutionResult(record=record, filter_result=result)
