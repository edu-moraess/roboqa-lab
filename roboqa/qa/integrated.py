"""Integrated transport and Fusion execution evidence for Quality Assurance (Phase 4.5-D3).

The existing Phase 4.2 metrics remain unchanged. This module adds explicit
replay-disposition and identity-preserving execution metrics, then offers a
thin entry point into the existing deterministic Quality Gate.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isclose

import numpy as np

from roboqa.fusion.result import FilterResult
from roboqa.integration.contracts import (
    DeliveryDisposition,
    TransportReplaySummary,
)
from roboqa.integration.execution import (
    ExecutionDisposition,
    FusionExecutionRecord,
)
from roboqa.qa.contracts import (
    MetricAvailability,
    MetricObservation,
    MetricScope,
    QualityReport,
)
from roboqa.qa.gate import (
    QualityGatePolicy,
    evaluate_quality_gate,
)
from roboqa.qa.metrics import (
    QualityMetricError,
    compute_quality_metrics,
)


class IntegratedQualityError(ValueError):
    """Raised when replay, execution, and Fusion evidence do not agree."""


def _metric(
    name: str,
    value: float | None,
    *,
    unit: str,
    scope: MetricScope,
    sample_count: int,
    sensor_id: str | None = None,
    availability: MetricAvailability = MetricAvailability.AVAILABLE,
) -> MetricObservation:
    return MetricObservation(
        name=name,
        value=value,
        unit=unit,
        scope=scope,
        sample_count=sample_count,
        sensor_id=sensor_id,
        availability=availability,
    )


def _same_telemetry(left, right) -> bool:
    return (
        left.sensor_id == right.sensor_id
        and isclose(left.timestamp, right.timestamp, rel_tol=0.0, abs_tol=1e-12)
        and left.measurement_dof == right.measurement_dof
        and isclose(left.nis, right.nis, rel_tol=0.0, abs_tol=1e-12)
        and np.array_equal(left.innovation, right.innovation)
        and np.array_equal(left.innovation_covariance, right.innovation_covariance)
    )


def _validate_evidence(
    result: FilterResult,
    replay_summary: TransportReplaySummary,
    execution_record: FusionExecutionRecord,
) -> None:
    if not isinstance(result, FilterResult):
        raise IntegratedQualityError("result must be a FilterResult")
    if not isinstance(replay_summary, TransportReplaySummary):
        raise IntegratedQualityError("replay_summary must be a TransportReplaySummary")
    if not isinstance(execution_record, FusionExecutionRecord):
        raise IntegratedQualityError("execution_record must be a FusionExecutionRecord")
    if execution_record.replay_summary != replay_summary:
        raise IntegratedQualityError(
            "execution record replay summary must exactly match supplied replay evidence"
        )
    if execution_record.run_id != replay_summary.run_id:
        raise IntegratedQualityError("replay and execution run IDs do not match")
    if result.diagnostics.n_update != execution_record.updated_count:
        raise IntegratedQualityError("FilterResult update count differs from execution outcomes")
    if result.diagnostics.n_rejected != execution_record.n_rejected:
        raise IntegratedQualityError("FilterResult rejection count differs from execution record")

    expected_telemetry = tuple(
        outcome.telemetry
        for outcome in execution_record.outcomes
        if outcome.disposition is ExecutionDisposition.UPDATED
    )
    actual_telemetry = tuple(result.telemetry)
    if len(expected_telemetry) != len(actual_telemetry) or any(
        not _same_telemetry(expected, actual)
        for expected, actual in zip(expected_telemetry, actual_telemetry, strict=True)
    ):
        raise IntegratedQualityError(
            "FilterResult telemetry does not match identity-preserving execution evidence"
        )


def _ratio(name: str, numerator: int, denominator: int, *, scope: MetricScope,
           sensor_id: str | None = None) -> MetricObservation:
    if denominator == 0:
        return _metric(
            name, None, unit="ratio", scope=scope, sample_count=0,
            sensor_id=sensor_id, availability=MetricAvailability.MISSING,
        )
    return _metric(
        name, numerator / denominator, unit="ratio", scope=scope,
        sample_count=denominator, sensor_id=sensor_id,
    )


def _append_count_metrics(
    output: list[MetricObservation],
    counts: Mapping[str, int],
    *,
    scope: MetricScope,
    sensor_id: str | None,
    source_count: int,
    delivered_count: int,
    accepted_count: int,
    updated_count: int,
    failed_count: int,
    not_executed_count: int,
) -> None:
    values = (
        ("transport_source_count", source_count, source_count),
        ("transport_delivered_count", delivered_count, delivered_count),
        ("transport_accepted_count", accepted_count, source_count),
        ("transport_duplicate_count", counts["duplicate"], delivered_count),
        ("transport_late_count", counts["late"], delivered_count),
        ("transport_invalid_count", counts["invalid"], delivered_count),
        ("transport_dropped_count", counts["dropped"], source_count),
        ("fusion_updated_count", updated_count, accepted_count),
        ("fusion_failed_count", failed_count, accepted_count),
        ("fusion_not_executed_count", not_executed_count, accepted_count),
    )
    for name, value, samples in values:
        output.append(_metric(
            name, float(value), unit="packets", scope=scope,
            sample_count=samples, sensor_id=sensor_id,
        ))

    output.extend((
        _ratio(
            "transport_acceptance_ratio", accepted_count, source_count,
            scope=scope, sensor_id=sensor_id,
        ),
        _ratio(
            "fusion_update_ratio", updated_count, source_count,
            scope=scope, sensor_id=sensor_id,
        ),
        _ratio(
            "fusion_execution_success_ratio", updated_count, accepted_count,
            scope=scope, sensor_id=sensor_id,
        ),
    ))


def compute_integrated_quality_metrics(
    result: FilterResult,
    *,
    replay_summary: TransportReplaySummary,
    execution_record: FusionExecutionRecord,
    ground_truth=None,
    envelopes: Sequence = (),
    expected_measurements: Mapping[str, int] | None = None,
    alignment_tolerance_s: float = 0.05,
) -> tuple[MetricObservation, ...]:
    """Combine existing estimator metrics with transport and execution evidence.

    Packet counts distinguish unique source packets from duplicate delivery
    attempts. Ratios with a zero denominator are MISSING, never fabricated as
    zero. Sensor-scoped metrics preserve the original normalized sensor IDs.
    """
    _validate_evidence(result, replay_summary, execution_record)
    try:
        base = compute_quality_metrics(
            result,
            ground_truth=ground_truth,
            envelopes=envelopes,
            expected_measurements=expected_measurements,
            alignment_tolerance_s=alignment_tolerance_s,
        )
    except QualityMetricError as exc:
        raise IntegratedQualityError(f"base quality metrics are invalid: {exc}") from exc

    replay_counts = {
        disposition: 0 for disposition in DeliveryDisposition
    }
    source_identities: set[tuple[str, str | None, int]] = set()
    sensor_source_counts: dict[str, set[tuple[str, str | None, int]]] = defaultdict(set)
    sensor_replay_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"delivered": 0, "accepted": 0, "duplicate": 0, "late": 0,
                 "invalid": 0, "dropped": 0}
    )
    for decision in replay_summary.decisions:
        disposition = decision.disposition
        replay_counts[disposition] += 1
        sensor_id = decision.identity.sensor_id
        if disposition is not DeliveryDisposition.DROPPED:
            sensor_replay_counts[sensor_id]["delivered"] += 1
        if disposition is not DeliveryDisposition.DUPLICATE:
            source_identities.add(decision.identity.key)
            sensor_source_counts[sensor_id].add(decision.identity.key)
        sensor_replay_counts[sensor_id][disposition.value] += 1

    if len(source_identities) != replay_summary.source_count:
        raise IntegratedQualityError("replay decisions do not match unique source packet count")

    sensor_execution_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"updated": 0, "failed": 0, "not_executed": 0}
    )
    for outcome in execution_record.outcomes:
        sensor_execution_counts[outcome.identity.sensor_id][outcome.disposition.value] += 1

    output = list(base)
    run_counts = {
        "duplicate": replay_counts[DeliveryDisposition.DUPLICATE],
        "late": replay_counts[DeliveryDisposition.LATE],
        "invalid": replay_counts[DeliveryDisposition.INVALID],
        "dropped": replay_counts[DeliveryDisposition.DROPPED],
    }
    _append_count_metrics(
        output,
        run_counts,
        scope=MetricScope.RUN,
        sensor_id=None,
        source_count=replay_summary.source_count,
        delivered_count=replay_summary.delivered_count,
        accepted_count=replay_summary.accepted_count,
        updated_count=execution_record.updated_count,
        failed_count=execution_record.failed_count,
        not_executed_count=execution_record.not_executed_count,
    )

    sensor_ids = sorted(set(sensor_source_counts) | set(sensor_execution_counts))
    for sensor_id in sensor_ids:
        replay = sensor_replay_counts[sensor_id]
        execution = sensor_execution_counts[sensor_id]
        source_count = len(sensor_source_counts[sensor_id])
        accepted_count = replay["accepted"]
        _append_count_metrics(
            output,
            replay,
            scope=MetricScope.SENSOR,
            sensor_id=sensor_id,
            source_count=source_count,
            delivered_count=replay["delivered"],
            accepted_count=accepted_count,
            updated_count=execution["updated"],
            failed_count=execution["failed"],
            not_executed_count=execution["not_executed"],
        )

    return tuple(output)


def evaluate_integrated_quality(
    result: FilterResult,
    *,
    replay_summary: TransportReplaySummary,
    execution_record: FusionExecutionRecord,
    policy: QualityGatePolicy,
    generated_at: str,
    configuration_hash: str,
    ground_truth=None,
    envelopes: Sequence = (),
    expected_measurements: Mapping[str, int] | None = None,
    alignment_tolerance_s: float = 0.05,
) -> QualityReport:
    """Evaluate integrated evidence with the existing deterministic Quality Gate.

    Source provenance comes from the replay source snapshot; output provenance
    comes from the identity-preserving Fusion execution hash. Configuration
    provenance remains explicit and caller-supplied.
    """
    metrics = compute_integrated_quality_metrics(
        result,
        replay_summary=replay_summary,
        execution_record=execution_record,
        ground_truth=ground_truth,
        envelopes=envelopes,
        expected_measurements=expected_measurements,
        alignment_tolerance_s=alignment_tolerance_s,
    )
    return evaluate_quality_gate(
        metrics,
        policy,
        run_id=replay_summary.run_id,
        generated_at=generated_at,
        source_hash=replay_summary.source_hash,
        output_hash=execution_record.execution_hash,
        configuration_hash=configuration_hash,
    )
