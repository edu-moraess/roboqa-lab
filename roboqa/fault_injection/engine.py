"""Deterministic orchestration for fault injection over raw envelopes (Phases 3.2–3.3-C)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np

from roboqa.fault_injection.contracts import (
    ContractError,
    FaultEvent,
    FaultResult,
    FaultScenario,
    FaultStatus,
    FaultType,
)
from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.fault_injection.operators import (
    ArrivalShiftOperator,
    BiasOperator,
    BurstLossOperator,
    DelayOperator,
    DropoutOperator,
    DuplicateOperator,
    EnvelopeFaultOperator,
    InvalidPayloadOperator,
    OutlierOperator,
    PacketLossOperator,
)
from roboqa.fault_injection.provenance import (
    FaultProvenance,
    build_configuration_hash,
    sha256_digest,
)


@dataclass(frozen=True)
class FaultExecution:
    """Immutable input/output snapshot and audit records for one execution."""
    input_envelopes: tuple[RawMeasurementEnvelope, ...]
    output_envelopes: tuple[RawMeasurementEnvelope, ...]
    result: FaultResult
    provenance: FaultProvenance


@dataclass(frozen=True)
class FaultCampaignExecution:
    """Ordered composition of scenarios with an explicit provenance chain."""
    input_envelopes: tuple[RawMeasurementEnvelope, ...]
    output_envelopes: tuple[RawMeasurementEnvelope, ...]
    executions: tuple[FaultExecution, ...]


def _portable(value):
    """Represent non-finite raw values explicitly so their hashes remain stable."""
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "positive_inf" if value > 0 else "negative_inf"}
        return value
    if isinstance(value, Mapping):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return value


def _envelope_document(envelope: RawMeasurementEnvelope) -> dict:
    return {
        "sensor_id": envelope.sensor_id,
        "acquisition_time": _portable(envelope.acquisition_time),
        "arrival_time": _portable(envelope.arrival_time),
        "processing_time": _portable(envelope.processing_time),
        "sequence_number": envelope.sequence_number,
        "payload": _portable(envelope.payload),
        "source_id": envelope.source_id,
    }


def _hash_envelopes(envelopes: Sequence[RawMeasurementEnvelope]) -> str:
    return sha256_digest([_envelope_document(item) for item in envelopes])


class FaultInjectionEngine:
    """Runs one scenario atomically using a local, seeded NumPy generator.

    Supported operators are explicit. Unsupported FaultType values fail closed.
    Original envelopes are immutable and are returned in the execution record.
    """

    def __init__(self, operators: Sequence[EnvelopeFaultOperator] | None = None):
        selected = tuple(operators) if operators is not None else (
            DropoutOperator(), PacketLossOperator(), BurstLossOperator(), DelayOperator(),
            ArrivalShiftOperator(), BiasOperator(), OutlierOperator(),
            InvalidPayloadOperator(), DuplicateOperator(),
        )
        if not selected:
            raise ContractError("at least one operator must be registered")
        registry: dict[FaultType, EnvelopeFaultOperator] = {}
        for operator in selected:
            if not isinstance(operator, EnvelopeFaultOperator):
                raise ContractError("every operator must implement EnvelopeFaultOperator")
            if operator.fault_type in registry:
                raise ContractError(f"duplicate operator for fault type {operator.fault_type.value}")
            registry[operator.fault_type] = operator
        self._operators = registry

    @property
    def supported_fault_types(self) -> tuple[FaultType, ...]:
        return tuple(sorted(self._operators, key=lambda item: item.value))

    def run(
        self,
        envelopes: Sequence[RawMeasurementEnvelope],
        scenario: FaultScenario,
        *,
        parent_run_id: str | None = None,
    ) -> FaultExecution:
        if not isinstance(scenario, FaultScenario):
            raise ContractError("scenario must be a FaultScenario")
        original = tuple(envelopes)
        if any(not isinstance(item, RawMeasurementEnvelope) for item in original):
            raise ContractError("all inputs must be RawMeasurementEnvelope instances")
        operator = self._operators.get(scenario.fault_type)
        if operator is None:
            raise ContractError(
                f"unsupported fault type for current engine: {scenario.fault_type.value}"
            )
        operator.validate_parameters(scenario.parameters)
        rng = np.random.default_rng(scenario.seed)
        output: list[tuple[int, RawMeasurementEnvelope]] = []
        events: list[FaultEvent] = []
        applied = 0
        skipped = 0

        for input_index, envelope in enumerate(original):
            eligible = (
                envelope.sensor_id in scenario.target_sensors
                and math.isfinite(envelope.acquisition_time)
                and math.isfinite(envelope.arrival_time)
                and scenario.contains(envelope.acquisition_time)
                and operator.is_applicable(envelope, scenario)
            )
            if not eligible:
                output.append((input_index, envelope))
                skipped += 1
                continue

            transformed = operator.apply(envelope, scenario, rng)
            if not isinstance(transformed, tuple) or any(
                not isinstance(item, RawMeasurementEnvelope) for item in transformed
            ):
                raise ContractError("operators must return a tuple of RawMeasurementEnvelope")
            if not transformed:
                # A dropout is a recorded applied event with no emitted envelope.
                event_arrival = envelope.arrival_time
            else:
                event_arrival = transformed[0].arrival_time
            events.append(FaultEvent(
                event_id=f"{scenario.scenario_id}:{input_index}:{envelope.sequence_number}",
                scenario_id=scenario.scenario_id,
                operator_id=operator.operator_id,
                sensor_id=envelope.sensor_id,
                acquisition_time=envelope.acquisition_time,
                arrival_time=event_arrival,
                processing_time=(
                    envelope.processing_time if envelope.processing_timestamp_finite else None
                ),
                sequence_number=envelope.sequence_number,
                fault_type=scenario.fault_type,
                severity=1.0,
                affected=transformed != (envelope,),
                metadata={
                    "input_index": input_index,
                    "output_count": len(transformed),
                    "operator_effect": transformed != (envelope,),
                },
            ))
            if transformed != (envelope,):
                applied += 1
            else:
                # A stochastic operator may inspect a packet but leave it unchanged.
                skipped += 1
            output.extend((input_index, item) for item in transformed)

        # Arrival order is transport order; sequence_number retains source identity.
        output.sort(key=lambda pair: (
            pair[1].arrival_time if math.isfinite(pair[1].arrival_time) else math.inf,
            pair[1].sequence_number,
            pair[0],
        ))
        output_envelopes = tuple(item for _, item in output)
        source_hash = _hash_envelopes(original)
        output_hash = _hash_envelopes(output_envelopes)
        status = (
            FaultStatus.NOT_APPLICABLE if applied == 0
            else FaultStatus.PARTIALLY_APPLIED if skipped > 0
            else FaultStatus.APPLIED
        )
        result = FaultResult(
            scenario_id=scenario.scenario_id,
            input_reference=f"sha256:{source_hash}",
            output_reference=f"sha256:{output_hash}",
            events=tuple(events),
            applied_count=applied,
            skipped_count=skipped,
            status=status,
            seed=scenario.seed,
        )
        configuration_hash = build_configuration_hash({
            "fault_type": scenario.fault_type.value,
            "target_sensors": list(scenario.target_sensors),
            "start_time": scenario.start_time,
            "end_time": scenario.end_time,
            "parameters": dict(scenario.parameters),
        })
        run_id = f"{scenario.scenario_id}-{source_hash[:12]}-{output_hash[:12]}"
        provenance = FaultProvenance(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            source_hash=source_hash,
            output_hash=output_hash,
            operator_version=operator.version,
            configuration_hash=configuration_hash,
            seed=scenario.seed,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            parent_run_id=parent_run_id,
        )
        return FaultExecution(
            input_envelopes=original,
            output_envelopes=output_envelopes,
            result=result,
            provenance=provenance,
        )

    def run_campaign(
        self,
        envelopes: Sequence[RawMeasurementEnvelope],
        scenarios: Sequence[FaultScenario],
    ) -> FaultCampaignExecution:
        """Apply scenarios sequentially in caller-declared order.

        Each scenario receives the previous scenario's output. The original
        input snapshot is retained, and provenance records form a parent chain.
        """
        original = tuple(envelopes)
        selected = tuple(scenarios)
        if not selected:
            raise ContractError("a campaign requires at least one scenario")
        if any(not isinstance(item, FaultScenario) for item in selected):
            raise ContractError("all campaign items must be FaultScenario instances")
        scenario_ids = [item.scenario_id for item in selected]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ContractError("campaign scenario_id values must be unique")

        current = original
        executions: list[FaultExecution] = []
        parent_run_id = None
        for scenario in selected:
            execution = self.run(current, scenario, parent_run_id=parent_run_id)
            executions.append(execution)
            current = execution.output_envelopes
            parent_run_id = execution.provenance.run_id

        return FaultCampaignExecution(
            input_envelopes=original,
            output_envelopes=current,
            executions=tuple(executions),
        )
