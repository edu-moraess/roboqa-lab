"""Deterministic, stateless operators for raw measurement envelopes (Phase 3.2)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from roboqa.fault_injection.contracts import ContractError, FaultScenario, FaultType
from roboqa.fault_injection.envelopes import RawMeasurementEnvelope


@runtime_checkable
class EnvelopeFaultOperator(Protocol):
    operator_id: str
    version: str
    fault_type: FaultType

    def validate_parameters(self, parameters: Mapping[str, object]) -> None: ...
    def is_applicable(self, envelope: RawMeasurementEnvelope, scenario: FaultScenario) -> bool: ...
    def apply(
        self,
        envelope: RawMeasurementEnvelope,
        scenario: FaultScenario,
        rng: np.random.Generator,
    ) -> tuple[RawMeasurementEnvelope, ...]: ...


def _finite_number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ContractError(f"{name} must be finite and >= {minimum}")
    return result


def _index(value: object, size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
        raise ContractError(f"index must be an integer in [0, {size - 1}]")
    return value


def _with_payload(
    envelope: RawMeasurementEnvelope, payload: Mapping[str, object]
) -> RawMeasurementEnvelope:
    return RawMeasurementEnvelope(
        sensor_id=envelope.sensor_id,
        acquisition_time=envelope.acquisition_time,
        arrival_time=envelope.arrival_time,
        sequence_number=envelope.sequence_number,
        payload=payload,
        source_id=envelope.source_id,
        processing_time=envelope.processing_time,
    )


class _BaseOperator:
    operator_id = "base"
    version = "1.0.0"
    fault_type: FaultType

    def validate_parameters(self, parameters: Mapping[str, object]) -> None:
        if not isinstance(parameters, Mapping):
            raise ContractError("operator parameters must be a mapping")

    def is_applicable(self, envelope, scenario):
        del envelope, scenario
        return True

    def describe(self) -> Mapping[str, object]:
        return {"operator_id": self.operator_id, "version": self.version,
                "fault_type": self.fault_type.value}


class DropoutOperator(_BaseOperator):
    operator_id = "dropout"
    fault_type = FaultType.SENSOR_DROPOUT

    def validate_parameters(self, parameters: Mapping[str, object]) -> None:
        super().validate_parameters(parameters)
        if parameters:
            raise ContractError("dropout operator accepts no parameters")

    def apply(self, envelope, scenario, rng):
        del scenario, rng
        return ()


class PacketLossOperator(_BaseOperator):
    """Independent Bernoulli packet loss using the scenario-local RNG."""

    operator_id = "packet-loss"
    fault_type = FaultType.PACKET_LOSS

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        probability = _finite_number(
            parameters.get("loss_probability"), "loss_probability", minimum=0.0
        )
        if probability > 1.0:
            raise ContractError("loss_probability must be in [0, 1]")

    def apply(self, envelope, scenario, rng):
        probability = float(scenario.parameters["loss_probability"])
        if rng.random() < probability:
            return ()
        return (envelope,)


class BurstLossOperator(_BaseOperator):
    """Remove packets whose acquisition times fall in declared burst intervals."""

    operator_id = "burst-loss"
    fault_type = FaultType.INTERMITTENT_OUTAGE

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        intervals = parameters.get("burst_intervals_s")
        if not isinstance(intervals, (tuple, list)) or not intervals:
            raise ContractError("burst_intervals_s must be a non-empty list of [start, end] pairs")
        previous_end = -math.inf
        for interval in intervals:
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                raise ContractError("each burst interval must contain [start, end]")
            start = _finite_number(interval[0], "burst start")
            end = _finite_number(interval[1], "burst end")
            if start >= end:
                raise ContractError("each burst interval must satisfy start < end")
            if start < previous_end:
                raise ContractError("burst intervals must be sorted and non-overlapping")
            previous_end = end

    def is_applicable(self, envelope, scenario):
        return any(
            start <= envelope.acquisition_time < end
            for start, end in scenario.parameters["burst_intervals_s"]
        )

    def apply(self, envelope, scenario, rng):
        del envelope, scenario, rng
        return ()


class DelayOperator(_BaseOperator):
    """Apply constant, uniform, or exponential non-negative transport delay."""

    operator_id = "delay"
    fault_type = FaultType.DELAY

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        distribution = parameters.get("distribution", "constant")
        if distribution == "constant":
            _finite_number(parameters.get("delay_s"), "delay_s", minimum=0.0)
        elif distribution == "uniform":
            minimum = _finite_number(parameters.get("min_delay_s"), "min_delay_s", minimum=0.0)
            maximum = _finite_number(parameters.get("max_delay_s"), "max_delay_s", minimum=0.0)
            if minimum >= maximum:
                raise ContractError("uniform delay requires min_delay_s < max_delay_s")
        elif distribution == "exponential":
            mean = _finite_number(parameters.get("mean_delay_s"), "mean_delay_s", minimum=0.0)
            if mean <= 0.0:
                raise ContractError("mean_delay_s must be > 0 for exponential delay")
        else:
            raise ContractError("distribution must be constant, uniform, or exponential")

    def apply(self, envelope, scenario, rng):
        parameters = scenario.parameters
        distribution = parameters.get("distribution", "constant")
        if distribution == "constant":
            delay = float(parameters["delay_s"])
        elif distribution == "uniform":
            delay = float(rng.uniform(parameters["min_delay_s"], parameters["max_delay_s"]))
        else:
            delay = float(rng.exponential(parameters["mean_delay_s"]))
        return (RawMeasurementEnvelope(
            sensor_id=envelope.sensor_id,
            acquisition_time=envelope.acquisition_time,
            arrival_time=envelope.arrival_time + delay,
            sequence_number=envelope.sequence_number,
            payload=envelope.payload,
            source_id=envelope.source_id,
            processing_time=envelope.processing_time,
        ),)


class ArrivalShiftOperator(_BaseOperator):
    """Shift transport arrival time; signed shifts can induce out-of-order arrival."""
    operator_id = "arrival-shift"
    fault_type = FaultType.OUT_OF_ORDER

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        if "shift_s" not in parameters:
            raise ContractError("shift_s is required")
        _finite_number(parameters["shift_s"], "shift_s")
        if "sequence_numbers" in parameters:
            selected = parameters["sequence_numbers"]
            if not isinstance(selected, (tuple, list)) or not selected:
                raise ContractError("sequence_numbers must be a non-empty integer list")
            if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in selected):
                raise ContractError("sequence_numbers must contain non-negative integers")
            if len(set(selected)) != len(selected):
                raise ContractError("sequence_numbers must be unique")

    def is_applicable(self, envelope, scenario):
        selected = scenario.parameters.get("sequence_numbers")
        return selected is None or envelope.sequence_number in selected

    def apply(self, envelope, scenario, rng):
        del rng
        shift = _finite_number(scenario.parameters["shift_s"], "shift_s")
        return (RawMeasurementEnvelope(
            sensor_id=envelope.sensor_id,
            acquisition_time=envelope.acquisition_time,
            arrival_time=envelope.arrival_time + shift,
            sequence_number=envelope.sequence_number,
            payload=envelope.payload,
            source_id=envelope.source_id,
            processing_time=envelope.processing_time,
        ),)


def _z_payload(envelope: RawMeasurementEnvelope) -> tuple[dict[str, object], list[float]]:
    if not isinstance(envelope.payload, Mapping) or "z" not in envelope.payload:
        raise ContractError("operator requires a mapping payload with a 'z' vector")
    z = envelope.payload["z"]
    if not isinstance(z, (tuple, list)) or not z:
        raise ContractError("payload['z'] must be a non-empty vector")
    try:
        values = [float(value) for value in z]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError("payload['z'] must contain numeric values") from exc
    return dict(envelope.payload), values


class BiasOperator(_BaseOperator):
    operator_id = "bias"
    fault_type = FaultType.BIAS

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        offset = parameters.get("offset")
        if not isinstance(offset, (tuple, list)) or not offset:
            raise ContractError("offset must be a non-empty numeric vector")
        for value in offset:
            _finite_number(value, "offset item")

    def apply(self, envelope, scenario, rng):
        del rng
        payload, z = _z_payload(envelope)
        offset = scenario.parameters["offset"]
        if len(offset) != len(z):
            raise ContractError("offset dimension must match payload['z']")
        payload["z"] = [value + float(delta) for value, delta in zip(z, offset)]
        return (_with_payload(envelope, payload),)


class OutlierOperator(_BaseOperator):
    operator_id = "outlier"
    fault_type = FaultType.OUTLIER

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        _finite_number(parameters.get("sigma"), "sigma", minimum=0.0)
        if float(parameters["sigma"]) == 0.0:
            raise ContractError("sigma must be > 0")
        if "index" in parameters and (
            isinstance(parameters["index"], bool) or not isinstance(parameters["index"], int)
            or parameters["index"] < 0
        ):
            raise ContractError("index must be a non-negative integer")

    def apply(self, envelope, scenario, rng):
        payload, z = _z_payload(envelope)
        index = _index(scenario.parameters.get("index", 0), len(z))
        sigma = _finite_number(scenario.parameters["sigma"], "sigma", minimum=0.0)
        if sigma == 0.0:
            raise ContractError("sigma must be > 0")
        z[index] += float(rng.normal(0.0, sigma))
        payload["z"] = z
        return (_with_payload(envelope, payload),)


class InvalidPayloadOperator(_BaseOperator):
    operator_id = "invalid-payload"
    fault_type = FaultType.INVALID_PAYLOAD

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        kind = parameters.get("kind", "nan")
        if kind not in ("nan", "positive_inf", "negative_inf"):
            raise ContractError("kind must be nan, positive_inf, or negative_inf")
        if "index" in parameters and (
            isinstance(parameters["index"], bool) or not isinstance(parameters["index"], int)
            or parameters["index"] < 0
        ):
            raise ContractError("index must be a non-negative integer")

    def apply(self, envelope, scenario, rng):
        del rng
        payload, z = _z_payload(envelope)
        index = _index(scenario.parameters.get("index", 0), len(z))
        kind = scenario.parameters.get("kind", "nan")
        z[index] = {"nan": float("nan"), "positive_inf": float("inf"),
                    "negative_inf": float("-inf")}[str(kind)]
        payload["z"] = z
        return (_with_payload(envelope, payload),)


class DuplicateOperator(_BaseOperator):
    operator_id = "duplicate"
    fault_type = FaultType.DUPLICATE

    def validate_parameters(self, parameters):
        super().validate_parameters(parameters)
        delay = parameters.get("duplicate_delay_s", 0.0)
        _finite_number(delay, "duplicate_delay_s", minimum=0.0)

    def apply(self, envelope, scenario, rng):
        del rng
        delay = _finite_number(
            scenario.parameters.get("duplicate_delay_s", 0.0),
            "duplicate_delay_s", minimum=0.0,
        )
        duplicate = RawMeasurementEnvelope(
            sensor_id=envelope.sensor_id,
            acquisition_time=envelope.acquisition_time,
            arrival_time=envelope.arrival_time + delay,
            sequence_number=envelope.sequence_number,
            payload=envelope.payload,
            source_id=envelope.source_id,
            processing_time=envelope.processing_time,
        )
        return (envelope, duplicate)


DEFAULT_OPERATORS: tuple[EnvelopeFaultOperator, ...] = (
    DropoutOperator(),
    PacketLossOperator(),
    DelayOperator(),
    ArrivalShiftOperator(),
    BiasOperator(),
    OutlierOperator(),
    InvalidPayloadOperator(),
    DuplicateOperator(),
)
