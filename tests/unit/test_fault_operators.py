import numpy as np
import pytest

from roboqa.fault_injection import ContractError, FaultScenario, FaultType, RawMeasurementEnvelope
from roboqa.fault_injection.operators import (
    ArrivalShiftOperator, BiasOperator, DelayOperator, DropoutOperator,
    DuplicateOperator, InvalidPayloadOperator, OutlierOperator, PacketLossOperator,
)


def _scenario(kind, parameters=None, seed=42):
    return FaultScenario("s", kind, ("gps",), 0.0, 10.0, seed, parameters or {})


def _envelope(t=1.0, arrival=1.1, z=(1.0, 2.0)):
    return RawMeasurementEnvelope("gps", t, arrival, 4, {"z": z, "R": [[1.0, 0.0], [0.0, 1.0]]})


def test_dropout_and_packet_loss_remove_only_targeted_envelope():
    source = _envelope()
    assert DropoutOperator().apply(source, _scenario(FaultType.SENSOR_DROPOUT), np.random.default_rng(1)) == ()
    assert PacketLossOperator().apply(source, _scenario(FaultType.PACKET_LOSS), np.random.default_rng(1)) == ()


def test_delay_preserves_acquisition_time_and_adds_transport_delay():
    source = _envelope()
    output = DelayOperator().apply(source, _scenario(FaultType.DELAY, {"delay_s": 2.5}), np.random.default_rng(1))[0]
    assert output.acquisition_time == source.acquisition_time
    assert output.arrival_time == pytest.approx(3.6)
    assert output.payload == source.payload


def test_arrival_shift_can_create_earlier_transport_arrival():
    source = _envelope()
    output = ArrivalShiftOperator().apply(source, _scenario(FaultType.OUT_OF_ORDER, {"shift_s": -2.0}), np.random.default_rng(1))[0]
    assert output.arrival_time < source.acquisition_time


def test_bias_is_applied_componentwise_without_mutating_input():
    source = _envelope()
    output = BiasOperator().apply(source, _scenario(FaultType.BIAS, {"offset": (3.0, -1.0)}), np.random.default_rng(1))[0]
    assert output.payload["z"] == (4.0, 1.0)
    assert source.payload["z"] == (1.0, 2.0)


def test_outlier_is_reproducible_for_same_seed():
    source = _envelope()
    scenario = _scenario(FaultType.OUTLIER, {"sigma": 10.0, "index": 1}, seed=123)
    a = OutlierOperator().apply(source, scenario, np.random.default_rng(scenario.seed))[0]
    b = OutlierOperator().apply(source, scenario, np.random.default_rng(scenario.seed))[0]
    assert a.payload == b.payload
    assert a.payload["z"][0] == 1.0


def test_invalid_payload_injects_nan_at_raw_boundary():
    source = _envelope()
    output = InvalidPayloadOperator().apply(source, _scenario(FaultType.INVALID_PAYLOAD), np.random.default_rng(1))[0]
    assert np.isnan(output.payload["z"][0])
    assert not output.payload_finite


def test_duplicate_preserves_identity_and_emits_two_packets():
    source = _envelope()
    output = DuplicateOperator().apply(source, _scenario(FaultType.DUPLICATE, {"duplicate_delay_s": 0.2}), np.random.default_rng(1))
    assert len(output) == 2
    assert output[0].sequence_number == output[1].sequence_number == source.sequence_number
    assert output[1].arrival_time == pytest.approx(source.arrival_time + 0.2)


@pytest.mark.parametrize("operator,params", [
    (DelayOperator(), {"delay_s": -1}),
    (ArrivalShiftOperator(), {}),
    (BiasOperator(), {"offset": ()}),
    (OutlierOperator(), {"sigma": 0}),
    (InvalidPayloadOperator(), {"kind": "unknown"}),
    (DuplicateOperator(), {"duplicate_delay_s": -1}),
])
def test_invalid_operator_parameters_are_rejected(operator, params):
    with pytest.raises(ContractError):
        operator.validate_parameters(params)
