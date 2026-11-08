import pytest

from roboqa.fault_injection import (
    ContractError, FaultInjectionEngine, FaultScenario, FaultStatus, FaultType,
    RawMeasurementEnvelope,
)


def _envelopes():
    return (
        RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"z": [1.0, 2.0]}),
        RawMeasurementEnvelope("imu", 1.0, 1.05, 1, {"z": [3.0]}),
        RawMeasurementEnvelope("gps", 2.0, 2.1, 2, {"z": [4.0, 5.0]}),
    )


def _scenario(kind=FaultType.DELAY, params=None, start=0.0, end=3.0, seed=7):
    if params is None:
        params = {"delay_s": 1.0} if kind == FaultType.DELAY else {}
    return FaultScenario("engine-test", kind, ("gps",), start, end, seed, params)


def test_engine_applies_only_matching_sensor_and_interval():
    execution = FaultInjectionEngine().run(_envelopes(), _scenario())
    assert execution.result.applied_count == 2
    assert execution.result.skipped_count == 1
    assert execution.result.status == FaultStatus.PARTIALLY_APPLIED
    assert execution.output_envelopes[0].sensor_id == "imu"
    assert execution.output_envelopes[1].arrival_time == pytest.approx(2.1)


def test_engine_dropout_preserves_unaffected_data_and_records_event():
    execution = FaultInjectionEngine().run(
        _envelopes(), _scenario(FaultType.SENSOR_DROPOUT, {})
    )
    assert len(execution.input_envelopes) == 3
    assert len(execution.output_envelopes) == 1
    assert execution.output_envelopes[0].sensor_id == "imu"
    assert execution.result.applied_count == 2
    assert len(execution.result.events) == 2


def test_engine_reorders_by_arrival_but_retains_sequence_identity():
    inputs = (
        RawMeasurementEnvelope("gps", 1.0, 1.0, 0, {"z": [1.0]}),
        RawMeasurementEnvelope("gps", 2.0, 2.0, 1, {"z": [2.0]}),
    )
    scenario = _scenario(FaultType.OUT_OF_ORDER, {"shift_s": -2.0, "sequence_numbers": [1]})
    execution = FaultInjectionEngine().run(inputs, scenario)
    assert [item.sequence_number for item in execution.output_envelopes] == [1, 0]


def test_engine_is_deterministic_and_provenance_hashes_are_stable():
    inputs = _envelopes()
    scenario = _scenario(FaultType.OUTLIER, {"sigma": 3.0, "index": 0}, seed=123)
    engine = FaultInjectionEngine()
    first = engine.run(inputs, scenario)
    second = engine.run(inputs, scenario)
    assert first.output_envelopes == second.output_envelopes
    assert first.provenance.source_hash == second.provenance.source_hash
    assert first.provenance.output_hash == second.provenance.output_hash
    assert first.provenance.configuration_hash == second.provenance.configuration_hash


def test_engine_hashes_nonfinite_raw_values_without_json_nan_extension():
    inputs = (RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"z": [float("nan")]}),)
    execution = FaultInjectionEngine().run(
        inputs, _scenario(FaultType.BIAS, {"offset": (1.0,)})
    )
    assert len(execution.output_envelopes) == 1
    assert execution.provenance.source_hash != execution.provenance.output_hash


def test_engine_fails_closed_for_unsupported_fault_type():
    scenario = _scenario(FaultType.STUCK_AT, {"value": 0.0})
    with pytest.raises(ContractError, match="unsupported fault type"):
        FaultInjectionEngine().run(_envelopes(), scenario)


def test_engine_does_not_accept_normalized_measurements():
    class NotEnvelope:
        pass
    with pytest.raises(ContractError):
        FaultInjectionEngine().run((NotEnvelope(),), _scenario())


def test_engine_validates_scenario_operator_parameters_before_processing():
    with pytest.raises(ContractError):
        FaultInjectionEngine().run(_envelopes(), _scenario(FaultType.DELAY, {"delay_s": -2}))
