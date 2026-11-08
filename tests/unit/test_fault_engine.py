import pytest

from roboqa.fault_injection import (
    ContractError,
    FaultInjectionEngine,
    FaultScenario,
    FaultStatus,
    FaultType,
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


def test_engine_packet_loss_probability_zero_preserves_packets_and_marks_no_effect():
    scenario = _scenario(FaultType.PACKET_LOSS, {"loss_probability": 0.0})
    execution = FaultInjectionEngine().run(_envelopes(), scenario)
    assert len(execution.output_envelopes) == 3
    assert execution.result.applied_count == 2
    assert all(not event.affected for event in execution.result.events)


def test_engine_packet_loss_probability_one_drops_only_target_sensor_packets():
    scenario = _scenario(FaultType.PACKET_LOSS, {"loss_probability": 1.0})
    execution = FaultInjectionEngine().run(_envelopes(), scenario)
    assert [item.sensor_id for item in execution.output_envelopes] == ["imu"]
    assert execution.result.applied_count == 2
    assert all(event.affected for event in execution.result.events)


def test_engine_burst_loss_drops_only_packets_inside_burst_intervals():
    inputs = tuple(
        RawMeasurementEnvelope("gps", timestamp, timestamp + 0.01, index, {"z": [index]})
        for index, timestamp in enumerate((1.0, 2.0, 3.0, 4.0, 5.0))
    )
    scenario = FaultScenario(
        "gps-bursts", FaultType.INTERMITTENT_OUTAGE, ("gps",), 0.0, 10.0, 3,
        {"burst_intervals_s": [[2.0, 4.0]]},
    )
    execution = FaultInjectionEngine().run(inputs, scenario)
    assert [item.sequence_number for item in execution.output_envelopes] == [0, 3, 4]
    assert execution.result.applied_count == 2
    assert execution.result.skipped_count == 3


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
    inputs = (RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"z": [float("nan"), 2.0]}),)
    execution = FaultInjectionEngine().run(
        inputs, _scenario(FaultType.BIAS, {"offset": (1.0, 1.0)})
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


def test_campaign_composes_scenarios_and_chains_provenance():
    inputs = (RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"z": [1.0]}),)
    bias = FaultScenario("bias-step", FaultType.BIAS, ("gps",), 0.0, 2.0, 10, {"offset": [2.0]})
    delay = FaultScenario("delay-step", FaultType.DELAY, ("gps",), 0.0, 2.0, 11, {"delay_s": 0.5})

    campaign = FaultInjectionEngine().run_campaign(inputs, (bias, delay))

    assert campaign.input_envelopes == inputs
    assert campaign.output_envelopes[0].payload["z"] == (3.0,)
    assert campaign.output_envelopes[0].arrival_time == pytest.approx(1.6)
    assert campaign.executions[0].provenance.parent_run_id is None
    assert campaign.executions[1].provenance.parent_run_id == campaign.executions[0].provenance.run_id


def test_campaign_rejects_empty_or_ambiguous_scenario_lists():
    engine = FaultInjectionEngine()
    with pytest.raises(ContractError, match="at least one scenario"):
        engine.run_campaign(_envelopes(), ())
    repeated = _scenario(FaultType.DELAY, {"delay_s": 0.1})
    with pytest.raises(ContractError, match="unique"):
        engine.run_campaign(_envelopes(), (repeated, repeated))


def test_engine_constructor_rejects_empty_invalid_and_duplicate_registries():
    from roboqa.fault_injection.operators import DelayOperator

    with pytest.raises(ContractError, match="at least one"):
        FaultInjectionEngine(())
    with pytest.raises(ContractError, match="implement EnvelopeFaultOperator"):
        FaultInjectionEngine((object(),))
    with pytest.raises(ContractError, match="duplicate operator"):
        FaultInjectionEngine((DelayOperator(), DelayOperator()))


def test_engine_rejects_invalid_scenario_and_operator_output_contract():
    engine = FaultInjectionEngine()
    with pytest.raises(ContractError, match="scenario must"):
        engine.run(_envelopes(), object())

    class BadOperator:
        operator_id = "bad"
        version = "1"
        fault_type = FaultType.DELAY

        def validate_parameters(self, parameters):
            return None

        def is_applicable(self, envelope, scenario):
            return True

        def apply(self, envelope, scenario, rng):
            return [envelope]

    bad_engine = FaultInjectionEngine((BadOperator(),))
    scenario = _scenario(FaultType.DELAY, {"delay_s": 1.0})
    with pytest.raises(ContractError, match="must return a tuple"):
        bad_engine.run(_envelopes(), scenario)


def test_engine_reports_not_applicable_when_no_envelope_matches():
    execution = FaultInjectionEngine().run(
        _envelopes(),
        FaultScenario("none", FaultType.DELAY, ("lidar",), 0.0, 3.0, 1, {"delay_s": 1.0}),
    )
    assert execution.result.status == FaultStatus.NOT_APPLICABLE
    assert execution.result.applied_count == 0
