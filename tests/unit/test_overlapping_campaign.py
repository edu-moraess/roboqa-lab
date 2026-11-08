import pytest

from roboqa.fault_injection import (
    ContractError,
    FaultInjectionEngine,
    FaultScenario,
    FaultType,
    RawMeasurementEnvelope,
)


def _envelope(sequence=0, acquisition=1.0, sensor="gps"):
    return RawMeasurementEnvelope(
        sensor, acquisition, acquisition + 0.1, sequence, {"z": [1.0, 2.0]}
    )


def _scenario(scenario_id, fault_type, parameters, *, start=0.0, end=5.0, seed=7):
    return FaultScenario(
        scenario_id, fault_type, ("gps",), start, end, seed, parameters
    )


def test_overlapping_campaign_order_is_independent_of_input_sequence():
    inputs = (_envelope(0), _envelope(1, acquisition=2.0))
    bias = _scenario("a-bias", FaultType.BIAS, {"offset": [2.0, -1.0]}, seed=11)
    delay = _scenario("b-delay", FaultType.DELAY, {"delay_s": 0.25}, seed=12)
    engine = FaultInjectionEngine()

    first = engine.run_overlapping_campaign(inputs, (delay, bias))
    second = engine.run_overlapping_campaign(inputs, (bias, delay))

    assert [item.result.scenario_id for item in first.executions] == [
        "a-bias", "b-delay"
    ]
    assert [item.result.scenario_id for item in second.executions] == [
        "a-bias", "b-delay"
    ]
    assert first.output_envelopes == second.output_envelopes
    assert first.executions[-1].result.output_reference == second.executions[-1].result.output_reference
    assert first.executions[0].provenance.parent_run_id is None
    assert first.executions[1].provenance.parent_run_id == first.executions[0].provenance.run_id
    assert first.output_envelopes[0].payload["z"] == (3.0, 1.0)
    assert first.output_envelopes[0].arrival_time == pytest.approx(1.35)


def test_overlapping_campaign_removal_prevents_later_operator_from_seeing_packet():
    inputs = (_envelope(),)
    dropout = _scenario("a-dropout", FaultType.SENSOR_DROPOUT, {})
    bias = _scenario("z-bias", FaultType.BIAS, {"offset": [10.0, 10.0]})

    campaign = FaultInjectionEngine().run_overlapping_campaign(
        inputs, (bias, dropout)
    )

    assert campaign.output_envelopes == ()
    assert campaign.executions[0].result.applied_count == 1
    assert campaign.executions[1].result.applied_count == 0
    assert campaign.executions[1].result.status.value == "NOT_APPLICABLE"


def test_overlapping_campaign_duplicates_are_visible_to_later_operators():
    inputs = (_envelope(),)
    duplicate = _scenario("a-duplicate", FaultType.DUPLICATE, {"duplicate_delay_s": 0.2})
    bias = _scenario("z-bias", FaultType.BIAS, {"offset": [1.0, -1.0]})

    campaign = FaultInjectionEngine().run_overlapping_campaign(
        inputs, (bias, duplicate)
    )

    assert len(campaign.output_envelopes) == 2
    assert campaign.executions[1].result.applied_count == 2
    assert all(item.payload["z"] == (2.0, 1.0) for item in campaign.output_envelopes)


def test_overlapping_campaign_rejects_single_or_non_overlapping_scenarios():
    engine = FaultInjectionEngine()
    only_one = _scenario("one", FaultType.BIAS, {"offset": [1.0, 1.0]})
    with pytest.raises(ContractError, match="at least two"):
        engine.run_overlapping_campaign((_envelope(),), (only_one,))

    left = _scenario("left", FaultType.BIAS, {"offset": [1.0, 1.0]}, start=0.0, end=2.0)
    right = _scenario("right", FaultType.DELAY, {"delay_s": 0.1}, start=2.0, end=4.0)
    with pytest.raises(ContractError, match="must overlap"):
        engine.run_overlapping_campaign((_envelope(),), (left, right))


def test_overlapping_campaign_rejects_invalid_or_duplicate_scenarios():
    engine = FaultInjectionEngine()
    valid = _scenario("same", FaultType.BIAS, {"offset": [1.0, 1.0]})
    with pytest.raises(ContractError, match="unique"):
        engine.run_overlapping_campaign((_envelope(),), (valid, valid))
    with pytest.raises(ContractError, match="FaultScenario"):
        engine.run_overlapping_campaign((_envelope(),), (valid, object()))


def test_overlapping_campaign_uses_half_open_overlap_boundaries():
    engine = FaultInjectionEngine()
    left = _scenario("left", FaultType.BIAS, {"offset": [1.0, 1.0]}, start=0.0, end=2.0)
    right = _scenario("right", FaultType.DELAY, {"delay_s": 0.1}, start=1.999, end=4.0)
    campaign = engine.run_overlapping_campaign((_envelope(),), (right, left))
    assert [item.result.scenario_id for item in campaign.executions] == ["left", "right"]
