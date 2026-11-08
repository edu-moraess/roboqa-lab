"""Adversarial cross-component tests for fault campaign composition."""
import pytest

from roboqa.fault_injection import (
    ContractError,
    FaultInjectionEngine,
    FaultScenario,
    FaultType,
    RawMeasurementEnvelope,
)


def _scenario(name, kind, parameters, *, start=0.0, end=5.0, seed=42):
    return FaultScenario(name, kind, ("gps",), start, end, seed, parameters)


def test_duplicate_then_bias_keeps_event_identity_unique_for_same_source_sequence():
    source = RawMeasurementEnvelope(
        "gps", 1.0, 1.1, 7, {"z": [1.0, 2.0], "metadata": {"tags": ["raw"]}}
    )
    duplicate = _scenario(
        "a-duplicate", FaultType.DUPLICATE, {"duplicate_delay_s": 0.2}
    )
    bias = _scenario("b-bias", FaultType.BIAS, {"offset": [3.0, -1.0]})

    campaign = FaultInjectionEngine().run_overlapping_campaign(
        (source,), (bias, duplicate)
    )

    second = campaign.executions[1]
    assert len(second.result.events) == 2
    assert len({event.event_id for event in second.result.events}) == 2
    assert {event.sequence_number for event in second.result.events} == {7}
    assert len(campaign.output_envelopes) == 2
    assert all(item.payload["z"] == (4.0, 1.0) for item in campaign.output_envelopes)
    assert source.payload["z"] == (1.0, 2.0)
    assert source.payload["metadata"]["tags"] == ("raw",)


def test_campaign_can_include_a_third_nonoverlapping_scenario_when_pair_overlaps():
    inputs = (RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"z": [1.0]}),)
    early = _scenario("a-early", FaultType.BIAS, {"offset": [1.0]}, start=0.0, end=4.0)
    overlap = _scenario("b-overlap", FaultType.DELAY, {"delay_s": 0.1}, start=2.0, end=6.0)
    later = _scenario("c-later", FaultType.OUTLIER, {"sigma": 1.0}, start=8.0, end=10.0)

    campaign = FaultInjectionEngine().run_overlapping_campaign(
        inputs, (later, overlap, early)
    )

    assert [item.result.scenario_id for item in campaign.executions] == [
        "a-early", "b-overlap", "c-later"
    ]
    assert campaign.executions[2].result.applied_count == 0
    assert campaign.output_envelopes[0].payload["z"][0] == 2.0


def test_invalid_or_unsupported_fault_fails_closed_without_mutating_source():
    source = RawMeasurementEnvelope(
        "gps", 1.0, 1.1, 0, {"z": [1.0], "metadata": {"tags": ["immutable"]}}
    )
    supported = _scenario("a-bias", FaultType.BIAS, {"offset": [2.0]})
    unsupported = _scenario("b-stuck", FaultType.STUCK_AT, {"value": 0.0})

    with pytest.raises(ContractError, match="unsupported fault type"):
        FaultInjectionEngine().run_overlapping_campaign(
            (source,), (supported, unsupported)
        )

    assert source.payload["z"] == (1.0,)
    assert source.payload["metadata"]["tags"] == ("immutable",)


def test_overlapping_campaign_rejects_non_envelope_input_before_execution():
    scenarios = (
        _scenario("a-bias", FaultType.BIAS, {"offset": [1.0]}),
        _scenario("b-delay", FaultType.DELAY, {"delay_s": 0.1}),
    )
    with pytest.raises(ContractError, match="RawMeasurementEnvelope"):
        FaultInjectionEngine().run_overlapping_campaign((object(),), scenarios)
