"""End-to-end integration tests for scheduled, overlapping fault campaigns."""
import math

from roboqa.fault_injection import (
    FaultInjectionEngine,
    FaultScenario,
    FaultSchedule,
    FaultType,
    RawMeasurementEnvelope,
    TemporalScheduler,
)


def _scenario(scenario_id, fault_type, parameters, *, start, end, seed):
    return FaultScenario(
        scenario_id, fault_type, ("gps",), start, end, seed, parameters
    )


def _inputs():
    return (
        RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"z": [10.0, 20.0]}),
        RawMeasurementEnvelope("gps", 2.0, 2.1, 1, {"z": [11.0, 21.0]}),
        RawMeasurementEnvelope("gps", 3.5, 3.6, 2, {"z": [12.0, 22.0]}),
        RawMeasurementEnvelope("gps", 5.0, 5.1, 3, {"z": [13.0, 23.0]}),
        RawMeasurementEnvelope("gps", 6.0, 6.1, 4, {"z": [14.0, 24.0]}),
    )


def _scheduled_overlapping_scenarios(schedule_order):
    templates = {
        "bias": FaultSchedule(
            _scenario(
                "gps-bias", FaultType.BIAS, {"offset": [1.0, -1.0]},
                start=0.0, end=4.0, seed=101,
            )
        ),
        "delay": FaultSchedule(
            _scenario(
                "gps-delay", FaultType.DELAY, {"delay_s": 0.5},
                start=2.0, end=6.0, seed=202,
            )
        ),
    }
    return TemporalScheduler().schedule(
        tuple(templates[name] for name in schedule_order),
        window_start=0.0,
        window_end=8.0,
    )


def test_scheduler_to_overlapping_campaign_composes_faults_end_to_end():
    inputs = _inputs()
    occurrences = _scheduled_overlapping_scenarios(("delay", "bias"))

    campaign = FaultInjectionEngine().run_overlapping_campaign(
        inputs, tuple(item.scenario for item in occurrences)
    )

    assert [item.result.scenario_id for item in campaign.executions] == [
        "gps-bias::occurrence:000000",
        "gps-delay::occurrence:000000",
    ]
    by_sequence = {item.sequence_number: item for item in campaign.output_envelopes}
    assert by_sequence[0].payload["z"] == (11.0, 19.0)
    assert by_sequence[1].payload["z"] == (12.0, 20.0)
    assert by_sequence[2].payload["z"] == (13.0, 21.0)
    assert by_sequence[3].payload["z"] == (13.0, 23.0)
    assert by_sequence[4].payload["z"] == (14.0, 24.0)
    assert by_sequence[0].arrival_time == 1.1
    assert by_sequence[1].arrival_time == 2.6
    assert by_sequence[2].arrival_time == 4.1
    assert by_sequence[3].arrival_time == 5.6
    assert by_sequence[4].arrival_time == 6.1
    assert campaign.input_envelopes == inputs


def test_schedule_and_campaign_are_reproducible_under_reversed_input_order():
    inputs = _inputs()
    first_occurrences = _scheduled_overlapping_scenarios(("bias", "delay"))
    second_occurrences = _scheduled_overlapping_scenarios(("delay", "bias"))
    first = FaultInjectionEngine().run_overlapping_campaign(
        inputs, tuple(item.scenario for item in first_occurrences)
    )
    second = FaultInjectionEngine().run_overlapping_campaign(
        inputs, tuple(item.scenario for item in second_occurrences)
    )

    assert [item.scenario_id for item in first_occurrences] == [
        item.scenario_id for item in second_occurrences
    ]
    assert first.output_envelopes == second.output_envelopes
    assert [
        item.provenance.source_hash for item in first.executions
    ] == [
        item.provenance.source_hash for item in second.executions
    ]
    assert [
        item.provenance.output_hash for item in first.executions
    ] == [
        item.provenance.output_hash for item in second.executions
    ]


def test_campaign_provenance_chain_matches_intermediate_hashes():
    occurrences = _scheduled_overlapping_scenarios(("bias", "delay"))
    campaign = FaultInjectionEngine().run_overlapping_campaign(
        _inputs(), tuple(item.scenario for item in occurrences)
    )

    first, second = campaign.executions
    assert first.provenance.parent_run_id is None
    assert second.provenance.parent_run_id == first.provenance.run_id
    assert first.provenance.output_hash == second.provenance.source_hash
    assert second.provenance.output_hash == campaign.executions[-1].result.output_reference.removeprefix("sha256:")
    assert first.result.output_reference == f"sha256:{first.provenance.output_hash}"
    assert second.result.input_reference == f"sha256:{second.provenance.source_hash}"


def test_half_open_fault_windows_leave_end_boundary_measurement_unchanged():
    inputs = _inputs()
    occurrences = _scheduled_overlapping_scenarios(("bias", "delay"))
    campaign = FaultInjectionEngine().run_overlapping_campaign(
        inputs, tuple(item.scenario for item in occurrences)
    )

    at_end = next(item for item in campaign.output_envelopes if item.sequence_number == 4)
    assert at_end.acquisition_time == 6.0
    assert at_end.arrival_time == 6.1
    assert at_end.payload["z"] == (14.0, 24.0)


def test_nonfinite_processing_clock_survives_transport_pipeline_but_is_omitted_from_event():
    envelope = RawMeasurementEnvelope(
        "gps", 2.5, 2.6, 9, {"z": [1.0, 2.0]}, processing_time=float("inf")
    )
    scenario = _scenario(
        "delay-with-invalid-processing-clock", FaultType.DELAY,
        {"delay_s": 0.25}, start=2.0, end=3.0, seed=77,
    )

    execution = FaultInjectionEngine().run((envelope,), scenario)

    assert execution.output_envelopes[0].processing_time == float("inf")
    assert execution.output_envelopes[0].arrival_time == 2.85
    assert not execution.output_envelopes[0].processing_timestamp_finite
    assert execution.result.events[0].processing_time is None
    assert execution.provenance.source_hash != execution.provenance.output_hash
    assert math.isinf(execution.input_envelopes[0].processing_time)
