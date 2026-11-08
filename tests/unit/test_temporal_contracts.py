import math

import pytest

from roboqa.fault_injection import (
    ContractError,
    FaultInjectionEngine,
    FaultScenario,
    FaultType,
    RawMeasurementEnvelope,
)


def _envelope(*, acquisition=1.0, arrival=1.2, processing=None):
    return RawMeasurementEnvelope(
        sensor_id="gps",
        acquisition_time=acquisition,
        arrival_time=arrival,
        sequence_number=7,
        payload={"z": [10.0, 20.0]},
        processing_time=processing,
    )


def test_three_clock_latency_metrics_are_derived_without_mutation():
    envelope = _envelope(processing=1.35)

    assert envelope.transport_latency_s == pytest.approx(0.2)
    assert envelope.processing_wait_s == pytest.approx(0.15)
    assert envelope.end_to_end_latency_s == pytest.approx(0.35)
    assert envelope.temporal_diagnostics == ()
    assert envelope.temporal_order_valid
    assert envelope.acquisition_time == 1.0
    assert envelope.arrival_time == 1.2
    assert envelope.processing_time == 1.35


def test_processing_clock_is_optional_for_backward_compatibility():
    envelope = RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"z": [1.0]})

    assert envelope.processing_time is None
    assert envelope.transport_latency_s == pytest.approx(0.1)
    assert envelope.processing_wait_s is None
    assert envelope.end_to_end_latency_s is None
    assert envelope.timestamps_finite
    assert envelope.is_valid_candidate


@pytest.mark.parametrize("value", [True, "1.2"])
def test_present_processing_clock_must_be_numeric(value):
    with pytest.raises(ContractError, match="processing_time must be numeric"):
        _envelope(processing=value)


def test_nonfinite_processing_clock_is_preserved_and_diagnosed():
    envelope = _envelope(processing=float("nan"))

    assert math.isnan(envelope.processing_time)
    assert envelope.timestamps_finite
    assert not envelope.processing_timestamp_finite
    assert not envelope.is_valid_candidate
    assert envelope.processing_wait_s is None
    assert envelope.end_to_end_latency_s is None
    assert envelope.temporal_diagnostics == ("non_finite_processing_time",)


def test_processing_before_arrival_is_diagnosed_not_rejected():
    envelope = _envelope(processing=1.1)

    assert envelope.processing_wait_s == pytest.approx(-0.1)
    assert "processing_precedes_arrival" in envelope.temporal_diagnostics
    assert not envelope.temporal_order_valid


def test_arrival_before_acquisition_is_preserved_as_transport_diagnostic():
    envelope = _envelope(acquisition=2.0, arrival=1.5, processing=2.2)

    assert envelope.transport_latency_s == pytest.approx(-0.5)
    assert "arrival_precedes_acquisition" in envelope.temporal_diagnostics
    assert envelope.temporal_order_valid


def test_fault_operator_preserves_processing_clock():
    envelope = _envelope(processing=1.35)
    scenario = FaultScenario(
        "delay-preserves-processing",
        FaultType.DELAY,
        ("gps",),
        0.0,
        2.0,
        42,
        {"delay_s": 0.5},
    )

    execution = FaultInjectionEngine().run((envelope,), scenario)
    output = execution.output_envelopes[0]

    assert output.acquisition_time == envelope.acquisition_time
    assert output.arrival_time == pytest.approx(1.7)
    assert output.processing_time == envelope.processing_time
    assert execution.result.events[0].processing_time == envelope.processing_time


def test_processing_clock_participates_in_provenance_hash():
    first = _envelope(processing=1.3)
    second = _envelope(processing=1.4)
    scenario = FaultScenario(
        "hash-processing-clock",
        FaultType.BIAS,
        ("gps",),
        0.0,
        2.0,
        42,
        {"offset": [0.5, -0.5]},
    )
    engine = FaultInjectionEngine()

    first_run = engine.run((first,), scenario)
    second_run = engine.run((second,), scenario)

    assert first_run.provenance.source_hash != second_run.provenance.source_hash
    assert first_run.provenance.output_hash != second_run.provenance.output_hash


def test_duplicate_operator_preserves_processing_clock_on_both_records():
    envelope = _envelope(processing=1.35)
    scenario = FaultScenario(
        "duplicate-preserves-processing",
        FaultType.DUPLICATE,
        ("gps",),
        0.0,
        2.0,
        42,
        {"duplicate_delay_s": 0.25},
    )

    output = FaultInjectionEngine().run((envelope,), scenario).output_envelopes

    assert len(output) == 2
    assert all(item.processing_time == 1.35 for item in output)


def test_nonfinite_processing_clock_does_not_block_transport_fault_injection():
    envelope = _envelope(processing=float("nan"))
    scenario = FaultScenario(
        "invalid-processing-clock",
        FaultType.DELAY,
        ("gps",),
        0.0,
        2.0,
        42,
        {"delay_s": 0.5},
    )

    execution = FaultInjectionEngine().run((envelope,), scenario)

    assert execution.result.applied_count == 1
    assert math.isnan(execution.output_envelopes[0].processing_time)
    assert execution.result.events[0].processing_time is None
    assert "non_finite_processing_time" in execution.output_envelopes[0].temporal_diagnostics


@pytest.mark.parametrize(
    ("fault_type", "parameters"),
    [
        (FaultType.BIAS, {"offset": [1.0, -1.0]}),
        (FaultType.OUTLIER, {"sigma": 0.5, "index": 0}),
        (FaultType.INVALID_PAYLOAD, {"kind": "nan", "index": 0}),
        (FaultType.OUT_OF_ORDER, {"shift_s": 0.1}),
    ],
)
def test_payload_and_arrival_operators_preserve_processing_clock(fault_type, parameters):
    envelope = _envelope(processing=1.35)
    scenario = FaultScenario(
        f"preserve-processing-{fault_type.value}",
        fault_type,
        ("gps",),
        0.0,
        2.0,
        42,
        parameters,
    )

    output = FaultInjectionEngine().run((envelope,), scenario).output_envelopes

    assert output
    assert all(item.processing_time == 1.35 for item in output)
