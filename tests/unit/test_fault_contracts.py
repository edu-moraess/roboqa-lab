import pytest

from roboqa.fault_injection import (
    ContractError,
    FaultEvent,
    FaultResult,
    FaultScenario,
    FaultStatus,
    FaultType,
    RecoveryRecord,
    RecoveryStatus,
)


def scenario(**overrides):
    values = dict(
        scenario_id="scenario-1",
        fault_type=FaultType.DELAY,
        target_sensors=("gps", "imu"),
        start_time=1.0,
        end_time=3.0,
        seed=42,
        parameters={"delay_s": 0.25, "nested": {"levels": [1, 2]}},
    )
    values.update(overrides)
    return FaultScenario(**values)


def event(**overrides):
    values = dict(
        event_id="event-1",
        scenario_id="scenario-1",
        operator_id="delay-v1",
        sensor_id="gps",
        acquisition_time=1.5,
        arrival_time=1.75,
        sequence_number=0,
        fault_type=FaultType.DELAY,
    )
    values.update(overrides)
    return FaultEvent(**values)


def test_scenario_interval_is_start_inclusive_end_exclusive():
    item = scenario()
    assert item.contains(1.0)
    assert item.contains(2.999)
    assert not item.contains(3.0)


@pytest.mark.parametrize("overrides", [
    {"scenario_id": " "},
    {"target_sensors": ()},
    {"target_sensors": ("gps", "gps")},
    {"start_time": 3.0, "end_time": 1.0},
    {"start_time": float("nan")},
    {"seed": -1},
    {"seed": True},
    {"parameters": {"bad": float("inf")}},
])
def test_invalid_scenario_is_rejected(overrides):
    with pytest.raises(ContractError):
        scenario(**overrides)


def test_scenario_parameters_are_deeply_immutable():
    item = scenario()
    with pytest.raises(TypeError):
        item.parameters["delay_s"] = 1.0
    with pytest.raises(TypeError):
        item.parameters["nested"]["new"] = 3
    assert item.parameters["nested"]["levels"] == (1, 2)


@pytest.mark.parametrize("overrides", [
    {"event_id": ""},
    {"sequence_number": -1},
    {"severity": 1.1},
    {"severity": float("nan")},
    {"affected": 1},
    {"metadata": {"bad": float("nan")}},
])
def test_invalid_event_is_rejected(overrides):
    with pytest.raises(ContractError):
        event(**overrides)


def test_event_preserves_distinct_acquisition_and_arrival_times():
    item = event(acquisition_time=10.0, arrival_time=12.5)
    assert item.acquisition_time == 10.0
    assert item.arrival_time == 12.5


def test_fault_result_validates_event_ownership_and_status():
    item = event()
    result = FaultResult(
        scenario_id="scenario-1",
        input_reference="input:sha256",
        output_reference="output:sha256",
        events=(item,),
        applied_count=1,
        skipped_count=0,
        status=FaultStatus.APPLIED,
        seed=42,
    )
    assert result.events == (item,)
    with pytest.raises(ContractError):
        FaultResult(
            scenario_id="other",
            input_reference="in",
            output_reference="out",
            events=(item,),
            applied_count=1,
            skipped_count=0,
            status=FaultStatus.APPLIED,
            seed=42,
        )
    with pytest.raises(ContractError):
        FaultResult(
            scenario_id="scenario-1",
            input_reference="in",
            output_reference="out",
            events=(),
            applied_count=0,
            skipped_count=0,
            status=FaultStatus.APPLIED,
            seed=42,
        )


def test_recovery_record_requires_consistent_times_and_nonnegative_errors():
    record = RecoveryRecord(
        recovery_id="recovery-1",
        scenario_id="scenario-1",
        sensor_id="gps",
        fault_end_time=10.0,
        sensor_return_time=12.0,
        recovery_time=1.5,
        baseline_error=0.2,
        recovery_error=0.3,
        recovery_status=RecoveryStatus.RECOVERED,
        evidence={"window_samples": 20},
    )
    assert record.recovery_time == 1.5
    with pytest.raises(ContractError):
        RecoveryRecord(
            recovery_id="r", scenario_id="s", sensor_id="gps",
            fault_end_time=10.0, sensor_return_time=9.0, recovery_time=1.0,
            baseline_error=0.0, recovery_error=0.1,
            recovery_status=RecoveryStatus.RECOVERED,
        )
