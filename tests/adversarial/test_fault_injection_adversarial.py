"""Adversarial validation of the Phase 3.1 contract boundary."""
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


def _scenario(**changes):
    values = {
        "scenario_id": "adversarial",
        "fault_type": FaultType.OUTLIER,
        "target_sensors": ("gps",),
        "start_time": 0.0,
        "end_time": 10.0,
        "seed": 7,
        "parameters": {},
    }
    values.update(changes)
    return FaultScenario(**values)


def _event(**changes):
    values = {
        "event_id": "evt-1",
        "scenario_id": "adversarial",
        "operator_id": "operator-v1",
        "sensor_id": "gps",
        "acquisition_time": 2.0,
        "arrival_time": 1.0,
        "sequence_number": 0,
        "fault_type": FaultType.OUTLIER,
    }
    values.update(changes)
    return FaultEvent(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"fault_type": "outlier"},
        {"target_sensors": ("gps", 7)},
        {"start_time": True},
        {"end_time": 10.0, "start_time": 10.0},
        {"seed": 2**64},
        {"parameters": {"nested": object()}},
    ],
)
def test_scenario_rejects_adversarial_contract_values(changes):
    with pytest.raises(ContractError):
        _scenario(**changes)


def test_temporal_clock_skew_is_preserved_not_silently_rewritten():
    event = _event(processing_time=3.0)
    assert event.acquisition_time == 2.0
    assert event.arrival_time == 1.0
    assert event.processing_time == 3.0


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence_number": True},
        {"fault_type": "outlier"},
        {"processing_time": float("inf")},
        {"metadata": {"nested": object()}},
    ],
)
def test_event_rejects_adversarial_contract_values(changes):
    with pytest.raises(ContractError):
        _event(**changes)


def test_result_rejects_inconsistent_counts_and_statuses():
    with pytest.raises(ContractError):
        FaultResult(
            scenario_id="adversarial", input_reference="in", output_reference="out",
            events=(_event(),), applied_count=0, skipped_count=0,
            status=FaultStatus.PARTIALLY_APPLIED, seed=7,
        )
    with pytest.raises(ContractError):
        FaultResult(
            scenario_id="adversarial", input_reference="in", output_reference="out",
            events=(), applied_count=1, skipped_count=0,
            status=FaultStatus.NOT_APPLICABLE, seed=7,
        )
    with pytest.raises(ContractError):
        FaultResult(
            scenario_id="adversarial", input_reference="in", output_reference="out",
            events=(), applied_count=1, skipped_count=0,
            status=FaultStatus.REJECTED, seed=7,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"baseline_error": -0.1},
        {"recovery_error": -0.1},
        {"recovery_time": 1.0, "sensor_return_time": None},
        {"recovery_status": "RECOVERED"},
        {"evidence": {"bad": object()}},
    ],
)
def test_recovery_record_rejects_adversarial_values(changes):
    values = {
        "recovery_id": "recovery-1",
        "scenario_id": "adversarial",
        "sensor_id": "gps",
        "fault_end_time": 10.0,
        "sensor_return_time": 11.0,
        "recovery_time": 2.0,
        "baseline_error": 0.1,
        "recovery_error": 0.2,
        "recovery_status": RecoveryStatus.RECOVERED,
        "evidence": {},
    }
    values.update(changes)
    with pytest.raises(ContractError):
        RecoveryRecord(**values)
