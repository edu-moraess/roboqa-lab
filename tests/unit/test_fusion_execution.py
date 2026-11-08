"""Identity-preserving Fusion schedule execution tests (Phase 4.5-D2)."""
import numpy as np
import pytest

from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.fusion.ekf import EKFNumericalError
from roboqa.integration import (
    TransportOrderingPolicy,
    adapt_accepted_envelopes,
    build_fusion_schedule,
    execute_fusion_schedule,
    replay_transport,
)
from roboqa.integration.execution import (
    ExecutionDisposition,
    FusionExecutionError,
    MeasurementExecutionOutcome,
)


def packet(sequence, sensor, acquisition, payload, *, source="execution-test"):
    return RawMeasurementEnvelope(
        sensor_id=sensor,
        source_id=source,
        sequence_number=sequence,
        acquisition_time=acquisition,
        arrival_time=acquisition + 0.01,
        processing_time=acquisition + 0.02,
        payload=payload,
    )


def make_schedule():
    source = (
        packet(1, "gps", 0.0, (10.0, 20.0)),
        packet(2, "speed", 0.5, (4.0,)),
        packet(3, "yaw_rate", 1.0, (0.05,)),
        packet(4, "gps", 1.5, (11.0, 20.5)),
    )
    replay = replay_transport(
        source, source,
        ordering_policy=TransportOrderingPolicy.ARRIVAL_ORDER,
    )
    adapted = adapt_accepted_envelopes(
        replay,
        sensor_covariances={
            "gps": np.eye(2),
            "speed": np.array([[0.25]]),
            "yaw_rate": np.array([[0.01]]),
        },
    )
    return build_fusion_schedule(replay, adapted)


def test_executes_schedule_and_maps_every_telemetry_to_source_identity():
    schedule = make_schedule()
    execution = execute_fusion_schedule(schedule)

    assert execution.record.run_id == schedule.run_id
    assert execution.record.schedule_hash == schedule.schedule_hash
    assert execution.record.filter_status.value == "ok"
    assert execution.record.updated_count == 4
    assert execution.record.failed_count == 0
    assert execution.record.not_executed_count == 0
    assert len(execution.filter_result.telemetry) == 4
    assert [
        (item.identity.sensor_id, item.identity.sequence_number)
        for item in execution.record.outcomes
    ] == [
        (item.identity.sensor_id, item.identity.sequence_number)
        for item in schedule.entries
    ]
    for outcome, telemetry in zip(
        execution.record.outcomes, execution.filter_result.telemetry
    ):
        assert outcome.disposition is ExecutionDisposition.UPDATED
        assert outcome.telemetry is telemetry
        assert outcome.acquisition_time == telemetry.timestamp
        assert outcome.identity.sensor_id == telemetry.sensor_id
    assert execution.record.execution_hash == execution.record.compute_hash()


def test_execution_hash_is_reproducible_for_same_schedule_and_configuration():
    schedule = make_schedule()
    first = execute_fusion_schedule(schedule)
    second = execute_fusion_schedule(schedule)

    assert first.record.execution_hash == second.record.execution_hash
    np.testing.assert_array_equal(first.filter_result.x, second.filter_result.x)
    np.testing.assert_array_equal(first.filter_result.P, second.filter_result.P)


def test_numerical_failure_is_attributed_and_remaining_items_are_not_executed(monkeypatch):
    schedule = make_schedule()
    original = __import__("roboqa.fusion.ekf", fromlist=["EKF"]).EKF.update
    calls = {"count": 0}

    def fail_second(self, measurement):
        calls["count"] += 1
        if calls["count"] == 2:
            raise EKFNumericalError("injected D2 failure")
        return original(self, measurement)

    monkeypatch.setattr("roboqa.fusion.ekf.EKF.update", fail_second)
    execution = execute_fusion_schedule(schedule)

    assert execution.record.filter_status.value == "numerical_failure"
    assert execution.record.updated_count == 1
    assert execution.record.failed_count == 1
    assert execution.record.not_executed_count == 2
    assert execution.record.outcomes[0].disposition is ExecutionDisposition.UPDATED
    assert execution.record.outcomes[1].disposition is ExecutionDisposition.FAILED
    assert "injected D2 failure" in execution.record.outcomes[1].reason
    assert all(
        item.disposition is ExecutionDisposition.NOT_EXECUTED
        for item in execution.record.outcomes[2:]
    )


def test_invalid_runner_configuration_does_not_blame_a_measurement():
    schedule = make_schedule()
    execution = execute_fusion_schedule(schedule, x0=np.zeros(4))

    assert execution.record.filter_status.value == "invalid_input"
    assert execution.record.updated_count == 0
    assert execution.record.failed_count == 0
    assert execution.record.not_executed_count == len(schedule.entries)
    assert all(
        item.disposition is ExecutionDisposition.NOT_EXECUTED
        for item in execution.record.outcomes
    )


def test_empty_schedule_preserves_empty_execution_and_replay_provenance():
    source = packet(1, "gps", 0.0, (1.0, 2.0))
    replay = replay_transport((source,), ())
    schedule = build_fusion_schedule(replay, ())
    execution = execute_fusion_schedule(schedule)

    assert execution.record.run_id == replay.summary.run_id
    assert execution.record.outcomes == ()
    assert execution.record.updated_count == 0
    assert execution.record.filter_status.value == "init_insufficient"
    assert execution.record.execution_hash == execution.record.compute_hash()


def test_non_updated_outcome_requires_reason_and_cannot_carry_telemetry():
    schedule = make_schedule()
    entry = schedule.entries[0]
    with pytest.raises(FusionExecutionError, match="explicit reason"):
        MeasurementExecutionOutcome(
            identity=entry.identity,
            acquisition_time=entry.measurement.t,
            disposition=ExecutionDisposition.FAILED,
        )
