import pytest

from roboqa.fault_injection import (
    ContractError,
    FaultScenario,
    FaultSchedule,
    FaultType,
    TemporalScheduler,
)


def _scenario(scenario_id="gps-dropout", *, start=2.0, end=4.0, seed=42,
              fault_type=FaultType.SENSOR_DROPOUT):
    return FaultScenario(scenario_id, fault_type, ("gps",), start, end, seed, {})


def test_one_shot_schedule_emits_original_interval_with_occurrence_identity():
    result = TemporalScheduler().schedule(
        (FaultSchedule(_scenario()),), window_start=0.0, window_end=10.0
    )
    assert len(result) == 1
    assert result[0].occurrence_index == 0
    assert result[0].scenario_id == "gps-dropout::occurrence:000000"
    assert result[0].start_time == 2.0
    assert result[0].end_time == 4.0


def test_fixed_interval_recurrence_is_anchored_and_deterministic():
    schedule = FaultSchedule(_scenario(), repeat_interval_s=5.0, max_occurrences=4)
    scheduler = TemporalScheduler()
    first = scheduler.schedule((schedule,), window_start=0.0, window_end=20.0)
    second = scheduler.schedule((schedule,), window_start=0.0, window_end=20.0)
    assert [item.start_time for item in first] == [2.0, 7.0, 12.0, 17.0]
    assert [item.scenario.seed for item in first] == [item.scenario.seed for item in second]
    assert [item.scenario_id for item in first] == [item.scenario_id for item in second]
    assert len({item.scenario.seed for item in first}) == 4


def test_horizon_selects_intersecting_occurrences_without_clipping_them():
    schedule = FaultSchedule(_scenario(), repeat_interval_s=5.0, max_occurrences=5)
    result = TemporalScheduler().schedule((schedule,), window_start=3.0, window_end=8.0)
    assert [(item.start_time, item.end_time) for item in result] == [(2.0, 4.0), (7.0, 9.0)]


def test_half_open_horizon_excludes_occurrence_starting_at_window_end():
    schedule = FaultSchedule(_scenario(), repeat_interval_s=2.0, max_occurrences=4)
    result = TemporalScheduler().schedule((schedule,), window_start=0.0, window_end=6.0)
    assert [item.start_time for item in result] == [2.0, 4.0]


def test_independent_schedules_can_overlap_and_are_sorted_stably():
    a = FaultSchedule(_scenario("zeta", start=3.0, end=5.0))
    b = FaultSchedule(_scenario("alpha", start=3.0, end=4.0, fault_type=FaultType.DELAY))
    result = TemporalScheduler().schedule((a, b), window_start=0.0, window_end=10.0)
    assert [item.scenario_id for item in result] == [
        "alpha::occurrence:000000", "zeta::occurrence:000000"
    ]


def test_occurrence_seed_changes_with_index_and_scenario_identity():
    scheduler = TemporalScheduler()
    first = scheduler.schedule(
        (FaultSchedule(_scenario(), repeat_interval_s=5.0, max_occurrences=2),),
        window_start=0.0, window_end=20.0,
    )
    other = scheduler.schedule(
        (FaultSchedule(_scenario("other"), repeat_interval_s=5.0, max_occurrences=2),),
        window_start=0.0, window_end=20.0,
    )
    assert first[0].scenario.seed != first[1].scenario.seed
    assert first[0].scenario.seed != other[0].scenario.seed


@pytest.mark.parametrize(
    ("interval", "max_occurrences"),
    [(0.0, None), (-1.0, None), (1.0, None), (5.0, 0), (5.0, True)],
)
def test_schedule_rejects_invalid_recurrence(interval, max_occurrences):
    with pytest.raises(ContractError):
        FaultSchedule(_scenario(start=0.0, end=2.0),
                      repeat_interval_s=interval, max_occurrences=max_occurrences)


def test_schedule_rejects_recurrence_limit_without_interval():
    with pytest.raises(ContractError, match="requires repeat_interval_s"):
        FaultSchedule(_scenario(), max_occurrences=2)


@pytest.mark.parametrize(
    ("window_start", "window_end"),
    [(1.0, 1.0), (2.0, 1.0), (float("nan"), 3.0), (0.0, float("inf"))],
)
def test_scheduler_rejects_invalid_windows(window_start, window_end):
    with pytest.raises(ContractError):
        TemporalScheduler().schedule(
            (FaultSchedule(_scenario()),),
            window_start=window_start, window_end=window_end,
        )


def test_scheduler_rejects_empty_invalid_or_duplicate_schedules():
    scheduler = TemporalScheduler()
    with pytest.raises(ContractError, match="at least one"):
        scheduler.schedule((), window_start=0.0, window_end=10.0)
    with pytest.raises(ContractError, match="FaultSchedule"):
        scheduler.schedule((object(),), window_start=0.0, window_end=10.0)
    duplicate = FaultSchedule(_scenario())
    with pytest.raises(ContractError, match="unique"):
        scheduler.schedule((duplicate, duplicate), window_start=0.0, window_end=10.0)


def test_scheduler_occurrences_are_ready_for_existing_fault_engine():
    from roboqa.fault_injection import FaultInjectionEngine, RawMeasurementEnvelope

    schedule = FaultSchedule(
        _scenario(), repeat_interval_s=5.0, max_occurrences=2,
    )
    occurrences = TemporalScheduler().schedule(
        (schedule,), window_start=0.0, window_end=10.0,
    )
    envelopes = (
        RawMeasurementEnvelope("gps", 2.5, 2.6, 0, {"z": [1.0]}),
        RawMeasurementEnvelope("gps", 7.5, 7.6, 1, {"z": [2.0]}),
    )
    executions = tuple(FaultInjectionEngine().run(envelopes, item.scenario) for item in occurrences)
    assert [item.result.applied_count for item in executions] == [1, 1]
    assert all(not item.output_envelopes for item in executions)
