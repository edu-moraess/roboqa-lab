"""Deterministic scheduling of fault scenarios (Phase 3.3-B)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

from roboqa.fault_injection.contracts import ContractError, FaultScenario


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FaultSchedule:
    """A scenario template and its optional fixed-interval recurrence."""

    scenario: FaultScenario
    repeat_interval_s: float | None = None
    max_occurrences: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, FaultScenario):
            raise ContractError("scenario must be a FaultScenario")
        if self.repeat_interval_s is None:
            if self.max_occurrences not in (None, 1):
                raise ContractError("max_occurrences > 1 requires repeat_interval_s")
            object.__setattr__(self, "max_occurrences", 1)
            return
        interval = _finite(self.repeat_interval_s, "repeat_interval_s")
        duration = self.scenario.end_time - self.scenario.start_time
        if interval <= 0.0:
            raise ContractError("repeat_interval_s must be > 0")
        if interval < duration:
            raise ContractError("repeat_interval_s must be >= scenario duration")
        object.__setattr__(self, "repeat_interval_s", interval)
        if self.max_occurrences is not None and (
            isinstance(self.max_occurrences, bool)
            or not isinstance(self.max_occurrences, int)
            or self.max_occurrences < 1
        ):
            raise ContractError("max_occurrences must be a positive integer or None")


@dataclass(frozen=True)
class ScheduledFaultOccurrence:
    """One immutable occurrence ready to pass to FaultInjectionEngine.run."""

    scenario: FaultScenario
    occurrence_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, FaultScenario):
            raise ContractError("scenario must be a FaultScenario")
        if (
            isinstance(self.occurrence_index, bool)
            or not isinstance(self.occurrence_index, int)
            or self.occurrence_index < 0
        ):
            raise ContractError("occurrence_index must be a non-negative integer")

    @property
    def start_time(self) -> float:
        return self.scenario.start_time

    @property
    def end_time(self) -> float:
        return self.scenario.end_time

    @property
    def scenario_id(self) -> str:
        return self.scenario.scenario_id


class TemporalScheduler:
    """Expand schedules into deterministic, half-open temporal occurrences."""

    MAX_GENERATED_OCCURRENCES = 100_000

    @staticmethod
    def _occurrence_seed(base_seed: int, scenario_id: str, index: int) -> int:
        material = f"roboqa-temporal-scheduler-v1|{base_seed}|{scenario_id}|{index}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def schedule(
        self,
        schedules: Sequence[FaultSchedule],
        *,
        window_start: float,
        window_end: float,
    ) -> tuple[ScheduledFaultOccurrence, ...]:
        """Expand schedules intersecting [window_start, window_end)."""
        start = _finite(window_start, "window_start")
        end = _finite(window_end, "window_end")
        if start >= end:
            raise ContractError("scheduler window must satisfy window_start < window_end")
        selected = tuple(schedules)
        if not selected:
            raise ContractError("at least one schedule is required")
        if any(not isinstance(item, FaultSchedule) for item in selected):
            raise ContractError("all schedules must be FaultSchedule instances")
        ids = [item.scenario.scenario_id for item in selected]
        if len(set(ids)) != len(ids):
            raise ContractError("schedule scenario IDs must be unique")

        occurrences: list[ScheduledFaultOccurrence] = []
        for item in selected:
            template = item.scenario
            duration = template.end_time - template.start_time
            interval = item.repeat_interval_s
            if interval is None:
                candidate_indices = (0,)
            else:
                first = max(
                    0, math.floor((start - template.start_time - duration) / interval) + 1
                )
                limit = item.max_occurrences or self.MAX_GENERATED_OCCURRENCES + 1
                candidate_indices = range(first, limit)

            for index in candidate_indices:
                if len(occurrences) >= self.MAX_GENERATED_OCCURRENCES:
                    raise ContractError("schedule expansion exceeds MAX_GENERATED_OCCURRENCES")
                occurrence_start = (
                    template.start_time if index == 0
                    else template.start_time + index * interval
                )
                if occurrence_start >= end:
                    break
                occurrence_end = occurrence_start + duration
                if occurrence_end <= start:
                    continue
                occurrence_id = f"{template.scenario_id}::occurrence:{index:06d}"
                scenario = FaultScenario(
                    scenario_id=occurrence_id,
                    fault_type=template.fault_type,
                    target_sensors=template.target_sensors,
                    start_time=occurrence_start,
                    end_time=occurrence_end,
                    seed=self._occurrence_seed(template.seed, template.scenario_id, index),
                    parameters=template.parameters,
                )
                occurrences.append(ScheduledFaultOccurrence(scenario, index))

        occurrences.sort(key=lambda item: (
            item.start_time,
            item.scenario_id.rsplit("::occurrence:", 1)[0],
            item.occurrence_index,
        ))
        return tuple(occurrences)
