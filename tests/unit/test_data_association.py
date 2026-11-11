import numpy as np
import pytest

from roboqa.data.association import associate_by_timestamp
from roboqa.data.schemas import GroundTruth, Measurement, SchemaError


def _gt(times):
    n = len(times)
    return GroundTruth(
        t=np.asarray(times, dtype=float),
        pos=np.zeros((n, 2)),
        yaw=np.zeros(n),
        speed=np.zeros(n),
        yaw_rate=np.zeros(n),
        source="synthetic",
    )


def _m(t):
    return Measurement(t=float(t), sensor="gps", z=np.zeros(2), R=np.eye(2))


def _kwargs(**extra):
    values = dict(
        measurement_sequence_id="measurements-A",
        ground_truth_sequence_id="reference-A",
        measurement_clock_domain="synthetic-sequence-clock",
        ground_truth_clock_domain="synthetic-sequence-clock",
        measurement_time_basis="elapsed_seconds_from_sequence_start",
        ground_truth_time_basis="elapsed_seconds_from_sequence_start",
        tolerance_s=1e-6,
    )
    values.update(extra)
    return values


def test_unique_timestamp_association_is_explicit_and_one_to_one():
    matches = associate_by_timestamp([_m(0.0), _m(1.0)], _gt([0.0, 1.0]), **_kwargs())
    assert [m.status for m in matches] == ["matched", "matched"]
    assert [m.ground_truth_index for m in matches] == [0, 1]
    assert [m.delta_t_s for m in matches] == [0.0, 0.0]


def test_no_match_remains_unmatched():
    result = associate_by_timestamp([_m(4.0)], _gt([0.0, 1.0]), **_kwargs())
    assert result[0].status == "unmatched"
    assert result[0].ground_truth_index is None


def test_multiple_candidates_remain_ambiguous():
    result = associate_by_timestamp(
        [_m(0.5)], _gt([0.0, 1.0]), **_kwargs(tolerance_s=0.6)
    )
    assert result[0].status == "ambiguous"
    assert result[0].ground_truth_index is None


def test_duplicate_claims_are_not_reused():
    result = associate_by_timestamp(
        [_m(1.0), _m(1.0)], _gt([1.0]), **_kwargs()
    )
    assert [item.status for item in result] == ["ambiguous", "ambiguous"]


@pytest.mark.parametrize("override", [
    {"measurement_clock_domain": "clock-A", "ground_truth_clock_domain": "clock-B"},
    {"measurement_time_basis": "UTC", "ground_truth_time_basis": "elapsed"},
])
def test_incomparable_time_declarations_fail_closed(override):
    with pytest.raises(SchemaError):
        associate_by_timestamp([_m(0.0)], _gt([0.0]), **_kwargs(**override))


def test_tolerance_must_be_finite_and_nonnegative():
    with pytest.raises(SchemaError):
        associate_by_timestamp([_m(0.0)], _gt([0.0]), **_kwargs(tolerance_s=-1.0))
