"""Unit / integration tests — runner and async sensors."""

import numpy as np
import pytest

from roboqa.data.schemas import Measurement, SchemaError
from roboqa.data.synthetic import make_synthetic_ground_truth
from roboqa.fusion.result import FilterStatus
from roboqa.fusion.runner import run_filter, simulate_measurements


def test_simulate_and_run_basic():
    gt = make_synthetic_ground_truth(n=50, dt=0.1, seed=42)
    ms = simulate_measurements(gt, seed=42, gps_every=5)
    result = run_filter(ms, provenance={"gt_source": gt.source, "seed": 42})
    assert result.status in (FilterStatus.OK, FilterStatus.WARNING)
    assert result.n >= 1
    assert result.diagnostics.n_update > 0
    assert "gps" in result.diagnostics.sensors_seen


def test_empty_measurements():
    result = run_filter([])
    assert result.status == FilterStatus.INIT_INSUFFICIENT


def test_irregular_timestamps():
    ms = [
        Measurement(t=0.0, sensor="gps", z=np.array([0.0, 0.0]), R=np.eye(2)),
        Measurement(t=0.07, sensor="speed", z=np.array([5.0]), R=np.array([[0.04]])),
        Measurement(t=0.35, sensor="yaw_rate", z=np.array([0.0]), R=np.array([[0.001]])),
        Measurement(t=0.40, sensor="gps", z=np.array([2.0, 0.1]), R=np.eye(2)),
    ]
    result = run_filter(ms)
    assert result.status in (FilterStatus.OK, FilterStatus.WARNING)
    assert result.diagnostics.n_update == 4


def test_no_gps_still_runs():
    ms = [
        Measurement(t=0.0, sensor="speed", z=np.array([5.0]), R=np.array([[0.04]])),
        Measurement(t=0.1, sensor="yaw_rate", z=np.array([0.0]), R=np.array([[0.001]])),
        Measurement(t=0.2, sensor="speed", z=np.array([5.0]), R=np.array([[0.04]])),
    ]
    result = run_filter(ms)
    assert result.status in (FilterStatus.OK, FilterStatus.WARNING)
    assert result.diagnostics.n_update == 3


def test_determinism():
    gt = make_synthetic_ground_truth(n=40, seed=7)
    ms = simulate_measurements(gt, seed=7)
    r1 = run_filter(ms, sigma_a=0.4, sigma_alpha=0.08)
    r2 = run_filter(ms, sigma_a=0.4, sigma_alpha=0.08)
    np.testing.assert_allclose(r1.x, r2.x, atol=1e-12)


def test_does_not_mutate_gt_or_measurements():
    gt = make_synthetic_ground_truth(n=30, seed=1)
    t0 = gt.t.copy()
    pos0 = gt.pos.copy()
    ms = simulate_measurements(gt, seed=1)
    z0 = [m.z.copy() for m in ms]
    run_filter(ms)
    np.testing.assert_array_equal(gt.t, t0)
    np.testing.assert_array_equal(gt.pos, pos0)
    for m, z in zip(ms, z0):
        np.testing.assert_array_equal(m.z, z)


def test_invalid_sensor_in_sequence():
    ms = [
        Measurement(t=0.0, sensor="gps", z=np.array([0.0, 0.0]), R=np.eye(2)),
        Measurement(t=0.1, sensor="unknown_sensor", z=np.array([1.0]), R=np.array([[1.0]])),
    ]
    result = run_filter(ms)
    assert result.status == FilterStatus.INVALID_INPUT
