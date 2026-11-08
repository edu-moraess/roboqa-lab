"""Unit / integration tests — runner and asynchronous sensors."""

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


def test_first_gps_is_applied_exactly_once():
    measurement = Measurement(
        t=0.0, sensor="gps", z=np.array([100.0, -50.0]), R=np.eye(2)
    )
    result = run_filter([measurement])
    assert result.status == FilterStatus.OK
    assert result.diagnostics.n_update == 1
    assert result.n == 2
    np.testing.assert_allclose(
        result.x[-1, :2], np.array([100.0, -50.0]) * (25.0 / 26.0)
    )


def test_future_t0_is_rejected_instead_of_dropping_earlier_data():
    measurement = Measurement(t=1.0, sensor="gps", z=np.zeros(2), R=np.eye(2))
    result = run_filter([measurement], t0=2.0)
    assert result.status == FilterStatus.INVALID_INPUT
    assert result.diagnostics.n_update == 0
    assert result.diagnostics.errors


def test_invalid_t0_is_rejected():
    measurement = Measurement(t=1.0, sensor="gps", z=np.zeros(2), R=np.eye(2))
    result = run_filter([measurement], t0=np.nan)
    assert result.status == FilterStatus.INVALID_INPUT


def test_determinism():
    gt = make_synthetic_ground_truth(n=40, seed=7)
    ms = simulate_measurements(gt, seed=7)
    r1 = run_filter(ms, sigma_a=0.4, sigma_alpha=0.08)
    r2 = run_filter(ms, sigma_a=0.4, sigma_alpha=0.08)
    np.testing.assert_allclose(r1.x, r2.x, atol=1e-12)
    np.testing.assert_allclose(r1.P, r2.P, atol=1e-12)


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


@pytest.mark.parametrize("sigma_a,sigma_alpha", [(-1.0, 0.1), (0.5, -0.1), (np.nan, 0.1)])
def test_invalid_runner_process_noise_parameters(sigma_a, sigma_alpha):
    measurement = Measurement(t=0.0, sensor="gps", z=np.zeros(2), R=np.eye(2))
    result = run_filter([measurement], sigma_a=sigma_a, sigma_alpha=sigma_alpha)
    assert result.status == FilterStatus.INVALID_INPUT

def test_custom_initial_state_and_covariance_are_used():
    measurement = Measurement(t=0.0, sensor="gps", z=np.array([2.0, 3.0]), R=np.eye(2))
    x0 = np.array([1.0, -1.0, 0.2, 4.0, 0.1])
    P0 = np.eye(5) * 2.0
    result = run_filter([measurement], x0=x0, P0=P0)
    assert result.status == FilterStatus.OK
    np.testing.assert_array_equal(result.x[0], x0)
    np.testing.assert_array_equal(result.P[0], P0)


def test_invalid_custom_initial_shapes_return_invalid_input():
    measurement = Measurement(t=0.0, sensor="gps", z=np.zeros(2), R=np.eye(2))
    result_x = run_filter([measurement], x0=np.zeros(4))
    result_p = run_filter([measurement], P0=np.eye(4))
    assert result_x.status == FilterStatus.INVALID_INPUT
    assert result_p.status == FilterStatus.INVALID_INPUT
    assert result_x.diagnostics.errors
    assert result_p.diagnostics.errors


def test_runner_reports_numerical_failure(monkeypatch):
    from roboqa.fusion.ekf import EKFNumericalError

    def fail_update(self, measurement):
        raise EKFNumericalError("injected numerical failure")

    monkeypatch.setattr("roboqa.fusion.runner.EKF.update", fail_update)
    measurement = Measurement(t=0.0, sensor="gps", z=np.zeros(2), R=np.eye(2))
    result = run_filter([measurement])
    assert result.status == FilterStatus.NUMERICAL_FAILURE
    assert result.diagnostics.n_update == 0
    assert "injected numerical failure" in result.diagnostics.errors[0]


def test_runner_reports_schema_failure_during_update(monkeypatch):
    from roboqa.data.schemas import SchemaError

    def fail_update(self, measurement):
        raise SchemaError("injected schema failure")

    monkeypatch.setattr("roboqa.fusion.runner.EKF.update", fail_update)
    measurement = Measurement(t=0.0, sensor="gps", z=np.zeros(2), R=np.eye(2))
    result = run_filter([measurement])
    assert result.status == FilterStatus.INVALID_INPUT
    assert result.diagnostics.n_update == 0
    assert "injected schema failure" in result.diagnostics.errors[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gps_sigma": -1.0},
        {"speed_sigma": np.nan},
        {"yawrate_sigma": np.inf},
    ],
)
def test_simulation_rejects_invalid_noise_parameters(kwargs):
    gt = make_synthetic_ground_truth(n=3, dt=0.1, seed=1)
    with pytest.raises(SchemaError):
        simulate_measurements(gt, **kwargs)


@pytest.mark.parametrize("gps_every", [0, -1, 1.5])
def test_simulation_rejects_invalid_gps_interval(gps_every):
    gt = make_synthetic_ground_truth(n=3, dt=0.1, seed=1)
    with pytest.raises(SchemaError, match="gps_every"):
        simulate_measurements(gt, gps_every=gps_every)

