"""Unit and adversarial tests for immutable EKF innovation telemetry."""

import numpy as np
import pytest

from roboqa.data.schemas import Measurement
from roboqa.fusion import InnovationTelemetry, run_filter
from roboqa.fusion.ekf import EKF


def test_gps_update_returns_correct_innovation_and_nis():
    x0 = np.array([1.0, -2.0, 0.1, 4.0, 0.02])
    P0 = np.diag([2.0, 3.0, 0.5, 1.0, 0.2])
    R = np.array([[0.5, 0.1], [0.1, 0.8]])
    measurement = Measurement(t=3.5, sensor="gps", z=np.array([2.5, -1.0]), R=R)
    ekf = EKF(x0, P0)
    ekf.set_time(3.5)
    expected_y = measurement.z - x0[:2]
    H = np.zeros((2, 5))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    expected_S = H @ P0 @ H.T + R
    expected_nis = float(expected_y @ np.linalg.solve(expected_S, expected_y))

    record = ekf.update(measurement)

    assert isinstance(record, InnovationTelemetry)
    assert (record.sensor_id, record.timestamp, record.measurement_dof) == ("gps", 3.5, 2)
    assert record.degrees_of_freedom == 2
    np.testing.assert_allclose(record.innovation, expected_y)
    np.testing.assert_allclose(record.innovation_covariance, expected_S)
    assert record.nis == pytest.approx(expected_nis)


@pytest.mark.parametrize(
    ("sensor", "z", "R", "dof"),
    [
        ("gps", np.array([1.0, 2.0]), np.eye(2), 2),
        ("speed", np.array([4.5]), np.array([[0.2]]), 1),
        ("yaw_rate", np.array([0.03]), np.array([[0.01]]), 1),
    ],
)
def test_telemetry_dof_matches_sensor_measurement_dimension(sensor, z, R, dof):
    ekf = EKF(np.array([0.0, 0.0, 0.0, 5.0, 0.0]), np.eye(5))
    ekf.set_time(0.0)
    record = ekf.update(Measurement(t=0.0, sensor=sensor, z=z, R=R))
    assert record.measurement_dof == dof
    assert record.innovation.shape == (dof,)
    assert record.innovation_covariance.shape == (dof, dof)
    assert np.isfinite(record.nis) and record.nis >= 0.0


def test_telemetry_arrays_are_defensive_and_read_only():
    y = np.array([1.0, -2.0])
    S = np.eye(2)
    record = InnovationTelemetry(1.0, "GPS", 2, y, S, 3.0)
    y[0] = 99.0
    S[0, 0] = 99.0
    assert record.sensor_id == "gps"
    assert record.innovation[0] == 1.0
    assert record.innovation_covariance[0, 0] == 1.0
    assert not record.innovation.flags.writeable
    assert not record.innovation_covariance.flags.writeable
    with pytest.raises(ValueError):
        record.innovation[0] = 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"measurement_dof": 0},
        {"measurement_dof": True},
        {"sensor_id": ""},
        {"timestamp": np.nan},
        {"nis": np.inf},
        {"nis": -1.0},
    ],
)
def test_telemetry_contract_rejects_invalid_metadata(kwargs):
    values = {
        "timestamp": 0.0,
        "sensor_id": "gps",
        "measurement_dof": 2,
        "innovation": np.zeros(2),
        "innovation_covariance": np.eye(2),
        "nis": 0.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        InnovationTelemetry(**values)


def test_telemetry_contract_rejects_shape_and_nonfinite_arrays():
    with pytest.raises(ValueError, match="shape"):
        InnovationTelemetry(0.0, "gps", 2, np.zeros(1), np.eye(2), 0.0)
    bad = np.eye(2)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        InnovationTelemetry(0.0, "gps", 2, np.zeros(2), bad, 0.0)


def test_runner_collects_one_record_per_successful_update_and_is_deterministic():
    measurements = [
        Measurement(0.0, "gps", np.array([0.0, 0.0]), np.eye(2)),
        Measurement(0.0, "speed", np.array([4.0]), np.array([[0.1]])),
        Measurement(0.1, "yaw_rate", np.array([0.02]), np.array([[0.01]])),
    ]
    first = run_filter(measurements)
    second = run_filter(measurements)
    assert len(first.telemetry) == first.diagnostics.n_update == 3
    assert [r.sensor_id for r in first.telemetry] == ["gps", "speed", "yaw_rate"]
    assert [r.timestamp for r in first.telemetry] == [0.0, 0.0, 0.1]
    np.testing.assert_allclose(
        [r.nis for r in first.telemetry], [r.nis for r in second.telemetry]
    )
    for left, right in zip(first.telemetry, second.telemetry):
        np.testing.assert_array_equal(left.innovation, right.innovation)
        np.testing.assert_array_equal(
            left.innovation_covariance, right.innovation_covariance
        )


def test_runner_keeps_prior_successful_telemetry_when_later_update_fails(monkeypatch):
    from roboqa.fusion.ekf import EKFNumericalError

    original_update = EKF.update
    calls = {"count": 0}

    def fail_second(self, measurement):
        calls["count"] += 1
        if calls["count"] == 2:
            raise EKFNumericalError("injected second-update failure")
        return original_update(self, measurement)

    monkeypatch.setattr(EKF, "update", fail_second)
    measurements = [
        Measurement(0.0, "gps", np.zeros(2), np.eye(2)),
        Measurement(0.1, "speed", np.array([4.0]), np.array([[0.1]])),
    ]
    result = run_filter(measurements)
    assert result.diagnostics.n_update == len(result.telemetry) == 1
    assert result.status.value == "numerical_failure"
