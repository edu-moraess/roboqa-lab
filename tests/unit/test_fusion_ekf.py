"""Unit tests — EKF predict/update."""

import numpy as np
import pytest

from roboqa.data.schemas import Measurement, SchemaError
from roboqa.fusion.ekf import EKF, EKFNumericalError


def _ekf():
    x0 = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    P0 = np.eye(5) * 1.0
    ekf = EKF(x0, P0, sigma_a=0.5, sigma_alpha=0.1)
    ekf.set_time(0.0)
    return ekf


def test_predict_moves_state():
    ekf = _ekf()
    ekf.predict(0.1)
    assert ekf.x[0] > 0
    assert ekf.t == 0.1


def test_predict_negative_dt_raises():
    ekf = _ekf()
    ekf.predict(0.2)
    with pytest.raises(SchemaError):
        ekf.predict(0.1)


def test_update_gps():
    ekf = _ekf()
    m = Measurement(t=0.0, sensor="gps", z=np.array([0.5, 0.1]), R=np.eye(2) * 2.25)
    ekf.update(m)
    assert np.all(np.isfinite(ekf.x))


def test_update_speed():
    ekf = _ekf()
    m = Measurement(t=0.0, sensor="speed", z=np.array([4.0]), R=np.array([[0.04]]))
    ekf.update(m)
    assert np.isfinite(ekf.x[3])


def test_update_yaw_rate():
    ekf = _ekf()
    m = Measurement(t=0.0, sensor="yaw_rate", z=np.array([0.1]), R=np.array([[0.001]]))
    ekf.update(m)
    assert np.isfinite(ekf.x[4])


def test_unknown_sensor_raises():
    ekf = _ekf()
    m = Measurement(t=0.0, sensor="lidar", z=np.array([1.0]), R=np.array([[1.0]]))
    with pytest.raises(SchemaError):
        ekf.update(m)


def test_covariance_stays_symmetric_psd():
    ekf = _ekf()
    for t in [0.1, 0.2, 0.3]:
        ekf.predict(t)
        m = Measurement(
            t=t, sensor="gps",
            z=np.array([ekf.x[0] + 0.1, ekf.x[1] - 0.1]),
            R=np.eye(2) * 1.0,
        )
        ekf.update(m)
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-8)
    eig = np.linalg.eigvalsh(ekf.P)
    assert np.all(eig >= -1e-6)


def test_invalid_P0_raises():
    with pytest.raises(SchemaError):
        EKF(np.zeros(5), np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_does_not_mutate_measurement():
    ekf = _ekf()
    z = np.array([1.0, 2.0])
    R = np.eye(2) * 1.0
    m = Measurement(t=0.0, sensor="gps", z=z, R=R)
    z_before = m.z.copy()
    ekf.update(m)
    np.testing.assert_array_equal(m.z, z_before)
    assert not m.z.flags.writeable


def test_predict_without_time_raises():
    ekf = EKF(np.zeros(5), np.eye(5))
    with pytest.raises(SchemaError):
        ekf.predict(0.1)


def test_x0_wrong_shape():
    with pytest.raises(SchemaError):
        EKF(np.zeros(4), np.eye(5))


def test_P0_nonsymmetric():
    P = np.eye(5)
    P[0, 1] = 0.5
    with pytest.raises(SchemaError):
        EKF(np.zeros(5), P)
