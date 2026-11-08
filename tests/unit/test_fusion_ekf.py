"""Unit tests — EKF predict/update and numerical invariants."""

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


def _snapshot(ekf):
    return ekf.x.copy(), ekf.P.copy(), ekf.t


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


def test_update_rejects_timestamp_mismatch_without_mutation():
    ekf = _ekf()
    before = _snapshot(ekf)
    m = Measurement(t=0.1, sensor="gps", z=np.array([0.5, 0.1]), R=np.eye(2))
    with pytest.raises(SchemaError, match="Timestamp"):
        ekf.update(m)
    np.testing.assert_array_equal(ekf.x, before[0])
    np.testing.assert_array_equal(ekf.P, before[1])
    assert ekf.t == before[2]


def test_update_requires_initialized_time():
    ekf = EKF(np.zeros(5), np.eye(5))
    m = Measurement(t=0.0, sensor="gps", z=np.zeros(2), R=np.eye(2))
    with pytest.raises(SchemaError, match="tempo inicial"):
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
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-10)
    eig = np.linalg.eigvalsh(ekf.P)
    assert np.all(eig >= -1e-10)


def test_predict_rejects_non_psd_candidate_atomically():
    ekf = _ekf()
    ekf.P = np.diag([1.0, 1.0, 1.0, 1.0, -10.0])
    before = _snapshot(ekf)
    with pytest.raises(EKFNumericalError, match="not PSD"):
        ekf.predict(0.1)
    np.testing.assert_array_equal(ekf.x, before[0])
    np.testing.assert_array_equal(ekf.P, before[1])
    assert ekf.t == before[2]


def test_update_rejects_non_psd_candidate_atomically():
    ekf = _ekf()
    ekf.P = np.diag([1.0, 1.0, 1.0, 1.0, -10.0])
    before = _snapshot(ekf)
    m = Measurement(t=0.0, sensor="gps", z=np.array([1.0, 2.0]), R=np.eye(2))
    with pytest.raises(EKFNumericalError, match="not PSD"):
        ekf.update(m)
    np.testing.assert_array_equal(ekf.x, before[0])
    np.testing.assert_array_equal(ekf.P, before[1])
    assert ekf.t == before[2]


def test_invalid_P0_raises():
    with pytest.raises(SchemaError):
        EKF(np.zeros(5), np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_does_not_mutate_measurement():
    ekf = _ekf()
    z = np.array([1.0, 2.0])
    R = np.eye(2)
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


@pytest.mark.parametrize("sigma_a,sigma_alpha", [(-1.0, 0.1), (0.5, -0.1), (np.nan, 0.1), (0.5, np.inf)])
def test_invalid_process_noise_parameters_rejected(sigma_a, sigma_alpha):
    with pytest.raises(SchemaError):
        EKF(np.zeros(5), np.eye(5), sigma_a=sigma_a, sigma_alpha=sigma_alpha)

def test_constructor_rejects_nonfinite_state_and_covariance():
    x0 = np.zeros(5)
    x0[0] = np.nan
    with pytest.raises(SchemaError, match="x0 contém"):
        EKF(x0, np.eye(5))

    P0 = np.eye(5)
    P0[0, 0] = np.inf
    with pytest.raises(SchemaError, match="P0 contém"):
        EKF(np.zeros(5), P0)


def test_constructor_rejects_indefinite_initial_covariance():
    P0 = np.eye(5)
    P0[0, 0] = -0.1
    with pytest.raises(SchemaError, match="não é PSD"):
        EKF(np.zeros(5), P0)


def test_set_time_rejects_nonfinite_timestamp():
    ekf = EKF(np.zeros(5), np.eye(5))
    with pytest.raises(SchemaError, match="timestamp não-finito"):
        ekf.set_time(np.nan)
    assert ekf.t is None


@pytest.mark.parametrize("t_next", [np.nan, np.inf, -np.inf])
def test_predict_rejects_nonfinite_timestamp_without_mutation(t_next):
    ekf = _ekf()
    before = _snapshot(ekf)
    with pytest.raises(SchemaError, match="t_next não-finito"):
        ekf.predict(t_next)
    np.testing.assert_array_equal(ekf.x, before[0])
    np.testing.assert_array_equal(ekf.P, before[1])
    assert ekf.t == before[2]


def test_predict_same_timestamp_is_noop():
    ekf = _ekf()
    before = _snapshot(ekf)
    ekf.predict(0.0)
    np.testing.assert_array_equal(ekf.x, before[0])
    np.testing.assert_array_equal(ekf.P, before[1])
    assert ekf.t == before[2]


def test_update_singular_innovation_is_atomic():
    ekf = EKF(np.zeros(5), np.zeros((5, 5)))
    ekf.set_time(0.0)
    before = _snapshot(ekf)
    measurement = Measurement(t=0.0, sensor="gps", z=np.ones(2), R=np.zeros((2, 2)))
    with pytest.raises(EKFNumericalError, match="singular/mal condicionada"):
        ekf.update(measurement)
    np.testing.assert_array_equal(ekf.x, before[0])
    np.testing.assert_array_equal(ekf.P, before[1])
    assert ekf.t == before[2]

