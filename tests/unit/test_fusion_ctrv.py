"""Unit tests — CTRV model and Jacobian."""

import numpy as np
import pytest

from roboqa.fusion.models.ctrv import (
    OMEGA_EPS,
    ctrv_f,
    ctrv_jacobian,
    normalize_angle,
    process_noise_Q,
)


def test_normalize_angle_scalar():
    assert abs(normalize_angle(0.0)) < 1e-15
    a = normalize_angle(3 * np.pi)
    assert -np.pi <= a < np.pi


def test_normalize_angle_array():
    a = normalize_angle(np.array([-3 * np.pi, 0.0, 3 * np.pi]))
    assert a.shape == (3,)
    assert np.all((a >= -np.pi) & (a < np.pi))


def test_ctrv_rectilinear():
    x = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    xn = ctrv_f(x, 0.1)
    np.testing.assert_allclose(xn[:2], [0.5, 0.0], atol=1e-10)


def test_ctrv_curve():
    x = np.array([0.0, 0.0, 0.0, 5.0, 0.2])
    xn = ctrv_f(x, 0.1)
    assert xn.shape == (5,)
    assert xn[0] > 0


def test_ctrv_negative_omega():
    x = np.array([0.0, 0.0, 0.0, 5.0, -0.2])
    xn = ctrv_f(x, 0.1)
    assert np.all(np.isfinite(xn))


def test_ctrv_zero_velocity():
    x = np.array([1.0, 2.0, 0.5, 0.0, 0.3])
    xn = ctrv_f(x, 0.2)
    np.testing.assert_allclose(xn[0], 1.0, atol=1e-10)


def test_ctrv_negative_dt_raises():
    with pytest.raises(ValueError):
        ctrv_f(np.zeros(5), -0.1)


def test_ctrv_continuity_near_eps():
    x = np.array([0.0, 0.0, 0.3, 4.0, OMEGA_EPS * 0.5])
    x_lo = ctrv_f(x, 0.1)
    x2 = x.copy()
    x2[4] = OMEGA_EPS * 1.5
    x_hi = ctrv_f(x2, 0.1)
    assert np.linalg.norm(x_lo[:2] - x_hi[:2]) < 0.05


def _numerical_jacobian(x, dt, eps=1e-6):
    F = np.zeros((5, 5))
    for i in range(5):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        F[:, i] = (ctrv_f(xp, dt) - ctrv_f(xm, dt)) / (2 * eps)
    return F


@pytest.mark.parametrize("omega", [0.0, 1e-6, 0.05, -0.1, 0.3])
@pytest.mark.parametrize("dt", [0.01, 0.1, 0.35])
def test_jacobian_finite_differences(omega, dt):
    x = np.array([1.0, -2.0, 0.4, 3.0, omega])
    F_an = ctrv_jacobian(x, dt)
    F_num = _numerical_jacobian(x, dt)
    np.testing.assert_allclose(F_an, F_num, atol=1e-4, rtol=1e-3)


def test_process_noise_Q_shape_psd():
    Q = process_noise_Q(0.1)
    assert Q.shape == (5, 5)
    assert np.allclose(Q, Q.T)
    eig = np.linalg.eigvalsh(Q)
    assert np.all(eig >= -1e-10)


def test_process_noise_Q_zero_dt():
    Q = process_noise_Q(0.0)
    np.testing.assert_array_equal(Q, np.zeros((5, 5)))
