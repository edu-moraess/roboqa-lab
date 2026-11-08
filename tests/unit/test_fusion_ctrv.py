"""Unit tests — CTRV model, Jacobian and discrete process covariance."""

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


@pytest.mark.parametrize("dt", [-0.1, np.nan, np.inf, -np.inf])
def test_ctrv_rejects_invalid_dt(dt):
    with pytest.raises(ValueError):
        ctrv_f(np.zeros(5), dt)
    with pytest.raises(ValueError):
        ctrv_jacobian(np.zeros(5), dt)


def test_ctrv_rejects_nonfinite_state():
    state = np.zeros(5)
    state[2] = np.nan
    with pytest.raises(ValueError):
        ctrv_f(state, 0.1)


def test_ctrv_continuity_near_eps():
    x_lo = np.array([0.0, 0.0, 0.3, 4.0, OMEGA_EPS * 0.999])
    x_hi = x_lo.copy()
    x_hi[4] = OMEGA_EPS * 1.001
    y_lo = ctrv_f(x_lo, 0.1)
    y_hi = ctrv_f(x_hi, 0.1)
    assert np.linalg.norm(y_lo[:3] - y_hi[:3]) < 1e-7


def _numerical_jacobian(x, dt, eps=1e-7):
    F = np.zeros((5, 5))
    for i in range(5):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        delta = ctrv_f(xp, dt) - ctrv_f(xm, dt)
        delta[2] = normalize_angle(delta[2])
        F[:, i] = delta / (2 * eps)
    return F


@pytest.mark.parametrize("omega", [0.0, 1e-6, OMEGA_EPS * 0.99, OMEGA_EPS * 1.01, 0.05, -0.1, 0.3])
@pytest.mark.parametrize("dt", [0.01, 0.1, 0.35])
def test_jacobian_finite_differences(omega, dt):
    x = np.array([1.0, -2.0, 0.4, 3.0, omega])
    F_an = ctrv_jacobian(x, dt)
    F_num = _numerical_jacobian(x, dt)
    np.testing.assert_allclose(F_an, F_num, atol=2e-5, rtol=2e-4)


def test_process_noise_Q_shape_symmetric_psd():
    Q = process_noise_Q(0.1, sigma_a=0.5, sigma_alpha=0.1, psi=0.3)
    assert Q.shape == (5, 5)
    np.testing.assert_allclose(Q, Q.T, atol=1e-14)
    eig = np.linalg.eigvalsh(Q)
    assert eig.min() >= -1e-12


def test_process_noise_Q_contains_physical_cross_terms():
    dt = 0.2
    sigma_a = 0.5
    Q = process_noise_Q(dt, sigma_a=sigma_a, sigma_alpha=0.0, psi=0.0)
    np.testing.assert_allclose(Q[0, 0], sigma_a**2 * dt**4 / 4)
    np.testing.assert_allclose(Q[0, 3], sigma_a**2 * dt**3 / 2)
    np.testing.assert_allclose(Q[3, 0], Q[0, 3])
    np.testing.assert_allclose(Q[3, 3], sigma_a**2 * dt**2)
    assert Q[1, 1] == 0.0


def test_process_noise_Q_rotates_with_heading():
    Q_x = process_noise_Q(0.2, sigma_a=0.5, sigma_alpha=0.0, psi=0.0)
    Q_y = process_noise_Q(0.2, sigma_a=0.5, sigma_alpha=0.0, psi=np.pi / 2)
    assert Q_x[0, 0] > 0.0
    assert Q_y[1, 1] > 0.0
    assert abs(Q_y[0, 0]) < 1e-15
    assert abs(Q_y[1, 3]) < 1e-15


def test_process_noise_Q_angular_cross_terms():
    dt = 0.2
    sigma_alpha = 0.1
    Q = process_noise_Q(dt, sigma_a=0.0, sigma_alpha=sigma_alpha)
    np.testing.assert_allclose(Q[2, 2], sigma_alpha**2 * dt**4 / 4)
    np.testing.assert_allclose(Q[2, 4], sigma_alpha**2 * dt**3 / 2)
    np.testing.assert_allclose(Q[4, 4], sigma_alpha**2 * dt**2)


def test_process_noise_Q_zero_dt():
    Q = process_noise_Q(0.0)
    np.testing.assert_array_equal(Q, np.zeros((5, 5)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": np.nan},
        {"dt": np.inf},
        {"dt": -0.1},
        {"sigma_a": np.nan},
        {"sigma_a": -0.1},
        {"sigma_alpha": np.inf},
        {"sigma_alpha": -0.1},
        {"psi": np.nan},
    ],
)
def test_process_noise_Q_rejects_invalid_parameters(kwargs):
    params = {"dt": 0.1, "sigma_a": 0.5, "sigma_alpha": 0.1, "psi": 0.0}
    params.update(kwargs)
    with pytest.raises(ValueError):
        process_noise_Q(**params)
