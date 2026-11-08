"""
CTRV — Constant Turn Rate and Velocity motion model.
Fase 2 — Fusion Engine
ARQTECH LABS

State: [x, y, psi, v, omega]^T
Units: m, m, rad, m/s, rad/s
Angle convention: psi normalized to [-pi, pi).
"""

from __future__ import annotations

import numpy as np

# Threshold below which the series expansion about omega=0 is used (rad/s).
OMEGA_EPS = 1e-5


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Normalize angle(s) to [-pi, pi)."""
    a = np.asarray(angle, dtype=np.float64)
    out = (a + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(angle):
        return float(out)
    return out


def _validated_state_dt(x: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
    state = np.asarray(x, dtype=np.float64)
    if state.shape != (5,):
        raise ValueError(f"state must have shape (5,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("state must contain only finite values")
    step = float(dt)
    if not np.isfinite(step) or step < 0.0:
        raise ValueError(f"dt must be finite and >= 0, got {dt}")
    return state, step


def ctrv_f(x: np.ndarray, dt: float) -> np.ndarray:
    """
    CTRV state transition.

    For |omega| > OMEGA_EPS uses the exact circular-arc formula.
    Near zero turn rate, a second-order Taylor expansion avoids cancellation.
    """
    x, dt = _validated_state_dt(x, dt)
    px, py, psi, v, omega = x
    psi = float(normalize_angle(psi))

    if abs(omega) > OMEGA_EPS:
        dpsi = omega * dt
        px_n = px + (v / omega) * (np.sin(psi + dpsi) - np.sin(psi))
        py_n = py + (v / omega) * (np.cos(psi) - np.cos(psi + dpsi))
    else:
        s, c = np.sin(psi), np.cos(psi)
        px_n = px + v * dt * c - v * omega * (dt ** 2) / 2.0 * s
        py_n = py + v * dt * s + v * omega * (dt ** 2) / 2.0 * c

    psi_n = float(normalize_angle(psi + omega * dt))
    return np.array([px_n, py_n, psi_n, v, omega], dtype=np.float64)


def ctrv_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    """Analytic Jacobian F = df/dx matching ctrv_f branching."""
    x, dt = _validated_state_dt(x, dt)
    _, _, psi, v, omega = x
    psi = float(normalize_angle(psi))
    F = np.eye(5, dtype=np.float64)

    if abs(omega) > OMEGA_EPS:
        dpsi = omega * dt
        s, c = np.sin(psi), np.cos(psi)
        s2, c2 = np.sin(psi + dpsi), np.cos(psi + dpsi)
        inv_w = 1.0 / omega

        F[0, 2] = v * inv_w * (c2 - c)
        F[1, 2] = v * inv_w * (s2 - s)
        F[0, 3] = inv_w * (s2 - s)
        F[1, 3] = inv_w * (c - c2)
        F[0, 4] = -v * (inv_w ** 2) * (s2 - s) + v * inv_w * (c2 * dt)
        F[1, 4] = -v * (inv_w ** 2) * (c - c2) + v * inv_w * (s2 * dt)
        F[2, 4] = dt
    else:
        s, c = np.sin(psi), np.cos(psi)
        F[0, 2] = -v * dt * s - v * omega * (dt ** 2) / 2.0 * c
        F[1, 2] = v * dt * c - v * omega * (dt ** 2) / 2.0 * s
        F[0, 3] = dt * c - omega * (dt ** 2) / 2.0 * s
        F[1, 3] = dt * s + omega * (dt ** 2) / 2.0 * c
        F[0, 4] = -v * (dt ** 2) / 2.0 * s
        F[1, 4] = v * (dt ** 2) / 2.0 * c
        F[2, 4] = dt

    return F


def process_noise_Q(
    dt: float,
    sigma_a: float = 0.5,
    sigma_alpha: float = 0.1,
    psi: float = 0.0,
) -> np.ndarray:
    """
    Discrete process covariance for interval-constant random accelerations.

    Model: during one interval dt, longitudinal acceleration a and angular
    acceleration alpha are independent, zero-mean random variables held
    constant over that interval, with standard deviations sigma_a [m/s^2]
    and sigma_alpha [rad/s^2]. The heading psi is the interval's initial
    heading. This is a sampled/zero-order-hold model, NOT continuous white
    acceleration noise.

    With u=[cos(psi), sin(psi)], the acceleration-to-state map is:
      delta_p = 0.5 * a * dt^2 * u; delta_v = a * dt
      delta_psi = 0.5 * alpha * dt^2; delta_omega = alpha * dt

    Q is B diag(sigma_a^2, sigma_alpha^2) B.T, so it is PSD by construction
    (up to floating-point roundoff) and includes position/velocity and
    heading/yaw-rate cross-covariances.
    """
    dt = float(dt)
    sigma_a = float(sigma_a)
    sigma_alpha = float(sigma_alpha)
    psi = float(psi)

    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError(f"dt must be finite and >= 0, got {dt}")
    if not np.isfinite(sigma_a) or sigma_a < 0.0:
        raise ValueError("sigma_a must be finite and >= 0")
    if not np.isfinite(sigma_alpha) or sigma_alpha < 0.0:
        raise ValueError("sigma_alpha must be finite and >= 0")
    if not np.isfinite(psi):
        raise ValueError("psi must be finite")
    if dt == 0.0 or (sigma_a == 0.0 and sigma_alpha == 0.0):
        return np.zeros((5, 5), dtype=np.float64)

    c, s = np.cos(psi), np.sin(psi)
    u = np.array([c, s], dtype=np.float64)
    B = np.zeros((5, 2), dtype=np.float64)
    B[0:2, 0] = 0.5 * (dt ** 2) * u
    B[3, 0] = dt
    B[2, 1] = 0.5 * (dt ** 2)
    B[4, 1] = dt

    variances = np.diag([sigma_a ** 2, sigma_alpha ** 2])
    Q = B @ variances @ B.T
    return 0.5 * (Q + Q.T)
