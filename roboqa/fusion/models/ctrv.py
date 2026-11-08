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

# Threshold below which the rectilinear limit is used (rad/s).
OMEGA_EPS = 1e-5


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Normalize angle(s) to [-pi, pi)."""
    a = np.asarray(angle, dtype=np.float64)
    out = (a + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(angle):
        return float(out)
    return out


def ctrv_f(x: np.ndarray, dt: float) -> np.ndarray:
    """CTRV state transition."""
    x = np.asarray(x, dtype=np.float64).reshape(5)
    dt = float(dt)
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt}")

    px, py, psi, v, omega = x
    psi = float(normalize_angle(psi))

    if abs(omega) > OMEGA_EPS:
        dpsi = omega * dt
        px_n = px + (v / omega) * (np.sin(psi + dpsi) - np.sin(psi))
        py_n = py + (v / omega) * (np.cos(psi) - np.cos(psi + dpsi))
        psi_n = float(normalize_angle(psi + dpsi))
    else:
        px_n = px + v * dt * np.cos(psi)
        py_n = py + v * dt * np.sin(psi)
        psi_n = psi

    return np.array([px_n, py_n, psi_n, v, omega], dtype=np.float64)


def ctrv_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    """Analytic Jacobian F = df/dx of the CTRV transition."""
    x = np.asarray(x, dtype=np.float64).reshape(5)
    dt = float(dt)
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt}")

    px, py, psi, v, omega = x
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
        F[0, 2] = -v * dt * s
        F[1, 2] = v * dt * c
        F[0, 3] = dt * c
        F[1, 3] = dt * s

    return F


def process_noise_Q(
    dt: float,
    sigma_a: float = 0.5,
    sigma_alpha: float = 0.1,
) -> np.ndarray:
    """Discrete process-noise covariance for CTRV (diagonal PSD)."""
    dt = float(dt)
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt}")
    if dt == 0.0:
        return np.zeros((5, 5), dtype=np.float64)

    sa2 = float(sigma_a) ** 2
    sal2 = float(sigma_alpha) ** 2

    q_v = sa2 * dt
    q_pos = sa2 * (dt ** 3) / 3.0
    q_w = sal2 * dt
    q_psi = sal2 * (dt ** 3) / 3.0

    Q = np.zeros((5, 5), dtype=np.float64)
    Q[0, 0] = q_pos
    Q[1, 1] = q_pos
    Q[2, 2] = q_psi
    Q[3, 3] = q_v
    Q[4, 4] = q_w
    return Q
