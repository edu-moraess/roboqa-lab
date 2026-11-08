"""
Extended Kalman Filter with CTRV motion model.
Fase 2 — Fusion Engine
ARQTECH LABS

Uses Joseph-form covariance update. predict/update validate all candidate
values before committing state, covariance, or time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from roboqa.data.schemas import Measurement, SchemaError
from roboqa.fusion.models.ctrv import (
    ctrv_f,
    ctrv_jacobian,
    normalize_angle,
    process_noise_Q,
)
from roboqa.fusion.sensors import get_sensor_model, validate_measurement_for_fusion


class EKFNumericalError(RuntimeError):
    """Raised when a numerical failure is detected."""


def _validate_covariance(P: np.ndarray, name: str) -> np.ndarray:
    """Return a symmetric candidate covariance or raise without mutating state."""
    candidate = np.asarray(P, dtype=np.float64)
    if candidate.shape != (5, 5) or not np.all(np.isfinite(candidate)):
        raise EKFNumericalError(f"{name} must be a finite 5x5 matrix.")
    try:
        scale = max(1.0, float(np.linalg.norm(candidate, ord=2)))
        symmetry_tol = 1e-10 * scale
        if not np.allclose(candidate, candidate.T, rtol=0.0, atol=symmetry_tol):
            raise EKFNumericalError(f"{name} is not symmetric within tolerance.")
        symmetric = 0.5 * (candidate + candidate.T)
        min_eig = float(np.linalg.eigvalsh(symmetric).min())
    except np.linalg.LinAlgError as exc:
        raise EKFNumericalError(f"Could not validate {name}: {exc}") from exc
    psd_tol = 1e-10 * scale
    if min_eig < -psd_tol:
        raise EKFNumericalError(
            f"{name} is not PSD (minimum eigenvalue={min_eig:.3e}, tolerance={psd_tol:.3e})."
        )
    return symmetric


class EKF:
    """Extended Kalman Filter (CTRV). State: [x, y, psi, v, omega]"""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        sigma_a: float = 0.5,
        sigma_alpha: float = 0.1,
    ) -> None:
        x0 = np.asarray(x0, dtype=np.float64).copy()
        P0 = np.asarray(P0, dtype=np.float64).copy()
        if x0.shape != (5,):
            raise SchemaError(f"x0 deve ter shape (5,), recebido {x0.shape}")
        if P0.shape != (5, 5):
            raise SchemaError(f"P0 deve ser (5,5), recebido {P0.shape}")
        if not np.all(np.isfinite(x0)):
            raise SchemaError("x0 contém valores não-finitos.")
        if not np.all(np.isfinite(P0)):
            raise SchemaError("P0 contém valores não-finitos.")
        if not np.allclose(P0, P0.T, rtol=0.0, atol=1e-8):
            raise SchemaError("P0 não é simétrica.")
        try:
            eig = np.linalg.eigvalsh(0.5 * (P0 + P0.T))
        except np.linalg.LinAlgError as exc:
            raise SchemaError(f"Não foi possível validar P0: {exc}") from exc
        if np.any(eig < -1e-8):
            raise SchemaError(f"P0 não é PSD (min eig={eig.min():.3e})")

        sigma_a = float(sigma_a)
        sigma_alpha = float(sigma_alpha)
        if not np.isfinite(sigma_a) or sigma_a < 0.0:
            raise SchemaError("sigma_a deve ser finito e >= 0.")
        if not np.isfinite(sigma_alpha) or sigma_alpha < 0.0:
            raise SchemaError("sigma_alpha deve ser finito e >= 0.")

        x0[2] = float(normalize_angle(x0[2]))
        self.x = x0
        self.P = 0.5 * (P0 + P0.T)
        self.sigma_a = sigma_a
        self.sigma_alpha = sigma_alpha
        self._t: Optional[float] = None

    @property
    def t(self) -> Optional[float]:
        return self._t

    def set_time(self, t: float) -> None:
        value = float(t)
        if not np.isfinite(value):
            raise SchemaError(f"timestamp não-finito: {t}")
        self._t = value

    def predict(self, t_next: float) -> None:
        """Propagate state and covariance to t_next (atomic)."""
        if self._t is None:
            raise SchemaError("EKF sem tempo inicial; chame set_time() antes de predict.")
        value = float(t_next)
        if not np.isfinite(value):
            raise SchemaError(f"t_next não-finito: {t_next}")
        dt = value - float(self._t)
        if dt < 0.0:
            raise SchemaError(
                f"predict com dt negativo ({dt:.6g}). "
                "Eventos fora de ordem não são suportados sem reprocessamento."
            )
        if dt == 0.0:
            return

        F = ctrv_jacobian(self.x, dt)
        Q = process_noise_Q(dt, self.sigma_a, self.sigma_alpha, psi=float(self.x[2]))
        x_new = ctrv_f(self.x, dt)
        P_new = _validate_covariance(F @ self.P @ F.T + Q, "P após predict")
        if not np.all(np.isfinite(x_new)):
            raise EKFNumericalError("Estado não-finito após predict.")

        self.x = x_new
        self.P = P_new
        self._t = value

    def update(self, measurement: Measurement) -> None:
        """Apply a measurement at the current filter timestamp (atomic)."""
        validate_measurement_for_fusion(measurement)
        if self._t is None:
            raise SchemaError("EKF sem tempo inicial; chame set_time() antes de update.")
        if abs(float(measurement.t) - self._t) > 1e-12:
            raise SchemaError(
                f"Timestamp da medição ({measurement.t:.12g}) difere do tempo "
                f"do filtro ({self._t:.12g}); execute predict até o evento primeiro."
            )

        h_fn, H_fn, _ = get_sensor_model(measurement.sensor)
        z = np.array(measurement.z, dtype=np.float64, copy=True)
        R = np.array(measurement.R, dtype=np.float64, copy=True)
        x_pred = self.x.copy()
        P_pred = self.P.copy()
        H = H_fn(x_pred)
        z_hat = h_fn(x_pred)
        y = z - z_hat

        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)
        if not np.all(np.isfinite(S)):
            raise EKFNumericalError("Matriz de inovação não-finita.")
        try:
            PHt = P_pred @ H.T
            K = np.linalg.solve(S, PHt.T).T
        except np.linalg.LinAlgError as exc:
            raise EKFNumericalError(
                f"Matriz de inovação singular/mal condicionada (sensor={measurement.sensor}): {exc}"
            ) from exc

        x_new = x_pred + K @ y
        x_new[2] = float(normalize_angle(x_new[2]))
        eye = np.eye(5)
        KH = K @ H
        I_KH = eye - KH
        P_new = _validate_covariance(
            I_KH @ P_pred @ I_KH.T + K @ R @ K.T,
            "P após update",
        )
        if not np.all(np.isfinite(x_new)):
            raise EKFNumericalError("Estado não-finito após update.")

        self.x = x_new
        self.P = P_new
