"""
Extended Kalman Filter with CTRV motion model.
Fase 2 — Fusion Engine
ARQTECH LABS

Uses Joseph-form covariance update for numerical robustness.
predict/update are atomic: state is committed only after full validation.
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
        if not np.allclose(P0, P0.T, atol=1e-8):
            raise SchemaError("P0 não é simétrica.")
        eig = np.linalg.eigvalsh((P0 + P0.T) / 2.0)
        if np.any(eig < -1e-8):
            raise SchemaError(f"P0 não é PSD (min eig={eig.min():.3e})")

        x0[2] = float(normalize_angle(x0[2]))
        self.x = x0
        self.P = P0
        self.sigma_a = float(sigma_a)
        self.sigma_alpha = float(sigma_alpha)
        self._t: Optional[float] = None

    @property
    def t(self) -> Optional[float]:
        return self._t

    def set_time(self, t: float) -> None:
        if not np.isfinite(t):
            raise SchemaError(f"timestamp não-finito: {t}")
        self._t = float(t)

    def predict(self, t_next: float) -> None:
        """Propagate state and covariance to t_next (atomic)."""
        if self._t is None:
            raise SchemaError("EKF sem tempo inicial; chame set_time() antes de predict.")
        if not np.isfinite(t_next):
            raise SchemaError(f"t_next não-finito: {t_next}")
        dt = float(t_next) - float(self._t)
        if dt < 0:
            raise SchemaError(
                f"predict com dt negativo ({dt:.6g}). "
                "Eventos fora de ordem não são suportados sem reprocessamento."
            )
        if dt == 0.0:
            return

        F = ctrv_jacobian(self.x, dt)
        Q = process_noise_Q(dt, self.sigma_a, self.sigma_alpha)
        x_new = ctrv_f(self.x, dt)
        P_new = F @ self.P @ F.T + Q
        P_new = 0.5 * (P_new + P_new.T)
        if not np.all(np.isfinite(x_new)):
            raise EKFNumericalError("Estado não-finito após predict.")
        if not np.all(np.isfinite(P_new)):
            raise EKFNumericalError("Covariância não-finita após predict.")
        self.x = x_new
        self.P = P_new
        self._t = float(t_next)

    def update(self, measurement: Measurement) -> None:
        """Measurement update (atomic)."""
        validate_measurement_for_fusion(measurement)
        h_fn, H_fn, _ = get_sensor_model(measurement.sensor)

        z = np.array(measurement.z, dtype=np.float64, copy=True)
        R = np.array(measurement.R, dtype=np.float64, copy=True)

        x_pred = self.x
        P_pred = self.P
        H = H_fn(x_pred)
        z_hat = h_fn(x_pred)
        y = z - z_hat

        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)

        try:
            PHt = P_pred @ H.T
            K = np.linalg.solve(S, PHt.T).T
        except np.linalg.LinAlgError as e:
            raise EKFNumericalError(
                f"Matriz de inovação singular/mal condicionada (sensor={measurement.sensor}): {e}"
            ) from e

        x_new = x_pred + K @ y
        x_new = x_new.copy()
        x_new[2] = float(normalize_angle(x_new[2]))

        eye = np.eye(5)
        KH = K @ H
        I_KH = eye - KH
        P_new = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        P_new = 0.5 * (P_new + P_new.T)

        if not np.all(np.isfinite(x_new)):
            raise EKFNumericalError("Estado não-finito após update.")
        if not np.all(np.isfinite(P_new)):
            raise EKFNumericalError("Covariância não-finita após update.")

        self.x = x_new
        self.P = P_new
