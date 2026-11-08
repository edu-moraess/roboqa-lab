"""
Observation models for GPS, speed and yaw-rate sensors.
Fase 2 — Fusion Engine
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from roboqa.data.schemas import Measurement, SchemaError


def h_gps(x: np.ndarray) -> np.ndarray:
    return np.array([x[0], x[1]], dtype=np.float64)


def H_gps(x: np.ndarray) -> np.ndarray:
    H = np.zeros((2, 5), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H


def h_speed(x: np.ndarray) -> np.ndarray:
    return np.array([x[3]], dtype=np.float64)


def H_speed(x: np.ndarray) -> np.ndarray:
    H = np.zeros((1, 5), dtype=np.float64)
    H[0, 3] = 1.0
    return H


def h_yaw_rate(x: np.ndarray) -> np.ndarray:
    return np.array([x[4]], dtype=np.float64)


def H_yaw_rate(x: np.ndarray) -> np.ndarray:
    H = np.zeros((1, 5), dtype=np.float64)
    H[0, 4] = 1.0
    return H


_SENSOR_MODELS: dict[str, tuple[Callable, Callable, int]] = {
    "gps": (h_gps, H_gps, 2),
    "speed": (h_speed, H_speed, 1),
    "yaw_rate": (h_yaw_rate, H_yaw_rate, 1),
}


def get_sensor_model(sensor: str) -> tuple[Callable, Callable, int]:
    key = sensor.strip().lower()
    if key not in _SENSOR_MODELS:
        raise SchemaError(
            f"Sensor desconhecido: '{sensor}'. "
            f"Suportados: {sorted(_SENSOR_MODELS)}"
        )
    return _SENSOR_MODELS[key]


def validate_measurement_for_fusion(m: Measurement) -> None:
    """Validate measurement dimensions against known sensor models."""
    h, H, dim = get_sensor_model(m.sensor)
    if m.z.shape[0] != dim:
        raise SchemaError(
            f"Sensor '{m.sensor}': z dim {m.z.shape[0]} != esperado {dim}"
        )
    if m.R.shape != (dim, dim):
        raise SchemaError(
            f"Sensor '{m.sensor}': R shape {m.R.shape} != ({dim}, {dim})"
        )
