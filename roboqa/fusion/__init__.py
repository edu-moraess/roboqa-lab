"""RoboQA Fusion Engine — EKF-CTRV multi-sensor fusion."""

from roboqa.fusion.ekf import EKF, EKFNumericalError
from roboqa.fusion.models.ctrv import (
    OMEGA_EPS,
    ctrv_f,
    ctrv_jacobian,
    normalize_angle,
    process_noise_Q,
)
from roboqa.fusion.result import (
    FilterDiagnostics,
    FilterResult,
    FilterStatus,
    InnovationTelemetry,
)
from roboqa.fusion.runner import run_filter, simulate_measurements
from roboqa.fusion.sensors import get_sensor_model, validate_measurement_for_fusion

__all__ = [
    "EKF",
    "EKFNumericalError",
    "OMEGA_EPS",
    "ctrv_f",
    "ctrv_jacobian",
    "normalize_angle",
    "process_noise_Q",
    "FilterDiagnostics",
    "FilterResult",
    "FilterStatus",
    "InnovationTelemetry",
    "run_filter",
    "simulate_measurements",
    "get_sensor_model",
    "validate_measurement_for_fusion",
]
