"""
Temporal orchestration of the EKF over asynchronous measurements.
Fase 2 — Fusion Engine; Phase 4.1 — Fusion Telemetry.

Policy:
- Measurements sorted by (t, sensor_name) for deterministic ordering.
- Out-of-order after start (t < current filter time) rejected with diagnosis.
- Original Measurement / GroundTruth objects are never mutated.
- Initialization uses a high-uncertainty prior (or caller-provided x0/P0).
  Every measurement is applied exactly once.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from roboqa.data.schemas import GroundTruth, Measurement, SchemaError
from roboqa.fusion.ekf import EKF, EKFNumericalError
from roboqa.fusion.result import (
    FilterDiagnostics,
    FilterResult,
    FilterStatus,
    InnovationTelemetry,
)
from roboqa.fusion.sensors import validate_measurement_for_fusion


def _default_x0_P0(
    measurements: Sequence[Measurement],
    x0: Optional[np.ndarray],
    P0: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build an initial prior without consuming any measurement."""
    warnings: list[str] = []
    x = np.zeros(5, dtype=np.float64)
    P = np.diag([25.0, 25.0, 0.5, 4.0, 0.25]).astype(np.float64)

    if not any(m.sensor.strip().lower() == "gps" for m in measurements):
        warnings.append(
            "Nenhum GPS na sequência; inicialização de posição em (0,0) com alta incerteza."
        )
    if x0 is not None:
        x = np.asarray(x0, dtype=np.float64).reshape(5)
    if P0 is not None:
        P = np.asarray(P0, dtype=np.float64).reshape(5, 5)
    return x, P, warnings


def _empty_result(status: FilterStatus, diag: FilterDiagnostics, provenance: dict, config: dict):
    return FilterResult(
        t=np.array([]),
        x=np.zeros((0, 5)),
        P=np.zeros((0, 5, 5)),
        status=status,
        diagnostics=diag,
        provenance=provenance,
        config=config,
    )


def run_filter(
    measurements: Sequence[Measurement],
    *,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    sigma_a: float = 0.5,
    sigma_alpha: float = 0.1,
    t0: Optional[float] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> FilterResult:
    diag = FilterDiagnostics()
    prov = dict(provenance or {})
    config = {"sigma_a": sigma_a, "sigma_alpha": sigma_alpha}

    if not measurements:
        return _empty_result(FilterStatus.INIT_INSUFFICIENT, diag, prov, config)

    for measurement in measurements:
        try:
            validate_measurement_for_fusion(measurement)
        except SchemaError as exc:
            diag.errors.append(str(exc))
            return _empty_result(FilterStatus.INVALID_INPUT, diag, prov, config)

    ordered = sorted(measurements, key=lambda m: (m.t, m.sensor))
    try:
        x_init, P_init, init_warnings = _default_x0_P0(ordered, x0, P0)
        diag.warnings.extend(init_warnings)
        start_t = float(ordered[0].t) if t0 is None else float(t0)
        if not np.isfinite(start_t):
            raise SchemaError("t0 deve ser finito.")
        if start_t > float(ordered[0].t) + 1e-12:
            raise SchemaError(
                "t0 não pode ser posterior à primeira medição; isso descartaria "
                "informação temporal anterior à inicialização."
            )
        ekf = EKF(x_init, P_init, sigma_a=sigma_a, sigma_alpha=sigma_alpha)
        ekf.set_time(start_t)
    except (SchemaError, ValueError, TypeError) as exc:
        diag.errors.append(str(exc))
        return _empty_result(FilterStatus.INVALID_INPUT, diag, prov, config)

    ts: list[float] = [float(ekf.t)]
    xs: list[np.ndarray] = [ekf.x.copy()]
    Ps: list[np.ndarray] = [ekf.P.copy()]
    telemetry: list[InnovationTelemetry] = []
    sensors_seen: set[str] = set()
    status = FilterStatus.OK

    for measurement in ordered:
        try:
            if measurement.t < float(ekf.t) - 1e-12:
                diag.n_rejected += 1
                diag.warnings.append(
                    f"Medição rejeitada (t={measurement.t:.6g} < "
                    f"filter_t={ekf.t:.6g}, sensor={measurement.sensor})."
                )
                status = FilterStatus.WARNING
                continue

            if measurement.t > float(ekf.t) + 1e-12:
                ekf.predict(measurement.t)
                diag.n_predict += 1

            update_evidence = ekf.update(measurement)
            telemetry.append(update_evidence)
            diag.n_update += 1
            sensors_seen.add(measurement.sensor)
            ts.append(float(ekf.t))
            xs.append(ekf.x.copy())
            Ps.append(ekf.P.copy())

        except EKFNumericalError as exc:
            diag.errors.append(str(exc))
            status = FilterStatus.NUMERICAL_FAILURE
            break
        except SchemaError as exc:
            diag.errors.append(str(exc))
            status = FilterStatus.INVALID_INPUT
            break

    diag.sensors_seen = sorted(sensors_seen)
    return FilterResult(
        t=np.asarray(ts, dtype=np.float64),
        x=np.asarray(xs, dtype=np.float64),
        P=np.asarray(Ps, dtype=np.float64),
        status=status,
        diagnostics=diag,
        provenance=prov,
        config={
            "sigma_a": float(sigma_a),
            "sigma_alpha": float(sigma_alpha),
            "t0": start_t,
            "n_measurements": len(ordered),
        },
        telemetry=tuple(telemetry),
    )


def simulate_measurements(
    gt: GroundTruth,
    seed: int = 42,
    gps_sigma: float = 1.5,
    gps_every: int = 5,
    speed_sigma: float = 0.2,
    yawrate_sigma: float = 0.02,
) -> list[Measurement]:
    """Generate synthetic measurements from GroundTruth (does not mutate gt)."""
    for name, value in (
        ("gps_sigma", gps_sigma),
        ("speed_sigma", speed_sigma),
        ("yawrate_sigma", yawrate_sigma),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise SchemaError(f"{name} deve ser finito e >= 0.")
    if not isinstance(gps_every, int) or gps_every < 1:
        raise SchemaError("gps_every deve ser inteiro >= 1")

    rng = np.random.default_rng(int(seed))
    ms: list[Measurement] = []
    for i in range(gt.n):
        t = float(gt.t[i])
        z_speed = np.array([float(gt.speed[i]) + rng.normal(0.0, speed_sigma)])
        ms.append(
            Measurement(t=t, sensor="speed", z=z_speed, R=np.array([[speed_sigma ** 2]]))
        )
        z_yaw = np.array([float(gt.yaw_rate[i]) + rng.normal(0.0, yawrate_sigma)])
        ms.append(
            Measurement(t=t, sensor="yaw_rate", z=z_yaw, R=np.array([[yawrate_sigma ** 2]]))
        )
        if i % gps_every == 0:
            z_gps = gt.pos[i] + rng.normal(0.0, gps_sigma, size=2)
            ms.append(
                Measurement(t=t, sensor="gps", z=z_gps, R=np.eye(2) * (gps_sigma ** 2))
            )
    return ms
