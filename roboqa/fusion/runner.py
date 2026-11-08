"""
Temporal orchestration of the EKF over asynchronous measurements.
Fase 2 — Fusion Engine

Policy:
- Measurements sorted by (t, sensor_name) for deterministic ordering.
- Out-of-order after start (t < current filter time) rejected with diagnosis.
- Original Measurement / GroundTruth objects are never mutated.
- Initialization uses a high-uncertainty prior (or caller-provided x0/P0).
  No measurement is consumed twice (no double-counting of first GPS).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from roboqa.data.schemas import GroundTruth, Measurement, SchemaError
from roboqa.fusion.ekf import EKF, EKFNumericalError
from roboqa.fusion.result import FilterDiagnostics, FilterResult, FilterStatus
from roboqa.fusion.sensors import validate_measurement_for_fusion


def _default_x0_P0(
    measurements: Sequence[Measurement],
    x0: Optional[np.ndarray],
    P0: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], Optional[int]]:
    """Build initial state/covariance without double-counting measurements."""
    warnings: list[str] = []
    skip_id: Optional[int] = None

    if x0 is not None and P0 is not None:
        return (
            np.asarray(x0, dtype=np.float64).reshape(5),
            np.asarray(P0, dtype=np.float64).reshape(5, 5),
            warnings,
            skip_id,
        )

    x = np.zeros(5, dtype=np.float64)
    P = np.diag([25.0, 25.0, 0.5, 4.0, 0.25]).astype(np.float64)

    has_gps = any(m.sensor.strip().lower() == "gps" for m in measurements)
    if not has_gps:
        warnings.append(
            "Nenhum GPS na sequência; inicialização de posição em (0,0) com alta incerteza."
        )

    if x0 is not None:
        x = np.asarray(x0, dtype=np.float64).reshape(5)
    if P0 is not None:
        P = np.asarray(P0, dtype=np.float64).reshape(5, 5)

    return x, P, warnings, skip_id


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

    if not measurements:
        return FilterResult(
            t=np.array([]),
            x=np.zeros((0, 5)),
            P=np.zeros((0, 5, 5)),
            status=FilterStatus.INIT_INSUFFICIENT,
            diagnostics=diag,
            provenance=prov,
            config={"sigma_a": sigma_a, "sigma_alpha": sigma_alpha},
        )

    for m in measurements:
        try:
            validate_measurement_for_fusion(m)
        except SchemaError as e:
            diag.errors.append(str(e))
            return FilterResult(
                t=np.array([]),
                x=np.zeros((0, 5)),
                P=np.zeros((0, 5, 5)),
                status=FilterStatus.INVALID_INPUT,
                diagnostics=diag,
                provenance=prov,
                config={"sigma_a": sigma_a, "sigma_alpha": sigma_alpha},
            )

    ordered = sorted(measurements, key=lambda m: (m.t, m.sensor))

    x_init, P_init, init_warns, _skip_id = _default_x0_P0(ordered, x0, P0)
    diag.warnings.extend(init_warns)

    try:
        ekf = EKF(x_init, P_init, sigma_a=sigma_a, sigma_alpha=sigma_alpha)
    except SchemaError as e:
        diag.errors.append(str(e))
        return FilterResult(
            t=np.array([]),
            x=np.zeros((0, 5)),
            P=np.zeros((0, 5, 5)),
            status=FilterStatus.INVALID_INPUT,
            diagnostics=diag,
            provenance=prov,
            config={"sigma_a": sigma_a, "sigma_alpha": sigma_alpha},
        )

    start_t = float(t0) if t0 is not None else float(ordered[0].t)
    ekf.set_time(start_t)

    ts: list[float] = []
    xs: list[np.ndarray] = []
    Ps: list[np.ndarray] = []
    sensors_seen: set[str] = set()

    ts.append(ekf.t)  # type: ignore[arg-type]
    xs.append(ekf.x.copy())
    Ps.append(ekf.P.copy())

    status = FilterStatus.OK

    for m in ordered:
        try:
            if ekf.t is not None and m.t < ekf.t - 1e-12:
                diag.n_rejected += 1
                diag.warnings.append(
                    f"Medição rejeitada (t={m.t:.6g} < filter_t={ekf.t:.6g}, sensor={m.sensor})."
                )
                status = FilterStatus.WARNING
                continue

            if ekf.t is not None and m.t > ekf.t + 1e-15:
                ekf.predict(m.t)
                diag.n_predict += 1

            ekf.update(m)
            diag.n_update += 1
            sensors_seen.add(m.sensor)

            ts.append(float(ekf.t) if ekf.t is not None else float(m.t))
            xs.append(ekf.x.copy())
            Ps.append(ekf.P.copy())

        except EKFNumericalError as e:
            diag.errors.append(str(e))
            status = FilterStatus.NUMERICAL_FAILURE
            break
        except SchemaError as e:
            diag.errors.append(str(e))
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
            "sigma_a": sigma_a,
            "sigma_alpha": sigma_alpha,
            "t0": start_t,
            "n_measurements": len(ordered),
        },
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
    if gps_every < 1:
        raise SchemaError("gps_every deve ser >= 1")
    rng = np.random.default_rng(int(seed))
    ms: list[Measurement] = []
    n = gt.n
    for i in range(n):
        t = float(gt.t[i])
        z_s = np.array([float(gt.speed[i]) + rng.normal(0.0, speed_sigma)])
        R_s = np.array([[speed_sigma ** 2]])
        ms.append(Measurement(t=t, sensor="speed", z=z_s, R=R_s))
        z_w = np.array([float(gt.yaw_rate[i]) + rng.normal(0.0, yawrate_sigma)])
        R_w = np.array([[yawrate_sigma ** 2]])
        ms.append(Measurement(t=t, sensor="yaw_rate", z=z_w, R=R_w))
        if i % gps_every == 0:
            z_g = gt.pos[i] + rng.normal(0.0, gps_sigma, size=2)
            R_g = np.eye(2) * (gps_sigma ** 2)
            ms.append(Measurement(t=t, sensor="gps", z=z_g, R=R_g))
    return ms
