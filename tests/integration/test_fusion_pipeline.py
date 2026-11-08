"""Integration — fusion pipeline exposes consistent innovation telemetry."""

import numpy as np

from roboqa.data.synthetic import make_synthetic_ground_truth
from roboqa.fusion.result import FilterStatus
from roboqa.fusion.runner import run_filter, simulate_measurements


def test_pipeline_position_error_reasonable():
    gt = make_synthetic_ground_truth(n=100, dt=0.1, speed=5.0, seed=42, radius=40.0)
    ms = simulate_measurements(
        gt, seed=42, gps_sigma=1.5, gps_every=5, speed_sigma=0.2, yawrate_sigma=0.02
    )
    result = run_filter(
        ms,
        sigma_a=0.5,
        sigma_alpha=0.1,
        provenance={"gt_source": gt.source, "seed": 42, "kind": "synthetic_experiment"},
    )
    assert result.status in (FilterStatus.OK, FilterStatus.WARNING)
    assert result.diagnostics.n_update > 50

    est_pos = result.position()
    errors = []
    for i in range(0, gt.n, 5):
        j = int(np.argmin(np.abs(result.t - gt.t[i])))
        errors.append(np.linalg.norm(est_pos[j] - gt.pos[i]))
    assert float(np.sqrt(np.mean(np.square(errors)))) < 5.0


def test_pipeline_provenance_preserved():
    gt = make_synthetic_ground_truth(n=20, seed=0)
    result = run_filter(simulate_measurements(gt, seed=0), provenance={"gt_source": gt.source})
    assert result.provenance.get("gt_source") == "synthetic"


def test_pipeline_telemetry_is_complete_and_sensor_dimension_aware():
    gt = make_synthetic_ground_truth(n=25, dt=0.1, seed=11)
    result = run_filter(simulate_measurements(gt, seed=11, gps_every=4))

    assert len(result.telemetry) == result.diagnostics.n_update
    assert {r.sensor_id for r in result.telemetry} == {"gps", "speed", "yaw_rate"}
    assert all(r.timestamp >= 0.0 for r in result.telemetry)
    assert all(r.measurement_dof == r.innovation.size for r in result.telemetry)
    assert all(
        r.innovation_covariance.shape == (r.measurement_dof, r.measurement_dof)
        for r in result.telemetry
    )
    assert all(np.isfinite(r.nis) and r.nis >= 0.0 for r in result.telemetry)
