"""Integration — full synthetic pipeline experiment."""

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
        t = gt.t[i]
        j = int(np.argmin(np.abs(result.t - t)))
        errors.append(np.linalg.norm(est_pos[j] - gt.pos[i]))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    assert rmse < 5.0, f"RMSE too high: {rmse}"


def test_pipeline_provenance_preserved():
    gt = make_synthetic_ground_truth(n=20, seed=0)
    ms = simulate_measurements(gt, seed=0)
    result = run_filter(ms, provenance={"gt_source": gt.source})
    assert result.provenance.get("gt_source") == "synthetic"
