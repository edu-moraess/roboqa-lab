"""
Red Team — Testes adversariais Fase 1 (Data Engine)
Ataques deliberados contra contratos, validação e proveniência.
"""

import numpy as np
import pytest

from roboqa.data.kitti import get_trajectory, latlon_to_local, load_kitti_oxts
from roboqa.data.schemas import GroundTruth, Measurement, SchemaError
from roboqa.data.synthetic import make_synthetic_ground_truth


@pytest.mark.parametrize("field,value", [
    ("t", np.array([0.0, np.nan])),
    ("t", np.array([0.0, np.inf])),
    ("pos", np.array([[0.0, 0.0], [np.nan, 0.0]])),
    ("yaw", np.array([0.0, np.inf])),
    ("speed", np.array([1.0, -np.inf])),
    ("yaw_rate", np.array([0.0, np.nan])),
])
def test_gt_nan_inf_rejected(field, value):
    kwargs = dict(
        t=np.array([0.0, 0.1]),
        pos=np.zeros((2, 2)),
        yaw=np.zeros(2),
        speed=np.ones(2),
        yaw_rate=np.zeros(2),
        source="synthetic",
    )
    kwargs[field] = value
    with pytest.raises(SchemaError):
        GroundTruth(**kwargs)


def test_empty_arrays():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([]), pos=np.empty((0, 2)),
            yaw=np.array([]), speed=np.array([]),
            yaw_rate=np.array([]), source="synthetic",
        )


def test_dimension_mismatch_pos():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.1, 0.2]),
            pos=np.zeros((2, 2)),
            yaw=np.zeros(3), speed=np.ones(3), yaw_rate=np.zeros(3),
            source="synthetic",
        )


def test_measurement_z_R_mismatch():
    with pytest.raises(SchemaError):
        Measurement(t=0.0, sensor="s", z=np.ones(3), R=np.eye(2))


def test_regressive_timestamps():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.2, 0.1]),
            pos=np.zeros((3, 2)), yaw=np.zeros(3),
            speed=np.ones(3), yaw_rate=np.zeros(3), source="synthetic",
        )


def test_duplicate_timestamps():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.1, 0.1 + 1e-15]),
            pos=np.zeros((3, 2)), yaw=np.zeros(3),
            speed=np.ones(3), yaw_rate=np.zeros(3), source="synthetic",
        )


def test_irregular_but_valid_timestamps():
    gt = GroundTruth(
        t=np.array([0.0, 0.05, 0.3, 1.0]),
        pos=np.zeros((4, 2)), yaw=np.zeros(4),
        speed=np.ones(4), yaw_rate=np.zeros(4), source="synthetic",
    )
    assert gt.n == 4


def test_asymmetric_R():
    with pytest.raises(SchemaError):
        Measurement(
            t=0.0, sensor="gps", z=np.zeros(2),
            R=np.array([[1.0, 0.9], [0.1, 1.0]]),
        )


def test_negative_eigenvalue_R():
    with pytest.raises(SchemaError):
        Measurement(
            t=0.0, sensor="gps", z=np.zeros(2),
            R=np.array([[1.0, 0.0], [0.0, -1e-3]]),
        )


@pytest.mark.parametrize("kwargs", [
    {"n": 0},
    {"n": 1},
    {"dt": 0.0},
    {"dt": -1.0},
    {"speed": 0.0},
    {"speed": -5.0},
    {"radius": 0.0},
    {"radius": -10.0},
    {"yaw_rate_amp": -0.01},
])
def test_synthetic_extreme_params(kwargs):
    with pytest.raises(SchemaError):
        make_synthetic_ground_truth(**kwargs)


def test_synthetic_large_n():
    gt = make_synthetic_ground_truth(n=2000, seed=0)
    assert gt.n == 2000
    assert np.all(np.isfinite(gt.pos))


def test_kitti_nonexistent_dir():
    with pytest.raises(SchemaError):
        load_kitti_oxts("/tmp/does_not_exist_roboqa_xyz")


def test_kitti_empty_dir(tmp_path):
    d = tmp_path / "empty_oxts"
    d.mkdir()
    with pytest.raises(SchemaError):
        load_kitti_oxts(d)


def test_kitti_corrupted_file(tmp_path):
    d = tmp_path / "bad_oxts"
    d.mkdir()
    (d / "0000000000.txt").write_text("not a number\n")
    with pytest.raises(SchemaError):
        load_kitti_oxts(d)


def test_kitti_incomplete_file(tmp_path):
    d = tmp_path / "short_oxts"
    d.mkdir()
    (d / "0000000000.txt").write_text("1.0 2.0 3.0\n")
    with pytest.raises(SchemaError):
        load_kitti_oxts(d)


def test_latlon_nan():
    with pytest.raises(SchemaError):
        latlon_to_local(np.array([np.nan]), np.array([0.0]))


def test_latlon_shape_mismatch():
    with pytest.raises(SchemaError):
        latlon_to_local(np.ones(3), np.ones(2))


def test_no_silent_fallback_prefer_kitti():
    with pytest.raises(SchemaError):
        get_trajectory(prefer_kitti=True, kitti_path=None)
    with pytest.raises(SchemaError):
        get_trajectory(prefer_kitti=True, kitti_path="/nonexistent")


def test_explicit_synthetic_ok():
    gt = get_trajectory(prefer_kitti=False, n=30, seed=99)
    assert gt.source == "synthetic"


def test_reproducibility_same_seed():
    a = make_synthetic_ground_truth(n=100, seed=42)
    b = make_synthetic_ground_truth(n=100, seed=42)
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.pos, b.pos)
    np.testing.assert_array_equal(a.yaw, b.yaw)
    np.testing.assert_array_equal(a.speed, b.speed)
    np.testing.assert_array_equal(a.yaw_rate, b.yaw_rate)


def test_reproducibility_different_seed():
    a = make_synthetic_ground_truth(n=100, seed=1)
    b = make_synthetic_ground_truth(n=100, seed=2)
    assert not np.allclose(a.speed, b.speed)


def test_array_isolation():
    t_ext = np.array([0.0, 0.1, 0.2])
    pos_ext = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = GroundTruth(
        t=t_ext, pos=pos_ext,
        yaw=np.zeros(3), speed=np.ones(3), yaw_rate=np.zeros(3),
        source="synthetic",
    )
    t_ext[0] = -999.0
    pos_ext[0, 0] = -999.0
    assert gt.t[0] == 0.0
    assert gt.pos[0, 0] == 0.0
    gt2 = gt.copy()
    gt.pos[0, 0] = 42.0
    assert gt2.pos[0, 0] == 0.0
