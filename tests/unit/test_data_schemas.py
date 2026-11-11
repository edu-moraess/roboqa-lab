"""Testes unitários — Fase 1 Data Engine (schemas + synthetic + kitti helpers)"""

import numpy as np
import pytest

from roboqa.data.kitti import download_kitti_drive, get_trajectory, latlon_to_local
from roboqa.data.schemas import GroundTruth, Measurement, SchemaError, validate_measurement_sequence
from roboqa.data.synthetic import make_synthetic_ground_truth


def test_synthetic_ground_truth_basic():
    gt = make_synthetic_ground_truth(n=100, dt=0.1, speed=5.0, seed=42)
    assert gt.n == 100
    assert gt.source == "synthetic"
    assert gt.pos.shape == (100, 2)
    assert np.all(np.isfinite(gt.pos))
    assert np.all(np.diff(gt.t) > 0)


def test_synthetic_reproducibility():
    gt1 = make_synthetic_ground_truth(n=50, seed=123)
    gt2 = make_synthetic_ground_truth(n=50, seed=123)
    np.testing.assert_array_equal(gt1.pos, gt2.pos)
    np.testing.assert_array_equal(gt1.yaw, gt2.yaw)


def test_synthetic_different_seeds_differ():
    gt1 = make_synthetic_ground_truth(n=50, seed=1)
    gt2 = make_synthetic_ground_truth(n=50, seed=2)
    assert not np.allclose(gt1.speed, gt2.speed)


def test_groundtruth_invalid_shape():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.1]),
            pos=np.array([1.0, 2.0]),
            yaw=np.array([0.0, 0.1]),
            speed=np.array([5.0, 5.0]),
            yaw_rate=np.array([0.0, 0.0]),
            source="synthetic",
        )


def test_groundtruth_empty():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([]),
            pos=np.empty((0, 2)),
            yaw=np.array([]),
            speed=np.array([]),
            yaw_rate=np.array([]),
            source="synthetic",
        )


def test_groundtruth_nan_rejected():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.1]),
            pos=np.array([[0.0, 0.0], [np.nan, 1.0]]),
            yaw=np.array([0.0, 0.1]),
            speed=np.array([5.0, 5.0]),
            yaw_rate=np.array([0.0, 0.0]),
            source="synthetic",
        )


def test_groundtruth_regressive_timestamps():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.1, 0.05]),
            pos=np.zeros((3, 2)),
            yaw=np.zeros(3),
            speed=np.ones(3),
            yaw_rate=np.zeros(3),
            source="synthetic",
        )


def test_groundtruth_duplicate_timestamps():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.1, 0.1]),
            pos=np.zeros((3, 2)),
            yaw=np.zeros(3),
            speed=np.ones(3),
            yaw_rate=np.zeros(3),
            source="synthetic",
        )


def test_groundtruth_invalid_source():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.1]),
            pos=np.zeros((2, 2)),
            yaw=np.zeros(2),
            speed=np.ones(2),
            yaw_rate=np.zeros(2),
            source="real_sensor",
        )


def test_groundtruth_immutability_of_arrays():
    t = np.array([0.0, 0.1, 0.2])
    pos = np.zeros((3, 2))
    gt = GroundTruth(
        t=t, pos=pos, yaw=np.zeros(3), speed=np.ones(3),
        yaw_rate=np.zeros(3), source="synthetic",
    )
    t[0] = 999.0
    pos[0, 0] = 999.0
    assert gt.t[0] == 0.0
    assert gt.pos[0, 0] == 0.0


def test_groundtruth_copy():
    gt = make_synthetic_ground_truth(n=20, seed=7)
    gt2 = gt.copy()
    assert gt2.source == gt.source
    np.testing.assert_array_equal(gt2.pos, gt.pos)
    assert gt2.pos is not gt.pos
    assert gt2.t is not gt.t
    assert not gt.pos.flags.writeable
    assert not gt2.pos.flags.writeable


def test_measurement_valid():
    m = Measurement(
        t=1.0,
        sensor="gps",
        z=np.array([10.0, 20.0]),
        R=np.eye(2) * 2.25,
    )
    assert m.sensor == "gps"
    assert m.z.shape == (2,)
    assert m.R.shape == (2, 2)


def test_measurement_asymmetric_R():
    with pytest.raises(SchemaError):
        Measurement(
            t=0.0,
            sensor="gps",
            z=np.array([0.0, 0.0]),
            R=np.array([[1.0, 0.5], [0.0, 1.0]]),
        )


def test_measurement_negative_eigenvalue():
    with pytest.raises(SchemaError):
        Measurement(
            t=0.0,
            sensor="gps",
            z=np.array([0.0, 0.0]),
            R=np.array([[1.0, 0.0], [0.0, -0.1]]),
        )


def test_measurement_nan_z():
    with pytest.raises(SchemaError):
        Measurement(
            t=0.0,
            sensor="gps",
            z=np.array([0.0, np.nan]),
            R=np.eye(2),
        )


def test_measurement_empty_sensor():
    with pytest.raises(SchemaError):
        Measurement(
            t=0.0,
            sensor="   ",
            z=np.array([0.0]),
            R=np.array([[1.0]]),
        )


def test_measurement_dimension_mismatch():
    with pytest.raises(SchemaError):
        Measurement(
            t=0.0,
            sensor="gps",
            z=np.array([0.0, 1.0]),
            R=np.eye(3),
        )


def test_measurement_nonfinite_t():
    with pytest.raises(SchemaError):
        Measurement(
            t=np.inf,
            sensor="gps",
            z=np.array([0.0]),
            R=np.array([[1.0]]),
        )


def test_validate_measurement_sequence_ok():
    ms = [
        Measurement(t=0.0, sensor="gps", z=np.zeros(2), R=np.eye(2)),
        Measurement(t=0.1, sensor="gps", z=np.zeros(2), R=np.eye(2)),
    ]
    validate_measurement_sequence(ms)


def test_validate_measurement_sequence_empty():
    validate_measurement_sequence([])


def test_validate_measurement_sequence_regressive():
    ms = [
        Measurement(t=0.2, sensor="gps", z=np.zeros(2), R=np.eye(2)),
        Measurement(t=0.1, sensor="gps", z=np.zeros(2), R=np.eye(2)),
    ]
    with pytest.raises(SchemaError):
        validate_measurement_sequence(ms)


def test_synthetic_invalid_n():
    with pytest.raises(SchemaError):
        make_synthetic_ground_truth(n=1)


def test_synthetic_invalid_dt():
    with pytest.raises(SchemaError):
        make_synthetic_ground_truth(dt=0.0)


def test_synthetic_invalid_speed():
    with pytest.raises(SchemaError):
        make_synthetic_ground_truth(speed=0.0)


def test_synthetic_invalid_radius():
    with pytest.raises(SchemaError):
        make_synthetic_ground_truth(radius=-1.0)


def test_synthetic_invalid_yaw_rate_amp():
    with pytest.raises(SchemaError):
        make_synthetic_ground_truth(yaw_rate_amp=-0.1)


def test_synthetic_yaw_rate_amp_used():
    gt0 = make_synthetic_ground_truth(n=50, seed=0, yaw_rate_amp=0.0)
    gt1 = make_synthetic_ground_truth(n=50, seed=0, yaw_rate_amp=0.5)
    assert not np.allclose(gt0.yaw, gt1.yaw)


def test_latlon_to_local_basic():
    lat = np.array([48.0, 48.001])
    lon = np.array([11.0, 11.001])
    x, y = latlon_to_local(lat, lon)
    assert x.shape == (2,)
    assert y.shape == (2,)
    assert np.all(np.isfinite(x))
    assert x[1] > 0
    assert y[1] > 0


def test_latlon_to_local_empty():
    with pytest.raises(SchemaError):
        latlon_to_local(np.array([]), np.array([]))


def test_latlon_to_local_shape_mismatch():
    with pytest.raises(SchemaError):
        latlon_to_local(np.array([1.0, 2.0]), np.array([1.0]))


def test_latlon_to_local_nan():
    with pytest.raises(SchemaError):
        latlon_to_local(np.array([48.0, np.nan]), np.array([11.0, 11.0]))


def test_get_trajectory_fallback():
    gt = get_trajectory(prefer_kitti=False, n=80, seed=7)
    assert gt.source == "synthetic"
    assert gt.n == 80


def test_get_trajectory_prefer_kitti_without_path():
    with pytest.raises(SchemaError):
        get_trajectory(prefer_kitti=True, kitti_path=None)


def test_get_trajectory_prefer_kitti_invalid_path():
    with pytest.raises(SchemaError):
        get_trajectory(prefer_kitti=True, kitti_path="/nonexistent/path/oxts")


def test_download_kitti_raises():
    with pytest.raises(NotImplementedError):
        download_kitti_drive()


def test_load_kitti_oxts_happy_path(tmp_path):
    d = tmp_path / "oxts_data"
    d.mkdir()
    (d / "0000000000.txt").write_text(
        "48.982543 11.390439 116.466827 0.035279 -0.022039 0.000328 "
        "0.0 0.0 5.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
    )
    (d / "0000000001.txt").write_text(
        "48.982550 11.390450 116.467000 0.035280 -0.022040 0.010328 "
        "0.0 0.0 5.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
    )
    from roboqa.data.kitti import load_kitti_oxts
    gt = load_kitti_oxts(d)
    assert gt.source == "kitti_real"
    assert gt.n == 2
    assert gt.pos.shape == (2, 2)
    assert np.all(np.isfinite(gt.yaw_rate))


def test_load_kitti_max_frames(tmp_path):
    d = tmp_path / "oxts_data"
    d.mkdir()
    for i in range(5):
        (d / f"{i:010d}.txt").write_text(
            f"48.98 11.39 116.0 0.0 0.0 0.{i} 0.0 0.0 5.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
        )
    from roboqa.data.kitti import load_kitti_oxts
    gt = load_kitti_oxts(d, max_frames=3)
    assert gt.n == 3


def test_load_kitti_max_frames_invalid(tmp_path):
    d = tmp_path / "oxts_data"
    d.mkdir()
    (d / "0000000000.txt").write_text("48.0 11.0 0 0 0 0 0 0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
    from roboqa.data.kitti import load_kitti_oxts
    with pytest.raises(SchemaError):
        load_kitti_oxts(d, max_frames=1)


def test_measurement_empty_z():
    with pytest.raises(SchemaError):
        Measurement(t=0.0, sensor="gps", z=np.array([]), R=np.empty((0, 0)))


def test_schemas_invalid_source_type():
    with pytest.raises(SchemaError):
        GroundTruth(
            t=np.array([0.0, 0.1]),
            pos=np.zeros((2, 2)),
            yaw=np.zeros(2),
            speed=np.ones(2),
            yaw_rate=np.zeros(2),
            source="invalid_source_xyz",
        )


def test_synthetic_seed_type_error():
    with pytest.raises(SchemaError):
        make_synthetic_ground_truth(seed="not_an_int")  # type: ignore


def test_groundtruth_arrays_readonly():
    gt = make_synthetic_ground_truth(n=20, seed=11)
    for name in ("t", "pos", "yaw", "speed", "yaw_rate"):
        arr = getattr(gt, name)
        assert not arr.flags.writeable, f"{name} deveria ser write=False"
        with pytest.raises((ValueError, RuntimeError)):
            if arr.ndim == 1:
                arr[0] = 999.0
            else:
                arr[0, 0] = 999.0


def test_measurement_arrays_readonly():
    m = Measurement(
        t=0.0,
        sensor="gps",
        z=np.array([1.0, 2.0]),
        R=np.eye(2),
    )
    assert not m.z.flags.writeable
    assert not m.R.flags.writeable
    with pytest.raises((ValueError, RuntimeError)):
        m.z[0] = 999.0
    with pytest.raises((ValueError, RuntimeError)):
        m.R[0, 0] = 999.0


def test_copy_produces_independent_readonly_arrays():
    gt = make_synthetic_ground_truth(n=15, seed=3)
    gt2 = gt.copy()
    assert gt2.t is not gt.t
    assert gt2.pos is not gt.pos
    assert not gt.t.flags.writeable
    assert not gt2.t.flags.writeable
    assert not gt.pos.flags.writeable
    assert not gt2.pos.flags.writeable
    with pytest.raises((ValueError, RuntimeError)):
        gt2.pos[0, 0] = 123.0


def test_groundtruth_geodetic_position_is_preserved_and_readonly():
    geodetic = np.array([
        [48.982543, 11.390439, 116.466827],
        [48.982550, 11.390450, 116.467000],
    ])
    gt = GroundTruth(
        t=np.array([0.0, 0.1]),
        pos=np.zeros((2, 2)),
        yaw=np.zeros(2),
        speed=np.ones(2),
        yaw_rate=np.zeros(2),
        source="kitti_real",
        geodetic_position=geodetic,
        geodetic_datum="WGS84",
    )
    geodetic[0, 0] = 0.0
    assert gt.geodetic_position is not None
    assert gt.geodetic_position[0, 0] == pytest.approx(48.982543)
    assert gt.geodetic_datum == "WGS84"
    assert not gt.geodetic_position.flags.writeable


def test_groundtruth_geodetic_requires_explicit_datum():
    with pytest.raises(SchemaError, match="geodetic_datum must be explicit"):
        GroundTruth(
            t=np.array([0.0, 0.1]),
            pos=np.zeros((2, 2)),
            yaw=np.zeros(2),
            speed=np.ones(2),
            yaw_rate=np.zeros(2),
            source="kitti_real",
            geodetic_position=np.zeros((2, 3)),
        )


def test_groundtruth_geodetic_rejects_invalid_latitude():
    with pytest.raises(SchemaError, match="latitude"):
        GroundTruth(
            t=np.array([0.0, 0.1]),
            pos=np.zeros((2, 2)),
            yaw=np.zeros(2),
            speed=np.ones(2),
            yaw_rate=np.zeros(2),
            source="kitti_real",
            geodetic_position=np.array([[91.0, 11.0, 100.0], [48.0, 11.0, 100.0]]),
            geodetic_datum="WGS84",
        )


def test_groundtruth_copy_preserves_geodetic_coordinates():
    gt = GroundTruth(
        t=np.array([0.0, 0.1]),
        pos=np.zeros((2, 2)),
        yaw=np.zeros(2),
        speed=np.ones(2),
        yaw_rate=np.zeros(2),
        source="kitti_real",
        geodetic_position=np.array([[48.0, 11.0, 100.0], [48.1, 11.1, 101.0]]),
        geodetic_datum="WGS84",
    )
    copied = gt.copy()
    assert copied.geodetic_datum == "WGS84"
    np.testing.assert_array_equal(copied.geodetic_position, gt.geodetic_position)
    assert copied.geodetic_position is not gt.geodetic_position
    assert not copied.geodetic_position.flags.writeable


def test_load_kitti_oxts_preserves_original_geodetic_coordinates(tmp_path):
    d = tmp_path / "oxts_data"
    d.mkdir()
    rows = [
        "48.982543 11.390439 116.466827 0.035279 -0.022039 0.000328 0 0 5.2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "48.982550 11.390450 116.467000 0.035280 -0.022040 0.010328 0 0 5.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
    ]
    for i, row in enumerate(rows):
        (d / f"{i:010d}.txt").write_text(row + "\\n")
    from roboqa.data.kitti import load_kitti_oxts
    gt = load_kitti_oxts(d)
    assert gt.geodetic_position is not None
    assert gt.geodetic_datum == "WGS84"
    np.testing.assert_allclose(
        gt.geodetic_position,
        np.array([[48.982543, 11.390439, 116.466827],
                  [48.982550, 11.390450, 116.467000]]),
    )
