"""Тесты прореживания облака точек."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.optimization.downsampling.voxel import downsample_point_cloud  # noqa: E402


def test_returns_original_points_when_target_is_not_smaller():
    points = np.arange(30, dtype=np.float64).reshape(10, 3)

    assert downsample_point_cloud(points, target_points=10) is points
    assert downsample_point_cloud(points, target_points=15) is points


def test_zero_target_returns_empty_array_with_same_shape():
    points = np.ones((10, 3), dtype=np.float32)

    result = downsample_point_cloud(points, target_points=0)

    assert result.shape == (0, 3)
    assert result.dtype == points.dtype


def test_result_does_not_exceed_target():
    rng = np.random.default_rng(0)
    points = rng.uniform(-10, 10, size=(200, 3))

    result = downsample_point_cloud(points, target_points=20)

    assert len(result) <= 20


def test_negative_target_is_rejected():
    points = np.ones((10, 3))

    try:
        downsample_point_cloud(points, target_points=-1)
    except ValueError as error:
        assert "отрицательным" in str(error)
    else:
        raise AssertionError("Отрицательный target_points должен вызывать ValueError")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"Все тесты пройдены: {len(tests)}")
