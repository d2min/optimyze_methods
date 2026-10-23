"""Тесты прореживания облака точек."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.optimization.downsampling.voxel import (  # noqa: E402
    _process_voxel_grid,
    downsample_point_cloud,
)


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


def test_voxel_size_is_refined_when_outliers_expand_bounding_box():
    rng = np.random.default_rng(3)
    points = np.vstack((
        rng.uniform(0, 1, size=(4900, 3)),
        [[100.0, 100.0, 100.0], [0.0, 0.0, 0.0]],
    ))

    result = downsample_point_cloud(points, target_points=1000)

    assert 800 <= len(result) <= 1000


def test_negative_target_is_rejected():
    points = np.ones((10, 3))

    try:
        downsample_point_cloud(points, target_points=-1)
    except ValueError as error:
        assert "отрицательным" in str(error)
    else:
        raise AssertionError("Отрицательный target_points должен вызывать ValueError")


def test_classes_stay_aligned_when_downsampling():
    rng = np.random.default_rng(1)
    points = rng.uniform(-10, 10, size=(200, 3))
    classes = np.full(len(points), 6, dtype=np.int32)

    result_points, result_classes = downsample_point_cloud(
        points, target_points=20, classes=classes
    )

    assert len(result_points) == len(result_classes) <= 20
    np.testing.assert_array_equal(result_classes, np.full(len(result_classes), 6))


def test_classes_are_returned_unchanged_when_downsampling_is_not_needed():
    points = np.arange(30, dtype=np.float64).reshape(10, 3)
    classes = np.arange(10, dtype=np.int32)

    result_points, result_classes = downsample_point_cloud(
        points, target_points=10, classes=classes
    )

    assert result_points is points
    assert result_classes is classes


def test_voxel_class_is_the_most_common_input_class():
    points = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1],
        [0.3, 0.1, 0.1],
        [1.1, 0.1, 0.1],
    ])
    classes = np.array([2, 2, 6, 9], dtype=np.int32)

    centroids, result_classes = _process_voxel_grid(
        points, voxel_size=1.0, p_min=np.zeros(3), classes=classes
    )

    np.testing.assert_allclose(centroids[:, 0], [0.2, 1.1])
    np.testing.assert_array_equal(result_classes, [2, 9])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"Все тесты пройдены: {len(tests)}")
