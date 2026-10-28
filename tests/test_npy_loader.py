"""Tests for loading point clouds from NPY files."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data.npy_loader import NPYLoader  # noqa: E402


def test_loads_class_and_xyz_columns():
    data = np.asfortranarray([[2, 1.0, 2.0, 3.0], [6, 4.0, 5.0, 6.0]])
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "points.npy"
        np.save(filepath, data)

        classes, points = NPYLoader(filepath).load()

    np.testing.assert_array_equal(classes, [2, 6])
    np.testing.assert_array_equal(points, data[:, 1:])
    assert points.flags.c_contiguous


def test_loads_xyz_columns_with_default_class():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "points.npy"
        np.save(filepath, points)

        classes, loaded_points = NPYLoader(filepath).load()

    np.testing.assert_array_equal(classes, [0, 0])
    np.testing.assert_array_equal(loaded_points, points)


def test_rejects_unsupported_array_shape():
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "points.npy"
        np.save(filepath, np.zeros((2, 5)))

        try:
            NPYLoader(filepath).load()
        except ValueError as error:
            assert "Неверная форма" in str(error)
        else:
            raise AssertionError("Ожидалась ошибка для массива неверной формы")