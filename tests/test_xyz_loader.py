"""Tests for loading coordinates from XYZ files."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data.xyz_loader import XYZLoader  # noqa: E402


def test_load_defaults_to_float64_for_large_coordinates():
    expected_points = np.array([
        [5000000.123456, 4000000.234567, 120.345678],
        [5000000.987654, 4000000.765432, 121.876543],
    ])
    data = np.column_stack(([2, 6], expected_points))

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "points.xyz"
        np.savetxt(filepath, data, fmt="%d %.6f %.6f %.6f")

        classes, points = XYZLoader(filepath).load()

    np.testing.assert_array_equal(classes, [2, 6])
    assert points.dtype == np.float64
    np.testing.assert_array_equal(points, expected_points)
