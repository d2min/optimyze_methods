"""Тесты CSV-экспортёров."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data.xyz_export import export_k_folks_to_csv, export_points_to_csv  # noqa: E402


def test_export_points_csv_has_xyz_header_and_values():
    points = np.array([[1.123456789123, 2.5, 3.75], [4.0, 5.0, 6.0]])
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "nested" / "points.csv"

        export_points_to_csv(points, filepath)

        lines = filepath.read_text().splitlines()
        assert lines[0] == "x,y,z"
        assert lines[1] == "1.1234567891,2.5000000000,3.7500000000"
        np.testing.assert_allclose(np.loadtxt(filepath, delimiter=",", skiprows=1), points)


def test_export_k_folks_csv_has_target_and_neighbor_columns():
    neighbors = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "neighbors.csv"

        export_k_folks_to_csv(neighbors, filepath)

        lines = filepath.read_text().splitlines()
        assert lines[0] == (
            "target_x,target_y,target_z,"
            "neighbor_1_x,neighbor_1_y,neighbor_1_z,"
            "neighbor_2_x,neighbor_2_y,neighbor_2_z"
        )
        np.testing.assert_allclose(np.loadtxt(filepath, delimiter=",", skiprows=1), neighbors[0])


def test_exporters_reject_wrong_shapes():
    for exporter, matrix in (
        (export_points_to_csv, np.zeros((2, 6))),
        (export_k_folks_to_csv, np.zeros((2, 4))),
    ):
        try:
            exporter(matrix, "unused.csv")
        except ValueError:
            pass
        else:
            raise AssertionError("Матрица неправильной формы должна вызвать ValueError")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"Все тесты пройдены: {len(tests)}")
