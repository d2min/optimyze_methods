"""Тесты поиска ближайших соседей по октаэдрической метрике."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.optimization.k_folks.k_folks import k_folks  # noqa: E402


def test_returns_self_and_neighbors_in_octahedral_radius_order():
    points = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 4.0, 0.0],
    ])

    result = k_folks(points, k=2, show_progress=False)

    assert result.shape == (4, 9)
    np.testing.assert_array_equal(result[:, :3], points)
    np.testing.assert_array_equal(result[0, 3:].reshape(2, 3), points[[2, 1]])
    np.testing.assert_array_equal(result[2, 3:].reshape(2, 3), points[[0, 1]])


def test_keeps_input_dtype_and_handles_zero_neighbors():
    points = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    result = k_folks(points, k=0, show_progress=False)

    assert result.shape == (2, 3)
    assert result.dtype == points.dtype
    np.testing.assert_array_equal(result, points)


def test_rejects_invalid_k_and_point_shape():
    points = np.zeros((2, 3))
    for invalid_k in (-1, 2, 1.5):
        try:
            k_folks(points, invalid_k, show_progress=False)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"k={invalid_k} должен вызвать ошибку")

    try:
        k_folks(np.zeros((2, 4)), 1, show_progress=False)
    except ValueError:
        pass
    else:
        raise AssertionError("Массив не формы (N, 3) должен вызвать ValueError")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"Все тесты пройдены: {len(tests)}")
