"""Поиск ближайших соседей по октаэдрической метрике."""
from operator import index
import sys

import numpy as np


def k_folks(points: np.ndarray, k: int, show_progress: bool = True) -> np.ndarray:
    """Вернуть каждую точку вместе с k ближайшими соседями.

    Для каждой target-точки она считается началом координат. Радиус другой
    точки равен сумме абсолютных разностей координат:
    ``abs(dx) + abs(dy) + abs(dz)``.

    Результат имеет форму ``(N, 3 * (k + 1))``. В каждой строке сначала идут
    XYZ target-точки, затем XYZ k соседей в порядке возрастания радиуса.
    Полный перебор занимает O(N²) времени. Расстояния считаются блоками через
    NumPy, поэтому промежуточная память ограничена размером одного блока.

    show_progress — отображать прогресс перебора в stderr.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points должен иметь форму (N, 3)")
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError("points должен содержать числовые координаты")
    if not np.isfinite(points).all():
        raise ValueError("points не должен содержать NaN или бесконечные значения")

    try:
        k = index(k)
    except TypeError as error:
        raise TypeError("k должен быть целым числом") from error
    if isinstance(k, bool):
        raise TypeError("k должен быть целым числом, не bool")
    if k < 0:
        raise ValueError("k не может быть отрицательным")

    n_points = len(points)
    max_neighbors = max(n_points - 1, 0)
    if k > max_neighbors:
        raise ValueError(f"k не может быть больше числа других точек ({max_neighbors})")

    neighbors = np.empty((n_points, k + 1, 3), dtype=points.dtype)
    neighbors[:, 0, :] = points
    if k == 0 or n_points == 0:
        return neighbors.reshape(n_points, 3 * (k + 1))

    coordinates = points.astype(np.float64, copy=False)
    # Keep pairwise-distance and partition-index temporaries bounded.
    max_block_pairs = 2_000_000
    block_rows = max(1, min(n_points, max_block_pairs // n_points))
    update_every = max(1, n_points // 100)
    next_progress = update_every

    for start in range(0, n_points, block_rows):
        stop = min(start + block_rows, n_points)
        block = coordinates[start:stop]

        radii = np.abs(block[:, 0, None] - coordinates[None, :, 0])
        radii += np.abs(block[:, 1, None] - coordinates[None, :, 1])
        radii += np.abs(block[:, 2, None] - coordinates[None, :, 2])
        rows = np.arange(stop - start)
        radii[rows, start + rows] = np.inf

        nearest = np.argpartition(radii, k - 1, axis=1)[:, :k]
        nearest_radii = np.take_along_axis(radii, nearest, axis=1)
        order = np.argsort(nearest_radii, axis=1, kind="stable")
        nearest = np.take_along_axis(nearest, order, axis=1)
        neighbors[start:stop, 1:, :] = points[nearest]

        completed = stop
        if show_progress and (completed >= next_progress or completed == n_points):
            percent = completed * 100 // n_points
            filled = percent * 30 // 100
            bar = "#" * filled + "-" * (30 - filled)
            print(f"\rПоиск соседей: [{bar}] {percent:3d}% "
                  f"({completed}/{n_points})", end="", file=sys.stderr, flush=True)
            next_progress = completed + update_every
    if show_progress:
        print(file=sys.stderr)

    return neighbors.reshape(n_points, 3 * (k + 1))
