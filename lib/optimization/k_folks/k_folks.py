"""Поиск ближайших соседей по октаэдрической метрике."""
from operator import index
import sys
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree


def k_folks(
    points: np.ndarray,
    k: int,
    show_progress: bool = True,
    classes: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Вернуть каждую точку вместе с k ближайшими соседями.

    Для каждой target-точки она считается началом координат. Радиус другой
    точки равен сумме абсолютных разностей координат:
    ``abs(dx) + abs(dy) + abs(dz)``.

    Результат имеет форму ``(N, 3 * (k + 1))``. В каждой строке сначала идут
    XYZ target-точки, затем XYZ k соседей в порядке возрастания радиуса.
    Поиск выполняется через cKDTree с метрикой Манхэттена (p=1), совпадающей
    с указанным радиусом. Ожидаемая сложность поиска — O(N log N + Nk),
    худшая — O(N² + Nk).

    show_progress — отображать прогресс перебора в stderr.

    features — необязательные признаки каждой точки формы (N, F), например
    hag, nx, ny, nz, ground. Если переданы, для target-точки и каждого соседа
    возвращается их строка признаков: массив (N, k + 1, F) в том же порядке,
    что и координаты. Возвращается последним элементом кортежа:
    (result, features) или (result, classes, features), если переданы и classes.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points должен иметь форму (N, 3)")
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError("points должен содержать числовые координаты")
    if not np.isfinite(points).all():
        raise ValueError("points не должен содержать NaN или бесконечные значения")
    if classes is not None:
        classes = np.asarray(classes)
        if classes.ndim != 1 or len(classes) != len(points):
            raise ValueError(
                f"classes должен иметь форму ({len(points)},), получена форма {classes.shape}"
            )

    if features is not None:
        features = np.asarray(features)
        if features.ndim != 2 or len(features) != len(points):
            raise ValueError(
                f"features должен иметь форму ({len(points)}, F), получена форма {features.shape}"
            )
        if not np.issubdtype(features.dtype, np.number):
            raise TypeError("features должен содержать числовые значения")
        if not np.isfinite(features).all():
            raise ValueError("features не должен содержать NaN или бесконечные значения")

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
    neighbor_classes = None
    if classes is not None:
        neighbor_classes = np.empty((n_points, k + 1), dtype=classes.dtype)
        neighbor_classes[:, 0] = classes
    neighbor_features = None
    if features is not None:
        neighbor_features = np.empty((n_points, k + 1, features.shape[1]), dtype=features.dtype)
        neighbor_features[:, 0, :] = features
    if k == 0 or n_points == 0:
        return _pack(neighbors.reshape(n_points, 3 * (k + 1)), neighbor_classes, neighbor_features)

    coordinates = points.astype(np.float64, copy=False)
    tree = cKDTree(coordinates)
    max_block_results = 2_000_000
    update_every = max(1, n_points // 100)
    block_rows = max(
        1, min(n_points, max_block_results // (k + 1), update_every)
    )
    next_progress = update_every
    for start in range(0, n_points, block_rows):
        stop = min(start + block_rows, n_points)
        _, nearest = tree.query(coordinates[start:stop], k=k + 1, p=1)
        nearest = np.asarray(nearest)

        # Put the target index last when returned. With duplicate points it may
        # be omitted; in that case the first k results are all valid neighbors.
        self_indices = np.arange(start, stop)[:, None]
        order = np.argsort(nearest == self_indices, axis=1, kind="stable")
        nearest = np.take_along_axis(nearest, order, axis=1)[:, :k]
        neighbors[start:stop, 1:, :] = points[nearest]
        if neighbor_classes is not None:
            neighbor_classes[start:stop, 1:] = classes[nearest]
        if neighbor_features is not None:
            neighbor_features[start:stop, 1:, :] = features[nearest]

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

    return _pack(neighbors.reshape(n_points, 3 * (k + 1)), neighbor_classes, neighbor_features)


def _pack(result, neighbor_classes, neighbor_features):
    """Форма возврата как раньше: result или (result, classes); features — последним."""
    out = (result,)
    if neighbor_classes is not None:
        out += (neighbor_classes,)
    if neighbor_features is not None:
        out += (neighbor_features,)
    return out[0] if len(out) == 1 else out
