"""Экспорт массивов облака точек и k_folks в CSV."""
from pathlib import Path

import numpy as np


def _validate_numeric_matrix(data: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(data)
    if matrix.ndim != 2:
        raise ValueError(f"{name} должен быть двумерной матрицей, получена форма {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} должен содержать числовые данные")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} не должен содержать NaN или inf")
    return matrix


def _write_csv(data: np.ndarray, filepath: str | Path, columns: list[str],
               fmt="%.10f") -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", fmt=fmt,
               header=",".join(columns), comments="")


def export_points_to_csv(points: np.ndarray, filepath: str | Path,
                         classes: np.ndarray = None) -> None:
    """Экспортировать облако точек формы (N, 3) с необязательной меткой."""
    matrix = _validate_numeric_matrix(points, "points")
    if matrix.shape[1] != 3:
        raise ValueError(f"points должен иметь форму (N, 3), получена форма {matrix.shape}")
    if classes is None:
        _write_csv(matrix, filepath, ["x", "y", "z"])
        return

    labels = np.asarray(classes)
    if labels.ndim != 1 or len(labels) != len(matrix):
        raise ValueError(
            f"classes должен иметь форму ({len(matrix)},), получена форма {labels.shape}"
        )
    if not np.issubdtype(labels.dtype, np.number):
        raise TypeError("classes должен содержать числовые метки")
    if not np.isfinite(labels).all():
        raise ValueError("classes не должен содержать NaN или inf")
    data = np.column_stack((labels, matrix))
    _write_csv(
        data, filepath, ["class", "x", "y", "z"],
        fmt=["%d", "%.10f", "%.10f", "%.10f"],
    )


def export_k_folks_to_csv(neighbors: np.ndarray, filepath: str | Path,
                          classes: np.ndarray = None) -> None:
    """Экспортировать матрицу k_folks формы (N, 3 * (k + 1)).

    В CSV каждая точка и каждый сосед получают отдельные координатные колонки:
    target_x,target_y,target_z,neighbor_1_x,...,neighbor_k_z.
    """
    matrix = _validate_numeric_matrix(neighbors, "neighbors")
    if matrix.shape[1] == 0 or matrix.shape[1] % 3 != 0:
        raise ValueError(
            "neighbors должен иметь форму (N, 3 * (k + 1)), "
            f"получена форма {matrix.shape}"
        )

    columns = ["target_x", "target_y", "target_z"]
    for neighbor_index in range(1, matrix.shape[1] // 3):
        columns.extend([
            f"neighbor_{neighbor_index}_x",
            f"neighbor_{neighbor_index}_y",
            f"neighbor_{neighbor_index}_z",
        ])
    if classes is None:
        _write_csv(matrix, filepath, columns)
        return

    labels = np.asarray(classes)
    expected_shape = (len(matrix), matrix.shape[1] // 3)
    if labels.shape != expected_shape:
        raise ValueError(
            f"classes должен иметь форму {expected_shape}, получена форма {labels.shape}"
        )
    if not np.issubdtype(labels.dtype, np.number):
        raise TypeError("classes должен содержать числовые метки")
    if not np.isfinite(labels).all():
        raise ValueError("classes не должен содержать NaN или inf")

    coordinate_groups = matrix.reshape(len(matrix), -1, 3)
    data = np.concatenate((coordinate_groups, labels[..., None]), axis=2).reshape(
        len(matrix), -1
    )
    labeled_columns = ["target_x", "target_y", "target_z", "target_class"]
    fmt = ["%.10f", "%.10f", "%.10f", "%d"]
    for neighbor_index in range(1, matrix.shape[1] // 3):
        labeled_columns.extend([
            f"neighbor_{neighbor_index}_x",
            f"neighbor_{neighbor_index}_y",
            f"neighbor_{neighbor_index}_z",
            f"neighbor_{neighbor_index}_class",
        ])
        fmt.extend(["%.10f", "%.10f", "%.10f", "%d"])
    _write_csv(data, filepath, labeled_columns, fmt=fmt)
