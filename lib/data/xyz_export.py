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


def _write_csv(data: np.ndarray, filepath: str | Path, columns: list[str]) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", fmt="%.3f",
               header=",".join(columns), comments="")


def export_points_to_csv(points: np.ndarray, filepath: str | Path) -> None:
    """Экспортировать облако точек формы (N, 3) с колонками x,y,z."""
    matrix = _validate_numeric_matrix(points, "points")
    if matrix.shape[1] != 3:
        raise ValueError(f"points должен иметь форму (N, 3), получена форма {matrix.shape}")
    _write_csv(matrix, filepath, ["x", "y", "z"])


def export_k_folks_to_csv(neighbors: np.ndarray, filepath: str | Path) -> None:
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
    _write_csv(matrix, filepath, columns)
