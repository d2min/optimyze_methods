"""Экспорт массивов облака точек и k_folks в CSV и NPY."""
import os
from pathlib import Path
import tempfile

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


def _write_npy(data: np.ndarray, filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)


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


def export_points_to_npy(points: np.ndarray, filepath: str | Path,
                         classes: np.ndarray = None,
                         features: np.ndarray = None) -> None:
    """Экспортировать точки в NPY: class (если есть), XYZ, признаки (если есть)."""
    matrix = _validate_numeric_matrix(points, "points")
    if matrix.shape[1] != 3:
        raise ValueError(f"points должен иметь форму (N, 3), получена форма {matrix.shape}")
    if classes is None and features is None:
        _write_npy(matrix, filepath)
        return

    parts = []
    if classes is not None:
        labels = np.asarray(classes)
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError(
                f"classes должен иметь форму ({len(matrix)},), получена форма {labels.shape}"
            )
        if not np.issubdtype(labels.dtype, np.number):
            raise TypeError("classes должен содержать числовые метки")
        if not np.isfinite(labels).all():
            raise ValueError("classes не должен содержать NaN или inf")
        parts.append(labels[:, None])
    parts.append(matrix)
    if features is not None:
        feature_matrix = _validate_numeric_matrix(features, "features")
        if len(feature_matrix) != len(matrix):
            raise ValueError(
                f"features должен иметь {len(matrix)} строк, "
                f"получено {len(feature_matrix)}"
            )
        parts.append(feature_matrix)
    if features is None:
        _write_npy(np.column_stack(parts), filepath)
        return

    dtype = np.result_type(matrix.dtype, feature_matrix.dtype, np.float32)
    _write_npy(
        np.column_stack([part.astype(dtype, copy=False) for part in parts]),
        filepath,
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


def export_k_folks_to_npy(neighbors: np.ndarray, filepath: str | Path,
                          classes: np.ndarray = None) -> None:
    """Экспортировать матрицу k_folks формы (N, 3 * (k + 1)) в NPY."""
    matrix = _validate_numeric_matrix(neighbors, "neighbors")
    if matrix.shape[1] == 0 or matrix.shape[1] % 3 != 0:
        raise ValueError(
            "neighbors должен иметь форму (N, 3 * (k + 1)), "
            f"получена форма {matrix.shape}"
        )
    if classes is None:
        _write_npy(matrix, filepath)
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
    _write_npy(data, filepath)


def k_folks_feature_columns(k: int, feature_names, with_class: bool = True) -> list[str]:
    """Названия столбцов экспорта k_folks с признаками.

    Группа на каждую точку: x, y, z, затем признаки, затем class (если есть).
    Порядок групп: target, neighbor_1, ..., neighbor_k.
    """
    names = []
    for prefix in ["target"] + [f"neighbor_{i}" for i in range(1, k + 1)]:
        names += [f"{prefix}_{c}" for c in ("x", "y", "z", *feature_names)]
        if with_class:
            names.append(f"{prefix}_class")
    return names


def _k_folks_feature_matrix(neighbors: np.ndarray, features: np.ndarray,
                            classes: np.ndarray = None) -> np.ndarray:
    matrix = _validate_numeric_matrix(neighbors, "neighbors")
    if matrix.shape[1] == 0 or matrix.shape[1] % 3 != 0:
        raise ValueError(
            "neighbors должен иметь форму (N, 3 * (k + 1)), "
            f"получена форма {matrix.shape}"
        )
    groups = matrix.shape[1] // 3
    feats = np.asarray(features)
    if feats.ndim != 3 or feats.shape[:2] != (len(matrix), groups):
        raise ValueError(
            f"features должен иметь форму ({len(matrix)}, {groups}, F), "
            f"получена форма {feats.shape}"
        )
    if not np.issubdtype(feats.dtype, np.number) or not np.isfinite(feats).all():
        raise ValueError("features должен содержать конечные числа")
    dtype = np.result_type(matrix.dtype, feats.dtype, np.float32)
    parts = [matrix.reshape(len(matrix), groups, 3).astype(dtype, copy=False),
             feats.astype(dtype, copy=False)]
    if classes is not None:
        labels = np.asarray(classes)
        if labels.shape != (len(matrix), groups):
            raise ValueError(
                f"classes должен иметь форму {(len(matrix), groups)}, получена форма {labels.shape}"
            )
        if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
            raise ValueError("classes должен содержать конечные числовые метки")
        parts.append(labels[..., None].astype(dtype, copy=False))
    return np.concatenate(parts, axis=2).reshape(len(matrix), -1)


def export_k_folks_features_to_npy(neighbors: np.ndarray, features: np.ndarray,
                                   filepath: str | Path,
                                   classes: np.ndarray = None) -> None:
    """Экспортировать k_folks вместе с признаками точек в NPY.

    neighbors — (N, 3 * (k + 1)) из k_folks; features — (N, k + 1, F) из
    k_folks(..., features=...); classes — (N, k + 1) или None.
    Результат (N, (k + 1) * G), где группа G = x, y, z, признаки..., class.
    Тип — float32 (float64, если координаты во float64).
    """
    _write_npy(_k_folks_feature_matrix(neighbors, features, classes), filepath)


def export_k_folks_feature_chunks_to_npy(
    chunks,
    filepath: str | Path,
    *,
    point_count: int,
    k: int,
    feature_count: int,
    dtype,
    with_classes: bool = True,
) -> None:
    """Записать ``(start, chunk)`` блоки iter_k_folks в NPY без сборки в RAM."""
    if point_count < 0 or k < 0 or feature_count < 0:
        raise ValueError("point_count, k и feature_count должны быть неотрицательными")
    group_width = 3 + feature_count + int(with_classes)
    output_shape = (point_count, (k + 1) * group_width)
    output_dtype = np.dtype(dtype)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as output:
            np.lib.format.write_array_header_2_0(
                output,
                {
                    "descr": np.lib.format.dtype_to_descr(output_dtype),
                    "fortran_order": False,
                    "shape": output_shape,
                },
            )
            written = 0
            for start, chunk in chunks:
                if start != written:
                    raise ValueError(
                        f"Получен блок с началом {start}, ожидалось {written}"
                    )
                if with_classes:
                    if not isinstance(chunk, tuple) or len(chunk) != 3:
                        raise ValueError(
                            "Каждый блок должен содержать neighbors, classes и features"
                        )
                    neighbors, classes, features = chunk
                else:
                    if not isinstance(chunk, tuple) or len(chunk) != 2:
                        raise ValueError("Каждый блок должен содержать neighbors и features")
                    neighbors, features = chunk
                    classes = None
                data = _k_folks_feature_matrix(neighbors, features, classes)
                stop = written + len(data)
                if stop > point_count:
                    raise ValueError("Блоки содержат больше строк, чем point_count")
                if data.dtype != output_dtype:
                    data = data.astype(output_dtype, copy=False)
                data.tofile(output)
                written = stop
            if written != point_count:
                raise ValueError(
                    f"Блоки содержат {written} строк, ожидалось {point_count}"
                )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def export_k_folks_features_to_csv(neighbors: np.ndarray, features: np.ndarray,
                                   filepath: str | Path, feature_names,
                                   classes: np.ndarray = None) -> None:
    """То же, что export_k_folks_features_to_npy, но в CSV с заголовком."""
    data = _k_folks_feature_matrix(neighbors, features, classes)
    k = neighbors.shape[1] // 3 - 1
    columns = k_folks_feature_columns(k, feature_names, with_class=classes is not None)
    per_group = ["%.10f"] * (3 + len(feature_names)) + (["%d"] if classes is not None else [])
    _write_csv(data, filepath, columns, fmt=per_group * (k + 1))
