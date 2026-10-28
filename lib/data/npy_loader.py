"""Загрузчик облаков точек из NumPy-файлов."""

from pathlib import Path

import numpy as np


class NPYLoader:
    """Загружает массивы (N, 4) в формате class, x, y, z или (N, 3) xyz."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        if self.filepath.suffix.lower() != ".npy":
            raise ValueError(f"Ожидался файл .npy: {filepath}")

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(self.filepath, allow_pickle=False)
        if data.ndim != 2 or data.shape[1] not in (3, 4):
            raise ValueError(
                "Неверная форма NPY-массива. Ожидались 3 колонки (x, y, z) "
                f"или 4 (class, x, y, z), получено {data.shape}"
            )
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("NPY-массив должен содержать числовые данные")
        if not np.isfinite(data).all():
            raise ValueError("NPY-массив не должен содержать NaN или inf")
        if data.shape[1] == 4:
            classes = data[:, 0].astype(np.int32)
            points = np.ascontiguousarray(data[:, 1:4])
        else:
            classes = np.zeros(len(data), dtype=np.int32)
            points = np.ascontiguousarray(data)
        return classes, points