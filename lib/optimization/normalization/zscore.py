"""
Z-score стандартизация координат по каждой оси: (x - mean) / std.

Параметры (mean, std) считаются по одной матрице точек (fit), а применяются
к другой (transform). Метод вызывается в каскаде после других преобразований,
поэтому работает только с numpy-матрицами, файлы не читает.

Пример:
    scaler = ZScoreScaler()
    scaler.fit(reference_points)            # 1) параметры по переданной матрице
    out = scaler.transform(other_points)    # 2) стандартизация другой матрицы, float32
    back = scaler.inverse_transform(out)    # обратно в метры, float64

Точки — матрица (N, 3): только координаты x y z, без столбца класса.
"""
from typing import Optional

import numpy as np


class ZScoreScaler:
    """Z-score стандартизация облака точек, отдельно по каждой оси.

    Поля после fit():
        mean_      — (3,) среднее по каждой оси, float64
        std_       — (3,) стандартное отклонение по каждой оси, float64
        n_samples_ — число точек, по которым считались параметры
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_samples_: Optional[int] = None

    def fit(self, points: np.ndarray) -> "ZScoreScaler":
        """Считает mean и std по матрице points (N, 3) и сохраняет их в поля.

        Считается во float64: на UTM-координатах (миллионы метров) во float32
        и среднее, и дисперсия теряют точность.
        Ось с нулевым разбросом (например, все z одинаковые) получает std = 1,
        чтобы не делить на ноль: такая ось только центрируется.
        """
        p = self._check(points, "fit")
        if len(p) == 0:
            raise ValueError("fit: пустая матрица, параметры не посчитать")
        mean = p.mean(axis=0)
        std = np.sqrt(((p - mean) ** 2).mean(axis=0))   # ddof=0
        std[std < self.eps] = 1.0
        self.mean_, self.std_, self.n_samples_ = mean, std, len(p)
        return self

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Стандартизирует points (M, 3) по сохранённым параметрам.

        points — любая матрица с тремя столбцами, M не обязано совпадать с N из fit().
        Считается во float64, результат — новый массив float32: после
        стандартизации значения порядка единиц, и точности float32 хватает
        (погрешность ~1e-7 std, это доли миллиметра). Исходный массив не меняется.
        """
        self._check_fitted()
        p = self._check(points, "transform")
        return ((p - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, points: np.ndarray) -> np.ndarray:
        """fit() и transform() по одной и той же матрице."""
        return self.fit(points).transform(points)

    def fit_transform_k_folks(self, neighbors: np.ndarray) -> np.ndarray:
        """Нормализовать результат k_folks, сохранив его двумерную форму.

        Ожидается матрица ``(N, 3 * (k + 1))``: в каждой строке записаны
        координаты target-точки и её соседей подряд. Статистики считаются по
        всем XYZ-тройкам матрицы, затем применяются отдельно к каждой тройке.
        """
        matrix = np.asarray(neighbors)
        if matrix.ndim != 2 or matrix.shape[1] == 0 or matrix.shape[1] % 3 != 0:
            raise ValueError(
                "fit_transform_k_folks: ожидается матрица формы "
                "(N, 3 * (k + 1)), получено "
                f"{matrix.shape}"
            )

        xyz = matrix.reshape(-1, 3)
        normalized = self.fit_transform(xyz)
        return normalized.reshape(matrix.shape)

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        """Обратно в исходные единицы: points * std + mean.

        Возвращает float64: исходные координаты бывают UTM (миллионы метров),
        во float32 они округлялись бы до 0.5 м.
        """
        self._check_fitted()
        p = self._check(points, "inverse_transform")
        return p * self.std_ + self.mean_

    def __repr__(self) -> str:
        if self.mean_ is None:
            return "ZScoreScaler(не обучен)"
        m = ", ".join(f"{v:.3f}" for v in self.mean_)
        s = ", ".join(f"{v:.3f}" for v in self.std_)
        return f"ZScoreScaler(mean=[{m}], std=[{s}], n={self.n_samples_})"

    def _check_fitted(self):
        if self.mean_ is None:
            raise RuntimeError("Сначала вызовите fit(points), чтобы посчитать параметры")

    def _check(self, points: np.ndarray, where: str) -> np.ndarray:
        p = np.asarray(points)
        if p.ndim != 2 or p.shape[1] != 3:
            hint = " Столбец класса передавать не нужно — только x y z." if p.ndim == 2 and p.shape[1] == 4 else ""
            raise ValueError(f"{where}: ожидается матрица (N, 3), получено {p.shape}.{hint}")
        if not np.isfinite(p).all():
            raise ValueError(f"{where}: в матрице есть NaN или inf")
        return p.astype(np.float64, copy=False)
