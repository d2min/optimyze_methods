"""
Нормали к поверхности по локальной окрестности (PCA).

Для каждой точки берутся k ближайших соседей (евклидово расстояние, сама точка
входит в окрестность), считается ковариационная матрица 3×3. Нормаль —
собственный вектор с наименьшим собственным значением: направление, вдоль
которого окрестность «тоньше» всего. Знак выбирается так, чтобы nz ≥ 0
(нормаль смотрит вверх). У вертикальных стен nz ≈ 0, и там направление вдоль
горизонтали выбирается условно — это ожидаемо.
"""
import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(points: np.ndarray, k: int = 20, chunk: int = 200_000) -> np.ndarray:
    """Единичные нормали (N, 3) float32, nz ≥ 0.

    k     — размер окрестности. 20 — стандартный выбор: достаточно, чтобы
            шум не разворачивал нормаль, и достаточно мало, чтобы не сглаживать рёбра.
    chunk — сколько точек обрабатывать за раз (ограничивает память).
    Если точек меньше 3, возвращается (0, 0, 1).
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points должен иметь форму (N, 3), получена форма {p.shape}")
    n = len(p)
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, 2] = 1.0
    if n < 3:
        return out
    p = p - p.mean(axis=0)                  # точность на UTM-координатах
    k = int(min(max(k, 3), n))
    tree = cKDTree(p)
    for s in range(0, n, chunk):
        q = p[s:s + chunk]
        _, idx = tree.query(q, k=k, workers=-1)
        nb = p[idx]                                       # (m, k, 3)
        X = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", X, X)
        _, vecs = np.linalg.eigh(cov)                     # по возрастанию
        nrm = vecs[:, :, 0]
        nrm *= np.where(nrm[:, 2:3] < 0, -1.0, 1.0)       # nz >= 0
        out[s:s + chunk] = nrm
    return out
