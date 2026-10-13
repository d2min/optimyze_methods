"""
Отклонение от локальной поверхности.

Для каждой точки по k соседям (без неё самой) строится плоскость (PCA),
считается расстояние точки до неё, нормированное на локальный разброс.
Ловит точки, «висящие» рядом с поверхностью: шум у земли, mixed pixels на краях,
отлетевшие от стен/крыш точки.

Ограничение: в кронах деревьев и проводах «поверхности» нет — там скор высокий
и у нормальных точек. Поэтому порог берите консервативный (5+) или применяйте
только к точкам у земли/строений.
"""
import numpy as np

from ._common import as_local, build_tree


def surface_score(points: np.ndarray, k: int = 20, chunk: int = 200_000) -> np.ndarray:
    """Нормированное расстояние до локальной плоскости. Считается кусками по chunk точек."""
    p = as_local(points)
    tree = build_tree(p)
    out = np.empty(len(p))
    for s in range(0, len(p), chunk):
        q = p[s:s + chunk]
        _, idx = tree.query(q, k=k + 1, workers=-1)
        nb = p[idx[:, 1:]]                         # (m, k, 3)
        c = nb.mean(axis=1)
        X = nb - c[:, None, :]
        cov = np.einsum("nki,nkj->nij", X, X) / k
        w, v = np.linalg.eigh(cov)                 # собственные значения по возрастанию
        normal = v[:, :, 0]
        dist = np.abs(np.einsum("ni,ni->n", q - c, normal))
        sigma = np.sqrt(np.maximum(w[:, 0], 0))
        # «пол» масштаба: на идеально ровной сетке sigma≈0, иначе делили бы на ноль
        floor = 0.05 * np.linalg.norm(X, axis=2).mean(axis=1)
        out[s:s + chunk] = dist / np.maximum(sigma, floor)
    return out


def surface_residual(points: np.ndarray, k: int = 20, max_score: float = 5.0) -> np.ndarray:
    """Маска keep: оставить точки со score <= max_score."""
    return surface_score(points, k=k) <= max_score
