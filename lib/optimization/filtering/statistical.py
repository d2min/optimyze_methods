"""
Классические фильтры по соседству: SOR, ROR, LOF.

Сильные стороны: просты, не требуют ничего кроме xyz.
Слабые: один глобальный порог на всё облако. В ALS плотность скачет в разы
(земля / кусты / кроны / стены), поэтому порог либо режет разреженные участки,
либо пропускает шум в плотных.
"""
import numpy as np

from ._common import as_local, build_tree, knn_mean_dist, neighbor_counts


def sor(points: np.ndarray, k: int = 8, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical Outlier Removal (как в PCL/Open3D/PDAL filters.outlier).

    Выброс — точка, у которой средняя дистанция до k соседей
    больше mean + std_ratio * std по всему облаку.
    """
    p = as_local(points)
    md = knn_mean_dist(p, k)
    return md <= md.mean() + std_ratio * md.std()


def ror(points: np.ndarray, radius: float, min_neighbors: int = 4) -> np.ndarray:
    """Radius Outlier Removal: оставить точки, у которых >= min_neighbors соседей в radius.

    radius удобно задавать в единицах density_scale(points).
    """
    p = as_local(points)
    return neighbor_counts(p, radius, build_tree(p)) >= min_neighbors


def lof(points: np.ndarray, k: int = 20, contamination="auto") -> np.ndarray:
    """Local Outlier Factor (scikit-learn).

    Сравнивает плотность точки с плотностью её соседей, поэтому мягче к перепадам
    плотности, чем SOR. Медленнее; на десятках миллионов точек — по тайлам.
    """
    from sklearn.neighbors import LocalOutlierFactor
    p = as_local(points)
    pred = LocalOutlierFactor(n_neighbors=k, contamination=contamination,
                              n_jobs=-1).fit_predict(p)
    return pred == 1
