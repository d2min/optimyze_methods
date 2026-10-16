"""Общие утилиты фильтров: перевод в локальные координаты, kd-дерево, шаг облака."""
import numpy as np
from scipy.spatial import cKDTree


def as_local(points: np.ndarray) -> np.ndarray:
    """float64 + вычитание центра облака.

    UTM-координаты (миллионы метров) во float32 имеют шаг до 0.5 м — соседи
    "слипаются". Центрирование во float64 возвращает сантиметровую точность
    (если исходные данные были загружены во float64).
    """
    p = np.asarray(points, dtype=np.float64)
    return p - p.mean(axis=0)


def build_tree(points: np.ndarray) -> cKDTree:
    return cKDTree(points, leafsize=32, balanced_tree=False, compact_nodes=False)


def knn_mean_dist(points: np.ndarray, k: int, tree: cKDTree = None) -> np.ndarray:
    """Средняя дистанция до k ближайших соседей (без самой точки)."""
    tree = tree or build_tree(points)
    d, _ = tree.query(points, k=k + 1, workers=-1)
    return d[:, 1:].mean(axis=1)


def density_scale(points: np.ndarray, k: int = 8, q: float = 0.9) -> float:
    """Характерный шаг облака: q-квантиль средней kNN-дистанции.

    Единица для порогов ALS_PRESET: параметры с суффиксом _s умножаются на s.
    Высокий квантиль (а не медиана) — потому что плотность в ALS неоднородна
    (кусты ~2 см, земля ~12 см), и порог по медиане резал бы разреженную землю.
    """
    return float(np.quantile(knn_mean_dist(points, k), q))
