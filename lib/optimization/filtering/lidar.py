"""
Фильтры для наземного/мобильного LiDAR (сенсор в известной точке, есть интенсивность):
снег, дождь, пыль, туман. Для ALS-данных (аэросъёмка, без положения сенсора и
интенсивности) не применимы — оставлены на случай TLS/MLS-съёмок.

DROR  — Charron et al., 2018 (радиус растёт с дальностью)
DSOR  — Kurup & Bos, 2021 (порог SOR растёт с дальностью)
LIOR  — Park et al., 2020 (низкая интенсивность + проверка плотности)
LIDROR — Afzalaghaeinaeini et al., 2022 (LIOR + DROR, для пыли)
"""
import numpy as np

from ._common import as_local, build_tree as _tree, knn_mean_dist, neighbor_counts as _nc


def ranges(xyz, sensor_origin=(0.0, 0.0, 0.0)):
    return np.linalg.norm(np.asarray(xyz, float) - np.asarray(sensor_origin, float), axis=1)


def neighbor_counts(xyz, radius, tree=None):
    return _nc(xyz, radius, tree or _tree(xyz))


# ---------------------------------------------------------------------------
# 2. Дальностно-адаптивные (LiDAR): DROR и DSOR
# ---------------------------------------------------------------------------

def dror(xyz: np.ndarray, sensor_origin=(0, 0, 0), angular_res_deg: float = 0.2,
         beta: float = 3.0, min_radius: float = 0.04, min_neighbors: int = 3,
         **_) -> np.ndarray:
    """Dynamic Radius Outlier Removal (Charron et al., 2018).
    Радиус поиска растёт с дальностью: R = max(min_radius, beta * range * alpha),
    alpha — угловое разрешение сенсора в радианах (расстояние между соседними лучами).
    """
    r = ranges(xyz, sensor_origin)
    alpha = np.deg2rad(angular_res_deg)
    radius = np.maximum(min_radius, beta * r * alpha)
    return neighbor_counts(xyz, radius) >= min_neighbors


def dsor(xyz: np.ndarray, sensor_origin=(0, 0, 0), k: int = 5, std_ratio: float = 0.01,
         range_mult: float = 0.05, **_) -> np.ndarray:
    """Dynamic Statistical Outlier Removal (Kurup & Bos, 2021).
    Глобальный порог SOR T = mean + std_ratio*std, затем масштабируется дальностью:
    T_d = T * range_mult * range. Выброс, если средняя kNN-дистанция > T_d.
    Параметры по умолчанию — как в оригинальной статье (для Ouster OS1-64, снег).
    """
    md = knn_mean_dist(xyz, k)
    thr = md.mean() + std_ratio * md.std()
    r = ranges(xyz, sensor_origin)
    return md <= thr * range_mult * r


# ---------------------------------------------------------------------------
# 3. С интенсивностью (пыль/снег/дождь дают слабый отражённый сигнал)
# ---------------------------------------------------------------------------

def lior(xyz: np.ndarray, intensity: np.ndarray, intensity_thr: float,
         radius: float = 0.1, min_neighbors: int = 3, **_) -> np.ndarray:
    """Low-Intensity Outlier Removal (Park et al., 2020).
    1) кандидаты в мусор — точки с intensity < intensity_thr;
    2) кандидата «спасаем», если вокруг него плотно (ROR по всему облаку) —
       это тёмный, но реальный объект.
    """
    keep = intensity >= intensity_thr
    cand = np.flatnonzero(~keep)
    if cand.size:
        tree = _tree(xyz)
        # запрос по кандидатам на дереве всего облака; neighbor_counts уже вычитает саму точку
        keep[cand] = neighbor_counts(xyz[cand], radius, tree=tree) >= min_neighbors
    return keep


def lidror(xyz: np.ndarray, intensity: np.ndarray, intensity_thr: float,
           sensor_origin=(0, 0, 0), angular_res_deg: float = 0.2, beta: float = 3.0,
           min_radius: float = 0.04, min_neighbors: int = 3, **_) -> np.ndarray:
    """LIDROR (Afzalaghaeinaeini et al., 2022): LIOR, но второй шаг — DROR, а не ROR.
    Предложен именно для пыли (off-road, карьеры, стройка)."""
    keep = intensity >= intensity_thr
    cand = np.flatnonzero(~keep)
    if cand.size:
        tree = _tree(xyz)
        r = ranges(xyz[cand], sensor_origin)
        radius = np.maximum(min_radius, beta * r * np.deg2rad(angular_res_deg))
        cnt = np.asarray(tree.query_ball_point(xyz[cand], r=radius,
                                               return_length=True, workers=-1)) - 1
        keep[cand] = cnt >= min_neighbors
    return keep


def auto_intensity_threshold(intensity: np.ndarray, sample_mask: np.ndarray | None = None,
                             quantile: float | None = None) -> float:
    """Порог интенсивности.
    - Если есть разметка мусора (sample_mask=True для мусора) — берём 95-й перцентиль мусора.
    - Иначе — порог Оцу по гистограмме (работает, если распределение бимодально)
      или заданный квантиль.
    """
    if sample_mask is not None and sample_mask.any():
        return float(np.percentile(intensity[sample_mask], 95))
    if quantile is not None:
        return float(np.quantile(intensity, quantile))
    # Оцу в лог-шкале: распределение интенсивности сильно скошено вправо,
    # в линейной шкале порог уезжает в середину «полезных» точек.
    x = np.log1p(np.maximum(intensity, 0))
    hist, edges = np.histogram(x, bins=256)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    return float(np.expm1(edges[np.nanargmax(s) + 1]))


