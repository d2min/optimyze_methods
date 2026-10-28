"""
Каскадный пайплайн очистки и метрики.

Шаги применяются последовательно к оставшимся точкам. Для каждой удалённой точки
запоминаются шаг (removed_by) и код шума (ASPRS: 7 — низкий/общий шум, 18 — высокий),
так что мусор можно сохранить отдельным .xyz и проверить глазами во вьюере.

Пример:
    from lib.data.xyz_loader import XYZLoader
    from lib.optimization.filtering import clean, ALS_PRESET

    classes, points = XYZLoader("cloud.xyz").load()
    res = clean(points, classes, ALS_PRESET)
    classes_clean, points_clean = classes[res.keep], points[res.keep]
"""
import time
from dataclasses import dataclass

import numpy as np

from . import clusters, height
from ._common import density_scale

LOW_NOISE, HIGH_NOISE = height.LOW_NOISE, height.HIGH_NOISE


def _mask_step(fn):
    """Шаг, возвращающий keep-маску → коды (0 / LOW_NOISE)."""
    def run(points, classes, **kw):
        keep = fn(points, **kw)
        return np.where(keep, 0, LOW_NOISE).astype(np.int16)
    return run


def _hag_step(points, classes, ground_class=2, verbose=True, **kw):
    if classes is None or not np.any(classes == ground_class):
        if verbose:
            print(f"    (пропущено: нет точек класса {ground_class})")
        return np.zeros(len(points), np.int16)
    return height.hag_noise(points, classes == ground_class, **kw)


def _resolve_scaled(params: dict, scale: float) -> dict:
    """Параметры с суффиксом _s задаются в единицах шага облака (density_scale):
    {"voxel_s": 2.5} → {"voxel": 2.5 * s}. Можно задать и нижнюю границу в метрах
    одновременно: {"isolation_s": 4, "isolation": 3.0} → max(3.0, 4 * s)."""
    out = dict(params)
    for k in [k for k in params if k.endswith("_s")]:
        base = k[:-2]
        v = params[k] * scale
        out[base] = max(v, params[base]) if base in params else v
        del out[k]
    return out


STEPS = {
    "hag": _hag_step,
    "column_gap": lambda p, c, **kw: height.column_gap_noise(p, **kw),
    "clusters": _mask_step(clusters.voxel_clusters),
}

# Пресет для аэролазерной съёмки (ALS). Порядок: от грубого и надёжного к тонкому.
# Настроен на высокую точность (лучше пропустить мусор, чем срезать сцену);
# расстояния привязаны к шагу облака s, поэтому работает от 2 до 300 точек/м².
ALS_PRESET = [
    # 1. Под рельефом больше чем на 1 м И изолированы (> max(1 м, 2s)) —
    #    многолучевые отражения. Нужен класс земли (2); если его нет — шаг пропускается.
    {"name": "hag", "below": -1.0, "isolation": 1.0, "isolation_s": 2.0},
    # 2. Вертикальные выбросы: в колонке 5×5 м отделены разрывом > 3 м, сегмент < 5 точек,
    #    изолированы (> max(3 м, 4s)) и в радиусе max(10 м, 8s) не больше 3 соседей —
    #    птицы, облака, атмосферные отражения. Фасады высоток и провода не трогает.
    {"name": "column_gap", "cell": 5.0, "gap": 3.0, "min_segment": 5,
     "isolation": 3.0, "isolation_s": 4.0,
     "confirm_radius": 10.0, "confirm_radius_s": 8.0, "confirm_max": 3},
    # 3. Оторванные группы: < 10 точек, отделённые от всего остального на ≥ max(5 м, 5s)
    #    (стаи птиц, сгустки в воздухе). Провода и стены связаны с соседями — остаются.
    {"name": "clusters", "voxel": 5.0, "voxel_s": 5.0, "min_cluster_size": 10},
]


@dataclass
class CleanResult:
    keep: np.ndarray        # bool (N,) — True = точка остаётся
    codes: np.ndarray       # int16 (N,) — 0 / 7 / 18
    removed_by: np.ndarray  # int16 (N,) — номер шага, -1 если осталась
    steps: list


def clean(points: np.ndarray, classes: np.ndarray = None, steps=ALS_PRESET,
          verbose: bool = True) -> CleanResult:
    n = len(points)
    alive = np.arange(n)
    codes = np.zeros(n, np.int16)
    removed_by = np.full(n, -1, np.int16)
    scale = None
    if any(k.endswith("_s") for st in steps for k in st):
        scale = density_scale(points)
        if verbose:
            print(f"  шаг облака s = {scale:.3f} м")
    for si, step in enumerate(steps):
        params = {k: v for k, v in step.items() if k != "name"}
        if scale is not None:
            params = _resolve_scaled(params, scale)
        t = time.perf_counter()
        sub_cls = None if classes is None else classes[alive]
        if step["name"] == "hag":
            c = STEPS[step["name"]](
                points[alive], sub_cls, verbose=verbose, **params
            )
        else:
            c = STEPS[step["name"]](points[alive], sub_cls, **params)
        bad = c > 0
        codes[alive[bad]] = c[bad]
        removed_by[alive[bad]] = si
        if verbose:
            print(f"  [{si}] {step['name']:<11} удалено {bad.sum():>8} "
                  f"({100 * bad.mean():6.3f}%)  осталось {(~bad).sum():>9}  "
                  f"{time.perf_counter() - t:6.1f} c")
        alive = alive[~bad]
    return CleanResult(removed_by < 0, codes, removed_by, list(steps))


def metrics(keep: np.ndarray, is_noise: np.ndarray) -> dict:
    """Метрики относительно разметки; положительный класс — мусор.

    precision  — доля удалённого, которая действительно мусор (не портим сцену);
    recall     — доля мусора, которую удалили;
    kept_valid — доля полезных точек, которые сохранились.
    """
    removed = ~keep
    tp = int(np.sum(removed & is_noise))
    fp = int(np.sum(removed & ~is_noise))
    fn = int(np.sum(keep & is_noise))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "kept_valid": 1 - fp / max(int((~is_noise).sum()), 1),
            "tp": tp, "fp": fp, "fn": fn}
