"""
Дополнительные признаки точек для модели.

    from optimization.features import compute_features
    feats = compute_features(points, classes)
    feats.values   # (N, 5) float32: hag, nx, ny, nz, ground
    feats.columns  # ("hag", "nx", "ny", "nz", "ground")

hag    — высота над землёй, м (может быть < 0: точка ниже найденной земли)
nx..nz — единичная нормаль к поверхности, nz ≥ 0
ground — 1, если точка — земля, иначе 0

Земля: класс 2 из разметки, если он есть в classes, иначе CSF по геометрии
(ground_source="auto"). ground_source="csf" — всегда CSF, без разметки.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ground import ground_mask, csf_ground, auto_cloth_resolution, GROUND_CLASS
from .normals import estimate_normals
from ..filtering.height import height_above_ground

FEATURE_COLUMNS = ("hag", "nx", "ny", "nz", "ground")


@dataclass
class PointFeatures:
    values: np.ndarray        # (N, 5) float32, столбцы FEATURE_COLUMNS
    columns: tuple            # названия столбцов
    ground_source: str        # "class" или "csf" — откуда взята земля


def compute_features(points: np.ndarray, classes: Optional[np.ndarray] = None,
                     ground_source: str = "auto", k_normals: int = 20,
                     hag_cell: float = 2.0, **csf_params) -> PointFeatures:
    """HAG, нормаль и флаг земли для каждой точки (N, 3).

    hag_cell — размер ячейки модели рельефа, м: в ячейке берётся медиана высот
               точек земли, между ячейками — интерполяция по 4 ближайшим.
    csf_params — параметры csf_ground (cloth_resolution, rigidness, ...).
    """
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points должен иметь форму (N, 3), получена форма {p.shape}")
    if classes is not None and len(classes) != len(p):
        raise ValueError(f"classes должен иметь длину {len(p)}, получено {len(classes)}")

    n = len(p)
    values = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    if n == 0:
        return PointFeatures(values, FEATURE_COLUMNS, "none")

    ground, source = ground_mask(p, classes, source=ground_source, **csf_params)
    if ground.any():
        values[:, 0] = height_above_ground(p, ground, cell=hag_cell)
    else:  # земля не найдена (например, облако из одних крыш) — HAG от минимума
        values[:, 0] = p[:, 2] - p[:, 2].min()
    values[:, 1:4] = estimate_normals(p, k=k_normals)
    values[:, 4] = ground
    return PointFeatures(values, FEATURE_COLUMNS, source)


__all__ = ["compute_features", "PointFeatures", "FEATURE_COLUMNS",
           "ground_mask", "csf_ground", "auto_cloth_resolution", "estimate_normals",
           "GROUND_CLASS"]
