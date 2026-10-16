"""
Фильтрация шума (мусора) в облаках точек.

Все фильтры принимают points (N, 3) — как возвращает XYZLoader.load() — и отдают
маску keep (True = оставить) или коды шума (0 / 7 / 18 по ASPRS LAS).
Подробности, выбор методов и ссылки на статьи — в README.md рядом.
"""
from .clusters import voxel_clusters
from .height import (column_gap, column_gap_noise, height_above_ground, hag_noise,
                     LOW_NOISE, HIGH_NOISE)
from .pipeline import clean, metrics, ALS_PRESET, CleanResult
from ._common import density_scale

__all__ = [
    "voxel_clusters",
    "column_gap", "column_gap_noise", "height_above_ground", "hag_noise",
    "clean", "metrics", "ALS_PRESET", "CleanResult", "density_scale",
    "LOW_NOISE", "HIGH_NOISE",
]
