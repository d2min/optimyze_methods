"""
Поиск точек земли: готовая разметка (класс 2) или CSF.

CSF (Cloth Simulation Filter, Zhang et al., 2016, Remote Sensing 8(6):501) —
облако переворачивается вверх ногами, сверху на него «падает» виртуальная ткань
с заданной жёсткостью. Ткань ложится на рельеф, но не проваливается между
домами и деревьями. Точки, оказавшиеся ближе class_threshold к ткани, — земля.
Используется официальная реализация авторов на C++ (пакет cloth-simulation-filter).
"""
from typing import Optional, Tuple

import numpy as np

GROUND_CLASS = 2  # ASPRS LAS: Ground


def _local(points: np.ndarray) -> np.ndarray:
    """float64 + центрирование: на UTM-координатах float32 даёт шаг до 0.5 м."""
    p = np.asarray(points, dtype=np.float64)
    return p - p.mean(axis=0)


def csf_ground(points: np.ndarray,
               cloth_resolution: Optional[float] = None,
               rigidness: int = 1,
               slope_smooth: bool = True,
               class_threshold: float = 0.5,
               iterations: int = 500,
               time_step: float = 0.65) -> np.ndarray:
    """Маска земли (N,) по CSF.

    cloth_resolution — шаг ткани, м. None — подбирается по плотности облака
                       (см. auto_cloth_resolution).
    rigidness        — жёсткость ткани: 1 — крутые склоны, 2 — смешанный рельеф,
                       3 — равнина и город. По умолчанию 1: на всех наших облаках
                       не хуже 2 и 3, а на обрывах Niagara заметно лучше.
    slope_smooth     — постобработка ткани на крутых склонах (обрывы, ущелья).
    class_threshold  — точка ближе этого расстояния к ткани (м) считается землёй.
    """
    try:
        import CSF
    except ImportError as e:  # pragma: no cover
        raise ImportError("Для поиска земли нужен пакет cloth-simulation-filter: "
                          "pip install cloth-simulation-filter") from e

    p = _local(points)
    if len(p) == 0:
        return np.zeros(0, dtype=bool)
    if cloth_resolution is None:
        cloth_resolution = auto_cloth_resolution(p)

    csf = CSF.CSF()
    csf.params.bSloopSmooth = bool(slope_smooth)
    csf.params.cloth_resolution = float(cloth_resolution)
    csf.params.rigidness = int(rigidness)
    csf.params.class_threshold = float(class_threshold)
    csf.params.interations = int(iterations)
    csf.params.time_step = float(time_step)
    csf.setPointCloud(p)

    ground_idx, off_ground_idx = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground_idx, off_ground_idx, False)  # False — не писать файл ткани

    mask = np.zeros(len(p), dtype=bool)
    mask[np.asarray(ground_idx, dtype=np.int64)] = True
    return mask


def auto_cloth_resolution(points: np.ndarray, lo: float = 0.5, hi: float = 1.5) -> float:
    """Шаг ткани по плотности: ~2 средних расстояния между точками в плане,
    в пределах [lo, hi] м. Слишком мелкая ткань проваливается в промежутки между
    точками земли, слишком крупная срезает бугры, бровки и обрывы.

    Подобрано на наших облаках (Univer2019, Niagara, Ogorod, terrain_points, Test_02):
    на холмистой синтетике шаг 1–2 м ломает HAG (ошибка до 5 м), а на Niagara
    шаг 2 м вдвое хуже, чем 1.5 м."""
    p = np.asarray(points, dtype=np.float64)
    area = max(np.ptp(p[:, 0]) * np.ptp(p[:, 1]), 1e-9)
    spacing = np.sqrt(area / max(len(p), 1))
    return float(np.clip(2.0 * spacing, lo, hi))


def ground_mask(points: np.ndarray, classes: Optional[np.ndarray] = None,
                source: str = "auto", ground_class: int = GROUND_CLASS,
                **csf_params) -> Tuple[np.ndarray, str]:
    """Маска земли и откуда она взята.

    source="auto"  — класс ground_class, если в classes есть хоть одна такая точка,
                     иначе CSF;
    source="class" — только разметка (ошибка, если земли в разметке нет);
    source="csf"   — только CSF, разметка не используется.
    Возвращает (mask (N,) bool, "class" | "csf").
    """
    if source not in ("auto", "class", "csf"):
        raise ValueError(f"source должен быть 'auto', 'class' или 'csf', получено {source!r}")
    has_labels = classes is not None and np.any(np.asarray(classes) == ground_class)
    if source == "class" and not has_labels:
        raise ValueError(f"source='class', но точек класса {ground_class} нет")
    if source == "class" or (source == "auto" and has_labels):
        return np.asarray(classes) == ground_class, "class"
    return csf_ground(points, **csf_params), "csf"
