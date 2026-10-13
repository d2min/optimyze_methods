"""
Вертикальные выбросы в аэролазерных (ALS) облаках.

Типичный мусор ALS:
  * low noise  — точки под землёй (многолучевое отражение, ошибки дальномера);
  * high noise — точки высоко в воздухе (птицы, облака, дымка, атмосферные отражения).
В стандарте ASPRS LAS для них есть классы 7 (Low Point / noise) и 18 (High Noise).

column_gap_noise не использует классы и модель рельефа: в каждой колонке (ячейка XY)
точки сортируются по Z и режутся на сегменты по вертикальным разрывам > gap.
Сегмент из < min_segment точек, если в колонке есть «основная масса», — шум.
Это обобщение PDAL filters.elm (Extended Local Minimum), которое ловит не только
одиночный нижний выброс, но и группы выбросов снизу и сверху.
"""
import numpy as np

from ._common import as_local, build_tree

LOW_NOISE = 7     # ASPRS: Low Point (noise)
HIGH_NOISE = 18   # ASPRS: High Noise


def _column_pass(p: np.ndarray, cell: float, gap: float, min_segment: int,
                 shift: float) -> np.ndarray:
    """Один проход по сетке со сдвигом shift. Возвращает коды 0 / LOW_NOISE / HIGH_NOISE."""
    ij = np.floor((p[:, :2] - p[:, :2].min(axis=0) + shift) / cell).astype(np.int64)
    ckey = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    order = np.lexsort((p[:, 2], ckey))            # по ячейке, внутри — по z
    ck = ckey[order]
    z = p[order, 2]
    n = len(z)

    new_cell = np.ones(n, bool)
    new_cell[1:] = ck[1:] != ck[:-1]
    new_seg = new_cell.copy()
    new_seg[1:] |= (z[1:] - z[:-1]) > gap
    seg = np.cumsum(new_seg) - 1                   # id сегмента
    cell_id = np.cumsum(new_cell) - 1              # id колонки
    seg_size = np.bincount(seg)
    seg_cell = cell_id[new_seg]                    # колонка каждого сегмента
    seg_zmin = z[new_seg]

    # «основная масса» колонки — самый большой сегмент
    n_cells = cell_id[-1] + 1
    best = np.zeros(n_cells, np.int64)
    best_size = np.zeros(n_cells, np.int64)
    order_s = np.lexsort((seg_size, seg_cell))     # последний в колонке — самый большой
    last = np.ones(len(order_s), bool)
    last[:-1] = seg_cell[order_s][1:] != seg_cell[order_s][:-1]
    best[seg_cell[order_s][last]] = order_s[last]
    best_size[seg_cell[order_s][last]] = seg_size[order_s][last]

    small = (seg_size < min_segment) & (best_size[seg_cell] >= min_segment)
    below = seg_zmin < seg_zmin[best[seg_cell]]
    code_seg = np.where(small, np.where(below, LOW_NOISE, HIGH_NOISE), 0)

    codes = np.empty(n, np.int16)
    codes[order] = code_seg[seg]
    return codes


def column_gap_noise(points: np.ndarray, cell: float = 5.0, gap: float = 3.0,
                     min_segment: int = 5, isolation: float = None,
                     isolation_k: int = 3, confirm_radius: float = None,
                     confirm_max: int = 3) -> np.ndarray:
    """Коды шума для каждой точки: 0 — норма, 7 — low noise, 18 — high noise.

    cell        — размер колонки в XY, м. Больше — устойчивее «основная масса»,
                  но стены/обрывы внутри ячейки дают больше сегментов.
    gap         — вертикальный разрыв, м, который отделяет сегмент.
    min_segment — сегменты меньше этого размера считаются шумом.
    isolation   — доп. проверка, м: кандидат подтверждается, только если средняя
                  дистанция до isolation_k соседей в 3D больше isolation.
                  Защищает провода ЛЭП и тонкие конструкции (у них есть соседи вдоль линии).
    confirm_radius, confirm_max — вторая проверка: кандидат остаётся шумом, только если
                  в радиусе confirm_radius (м) у него не больше confirm_max соседей.
                  Отличает «точку в пустом небе» от редкой точки на фасаде высотки:
                  у стены в радиусе 10 м всегда есть другие точки стены.

    Два прохода по сеткам со сдвигом на пол-ячейки: точка у границы ячейки
    иначе может оказаться одна в «пустой» колонке и не быть проверенной.
    """
    p = as_local(points)
    a = _column_pass(p, cell, gap, min_segment, shift=0.0)
    b = _column_pass(p, cell, gap, min_segment, shift=cell / 2)
    codes = np.where(a > 0, a, b)
    if isolation is None and confirm_radius is None:
        return codes
    cand = np.flatnonzero(codes > 0)
    if cand.size:
        tree = build_tree(p)
        ok = np.ones(cand.size, bool)
        if isolation is not None:
            d, _ = tree.query(p[cand], k=isolation_k + 1, workers=-1)
            ok &= d[:, 1:].mean(axis=1) > isolation
        if confirm_radius is not None:
            cnt = np.asarray(tree.query_ball_point(p[cand], r=confirm_radius,
                                                   return_length=True, workers=-1)) - 1
            ok &= cnt <= confirm_max
        codes[cand[~ok]] = 0
    return codes


def column_gap(points: np.ndarray, **kw) -> np.ndarray:
    """Маска keep для column_gap_noise."""
    return column_gap_noise(points, **kw) == 0


def height_above_ground(points: np.ndarray, ground_mask: np.ndarray,
                        cell: float = 2.0, k: int = 4, method: str = "median") -> np.ndarray:
    """Высота над землёй (HAG) по точкам земли (например, classes == 2).

    Модель рельефа: медиана Z земли в ячейках cell×cell (медиана устойчива
    к попавшему в класс земли шуму), затем IDW-интерполяция по k ближайшим ячейкам.
    """
    from scipy.spatial import cKDTree
    p = as_local(points)
    g = p[ground_mask]
    if len(g) == 0:
        raise ValueError("Нет точек земли: проверьте ground_mask")
    o = p[:, :2].min(axis=0)
    ij = np.floor((g[:, :2] - o) / cell).astype(np.int64)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    srt = np.lexsort((g[:, 2], key))
    ks, zs = key[srt], g[srt, 2]
    uk, start, cnt = np.unique(ks, return_index=True, return_counts=True)
    ny = ij[:, 1].max() + 1
    cen = np.column_stack([uk // ny, uk % ny]) * cell + o + cell / 2
    d, idx = cKDTree(cen).query(p[:, :2], k=min(k, len(cen)), workers=-1)
    if d.ndim == 1:
        d, idx = d[:, None], idx[:, None]
    if method == "min":
        # нижняя огибающая: самая низкая точка земли в k ближайших ячейках
        return p[:, 2] - zs[start][idx].min(axis=1)
    # медиана отсортированного по z куска = элемент(ы) посередине
    med = 0.5 * (zs[start + (cnt - 1) // 2] + zs[start + cnt // 2])
    w = 1.0 / np.maximum(d, cell / 4)
    dtm = (med[idx] * w).sum(axis=1) / w.sum(axis=1)
    return p[:, 2] - dtm


def hag_noise(points: np.ndarray, ground_mask: np.ndarray, below: float = -1.0,
              above: float = None, cell: float = 2.0, isolation: float = None,
              isolation_k: int = 3) -> np.ndarray:
    """Коды шума по высоте над землёй: 7 — ниже below (м), 18 — выше above (м).

    below=-1.0: всё, что больше чем на метр под рельефом, физически невозможно
    для съёмки сверху (кроме ям/подвалов — их видно по соседям-земле).
    above: задайте по самому высокому объекту сцены с запасом (например, 1.2 × высота
    самого высокого здания); None — верх не проверять.
    isolation: подтверждать кандидата, только если средняя дистанция до isolation_k
    соседей > isolation (м). Защищает от ошибок модели рельефа на обрывах, в ущельях,
    у подпорных стен: там «подземные» точки на самом деле плотная нормальная поверхность.

    Рельеф здесь — нижняя огибающая (method="min"), а не медиана: на обрыве медиана
    ячейки выше её нижних точек, и настоящая земля у подножия выглядела бы «подземной»
    (на Niagara так терялось 97 точек земли). Шумом считается только то, что ниже
    самой низкой точки земли во всей окрестности.
    """
    h = height_above_ground(points, ground_mask, cell=cell, method="min")
    codes = np.zeros(len(h), np.int16)
    codes[h < below] = LOW_NOISE
    if above is not None:
        codes[h > above] = HIGH_NOISE
    if isolation is not None:
        cand = np.flatnonzero(codes > 0)
        if cand.size:
            p = as_local(points)
            d, _ = build_tree(p).query(p[cand], k=isolation_k + 1, workers=-1)
            codes[cand[d[:, 1:].mean(axis=1) <= isolation]] = 0
    return codes
