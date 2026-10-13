"""
Удаление мелких изолированных групп точек ("острова" в воздухе):
стаи птиц, сгустки атмосферного шума, случайно попавшие объекты.

Ловит то, что пропускают SOR/ROR: внутри маленького сгустка у точек есть соседи,
поэтому поточечные фильтры считают их нормальными.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ._common import as_local


def voxel_clusters(points: np.ndarray, voxel: float, min_cluster_size: int = 50,
                   return_labels: bool = False):
    """Связные компоненты на воксельной сетке (26-связность).

    Масштабируется на десятки миллионов точек: память O(число занятых вокселей),
    без перебора пар точек. Точки связаны, если их воксели соседние
    (расстояние между точками до ~2·sqrt(3)·voxel).

    voxel — должен быть больше шага облака в разреженных местах (иначе земля
    рассыпается на куски) и меньше расстояния от мусора до объектов.
    """
    p = as_local(points)
    ijk = np.floor((p - p.min(axis=0)) / voxel).astype(np.int64) + 1
    dims = ijk.max(axis=0) + 2
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    ukey, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    m = len(ukey)
    rows, cols = [], []
    # 13 «половинных» соседей: граф неориентированный, вторую половину не нужно
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
            if (a, b, c) > (0, 0, 0)]
    for a, b, c in offs:
        nk = ukey + (a * dims[1] + b) * dims[2] + c
        pos = np.minimum(np.searchsorted(ukey, nk), m - 1)
        hit = ukey[pos] == nk
        rows.append(np.flatnonzero(hit))
        cols.append(pos[hit])
    r = np.concatenate(rows)
    cl = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.int8), (r, cl)), shape=(m, m))
    _, lab = connected_components(g, directed=False)
    comp_size = np.bincount(lab, weights=cnt)  # размер компоненты в точках
    keep = comp_size[lab[inv]] >= min_cluster_size
    if return_labels:
        return keep, lab[inv]
    return keep
