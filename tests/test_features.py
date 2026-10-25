"""Тесты признаков: земля (класс 2 / CSF), HAG, нормали, k_folks и экспорт с признаками.

    python tests/test_features.py      # без pytest
    python -m pytest tests/            # если pytest установлен
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.optimization.features import (  # noqa: E402
    FEATURE_COLUMNS, compute_features, estimate_normals, ground_mask,
)
from lib.optimization.k_folks.k_folks import k_folks  # noqa: E402
from lib.data.xyz_export import (  # noqa: E402
    export_k_folks_features_to_csv, export_k_folks_features_to_npy,
    k_folks_feature_columns,
)
from lib.optimization.normalization import ZScoreScaler  # noqa: E402

rng = np.random.default_rng(0)


def _scene(n_ground=20000, seed=0):
    """Холмистый рельеф + «дом» 6 м и «дерево» 10 м. Классы: 2 земля, 6 дом, 5 дерево."""
    r = np.random.default_rng(seed)
    xy = r.uniform(0, 60, (n_ground, 2))
    ground_z = lambda x, y: 2 * np.sin(x / 15) + 0.05 * y
    g = np.column_stack([xy, ground_z(xy[:, 0], xy[:, 1])])
    hx = r.uniform(20, 30, (3000, 2))
    house = np.column_stack([hx, ground_z(hx[:, 0], hx[:, 1]) + 6.0])
    t = r.normal(0, 1.2, (2000, 3)) + [45, 15, 0]
    tree = np.column_stack([t[:, :2], ground_z(t[:, 0], t[:, 1]) + 10 + t[:, 2]])
    pts = np.vstack([g, house, tree])
    cls = np.concatenate([np.full(len(g), 2), np.full(len(house), 6), np.full(len(tree), 5)])
    true_hag = pts[:, 2] - ground_z(pts[:, 0], pts[:, 1])
    return pts, cls, true_hag


def test_normals_on_tilted_plane_point_up():
    xy = rng.uniform(0, 20, (5000, 2))
    plane = np.column_stack([xy, 0.5 * xy[:, 0]])
    n = estimate_normals(plane)
    expected = np.array([-0.5, 0.0, 1.0]) / np.sqrt(1.25)
    np.testing.assert_allclose(np.median(n, axis=0), expected, atol=1e-3)
    assert (n[:, 2] >= 0).all()
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-5)
    assert n.dtype == np.float32


def test_normals_follow_dome_radius():
    v = rng.normal(size=(20000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] > 0.2] * 5
    n = estimate_normals(v)
    cos = np.abs((n * v / 5).sum(1))
    assert np.median(cos) > 0.999


def test_normals_tiny_clouds():
    assert estimate_normals(np.zeros((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(estimate_normals(np.zeros((2, 3))), [[0, 0, 1], [0, 0, 1]])


def test_ground_source_auto_class_csf():
    pts, cls, _ = _scene()
    m, src = ground_mask(pts, cls)
    assert src == "class" and np.array_equal(m, cls == 2)
    m, src = ground_mask(pts, None)
    assert src == "csf"
    m2, src2 = ground_mask(pts, cls, source="csf")
    assert src2 == "csf"
    # CSF без разметки находит землю и не берёт крышу и крону
    assert m[cls == 2].mean() > 0.9
    assert m[cls == 6].mean() < 0.05 and m[cls == 5].mean() < 0.05
    try:
        ground_mask(pts, np.full(len(pts), 6), source="class")
    except ValueError:
        pass
    else:
        raise AssertionError("source='class' без земли должен падать")


def test_hag_matches_true_height_with_class_and_csf():
    pts, cls, true_hag = _scene()
    for src in ("auto", "csf"):
        f = compute_features(pts, cls, ground_source=src)
        hag = f.values[:, 0]
        assert np.median(np.abs(hag - true_hag)[cls == 6]) < 0.3, src
        assert np.median(np.abs(hag - true_hag)[cls == 2]) < 0.2, src


def test_compute_features_shape_and_columns():
    pts, cls, _ = _scene(n_ground=3000)
    f = compute_features(pts, cls)
    assert f.values.shape == (len(pts), 5) and f.values.dtype == np.float32
    assert f.columns == FEATURE_COLUMNS == ("hag", "nx", "ny", "nz", "ground")
    assert set(np.unique(f.values[:, 4])) <= {0.0, 1.0}
    assert (f.values[:, 3] >= 0).all()
    empty = compute_features(np.zeros((0, 3)))
    assert empty.values.shape == (0, 5)


def test_k_folks_without_features_is_unchanged():
    pts = rng.normal(size=(50, 3))
    cls = rng.integers(0, 5, 50)
    a = k_folks(pts, k=4, show_progress=False)
    b, bc = k_folks(pts, k=4, show_progress=False, classes=cls)
    assert isinstance(a, np.ndarray) and a.shape == (50, 15)
    np.testing.assert_array_equal(a, b)
    assert bc.shape == (50, 5)


def test_k_folks_features_follow_neighbors():
    pts = rng.normal(size=(60, 3))
    cls = rng.integers(0, 5, 60)
    feats = np.column_stack([np.arange(60), pts]).astype(np.float32)   # id точки + её xyz
    res, nc, nf = k_folks(pts, k=5, show_progress=False, classes=cls, features=feats)
    groups = res.reshape(60, 6, 3)
    np.testing.assert_allclose(nf[..., 1:], groups, atol=1e-6)        # признаки той же точки
    np.testing.assert_array_equal(cls[nf[..., 0].astype(int)], nc)
    res2, nf2 = k_folks(pts, k=5, show_progress=False, features=feats)
    np.testing.assert_array_equal(res, res2)
    np.testing.assert_array_equal(nf, nf2)
    r0, f0 = k_folks(pts, k=0, show_progress=False, features=feats)
    assert f0.shape == (60, 1, 4)


def test_export_layout_x_y_z_features_class():
    pts = rng.normal(size=(20, 3)).astype(np.float32)
    cls = rng.integers(0, 7, 20)
    feats = rng.normal(size=(20, 5)).astype(np.float32)
    res, nc, nf = k_folks(pts, k=3, show_progress=False, classes=cls, features=feats)
    with tempfile.TemporaryDirectory() as d:
        export_k_folks_features_to_npy(res, nf, Path(d) / "a.npy", classes=nc)
        data = np.load(Path(d) / "a.npy")
        assert data.shape == (20, 4 * 9) and data.dtype == np.float32
        g = data.reshape(20, 4, 9)
        np.testing.assert_array_equal(g[..., :3], res.reshape(20, 4, 3))
        np.testing.assert_array_equal(g[..., 3:8], nf)
        np.testing.assert_array_equal(g[..., 8], nc)
        export_k_folks_features_to_csv(res, nf, Path(d) / "a.csv", FEATURE_COLUMNS, classes=nc)
        header = (Path(d) / "a.csv").read_text().splitlines()[0].split(",")
        assert header == k_folks_feature_columns(3, FEATURE_COLUMNS)
        assert header[:9] == ["target_x", "target_y", "target_z", "target_hag", "target_nx",
                              "target_ny", "target_nz", "target_ground", "target_class"]
        export_k_folks_features_to_npy(res, nf, Path(d) / "b.npy")
        assert np.load(Path(d) / "b.npy").shape == (20, 4 * 8)


def test_zscore_single_column_for_hag():
    hag = rng.normal(5, 3, size=(1000, 1))
    out = ZScoreScaler(n_columns=1).fit_transform(hag)
    assert out.shape == (1000, 1)
    np.testing.assert_allclose(out.mean(), 0, atol=1e-5)
    try:
        ZScoreScaler().fit(hag)      # по умолчанию по-прежнему 3 столбца
    except ValueError:
        pass
    else:
        raise AssertionError("ZScoreScaler() должен требовать 3 столбца")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"Все тесты пройдены: {len(tests)}")
