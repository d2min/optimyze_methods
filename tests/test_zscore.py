"""Тесты ZScoreScaler.

    python tests/test_zscore.py      # без pytest
    python -m pytest tests/          # если pytest установлен
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.optimization.normalization import ZScoreScaler  # noqa: E402

rng = np.random.default_rng(0)


def test_params_come_from_fit_matrix_only():
    ref = rng.normal([10, 20, 30], [1, 2, 3], size=(5000, 3))
    other = rng.normal([500, -7, 0], [9, 9, 9], size=(123, 3))
    s = ZScoreScaler().fit(ref)
    np.testing.assert_allclose(s.mean_, ref.mean(axis=0))
    np.testing.assert_allclose(s.std_, ref.std(axis=0))
    out = s.transform(other)
    # другая матрица стандартизуется по параметрам ref, а не по своим
    np.testing.assert_allclose(out, (other - ref.mean(0)) / ref.std(0), rtol=1e-6, atol=1e-5)
    assert out.shape == other.shape
    assert s.n_samples_ == 5000


def test_fit_transform_gives_zero_mean_unit_std_per_axis():
    p = rng.normal([3, -4, 100], [5, 0.1, 20], size=(10000, 3))
    out = ZScoreScaler().fit_transform(p).astype(np.float64)
    np.testing.assert_allclose(out.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(out.std(0), 1, atol=1e-5)


def test_output_is_float32_and_input_not_modified():
    for dt in (np.float64, np.float32, np.int64):
        p = (rng.normal(size=(100, 3)) * 10).astype(dt)
        before = p.copy()
        out = ZScoreScaler().fit(p).transform(p)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(p, before)


def test_inverse_restores_points():
    ref = rng.normal(size=(1000, 3)) * 50
    other = rng.normal(size=(77, 3)) * 50 + 3
    s = ZScoreScaler().fit(ref)
    back = s.inverse_transform(s.transform(other))
    assert back.dtype == np.float64
    assert np.abs(back - other).max() < 1e-4


def test_utm_coordinates_float32_output_is_precise_enough():
    base = np.array([502964.94, 4985000.22, 50.85])
    ref = base + rng.uniform(-500, 500, size=(100000, 3)) * [1, 1, 0.1]
    s = ZScoreScaler().fit(ref)
    back = s.inverse_transform(s.transform(ref))
    assert np.abs(back - ref).max() < 1e-3   # меньше миллиметра после float32


def test_constant_axis_does_not_divide_by_zero():
    p = np.column_stack([rng.normal(size=100), rng.normal(size=100), np.full(100, 7.0)])
    s = ZScoreScaler().fit(p)
    assert s.std_[2] == 1.0
    np.testing.assert_allclose(s.transform(p)[:, 2], 0)


def test_errors():
    s = ZScoreScaler()
    cases = [(lambda: s.transform(np.zeros((3, 3))), "не обучен"),
             (lambda: s.fit(np.zeros(3)), "1D"),
             (lambda: s.fit(np.zeros((0, 3))), "пусто"),
             (lambda: s.fit(np.array([[1.0, np.nan, 2.0]])), "NaN")]
    for bad, name in cases:
        try:
            bad()
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"нет ошибки: {name}")
    s.fit(rng.normal(size=(10, 3)))
    try:
        s.transform(np.zeros((5, 4)))   # передали столбец класса
    except ValueError as e:
        assert "класса" in str(e)
    else:
        raise AssertionError("нет ошибки при 4 столбцах")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"Все тесты пройдены: {len(tests)}")
