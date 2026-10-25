# Признаки точек: HAG, нормаль, земля

```python
from optimization.features import compute_features

feats = compute_features(points, classes)   # points (N, 3), classes (N,) или None
feats.values         # (N, 5) float32: hag, nx, ny, nz, ground
feats.ground_source  # "class" — земля из разметки, "csf" — найдена CSF
```

| Признак | Что это |
|---|---|
| `hag` | высота над землёй, м; бывает < 0 (точка ниже найденной земли) |
| `nx ny nz` | единичная нормаль к поверхности по 20 ближайшим соседям (PCA), `nz ≥ 0` |
| `ground` | 1 — точка земли, 0 — нет |

## Откуда земля

- Если в `classes` есть класс 2 — земля берётся из разметки (`ground_source="auto"`, по умолчанию).
- Иначе — **CSF** (Cloth Simulation Filter, Zhang et al., 2016, *Remote Sensing* 8(6):501):
  облако переворачивается, сверху падает виртуальная ткань, точки рядом с тканью — земля.
  Используется реализация авторов на C++: пакет `cloth-simulation-filter`.
- `ground_source="csf"` — всегда CSF, разметка не используется.

HAG: по точкам земли строится рельеф (медиана высот в ячейках 2 × 2 м, между ячейками —
интерполяция по 4 ближайшим), из высоты точки вычитается высота рельефа под ней.

## Параметры CSF

По умолчанию: жёсткость ткани 1, сглаживание склонов включено, шаг ткани — два средних
расстояния между точками, от 0.5 до 1.5 м. Подобрано на Univer2019, Niagara, Ogorod,
terrain_points и Test_02 по совпадению HAG с HAG по разметке:

| Облако | Медиана расхождения HAG | 95% точек в пределах |
|---|---:|---:|
| Univer2019 | 5 см | 0.23 м |
| Ogorod | 5 см | 0.38 м |
| terrain_points | 1 см | 0.38 м |
| Test_02 | 14 см | 0.38 м |
| Niagara | 12 см | 8 м |

На Niagara хуже из-за заросших обрывов ущелья: под кронами на крутом склоне ткань
не находит землю. Это известное ограничение CSF.

## В пайплайне (`lib/__main__.py`)

прореживание → очистка → `compute_features` → `k_folks(..., features=...)` →
`export_k_folks_features_to_npy`. На каждую точку и каждого соседа сохраняется группа
`x y z hag nx ny nz ground class` (9 чисел, float32). Названия столбцов —
`k_folks_feature_columns(k, FEATURE_COLUMNS)` из `data/xyz_export.py`.

Без `features` функции `k_folks` и `export_k_folks_to_npy/csv` работают как раньше.
