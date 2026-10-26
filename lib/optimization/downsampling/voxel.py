from typing import Optional, Tuple, Union

import numpy as np


def _compute_curvature_via_pca(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Вычисляет кривизну для каждой точки через PCA локального окружения.
    Использует воксельную сетку для группировки точек и вычисления ковариационных матриц.

    Математическая основа:
    - Для каждого вокселя строится ковариационная матрица 3x3
    - Вычисляются собственные значения λ1 ≤ λ2 ≤ λ3 (в возрастающем порядке)
    - Кривизна (Surface Variation) = λ1 / (λ1 + λ2 + λ3)

    :param points: Массив координат формы (N, 3).
    :param voxel_size: Размер вокселя для локальной группировки.
    :return: Массив кривизны формы (N,).
    """
    p_min = np.min(points, axis=0)
    pts_shifted = points - p_min

    # Квантование в мелкие воксели для оценки локальной структуры
    voxel_idx = np.floor(pts_shifted / voxel_size).astype(np.int32)

    # Преобразование в 1D ключи через np.void
    void_dt = np.dtype((np.void, voxel_idx.dtype.itemsize * 3))
    void_view = voxel_idx.view(void_dt).squeeze(axis=-1)

    _, inverse_indices = np.unique(void_view, return_inverse=True)
    num_voxels = inverse_indices.max() + 1

    # Подсчет точек в каждом вокселе
    counts = np.bincount(inverse_indices, minlength=num_voxels)
    valid_indices = np.flatnonzero(counts >= 3)
    curvature_voxel = np.zeros(num_voxels, dtype=np.float64)
    if not len(valid_indices):
        return curvature_voxel[inverse_indices]

    compact_indices = np.full(num_voxels, -1, dtype=np.intp)
    compact_indices[valid_indices] = np.arange(len(valid_indices))
    valid_points_mask = counts[inverse_indices] >= 3
    valid_inverse_indices = compact_indices[inverse_indices[valid_points_mask]]
    valid_points = points[valid_points_mask]
    valid_counts = counts[valid_indices]

    # Вычисляем суммы для ковариационной матрицы через np.bincount
    # C = E[XX^T] - E[X]E[X]^T
    sums = np.column_stack([
        np.bincount(
            valid_inverse_indices,
            weights=values,
            minlength=len(valid_indices),
        )
        for values in (
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
        )
    ])
    products = (
        valid_points[:, 0] ** 2,
        valid_points[:, 1] ** 2,
        valid_points[:, 2] ** 2,
        valid_points[:, 0] * valid_points[:, 1],
        valid_points[:, 0] * valid_points[:, 2],
        valid_points[:, 1] * valid_points[:, 2],
    )
    sum_xx, sum_yy, sum_zz, sum_xy, sum_xz, sum_yz = (
        np.bincount(
            valid_inverse_indices,
            weights=values,
            minlength=len(valid_indices),
        )
        for values in products
    )
    means = sums / valid_counts[:, None]

    cov = np.empty((len(valid_indices), 3, 3), dtype=np.float64)
    cov[:, 0, 0] = sum_xx / valid_counts - means[:, 0] ** 2
    cov[:, 1, 1] = sum_yy / valid_counts - means[:, 1] ** 2
    cov[:, 2, 2] = sum_zz / valid_counts - means[:, 2] ** 2
    cov[:, 0, 1] = cov[:, 1, 0] = (
        sum_xy / valid_counts - means[:, 0] * means[:, 1]
    )
    cov[:, 0, 2] = cov[:, 2, 0] = (
        sum_xz / valid_counts - means[:, 0] * means[:, 2]
    )
    cov[:, 1, 2] = cov[:, 2, 1] = (
        sum_yz / valid_counts - means[:, 1] * means[:, 2]
    )

    eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0)
    lambda_sum = np.sum(eigenvalues, axis=1)
    valid_lambda = lambda_sum > 1e-10
    curvature_voxel[valid_indices[valid_lambda]] = (
        eigenvalues[valid_lambda, 0] / lambda_sum[valid_lambda]
    )

    # Присваиваем кривизну вокселя всем точкам внутри него
    curvature = curvature_voxel[inverse_indices]

    return curvature


def _process_voxel_grid(points: np.ndarray, voxel_size: float, p_min: np.ndarray,
                        classes: Optional[np.ndarray] = None):
    """
    Внутренняя векторизованная функция для обработки воксельной сетки.
    Выполняет квантование, хеширование и вычисление центроидов.

    :param points: Массив координат формы (N, 3).
    :param voxel_size: Размер вокселя.
    :param p_min: Минимальные координаты для сдвига.
    :return: Массив центроидов формы (M, 3), а при переданных классах —
             также наиболее частый класс каждой группы.
    """
    if points.shape[0] == 0:
        centroids = np.empty((0, 3), dtype=points.dtype)
        if classes is None:
            return centroids
        return centroids, np.empty(0, dtype=classes.dtype)

    pts_shifted = points - p_min
    voxel_idx = np.floor(pts_shifted / voxel_size).astype(np.int32)

    void_dt = np.dtype((np.void, voxel_idx.dtype.itemsize * 3))
    void_view = voxel_idx.view(void_dt).squeeze(axis=-1)

    _, inverse_indices = np.unique(void_view, return_inverse=True)
    num_voxels = inverse_indices.max() + 1

    counts = np.bincount(inverse_indices, minlength=num_voxels)

    centroids = np.empty((num_voxels, 3), dtype=points.dtype)
    for i in range(3):
        centroids[:, i] = np.bincount(
            inverse_indices, weights=points[:, i], minlength=num_voxels
        ) / counts

    if classes is not None:
        unique_classes, class_indices = np.unique(classes, return_inverse=True)
        output_classes = np.empty(num_voxels, dtype=classes.dtype)
        if len(unique_classes) <= 16:
            class_counts = np.zeros(num_voxels, dtype=np.int64)
            for class_index, point_class in enumerate(unique_classes):
                current_counts = np.bincount(
                    inverse_indices[class_indices == class_index],
                    minlength=num_voxels,
                )
                replace = current_counts > class_counts
                output_classes[replace] = point_class
                class_counts[replace] = current_counts[replace]
        else:
            voxel_class_pairs = np.column_stack((inverse_indices, class_indices))
            unique_pairs, pair_counts = np.unique(
                voxel_class_pairs, axis=0, return_counts=True
            )
            max_counts = np.zeros(num_voxels, dtype=np.int64)
            np.maximum.at(max_counts, unique_pairs[:, 0], pair_counts)

            winners = np.flatnonzero(
                pair_counts == max_counts[unique_pairs[:, 0]]
            )
            first_winner = winners[
                np.r_[
                    True,
                    unique_pairs[winners[1:], 0] != unique_pairs[winners[:-1], 0],
                ]
            ]
            output_classes[unique_pairs[first_winner, 0]] = unique_classes[
                unique_pairs[first_winner, 1]
            ]
        return centroids, output_classes

    return centroids


def downsample_point_cloud(
    points: np.ndarray,
    target_points: int,
    classes: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Основной конвейер прореживания облака точек с адаптивной кривизной.

    Оценивает кривизну через PCA и применяет адаптивные размеры вокселей.
    После первоначальной оценки размер вокселя подстраивается по фактическому
    числу занятых вокселей, чтобы разреженные и вытянутые облака не
    прореживались значительно сильнее target_points. При превышении целевого
    количества результат ограничивается target_points.

    :param points: Массив координат формы (N, 3) типа float32 или float64.
    :param target_points: Максимальное количество точек после прореживания.
    :param classes: Необязательная метка каждой исходной точки формы (N,).
    :return: При отсутствии classes — массив точек (M, 3). При переданных
             classes — кортеж (точки, метки), где метка центроида является
             наиболее частой меткой исходных точек в его вокселе.
    """
    N = points.shape[0]
    if classes is not None:
        classes = np.asarray(classes)
        if classes.ndim != 1 or len(classes) != N:
            raise ValueError(
                f"classes должен иметь форму ({N},), получена форма {classes.shape}"
            )
    if target_points < 0:
        raise ValueError("target_points не может быть отрицательным")
    if target_points == 0:
        if classes is None:
            return points[:0]
        return points[:0], classes[:0]
    if N <= target_points:
        if classes is not None:
            return points, classes
        return points

    # ШАГ 1: Инициализация и оценка базового размера вокселя (V_base)
    p_min = np.min(points, axis=0)
    p_max = np.max(points, axis=0)
    dims = p_max - p_min
    dims[dims == 0] = 1e-6  # Защита от вырожденных случаев

    volume = np.prod(dims)
    V_base = np.cbrt(volume / target_points)

    # ШАГ 2: Грубое прореживание для экономии памяти на экстремально больших данных
    if N > 10_000_000:
        indices = np.random.choice(N, 10_000_000, replace=False)
        points = points[indices]
        if classes is not None:
            classes = classes[indices]
        N = 10_000_000

    # ШАГ 3: Вычисление кривизны через PCA на воксельной сетке
    # Используем уменьшенный воксель для захвата локальной структуры
    curvature_voxel_size = V_base / 4.0
    curvature = _compute_curvature_via_pca(points, curvature_voxel_size)

    # Адаптивное разделение по порогу кривизны
    # 80-й процентиль выделяет реальные ~20% самых "признаковых" точек
    # (границы объектов, углы, перегибы), а не шум на плоских поверхностях
    curvature_threshold = np.percentile(curvature, 80)

    high_mask = curvature > curvature_threshold
    low_mask = ~high_mask

    pts_high, pts_low = points[high_mask], points[low_mask]
    classes_high = None if classes is None else classes[high_mask]
    classes_low = None if classes is None else classes[low_mask]

    # Обработка зон с высокой детализацией (уменьшенный воксель для сохранения деталей)
    high_result = _process_voxel_grid(pts_high, V_base * 0.5, p_min, classes_high)
    # Обработка зон с низкой детализацией (базовый воксель)
    low_result = _process_voxel_grid(pts_low, V_base, p_min, classes_low)
    if classes is None:
        voxels_high, voxels_low = high_result, low_result
    else:
        voxels_high, labels_high = high_result
        voxels_low, labels_low = low_result

    # Подстраиваем размер вокселя по фактическому числу занятых ячеек.
    # Оценка через объем bounding box может сильно промахиваться для облаков
    # с выбросами, плоской/вытянутой формой или неравномерной плотностью.
    for _ in range(12):
        n_voxels = voxels_high.shape[0] + voxels_low.shape[0]
        if n_voxels >= target_points:
            break

        scale = max(0.1, min(0.8, 0.8 * np.cbrt(n_voxels / target_points)))
        V_base *= scale
        high_result = _process_voxel_grid(
            pts_high, V_base * 0.5, p_min, classes_high
        )
        low_result = _process_voxel_grid(pts_low, V_base, p_min, classes_low)
        if classes is None:
            voxels_high, voxels_low = high_result, low_result
        else:
            voxels_high, labels_high = high_result
            voxels_low, labels_low = low_result

    # ШАГ 4: Умное отсечение с приоритетом зон высокой кривизны
    # Гарантируем, что геометрические признаки (границы, углы) сохранятся в первую очередь
    n_high = voxels_high.shape[0]
    n_low = voxels_low.shape[0]

    if n_high >= target_points:
        # Случай A: Зона высокой кривизны сама по себе превышает лимит.
        # Все точки здесь ценные, поэтому выбираем случайно из них
        # (дальнейшее уточнение потребовало бы повторного PCA, что дорого).
        indices = np.random.choice(n_high, target_points, replace=False)
        downsampled_points = voxels_high[indices]
        if classes is not None:
            downsampled_classes = labels_high[indices]

    elif n_high + n_low <= target_points:
        # Случай B: Даже объединение обеих зон не превышает лимит.
        # Берём всё — дополнительное прореживание не нужно.
        downsampled_points = np.vstack((voxels_high, voxels_low))
        if classes is not None:
            downsampled_classes = np.concatenate((labels_high, labels_low))

    else:
        # Случай C (основной): High-зона помещается, добираем остаток из Low-зоны.
        # Все признаки сохранены, "плоские" точки прореживаются равномерно.
        remaining = target_points - n_high
        indices = np.random.choice(n_low, remaining, replace=False)
        downsampled_points = np.vstack((voxels_high, voxels_low[indices]))
        if classes is not None:
            downsampled_classes = np.concatenate((labels_high, labels_low[indices]))

    if classes is not None:
        return downsampled_points, downsampled_classes
    return downsampled_points
