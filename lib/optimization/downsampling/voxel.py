import numpy as np


def _compute_curvature_via_pca(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Вычисляет кривизну для каждой точки через PCA локального окружения.
    Использует воксельную сетку для группировки точек и вычисления ковариационных матриц.
    
    Математическая основа:
    - Для каждого вокселя строится ковариационная матрица 3x3
    - Вычисляются собственные значения λ1 ≥ λ2 ≥ λ3
    - Кривизна = λ_min / (λ1 + λ2 + λ3)
    
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
    
    # Вычисляем суммы для ковариационной матрицы через np.bincount
    # C = E[XX^T] - E[X]E[X]^T
    sum_x = np.bincount(inverse_indices, weights=points[:, 0], minlength=num_voxels)
    sum_y = np.bincount(inverse_indices, weights=points[:, 1], minlength=num_voxels)
    sum_z = np.bincount(inverse_indices, weights=points[:, 2], minlength=num_voxels)
    
    sum_xx = np.bincount(inverse_indices, weights=points[:, 0] ** 2, minlength=num_voxels)
    sum_yy = np.bincount(inverse_indices, weights=points[:, 1] ** 2, minlength=num_voxels)
    sum_zz = np.bincount(inverse_indices, weights=points[:, 2] ** 2, minlength=num_voxels)
    
    sum_xy = np.bincount(inverse_indices, weights=points[:, 0] * points[:, 1], minlength=num_voxels)
    sum_xz = np.bincount(inverse_indices, weights=points[:, 0] * points[:, 2], minlength=num_voxels)
    sum_yz = np.bincount(inverse_indices, weights=points[:, 1] * points[:, 2], minlength=num_voxels)
    
    # Защита от деления на ноль и вырожденных случаев
    valid_mask = counts >= 3  # Минимум 3 точки для PCA
    
    # Инициализируем ковариационные матрицы нулями
    cov = np.zeros((num_voxels, 3, 3), dtype=np.float64)
    
    # Вычисляем ковариацию только для валидных вокселей
    if np.any(valid_mask):
        mean_x = np.zeros(num_voxels, dtype=np.float64)
        mean_y = np.zeros(num_voxels, dtype=np.float64)
        mean_z = np.zeros(num_voxels, dtype=np.float64)
        
        mean_x[valid_mask] = sum_x[valid_mask] / counts[valid_mask]
        mean_y[valid_mask] = sum_y[valid_mask] / counts[valid_mask]
        mean_z[valid_mask] = sum_z[valid_mask] / counts[valid_mask]
        
        cov[valid_mask, 0, 0] = (sum_xx[valid_mask] / counts[valid_mask]) - mean_x[valid_mask] ** 2
        cov[valid_mask, 1, 1] = (sum_yy[valid_mask] / counts[valid_mask]) - mean_y[valid_mask] ** 2
        cov[valid_mask, 2, 2] = (sum_zz[valid_mask] / counts[valid_mask]) - mean_z[valid_mask] ** 2
        cov[valid_mask, 0, 1] = (sum_xy[valid_mask] / counts[valid_mask]) - mean_x[valid_mask] * mean_y[valid_mask]
        cov[valid_mask, 1, 0] = cov[valid_mask, 0, 1]
        cov[valid_mask, 0, 2] = (sum_xz[valid_mask] / counts[valid_mask]) - mean_x[valid_mask] * mean_z[valid_mask]
        cov[valid_mask, 2, 0] = cov[valid_mask, 0, 2]
        cov[valid_mask, 1, 2] = (sum_yz[valid_mask] / counts[valid_mask]) - mean_y[valid_mask] * mean_z[valid_mask]
        cov[valid_mask, 2, 1] = cov[valid_mask, 1, 2]
    
    # Вычисляем собственные значения (batch-операция)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)  # Обнуляем отрицательные значения (численные ошибки)
    
    # Кривизна = λ_min / (λ1 + λ2 + λ3)
    lambda_sum = np.sum(eigenvalues, axis=1)
    curvature_voxel = np.zeros(num_voxels, dtype=np.float64)
    valid_lambda = lambda_sum > 1e-10
    curvature_voxel[valid_lambda] = eigenvalues[valid_lambda, 0] / lambda_sum[valid_lambda]
    
    # Присваиваем кривизну вокселя всем точкам внутри него
    curvature = curvature_voxel[inverse_indices]
    
    return curvature


def _process_voxel_grid(points: np.ndarray, voxel_size: float, p_min: np.ndarray) -> np.ndarray:
    """
    Внутренняя векторизованная функция для обработки воксельной сетки.
    Выполняет квантование, хеширование и вычисление центроидов.
    
    :param points: Массив координат формы (N, 3).
    :param voxel_size: Размер вокселя.
    :param p_min: Минимальные координаты для сдвига.
    :return: Массив центроидов формы (M, 3).
    """
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
        
    return centroids


def downsample_point_cloud(points: np.ndarray, target_points: int) -> np.ndarray:
    """
    Основной конвейер прореживания облака точек с адаптивной кривизной.
    
    Алгоритм:
    1. Оценивает базовый размер вокселя на основе bounding box и целевого количества точек
    2. Вычисляет кривизну через PCA локального окружения (воксельная сетка)
    3. Разделяет точки на зоны высокой и низкой детализации
    4. Применяет адаптивное прореживание с разными размерами вокселей
    5. Выполняет финальную корректировку для точного попадания в целевое количество
    
    :param points: Массив координат формы (N, 3) типа float32 или float64.
    :param target_points: Целевое количество точек после прореживания.
    :return: Массив прореженных точек формы (M, 3), где M <= target_points.
    """
    N = points.shape[0]
    if N == 0 or target_points <= 0:
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
        N = 10_000_000

    # ШАГ 3: Вычисление кривизны через PCA на воксельной сетке
    # Используем уменьшенный воксель для захвата локальной структуры
    curvature_voxel_size = V_base / 4.0
    curvature = _compute_curvature_via_pca(points, curvature_voxel_size)
    
    # Адаптивное разделение по порогу кривизны (медиана как динамический порог)
    curvature_threshold = np.median(curvature)
    
    high_mask = curvature > curvature_threshold
    low_mask = ~high_mask
    
    pts_high, pts_low = points[high_mask], points[low_mask]
    
    # Обработка зон с высокой детализацией (уменьшенный воксель для сохранения деталей)
    voxels_high = _process_voxel_grid(pts_high, V_base * 0.5, p_min)
    # Обработка зон с низкой детализацией (базовый воксель)
    voxels_low = _process_voxel_grid(pts_low, V_base, p_min)
    
    # ШАГ 4: Объединение результатов
    downsampled_points = np.vstack((voxels_high, voxels_low))

    # ШАГ 5: Финальная корректировка для точного попадания в целевое количество
    current_N = downsampled_points.shape[0]
    if current_N > target_points * 1.1:
        final_indices = np.random.choice(current_N, target_points, replace=False)
        downsampled_points = downsampled_points[final_indices]

    return downsampled_points