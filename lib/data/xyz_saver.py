import numpy as np
import os

def save_points_to_xyz(points: np.ndarray, filepath: str = "./exports/points.xyz",
                       classes: np.ndarray = None) -> None:
    """
    Сохраняет массив точек в файл .xyz.
    Перезаписывает файл при каждом вызове.
    Автоматически исправляет ситуацию, если вместо папки там файл/симлинк.
    """
    abs_path = os.path.abspath(filepath)
    dir_path = os.path.dirname(abs_path)
    
    # Пытаемся создать директорию
    try:
        os.makedirs(dir_path, exist_ok=True)
    except FileExistsError:
        # exist_ok=True не сработал — значит там НЕ директория
        if os.path.isfile(dir_path) or os.path.islink(dir_path):
            print(f"⚠️ По пути '{dir_path}' находится файл или симлинк. Удаляем и создаем настоящую папку...")
            os.remove(dir_path)
            os.makedirs(dir_path)
        else:
            raise OSError(f"Невозможно создать директорию '{dir_path}'. Проверьте права доступа.")
    
    # Сохраняем файл (np.savetxt автоматически перезаписывает существующий файл)
    # classes — необязательно; без них, как и раньше, пишется класс 0
    cls = np.zeros(points.shape[0], dtype=int) if classes is None else np.asarray(classes)
    output_data = np.column_stack((cls, points))
    np.savetxt(filepath, output_data, fmt='%d %.6f %.6f %.6f')
    print(f"✅ Файл сохранен: {abs_path}")