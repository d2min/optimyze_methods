"""
Загрузчик файлов формата .xyz (класс x y z)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Загрузчик файлов формата .xyz
    
# Ожидаемый формат каждой строки:
# class x y z
    
# Пример:
# 2 15.430000 20.110000 5.000000
# 6 16.120000 21.050000 8.500000
class XYZLoader:
    # Инициализация загрузки
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        if not self.filepath.suffix.lower() == '.xyz':
            logger.warning(f"У файла неверное расширение, нужно .xyz: {filepath}")
    
    # Выгрузка данных из файла
    # Можно задать ограничение точек для тестирования
    def load(self, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Загрузка файла: {self.filepath}")
        
        try:
            # Читаем файл через NumPy
            data = np.loadtxt(
                self.filepath,
                dtype=np.float32,
                max_rows=max_points
            )
            
            logger.info(f"Загружено строк: {data.shape[0]}")
            
            # Валидация структуры данных
            if data.ndim != 2 or data.shape[1] != 4:
                raise ValueError(
                    f"Неверный формат данных. Ожидалось 4 колонки (class, x, y, z), "
                    f"получено {data.shape[1] if data.ndim == 2 else 'не 2D массив'}"
                )
            
            # Разделяем на классы и координаты
            classes = data[:, 0].astype(np.int32)
            points = data[:, 1:4]
            
            logger.info(f"Уникальные классы: {np.unique(classes)}")
            logger.info(f"Диапазон координат: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                       f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                       f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
            
            return classes, points
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            raise
    
    # Загрузка данных сразу в формат Open3D PointCloud
    def load_with_open3d(self, max_points: Optional[int] = None):
        import open3d as o3d
        
        try:
            classes, points = self.load(max_points)
            
            # Создаем облако точек
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Добавляем классы как пользовательский атрибут
            # Open3D не поддерживает целочисленные атрибуты напрямую,
            # поэтому сохраняем как float
            pcd.point_normals = o3d.utility.Vector3dVector(
                np.column_stack([classes, np.zeros_like(classes), np.zeros_like(classes)])
            )
            
            logger.info(f"Создано Open3D PointCloud с {len(pcd.points)} точками")
            
            return pcd, classes
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            raise

# Пример использования
if __name__ == "__main__":
    # Тестирование загрузчика
    filepath = "examples/datasets/terrain_points.xyz"
    
    try:
        xyz_loader = XYZLoader(filepath)
        classes, points = xyz_loader.load(max_points=100000)
        
        print(f"   Первые 5 точек:")
        for i in range(min(5, len(points))):
            print(f"   Класс {classes[i]}: ({points[i, 0]:.3f}, {points[i, 1]:.3f}, {points[i, 2]:.3f})")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")