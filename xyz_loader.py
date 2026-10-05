"""
Загрузчик файлов формата .xyz (класс x y z)
Оптимизирован для работы с большими облаками точек (миллионы точек)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class XYZLoader:
    """
    Загрузчик файлов формата .xyz
    
    Ожидаемый формат каждой строки:
    class x y z
    
    Пример:
    2 15.430000 20.110000 5.000000
    6 16.120000 21.050000 8.500000
    """
    
    def __init__(self, filepath: str):
        """
        Инициализация загрузчика
        
        Args:
            filepath: Путь к .xyz файлу
        """
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        if not self.filepath.suffix.lower() == '.xyz':
            logger.warning(f"Файл не имеет расширения .xyz: {filepath}")
    
    def load(self, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загрузка данных из .xyz файла
        
        Args:
            max_points: Максимальное количество точек для загрузки (для тестирования)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - classes: массив классов形状 (N,)
                - points: массив координат形状 (N, 3)
        """
        logger.info(f"Загрузка файла: {self.filepath}")
        
        try:
            # Читаем файл через NumPy (быстро и эффективно)
            # dtype=np.float32 экономит память
            data = np.loadtxt(
                self.filepath,
                dtype=np.float32,
                max_rows=max_points  # Ограничение для тестирования
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
            
            logger.info(f"Классы: {classes.shape}, Координаты: {points.shape}")
            logger.info(f"Уникальные классы: {np.unique(classes)}")
            logger.info(f"Диапазон координат: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                       f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                       f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
            
            return classes, points
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            raise
    
    def load_with_open3d(self, max_points: Optional[int] = None):
        """
        Загрузка данных сразу в формат Open3D PointCloud
        
        Args:
            max_points: Максимальное количество точек для загрузки
            
        Returns:
            o3d.geometry.PointCloud: Облако точек Open3D
        """
        import open3d as o3d
        
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


def load_xyz(filepath: str, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Удобная функция для быстрой загрузки .xyz файла
    
    Args:
        filepath: Путь к файлу
        max_points: Максимальное количество точек
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: classes, points
    """
    loader = XYZLoader(filepath)
    return loader.load(max_points)


# Пример использования
if __name__ == "__main__":
    # Тестирование загрузчика
    filepath = "examples/terrain_points.xyz"
    
    try:
        classes, points = load_xyz(filepath, max_points=100000)  # Для теста ограничим
        
        print(f"\n✅ Успешно загружено:")
        print(f"   Точек: {len(points):,}")
        print(f"   Классов: {len(np.unique(classes))}")
        print(f"   Первые 5 точек:")
        for i in range(min(5, len(points))):
            print(f"   Класс {classes[i]}: ({points[i, 0]:.3f}, {points[i, 1]:.3f}, {points[i, 2]:.3f})")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")