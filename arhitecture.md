red_team/
├── data/
│   ├── __init__.py
│   ├── xyz_loader.py          # Загрузчик .xyz файлов
│   └── xyz_saver.py           # Сохранение результатов
├── optimization/
│   ├── __init__.py
│   ├── voxelization/
│   │   ├── __init__.py
│   │   └── voxel_grid.py      # Вокселизация
│   ├── filtering/
│   │   ├── __init__.py
│   │   ├── statistical_outlier.py  # SOR фильтрация
│   │   └── radius_outlier.py       # Радиусная фильтрация
│   ├── normalization/
│   │   ├── __init__.py
│   │   └── normalize.py       # Нормализация координат
│   └── downsampling/
│       ├── __init__.py
│       └── random_sampling.py # Случайное прореживание
├── tests/
│   ├── __init__.py
│   └── test_loader.py         # Тесты загрузчика
├── examples/
│   └── terrain_points.xyz     # Тестовый файл
── README.md                  # Документация