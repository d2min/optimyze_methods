from pathlib import Path

from data.xyz_loader import XYZLoader
from data.xyz_saver import save_points_to_xyz
from optimization.downsampling.voxel import downsample_point_cloud


if __name__ == "__main__":
    filepath = "assets/dataset.xyz"
    
    try:
        xyzLoader = XYZLoader(filepath)
        classes, points = xyzLoader.load()

        voxelsPoints = downsample_point_cloud(points=points, target_points=len(points) / 10)

        print(f"Размер изначальный - {len(points)} точек")
        print(f"Размер вокселей - {len(voxelsPoints)} точек")

        save_points_to_xyz(points=voxelsPoints)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")