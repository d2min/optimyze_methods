import shutil
import sys
from pathlib import Path

from data.xyz_loader import XYZLoader
from data.xyz_export import export_k_folks_to_csv, export_points_to_csv, export_k_folks_to_npy, export_points_to_npy
from optimization.k_folks import k_folks
from optimization.normalization.zscore import ZScoreScaler
from optimization.downsampling.voxel import downsample_point_cloud
from optimization.filtering import clean


def clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def findTarget(len: int) -> int:
    if len > 10_000_000:
        return len // 10
    
    if len > 5_000_000:
        return len // 5

    if len > 3_000_000:
        return len // 3

    if len > 500_000:
        return len // 2

    if len > 100_000:
        return len - 1
    
    return len

def main() -> int:
    if len(sys.argv) < 2:
        print("Ошибка: укажите путь к папке с файлами .xyz!")
        return 1

    input_dir = Path(sys.argv[1]).resolve()
    if not input_dir.is_dir():
        print(f"Ошибка: папка не найдена: {input_dir}")
        return 1

    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".xyz"
    )
    if not files:
        print(f"В папке нет файлов .xyz: {input_dir}")
        return 1

    export_dir = Path("exports").resolve()
    if export_dir == input_dir or export_dir in input_dir.parents:
        print("Ошибка: папка экспорта не должна содержать входную папку.")
        return 1

    try:
        clear_directory(export_dir)
    except OSError as error:
        print(f"Ошибка при очистке папки экспорта: {error}")
        return 1

    failures = 0
    for filepath in files:
        try:
            print(f"\n------------------- Обработка {filepath.name} ---------------------")

            zscore = ZScoreScaler()
            classes, points = XYZLoader(str(filepath)).load()
            
            # Voxel downsampling
            result_points, result_classes = downsample_point_cloud(
                points=points,
                target_points= findTarget(len=len(points)),
                classes=classes,
            )

            # Filtration
            result_clean = clean(result_points, result_classes)
            result_points = result_points[result_clean.keep]
            result_classes = result_classes[result_clean.keep]

            # Получение ближайших точек
            folks_result, folks_classes = k_folks(
                points=result_points, k=400, classes=result_classes
            )

            # Z-score
            folks_result = zscore.fit_transform_k_folks(folks_result)
            print(f"Z-score для folks точек прошла")

            result_points = zscore.fit_transform(result_points)
            print(f"Z-score для облока точек прошла")

            output_path = export_dir / f"{filepath.stem}.csv"
            export_points_to_npy(
                points=result_points,
                filepath=output_path,
                classes=result_classes,
            )
            print(f"Успешный импорт облока")

            output_path = export_dir / f"{filepath.stem}_k_folks.csv"
            export_k_folks_to_npy(
                neighbors=folks_result,
                filepath=output_path,
                classes=folks_classes,
            )
            print(f"Успешный импорт folks точек")

            print(f"--------------------------------------------------------------")

        except Exception as error:
            failures += 1
            print(f"❌ Ошибка при обработке {filepath.name}: {error}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
