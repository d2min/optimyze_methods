import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from data.xyz_loader import XYZLoader
from data.xyz_export import export_k_folks_to_csv, export_points_to_csv, export_k_folks_to_npy, export_points_to_npy
from data.xyz_export import export_k_folks_features_to_npy
from optimization.k_folks import k_folks
from optimization.normalization.zscore import ZScoreScaler
from optimization.downsampling.voxel import downsample_point_cloud
from optimization.filtering import clean
from optimization.features import compute_features


@contextmanager
def measure_time(label: str) -> Iterator[None]:
    started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        print(f"Время выполнения {label}: {elapsed_seconds:.3f} сек.")


def clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def findTarget(len: int) -> int:
    if len > 5_000_000:
        return 2_000_000

    if len > 3_000_000:
        return 1_000_000

    if len > 500_000:
        return 200_000

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
    loop_started_at = time.perf_counter()
    for filepath in files:
        try:
            print(f"\n------------------- Обработка {filepath.name} ---------------------")

            zscore = ZScoreScaler()
            with measure_time("XYZLoader.load"):
                classes, points = XYZLoader(str(filepath)).load()
            
            # Voxel downsampling
            with measure_time("downsample_point_cloud"):
                result_points, result_classes = downsample_point_cloud(
                    points=points,
                    target_points=findTarget(len=len(points)),
                    classes=classes,
                )

            # Filtration
            with measure_time("clean"):
                result_clean = clean(result_points, result_classes)
            result_points = result_points[result_clean.keep]
            result_classes = result_classes[result_clean.keep]

            print(f'Общее число точек после обработки {len(result_points)}')

            # Признаки: hag, nx, ny, nz, ground (земля — класс 2, если есть, иначе CSF)
            with measure_time("compute_features"):
                point_features = compute_features(result_points, result_classes)
            print(f"Земля для HAG взята из: {point_features.ground_source}")

            # Получение ближайших точек
            with measure_time("k_folks"):
                folks_result, folks_classes, folks_features = k_folks(
                    points=result_points, k=400, classes=result_classes,
                    features=point_features.values,
                )

            # Z-score
            # ! Пока отключили но не удаляем
            # folks_result = zscore.fit_transform_k_folks(folks_result)
            # HAG стандартизуется отдельно, нормали и ground — нет:
            # hag = folks_features[..., 0]
            # folks_features[..., 0] = ZScoreScaler(n_columns=1).fit_transform(
            #     hag.reshape(-1, 1)).reshape(hag.shape)
            # print(f"Z-score для folks точек прошла")

            # result_points = zscore.fit_transform(result_points)
            # print(f"Z-score для облока точек прошла")
            # !

            # ! Отключил но не удалять
            # output_path = export_dir / f"{filepath.stem}.csv"
            # export_points_to_npy(
            #     points=result_points,
            #     filepath=output_path,
            #     classes=result_classes,
            # )
            # print(f"Успешный импорт облока")
            # !

            output_path = export_dir / f"{filepath.stem}_k_folks.csv"
            # Группа на каждую точку: x y z hag nx ny nz ground class
            with measure_time("export_k_folks_features_to_npy"):
                export_k_folks_features_to_npy(
                    neighbors=folks_result,
                    features=folks_features,
                    filepath=output_path,
                    classes=folks_classes,
                )

            print(f"Успешный импорт folks точек")

            print(f"--------------------------------------------------------------")

        except Exception as error:
            failures += 1
            print(f"❌ Ошибка при обработке {filepath.name}: {error}")

    elapsed_seconds = int(time.perf_counter() - loop_started_at)
    print(
        f"Общее время обработки: "
        f"{elapsed_seconds // 60}:{elapsed_seconds % 60:02d}"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
