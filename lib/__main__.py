"""Command-line entry point for folder processing."""
import logging
import sys
import time
from pathlib import Path

if __package__:
    from .pipeline import iter_process_directory
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib.pipeline import iter_process_directory


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("Ошибка: укажите путь к папке с файлами .xyz или .npy!")
        return 1

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    started_at = time.perf_counter()
    failures = 0
    try:
        results = iter_process_directory(
            args[0], clear_export_dir=True, verbose=True
        )
        for result in results:
            print(
                f"\n------------------- Обработка "
                f"{result.input_path.name} ---------------------"
            )
            for name, elapsed in result.timings.items():
                print(f"Время выполнения {name}: {elapsed:.3f} сек.")
            if not result.success:
                failures += 1
                print(
                    f"❌ Ошибка при обработке {result.input_path.name}: "
                    f"{result.error}"
                )
                continue

            print(f"Общее число точек после обработки {result.output_points}")
            print(f"Земля для HAG взята из: {result.ground_source}")
            print("Успешный импорт folks точек")
            print("--------------------------------------------------------------")
    except (NotADirectoryError, ValueError, OSError) as error:
        print(f"Ошибка: {error}")
        return 1

    elapsed_seconds = int(time.perf_counter() - started_at)
    print(
        f"Общее время обработки: "
        f"{elapsed_seconds // 60}:{elapsed_seconds % 60:02d}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
