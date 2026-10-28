"""Потоковый пример: python lib/exemple.py [--aggregate]."""
import argparse
import sys
from pathlib import Path

if __package__:
    from . import iter_process_directory, process_directory
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import iter_process_directory, process_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="обработать все файлы и вернуть общий отчет",
    )
    args = parser.parse_args()

    input_dir = Path(__file__).resolve().parent.parent / "assets"
    if not args.aggregate:
        for result in iter_process_directory(input_dir):
            print(
                f"Готово: {result.input_path.name}; "
                f"успешно: {result.success}; "
                f"точек: {result.output_points}; "
                f"ошибка: {result.error}"
            )
        return

    report = process_directory(input_dir)
    print(f"Успешно: {report.success}; всего файлов: {len(report.files)}")
    for result in report.files:
        print(result.input_path, result.success, result.error)


if __name__ == "__main__":
    main()