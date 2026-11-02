"""Print the JSON report for a folder of point-cloud files."""
import argparse
import tempfile
from pathlib import Path

from lib import process_directory_json


def main() -> None:
    default_input_dir = Path(__file__).resolve().parent / "assets"
    parser = argparse.ArgumentParser(
        description="Process point clouds and print the JSON report."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=default_input_dir,
        help="folder with .xyz or .npy files (default: assets)",
    )
    parser.add_argument(
        "--export-dir",
        help="output folder (default: a temporary folder)",
    )
    args = parser.parse_args()

    if args.export_dir:
        report_json = process_directory_json(args.input_dir, args.export_dir)
        print(report_json)
        return

    with tempfile.TemporaryDirectory(prefix="optimyze-json-example-") as export_dir:
        report_json = process_directory_json(args.input_dir, export_dir)
        print(report_json)


if __name__ == "__main__":
    main()
