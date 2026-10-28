"""Reusable API for processing folders of XYZ point-cloud files."""
from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TypeVar

from .data.xyz_export import export_k_folks_features_to_npy
from .data.xyz_loader import XYZLoader
from .optimization.downsampling.voxel import downsample_point_cloud
from .optimization.features import compute_features
from .optimization.filtering import clean
from .optimization.k_folks import k_folks

T = TypeVar("T")


@dataclass
class FileProcessingResult:
    """Outcome and measurements for one processed XYZ file."""

    input_path: Path
    success: bool
    input_points: int | None = None
    output_points: int | None = None
    output_path: Path | None = None
    ground_source: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ProcessingResult:
    """Aggregate result returned by :func:`process_directory`."""

    input_dir: Path
    export_dir: Path
    files: list[FileProcessingResult]
    elapsed_seconds: float

    @property
    def success(self) -> bool:
        return all(result.success for result in self.files)

    @property
    def exit_code(self) -> int:
        return 0 if self.success else 1


def _find_target(point_count: int) -> int:
    if point_count > 5_000_000:
        return 2_000_000
    if point_count > 3_000_000:
        return 1_000_000
    if point_count > 500_000:
        return 200_000
    if point_count > 100_000:
        return point_count - 1
    return point_count


def _clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _process_file(filepath: Path, export_dir: Path, verbose: bool) -> FileProcessingResult:
    result = FileProcessingResult(input_path=filepath, success=False)

    def run_step(name: str, operation: Callable[[], T]) -> T:
        started_at = time.perf_counter()
        value = operation()
        result.timings[name] = time.perf_counter() - started_at
        return value

    try:
        classes, points = run_step(
            "XYZLoader.load", lambda: XYZLoader(str(filepath)).load()
        )
        result.input_points = len(points)

        result_points, result_classes = run_step(
            "downsample_point_cloud",
            lambda: downsample_point_cloud(
                points=points,
                target_points=_find_target(len(points)),
                classes=classes,
            ),
        )

        result_clean = run_step(
            "clean", lambda: clean(result_points, result_classes, verbose=verbose)
        )
        result_points = result_points[result_clean.keep]
        result_classes = result_classes[result_clean.keep]
        result.output_points = len(result_points)

        point_features = run_step(
            "compute_features",
            lambda: compute_features(result_points, result_classes),
        )
        result.ground_source = point_features.ground_source

        folks_result, folks_classes, folks_features = run_step(
            "k_folks",
            lambda: k_folks(
                points=result_points,
                k=400,
                classes=result_classes,
                features=point_features.values,
                show_progress=verbose,
            ),
        )

        requested_path = export_dir / f"{filepath.stem}_k_folks.csv"
        run_step(
            "export_k_folks_features_to_npy",
            lambda: export_k_folks_features_to_npy(
                neighbors=folks_result,
                features=folks_features,
                filepath=requested_path,
                classes=folks_classes,
            ),
        )
        result.output_path = requested_path.with_name(requested_path.name + ".npy")
        result.success = True
    except Exception as error:
        result.error = str(error)

    return result


def iter_process_directory(
    input_dir: str | Path,
    export_dir: str | Path = "exports",
    *,
    clear_export_dir: bool = False,
    verbose: bool = False,
) -> Iterator[FileProcessingResult]:
    """Yield one result per input file as it finishes processing.

    Invalid directories and export-directory setup errors are raised to the
    caller. Errors processing an individual file are reported in its result.
    """
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Папка не найдена: {input_path}")

    files = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".xyz"
    )
    if not files:
        raise ValueError(f"В папке нет файлов .xyz: {input_path}")

    output_dir = Path(export_dir).resolve()
    if output_dir == input_path or output_dir in input_path.parents:
        raise ValueError("Папка экспорта не должна содержать входную папку.")
    if clear_export_dir:
        _clear_directory(output_dir)

    for filepath in files:
        yield _process_file(filepath, output_dir, verbose)


def process_directory(
    input_dir: str | Path,
    export_dir: str | Path = "exports",
    *,
    clear_export_dir: bool = False,
    verbose: bool = False,
) -> ProcessingResult:
    """Process all XYZ files and return a structured aggregate result."""
    started_at = time.perf_counter()
    input_path = Path(input_dir).resolve()
    output_dir = Path(export_dir).resolve()
    files = list(iter_process_directory(
        input_path,
        output_dir,
        clear_export_dir=clear_export_dir,
        verbose=verbose,
    ))
    return ProcessingResult(
        input_dir=input_path,
        export_dir=output_dir,
        files=files,
        elapsed_seconds=time.perf_counter() - started_at,
    )
