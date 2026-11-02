"""Reusable API for processing folders of XYZ point-cloud files."""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TypeVar

import numpy as np

from .data.xyz_export import export_points_to_npy
from .data.npy_loader import NPYLoader
from .data.xyz_loader import XYZLoader
from .data.xyz_saver import save_points_to_xyz
from .optimization.downsampling.voxel import downsample_point_cloud
from .optimization.features import compute_features
from .optimization.filtering import clean, density_scale

T = TypeVar("T")
CoordinateRange = dict[str, dict[str, float | None]]
FilterStatistics = dict[str, int | float]


@dataclass
class FileProcessingResult:
    """Outcome and measurements for one processed XYZ file."""

    input_path: Path
    success: bool
    input_points: int | None = None
    unique_classes: list[int] | None = None
    point_cloud_step_before_downsampling: float | None = None
    point_cloud_step_after_downsampling: float | None = None
    points_removed_by_filter: int | None = None
    filters_result: dict[str, FilterStatistics] = field(default_factory=dict)
    coordinate_range: CoordinateRange | None = None
    output_points: int | None = None
    output_path: Path | None = None
    ground_source: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this file's result."""
        return {
            "file_name": self.input_path.name,
            "success": self.success,
            "rows_loaded": self.input_points,
            "unique_classes": self.unique_classes,
            "point_cloud_step_before_downsampling": (
                self.point_cloud_step_before_downsampling
            ),
            "point_cloud_step_after_downsampling": (
                self.point_cloud_step_after_downsampling
            ),
            "points_removed_by_filter": self.points_removed_by_filter,
            "filters_result": self.filters_result,
            "coordinate_range": self.coordinate_range,
            "output_points": self.output_points,
            "error": self.error,
        }


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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the complete report."""
        return {
            "success": self.success,
            "input_dir": str(self.input_dir),
            "export_dir": str(self.export_dir),
            "elapsed_seconds": self.elapsed_seconds,
            "files": [result.to_dict() for result in self.files],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the complete report as standards-compliant JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )


def _coordinate_range(points: np.ndarray) -> CoordinateRange:
    ranges: CoordinateRange = {}
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        values = points[:, axis_index]
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            ranges[axis_name] = {
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
            }
        else:
            ranges[axis_name] = {"min": None, "max": None}
    return ranges


def _find_target(point_count: int) -> int:
    if point_count > 5_000_000:
        return 3_000_000
    if point_count > 3_000_000:
        return 1_000_000
    if point_count > 500_000:
        return 200_000
    if point_count > 100_000:
        return 100_000
    return point_count - 1


def _point_cloud_step(points: np.ndarray) -> float | None:
    if len(points) < 2:
        return None
    return density_scale(points, k=min(8, len(points) - 1))


def _clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _process_file(
    filepath: Path, export_dir: Path, verbose: bool, debug: bool
) -> FileProcessingResult:
    result = FileProcessingResult(input_path=filepath, success=False)

    def run_step(name: str, operation: Callable[[], T]) -> T:
        started_at = time.perf_counter()
        value = operation()
        result.timings[name] = time.perf_counter() - started_at
        return value

    try:
        loader = NPYLoader if filepath.suffix.lower() == ".npy" else XYZLoader
        classes, points = run_step(
            f"{loader.__name__}.load", lambda: loader(str(filepath)).load()
        )
        result.input_points = len(points)
        result.unique_classes = [int(value) for value in np.unique(classes)]
        result.coordinate_range = _coordinate_range(points)
        result.point_cloud_step_before_downsampling = run_step(
            "point_cloud_step_before_downsampling",
            lambda: _point_cloud_step(points),
        )

        result_points, result_classes = run_step(
            "downsample_point_cloud",
            lambda: downsample_point_cloud(
                points=points,
                target_points=_find_target(len(points)),
                classes=classes,
            ),
        )

        if debug:
            result.point_cloud_step_after_downsampling = run_step(
                "point_cloud_step_after_downsampling",
                lambda: _point_cloud_step(result_points),
            )
            result.output_points = len(result_points)
            output_path = export_dir / f"{filepath.stem}_downsampled.xyz"
            run_step(
                "export_downsampled_xyz",
                lambda: save_points_to_xyz(
                    result_points, output_path, classes=result_classes
                ),
            )
            result.output_path = output_path
            result.success = True
            return result

        result_clean = run_step(
            "clean", lambda: clean(result_points, result_classes, verbose=verbose)
        )
        result.point_cloud_step_after_downsampling = result_clean.cloud_scale
        if result.point_cloud_step_after_downsampling is None:
            result.point_cloud_step_after_downsampling = run_step(
                "point_cloud_step_after_downsampling",
                lambda: _point_cloud_step(result_points),
            )
        result.filters_result = result_clean.filters_result
        result_points = result_points[result_clean.keep]
        result_classes = result_classes[result_clean.keep]
        result.points_removed_by_filter = int(
            np.count_nonzero(~result_clean.keep)
        )
        result.output_points = len(result_points)

        point_features = run_step(
            "compute_features",
            lambda: compute_features(result_points, result_classes),
        )
        result.ground_source = point_features.ground_source

        output_path = export_dir / f"{filepath.stem}_cleaned.npy"
        run_step(
            "export_points_to_npy",
            lambda: export_points_to_npy(
                result_points,
                output_path,
                classes=result_classes,
                features=point_features.values,
            ),
        )
        result.output_path = output_path
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
    debug: bool = False,
) -> Iterator[FileProcessingResult]:
    """Yield one result per input file as it finishes processing.

    By default, clean the downsampled points and export them as NPY.
    With ``debug=True``, export the downsampled points as XYZ and skip cleanup.
    Feature calculation is included in NPY output; nearest-neighbor generation
    is currently disabled.

    Invalid directories and export-directory setup errors are raised to the
    caller. Errors processing an individual file are reported in its result.
    """
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Папка не найдена: {input_path}")

    files = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".xyz", ".npy"}
    )
    if not files:
        raise ValueError(f"В папке нет файлов .xyz или .npy: {input_path}")

    output_dir = Path(export_dir).resolve()
    if output_dir == input_path or output_dir in input_path.parents:
        raise ValueError("Папка экспорта не должна содержать входную папку.")
    if clear_export_dir:
        _clear_directory(output_dir)

    for filepath in files:
        yield _process_file(filepath, output_dir, verbose, debug)


def process_directory(
    input_dir: str | Path,
    export_dir: str | Path = "exports",
    *,
    clear_export_dir: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> ProcessingResult:
    """Process all point-cloud files and return a structured aggregate result."""
    started_at = time.perf_counter()
    input_path = Path(input_dir).resolve()
    output_dir = Path(export_dir).resolve()
    files = list(iter_process_directory(
        input_path,
        output_dir,
        clear_export_dir=clear_export_dir,
        verbose=verbose,
        debug=debug,
    ))
    return ProcessingResult(
        input_dir=input_path,
        export_dir=output_dir,
        files=files,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def process_directory_json(
    input_dir: str | Path,
    export_dir: str | Path = "exports",
    *,
    clear_export_dir: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Process a folder and return its complete report as a JSON string."""
    return process_directory(
        input_dir,
        export_dir,
        clear_export_dir=clear_export_dir,
        verbose=verbose,
        debug=debug,
    ).to_json()
