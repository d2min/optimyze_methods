"""Tests for the public folder-processing API."""
import json
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import iter_process_directory, process_directory  # noqa: E402
from lib import process_directory_json  # noqa: E402
from lib import pipeline  # noqa: E402
from lib import __main__ as cli  # noqa: E402
from lib.optimization.filtering import clean  # noqa: E402
from lib.optimization.filtering import pipeline as filtering_pipeline  # noqa: E402


def _mock_pipeline():
    points = np.zeros((3, 3), dtype=np.float32)
    classes = np.zeros(3, dtype=np.int32)

    class Loader:
        def __init__(self, filepath):
            self.filepath = filepath

        def load(self):
            if Path(self.filepath).stem == "bad":
                raise ValueError("invalid input")
            return classes, points

    class NPYLoader(Loader):
        pass

    return (
        patch.object(pipeline, "XYZLoader", Loader),
        patch.object(pipeline, "NPYLoader", NPYLoader),
        patch.object(
            pipeline, "downsample_point_cloud",
            lambda points, **_kwargs: (points, classes),
        ),
        patch.object(
            pipeline, "clean",
            lambda points, _classes=None, **_kwargs: SimpleNamespace(
                keep=np.ones(len(points), dtype=bool),
                filters_result={},
                cloud_scale=1.0,
            ),
        ),
        patch.object(
            pipeline, "compute_features",
            lambda *_args, **_kwargs: SimpleNamespace(
                values=np.tile([10, 0, 0, 1, 1], (3, 1)).astype(np.float32),
                ground_source="class",
            ),
        ),
    )


def test_directory_api_returns_results_and_yields_per_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"
        export_dir = root / "output"
        input_dir.mkdir()
        (input_dir / "good.xyz").touch()
        (input_dir / "points.npy").touch()
        (input_dir / "bad.xyz").touch()
        (input_dir / "ignored.txt").touch()

        patches = _mock_pipeline()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            report = process_directory(input_dir, export_dir)
            assert report.success is False
            assert report.exit_code == 1
            assert len(report.files) == 3
            good = next(
                (result for result in report.files if result.success),
                None,
            )
            assert good is not None, report.files
            assert good.output_points == 3
            assert good.input_points == 3
            assert good.unique_classes == [0]
            assert good.point_cloud_step_before_downsampling is not None
            assert good.point_cloud_step_after_downsampling == 1.0
            assert good.points_removed_by_filter == 0
            assert good.coordinate_range == {
                "x": {"min": 0.0, "max": 0.0},
                "y": {"min": 0.0, "max": 0.0},
                "z": {"min": 0.0, "max": 0.0},
            }
            assert good.ground_source == "class"
            assert good.output_path.is_file()
            npy = next(result for result in report.files if result.input_path.suffix == ".npy")
            assert npy.success
            assert "NPYLoader.load" in npy.timings
            assert "compute_features" in npy.timings
            assert npy.output_path.name == "points_cleaned.npy"
            expected = np.tile([0, 0, 0, 0, 10, 0, 0, 1, 1], (3, 1))
            np.testing.assert_array_equal(np.load(npy.output_path), expected)
            bad = next(result for result in report.files if not result.success)
            assert bad.error == "invalid input"

        (input_dir / "bad.xyz").unlink()
        (input_dir / "points.npy").unlink()
        patches = _mock_pipeline()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            yielded = list(iter_process_directory(input_dir, export_dir))
        assert len(yielded) == 1
        assert yielded[0].success


def test_process_directory_json_reports_file_statistics():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"
        input_dir.mkdir()
        (input_dir / "cloud.xyz").touch()
        classes = np.array([2, 6, 6], dtype=np.int32)
        points = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32
        )

        class Loader:
            def __init__(self, _filepath):
                pass

            def load(self):
                return classes, points

        with (
            patch.object(pipeline, "XYZLoader", Loader),
            patch.object(pipeline, "NPYLoader", Loader),
            patch.object(
                pipeline, "downsample_point_cloud",
                lambda points, classes, **_kwargs: (points, classes),
            ),
            patch.object(
                pipeline,
                "clean",
                lambda points, _classes=None, **_kwargs: SimpleNamespace(
                    keep=np.array([True, False, True]),
                    cloud_scale=0.5,
                    filters_result={
                        "hag": {
                            "deleted_points_count": 1,
                            "percent_deleted_points": 50.0,
                            "number_of_remaining": 1,
                            "filter_worktime": 0.25,
                        }
                    },
                ),
            ),
            patch.object(
                pipeline,
                "compute_features",
                lambda points, *_args, **_kwargs: SimpleNamespace(
                    values=np.zeros((len(points), 5), dtype=np.float32),
                    ground_source="class",
                ),
            ),
        ):
            report = json.loads(
                process_directory_json(input_dir, root / "output")
            )

    assert report["success"] is True
    file_result = report["files"][0]
    assert file_result["file_name"] == "cloud.xyz"
    assert "input_path" not in file_result
    assert "timings" not in file_result
    assert "ground_source" not in file_result
    assert "output_path" not in file_result
    assert file_result["rows_loaded"] == 3
    assert file_result["unique_classes"] == [2, 6]
    assert file_result["point_cloud_step_before_downsampling"] is not None
    assert file_result["point_cloud_step_after_downsampling"] == 0.5
    assert file_result["points_removed_by_filter"] == 1
    assert file_result["filters_result"] == {
        "hag": {
            "deleted_points_count": 1,
            "percent_deleted_points": 50.0,
            "number_of_remaining": 1,
            "filter_worktime": 0.25,
        }
    }
    assert file_result["coordinate_range"] == {
        "x": {"min": 1.0, "max": 7.0},
        "y": {"min": 2.0, "max": 8.0},
        "z": {"min": 3.0, "max": 9.0},
    }


def test_directory_iterator_processes_each_file_lazily():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"
        input_dir.mkdir()
        (input_dir / "first.xyz").touch()
        (input_dir / "second.xyz").touch()
        loaded_files = []

        patches = list(_mock_pipeline())
        loader_patch = patches[0]
        original_loader = loader_patch.new

        class TrackingLoader(original_loader):
            def load(self):
                loaded_files.append(Path(self.filepath).name)
                return super().load()

        loader_patch = patch.object(pipeline, "XYZLoader", TrackingLoader)
        with loader_patch, patches[1], patches[2], patches[3], patches[4]:
            results = iter_process_directory(input_dir, root / "output")
            assert loaded_files == []
            first = next(results)
            assert first.input_path.name == "first.xyz"
            assert loaded_files == ["first.xyz"]
            second = next(results)
            assert second.input_path.name == "second.xyz"
            assert loaded_files == ["first.xyz", "second.xyz"]


def test_debug_exports_downsampled_xyz_and_skips_remaining_steps():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"
        input_dir.mkdir()
        (input_dir / "cloud.xyz").touch()
        export_dir = root / "output"
        patches = _mock_pipeline()
        with (
            patches[0],
            patches[2],
            patch.object(pipeline, "clean", side_effect=AssertionError("clean called")),
            patch.object(
                pipeline, "compute_features",
                side_effect=AssertionError("features called"),
            ),
        ):
            result = next(iter_process_directory(input_dir, export_dir, debug=True))

        assert result.success, result.error
        assert result.output_points == 3
        assert result.points_removed_by_filter is None
        assert result.output_path == export_dir / "cloud_downsampled.xyz"
        exported = np.loadtxt(result.output_path)
        assert exported.shape == (3, 4)
        np.testing.assert_array_equal(exported[:, 0], np.zeros(3))
        assert "export_downsampled_xyz" in result.timings


def test_cli_debug_flag_enables_debug_pipeline():
    received = {}

    def fake_iter_process_directory(input_dir, **kwargs):
        received["input_dir"] = input_dir
        received.update(kwargs)
        return iter(())

    with (
        patch.object(cli, "iter_process_directory", fake_iter_process_directory),
        redirect_stdout(StringIO()),
    ):
        assert cli.main([".\\assets\\", "--debug"]) == 0

    assert received["debug"] is True
    assert received["clear_export_dir"] is True


def test_silent_clean_does_not_print_when_hag_step_is_skipped():
    output = StringIO()
    with redirect_stdout(output):
        clean(
            np.zeros((2, 3)),
            classes=np.zeros(2, dtype=np.int32),
            steps=[{"name": "hag"}],
            verbose=False,
        )
    assert output.getvalue() == ""


def test_clean_returns_per_filter_statistics():
    with patch.object(
        filtering_pipeline,
        "STEPS",
        {"test_filter": lambda points, _classes, **_kwargs: np.array([0, 7, 0])},
    ):
        result = clean(
            np.zeros((3, 3)),
            steps=[{"name": "test_filter"}],
            verbose=False,
        )

    assert result.filters_result["test_filter"]["deleted_points_count"] == 1
    assert result.filters_result["test_filter"]["percent_deleted_points"] == 100 / 3
    assert result.filters_result["test_filter"]["number_of_remaining"] == 2
    assert isinstance(result.filters_result["test_filter"]["filter_worktime"], float)


if __name__ == "__main__":
    test_directory_api_returns_results_and_yields_per_file()
    print("ok  test_directory_api_returns_results_and_yields_per_file")
    test_process_directory_json_reports_file_statistics()
    print("ok  test_process_directory_json_reports_file_statistics")
    test_directory_iterator_processes_each_file_lazily()
    print("ok  test_directory_iterator_processes_each_file_lazily")
    test_debug_exports_downsampled_xyz_and_skips_remaining_steps()
    print("ok  test_debug_exports_downsampled_xyz_and_skips_remaining_steps")
    test_cli_debug_flag_enables_debug_pipeline()
    print("ok  test_cli_debug_flag_enables_debug_pipeline")
    test_silent_clean_does_not_print_when_hag_step_is_skipped()
    print("ok  test_silent_clean_does_not_print_when_hag_step_is_skipped")
    test_clean_returns_per_filter_statistics()
    print("ok  test_clean_returns_per_filter_statistics")
