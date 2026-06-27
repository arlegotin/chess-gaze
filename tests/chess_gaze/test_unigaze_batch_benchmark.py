from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pytest import CaptureFixture, MonkeyPatch

import chess_gaze.unigaze_batch_benchmark as benchmark
from chess_gaze.run_equivalence import EquivalenceReport, EquivalenceTolerances
from chess_gaze.unigaze_batch_benchmark import (
    BenchmarkCandidateResult,
    UniGazeBatchBenchmarkReport,
    selected_mps_batch_size,
)

BenchmarkDevice = Literal["cpu", "mps"]
BenchmarkStatus = Literal[
    "passed",
    "preflight_failed",
    "analyze_failed",
    "equivalence_failed",
    "oom",
    "unsupported_op",
]
ForwardStatus = Literal["not_run", "passed", "failed"]


def test_benchmark_report_selects_fastest_passing_mps_batch_size() -> None:
    report = _report(
        [
            _candidate(
                device="cpu",
                batch_size=1,
                analysis_wall_seconds=100.0,
                full_run_dir="cpu1",
            ),
            _candidate(
                device="mps",
                batch_size=8,
                analysis_wall_seconds=80.0,
                full_run_dir="mps8",
                peak_mps_memory_bytes=123,
            ),
            _candidate(
                device="mps",
                batch_size=16,
                analysis_wall_seconds=70.0,
                full_run_dir="mps16",
                peak_mps_memory_bytes=456,
            ),
        ]
    )

    assert selected_mps_batch_size(report) == 16


def test_benchmark_report_ignores_failed_mps_candidates() -> None:
    report = _report(
        [
            _candidate(
                device="mps",
                batch_size=32,
                status="oom",
                analysis_wall_seconds=None,
                frames_per_second=None,
                full_run_dir=None,
                qa_final_status=None,
                qa_decoded_frames=None,
                qa_counts_match=None,
                qa_schema_validation_passed=None,
                equivalence_report_path=None,
                max_appearance_pitch_yaw_delta_radians=None,
                max_scene_ray_component_delta=None,
                max_monitor_uv_delta_m=None,
                error_code="OOM",
                error_message="simulated",
            )
        ]
    )

    assert selected_mps_batch_size(report) is None


def test_benchmark_report_round_trips_through_json(tmp_path: Path) -> None:
    report = _report(
        [
            _candidate(
                device="mps",
                batch_size=4,
                analysis_wall_seconds=12.5,
                full_run_dir="mps4",
            )
        ],
        selected_device="mps",
        selected_batch_size=4,
        selected_reason="fastest passing MPS batch size above 1",
    )
    path = tmp_path / "benchmark.json"

    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    loaded = UniGazeBatchBenchmarkReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )

    assert loaded == report


def test_print_selected_batch_size_cli(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    report = _report(
        [
            _candidate(
                device="cpu",
                batch_size=1,
                analysis_wall_seconds=12.0,
                full_run_dir="cpu1",
            ),
            _candidate(
                device="mps",
                batch_size=2,
                analysis_wall_seconds=11.0,
                full_run_dir="mps2",
            ),
            _candidate(
                device="mps",
                batch_size=4,
                analysis_wall_seconds=9.0,
                full_run_dir="mps4",
            ),
        ]
    )
    report_path = tmp_path / "benchmark.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")

    exit_code = benchmark.main(
        ["--report", str(report_path), "--print-selected-batch-size"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "4\n"
    assert captured.err == ""


def test_benchmark_cli_writes_candidate_rows_and_removes_mps_env(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"not a real video")
    models_root = tmp_path / "models"
    unigaze_model = models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    unigaze_model.parent.mkdir(parents=True)
    unigaze_model.write_bytes(b"model")
    baseline_json = tmp_path / "baseline.json"
    baseline_run_dir = tmp_path / "baseline-run"
    baseline_json.write_text(
        json.dumps(
            {
                "run_dir": str(baseline_run_dir),
                "qa_decoded_frames": 20,
                "source_video_sha256": "e" * 64,
                "unigaze_sha256": "a" * 64,
                "torch_version": "2.12.1",
                "unigaze_version": "0.1.3",
                "mps_available": True,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "benchmark.json"
    commands: list[list[str]] = []
    subprocess_envs: list[dict[str, str]] = []
    compared_runs: list[tuple[str, str, tuple[float, float, float] | None]] = []
    forward_calls: list[tuple[str, int, object]] = []

    monkeypatch.setattr(benchmark, "CANDIDATE_DEVICES", ("cpu", "mps"))
    monkeypatch.setattr(benchmark, "BATCH_SIZES", (1, 2))
    monkeypatch.setattr(benchmark, "CURRENT_FLOW_BASELINE_PATH", baseline_json)
    monkeypatch.setattr(benchmark, "_git_revision", lambda: "abc123")
    monkeypatch.setattr(benchmark, "_torch_version", lambda: "2.12.1")
    monkeypatch.setattr(benchmark, "_unigaze_version", lambda: "0.1.3")
    monkeypatch.setattr(benchmark, "_mps_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    monkeypatch.setenv("PYTORCH_MPS_PREFER_METAL", "1")

    perf_times = iter([0.0, 10.0, 10.0, 11.0, 11.0, 17.0, 17.0, 21.0])
    monkeypatch.setattr(
        "chess_gaze.unigaze_batch_benchmark.time.perf_counter",
        lambda: next(perf_times),
    )

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> object:
        assert capture_output is True
        assert text is True
        commands.append(command)
        subprocess_envs.append(env)
        device = command[command.index("--unigaze-device") + 1]
        batch_size = command[command.index("--unigaze-batch-size") + 1]
        if device == "cpu" and batch_size == "2":
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="MPS backend out of memory while allocating",
            )
        run_dir = tmp_path / "source" / "runs" / f"{device}{batch_size}"
        _write_qa_summary(run_dir, decoded_frames=20)
        return SimpleNamespace(
            returncode=0,
            stdout=f"{run_dir}\nviewer: {run_dir / 'viewer' / 'index.html'}\n",
            stderr="",
        )

    def fake_compare_runs(
        baseline: str | Path, candidate: str | Path, **kwargs: object
    ) -> object:
        tolerances = kwargs.get("tolerances")
        assert tolerances is None or isinstance(tolerances, EquivalenceTolerances)
        compared_runs.append(
            (
                str(baseline),
                str(candidate),
                None
                if tolerances is None
                else (
                    tolerances.appearance_pitch_yaw_radians,
                    tolerances.scene_ray_component,
                    tolerances.monitor_uv_m,
                ),
            )
        )
        return EquivalenceReport(
            baseline_run_dir=str(baseline),
            candidate_run_dir=str(candidate),
            passed=True,
            exact_mismatch_count=0,
            numeric_mismatch_count=0,
            validation_errors=[],
            mismatches=[],
            max_appearance_pitch_yaw_delta_radians=0.0,
            max_scene_ray_component_delta=0.0,
            max_monitor_uv_delta_m=0.0,
        )

    monkeypatch.setattr("chess_gaze.unigaze_batch_benchmark.subprocess.run", fake_run)
    monkeypatch.setattr(benchmark, "compare_runs", fake_compare_runs)
    benchmark_crops = object()

    def fake_load_forward_benchmark_crops(run_dir: Path) -> object:
        assert run_dir == tmp_path / "source" / "runs" / "cpu1"
        return benchmark_crops

    def fake_benchmark_unigaze_forward(
        *,
        models_root: Path,
        device: str,
        batch_size: int,
        normalized_crops: object,
    ) -> object:
        assert models_root == tmp_path / "models"
        assert normalized_crops is benchmark_crops
        forward_calls.append((device, batch_size, normalized_crops))
        return (
            benchmark._ForwardBenchmarkTiming(
                preflight_seconds=0.25,
                repetitions_seconds=[1.1, 1.0, 1.2],
                median_seconds=1.1,
                crop_count=4,
                peak_mps_memory_bytes=123 if device == "mps" else None,
            ),
            None,
        )

    monkeypatch.setattr(
        benchmark, "_load_forward_benchmark_crops", fake_load_forward_benchmark_crops
    )
    monkeypatch.setattr(
        benchmark, "_benchmark_unigaze_forward", fake_benchmark_unigaze_forward
    )

    exit_code = benchmark.main(
        [
            "--video",
            str(video_path),
            "--models-root",
            str(models_root),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = UniGazeBatchBenchmarkReport.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert [(row.device, row.batch_size) for row in report.candidate_results] == [
        ("cpu", 1),
        ("cpu", 2),
        ("mps", 1),
        ("mps", 2),
    ]
    assert report.selected_device == "mps"
    assert report.selected_batch_size == 2
    assert forward_calls == [
        ("cpu", 1, benchmark_crops),
        ("cpu", 2, benchmark_crops),
        ("mps", 1, benchmark_crops),
        ("mps", 2, benchmark_crops),
    ]
    assert report.candidate_results[0].preflight_seconds == 0.25
    assert report.candidate_results[0].unigaze_forward_status == "passed"
    assert report.candidate_results[0].unigaze_forward_crop_count == 4
    assert report.candidate_results[0].unigaze_forward_repetitions_seconds == [
        1.1,
        1.0,
        1.2,
    ]
    assert report.candidate_results[0].unigaze_forward_median_seconds == 1.1
    assert report.candidate_results[0].full_run_dir_retained is True
    assert (
        report.candidate_results[0].full_run_dir_retention_reason
        == "fresh_cpu1_equivalence_baseline"
    )
    assert report.candidate_results[1].status == "oom"
    assert report.candidate_results[1].error_code == "OOM"
    assert report.candidate_results[1].full_run_dir is None
    assert report.candidate_results[1].unigaze_forward_status == "passed"
    assert report.candidate_results[1].unigaze_forward_median_seconds == 1.1
    assert report.candidate_results[2].full_run_dir_retained is False
    assert (
        report.candidate_results[2].full_run_dir_retention_reason == "metrics_captured"
    )
    assert report.candidate_results[3].full_run_dir_retained is True
    assert (
        report.candidate_results[3].full_run_dir_retention_reason
        == "fastest_passing_mps_candidate_so_far"
    )
    assert not (tmp_path / "source" / "runs" / "mps1").exists()
    assert (tmp_path / "source" / "runs" / "mps2").is_dir()
    assert (tmp_path / "source" / "runs" / "cpu1").is_dir()
    assert compared_runs == [
        (
            str(baseline_run_dir),
            str(tmp_path / "source" / "runs" / "cpu1"),
            None,
        ),
        (
            str(tmp_path / "source" / "runs" / "cpu1"),
            str(tmp_path / "source" / "runs" / "mps1"),
            (1e-3, 1e-3, 2e-3),
        ),
        (
            str(tmp_path / "source" / "runs" / "cpu1"),
            str(tmp_path / "source" / "runs" / "mps2"),
            (1e-3, 1e-3, 2e-3),
        ),
    ]
    assert all(command[:2] == ["chess-gaze", "analyze"] for command in commands)
    for env in subprocess_envs:
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in env
        assert "PYTORCH_MPS_FAST_MATH" not in env
        assert "PYTORCH_MPS_PREFER_METAL" not in env
    equivalence_reports = sorted(output_path.parent.glob("*-equivalence.json"))
    assert [path.name for path in equivalence_reports] == [
        "benchmark-cpu1-equivalence.json",
        "benchmark-mps1-equivalence.json",
        "benchmark-mps2-equivalence.json",
    ]


def test_forward_benchmark_failure_does_not_skip_full_candidate_run(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"not a real video")
    models_root = tmp_path / "models"
    unigaze_model = models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    unigaze_model.parent.mkdir(parents=True)
    unigaze_model.write_bytes(b"model")
    baseline_json = tmp_path / "baseline.json"
    baseline_run_dir = tmp_path / "baseline-run"
    baseline_json.write_text(
        json.dumps(
            {
                "run_dir": str(baseline_run_dir),
                "qa_decoded_frames": 20,
                "source_video_sha256": "e" * 64,
                "unigaze_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "benchmark.json"
    commands: list[tuple[str, str]] = []

    monkeypatch.setattr(benchmark, "CANDIDATE_DEVICES", ("cpu", "mps"))
    monkeypatch.setattr(benchmark, "BATCH_SIZES", (1, 2))
    monkeypatch.setattr(benchmark, "CURRENT_FLOW_BASELINE_PATH", baseline_json)
    monkeypatch.setattr(benchmark, "_git_revision", lambda: "abc123")
    monkeypatch.setattr(benchmark, "_torch_version", lambda: "2.12.1")
    monkeypatch.setattr(benchmark, "_unigaze_version", lambda: "0.1.3")
    monkeypatch.setattr(benchmark, "_mps_available", lambda: True)

    perf_times = iter([0.0, 10.0, 10.0, 12.0, 12.0, 18.0, 18.0, 21.0])
    monkeypatch.setattr(
        "chess_gaze.unigaze_batch_benchmark.time.perf_counter",
        lambda: next(perf_times),
    )

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> object:
        del capture_output, text, env
        device = command[command.index("--unigaze-device") + 1]
        batch_size = command[command.index("--unigaze-batch-size") + 1]
        commands.append((device, batch_size))
        run_dir = tmp_path / "source" / "runs" / f"{device}{batch_size}"
        _write_qa_summary(run_dir, decoded_frames=20)
        return SimpleNamespace(returncode=0, stdout=f"{run_dir}\n", stderr="")

    def fake_compare_runs(
        baseline: str | Path, candidate: str | Path, **kwargs: object
    ) -> object:
        del kwargs
        return EquivalenceReport(
            baseline_run_dir=str(baseline),
            candidate_run_dir=str(candidate),
            passed=True,
            exact_mismatch_count=0,
            numeric_mismatch_count=0,
            validation_errors=[],
            mismatches=[],
            max_appearance_pitch_yaw_delta_radians=0.0,
            max_scene_ray_component_delta=0.0,
            max_monitor_uv_delta_m=0.0,
        )

    monkeypatch.setattr("chess_gaze.unigaze_batch_benchmark.subprocess.run", fake_run)
    monkeypatch.setattr(benchmark, "compare_runs", fake_compare_runs)
    monkeypatch.setattr(
        benchmark,
        "_load_forward_benchmark_crops",
        lambda run_dir: object(),
    )

    def fake_benchmark_unigaze_forward(
        *,
        models_root: Path,
        device: str,
        batch_size: int,
        normalized_crops: object,
    ) -> object:
        del models_root, normalized_crops
        if device == "mps" and batch_size == 2:
            return (
                None,
                benchmark._ForwardBenchmarkFailure(
                    preflight_seconds=0.5,
                    error_code="UNIGAZE_FORWARD_FAILED",
                    error_message="simulated forward-only failure",
                ),
            )
        return (
            benchmark._ForwardBenchmarkTiming(
                preflight_seconds=0.25,
                repetitions_seconds=[1.0, 1.1, 1.2],
                median_seconds=1.1,
                crop_count=4,
                peak_mps_memory_bytes=None,
            ),
            None,
        )

    monkeypatch.setattr(
        benchmark, "_benchmark_unigaze_forward", fake_benchmark_unigaze_forward
    )

    exit_code = benchmark.main(
        [
            "--video",
            str(video_path),
            "--models-root",
            str(models_root),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert commands == [("cpu", "1"), ("cpu", "2"), ("mps", "1"), ("mps", "2")]
    report = UniGazeBatchBenchmarkReport.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    mps2 = report.candidate_results[3]
    assert mps2.status == "passed"
    assert mps2.unigaze_forward_status == "failed"
    assert mps2.unigaze_forward_error_code == "UNIGAZE_FORWARD_FAILED"
    assert mps2.analysis_wall_seconds == 3.0
    assert report.selected_batch_size == 2


def test_prune_candidate_run_dir_requires_candidate_created_run_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "valuable-existing-run"
    _write_qa_summary(run_dir, decoded_frames=20)
    result = _candidate(
        device="mps",
        batch_size=2,
        analysis_wall_seconds=12.0,
        full_run_dir=str(run_dir),
    )

    pruned = benchmark._prune_candidate_run_dir(
        result,
        reason="metrics_captured",
        safe_prune_run_dirs=set(),
    )

    assert pruned.full_run_dir_retained is True
    assert (
        pruned.full_run_dir_retention_reason
        == "not_pruned_not_created_by_benchmark_candidate"
    )
    assert run_dir.is_dir()


def test_benchmark_cli_does_not_select_mps_when_cpu1_baseline_fails(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"not a real video")
    models_root = tmp_path / "models"
    unigaze_model = models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    unigaze_model.parent.mkdir(parents=True)
    unigaze_model.write_bytes(b"model")
    baseline_json = tmp_path / "baseline.json"
    baseline_run_dir = tmp_path / "baseline-run"
    baseline_json.write_text(
        json.dumps(
            {
                "run_dir": str(baseline_run_dir),
                "qa_decoded_frames": 20,
                "source_video_sha256": "e" * 64,
                "unigaze_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "benchmark.json"

    monkeypatch.setattr(benchmark, "CANDIDATE_DEVICES", ("cpu", "mps"))
    monkeypatch.setattr(benchmark, "BATCH_SIZES", (1, 2))
    monkeypatch.setattr(benchmark, "CURRENT_FLOW_BASELINE_PATH", baseline_json)
    monkeypatch.setattr(benchmark, "_git_revision", lambda: "abc123")
    monkeypatch.setattr(benchmark, "_torch_version", lambda: "2.12.1")
    monkeypatch.setattr(benchmark, "_unigaze_version", lambda: "0.1.3")
    monkeypatch.setattr(benchmark, "_mps_available", lambda: True)

    perf_times = iter([0.0, 10.0, 10.0, 11.0, 11.0, 15.0, 15.0, 18.0])
    monkeypatch.setattr(
        "chess_gaze.unigaze_batch_benchmark.time.perf_counter",
        lambda: next(perf_times),
    )

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> object:
        del capture_output, text, env
        device = command[command.index("--unigaze-device") + 1]
        batch_size = command[command.index("--unigaze-batch-size") + 1]
        run_dir = tmp_path / f"{device}{batch_size}"
        _write_qa_summary(run_dir, decoded_frames=20)
        return SimpleNamespace(returncode=0, stdout=f"{run_dir}\n", stderr="")

    def fake_compare_runs(baseline: str | Path, candidate: str | Path) -> object:
        cpu1_equivalence = str(baseline) == str(baseline_run_dir) and str(
            candidate
        ) == str(tmp_path / "cpu1")
        return EquivalenceReport(
            baseline_run_dir=str(baseline),
            candidate_run_dir=str(candidate),
            passed=not cpu1_equivalence,
            exact_mismatch_count=1 if cpu1_equivalence else 0,
            numeric_mismatch_count=0,
            validation_errors=[],
            mismatches=["cpu1 drifted"] if cpu1_equivalence else [],
            max_appearance_pitch_yaw_delta_radians=0.0,
            max_scene_ray_component_delta=0.0,
            max_monitor_uv_delta_m=0.0,
        )

    monkeypatch.setattr("chess_gaze.unigaze_batch_benchmark.subprocess.run", fake_run)
    monkeypatch.setattr(benchmark, "compare_runs", fake_compare_runs)

    exit_code = benchmark.main(
        [
            "--video",
            str(video_path),
            "--models-root",
            str(models_root),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = UniGazeBatchBenchmarkReport.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert report.candidate_results[0].status == "equivalence_failed"
    assert report.candidate_results[2].status == "equivalence_failed"
    assert report.candidate_results[3].status == "equivalence_failed"
    assert report.selected_batch_size is None
    assert report.selected_device is None
    assert selected_mps_batch_size(report) is None

    exit_code = benchmark.main(
        ["--report", str(output_path), "--print-selected-batch-size"]
    )
    assert exit_code == 1


def test_benchmark_cli_does_not_select_mps_without_current_flow_baseline(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"not a real video")
    models_root = tmp_path / "models"
    unigaze_model = models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    unigaze_model.parent.mkdir(parents=True)
    unigaze_model.write_bytes(b"model")
    output_path = tmp_path / "benchmark.json"
    compared_runs: list[tuple[str, str]] = []

    monkeypatch.setattr(benchmark, "CANDIDATE_DEVICES", ("cpu", "mps"))
    monkeypatch.setattr(benchmark, "BATCH_SIZES", (1, 2))
    monkeypatch.setattr(
        benchmark,
        "CURRENT_FLOW_BASELINE_PATH",
        tmp_path / "missing-baseline.json",
    )
    monkeypatch.setattr(benchmark, "_git_revision", lambda: "abc123")
    monkeypatch.setattr(benchmark, "_torch_version", lambda: "2.12.1")
    monkeypatch.setattr(benchmark, "_unigaze_version", lambda: "0.1.3")
    monkeypatch.setattr(benchmark, "_mps_available", lambda: True)

    perf_times = iter([0.0, 10.0, 10.0, 11.0, 11.0, 15.0, 15.0, 18.0])
    monkeypatch.setattr(
        "chess_gaze.unigaze_batch_benchmark.time.perf_counter",
        lambda: next(perf_times),
    )

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> object:
        del capture_output, text, env
        device = command[command.index("--unigaze-device") + 1]
        batch_size = command[command.index("--unigaze-batch-size") + 1]
        run_dir = tmp_path / f"{device}{batch_size}"
        _write_qa_summary(run_dir, decoded_frames=20)
        return SimpleNamespace(returncode=0, stdout=f"{run_dir}\n", stderr="")

    def fake_compare_runs(baseline: str | Path, candidate: str | Path) -> object:
        compared_runs.append((str(baseline), str(candidate)))
        return EquivalenceReport(
            baseline_run_dir=str(baseline),
            candidate_run_dir=str(candidate),
            passed=True,
            exact_mismatch_count=0,
            numeric_mismatch_count=0,
            validation_errors=[],
            mismatches=[],
            max_appearance_pitch_yaw_delta_radians=0.0,
            max_scene_ray_component_delta=0.0,
            max_monitor_uv_delta_m=0.0,
        )

    monkeypatch.setattr("chess_gaze.unigaze_batch_benchmark.subprocess.run", fake_run)
    monkeypatch.setattr(benchmark, "compare_runs", fake_compare_runs)

    exit_code = benchmark.main(
        [
            "--video",
            str(video_path),
            "--models-root",
            str(models_root),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = UniGazeBatchBenchmarkReport.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert report.candidate_results[0].status == "equivalence_failed"
    assert report.candidate_results[0].error_code == "CURRENT_FLOW_BASELINE_MISSING"
    assert report.candidate_results[2].status == "equivalence_failed"
    assert report.candidate_results[3].status == "equivalence_failed"
    assert compared_runs == []
    assert report.selected_batch_size is None
    assert selected_mps_batch_size(report) is None


def _report(
    candidate_results: list[BenchmarkCandidateResult],
    *,
    selected_device: Literal["mps"] | None = None,
    selected_batch_size: int | None = None,
    selected_reason: str | None = None,
) -> UniGazeBatchBenchmarkReport:
    return UniGazeBatchBenchmarkReport(
        git_revision="abc123",
        source_video="artifacts/input/nakamura_1.mp4",
        source_video_sha256="e" * 64,
        decoded_frame_count=1973,
        torch_version="2.12.1",
        unigaze_version="0.1.3",
        mps_available=True,
        mps_fallback_env="unset",
        mps_fast_math_env="unset",
        mps_prefer_metal_env="unset",
        model_asset_sha256="a" * 64,
        baseline_run_dir="artifacts/output/nakamura_1/runs/baseline",
        candidate_results=candidate_results,
        selected_device=selected_device,
        selected_batch_size=selected_batch_size,
        selected_reason=selected_reason,
    )


def _candidate(
    *,
    device: BenchmarkDevice,
    batch_size: int,
    status: BenchmarkStatus = "passed",
    preflight_seconds: float | None = None,
    analysis_wall_seconds: float | None,
    frames_per_second: float | None = 19.73,
    unigaze_forward_status: ForwardStatus = "not_run",
    unigaze_forward_repetitions_seconds: list[float] | None = None,
    unigaze_forward_median_seconds: float | None = None,
    unigaze_forward_crop_count: int | None = None,
    unigaze_forward_error_code: str | None = None,
    unigaze_forward_error_message: str | None = None,
    full_run_dir: str | None,
    full_run_dir_retained: bool | None = None,
    full_run_dir_retention_reason: str | None = None,
    qa_final_status: str | None = "complete",
    qa_decoded_frames: int | None = 1973,
    qa_counts_match: bool | None = True,
    qa_schema_validation_passed: bool | None = True,
    equivalence_report_path: str | None = "equivalence.json",
    max_appearance_pitch_yaw_delta_radians: float | None = 0.0,
    max_scene_ray_component_delta: float | None = 0.0,
    max_monitor_uv_delta_m: float | None = 0.0,
    peak_mps_memory_bytes: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> BenchmarkCandidateResult:
    return BenchmarkCandidateResult(
        device=device,
        batch_size=batch_size,
        status=status,
        preflight_seconds=preflight_seconds,
        analysis_wall_seconds=analysis_wall_seconds,
        frames_per_second=frames_per_second,
        unigaze_forward_status=unigaze_forward_status,
        unigaze_forward_repetitions_seconds=(
            []
            if unigaze_forward_repetitions_seconds is None
            else (unigaze_forward_repetitions_seconds)
        ),
        unigaze_forward_median_seconds=unigaze_forward_median_seconds,
        unigaze_forward_crop_count=unigaze_forward_crop_count,
        unigaze_forward_error_code=unigaze_forward_error_code,
        unigaze_forward_error_message=unigaze_forward_error_message,
        full_run_dir=full_run_dir,
        full_run_dir_retained=full_run_dir_retained,
        full_run_dir_retention_reason=full_run_dir_retention_reason,
        qa_final_status=qa_final_status,
        qa_decoded_frames=qa_decoded_frames,
        qa_counts_match=qa_counts_match,
        qa_schema_validation_passed=qa_schema_validation_passed,
        equivalence_report_path=equivalence_report_path,
        max_appearance_pitch_yaw_delta_radians=(max_appearance_pitch_yaw_delta_radians),
        max_scene_ray_component_delta=max_scene_ray_component_delta,
        max_monitor_uv_delta_m=max_monitor_uv_delta_m,
        peak_mps_memory_bytes=peak_mps_memory_bytes,
        error_code=error_code,
        error_message=error_message,
    )


def _write_qa_summary(run_dir: Path, *, decoded_frames: int) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "records").mkdir()
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "qa_summary.json").write_text(
        json.dumps(
            {
                "final_status": "complete",
                "counts": {"decoded_frames": decoded_frames},
                "artifact_validation": {
                    "counts_match": True,
                    "schema_validation_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
