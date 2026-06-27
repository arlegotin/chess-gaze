from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from chess_gaze.geometry import StrictSchemaModel
from chess_gaze.model_assets import sha256_file
from chess_gaze.run_equivalence import EquivalenceReport, compare_runs

CANDIDATE_DEVICES: tuple[Literal["cpu", "mps"], ...] = ("cpu", "mps")
BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 7, 8, 16, 32, 64)
OPTIONAL_MPS_EXTENSION_BATCH_SIZE = 128
CURRENT_FLOW_BASELINE_PATH = Path(
    "artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.json"
)
UNIGAZE_MODEL_RELATIVE_PATH = Path("unigaze/unigaze_h14_joint.safetensors")
MPS_ENV_VARS = (
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
)


class BenchmarkCandidateResult(StrictSchemaModel):
    device: Literal["cpu", "mps"]
    batch_size: int
    status: Literal[
        "passed",
        "preflight_failed",
        "analyze_failed",
        "equivalence_failed",
        "oom",
        "unsupported_op",
    ]
    preflight_seconds: float | None
    analysis_wall_seconds: float | None
    frames_per_second: float | None
    unigaze_forward_repetitions_seconds: list[float] = Field(default_factory=list)
    unigaze_forward_median_seconds: float | None
    full_run_dir: str | None
    qa_final_status: str | None
    qa_decoded_frames: int | None
    qa_counts_match: bool | None
    qa_schema_validation_passed: bool | None
    equivalence_report_path: str | None
    max_appearance_pitch_yaw_delta_radians: float | None
    max_scene_ray_component_delta: float | None
    max_monitor_uv_delta_m: float | None
    peak_mps_memory_bytes: int | None
    error_code: str | None
    error_message: str | None


class UniGazeBatchBenchmarkReport(StrictSchemaModel):
    schema_version: Literal["unigaze-batch-benchmark-v1"] = "unigaze-batch-benchmark-v1"
    git_revision: str
    source_video: str
    source_video_sha256: str
    decoded_frame_count: int
    torch_version: str
    unigaze_version: str
    mps_available: bool
    mps_fallback_env: str
    mps_fast_math_env: str
    mps_prefer_metal_env: str
    model_asset_sha256: str
    baseline_run_dir: str
    candidate_results: list[BenchmarkCandidateResult]
    selected_device: Literal["mps"] | None
    selected_batch_size: int | None
    selected_reason: str | None


def selected_mps_batch_size(report: UniGazeBatchBenchmarkReport) -> int | None:
    return _selected_mps_batch_size_from_results(report.candidate_results)


def _selected_mps_batch_size_from_results(
    candidate_results: list[BenchmarkCandidateResult],
) -> int | None:
    if not any(
        result.device == "cpu"
        and result.batch_size == 1
        and result.status == "passed"
        for result in candidate_results
    ):
        return None
    passing = [
        result
        for result in candidate_results
        if result.device == "mps"
        and result.batch_size > 1
        and result.status == "passed"
        and result.analysis_wall_seconds is not None
    ]
    if not passing:
        return None
    return min(passing, key=_required_analysis_wall_seconds).batch_size


def _required_analysis_wall_seconds(result: BenchmarkCandidateResult) -> float:
    if result.analysis_wall_seconds is None:
        raise ValueError("passing benchmark result must include analysis_wall_seconds")
    return result.analysis_wall_seconds


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.report is not None:
        if not args.print_selected_batch_size:
            parser.error("--report requires --print-selected-batch-size")
        return _print_selected_batch_size(Path(args.report))

    if args.print_selected_batch_size:
        parser.error("--print-selected-batch-size requires --report")

    if args.video is None or args.models_root is None or args.output is None:
        parser.error("--video, --models-root, and --output are required")

    return _run_benchmark(
        video_path=Path(args.video),
        models_root=Path(args.models_root),
        output_path=Path(args.output),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chess_gaze.unigaze_batch_benchmark"
    )
    parser.add_argument("--video", default=None)
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--print-selected-batch-size", action="store_true")
    return parser


def _print_selected_batch_size(report_path: Path) -> int:
    report = UniGazeBatchBenchmarkReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    selected = selected_mps_batch_size(report)
    if selected is None:
        print("NO_PASSING_MPS_BATCH_SIZE", file=sys.stderr)
        return 1
    print(selected)
    return 0


def _run_benchmark(
    *,
    video_path: Path,
    models_root: Path,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_metadata = _read_json_object(CURRENT_FLOW_BASELINE_PATH)
    baseline_run_dir = _string_or_default(baseline_metadata.get("run_dir"), "")
    candidate_results: list[BenchmarkCandidateResult] = []
    fresh_cpu1_run_dir: Path | None = None

    for device, batch_size in _candidate_grid():
        equivalence_baseline = _equivalence_baseline_for_candidate(
            device=device,
            batch_size=batch_size,
            baseline_run_dir=baseline_run_dir,
            fresh_cpu1_run_dir=fresh_cpu1_run_dir,
        )
        result = _run_candidate(
            video_path=video_path,
            models_root=models_root,
            benchmark_output_path=output_path,
            device=device,
            batch_size=batch_size,
            equivalence_baseline=equivalence_baseline,
        )
        candidate_results.append(result)
        if (
            device == "cpu"
            and batch_size == 1
            and result.full_run_dir is not None
            and result.status == "passed"
            and _candidate_has_valid_artifacts(result)
        ):
            fresh_cpu1_run_dir = Path(result.full_run_dir)

    selected_batch_size = _selected_mps_batch_size_from_results(candidate_results)
    report = _build_report(
        video_path=video_path,
        models_root=models_root,
        baseline_metadata=baseline_metadata,
        baseline_run_dir=baseline_run_dir,
        candidate_results=candidate_results,
        selected_batch_size=selected_batch_size,
    )
    _write_report(output_path, report)
    return 0


def _candidate_grid() -> list[tuple[Literal["cpu", "mps"], int]]:
    return [
        (device, batch_size)
        for device in CANDIDATE_DEVICES
        for batch_size in BATCH_SIZES
    ]


def _equivalence_baseline_for_candidate(
    *,
    device: Literal["cpu", "mps"],
    batch_size: int,
    baseline_run_dir: str,
    fresh_cpu1_run_dir: Path | None,
) -> Path | None:
    if device == "cpu" and batch_size == 1 and baseline_run_dir:
        return Path(baseline_run_dir)
    return fresh_cpu1_run_dir


def _run_candidate(
    *,
    video_path: Path,
    models_root: Path,
    benchmark_output_path: Path,
    device: Literal["cpu", "mps"],
    batch_size: int,
    equivalence_baseline: Path | None,
) -> BenchmarkCandidateResult:
    start = time.perf_counter()
    completed = _run_analysis_subprocess(
        video_path=video_path,
        models_root=models_root,
        output_root=_analysis_output_root(benchmark_output_path),
        device=device,
        batch_size=batch_size,
    )
    analysis_wall_seconds = time.perf_counter() - start

    if completed.returncode != 0:
        status, error_code = _status_for_failed_process(
            f"{completed.stdout}\n{completed.stderr}"
        )
        return _candidate_result(
            device=device,
            batch_size=batch_size,
            status=status,
            analysis_wall_seconds=analysis_wall_seconds,
            error_code=error_code,
            error_message=_combined_process_output(completed),
        )

    run_dir_text = _parse_run_dir(completed.stdout)
    if run_dir_text is None:
        return _candidate_result(
            device=device,
            batch_size=batch_size,
            status="analyze_failed",
            analysis_wall_seconds=analysis_wall_seconds,
            error_code="RUN_DIR_NOT_FOUND",
            error_message="chess-gaze analyze did not print a run directory",
        )

    run_dir = Path(run_dir_text)
    qa_summary, qa_error = _load_qa_summary(run_dir)
    if qa_error is not None:
        return _candidate_result(
            device=device,
            batch_size=batch_size,
            status="analyze_failed",
            analysis_wall_seconds=analysis_wall_seconds,
            full_run_dir=str(run_dir),
            error_code="QA_SUMMARY_UNREADABLE",
            error_message=qa_error,
        )

    qa_facts = _qa_facts(qa_summary)
    base_result = _candidate_result(
        device=device,
        batch_size=batch_size,
        status="passed",
        analysis_wall_seconds=analysis_wall_seconds,
        frames_per_second=_frames_per_second(
            qa_facts.decoded_frames, analysis_wall_seconds
        ),
        full_run_dir=str(run_dir),
        qa_final_status=qa_facts.final_status,
        qa_decoded_frames=qa_facts.decoded_frames,
        qa_counts_match=qa_facts.counts_match,
        qa_schema_validation_passed=qa_facts.schema_validation_passed,
    )
    if not qa_facts.passed:
        return base_result.model_copy(
            update={
                "status": "analyze_failed",
                "error_code": "QA_SUMMARY_FAILED",
                "error_message": (
                    "qa_summary.json did not report complete validated artifacts"
                ),
            }
        )

    if equivalence_baseline is None:
        if device == "cpu" and batch_size == 1:
            equivalence_baseline = run_dir
        else:
            return base_result.model_copy(
                update={
                    "status": "equivalence_failed",
                    "error_code": "EQUIVALENCE_BASELINE_MISSING",
                    "error_message": (
                        "fresh cpu/1 run was not available for equivalence"
                    ),
                }
            )

    equivalence_report_path = _equivalence_report_path(
        benchmark_output_path, device, batch_size
    )
    try:
        equivalence_report = compare_runs(equivalence_baseline, run_dir)
    except Exception as exc:
        return base_result.model_copy(
            update={
                "status": "equivalence_failed",
                "error_code": type(exc).__name__,
                "error_message": str(exc),
            }
        )

    _write_equivalence_report(equivalence_report_path, equivalence_report)
    updates: dict[str, Any] = {
        "equivalence_report_path": str(equivalence_report_path),
        "max_appearance_pitch_yaw_delta_radians": (
            equivalence_report.max_appearance_pitch_yaw_delta_radians
        ),
        "max_scene_ray_component_delta": (
            equivalence_report.max_scene_ray_component_delta
        ),
        "max_monitor_uv_delta_m": equivalence_report.max_monitor_uv_delta_m,
    }
    if not equivalence_report.passed:
        updates.update(
            {
                "status": "equivalence_failed",
                "error_code": "EQUIVALENCE_FAILED",
                "error_message": _equivalence_error_message(equivalence_report),
            }
        )
    return base_result.model_copy(update=updates)


def _candidate_has_valid_artifacts(result: BenchmarkCandidateResult) -> bool:
    return (
        result.qa_final_status == "complete"
        and result.qa_counts_match is True
        and result.qa_schema_validation_passed is True
    )


def _run_analysis_subprocess(
    *,
    video_path: Path,
    models_root: Path,
    output_root: Path,
    device: Literal["cpu", "mps"],
    batch_size: int,
) -> subprocess.CompletedProcess[str]:
    command = [
        "chess-gaze",
        "analyze",
        str(video_path),
        "--output-root",
        str(output_root),
        "--models-root",
        str(models_root),
        "--unigaze-device",
        device,
        "--unigaze-batch-size",
        str(batch_size),
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=_benchmark_subprocess_env(),
    )


def _candidate_result(
    *,
    device: Literal["cpu", "mps"],
    batch_size: int,
    status: Literal[
        "passed",
        "preflight_failed",
        "analyze_failed",
        "equivalence_failed",
        "oom",
        "unsupported_op",
    ],
    analysis_wall_seconds: float | None,
    frames_per_second: float | None = None,
    full_run_dir: str | None = None,
    qa_final_status: str | None = None,
    qa_decoded_frames: int | None = None,
    qa_counts_match: bool | None = None,
    qa_schema_validation_passed: bool | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> BenchmarkCandidateResult:
    return BenchmarkCandidateResult(
        device=device,
        batch_size=batch_size,
        status=status,
        preflight_seconds=None,
        analysis_wall_seconds=analysis_wall_seconds,
        frames_per_second=frames_per_second,
        unigaze_forward_repetitions_seconds=[],
        unigaze_forward_median_seconds=None,
        full_run_dir=full_run_dir,
        qa_final_status=qa_final_status,
        qa_decoded_frames=qa_decoded_frames,
        qa_counts_match=qa_counts_match,
        qa_schema_validation_passed=qa_schema_validation_passed,
        equivalence_report_path=None,
        max_appearance_pitch_yaw_delta_radians=None,
        max_scene_ray_component_delta=None,
        max_monitor_uv_delta_m=None,
        peak_mps_memory_bytes=None,
        error_code=error_code,
        error_message=error_message,
    )


class _QAFacts(StrictSchemaModel):
    final_status: str | None
    decoded_frames: int | None
    counts_match: bool | None
    schema_validation_passed: bool | None

    @property
    def passed(self) -> bool:
        return (
            self.final_status == "complete"
            and self.counts_match is True
            and self.schema_validation_passed is True
        )


def _qa_facts(qa_summary: dict[str, Any]) -> _QAFacts:
    counts = _mapping_or_empty(qa_summary.get("counts"))
    artifact_validation = _mapping_or_empty(qa_summary.get("artifact_validation"))
    return _QAFacts(
        final_status=_string_or_none(qa_summary.get("final_status")),
        decoded_frames=_int_or_none(counts.get("decoded_frames")),
        counts_match=_bool_or_none(artifact_validation.get("counts_match")),
        schema_validation_passed=_bool_or_none(
            artifact_validation.get("schema_validation_passed")
        ),
    )


def _frames_per_second(
    decoded_frames: int | None, analysis_wall_seconds: float | None
) -> float | None:
    if (
        decoded_frames is None
        or analysis_wall_seconds is None
        or analysis_wall_seconds <= 0.0
    ):
        return None
    return decoded_frames / analysis_wall_seconds


def _load_qa_summary(run_dir: Path) -> tuple[dict[str, Any], str | None]:
    path = run_dir / "qa_summary.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"Unable to read qa_summary.json at {path}: {exc}"
    if not isinstance(value, dict):
        return {}, f"Invalid qa_summary.json at {path}: expected object"
    return value, None


def _parse_run_dir(stdout: str) -> str | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _status_for_failed_process(
    output: str,
) -> tuple[
    Literal["analyze_failed", "oom", "unsupported_op"],
    str,
]:
    normalized = output.lower()
    if "out of memory" in normalized or "oom" in normalized:
        return "oom", "OOM"
    if "unsupported" in normalized or "not implemented for" in normalized:
        return "unsupported_op", "UNSUPPORTED_OP"
    return "analyze_failed", "ANALYZE_FAILED"


def _combined_process_output(completed: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return output.strip()


def _benchmark_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in MPS_ENV_VARS:
        env.pop(name, None)
    return env


def _analysis_output_root(benchmark_output_path: Path) -> Path:
    if benchmark_output_path.parent.name == "benchmarks":
        return benchmark_output_path.parent.parent
    return benchmark_output_path.parent


def _equivalence_report_path(
    benchmark_output_path: Path, device: Literal["cpu", "mps"], batch_size: int
) -> Path:
    return benchmark_output_path.with_name(
        f"{benchmark_output_path.stem}-{device}{batch_size}-equivalence.json"
    )


def _write_equivalence_report(path: Path, report: EquivalenceReport) -> None:
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _equivalence_error_message(report: EquivalenceReport) -> str:
    if report.validation_errors:
        return report.validation_errors[0]
    if report.mismatches:
        return report.mismatches[0]
    return (
        "equivalence report failed with "
        f"{report.exact_mismatch_count} exact and "
        f"{report.numeric_mismatch_count} numeric mismatches"
    )


def _build_report(
    *,
    video_path: Path,
    models_root: Path,
    baseline_metadata: dict[str, Any],
    baseline_run_dir: str,
    candidate_results: list[BenchmarkCandidateResult],
    selected_batch_size: int | None,
) -> UniGazeBatchBenchmarkReport:
    selected_reason = (
        "fastest passing MPS batch size above 1 by analysis_wall_seconds"
        if selected_batch_size is not None
        else "no passing MPS batch size above 1"
    )
    return UniGazeBatchBenchmarkReport(
        git_revision=_git_revision(),
        source_video=str(video_path),
        source_video_sha256=_source_video_sha256(video_path, baseline_metadata),
        decoded_frame_count=_decoded_frame_count(candidate_results, baseline_metadata),
        torch_version=_metadata_or_current(
            baseline_metadata, "torch_version", _torch_version
        ),
        unigaze_version=_metadata_or_current(
            baseline_metadata, "unigaze_version", _unigaze_version
        ),
        mps_available=_mps_available(),
        mps_fallback_env="unset",
        mps_fast_math_env="unset",
        mps_prefer_metal_env="unset",
        model_asset_sha256=_model_asset_sha256(models_root, baseline_metadata),
        baseline_run_dir=baseline_run_dir,
        candidate_results=candidate_results,
        selected_device="mps" if selected_batch_size is not None else None,
        selected_batch_size=selected_batch_size,
        selected_reason=selected_reason,
    )


def _write_report(path: Path, report: UniGazeBatchBenchmarkReport) -> None:
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return value


def _source_video_sha256(video_path: Path, baseline_metadata: dict[str, Any]) -> str:
    try:
        return sha256_file(video_path)
    except OSError:
        return _string_or_default(baseline_metadata.get("source_video_sha256"), "")


def _model_asset_sha256(models_root: Path, baseline_metadata: dict[str, Any]) -> str:
    model_path = models_root / UNIGAZE_MODEL_RELATIVE_PATH
    try:
        return sha256_file(model_path)
    except OSError:
        return _string_or_default(baseline_metadata.get("unigaze_sha256"), "")


def _decoded_frame_count(
    candidate_results: list[BenchmarkCandidateResult],
    baseline_metadata: dict[str, Any],
) -> int:
    for result in candidate_results:
        if result.device == "cpu" and result.batch_size == 1:
            if result.qa_decoded_frames is not None:
                return result.qa_decoded_frames
    baseline_count = _int_or_none(baseline_metadata.get("qa_decoded_frames"))
    if baseline_count is not None:
        return baseline_count
    for result in candidate_results:
        if result.qa_decoded_frames is not None:
            return result.qa_decoded_frames
    return 0


def _metadata_or_current(
    baseline_metadata: dict[str, Any], key: str, current: Callable[[], str]
) -> str:
    value = current()
    if value:
        return value
    return _string_or_default(baseline_metadata.get(key), "unknown")


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        return "unknown"
    revision = completed.stdout.strip()
    return revision or "unknown"


def _torch_version() -> str:
    try:
        import torch
    except ImportError:
        return ""
    return str(torch.__version__)


def _unigaze_version() -> str:
    try:
        return version("unigaze")
    except PackageNotFoundError:
        return ""


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
