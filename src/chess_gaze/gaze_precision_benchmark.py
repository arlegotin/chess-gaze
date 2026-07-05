from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Literal

from chess_gaze.frame_records import CalibrationRecord, FrameRecord
from chess_gaze.gaze_observation import pitch_yaw_to_unit_vector
from chess_gaze.geometry import StrictSchemaModel
from chess_gaze.scene_records import SceneSummary


class GazePrecisionRunMetrics(StrictSchemaModel):
    schema_version: Literal["gaze-precision-run-metrics-v1"] = (
        "gaze-precision-run-metrics-v1"
    )
    run_dir: str
    unigaze_preprocessing_profile: str
    frame_count: int
    valid_appearance_gaze_frames: int
    valid_appearance_gaze_rate: float
    valid_sphere_hit_frames: int | None
    valid_target_plane_hit_frames: int | None
    yaw_median_radians: float | None
    pitch_median_radians: float | None
    ray_step_median_radians: float | None
    ray_step_p95_radians: float | None
    ray_step_p99_radians: float | None


class GazePrecisionComparisonReport(StrictSchemaModel):
    schema_version: Literal["gaze-precision-comparison-v1"] = (
        "gaze-precision-comparison-v1"
    )
    generated_at_utc: str
    baseline: GazePrecisionRunMetrics
    candidate: GazePrecisionRunMetrics
    valid_appearance_gaze_rate_delta: float
    valid_sphere_hit_delta: int | None
    valid_target_plane_hit_delta: int | None
    ray_step_median_delta_radians: float | None
    ray_step_p95_delta_radians: float | None
    ray_step_p99_delta_radians: float | None


def build_gaze_precision_run_metrics(run_dir: Path) -> GazePrecisionRunMetrics:
    calibration = _load_calibration(run_dir / "calibration.json")
    frames = _load_frame_records(run_dir / "records" / "frames.jsonl")
    scene_summary = _load_scene_summary(run_dir / "scene" / "scene_summary.json")
    valid_gazes = [
        frame.appearance_gaze
        for frame in frames
        if frame.appearance_gaze.valid
        and frame.appearance_gaze.yaw_radians is not None
        and frame.appearance_gaze.pitch_radians is not None
    ]
    yaw_values = [
        gaze.yaw_radians for gaze in valid_gazes if gaze.yaw_radians is not None
    ]
    pitch_values = [
        gaze.pitch_radians for gaze in valid_gazes if gaze.pitch_radians is not None
    ]
    ray_steps = _ray_step_angles(frames)
    frame_count = len(frames)
    valid_count = len(valid_gazes)
    return GazePrecisionRunMetrics(
        run_dir=str(run_dir),
        unigaze_preprocessing_profile=calibration.unigaze_preprocessing_profile,
        frame_count=frame_count,
        valid_appearance_gaze_frames=valid_count,
        valid_appearance_gaze_rate=(valid_count / frame_count if frame_count else 0.0),
        valid_sphere_hit_frames=(
            None if scene_summary is None else scene_summary.valid_sphere_hit_frames
        ),
        valid_target_plane_hit_frames=(
            None
            if scene_summary is None
            else scene_summary.valid_target_plane_hit_frames
        ),
        yaw_median_radians=_median_or_none(yaw_values),
        pitch_median_radians=_median_or_none(pitch_values),
        ray_step_median_radians=_median_or_none(ray_steps),
        ray_step_p95_radians=_percentile_or_none(ray_steps, 0.95),
        ray_step_p99_radians=_percentile_or_none(ray_steps, 0.99),
    )


def compare_gaze_precision_runs(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    *,
    generated_at_utc: datetime | None = None,
) -> GazePrecisionComparisonReport:
    baseline = build_gaze_precision_run_metrics(baseline_run_dir)
    candidate = build_gaze_precision_run_metrics(candidate_run_dir)
    timestamp = generated_at_utc or datetime.now(UTC)
    return GazePrecisionComparisonReport(
        generated_at_utc=timestamp.replace(microsecond=0).isoformat(),
        baseline=baseline,
        candidate=candidate,
        valid_appearance_gaze_rate_delta=(
            candidate.valid_appearance_gaze_rate - baseline.valid_appearance_gaze_rate
        ),
        valid_sphere_hit_delta=_optional_int_delta(
            baseline.valid_sphere_hit_frames,
            candidate.valid_sphere_hit_frames,
        ),
        valid_target_plane_hit_delta=_optional_int_delta(
            baseline.valid_target_plane_hit_frames,
            candidate.valid_target_plane_hit_frames,
        ),
        ray_step_median_delta_radians=_optional_float_delta(
            baseline.ray_step_median_radians,
            candidate.ray_step_median_radians,
        ),
        ray_step_p95_delta_radians=_optional_float_delta(
            baseline.ray_step_p95_radians,
            candidate.ray_step_p95_radians,
        ),
        ray_step_p99_delta_radians=_optional_float_delta(
            baseline.ray_step_p99_radians,
            candidate.ray_step_p99_radians,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gaze-precision-benchmark")
    parser.add_argument("baseline_run_dir")
    parser.add_argument("candidate_run_dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = compare_gaze_precision_runs(
        Path(args.baseline_run_dir),
        Path(args.candidate_run_dir),
    )
    payload = json.dumps(
        report.model_dump(mode="json"),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if args.output is not None:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def _load_calibration(path: Path) -> CalibrationRecord:
    return CalibrationRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _load_frame_records(path: Path) -> list[FrameRecord]:
    return [
        FrameRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_scene_summary(path: Path) -> SceneSummary | None:
    if not path.exists():
        return None
    return SceneSummary.model_validate_json(path.read_text(encoding="utf-8"))


def _ray_step_angles(frames: list[FrameRecord]) -> list[float]:
    steps: list[float] = []
    previous: tuple[float, float, float] | None = None
    for frame in frames:
        gaze = frame.appearance_gaze
        if not gaze.valid or gaze.pitch_radians is None or gaze.yaw_radians is None:
            previous = None
            continue
        current = pitch_yaw_to_unit_vector(
            pitch_radians=gaze.pitch_radians,
            yaw_radians=gaze.yaw_radians,
        )
        if previous is not None:
            steps.append(_angular_distance(previous, current))
        previous = current
    return steps


def _angular_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    dot = (first[0] * second[0]) + (first[1] * second[1]) + (first[2] * second[2])
    return math.acos(max(-1.0, min(1.0, dot)))


def _median_or_none(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _percentile_or_none(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    return float(
        ordered[lower_index]
        + ((ordered[upper_index] - ordered[lower_index]) * fraction)
    )


def _optional_int_delta(baseline: int | None, candidate: int | None) -> int | None:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


def _optional_float_delta(
    baseline: float | None, candidate: float | None
) -> float | None:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


if __name__ == "__main__":
    raise SystemExit(main())
