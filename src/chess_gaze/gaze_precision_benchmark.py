from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast

from chess_gaze.artifact_runs import frame_id
from chess_gaze.frame_records import (
    CalibrationRecord,
    FrameRecord,
    RunManifest,
    VideoManifest,
    read_run_manifest_artifact_json,
)
from chess_gaze.gaze_observation import pitch_yaw_to_unit_vector
from chess_gaze.geometry import StrictSchemaModel
from chess_gaze.scene_records import SceneSummary

GazePrecisionExperimentalVariable = Literal["unigaze_preprocessing"]

EXPERIMENTAL_VARIABLE_FIELDS: dict[
    GazePrecisionExperimentalVariable, frozenset[str]
] = {
    "unigaze_preprocessing": frozenset(
        {
            "calibration.unigaze_preprocessing_profile",
            "calibration.unigaze_face_crop_scale",
            "calibration.unigaze_image_mean_rgb",
            "calibration.unigaze_image_std_rgb",
        }
    )
}


@dataclass(frozen=True)
class _RunArtifacts:
    run_dir: Path
    run_manifest: RunManifest
    video_manifest: VideoManifest
    calibration: CalibrationRecord
    frames: list[FrameRecord]
    scene_summary: SceneSummary | None


class GazePrecisionRunMetrics(StrictSchemaModel):
    schema_version: Literal["gaze-precision-run-metrics-v2"] = (
        "gaze-precision-run-metrics-v2"
    )
    run_dir: str
    source_path: str
    source_sha256: str
    frame_width: int
    frame_height: int
    frame_count_decoded: int
    pts_sequence_sha256: str | None
    pts_sequence_usable: bool
    unigaze_model_id: str | None
    unigaze_model_checksum_sha256: str | None
    unigaze_preprocessing_profile: str
    frame_count: int
    valid_appearance_gaze_frames: int
    valid_appearance_gaze_rate: float
    valid_sphere_hit_frames: int | None
    valid_target_plane_hit_frames: int | None
    in_bounds_target_plane_hit_frames: int | None
    yaw_median_radians: float | None
    pitch_median_radians: float | None
    ray_step_median_radians: float | None
    ray_step_p95_radians: float | None
    ray_step_p99_radians: float | None
    ray_speed_median_degrees_per_second: float | None
    ray_speed_p95_degrees_per_second: float | None
    ray_speed_p99_degrees_per_second: float | None


class GazePrecisionComparisonReport(StrictSchemaModel):
    schema_version: Literal["gaze-precision-comparison-v2"] = (
        "gaze-precision-comparison-v2"
    )
    generated_at_utc: str
    experimental_variable: GazePrecisionExperimentalVariable
    baseline: GazePrecisionRunMetrics
    candidate: GazePrecisionRunMetrics
    valid_appearance_gaze_rate_delta: float
    valid_sphere_hit_delta: int | None
    valid_target_plane_hit_delta: int | None
    in_bounds_target_plane_hit_delta: int | None
    ray_step_median_delta_radians: float | None
    ray_step_p95_delta_radians: float | None
    ray_step_p99_delta_radians: float | None
    ray_speed_median_delta_degrees_per_second: float | None
    ray_speed_p95_delta_degrees_per_second: float | None
    ray_speed_p99_delta_degrees_per_second: float | None


def build_gaze_precision_run_metrics(run_dir: Path) -> GazePrecisionRunMetrics:
    return _build_gaze_precision_run_metrics(_load_run_artifacts(run_dir))


def _build_gaze_precision_run_metrics(
    artifacts: _RunArtifacts,
) -> GazePrecisionRunMetrics:
    frames = artifacts.frames
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
    ray_steps, ray_speeds = _ray_steps(
        frames,
        timestamps_usable=artifacts.video_manifest.pts_sequence_usable,
    )
    frame_count = len(frames)
    valid_count = len(valid_gazes)
    scene_summary = artifacts.scene_summary
    return GazePrecisionRunMetrics(
        run_dir=str(artifacts.run_dir),
        source_path=artifacts.video_manifest.source_path,
        source_sha256=artifacts.video_manifest.source_sha256,
        frame_width=artifacts.video_manifest.frame_width,
        frame_height=artifacts.video_manifest.frame_height,
        frame_count_decoded=artifacts.video_manifest.frame_count_decoded,
        pts_sequence_sha256=artifacts.video_manifest.pts_sequence_sha256,
        pts_sequence_usable=artifacts.video_manifest.pts_sequence_usable,
        unigaze_model_id=artifacts.run_manifest.inference.unigaze_model_id,
        unigaze_model_checksum_sha256=(
            artifacts.run_manifest.inference.unigaze_model_checksum_sha256
        ),
        unigaze_preprocessing_profile=(
            artifacts.calibration.unigaze_preprocessing_profile
        ),
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
        in_bounds_target_plane_hit_frames=(
            None
            if scene_summary is None
            else scene_summary.in_bounds_target_plane_hit_frames
        ),
        yaw_median_radians=_median_or_none(yaw_values),
        pitch_median_radians=_median_or_none(pitch_values),
        ray_step_median_radians=_median_or_none(ray_steps),
        ray_step_p95_radians=_percentile_or_none(ray_steps, 0.95),
        ray_step_p99_radians=_percentile_or_none(ray_steps, 0.99),
        ray_speed_median_degrees_per_second=(
            None if ray_speeds is None else _median_or_none(ray_speeds)
        ),
        ray_speed_p95_degrees_per_second=(
            None if ray_speeds is None else _percentile_or_none(ray_speeds, 0.95)
        ),
        ray_speed_p99_degrees_per_second=(
            None if ray_speeds is None else _percentile_or_none(ray_speeds, 0.99)
        ),
    )


def compare_gaze_precision_runs(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    *,
    experimental_variable: GazePrecisionExperimentalVariable,
    generated_at_utc: datetime | None = None,
) -> GazePrecisionComparisonReport:
    allowed_fields = EXPERIMENTAL_VARIABLE_FIELDS.get(experimental_variable)
    if allowed_fields is None:
        raise ValueError(
            f"unsupported experimental_variable: {experimental_variable!r}"
        )
    baseline_artifacts = _load_run_artifacts(baseline_run_dir)
    candidate_artifacts = _load_run_artifacts(candidate_run_dir)
    _validate_comparison(
        baseline_artifacts,
        candidate_artifacts,
        experimental_variable=experimental_variable,
        allowed_fields=allowed_fields,
    )
    baseline = _build_gaze_precision_run_metrics(baseline_artifacts)
    candidate = _build_gaze_precision_run_metrics(candidate_artifacts)
    timestamp = generated_at_utc or datetime.now(UTC)
    return GazePrecisionComparisonReport(
        generated_at_utc=timestamp.replace(microsecond=0).isoformat(),
        experimental_variable=experimental_variable,
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
        in_bounds_target_plane_hit_delta=_optional_int_delta(
            baseline.in_bounds_target_plane_hit_frames,
            candidate.in_bounds_target_plane_hit_frames,
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
        ray_speed_median_delta_degrees_per_second=_optional_float_delta(
            baseline.ray_speed_median_degrees_per_second,
            candidate.ray_speed_median_degrees_per_second,
        ),
        ray_speed_p95_delta_degrees_per_second=_optional_float_delta(
            baseline.ray_speed_p95_degrees_per_second,
            candidate.ray_speed_p95_degrees_per_second,
        ),
        ray_speed_p99_delta_degrees_per_second=_optional_float_delta(
            baseline.ray_speed_p99_degrees_per_second,
            candidate.ray_speed_p99_degrees_per_second,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gaze-precision-benchmark")
    parser.add_argument("baseline_run_dir")
    parser.add_argument("candidate_run_dir")
    parser.add_argument(
        "--experimental-variable",
        required=True,
        choices=tuple(EXPERIMENTAL_VARIABLE_FIELDS),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = compare_gaze_precision_runs(
        Path(args.baseline_run_dir),
        Path(args.candidate_run_dir),
        experimental_variable=cast(
            GazePrecisionExperimentalVariable, args.experimental_variable
        ),
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


def _load_run_artifacts(run_dir: Path) -> _RunArtifacts:
    run_manifest = read_run_manifest_artifact_json(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    video_manifest = VideoManifest.model_validate_json(
        (run_dir / "video_manifest.json").read_text(encoding="utf-8")
    )
    artifacts = _RunArtifacts(
        run_dir=run_dir,
        run_manifest=run_manifest,
        video_manifest=video_manifest,
        calibration=_load_calibration(run_dir / "calibration.json"),
        frames=_load_frame_records(run_dir / "records" / "frames.jsonl"),
        scene_summary=_load_scene_summary(run_dir / "scene" / "scene_summary.json"),
    )
    _validate_run_artifacts(artifacts)
    return artifacts


def _validate_run_artifacts(artifacts: _RunArtifacts) -> None:
    embedded_video = artifacts.run_manifest.video.model_dump(mode="json")
    standalone_video = artifacts.video_manifest.model_dump(mode="json")
    embedded_mismatches = _mismatched_paths(embedded_video, standalone_video)
    if embedded_mismatches:
        paths = [f"run_manifest.video.{path}" for path in embedded_mismatches]
        raise ValueError(
            "embedded video manifest differs from video_manifest.json: "
            + ", ".join(paths)
        )

    identity_issues: list[str] = []
    if len(artifacts.frames) != artifacts.video_manifest.frame_count_decoded:
        identity_issues.append("video.frame_count_decoded")
    expected_indices = list(range(len(artifacts.frames)))
    if [frame.frame_index for frame in artifacts.frames] != expected_indices:
        identity_issues.append("frames.frame_index")
    if [frame.frame_id for frame in artifacts.frames] != [
        frame_id(index) for index in expected_indices
    ]:
        identity_issues.append("frames.frame_id")
    if identity_issues:
        raise ValueError(
            "invalid run frame identity: " + ", ".join(sorted(identity_issues))
        )


def _validate_comparison(
    baseline: _RunArtifacts,
    candidate: _RunArtifacts,
    *,
    experimental_variable: GazePrecisionExperimentalVariable,
    allowed_fields: frozenset[str],
) -> None:
    for label, artifacts in (("baseline", baseline), ("candidate", candidate)):
        if not artifacts.run_manifest.inference.unigaze_model_checksum_sha256:
            raise ValueError(
                f"{label} is missing inference.unigaze_model_checksum_sha256"
            )

    differences = _mismatched_paths(
        _comparison_fields(baseline),
        _comparison_fields(candidate),
    )
    unexpected = [path for path in differences if path not in allowed_fields]
    if unexpected:
        raise ValueError(
            "runs differ outside declared experimental variable: "
            + ", ".join(unexpected)
        )
    if not any(path in allowed_fields for path in differences):
        raise ValueError(
            f"experimental_variable {experimental_variable!r} has no actual "
            "difference; expected one of: " + ", ".join(sorted(allowed_fields))
        )


def _comparison_fields(artifacts: _RunArtifacts) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "input_path": artifacts.run_manifest.input_path,
        "frames.frame_id": tuple(frame.frame_id for frame in artifacts.frames),
        "frames.frame_index": tuple(frame.frame_index for frame in artifacts.frames),
        "frames.timestamp_seconds": tuple(
            frame.timestamp_seconds for frame in artifacts.frames
        ),
    }
    for prefix, payload in (
        ("video", artifacts.video_manifest.model_dump(mode="json")),
        ("inference", artifacts.run_manifest.inference.model_dump(mode="json")),
        (
            "frame_image_retention",
            artifacts.run_manifest.frame_image_retention.model_dump(mode="json"),
        ),
        (
            "crop_image_retention",
            artifacts.run_manifest.crop_image_retention.model_dump(mode="json"),
        ),
        (
            "qa_summary_policy",
            artifacts.run_manifest.qa_summary_policy.model_dump(mode="json"),
        ),
        ("calibration", artifacts.calibration.model_dump(mode="json")),
    ):
        fields.update(_flatten_fields(prefix, payload))
    return fields


def _flatten_fields(prefix: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        flattened.update(_flatten_fields(f"{prefix}.{key}", value[key]))
    return flattened


def _mismatched_paths(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    return sorted(
        path
        for path in baseline.keys() | candidate.keys()
        if baseline.get(path) != candidate.get(path)
    )


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


def _ray_steps(
    frames: list[FrameRecord],
    *,
    timestamps_usable: bool,
) -> tuple[list[float], list[float] | None]:
    steps: list[float] = []
    speeds: list[float] | None = [] if timestamps_usable else None
    if timestamps_usable:
        for previous_frame, current_frame in pairwise(frames):
            delta_seconds = (
                current_frame.timestamp_seconds - previous_frame.timestamp_seconds
            )
            if not math.isfinite(delta_seconds) or delta_seconds <= 0.0:
                raise ValueError(
                    "frames.timestamp_seconds must contain finite positive deltas "
                    "when video.pts_sequence_usable is true"
                )

    previous: tuple[float, float, float] | None = None
    previous_timestamp: float | None = None
    for frame in frames:
        gaze = frame.appearance_gaze
        if not gaze.valid or gaze.pitch_radians is None or gaze.yaw_radians is None:
            previous = None
            previous_timestamp = None
            continue
        current = pitch_yaw_to_unit_vector(
            pitch_radians=gaze.pitch_radians,
            yaw_radians=gaze.yaw_radians,
        )
        if previous is not None:
            step = _angular_distance(previous, current)
            steps.append(step)
            if speeds is not None and previous_timestamp is not None:
                speeds.append(
                    math.degrees(step) / (frame.timestamp_seconds - previous_timestamp)
                )
        previous = current
        previous_timestamp = frame.timestamp_seconds
    return steps, speeds


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
