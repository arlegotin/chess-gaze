from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from chess_gaze.geometry import StrictSchemaModel


class EquivalenceTolerances(StrictSchemaModel):
    appearance_pitch_yaw_radians: float = 1e-6
    scene_ray_component: float = 1e-6
    sphere_hit_angle_radians: float = 1e-6


class EquivalenceReport(StrictSchemaModel):
    schema_version: Literal["run-equivalence-v1"] = "run-equivalence-v1"
    baseline_run_dir: str
    candidate_run_dir: str
    passed: bool
    exact_mismatch_count: int
    numeric_mismatch_count: int
    validation_errors: list[str] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)
    max_appearance_pitch_yaw_delta_radians: float
    max_scene_ray_component_delta: float
    max_sphere_hit_angle_delta_radians: float


@dataclass(frozen=True)
class _RunArtifacts:
    frames: list[dict[str, Any]]
    scene_frames: list[dict[str, Any]]
    qa_summary: dict[str, Any]
    viewer_data: dict[str, Any]


@dataclass
class _Comparison:
    tolerances: EquivalenceTolerances
    exact_mismatches: list[str] = field(default_factory=list)
    numeric_mismatches: list[str] = field(default_factory=list)
    max_appearance_pitch_yaw_delta_radians: float = 0.0
    max_scene_ray_component_delta: float = 0.0
    max_sphere_hit_angle_delta_radians: float = 0.0

    def exact(self, path: str, baseline: Any, candidate: Any) -> None:
        if baseline != candidate:
            self.exact_mismatches.append(
                f"{path} exact mismatch: baseline={_format_value(baseline)} "
                f"candidate={_format_value(candidate)}"
            )

    def numeric(
        self,
        path: str,
        baseline: Any,
        candidate: Any,
        *,
        tolerance: float,
        max_delta_kind: Literal["appearance", "scene_ray", "sphere_hit_angle"],
    ) -> None:
        baseline_number = _finite_number_or_none(baseline)
        candidate_number = _finite_number_or_none(candidate)
        if baseline_number is None or candidate_number is None:
            self.numeric_mismatches.append(
                f"{path} numeric presence mismatch: "
                f"baseline={_format_value(baseline)} "
                f"candidate={_format_value(candidate)}"
            )
            return

        delta = abs(baseline_number - candidate_number)
        self._record_max_delta(max_delta_kind, delta)
        if delta > tolerance:
            self.numeric_mismatches.append(
                f"{path} numeric mismatch: baseline={_format_float(baseline_number)} "
                f"candidate={_format_float(candidate_number)} "
                f"delta={_format_float(delta)} "
                f"tolerance={_format_float(tolerance)}"
            )

    def _record_max_delta(
        self,
        max_delta_kind: Literal["appearance", "scene_ray", "sphere_hit_angle"],
        delta: float,
    ) -> None:
        if max_delta_kind == "appearance":
            self.max_appearance_pitch_yaw_delta_radians = max(
                self.max_appearance_pitch_yaw_delta_radians,
                delta,
            )
        elif max_delta_kind == "scene_ray":
            self.max_scene_ray_component_delta = max(
                self.max_scene_ray_component_delta,
                delta,
            )
        else:
            self.max_sphere_hit_angle_delta_radians = max(
                self.max_sphere_hit_angle_delta_radians,
                delta,
            )


def compare_runs(
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    *,
    tolerances: EquivalenceTolerances | None = None,
) -> EquivalenceReport:
    baseline_path = Path(baseline_run_dir)
    candidate_path = Path(candidate_run_dir)
    active_tolerances = tolerances or EquivalenceTolerances()
    validation_errors: list[str] = []

    baseline = _load_run_artifacts(
        baseline_path,
        label="baseline",
        validation_errors=validation_errors,
    )
    candidate = _load_run_artifacts(
        candidate_path,
        label="candidate",
        validation_errors=validation_errors,
    )

    comparison = _Comparison(active_tolerances)
    _compare_frame_records(comparison, baseline.frames, candidate.frames)
    _compare_scene_frame_records(
        comparison,
        baseline.scene_frames,
        candidate.scene_frames,
    )
    _compare_summary_counts(comparison, baseline.qa_summary, candidate.qa_summary)
    _compare_viewer_counts(comparison, baseline.viewer_data, candidate.viewer_data)

    mismatches = comparison.exact_mismatches + comparison.numeric_mismatches
    passed = not validation_errors and not mismatches
    return EquivalenceReport(
        baseline_run_dir=str(baseline_path),
        candidate_run_dir=str(candidate_path),
        passed=passed,
        exact_mismatch_count=len(comparison.exact_mismatches),
        numeric_mismatch_count=len(comparison.numeric_mismatches),
        validation_errors=validation_errors,
        mismatches=mismatches,
        max_appearance_pitch_yaw_delta_radians=(
            comparison.max_appearance_pitch_yaw_delta_radians
        ),
        max_scene_ray_component_delta=comparison.max_scene_ray_component_delta,
        max_sphere_hit_angle_delta_radians=(
            comparison.max_sphere_hit_angle_delta_radians
        ),
    )


def _load_run_artifacts(
    run_dir: Path,
    *,
    label: str,
    validation_errors: list[str],
) -> _RunArtifacts:
    records_dir = run_dir / "records"
    return _RunArtifacts(
        frames=_read_jsonl(
            records_dir / "frames.jsonl",
            label=f"{label} frame records",
            validation_errors=validation_errors,
        ),
        scene_frames=_read_jsonl(
            records_dir / "scene_frames.jsonl",
            label=f"{label} scene frame records",
            validation_errors=validation_errors,
        ),
        qa_summary=_read_json(
            run_dir / "qa_summary.json",
            label=f"{label} QA summary",
            validation_errors=validation_errors,
        ),
        viewer_data=_read_json(
            run_dir / "viewer" / "scene-data.json",
            label=f"{label} viewer scene data",
            validation_errors=validation_errors,
        ),
    )


def _read_jsonl(
    path: Path,
    *,
    label: str,
    validation_errors: list[str],
) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        validation_errors.append(f"Unable to read {label} at {path}: {exc}")
        return []

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            validation_errors.append(
                f"Invalid {label} JSON at {path}:{line_number}: {exc}"
            )
            continue
        if not isinstance(value, dict):
            validation_errors.append(
                f"Invalid {label} record at {path}:{line_number}: "
                f"expected object, got {type(value).__name__}"
            )
            continue
        records.append(value)
    return records


def _read_json(
    path: Path,
    *,
    label: str,
    validation_errors: list[str],
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        validation_errors.append(f"Invalid {label} at {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        validation_errors.append(
            f"Invalid {label} at {path}: expected object, got {type(value).__name__}"
        )
        return {}
    return value


def _compare_frame_records(
    comparison: _Comparison,
    baseline_frames: list[dict[str, Any]],
    candidate_frames: list[dict[str, Any]],
) -> None:
    comparison.exact(
        "records.frames.count",
        len(baseline_frames),
        len(candidate_frames),
    )
    for index, (baseline, candidate) in enumerate(
        zip(baseline_frames, candidate_frames, strict=False)
    ):
        prefix = f"frames[{index}]"
        comparison.exact(
            f"{prefix}.frame_id", baseline.get("frame_id"), candidate.get("frame_id")
        )
        comparison.exact(
            f"{prefix}.frame_index",
            baseline.get("frame_index"),
            candidate.get("frame_index"),
        )
        comparison.exact(
            f"{prefix}.status", baseline.get("status"), candidate.get("status")
        )
        _compare_frame_validity_fields(comparison, prefix, baseline, candidate)
        _compare_appearance_gaze(comparison, prefix, baseline, candidate)
        comparison.exact(
            f"{prefix}.errors.codes",
            _error_codes(baseline.get("errors")),
            _error_codes(candidate.get("errors")),
        )


def _compare_frame_validity_fields(
    comparison: _Comparison,
    prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for field_name, validity_key in (
        ("face", "present"),
        ("left_eye", "present"),
        ("right_eye", "present"),
        ("head_pose", "valid"),
        ("geometric_gaze", "valid"),
        ("appearance_gaze", "valid"),
        ("recommended_gaze", "valid"),
    ):
        baseline_record = _mapping_or_empty(baseline.get(field_name))
        candidate_record = _mapping_or_empty(candidate.get(field_name))
        comparison.exact(
            f"{prefix}.{field_name}.{validity_key}",
            baseline_record.get(validity_key),
            candidate_record.get(validity_key),
        )
        comparison.exact(
            f"{prefix}.{field_name}.reason_invalid",
            baseline_record.get("reason_invalid"),
            candidate_record.get("reason_invalid"),
        )


def _compare_appearance_gaze(
    comparison: _Comparison,
    prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    baseline_gaze = _mapping_or_empty(baseline.get("appearance_gaze"))
    candidate_gaze = _mapping_or_empty(candidate.get("appearance_gaze"))
    if (
        baseline_gaze.get("valid") is not True
        or candidate_gaze.get("valid") is not True
    ):
        return
    for component in ("pitch_radians", "yaw_radians"):
        comparison.numeric(
            f"{prefix}.appearance_gaze.{component}",
            baseline_gaze.get(component),
            candidate_gaze.get(component),
            tolerance=comparison.tolerances.appearance_pitch_yaw_radians,
            max_delta_kind="appearance",
        )


def _compare_scene_frame_records(
    comparison: _Comparison,
    baseline_frames: list[dict[str, Any]],
    candidate_frames: list[dict[str, Any]],
) -> None:
    comparison.exact(
        "records.scene_frames.count",
        len(baseline_frames),
        len(candidate_frames),
    )
    for index, (baseline, candidate) in enumerate(
        zip(baseline_frames, candidate_frames, strict=False)
    ):
        prefix = f"scene_frames[{index}]"
        comparison.exact(
            f"{prefix}.frame_id", baseline.get("frame_id"), candidate.get("frame_id")
        )
        comparison.exact(
            f"{prefix}.frame_index",
            baseline.get("frame_index"),
            candidate.get("frame_index"),
        )
        comparison.exact(
            f"{prefix}.source_frame_status",
            baseline.get("source_frame_status"),
            candidate.get("source_frame_status"),
        )
        for field_name in (
            "valid_for_scene_center",
            "valid_for_sphere_projection",
        ):
            comparison.exact(
                f"{prefix}.{field_name}",
                baseline.get(field_name),
                candidate.get(field_name),
            )
        _compare_scene_validity_fields(comparison, prefix, baseline, candidate)
        _compare_scene_ray(comparison, prefix, baseline, candidate)
        _compare_sphere_hit(comparison, prefix, baseline, candidate)
        comparison.exact(
            f"{prefix}.diagnostics.source_error_codes",
            _get_path(baseline, ("diagnostics", "source_error_codes")),
            _get_path(candidate, ("diagnostics", "source_error_codes")),
        )


def _compare_scene_validity_fields(
    comparison: _Comparison,
    prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for field_name in (
        "left_eye",
        "right_eye",
        "eye_midpoint",
        "head",
        "unigaze_ray",
        "sphere_hit",
    ):
        _compare_valid_reason_record(
            comparison,
            f"{prefix}.{field_name}",
            _mapping_or_empty(baseline.get(field_name)),
            _mapping_or_empty(candidate.get(field_name)),
        )


def _compare_valid_reason_record(
    comparison: _Comparison,
    prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for field_name in ("valid", "reason_invalid", "source_reason_invalid"):
        comparison.exact(
            f"{prefix}.{field_name}",
            baseline.get(field_name),
            candidate.get(field_name),
        )


def _compare_scene_ray(
    comparison: _Comparison,
    prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    baseline_ray = _mapping_or_empty(baseline.get("unigaze_ray"))
    candidate_ray = _mapping_or_empty(candidate.get("unigaze_ray"))
    if baseline_ray.get("valid") is not True or candidate_ray.get("valid") is not True:
        return
    for vector_name in ("direction_camera", "direction_scene"):
        baseline_vector = _mapping_or_empty(baseline_ray.get(vector_name))
        candidate_vector = _mapping_or_empty(candidate_ray.get(vector_name))
        for component in ("x", "y", "z"):
            comparison.numeric(
                f"{prefix}.unigaze_ray.{vector_name}.{component}",
                baseline_vector.get(component),
                candidate_vector.get(component),
                tolerance=comparison.tolerances.scene_ray_component,
                max_delta_kind="scene_ray",
            )


def _compare_sphere_hit(
    comparison: _Comparison,
    prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    baseline_hit = _sphere_hit_record(baseline)
    candidate_hit = _sphere_hit_record(candidate)
    if baseline_hit.get("valid") is not True or candidate_hit.get("valid") is not True:
        return
    hit_path = f"{prefix}.sphere_hit"
    for component in ("theta_radians", "phi_radians"):
        comparison.numeric(
            f"{hit_path}.{component}",
            baseline_hit.get(component),
            candidate_hit.get(component),
            tolerance=comparison.tolerances.sphere_hit_angle_radians,
            max_delta_kind="sphere_hit_angle",
        )
    comparison.exact(
        f"{hit_path}.hemisphere",
        baseline_hit.get("hemisphere"),
        candidate_hit.get("hemisphere"),
    )


def _compare_summary_counts(
    comparison: _Comparison,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> None:
    for field_name in (
        "decoded_frames",
        "frame_records",
        "scene_frame_records",
        "raw_frames",
        "processed_frames",
        "crop_files",
    ):
        comparison.exact(
            f"qa_summary.counts.{field_name}",
            _get_path(baseline_summary, ("counts", field_name)),
            _get_path(candidate_summary, ("counts", field_name)),
        )
    for field_name in ("counts_match", "schema_validation_passed"):
        comparison.exact(
            f"qa_summary.artifact_validation.{field_name}",
            _get_path(baseline_summary, ("artifact_validation", field_name)),
            _get_path(candidate_summary, ("artifact_validation", field_name)),
        )


def _compare_viewer_counts(
    comparison: _Comparison,
    baseline_viewer: dict[str, Any],
    candidate_viewer: dict[str, Any],
) -> None:
    comparison.exact(
        "viewer.frame_count",
        baseline_viewer.get("frame_count"),
        candidate_viewer.get("frame_count"),
    )


def _sphere_hit_record(frame: dict[str, Any]) -> dict[str, Any]:
    sphere_hit = frame.get("sphere_hit")
    if isinstance(sphere_hit, dict):
        return sphere_hit
    return {}


def _error_codes(errors: Any) -> Any:
    if not isinstance(errors, list):
        return errors
    codes: list[Any] = []
    for error in errors:
        if isinstance(error, dict):
            codes.append(error.get("code"))
        else:
            codes.append(error)
    return codes


def _get_path(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _finite_number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return repr(value)


def _format_float(value: float) -> str:
    return f"{value:.12g}"
