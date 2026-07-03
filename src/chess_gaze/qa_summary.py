from __future__ import annotations

import json
import mmap
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.errors import CliErrorCode, ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    CalibrationRecord,
    CropImageRetentionPolicy,
    FrameErrorRecord,
    FrameImageRetentionPolicy,
    FrameRecord,
    RunManifest,
    VideoManifest,
    read_run_manifest_artifact_json,
)
from chess_gaze.geometry import StrictSchemaModel
from chess_gaze.image_io import atomic_write_bytes
from chess_gaze.scene_records import (
    SceneAssumptionRecord,
    SceneAxisBasisRecord,
    SceneFrameRecord,
    SceneGazeSphereRecord,
    SceneManifest,
    SceneSummary,
)

QA_SUMMARY_BYTE_COUNT_STABILIZATION_ATTEMPTS = 5
QA_SAMPLE_COUNT = 30
REPRESENTATIVE_FAILURE_COUNT = 20
QUALITY_FAILURE_COUNT = 20
SOURCE_ARTIFACTS = {
    "run_manifest": "run_manifest.json",
    "calibration": "calibration.json",
    "video_manifest": "video_manifest.json",
    "frames_jsonl": "records/frames.jsonl",
    "errors_jsonl": "records/errors.jsonl",
    "scene_manifest": "scene/scene_manifest.json",
    "scene_summary": "scene/scene_summary.json",
    "scene_frames_jsonl": "records/scene_frames.jsonl",
    "viewer_index": "viewer/index.html",
    "viewer_scene_data": "viewer/scene-data.json",
    "raw_frames": "raw_frames",
    "processed_frames": "processed_frames",
    "crops": "crops",
}
ERROR_SEVERITY_CODES = frozenset(
    {
        ErrorCode.RAW_FRAME_WRITE_FAILED,
        ErrorCode.PROCESSED_FRAME_WRITE_FAILED,
        ErrorCode.SCHEMA_VALIDATION_FAILED,
    }
)


class ArtifactValidationError(RuntimeError):
    def __init__(self, code: CliErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class ArtifactCounts(StrictSchemaModel):
    decoded_frames: int
    frame_records: int
    scene_frame_records: int
    raw_frames: int
    processed_frames: int
    crop_files: int


class ByteCounts(StrictSchemaModel):
    raw_frames_bytes: int
    processed_frames_bytes: int
    crops_bytes: int
    jsonl_bytes: int
    scene_jsonl_bytes: int
    scene_bytes: int
    viewer_bytes: int
    total_run_bytes: int


class DetectionRates(StrictSchemaModel):
    face_present_rate: float
    both_eyes_present_rate: float
    left_eye_only_rate: float
    right_eye_only_rate: float
    left_iris_present_rate: float
    right_iris_present_rate: float
    head_pose_valid_rate: float
    face_gaze_valid_rate: float
    recommended_gaze_valid_rate: float


class DiskSpaceSummary(StrictSchemaModel):
    preflight_estimate_bytes: int
    closeout_free_bytes: int
    closeout_total_bytes: int


class ArtifactValidationResult(StrictSchemaModel):
    source_artifacts: dict[str, str]
    counts: ArtifactCounts
    byte_counts: ByteCounts
    schema_validation_passed: bool
    counts_match: bool
    validation_errors: list[str]
    final_status: Literal["complete", "failed"]


class QASummary(StrictSchemaModel):
    schema_version: Literal["qa-summary-v1"] = "qa-summary-v1"
    run_id: str
    source_video_path: str
    source_video_sha256: str
    source_artifacts: dict[str, str]
    counts: ArtifactCounts
    byte_counts: ByteCounts
    rates: DetectionRates
    errors_by_code: dict[str, int]
    errors_by_severity: dict[str, int]
    worst_blur_frame_ids: list[str]
    worst_exposure_frame_ids: list[str]
    qa_sample_frame_ids: list[str]
    representative_failure_frame_ids: list[str]
    status_transitions: list[str]
    final_status: Literal["complete", "failed"]
    disk_space: DiskSpaceSummary
    artifact_validation: ArtifactValidationResult
    built_from_disk_at_utc: str


@dataclass
class _FrameRecordSummary:
    count: int = 0
    frame_index_ids: list[tuple[int, str]] = field(default_factory=list)
    face_present_count: int = 0
    both_eyes_present_count: int = 0
    left_eye_only_count: int = 0
    right_eye_only_count: int = 0
    left_iris_present_count: int = 0
    right_iris_present_count: int = 0
    head_pose_valid_count: int = 0
    appearance_gaze_valid_count: int = 0
    recommended_gaze_valid_count: int = 0
    hard_failure_candidates: list[tuple[int, str]] = field(default_factory=list)

    @property
    def frame_indices(self) -> list[int]:
        return [frame_index for frame_index, _frame_id in self.frame_index_ids]


@dataclass
class _FrameErrorSummary:
    count: int = 0
    errors_by_code: Counter[str] = field(default_factory=Counter)
    errors_by_severity: Counter[str] = field(default_factory=Counter)
    hard_failure_candidates: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class _SceneFrameSummary:
    count: int = 0
    frame_indices: list[int] = field(default_factory=list)


class _ViewerSceneDataEnvelope(StrictSchemaModel):
    schema_version: Literal["gaze-scene-viewer-data-v2"]
    run_id: str
    source_video_stem: str
    frame_count: int
    frames_count: int
    valid_hit_points_count: int
    gaze_sphere: SceneGazeSphereRecord
    axis_basis: SceneAxisBasisRecord
    assumptions: list[SceneAssumptionRecord]
    summary: SceneSummary


@dataclass(frozen=True)
class _LoadedRunArtifacts:
    run_manifest: RunManifest
    video_manifest: VideoManifest
    frame_records: _FrameRecordSummary
    error_records: _FrameErrorSummary
    scene_manifest: SceneManifest | None
    scene_summary: SceneSummary | None
    scene_frame_records: _SceneFrameSummary
    raw_frame_paths: list[Path]
    processed_frame_paths: list[Path]
    crop_paths: list[Path]
    scene_paths: list[Path]
    viewer_paths: list[Path]
    schema_validation_errors: list[str]


def validate_run_artifacts(run_layout: RunLayout) -> ArtifactValidationResult:
    loaded = _load_run_artifacts(run_layout)
    return _validate_loaded_run_artifacts(run_layout, loaded)


def build_qa_summary(run_layout: RunLayout) -> QASummary:
    loaded = _load_run_artifacts(run_layout)
    artifact_validation = _validate_loaded_run_artifacts(run_layout, loaded)
    disk_space = _disk_space_summary(run_layout, artifact_validation.byte_counts)
    final_status = artifact_validation.final_status

    return QASummary(
        run_id=loaded.run_manifest.run_id,
        source_video_path=loaded.run_manifest.input_path,
        source_video_sha256=loaded.video_manifest.source_sha256,
        source_artifacts=artifact_validation.source_artifacts,
        counts=artifact_validation.counts,
        byte_counts=artifact_validation.byte_counts,
        rates=_detection_rates(loaded.frame_records),
        errors_by_code=_errors_by_code(loaded.error_records),
        errors_by_severity=_errors_by_severity(loaded.error_records),
        worst_blur_frame_ids=_worst_blur_frame_ids(loaded.raw_frame_paths),
        worst_exposure_frame_ids=_worst_exposure_frame_ids(loaded.raw_frame_paths),
        qa_sample_frame_ids=_qa_sample_frame_ids(loaded.frame_records),
        representative_failure_frame_ids=_representative_failure_frame_ids(
            loaded.frame_records, loaded.error_records
        ),
        status_transitions=_status_transitions(final_status),
        final_status=final_status,
        disk_space=disk_space,
        artifact_validation=artifact_validation,
        built_from_disk_at_utc=_format_utc(datetime.now(UTC)),
    )


def write_qa_summary(
    run_layout: RunLayout,
    qa_summary_path: Path,
    *,
    qa_summary: QASummary | None = None,
) -> QASummary:
    if qa_summary is None:
        qa_summary = build_qa_summary(run_layout)
    qa_summary = _stabilize_qa_summary_byte_count(
        run_layout, qa_summary, qa_summary_path
    )
    _write_json(qa_summary_path, qa_summary.model_dump(mode="json"))
    return qa_summary


def _stabilize_qa_summary_byte_count(
    run_layout: RunLayout, qa_summary: QASummary, qa_summary_path: Path
) -> QASummary:
    existing_qa_summary_bytes = (
        qa_summary_path.stat().st_size if qa_summary_path.exists() else 0
    )
    run_bytes_without_qa_summary = (
        _total_file_size(_artifact_files(run_layout.run_dir))
        - existing_qa_summary_bytes
    )
    stable_total_run_bytes: int | None = None

    for _attempt in range(QA_SUMMARY_BYTE_COUNT_STABILIZATION_ATTEMPTS):
        candidate_total_run_bytes = run_bytes_without_qa_summary + len(
            _json_bytes(qa_summary.model_dump(mode="json"))
        )
        qa_summary = _qa_summary_with_total_run_bytes(
            qa_summary,
            candidate_total_run_bytes,
        )
        if candidate_total_run_bytes == stable_total_run_bytes:
            return qa_summary
        stable_total_run_bytes = candidate_total_run_bytes

    return qa_summary


def _qa_summary_with_total_run_bytes(
    qa_summary: QASummary, total_run_bytes: int
) -> QASummary:
    byte_counts = qa_summary.byte_counts.model_copy(
        update={"total_run_bytes": total_run_bytes}
    )
    artifact_validation = qa_summary.artifact_validation.model_copy(
        update={"byte_counts": byte_counts}
    )
    disk_space = qa_summary.disk_space.model_copy(
        update={"preflight_estimate_bytes": max(total_run_bytes, 1)}
    )
    return qa_summary.model_copy(
        update={
            "byte_counts": byte_counts,
            "artifact_validation": artifact_validation,
            "disk_space": disk_space,
        }
    )


def _write_json(path: Path, payload: object) -> None:
    atomic_write_bytes(path, _json_bytes(payload))


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _load_run_artifacts(run_layout: RunLayout) -> _LoadedRunArtifacts:
    run_manifest = _read_run_manifest(run_layout.run_dir / "run_manifest.json")
    _read_json_model(run_layout.run_dir / "calibration.json", CalibrationRecord)
    video_manifest = _read_json_model(
        run_layout.run_dir / "video_manifest.json", VideoManifest
    )
    frame_records, frame_record_errors = _read_frame_record_summary(
        run_layout.records_dir / "frames.jsonl"
    )
    error_records, frame_error_record_errors = _read_frame_error_summary(
        run_layout.records_dir / "errors.jsonl"
    )
    scene_manifest, scene_manifest_errors = _read_optional_json_model(
        run_layout.scene_dir / "scene_manifest.json",
        SceneManifest,
        "scene manifest",
    )
    scene_summary, scene_summary_errors = _read_optional_json_model(
        run_layout.scene_dir / "scene_summary.json",
        SceneSummary,
        "scene summary",
    )
    scene_frame_records, scene_frame_record_errors = _read_scene_frame_summary(
        run_layout.records_dir / "scene_frames.jsonl"
    )
    viewer_scene_data_errors = _viewer_scene_data_validation_errors(
        run_layout.viewer_dir / "scene-data.json",
        run_manifest=run_manifest,
        video_manifest=video_manifest,
        scene_summary=scene_summary,
    )
    viewer_index_errors = _required_file_errors(
        run_layout.viewer_dir / "index.html", "viewer index"
    )
    return _LoadedRunArtifacts(
        run_manifest=run_manifest,
        video_manifest=video_manifest,
        frame_records=frame_records,
        error_records=error_records,
        scene_manifest=scene_manifest,
        scene_summary=scene_summary,
        scene_frame_records=scene_frame_records,
        raw_frame_paths=_artifact_files(run_layout.raw_frames_dir),
        processed_frame_paths=_artifact_files(run_layout.processed_frames_dir),
        crop_paths=_artifact_files(run_layout.crops_dir),
        scene_paths=_artifact_files(run_layout.scene_dir),
        viewer_paths=_artifact_files(run_layout.viewer_dir),
        schema_validation_errors=(
            frame_record_errors
            + frame_error_record_errors
            + scene_manifest_errors
            + scene_summary_errors
            + scene_frame_record_errors
            + viewer_scene_data_errors
            + viewer_index_errors
        ),
    )


def _read_run_manifest(path: Path) -> RunManifest:
    try:
        return read_run_manifest_artifact_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            f"Invalid JSON artifact at {path}: {exc}",
        ) from exc


def _validate_loaded_run_artifacts(
    run_layout: RunLayout, loaded: _LoadedRunArtifacts
) -> ArtifactValidationResult:
    counts = _artifact_counts(loaded)
    byte_counts = _byte_counts(run_layout, loaded)
    count_validation_errors = _count_validation_errors(
        counts,
        loaded.frame_records,
        loaded.scene_frame_records,
        loaded.run_manifest.frame_image_retention,
        loaded.run_manifest.crop_image_retention,
    )
    validation_errors = loaded.schema_validation_errors + count_validation_errors
    final_status: Literal["complete", "failed"] = (
        "complete" if not validation_errors else "failed"
    )
    return ArtifactValidationResult(
        source_artifacts=dict(SOURCE_ARTIFACTS),
        counts=counts,
        byte_counts=byte_counts,
        schema_validation_passed=not loaded.schema_validation_errors,
        counts_match=not count_validation_errors,
        validation_errors=validation_errors,
        final_status=final_status,
    )


def _read_json_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            f"Invalid JSON artifact at {path}: {exc}",
        ) from exc


def _read_optional_json_model[ModelT: BaseModel](
    path: Path, model_type: type[ModelT], artifact_name: str
) -> tuple[ModelT | None, list[str]]:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8")), []
    except (OSError, ValueError) as exc:
        return None, [
            (
                f"{CliErrorCode.SCHEMA_VALIDATION_FAILED.value}: "
                f"Invalid {artifact_name} at {path}: {exc}"
            )
        ]


def _read_frame_record_summary(
    path: Path,
) -> tuple[_FrameRecordSummary, list[str]]:
    summary = _FrameRecordSummary()
    validation_errors: list[str] = []
    for line_number, line in _iter_jsonl_lines(path, "frame", validation_errors):
        try:
            record = FrameRecord.model_validate_json(line)
        except ValueError as exc:
            validation_errors.append(
                f"{CliErrorCode.SCHEMA_VALIDATION_FAILED.value}: "
                f"Invalid frame record at {path}:{line_number}: {exc}"
            )
            continue
        _accumulate_frame_record(summary, record)
    return summary, validation_errors


def _accumulate_frame_record(summary: _FrameRecordSummary, record: FrameRecord) -> None:
    summary.count += 1
    summary.frame_index_ids.append((record.frame_index, record.frame_id))
    if record.face.present:
        summary.face_present_count += 1
    if record.left_eye.present and record.right_eye.present:
        summary.both_eyes_present_count += 1
    if record.left_eye.present and not record.right_eye.present:
        summary.left_eye_only_count += 1
    if record.right_eye.present and not record.left_eye.present:
        summary.right_eye_only_count += 1
    if record.left_eye.iris_landmarks:
        summary.left_iris_present_count += 1
    if record.right_eye.iris_landmarks:
        summary.right_iris_present_count += 1
    if record.head_pose.valid:
        summary.head_pose_valid_count += 1
    if record.appearance_gaze.valid:
        summary.appearance_gaze_valid_count += 1
    if record.recommended_gaze.valid:
        summary.recommended_gaze_valid_count += 1
    if record.status is FrameStatus.ERROR:
        summary.hard_failure_candidates.append((record.frame_index, record.frame_id))


def _read_frame_error_summary(
    path: Path,
) -> tuple[_FrameErrorSummary, list[str]]:
    summary = _FrameErrorSummary()
    validation_errors: list[str] = []
    for line_number, line in _iter_jsonl_lines(path, "frame error", validation_errors):
        try:
            record = FrameErrorRecord.model_validate_json(line)
        except ValueError as exc:
            validation_errors.append(
                f"{CliErrorCode.SCHEMA_VALIDATION_FAILED.value}: "
                f"Invalid frame error record at {path}:{line_number}: {exc}"
            )
            continue
        summary.count += 1
        code = record.code.value
        severity = _severity(record.code)
        summary.errors_by_code[code] += 1
        summary.errors_by_severity[severity] += 1
        if severity == "error":
            summary.hard_failure_candidates.append(
                (record.frame_index, record.frame_id)
            )
    return summary, validation_errors


def _read_scene_frame_summary(
    path: Path,
) -> tuple[_SceneFrameSummary, list[str]]:
    summary = _SceneFrameSummary()
    validation_errors: list[str] = []
    for line_number, line in _iter_jsonl_lines(path, "scene frame", validation_errors):
        try:
            record = SceneFrameRecord.model_validate_json(line)
        except ValueError as exc:
            validation_errors.append(
                f"{CliErrorCode.SCHEMA_VALIDATION_FAILED.value}: "
                f"Invalid scene frame record at {path}:{line_number}: {exc}"
            )
            continue
        summary.count += 1
        summary.frame_indices.append(record.frame_index)
    return summary, validation_errors


def _iter_jsonl_lines(
    path: Path, record_name: str, validation_errors: list[str]
) -> Iterator[tuple[int, str]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield line_number, line
    except (OSError, UnicodeDecodeError) as exc:
        validation_errors.append(
            f"{CliErrorCode.SCHEMA_VALIDATION_FAILED.value}: "
            f"Unable to read {record_name} JSONL artifact at {path}: {exc}"
        )


def _viewer_scene_data_validation_errors(
    path: Path,
    *,
    run_manifest: RunManifest,
    video_manifest: VideoManifest,
    scene_summary: SceneSummary | None,
) -> list[str]:
    try:
        envelope = _read_viewer_scene_data_envelope(path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [
            (
                f"{CliErrorCode.SCHEMA_VALIDATION_FAILED.value}: "
                f"Invalid viewer scene data at {path}: {exc}"
            )
        ]

    errors: list[str] = []
    expected_source_video_stem = Path(video_manifest.source_path).stem
    if envelope.run_id != run_manifest.run_id:
        errors.append(
            "viewer scene data run_id does not match run manifest: "
            f"{envelope.run_id} != {run_manifest.run_id}"
        )
    if envelope.source_video_stem != expected_source_video_stem:
        errors.append(
            "viewer scene data source video stem does not match video manifest: "
            f"{envelope.source_video_stem} != {expected_source_video_stem}"
        )
    if envelope.frame_count != video_manifest.frame_count_decoded:
        errors.append(
            "viewer scene data frame_count does not match decoded frame count: "
            f"{envelope.frame_count} != {video_manifest.frame_count_decoded}"
        )
    if envelope.frames_count != video_manifest.frame_count_decoded:
        errors.append(
            "viewer scene data frames count does not match decoded frame count: "
            f"{envelope.frames_count} != {video_manifest.frame_count_decoded}"
        )
    if scene_summary is not None:
        if envelope.summary != scene_summary:
            errors.append("viewer scene data summary does not match scene summary")
        if envelope.valid_hit_points_count != scene_summary.valid_sphere_hit_frames:
            errors.append(
                "viewer scene data valid hit point count does not match scene "
                "summary: "
                f"{envelope.valid_hit_points_count} != "
                f"{scene_summary.valid_sphere_hit_frames}"
            )
    if errors:
        return [
            (
                f"{CliErrorCode.SCHEMA_VALIDATION_FAILED.value}: "
                f"Invalid viewer scene data at {path}: {error}"
            )
            for error in errors
        ]
    return []


def _read_viewer_scene_data_envelope(path: Path) -> _ViewerSceneDataEnvelope:
    with path.open("rb") as handle:
        if path.stat().st_size == 0:
            raise ValueError("empty JSON document")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            payload = _scan_viewer_scene_data_payload(data)
    return _ViewerSceneDataEnvelope.model_validate(payload)


def _scan_viewer_scene_data_payload(data: mmap.mmap) -> dict[str, object]:
    envelope_keys = {
        "schema_version",
        "run_id",
        "source_video_stem",
        "frame_count",
        "gaze_sphere",
        "axis_basis",
        "assumptions",
        "summary",
    }
    payload: dict[str, object] = {}
    array_counts: dict[str, int] = {}
    index = _skip_json_whitespace(data, 0)
    index = _expect_json_byte(data, index, ord("{"))
    index = _skip_json_whitespace(data, index)
    if _json_byte(data, index) == ord("}"):
        raise ValueError("viewer scene data object is empty")

    while True:
        key, index = _parse_json_string(data, index)
        index = _skip_json_whitespace(data, index)
        index = _expect_json_byte(data, index, ord(":"))
        index = _skip_json_whitespace(data, index)

        if key in {"frames", "valid_hit_points"}:
            if key in array_counts:
                raise ValueError(f"duplicate top-level key: {key}")
            count, index = _count_and_skip_json_array(data, index)
            array_counts[key] = count
        elif key in envelope_keys:
            if key in payload:
                raise ValueError(f"duplicate top-level key: {key}")
            value_end = _skip_json_value(data, index)
            value = json.loads(bytes(data[index:value_end]).decode("utf-8"))
            if key == "assumptions":
                value = _coerce_assumption_triplets(value)
            payload[key] = value
            index = value_end
        else:
            index = _skip_json_value(data, index)

        index = _skip_json_whitespace(data, index)
        byte = _json_byte(data, index)
        if byte == ord(","):
            index = _skip_json_whitespace(data, index + 1)
            continue
        if byte == ord("}"):
            index = _skip_json_whitespace(data, index + 1)
            if index != len(data):
                raise ValueError("unexpected data after viewer scene data object")
            break
        raise ValueError("expected ',' or '}' in viewer scene data object")

    missing = (
        envelope_keys
        | {
            "frames",
            "valid_hit_points",
        }
    ) - (set(payload) | set(array_counts))
    if missing:
        raise ValueError(f"missing top-level keys: {', '.join(sorted(missing))}")
    payload["frames_count"] = array_counts["frames"]
    payload["valid_hit_points_count"] = array_counts["valid_hit_points"]
    return payload


def _coerce_assumption_triplets(value: object) -> object:
    if not isinstance(value, list):
        return value
    coerced: list[object] = []
    for item in value:
        if not isinstance(item, dict):
            coerced.append(item)
            continue
        record = dict(item)
        record_value = record.get("value")
        if isinstance(record_value, list) and len(record_value) == 3:
            record["value"] = tuple(record_value)
        coerced.append(record)
    return coerced


def _skip_json_whitespace(data: mmap.mmap, index: int) -> int:
    while index < len(data) and _json_byte(data, index) in b" \t\r\n":
        index += 1
    return index


def _expect_json_byte(data: mmap.mmap, index: int, expected: int) -> int:
    if _json_byte(data, index) != expected:
        raise ValueError(f"expected {chr(expected)!r} at byte {index}")
    return index + 1


def _json_byte(data: mmap.mmap, index: int) -> int:
    if index >= len(data):
        raise ValueError("unexpected end of JSON document")
    return data[index]


def _parse_json_string(data: mmap.mmap, index: int) -> tuple[str, int]:
    if _json_byte(data, index) != ord('"'):
        raise ValueError(f"expected string at byte {index}")
    cursor = index + 1
    escaped = False
    while cursor < len(data):
        byte = _json_byte(data, cursor)
        if escaped:
            escaped = False
            cursor += 1
            continue
        if byte == ord("\\"):
            escaped = True
            cursor += 1
            continue
        if byte == ord('"'):
            raw = bytes(data[index : cursor + 1]).decode("utf-8")
            return str(json.loads(raw)), cursor + 1
        cursor += 1
    raise ValueError("unterminated string")


def _count_and_skip_json_array(data: mmap.mmap, index: int) -> tuple[int, int]:
    index = _expect_json_byte(data, index, ord("["))
    index = _skip_json_whitespace(data, index)
    if _json_byte(data, index) == ord("]"):
        return 0, index + 1

    count = 0
    while True:
        index = _skip_json_value(data, index)
        count += 1
        index = _skip_json_whitespace(data, index)
        byte = _json_byte(data, index)
        if byte == ord(","):
            index = _skip_json_whitespace(data, index + 1)
            continue
        if byte == ord("]"):
            return count, index + 1
        raise ValueError("expected ',' or ']' in viewer scene data array")


def _skip_json_value(data: mmap.mmap, index: int) -> int:
    index = _skip_json_whitespace(data, index)
    byte = _json_byte(data, index)
    if byte == ord('"'):
        _value, end = _parse_json_string(data, index)
        return end
    if byte in {ord("{"), ord("[")}:
        return _skip_json_container(data, index)

    cursor = index
    while cursor < len(data) and _json_byte(data, cursor) not in b",]} \t\r\n":
        cursor += 1
    if cursor == index:
        raise ValueError(f"expected JSON value at byte {index}")
    json.loads(bytes(data[index:cursor]).decode("utf-8"))
    return cursor


def _skip_json_container(data: mmap.mmap, index: int) -> int:
    opening = _json_byte(data, index)
    if opening == ord("{"):
        stack = [ord("}")]
    elif opening == ord("["):
        stack = [ord("]")]
    else:
        raise ValueError(f"expected JSON container at byte {index}")

    cursor = index + 1
    in_string = False
    escaped = False
    while cursor < len(data):
        byte = _json_byte(data, cursor)
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            cursor += 1
            continue

        if byte == ord('"'):
            in_string = True
        elif byte == ord("{"):
            stack.append(ord("}"))
        elif byte == ord("["):
            stack.append(ord("]"))
        elif byte in {ord("}"), ord("]")}:
            expected = stack.pop()
            if byte != expected:
                raise ValueError(f"mismatched JSON delimiter at byte {cursor}")
            if not stack:
                return cursor + 1
        cursor += 1
    raise ValueError("unterminated JSON container")


def _required_file_errors(path: Path, artifact_name: str) -> list[str]:
    if path.is_file():
        return []
    return [
        (
            f"{CliErrorCode.SCHEMA_VALIDATION_FAILED.value}: "
            f"Missing {artifact_name} artifact at {path}"
        )
    ]


def _artifact_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _artifact_counts(loaded: _LoadedRunArtifacts) -> ArtifactCounts:
    return ArtifactCounts(
        decoded_frames=loaded.video_manifest.frame_count_decoded,
        frame_records=loaded.frame_records.count,
        scene_frame_records=loaded.scene_frame_records.count,
        raw_frames=len(loaded.raw_frame_paths),
        processed_frames=len(loaded.processed_frame_paths),
        crop_files=len(loaded.crop_paths),
    )


def _byte_counts(run_layout: RunLayout, loaded: _LoadedRunArtifacts) -> ByteCounts:
    frames_jsonl_path = run_layout.records_dir / "frames.jsonl"
    errors_jsonl_path = run_layout.records_dir / "errors.jsonl"
    scene_frames_jsonl_path = run_layout.records_dir / "scene_frames.jsonl"
    return ByteCounts(
        raw_frames_bytes=_total_file_size(loaded.raw_frame_paths),
        processed_frames_bytes=_total_file_size(loaded.processed_frame_paths),
        crops_bytes=_total_file_size(loaded.crop_paths),
        jsonl_bytes=_total_file_size([frames_jsonl_path, errors_jsonl_path]),
        scene_jsonl_bytes=_total_file_size([scene_frames_jsonl_path]),
        scene_bytes=_total_file_size(loaded.scene_paths),
        viewer_bytes=_total_file_size(loaded.viewer_paths),
        total_run_bytes=_total_file_size(_artifact_files(run_layout.run_dir)),
    )


def _total_file_size(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())


def _count_validation_errors(
    counts: ArtifactCounts,
    frame_records: _FrameRecordSummary,
    scene_frame_records: _SceneFrameSummary,
    frame_image_retention: FrameImageRetentionPolicy,
    crop_image_retention: CropImageRetentionPolicy,
) -> list[str]:
    errors: list[str] = []
    if counts.frame_records != counts.decoded_frames:
        errors.append(
            "frame record count does not match decoded frame count: "
            f"{counts.frame_records} != {counts.decoded_frames}"
        )
    expected_frame_images = (
        counts.decoded_frames if frame_image_retention.save_frame_images else 0
    )
    errors.extend(
        _frame_image_count_validation_errors(
            raw_frames=counts.raw_frames,
            processed_frames=counts.processed_frames,
            expected_frame_images=expected_frame_images,
            save_frame_images=frame_image_retention.save_frame_images,
        )
    )
    if not crop_image_retention.save_crop_images and counts.crop_files != 0:
        errors.append(
            "crop file count does not match crop image retention policy: "
            f"{counts.crop_files} != 0"
        )
    if counts.scene_frame_records != counts.decoded_frames:
        errors.append(
            "scene frame record count does not match decoded frame count: "
            f"{counts.scene_frame_records} != {counts.decoded_frames}"
        )

    observed_indices = sorted(frame_records.frame_indices)
    if observed_indices != list(range(counts.decoded_frames)):
        errors.append("frame records are not contiguous from decoded frame zero")
    observed_scene_indices = sorted(scene_frame_records.frame_indices)
    if observed_scene_indices != list(range(counts.decoded_frames)):
        errors.append("scene frame records are not contiguous from decoded frame zero")
    return errors


def _frame_image_count_validation_errors(
    *,
    raw_frames: int,
    processed_frames: int,
    expected_frame_images: int,
    save_frame_images: bool,
) -> list[str]:
    errors: list[str] = []
    if raw_frames != expected_frame_images:
        errors.append(
            _frame_image_count_error(
                "raw",
                observed=raw_frames,
                expected=expected_frame_images,
                save_frame_images=save_frame_images,
            )
        )
    if processed_frames != expected_frame_images:
        errors.append(
            _frame_image_count_error(
                "processed",
                observed=processed_frames,
                expected=expected_frame_images,
                save_frame_images=save_frame_images,
            )
        )
    return errors


def _frame_image_count_error(
    frame_kind: str,
    *,
    observed: int,
    expected: int,
    save_frame_images: bool,
) -> str:
    if save_frame_images:
        return (
            f"{frame_kind} frame count does not match decoded frame count: "
            f"{observed} != {expected}"
        )
    return (
        f"{frame_kind} frame count does not match frame image retention policy: "
        f"{observed} != {expected}"
    )


def _detection_rates(frame_records: _FrameRecordSummary) -> DetectionRates:
    total = frame_records.count
    return DetectionRates(
        face_present_rate=_rate(total, frame_records.face_present_count),
        both_eyes_present_rate=_rate(
            total,
            frame_records.both_eyes_present_count,
        ),
        left_eye_only_rate=_rate(
            total,
            frame_records.left_eye_only_count,
        ),
        right_eye_only_rate=_rate(
            total,
            frame_records.right_eye_only_count,
        ),
        left_iris_present_rate=_rate(total, frame_records.left_iris_present_count),
        right_iris_present_rate=_rate(total, frame_records.right_iris_present_count),
        head_pose_valid_rate=_rate(total, frame_records.head_pose_valid_count),
        face_gaze_valid_rate=_rate(total, frame_records.appearance_gaze_valid_count),
        recommended_gaze_valid_rate=_rate(
            total, frame_records.recommended_gaze_valid_count
        ),
    )


def _rate(total: int, count: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _errors_by_code(error_records: _FrameErrorSummary) -> dict[str, int]:
    return dict(sorted(error_records.errors_by_code.items()))


def _errors_by_severity(error_records: _FrameErrorSummary) -> dict[str, int]:
    return dict(sorted(error_records.errors_by_severity.items()))


def _severity(code: ErrorCode) -> Literal["error", "warning"]:
    if code in ERROR_SEVERITY_CODES:
        return "error"
    return "warning"


def _worst_blur_frame_ids(raw_frame_paths: list[Path]) -> list[str]:
    scores = _quality_scores(raw_frame_paths)
    return [
        frame_id_value
        for frame_id_value, _blur_score, _exposure_error in sorted(
            scores, key=lambda score: (score[1], score[0])
        )[:QUALITY_FAILURE_COUNT]
    ]


def _worst_exposure_frame_ids(raw_frame_paths: list[Path]) -> list[str]:
    scores = _quality_scores(raw_frame_paths)
    return [
        frame_id_value
        for frame_id_value, _blur_score, _exposure_error in sorted(
            scores, key=lambda score: (-score[2], score[0])
        )[:QUALITY_FAILURE_COUNT]
    ]


def _quality_scores(raw_frame_paths: list[Path]) -> list[tuple[str, float, float]]:
    scores: list[tuple[str, float, float]] = []
    for path in raw_frame_paths:
        try:
            with Image.open(path) as image:
                gray = np.asarray(image.convert("L"), dtype=np.float32)
        except OSError:
            continue
        horizontal_gradient = np.diff(gray, axis=1)
        vertical_gradient = np.diff(gray, axis=0)
        blur_score = float(
            np.var(horizontal_gradient, dtype=np.float64)
            + np.var(vertical_gradient, dtype=np.float64)
        )
        exposure_error = abs(float(np.mean(gray, dtype=np.float64)) - 127.5)
        scores.append((path.stem, blur_score, exposure_error))
    return scores


def _qa_sample_frame_ids(frame_records: _FrameRecordSummary) -> list[str]:
    ordered_records = sorted(frame_records.frame_index_ids)
    if len(ordered_records) <= QA_SAMPLE_COUNT:
        return [frame_id_value for _frame_index, frame_id_value in ordered_records]

    last_index = len(ordered_records) - 1
    sample_indices = [
        round(sample_index * last_index / (QA_SAMPLE_COUNT - 1))
        for sample_index in range(QA_SAMPLE_COUNT)
    ]
    return [ordered_records[index][1] for index in sample_indices]


def _representative_failure_frame_ids(
    frame_records: _FrameRecordSummary, error_records: _FrameErrorSummary
) -> list[str]:
    failed_frames: dict[str, tuple[int, str]] = {}
    for frame_index, frame_id_value in frame_records.hard_failure_candidates:
        failed_frames[frame_id_value] = (frame_index, frame_id_value)
    for frame_index, frame_id_value in error_records.hard_failure_candidates:
        failed_frames.setdefault(frame_id_value, (frame_index, frame_id_value))

    return [
        frame_id_value
        for _frame_index, frame_id_value in sorted(failed_frames.values())[
            :REPRESENTATIVE_FAILURE_COUNT
        ]
    ]


def _status_transitions(final_status: Literal["complete", "failed"]) -> list[str]:
    return ["created", "processing", "revalidating", final_status]


def _disk_space_summary(
    run_layout: RunLayout, byte_counts: ByteCounts
) -> DiskSpaceSummary:
    disk_usage = shutil.disk_usage(_nearest_existing_parent(run_layout.run_dir))
    return DiskSpaceSummary(
        preflight_estimate_bytes=max(byte_counts.total_run_bytes, 1),
        closeout_free_bytes=disk_usage.free,
        closeout_total_bytes=disk_usage.total,
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate


def _format_utc(value: datetime) -> str:
    return (
        value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
