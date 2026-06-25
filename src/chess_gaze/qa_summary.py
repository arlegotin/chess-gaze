from __future__ import annotations

import shutil
from collections import Counter
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
    FrameErrorRecord,
    FrameRecord,
    RunManifest,
    VideoManifest,
)
from chess_gaze.geometry import StrictSchemaModel

QA_SAMPLE_COUNT = 30
REPRESENTATIVE_FAILURE_COUNT = 20
QUALITY_FAILURE_COUNT = 20
SOURCE_ARTIFACTS = {
    "run_manifest": "run_manifest.json",
    "calibration": "calibration.json",
    "video_manifest": "video_manifest.json",
    "frames_jsonl": "records/frames.jsonl",
    "errors_jsonl": "records/errors.jsonl",
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
    raw_frames: int
    processed_frames: int
    crop_files: int


class ByteCounts(StrictSchemaModel):
    raw_frames_bytes: int
    processed_frames_bytes: int
    crops_bytes: int
    jsonl_bytes: int
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


class _LoadedRunArtifacts(StrictSchemaModel):
    run_manifest: RunManifest
    video_manifest: VideoManifest
    frame_records: list[FrameRecord]
    error_records: list[FrameErrorRecord]
    raw_frame_paths: list[Path]
    processed_frame_paths: list[Path]
    crop_paths: list[Path]


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
        source_artifacts=dict(SOURCE_ARTIFACTS),
        counts=artifact_validation.counts,
        byte_counts=artifact_validation.byte_counts,
        rates=_detection_rates(loaded.frame_records),
        errors_by_code=_errors_by_code(loaded.error_records),
        errors_by_severity=_errors_by_severity(loaded.error_records),
        worst_blur_frame_ids=_worst_blur_frame_ids(loaded.raw_frame_paths),
        worst_exposure_frame_ids=_worst_exposure_frame_ids(loaded.raw_frame_paths),
        qa_sample_frame_ids=_qa_sample_frame_ids(loaded.frame_records),
        representative_failure_frame_ids=_representative_failure_frame_ids(
            loaded.frame_records
        ),
        status_transitions=_status_transitions(final_status),
        final_status=final_status,
        disk_space=disk_space,
        artifact_validation=artifact_validation,
        built_from_disk_at_utc=_format_utc(datetime.now(UTC)),
    )


def _load_run_artifacts(run_layout: RunLayout) -> _LoadedRunArtifacts:
    run_manifest = _read_json_model(
        run_layout.run_dir / "run_manifest.json", RunManifest
    )
    _read_json_model(run_layout.run_dir / "calibration.json", CalibrationRecord)
    video_manifest = _read_json_model(
        run_layout.run_dir / "video_manifest.json", VideoManifest
    )
    frame_records = _read_jsonl_model(
        run_layout.records_dir / "frames.jsonl", FrameRecord, "frame"
    )
    error_records = _read_jsonl_model(
        run_layout.records_dir / "errors.jsonl", FrameErrorRecord, "frame error"
    )
    return _LoadedRunArtifacts(
        run_manifest=run_manifest,
        video_manifest=video_manifest,
        frame_records=frame_records,
        error_records=error_records,
        raw_frame_paths=_artifact_files(run_layout.raw_frames_dir),
        processed_frame_paths=_artifact_files(run_layout.processed_frames_dir),
        crop_paths=_artifact_files(run_layout.crops_dir),
    )


def _validate_loaded_run_artifacts(
    run_layout: RunLayout, loaded: _LoadedRunArtifacts
) -> ArtifactValidationResult:
    counts = _artifact_counts(loaded)
    byte_counts = _byte_counts(run_layout, loaded)
    validation_errors = _count_validation_errors(counts, loaded.frame_records)
    final_status: Literal["complete", "failed"] = (
        "complete" if not validation_errors else "failed"
    )
    return ArtifactValidationResult(
        source_artifacts=dict(SOURCE_ARTIFACTS),
        counts=counts,
        byte_counts=byte_counts,
        schema_validation_passed=True,
        counts_match=not validation_errors,
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


def _read_jsonl_model[ModelT: BaseModel](
    path: Path, model_type: type[ModelT], record_name: str
) -> list[ModelT]:
    records: list[ModelT] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactValidationError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            f"Unable to read {record_name} JSONL artifact at {path}: {exc}",
        ) from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(model_type.model_validate_json(line))
        except ValueError as exc:
            raise ArtifactValidationError(
                CliErrorCode.SCHEMA_VALIDATION_FAILED,
                f"Invalid {record_name} record at {path}:{line_number}: {exc}",
            ) from exc
    return records


def _artifact_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _artifact_counts(loaded: _LoadedRunArtifacts) -> ArtifactCounts:
    decoded_frame_count = _decoded_frame_count(loaded.frame_records)
    return ArtifactCounts(
        decoded_frames=decoded_frame_count,
        frame_records=len(loaded.frame_records),
        raw_frames=len(loaded.raw_frame_paths),
        processed_frames=len(loaded.processed_frame_paths),
        crop_files=len(loaded.crop_paths),
    )


def _decoded_frame_count(frame_records: list[FrameRecord]) -> int:
    if not frame_records:
        return 0
    return max(record.frame_index for record in frame_records) + 1


def _byte_counts(run_layout: RunLayout, loaded: _LoadedRunArtifacts) -> ByteCounts:
    frames_jsonl_path = run_layout.records_dir / "frames.jsonl"
    errors_jsonl_path = run_layout.records_dir / "errors.jsonl"
    return ByteCounts(
        raw_frames_bytes=_total_file_size(loaded.raw_frame_paths),
        processed_frames_bytes=_total_file_size(loaded.processed_frame_paths),
        crops_bytes=_total_file_size(loaded.crop_paths),
        jsonl_bytes=_total_file_size([frames_jsonl_path, errors_jsonl_path]),
        total_run_bytes=_total_file_size(_artifact_files(run_layout.run_dir)),
    )


def _total_file_size(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())


def _count_validation_errors(
    counts: ArtifactCounts, frame_records: list[FrameRecord]
) -> list[str]:
    errors: list[str] = []
    if counts.frame_records != counts.decoded_frames:
        errors.append(
            "frame record count does not match decoded frame count: "
            f"{counts.frame_records} != {counts.decoded_frames}"
        )
    if counts.raw_frames != counts.decoded_frames:
        errors.append(
            "raw frame count does not match decoded frame count: "
            f"{counts.raw_frames} != {counts.decoded_frames}"
        )
    if counts.processed_frames != counts.decoded_frames:
        errors.append(
            "processed frame count does not match decoded frame count: "
            f"{counts.processed_frames} != {counts.decoded_frames}"
        )

    observed_indices = sorted(record.frame_index for record in frame_records)
    if observed_indices != list(range(counts.decoded_frames)):
        errors.append("frame records are not contiguous from decoded frame zero")
    return errors


def _detection_rates(frame_records: list[FrameRecord]) -> DetectionRates:
    total = len(frame_records)
    return DetectionRates(
        face_present_rate=_rate(
            total, sum(record.face.present for record in frame_records)
        ),
        both_eyes_present_rate=_rate(
            total,
            sum(
                record.left_eye.present and record.right_eye.present
                for record in frame_records
            ),
        ),
        left_eye_only_rate=_rate(
            total,
            sum(
                record.left_eye.present and not record.right_eye.present
                for record in frame_records
            ),
        ),
        right_eye_only_rate=_rate(
            total,
            sum(
                record.right_eye.present and not record.left_eye.present
                for record in frame_records
            ),
        ),
        left_iris_present_rate=_rate(
            total,
            sum(bool(record.left_eye.iris_landmarks) for record in frame_records),
        ),
        right_iris_present_rate=_rate(
            total,
            sum(bool(record.right_eye.iris_landmarks) for record in frame_records),
        ),
        head_pose_valid_rate=_rate(
            total, sum(record.head_pose.valid for record in frame_records)
        ),
        face_gaze_valid_rate=_rate(
            total, sum(record.appearance_gaze.valid for record in frame_records)
        ),
        recommended_gaze_valid_rate=_rate(
            total, sum(record.recommended_gaze.valid for record in frame_records)
        ),
    )


def _rate(total: int, count: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _errors_by_code(error_records: list[FrameErrorRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.code.value for record in error_records).items()))


def _errors_by_severity(error_records: list[FrameErrorRecord]) -> dict[str, int]:
    return dict(
        sorted(Counter(_severity(record.code) for record in error_records).items())
    )


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


def _qa_sample_frame_ids(frame_records: list[FrameRecord]) -> list[str]:
    ordered_records = sorted(frame_records, key=lambda record: record.frame_index)
    if len(ordered_records) <= QA_SAMPLE_COUNT:
        return [record.frame_id for record in ordered_records]

    last_index = len(ordered_records) - 1
    sample_indices = [
        round(sample_index * last_index / (QA_SAMPLE_COUNT - 1))
        for sample_index in range(QA_SAMPLE_COUNT)
    ]
    return [ordered_records[index].frame_id for index in sample_indices]


def _representative_failure_frame_ids(frame_records: list[FrameRecord]) -> list[str]:
    failed_records = [
        record
        for record in frame_records
        if record.status is not FrameStatus.OK or record.errors
    ]
    return [
        record.frame_id
        for record in sorted(
            failed_records, key=lambda record: (record.frame_index, record.frame_id)
        )[:REPRESENTATIVE_FAILURE_COUNT]
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
