from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TextIO

from pydantic import ValidationError

from chess_gaze.artifact_runs import RunLayout, frame_id, run_layout_from_dir
from chess_gaze.frame_records import (
    CalibrationRecord,
    CropImageRetentionPolicy,
    ErrorRecord,
    FrameErrorRecord,
    FrameImageRetentionPolicy,
    FrameRecord,
    InferenceRuntimeRecord,
    QASummaryPolicy,
    RunManifest,
    VideoManifest,
    read_run_manifest_artifact_json,
)
from chess_gaze.geometry import StrictSchemaModel
from chess_gaze.image_io import atomic_write_bytes
from chess_gaze.qa_summary import QASummary


@dataclass(frozen=True)
class ResumePreparation:
    committed_records: tuple[FrameRecord, ...]
    next_frame_index: int
    analysis_state: AnalysisState


class AnalysisState(StrictSchemaModel):
    schema_version: Literal["analysis-state-v1"] = "analysis-state-v1"
    run_id: str
    input_path: str
    source_video_sha256: str
    frame_count_decoded: int
    next_frame_index: int
    status: Literal["processing", "revalidating", "complete", "failed"]
    updated_at_utc: str


def find_latest_resumable_run(
    runs_root: Path,
    video_path: Path,
    video_manifest: VideoManifest,
    calibration: CalibrationRecord,
    inference: InferenceRuntimeRecord,
    frame_image_retention: FrameImageRetentionPolicy,
    crop_image_retention: CropImageRetentionPolicy,
    qa_summary_policy: QASummaryPolicy,
) -> RunLayout | None:
    if not runs_root.exists():
        return None

    expected_video_path = str(video_path)
    for run_dir in sorted(
        (
            path
            for path in runs_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: path.name,
        reverse=True,
    ):
        if not _run_matches(
            run_dir,
            expected_video_path=expected_video_path,
            video_manifest=video_manifest,
            calibration=calibration,
            inference=inference,
            frame_image_retention=frame_image_retention,
            crop_image_retention=crop_image_retention,
            expected_qa_summary_policy=qa_summary_policy,
        ):
            continue
        if _run_is_complete(run_dir):
            continue
        layout = run_layout_from_dir(run_dir)
        try:
            validate_resume_cleanup_paths(layout)
        except ValueError:
            continue
        return layout
    return None


def prepare_resume_run(
    layout: RunLayout,
    video_manifest: VideoManifest,
    *,
    clock: Callable[[], datetime],
) -> ResumePreparation:
    validate_resume_cleanup_paths(layout)
    committed_records = _committed_frame_records(
        layout.records_dir / "frames.jsonl",
        frame_count_decoded=video_manifest.frame_count_decoded,
    )
    next_frame_index = len(committed_records)
    _rewrite_frames_jsonl(layout.records_dir / "frames.jsonl", committed_records)
    _rewrite_errors_jsonl(layout.records_dir / "errors.jsonl", committed_records)
    _delete_uncommitted_frame_artifacts(layout, next_frame_index=next_frame_index)
    _delete_derived_artifacts(layout)
    analysis_state = AnalysisState(
        run_id=layout.run_dir.name,
        input_path=video_manifest.source_path,
        source_video_sha256=video_manifest.source_sha256,
        frame_count_decoded=video_manifest.frame_count_decoded,
        next_frame_index=next_frame_index,
        status="processing",
        updated_at_utc=_format_utc(clock()),
    )
    return ResumePreparation(
        committed_records=tuple(committed_records),
        next_frame_index=next_frame_index,
        analysis_state=analysis_state,
    )


def write_initial_run_artifacts(
    layout: RunLayout,
    *,
    created_at: datetime,
    input_path: Path,
    video_manifest: VideoManifest,
    calibration: CalibrationRecord,
    inference: InferenceRuntimeRecord,
    frame_image_retention: FrameImageRetentionPolicy,
    crop_image_retention: CropImageRetentionPolicy,
    qa_summary_policy: QASummaryPolicy,
) -> None:
    _write_json(
        layout.run_dir / "run_manifest.json",
        RunManifest(
            run_id=layout.run_dir.name,
            created_at_utc=_format_utc(created_at),
            input_path=str(input_path),
            video=video_manifest,
            inference=inference,
            frame_image_retention=frame_image_retention,
            crop_image_retention=crop_image_retention,
            qa_summary_policy=qa_summary_policy,
        ).model_dump(mode="json"),
    )
    _write_json(
        layout.run_dir / "calibration.json",
        calibration.model_dump(mode="json"),
    )
    _write_json(
        layout.run_dir / "video_manifest.json",
        video_manifest.model_dump(mode="json"),
    )
    (layout.records_dir / "frames.jsonl").touch()
    (layout.records_dir / "errors.jsonl").touch()


def new_analysis_state(
    layout: RunLayout,
    *,
    video_manifest: VideoManifest,
    next_frame_index: int,
    status: Literal["processing", "revalidating", "complete", "failed"],
    clock: Callable[[], datetime],
) -> AnalysisState:
    return AnalysisState(
        run_id=layout.run_dir.name,
        input_path=video_manifest.source_path,
        source_video_sha256=video_manifest.source_sha256,
        frame_count_decoded=video_manifest.frame_count_decoded,
        next_frame_index=next_frame_index,
        status=status,
        updated_at_utc=_format_utc(clock()),
    )


def update_analysis_state(
    state: AnalysisState,
    *,
    next_frame_index: int | None = None,
    status: Literal["processing", "revalidating", "complete", "failed"] | None = None,
    clock: Callable[[], datetime],
) -> AnalysisState:
    return AnalysisState(
        run_id=state.run_id,
        input_path=state.input_path,
        source_video_sha256=state.source_video_sha256,
        frame_count_decoded=state.frame_count_decoded,
        next_frame_index=(
            state.next_frame_index if next_frame_index is None else next_frame_index
        ),
        status=state.status if status is None else status,
        updated_at_utc=_format_utc(clock()),
    )


def validate_resume_cleanup_paths(layout: RunLayout) -> None:
    _require_within_run_root(layout.run_dir, root=layout.run_dir)

    for directory in (
        layout.raw_frames_dir,
        layout.processed_frames_dir,
        layout.face_crops_dir,
        layout.left_eye_crops_dir,
        layout.right_eye_crops_dir,
        layout.scene_dir,
        layout.viewer_dir,
    ):
        _validate_directory_tree_within_run_root(directory, root=layout.run_dir)

    for path in (
        layout.records_dir,
        layout.records_dir / "frames.jsonl",
        layout.records_dir / "errors.jsonl",
        layout.records_dir / "scene_frames.jsonl",
        layout.run_dir / "qa_summary.json",
        layout.run_dir / "analysis_state.json",
    ):
        _require_within_run_root(path, root=layout.run_dir)


def write_analysis_state(layout: RunLayout, state: AnalysisState) -> Path:
    path = layout.run_dir / "analysis_state.json"
    atomic_write_bytes(path, state.model_dump_json().encode("utf-8"))
    return path


def commit_processed_records(
    processed: Iterable[tuple[FrameRecord, list[ErrorRecord]]],
    *,
    frames_handle: TextIO,
    errors_handle: TextIO,
) -> tuple[int, int]:
    next_frame_index = 0
    frame_error_count = 0
    for record, frame_errors in processed:
        frames_handle.write(record.model_dump_json() + "\n")
        next_frame_index = record.frame_index + 1
        frame_error_count += len(frame_errors)
    flush_jsonl_checkpoint(frames_handle, errors_handle)
    return next_frame_index, frame_error_count


def flush_jsonl_checkpoint(*handles: TextIO) -> None:
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())


def _run_matches(
    run_dir: Path,
    *,
    expected_video_path: str,
    video_manifest: VideoManifest,
    calibration: CalibrationRecord,
    inference: InferenceRuntimeRecord,
    frame_image_retention: FrameImageRetentionPolicy,
    crop_image_retention: CropImageRetentionPolicy,
    expected_qa_summary_policy: QASummaryPolicy,
) -> bool:
    try:
        run_manifest = read_run_manifest_artifact_json(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        persisted_video_manifest = VideoManifest.model_validate_json(
            (run_dir / "video_manifest.json").read_text(encoding="utf-8")
        )
        persisted_calibration = CalibrationRecord.model_validate_json(
            (run_dir / "calibration.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return False

    return (
        run_manifest.input_path == expected_video_path
        and run_manifest.video == video_manifest
        and persisted_video_manifest == video_manifest
        and run_manifest.inference == inference
        and run_manifest.frame_image_retention == frame_image_retention
        and run_manifest.crop_image_retention == crop_image_retention
        and run_manifest.qa_summary_policy == expected_qa_summary_policy
        and persisted_calibration == calibration
    )


def _run_is_complete(run_dir: Path) -> bool:
    if _run_has_complete_qa_summary(run_dir):
        return True

    try:
        run_manifest = read_run_manifest_artifact_json(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return False

    if run_manifest.qa_summary_policy.generate_qa_summary:
        return False

    return _run_has_complete_no_qa_state(run_dir)


def _run_has_complete_qa_summary(run_dir: Path) -> bool:
    qa_summary_path = run_dir / "qa_summary.json"
    if not qa_summary_path.exists():
        return False

    try:
        qa_summary = QASummary.model_validate_json(
            qa_summary_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return False

    validation = qa_summary.artifact_validation
    return (
        qa_summary.final_status == "complete"
        and validation.final_status == "complete"
        and validation.schema_validation_passed
        and validation.counts_match
    )


def _run_has_complete_no_qa_state(run_dir: Path) -> bool:
    try:
        state = AnalysisState.model_validate_json(
            (run_dir / "analysis_state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return False

    return (
        state.status == "complete"
        and state.next_frame_index == state.frame_count_decoded
        and all(
            path.is_file() for path in _required_no_qa_completion_artifacts(run_dir)
        )
    )


def _required_no_qa_completion_artifacts(run_dir: Path) -> list[Path]:
    return [
        run_dir / "records" / "frames.jsonl",
        run_dir / "records" / "errors.jsonl",
        run_dir / "records" / "scene_frames.jsonl",
        run_dir / "scene" / "scene_manifest.json",
        run_dir / "scene" / "scene_summary.json",
        run_dir / "viewer" / "index.html",
        run_dir / "viewer" / "scene-data.json",
    ]


def _committed_frame_records(
    frames_jsonl_path: Path, *, frame_count_decoded: int
) -> list[FrameRecord]:
    if not frames_jsonl_path.exists():
        return []

    committed_records: list[FrameRecord] = []
    for raw_line in frames_jsonl_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            break

        try:
            record = FrameRecord.model_validate_json(raw_line)
        except ValidationError:
            break

        expected_frame_index = len(committed_records)
        if (
            record.frame_index != expected_frame_index
            or record.frame_id != frame_id(expected_frame_index)
            or record.frame_index >= frame_count_decoded
        ):
            break

        committed_records.append(record)
    return committed_records


def _rewrite_frames_jsonl(
    frames_jsonl_path: Path, committed_records: list[FrameRecord]
) -> None:
    atomic_write_bytes(
        frames_jsonl_path,
        _jsonl_bytes(record.model_dump_json() for record in committed_records),
    )


def _rewrite_errors_jsonl(
    errors_jsonl_path: Path, committed_records: list[FrameRecord]
) -> None:
    error_lines: list[str] = []
    for record in committed_records:
        for error in record.errors:
            error_lines.append(
                FrameErrorRecord(
                    frame_id=record.frame_id,
                    frame_index=record.frame_index,
                    code=error.code,
                    message=error.message,
                ).model_dump_json()
            )

    atomic_write_bytes(errors_jsonl_path, _jsonl_bytes(error_lines))


def _delete_uncommitted_frame_artifacts(
    layout: RunLayout, *, next_frame_index: int
) -> None:
    for directory in (
        layout.raw_frames_dir,
        layout.processed_frames_dir,
        layout.face_crops_dir,
        layout.left_eye_crops_dir,
        layout.right_eye_crops_dir,
    ):
        _delete_frame_indexed_files(
            directory,
            next_frame_index=next_frame_index,
            root=layout.run_dir,
        )


def _delete_frame_indexed_files(
    directory: Path,
    *,
    next_frame_index: int,
    root: Path,
) -> None:
    _validate_directory_tree_within_run_root(directory, root=root)
    if not directory.exists():
        return

    for path in directory.iterdir():
        if not path.is_file():
            continue

        index = _frame_index_from_path(path)
        if index is not None and index >= next_frame_index:
            _unlink_if_file(path, root=root)


def _delete_derived_artifacts(layout: RunLayout) -> None:
    for path in (
        layout.records_dir / "scene_frames.jsonl",
        layout.run_dir / "qa_summary.json",
        layout.run_dir / "analysis_state.json",
    ):
        _unlink_if_file(path, root=layout.run_dir)

    for directory in (layout.scene_dir, layout.viewer_dir):
        _validate_directory_tree_within_run_root(directory, root=layout.run_dir)
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                _unlink_if_file(path, root=layout.run_dir)


def _unlink_if_file(path: Path, *, root: Path) -> None:
    _require_within_run_root(path, root=root)

    if path.is_file():
        path.unlink()


def _require_within_run_root(path: Path, *, root: Path) -> None:
    if root.is_symlink():
        raise ValueError(f"refusing to delete outside run root: {path}")

    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"refusing to delete outside run root: {path}") from exc


def _validate_directory_tree_within_run_root(directory: Path, *, root: Path) -> None:
    _require_within_run_root(directory, root=root)
    if not directory.exists():
        return

    for path in directory.rglob("*"):
        _require_within_run_root(path, root=root)


def _frame_index_from_path(path: Path) -> int | None:
    name = path.stem
    if len(name) != 10 or not name.startswith("f") or not name[1:].isdigit():
        return None
    return int(name[1:])


def _jsonl_bytes(lines: Iterable[str]) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    data = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(path, data)


def _format_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
