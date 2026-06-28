from __future__ import annotations

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
    FrameErrorRecord,
    FrameRecord,
    InferenceRuntimeRecord,
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
    status: Literal["processing", "complete", "failed"]
    updated_at_utc: str


def find_latest_resumable_run(
    runs_root: Path,
    video_path: Path,
    video_manifest: VideoManifest,
    calibration: CalibrationRecord,
    inference: InferenceRuntimeRecord,
) -> RunLayout | None:
    if not runs_root.exists():
        return None

    expected_video_path = str(video_path)
    for run_dir in sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    ):
        if not _run_matches(
            run_dir,
            expected_video_path=expected_video_path,
            video_manifest=video_manifest,
            calibration=calibration,
            inference=inference,
        ):
            continue
        if _run_is_complete(run_dir):
            continue
        return run_layout_from_dir(run_dir)
    return None


def prepare_resume_run(
    layout: RunLayout,
    video_manifest: VideoManifest,
    *,
    clock: Callable[[], datetime],
) -> ResumePreparation:
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


def write_analysis_state(layout: RunLayout, state: AnalysisState) -> Path:
    path = layout.run_dir / "analysis_state.json"
    atomic_write_bytes(path, state.model_dump_json().encode("utf-8"))
    return path


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
    except (FileNotFoundError, ValidationError, ValueError):
        return False

    return (
        run_manifest.input_path == expected_video_path
        and run_manifest.video == video_manifest
        and persisted_video_manifest == video_manifest
        and run_manifest.inference == inference
        and persisted_calibration == calibration
    )


def _run_is_complete(run_dir: Path) -> bool:
    qa_summary_path = run_dir / "qa_summary.json"
    if not qa_summary_path.exists():
        return False

    try:
        qa_summary = QASummary.model_validate_json(
            qa_summary_path.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ValidationError, ValueError):
        return False

    validation = qa_summary.artifact_validation
    return (
        qa_summary.final_status == "complete"
        and validation.final_status == "complete"
        and validation.schema_validation_passed
        and validation.counts_match
    )


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
        _delete_frame_indexed_files(directory, next_frame_index=next_frame_index)


def _delete_frame_indexed_files(directory: Path, *, next_frame_index: int) -> None:
    if not directory.exists():
        return

    for path in directory.iterdir():
        if not path.is_file():
            continue

        index = _frame_index_from_path(path)
        if index is not None and index >= next_frame_index:
            path.unlink()


def _delete_derived_artifacts(layout: RunLayout) -> None:
    for path in (
        layout.records_dir / "scene_frames.jsonl",
        layout.run_dir / "qa_summary.json",
        layout.run_dir / "analysis_state.json",
    ):
        _unlink_if_file(path, root=layout.run_dir)

    for directory in (layout.scene_dir, layout.viewer_dir):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                _unlink_if_file(path, root=layout.run_dir)


def _unlink_if_file(path: Path, *, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive boundary
        raise ValueError(f"refusing to delete outside run root: {path}") from exc

    if path.is_file():
        path.unlink()


def _frame_index_from_path(path: Path) -> int | None:
    name = path.stem
    if len(name) != 10 or not name.startswith("f") or not name[1:].isdigit():
        return None
    return int(name[1:])


def _jsonl_bytes(lines: Iterable[str]) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def _format_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
