from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chess_gaze.analysis_resume import (
    AnalysisState,
    find_latest_resumable_run,
    flush_jsonl_checkpoint,
    prepare_resume_run,
    write_analysis_state,
)
from chess_gaze.artifact_runs import RunLayout, create_run_layout
from chess_gaze.calibration import default_calibration
from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    ErrorRecord,
    EyeRecord,
    FaceRecord,
    FrameRecord,
    GazeAngles,
    HeadPoseRecord,
    RunManifest,
    VideoManifest,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.qa_summary import (
    ArtifactCounts,
    ArtifactValidationResult,
    ByteCounts,
    DetectionRates,
    DiskSpaceSummary,
    QASummary,
)
from chess_gaze.unigaze_runtime import external_observer_inference_record


def test_prepare_resume_run_repairs_committed_prefix_and_rebuilds_errors(
    tmp_path: Path,
) -> None:
    layout = _make_resume_layout(tmp_path, frame_count=4)
    records = [_fake_frame_record(0), _fake_frame_record(1, with_error=True)]
    (layout.records_dir / "frames.jsonl").write_text(
        records[0].model_dump_json()
        + "\n"
        + records[1].model_dump_json()
        + "\n"
        + '{"frame_id":"f000000999","frame_index":999}\n',
        encoding="utf-8",
    )
    (layout.records_dir / "errors.jsonl").write_text(
        '{"frame_id":"f000000002","frame_index":2,"code":"FACE_NOT_FOUND","message":"stale"}\n',
        encoding="utf-8",
    )
    (layout.raw_frames_dir / "f000000000.png").write_bytes(b"raw0")
    (layout.raw_frames_dir / "f000000002.png").write_bytes(b"raw2")
    (layout.processed_frames_dir / "f000000001.jpg").write_bytes(b"processed1")
    (layout.processed_frames_dir / "f000000002.jpg").write_bytes(b"processed2")
    (layout.left_eye_crops_dir / "f000000002.png").write_bytes(b"crop")
    (layout.records_dir / "scene_frames.jsonl").write_text("stale\n", encoding="utf-8")
    (layout.scene_dir / "scene_manifest.json").write_text("{}", encoding="utf-8")
    (layout.viewer_dir / "index.html").write_text("stale", encoding="utf-8")
    (layout.run_dir / "qa_summary.json").write_text("{}", encoding="utf-8")

    preparation = prepare_resume_run(
        layout,
        _video_manifest(frame_count=4),
        clock=lambda: datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
    )

    assert preparation.next_frame_index == 2
    assert [record.frame_index for record in preparation.committed_records] == [0, 1]
    repaired_lines = (
        (layout.records_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(repaired_lines) == 2
    error_lines = (
        (layout.records_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(error_lines) == 1
    assert json.loads(error_lines[0])["frame_id"] == "f000000001"
    assert not (layout.raw_frames_dir / "f000000002.png").exists()
    assert not (layout.processed_frames_dir / "f000000002.jpg").exists()
    assert not (layout.left_eye_crops_dir / "f000000002.png").exists()
    assert not (layout.records_dir / "scene_frames.jsonl").exists()
    assert not (layout.scene_dir / "scene_manifest.json").exists()
    assert not (layout.viewer_dir / "index.html").exists()
    assert not (layout.run_dir / "qa_summary.json").exists()


def test_find_latest_resumable_run_ignores_complete_and_incompatible_runs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "output" / "clip" / "runs"
    compatible = _make_compatible_run(runs_root / "20260628T100000Z-good")
    complete = _make_compatible_run(runs_root / "20260628T110000Z-complete")
    _make_compatible_run(
        runs_root / "20260628T120000Z-wrong",
        source_sha256="wrong",
    )
    _write_complete_qa_summary(complete)

    result = find_latest_resumable_run(
        runs_root,
        Path("artifacts/input/clip.mp4"),
        _video_manifest(frame_count=4),
        default_calibration(),
        external_observer_inference_record(),
    )

    assert result == compatible


def test_prepare_resume_run_refuses_to_delete_frame_artifacts_outside_run_root(
    tmp_path: Path,
) -> None:
    layout = _make_resume_layout(tmp_path, frame_count=2)
    committed_record = _fake_frame_record(0)
    (layout.records_dir / "frames.jsonl").write_text(
        committed_record.model_dump_json() + "\n",
        encoding="utf-8",
    )
    outside_dir = tmp_path / "outside-raw-frames"
    outside_dir.mkdir()
    outside_file = outside_dir / "f000000001.png"
    outside_file.write_bytes(b"outside")
    malformed_layout = RunLayout(
        run_dir=layout.run_dir,
        raw_frames_dir=outside_dir,
        processed_frames_dir=layout.processed_frames_dir,
        crops_dir=layout.crops_dir,
        face_crops_dir=layout.face_crops_dir,
        eyes_crops_dir=layout.eyes_crops_dir,
        left_eye_crops_dir=layout.left_eye_crops_dir,
        right_eye_crops_dir=layout.right_eye_crops_dir,
        records_dir=layout.records_dir,
    )

    with pytest.raises(ValueError, match="outside run root"):
        prepare_resume_run(
            malformed_layout,
            _video_manifest(frame_count=2),
            clock=lambda: datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
        )

    assert outside_file.exists()


def test_prepare_resume_run_refuses_to_traverse_symlinked_run_subdirectories(
    tmp_path: Path,
) -> None:
    layout = _make_resume_layout(tmp_path, frame_count=1)
    (layout.records_dir / "frames.jsonl").write_text("", encoding="utf-8")
    outside_scene_dir = tmp_path / "outside-scene"
    outside_scene_dir.mkdir()
    outside_file = outside_scene_dir / "scene_manifest.json"
    outside_file.write_text('{"preserve": true}', encoding="utf-8")
    layout.scene_dir.rmdir()
    layout.scene_dir.symlink_to(outside_scene_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="outside run root"):
        prepare_resume_run(
            layout,
            _video_manifest(frame_count=1),
            clock=lambda: datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
        )

    assert outside_file.exists()


def test_write_analysis_state_writes_expected_json(tmp_path: Path) -> None:
    layout = _make_resume_layout(tmp_path, frame_count=3)
    state = AnalysisState(
        run_id=layout.run_dir.name,
        input_path="artifacts/input/clip.mp4",
        source_video_sha256="b" * 64,
        frame_count_decoded=3,
        next_frame_index=2,
        status="processing",
        updated_at_utc="2026-06-28T12:00:00Z",
    )

    result = write_analysis_state(layout, state)

    assert result == layout.run_dir / "analysis_state.json"
    assert (
        AnalysisState.model_validate_json(result.read_text(encoding="utf-8")) == state
    )


def test_flush_jsonl_checkpoint_flushes_and_fsyncs_each_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsync_calls: list[int] = []

    def record_fsync(fileno: int) -> None:
        fsync_calls.append(fileno)

    monkeypatch.setattr(os, "fsync", record_fsync)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    with (
        first_path.open("w", encoding="utf-8") as first_handle,
        second_path.open(
            "w",
            encoding="utf-8",
        ) as second_handle,
    ):
        first_handle.write('{"frame":0}\n')
        second_handle.write('{"frame":1}\n')

        flush_jsonl_checkpoint(first_handle, second_handle)

        assert fsync_calls == [first_handle.fileno(), second_handle.fileno()]
        assert first_path.read_text(encoding="utf-8") == '{"frame":0}\n'
        assert second_path.read_text(encoding="utf-8") == '{"frame":1}\n'


def _make_resume_layout(tmp_path: Path, *, frame_count: int) -> RunLayout:
    source = tmp_path / "input" / "clip.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"video")

    layout = create_run_layout(
        input_path=source,
        output_root=tmp_path / "output" / "clip" / "runs",
        clock=lambda: datetime(2026, 6, 28, 9, 0, tzinfo=UTC),
        run_suffix="resume01",
    )
    _write_run_metadata(layout, source_path=source, frame_count=frame_count)
    return layout


def _make_compatible_run(run_dir: Path, *, source_sha256: str = "a" * 64) -> RunLayout:
    run_dir.mkdir(parents=True)
    layout = RunLayout(
        run_dir=run_dir,
        raw_frames_dir=run_dir / "raw_frames",
        processed_frames_dir=run_dir / "processed_frames",
        crops_dir=run_dir / "crops",
        face_crops_dir=run_dir / "crops" / "face",
        eyes_crops_dir=run_dir / "crops" / "eyes",
        left_eye_crops_dir=run_dir / "crops" / "eyes" / "left",
        right_eye_crops_dir=run_dir / "crops" / "eyes" / "right",
        records_dir=run_dir / "records",
    )
    for directory in (
        layout.raw_frames_dir,
        layout.processed_frames_dir,
        layout.face_crops_dir,
        layout.left_eye_crops_dir,
        layout.right_eye_crops_dir,
        layout.records_dir,
        layout.scene_dir,
        layout.viewer_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _write_run_metadata(
        layout,
        source_path=Path("artifacts/input/clip.mp4"),
        frame_count=4,
        source_sha256=source_sha256,
    )
    return layout


def _write_run_metadata(
    layout: RunLayout,
    *,
    source_path: Path,
    frame_count: int,
    source_sha256: str = "a" * 64,
) -> None:
    video_manifest = _video_manifest(
        frame_count=frame_count,
        source_path=source_path,
        source_sha256=source_sha256,
    )
    run_manifest = RunManifest(
        run_id=layout.run_dir.name,
        created_at_utc="2026-06-28T10:00:00Z",
        input_path=str(source_path),
        video=video_manifest,
        inference=external_observer_inference_record(),
    )
    (layout.run_dir / "run_manifest.json").write_text(
        run_manifest.model_dump_json(),
        encoding="utf-8",
    )
    (layout.run_dir / "video_manifest.json").write_text(
        video_manifest.model_dump_json(),
        encoding="utf-8",
    )
    (layout.run_dir / "calibration.json").write_text(
        default_calibration().model_dump_json(),
        encoding="utf-8",
    )


def _write_complete_qa_summary(layout: RunLayout) -> None:
    summary = QASummary(
        run_id=layout.run_dir.name,
        source_video_path="artifacts/input/clip.mp4",
        source_video_sha256="a" * 64,
        source_artifacts={"frames_jsonl": "records/frames.jsonl"},
        counts=ArtifactCounts(
            decoded_frames=4,
            frame_records=4,
            scene_frame_records=4,
            raw_frames=4,
            processed_frames=4,
            crop_files=0,
        ),
        byte_counts=ByteCounts(
            raw_frames_bytes=0,
            processed_frames_bytes=0,
            crops_bytes=0,
            jsonl_bytes=0,
            scene_jsonl_bytes=0,
            scene_bytes=0,
            viewer_bytes=0,
            total_run_bytes=0,
        ),
        rates=DetectionRates(
            face_present_rate=1.0,
            both_eyes_present_rate=1.0,
            left_eye_only_rate=0.0,
            right_eye_only_rate=0.0,
            left_iris_present_rate=1.0,
            right_iris_present_rate=1.0,
            head_pose_valid_rate=1.0,
            face_gaze_valid_rate=1.0,
            recommended_gaze_valid_rate=1.0,
        ),
        errors_by_code={},
        errors_by_severity={},
        worst_blur_frame_ids=[],
        worst_exposure_frame_ids=[],
        qa_sample_frame_ids=[],
        representative_failure_frame_ids=[],
        status_transitions=["processing", "complete"],
        final_status="complete",
        disk_space=DiskSpaceSummary(
            preflight_estimate_bytes=0,
            closeout_free_bytes=0,
            closeout_total_bytes=0,
        ),
        artifact_validation=ArtifactValidationResult(
            source_artifacts={"frames_jsonl": "records/frames.jsonl"},
            counts=ArtifactCounts(
                decoded_frames=4,
                frame_records=4,
                scene_frame_records=4,
                raw_frames=4,
                processed_frames=4,
                crop_files=0,
            ),
            byte_counts=ByteCounts(
                raw_frames_bytes=0,
                processed_frames_bytes=0,
                crops_bytes=0,
                jsonl_bytes=0,
                scene_jsonl_bytes=0,
                scene_bytes=0,
                viewer_bytes=0,
                total_run_bytes=0,
            ),
            schema_validation_passed=True,
            counts_match=True,
            validation_errors=[],
            final_status="complete",
        ),
        built_from_disk_at_utc="2026-06-28T12:00:00Z",
    )
    (layout.run_dir / "qa_summary.json").write_text(
        summary.model_dump_json(),
        encoding="utf-8",
    )


def _video_manifest(
    *,
    frame_count: int,
    source_path: Path = Path("artifacts/input/clip.mp4"),
    source_sha256: str = "a" * 64,
) -> VideoManifest:
    return VideoManifest(
        source_path=str(source_path),
        source_sha256=source_sha256,
        frame_width=1920,
        frame_height=1080,
        frame_count_decoded=frame_count,
    )


def _fake_frame_record(frame_index: int, *, with_error: bool = False) -> FrameRecord:
    return FrameRecord(
        frame_id=f"f{frame_index:09d}",
        frame_index=frame_index,
        status=FrameStatus.ERROR if with_error else FrameStatus.OK,
        timestamp_seconds=frame_index / 30.0,
        face=FaceRecord(
            present=not with_error,
            bounding_box=_box(10.0, 10.0, 20.0, 20.0) if not with_error else None,
            landmarks=(
                [_point(12.0, 12.0), _point(18.0, 12.0)] if not with_error else None
            ),
            reason_invalid=None if not with_error else ErrorCode.FACE_NOT_FOUND,
        ),
        left_eye=_eye_record(not with_error, ErrorCode.LEFT_EYE_NOT_FOUND),
        right_eye=_eye_record(not with_error, ErrorCode.RIGHT_EYE_NOT_FOUND),
        head_pose=HeadPoseRecord(
            valid=not with_error,
            yaw_radians=0.01 if not with_error else None,
            pitch_radians=0.02 if not with_error else None,
            roll_radians=0.03 if not with_error else None,
            reason_invalid=None if not with_error else ErrorCode.HEAD_POSE_INVALID,
        ),
        geometric_gaze=_gaze_angles(not with_error, ErrorCode.GAZE_MODEL_FAILED),
        appearance_gaze=_gaze_angles(not with_error, ErrorCode.GAZE_MODEL_FAILED),
        recommended_gaze=_gaze_angles(
            not with_error,
            ErrorCode.GAZE_ESTIMATORS_DISAGREE,
        ),
        errors=(
            [
                ErrorRecord(
                    code=ErrorCode.FACE_NOT_FOUND,
                    message="No face detected in frame.",
                )
            ]
            if with_error
            else []
        ),
    )


def _gaze_angles(valid: bool, invalid_reason: ErrorCode) -> GazeAngles:
    return GazeAngles(
        valid=valid,
        yaw_radians=0.01 if valid else None,
        pitch_radians=0.02 if valid else None,
        reason_invalid=None if valid else invalid_reason,
    )


def _eye_record(present: bool, invalid_reason: ErrorCode) -> EyeRecord:
    return EyeRecord(
        present=present,
        bounding_box=_box(12.0, 12.0, 18.0, 18.0) if present else None,
        pupil_center=_point(15.0, 15.0) if present else None,
        iris_landmarks=[
            _point(14.0, 15.0),
            _point(16.0, 15.0),
            _point(15.0, 14.0),
            _point(15.0, 16.0),
        ]
        if present
        else None,
        reason_invalid=None if present else invalid_reason,
    )


def _point(x: float, y: float) -> Point2D:
    return Point2D(space=CoordinateSpace.IMAGE_PX, x=x, y=y)


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> BBox:
    return BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )
