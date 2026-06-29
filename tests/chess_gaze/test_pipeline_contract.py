from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO, cast

import av
import numpy as np
import pytest

from chess_gaze.errors import CliErrorCode, ErrorCode, FrameStatus
from chess_gaze.frame_observation import ModelInferenceError
from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.model_assets import ResolvedModelAsset
from chess_gaze.pipeline import (
    AnalysisProgressEvent,
    AnalyzeRequest,
    ObserverBundle,
    ObserverFrame,
    PipelineError,
    analyze_video,
)
from chess_gaze.qa_summary import QASummary
from chess_gaze.scene_artifacts import (
    build_scene_artifacts as real_build_scene_artifacts,
)
from chess_gaze.scene_records import SceneSummary, ViewerSceneData


def make_tiny_video(path: Path, frame_count: int = 3) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=3)
    stream.width = 96
    stream.height = 72
    stream.pix_fmt = "yuv420p"
    for index in range(frame_count):
        image = np.full((72, 96, 3), index * 40, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


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


def _missing_eye_record(
    frame: ObserverFrame, *, code: ErrorCode, message: str
) -> dict[str, object]:
    return {
        "present": False,
        "bounding_box": None,
        "pupil_center": None,
        "iris_landmarks": None,
        "reason_invalid": code,
    }


def _observed_eye_record(
    *,
    x_offset: float,
    y_offset: float,
    frame_index: int,
) -> dict[str, object]:
    pupil = _point(x_offset + frame_index, y_offset)
    return {
        "present": True,
        "bounding_box": _box(
            x_offset - 8.0, y_offset - 6.0, x_offset + 8.0, y_offset + 6.0
        ),
        "pupil_center": pupil,
        "iris_landmarks": [
            _point(pupil.x - 3.0, pupil.y),
            _point(pupil.x + 3.0, pupil.y),
            _point(pupil.x, pupil.y - 3.0),
            _point(pupil.x, pupil.y + 3.0),
        ],
        "reason_invalid": None,
    }


def _fake_record(
    frame: ObserverFrame,
    *,
    face_present: bool = True,
    left_eye_present: bool = True,
    right_eye_present: bool = True,
) -> FrameRecord:
    errors = []
    if face_present:
        face = {
            "present": True,
            "bounding_box": _box(
                20.0 + frame.frame_index,
                10.0,
                76.0 + frame.frame_index,
                58.0,
            ),
            "landmarks": [
                _point(34.0 + frame.frame_index, 24.0),
                _point(62.0 + frame.frame_index, 24.0),
                _point(48.0 + frame.frame_index, 38.0),
            ],
            "reason_invalid": None,
        }
    else:
        face = {
            "present": False,
            "bounding_box": None,
            "landmarks": None,
            "reason_invalid": ErrorCode.FACE_NOT_FOUND,
        }
        errors.append(
            {
                "code": ErrorCode.FACE_NOT_FOUND,
                "message": "No face detected by fake observer.",
            }
        )

    if left_eye_present:
        left_eye = _observed_eye_record(
            x_offset=38.0,
            y_offset=30.0,
            frame_index=frame.frame_index,
        )
    else:
        left_eye = _missing_eye_record(
            frame,
            code=ErrorCode.LEFT_EYE_NOT_FOUND,
            message="Left eye missing.",
        )
        errors.append(
            {
                "code": ErrorCode.LEFT_EYE_NOT_FOUND,
                "message": "Left eye missing.",
            }
        )

    if right_eye_present:
        right_eye = _observed_eye_record(
            x_offset=58.0,
            y_offset=30.0,
            frame_index=frame.frame_index,
        )
    else:
        right_eye = _missing_eye_record(
            frame,
            code=ErrorCode.RIGHT_EYE_NOT_FOUND,
            message="Right eye missing.",
        )
        errors.append(
            {
                "code": ErrorCode.RIGHT_EYE_NOT_FOUND,
                "message": "Right eye missing.",
            }
        )

    valid = face_present and left_eye_present and right_eye_present
    gaze_yaw = frame.frame_index / 100.0
    gaze_pitch = -frame.frame_index / 200.0
    invalid_reason = None if valid else ErrorCode.GAZE_ESTIMATORS_DISAGREE
    gaze = {
        "valid": valid,
        "yaw_radians": gaze_yaw if valid else None,
        "pitch_radians": gaze_pitch if valid else None,
        "reason_invalid": invalid_reason,
    }
    head_pose = {
        "valid": face_present,
        "yaw_radians": frame.frame_index / 250.0 if face_present else None,
        "pitch_radians": 0.01 if face_present else None,
        "roll_radians": 0.02 if face_present else None,
        "reason_invalid": None if face_present else ErrorCode.HEAD_POSE_INVALID,
    }
    if not face_present:
        errors.append(
            {
                "code": ErrorCode.HEAD_POSE_INVALID,
                "message": "Head pose unavailable without face.",
            }
        )
    if not valid:
        errors.append(
            {
                "code": ErrorCode.GAZE_ESTIMATORS_DISAGREE,
                "message": "Recommended gaze unavailable for incomplete fake record.",
            }
        )

    return FrameRecord.model_validate(
        {
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "status": FrameStatus.OK if valid else FrameStatus.ERROR,
            "timestamp_seconds": frame.timestamp_seconds,
            "face": face,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "head_pose": head_pose,
            "geometric_gaze": gaze,
            "appearance_gaze": gaze,
            "recommended_gaze": gaze,
            "errors": errors,
        }
    )


def _records_from(result_path: Path) -> list[FrameRecord]:
    return [
        FrameRecord.model_validate_json(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ]


def _jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_model_registry_with_assets(models_root: Path, registry_path: Path) -> None:
    mediapipe_path = models_root / "mediapipe" / "face_landmarker.task"
    unigaze_path = models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    mediapipe_path.parent.mkdir(parents=True)
    unigaze_path.parent.mkdir(parents=True)
    mediapipe_path.write_bytes(b"mediapipe")
    unigaze_path.write_bytes(b"unigaze")
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "mediapipe-face-landmarker",
                        "task_name": "face_landmarks",
                        "expected_relative_path": "mediapipe/face_landmarker.task",
                        "checksum_sha256": None,
                        "source_url": "https://example.invalid/mediapipe",
                        "license": "Google AI Edge Terms",
                        "requires_license_approval": False,
                        "license_approved": True,
                        "license_approved_by": "repo_owner",
                        "license_approved_at": "2026-06-25",
                        "input_contract": {"running_mode": "IMAGE"},
                        "output_contract": {"landmarks": "face mesh"},
                    },
                    {
                        "model_id": "unigaze-h14-joint",
                        "task_name": "gaze_estimation",
                        "expected_relative_path": (
                            "unigaze/unigaze_h14_joint.safetensors"
                        ),
                        "checksum_sha256": None,
                        "source_url": "https://example.invalid/unigaze",
                        "license": "MG-NC-RAI-2.0",
                        "requires_license_approval": True,
                        "license_approved": True,
                        "license_approved_by": "repo_owner",
                        "license_approved_at": "2026-06-25",
                        "input_contract": {"input_size_px": 224},
                        "output_contract": {"order": "pitch_yaw_radians"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_analyze_video_does_not_retain_raw_or_processed_frame_images_by_default(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=4)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.decoded_frame_count == 4
    assert list(result.layout.raw_frames_dir.glob("*.png")) == []
    assert list(result.layout.processed_frames_dir.glob("*.jpg")) == []
    records = _records_from(result.frames_jsonl_path)
    assert len(records) == 4
    assert [record.frame_id for record in records] == [
        "f000000000",
        "f000000001",
        "f000000002",
        "f000000003",
    ]
    assert [record.recommended_gaze.yaw_radians for record in records] == [
        0.0,
        0.01,
        0.02,
        0.03,
    ]
    assert (result.layout.run_dir / "run_manifest.json").is_file()
    assert (result.layout.run_dir / "video_manifest.json").is_file()
    assert (result.layout.run_dir / "calibration.json").is_file()
    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["frame_image_retention"] == {
        "schema_version": "frame-image-retention-v1",
        "save_frame_images": False,
    }
    assert manifest["crop_image_retention"] == {
        "schema_version": "crop-image-retention-v1",
        "save_crop_images": False,
    }
    assert result.qa_summary_path.is_file()
    summary = QASummary.model_validate_json(
        result.qa_summary_path.read_text(encoding="utf-8")
    )
    assert summary.counts.raw_frames == 0
    assert summary.counts.processed_frames == 0
    assert summary.artifact_validation.counts_match is True
    assert summary.final_status == "complete"
    assert summary.byte_counts.total_run_bytes == sum(
        path.stat().st_size
        for path in result.layout.run_dir.rglob("*")
        if path.is_file()
    )


def test_analyze_video_retains_raw_and_processed_frame_images_when_requested(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=3)

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            save_frame_images=True,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.decoded_frame_count == 3
    assert len(list(result.layout.raw_frames_dir.glob("*.png"))) == 3
    assert len(list(result.layout.processed_frames_dir.glob("*.jpg"))) == 3
    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["frame_image_retention"] == {
        "schema_version": "frame-image-retention-v1",
        "save_frame_images": True,
    }
    assert manifest["crop_image_retention"] == {
        "schema_version": "crop-image-retention-v1",
        "save_crop_images": False,
    }
    summary = QASummary.model_validate_json(
        result.qa_summary_path.read_text(encoding="utf-8")
    )
    assert summary.counts.raw_frames == 3
    assert summary.counts.processed_frames == 3
    assert summary.artifact_validation.counts_match is True


def test_analyze_video_resumes_latest_compatible_partial_run(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=5)
    observed_frames: list[int] = []

    def interrupt_after_two(frame: ObserverFrame) -> FrameRecord:
        observed_frames.append(frame.frame_index)
        if frame.frame_index == 2:
            raise RuntimeError("simulated interruption")
        return _fake_record(frame)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        analyze_video(
            AnalyzeRequest(video_path=video_path, output_root=output_root),
            observers=ObserverBundle(frame_observer=interrupt_after_two),
        )

    [run_dir] = (output_root / "tiny" / "runs").iterdir()
    assert observed_frames == [0, 1, 2]
    assert len(_records_from(run_dir / "records" / "frames.jsonl")) == 2

    resumed_observed_frames: list[int] = []

    def resumed_observer(frame: ObserverFrame) -> FrameRecord:
        resumed_observed_frames.append(frame.frame_index)
        return _fake_record(frame)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=output_root),
        observers=ObserverBundle(frame_observer=resumed_observer),
    )

    assert result.layout.run_dir == run_dir
    assert resumed_observed_frames == [2, 3, 4]
    assert [
        record.frame_index for record in _records_from(result.frames_jsonl_path)
    ] == [0, 1, 2, 3, 4]
    summary = QASummary.model_validate_json(
        result.qa_summary_path.read_text(encoding="utf-8")
    )
    assert summary.final_status == "complete"
    assert summary.counts.frame_records == 5


def test_analyze_video_resumes_partial_batched_run(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=5)
    interrupted_batches: list[list[int]] = []

    def interrupting_batch(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
        frame_indices = [frame.frame_index for frame in frames]
        interrupted_batches.append(frame_indices)
        if frame_indices == [2, 3]:
            raise RuntimeError("simulated batch interruption")
        return [_fake_record(frame) for frame in frames]

    with pytest.raises(RuntimeError, match="simulated batch interruption"):
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                unigaze_batch_size=2,
            ),
            observers=ObserverBundle(
                frame_observer=_fake_record,
                frame_batch_observer=interrupting_batch,
            ),
        )

    [run_dir] = (output_root / "tiny" / "runs").iterdir()
    assert interrupted_batches == [[0, 1], [2, 3]]
    assert len(_records_from(run_dir / "records" / "frames.jsonl")) == 2

    resumed_batches: list[list[int]] = []

    def resumed_batch(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
        resumed_batches.append([frame.frame_index for frame in frames])
        return [_fake_record(frame) for frame in frames]

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=output_root,
            unigaze_batch_size=2,
        ),
        observers=ObserverBundle(
            frame_observer=_fake_record,
            frame_batch_observer=resumed_batch,
        ),
    )

    assert result.layout.run_dir == run_dir
    assert resumed_batches == [[2, 3], [4]]
    assert [
        record.frame_index for record in _records_from(result.frames_jsonl_path)
    ] == [0, 1, 2, 3, 4]


def test_analyze_video_no_resume_forces_fresh_run_for_partial_run(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=3)

    def interrupt_after_first(frame: ObserverFrame) -> FrameRecord:
        if frame.frame_index == 1:
            raise RuntimeError("simulated interruption")
        return _fake_record(frame)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        analyze_video(
            AnalyzeRequest(video_path=video_path, output_root=output_root),
            observers=ObserverBundle(frame_observer=interrupt_after_first),
        )

    [partial_run_dir] = (output_root / "tiny" / "runs").iterdir()

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=output_root,
            resume=False,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.layout.run_dir != partial_run_dir
    assert {path.name for path in (output_root / "tiny" / "runs").iterdir()} == {
        partial_run_dir.name,
        result.layout.run_dir.name,
    }
    assert [
        record.frame_index for record in _records_from(result.frames_jsonl_path)
    ] == [0, 1, 2]


def test_analyze_video_does_not_resume_complete_run(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=2)

    first = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=output_root),
        observers=ObserverBundle(frame_observer=_fake_record),
    )
    second = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=output_root),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert second.layout.run_dir != first.layout.run_dir


def test_analyze_video_uses_batch_observer_without_reordering_frames(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=5)
    observed_batches: list[list[str]] = []

    def fake_batch_record(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
        observed_batches.append([frame.frame_id for frame in frames])
        return [_fake_record(frame) for frame in frames]

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            unigaze_batch_size=2,
        ),
        observers=ObserverBundle(
            frame_observer=_fake_record,
            frame_batch_observer=fake_batch_record,
        ),
    )

    records = _records_from(result.frames_jsonl_path)
    assert observed_batches == [
        ["f000000000", "f000000001"],
        ["f000000002", "f000000003"],
        ["f000000004"],
    ]
    assert [record.frame_id for record in records] == [
        "f000000000",
        "f000000001",
        "f000000002",
        "f000000003",
        "f000000004",
    ]
    assert list(result.layout.raw_frames_dir.glob("*.png")) == []
    assert list(result.layout.processed_frames_dir.glob("*.jpg")) == []
    assert result.decoded_frame_count == 5


def test_analyze_video_reports_committed_progress_after_each_batch(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=5)
    events: list[tuple[Path, int, int]] = []

    def progress_callback(event: AnalysisProgressEvent) -> None:
        events.append((event.run_dir, event.completed_frames, event.total_frames))

    def fake_batch_record(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
        return [_fake_record(frame) for frame in frames]

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            unigaze_batch_size=2,
            progress_callback=progress_callback,
        ),
        observers=ObserverBundle(
            frame_observer=_fake_record,
            frame_batch_observer=fake_batch_record,
        ),
    )

    assert events == [
        (result.layout.run_dir, 0, 5),
        (result.layout.run_dir, 2, 5),
        (result.layout.run_dir, 4, 5),
        (result.layout.run_dir, 5, 5),
    ]


def test_analyze_video_reports_resumed_committed_progress_from_resume_point(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=5)

    def interrupting_batch(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
        frame_indices = [frame.frame_index for frame in frames]
        if frame_indices == [2, 3]:
            raise RuntimeError("simulated batch interruption")
        return [_fake_record(frame) for frame in frames]

    with pytest.raises(RuntimeError, match="simulated batch interruption"):
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                unigaze_batch_size=2,
            ),
            observers=ObserverBundle(
                frame_observer=_fake_record,
                frame_batch_observer=interrupting_batch,
            ),
        )

    resumed_events: list[tuple[int, int]] = []

    def progress_callback(event: AnalysisProgressEvent) -> None:
        resumed_events.append((event.completed_frames, event.total_frames))

    def resumed_batch(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
        return [_fake_record(frame) for frame in frames]

    analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=output_root,
            unigaze_batch_size=2,
            progress_callback=progress_callback,
        ),
        observers=ObserverBundle(
            frame_observer=_fake_record,
            frame_batch_observer=resumed_batch,
        ),
    )

    assert resumed_events == [(2, 5), (4, 5), (5, 5)]


def test_batch_observer_identity_mismatch_fails_schema_validation(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=2)

    def wrong_batch_record(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
        records = [_fake_record(frame) for frame in frames]
        payload = records[0].model_dump(mode="python")
        payload["frame_index"] = 99
        return [FrameRecord.model_validate(payload), records[1]]

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=tmp_path / "output",
                unigaze_batch_size=2,
            ),
            observers=ObserverBundle(
                frame_observer=_fake_record,
                frame_batch_observer=wrong_batch_record,
            ),
        )

    assert exc_info.value.code is CliErrorCode.SCHEMA_VALIDATION_FAILED


def test_single_observer_fallback_keeps_immediate_frame_processing(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=2)
    observed_frames: list[str] = []

    def fail_on_first_frame(frame: ObserverFrame) -> FrameRecord:
        observed_frames.append(frame.frame_id)
        if frame.frame_index == 0:
            raise RuntimeError("stop after first frame")
        return _fake_record(frame)

    with pytest.raises(RuntimeError, match="stop after first frame"):
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                unigaze_batch_size=2,
            ),
            observers=ObserverBundle(frame_observer=fail_on_first_frame),
        )

    assert observed_frames == ["f000000000"]
    [run_dir] = (output_root / "tiny" / "runs").iterdir()
    assert list((run_dir / "raw_frames").glob("*.png")) == []


def test_batch_observer_record_count_mismatch_fails_schema_validation(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=2)

    def short_batch_record(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
        return [_fake_record(frames[0])]

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=tmp_path / "output",
                unigaze_batch_size=2,
            ),
            observers=ObserverBundle(
                frame_observer=_fake_record,
                frame_batch_observer=short_batch_record,
            ),
        )

    assert exc_info.value.code is CliErrorCode.SCHEMA_VALIDATION_FAILED


def test_default_model_batch_inference_failure_returns_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "models"
    registry_path = tmp_path / "model_registry.json"
    make_tiny_video(video_path, frame_count=2)
    _write_model_registry_with_assets(models_root, registry_path)

    from chess_gaze import pipeline
    from chess_gaze.frame_records import InferenceRuntimeRecord

    def fake_prepare_unigaze_runtime(
        asset: object, *, device: str, batch_size: int, input_size_px: int
    ) -> object:
        del asset, input_size_px
        return SimpleNamespace(
            model=object(),
            inference=InferenceRuntimeRecord(
                observer_source="default_model_observer",
                unigaze_model_id="unigaze-h14-joint",
                unigaze_device=cast(Any, device),
                unigaze_batch_size=batch_size,
                torch_version="test-torch",
                torch_mps_available=True,
                mps_fallback_env="unset",
                mps_fast_math_env="unset",
                mps_prefer_metal_env="unset",
                mps_preflight_passed=None,
            ),
        )

    def fake_default_observer_bundle_factory(
        resolved_assets: list[Any],
        calibration: object,
        run_layout: object,
        gaze_model: object,
        save_crop_images: bool,
    ) -> ObserverBundle:
        del resolved_assets, calibration, run_layout, gaze_model, save_crop_images

        def fail_batch(frames: Sequence[ObserverFrame]) -> list[FrameRecord]:
            del frames
            raise ModelInferenceError("UniGaze batch inference failed: simulated")

        return ObserverBundle(
            frame_observer=_fake_record,
            frame_batch_observer=fail_batch,
        )

    monkeypatch.setattr(
        pipeline, "prepare_unigaze_runtime", fake_prepare_unigaze_runtime
    )
    monkeypatch.setattr(
        pipeline,
        "default_observer_bundle_factory",
        fake_default_observer_bundle_factory,
    )

    with pytest.raises(PipelineError, match="simulated") as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                models_root=models_root,
                model_registry_path=registry_path,
                unigaze_device="cpu",
                unigaze_batch_size=2,
            )
        )

    assert exc_info.value.code is CliErrorCode.USAGE


def test_analyze_video_writes_scene_artifacts_and_viewer_files(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=4)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.scene_manifest_path == result.layout.scene_dir / "scene_manifest.json"
    assert result.scene_summary_path == result.layout.scene_dir / "scene_summary.json"
    assert result.scene_frames_jsonl_path == result.layout.records_dir / (
        "scene_frames.jsonl"
    )
    assert result.viewer_index_path == result.layout.viewer_dir / "index.html"
    assert result.viewer_scene_data_path == result.layout.viewer_dir / "scene-data.json"
    assert result.scene_manifest_path.is_file()
    assert result.scene_summary_path.is_file()
    assert result.scene_frames_jsonl_path.is_file()
    assert result.viewer_index_path.is_file()
    assert result.viewer_scene_data_path.is_file()

    summary = QASummary.model_validate_json(
        result.qa_summary_path.read_text(encoding="utf-8")
    )
    assert result.validated_record_count == 4
    assert result.valid_scene_frame_count == 4
    assert result.valid_monitor_hit_count == 4
    assert summary.counts.frame_records == 4
    assert summary.counts.scene_frame_records == 4
    assert summary.counts.scene_frame_records == summary.counts.decoded_frames
    assert summary.source_artifacts == summary.artifact_validation.source_artifacts
    assert {
        "scene_manifest",
        "scene_summary",
        "scene_frames_jsonl",
        "viewer_index",
        "viewer_scene_data",
    }.issubset(summary.source_artifacts)
    scene_frame_lines = _jsonl(result.scene_frames_jsonl_path)
    assert [line["frame_index"] for line in scene_frame_lines] == [0, 1, 2, 3]
    viewer_data = json.loads(result.viewer_scene_data_path.read_text(encoding="utf-8"))
    assert viewer_data["schema_version"] == "gaze-scene-viewer-data-v1"
    assert viewer_data["frame_count"] == 4
    scene_summary = SceneSummary.model_validate_json(
        result.scene_summary_path.read_text(encoding="utf-8")
    )
    viewer_scene_data = ViewerSceneData.model_validate_json(
        result.viewer_scene_data_path.read_text(encoding="utf-8")
    )
    assert scene_summary.artifact_validation.viewer_exists is True
    assert viewer_scene_data.summary.artifact_validation.viewer_exists is True
    assert summary.final_status == "complete"
    assert summary.byte_counts.scene_jsonl_bytes == (
        result.scene_frames_jsonl_path.stat().st_size
    )
    assert summary.byte_counts.scene_bytes >= (
        result.scene_manifest_path.stat().st_size
        + result.scene_summary_path.stat().st_size
    )
    assert summary.byte_counts.viewer_bytes >= (
        result.viewer_index_path.stat().st_size
        + result.viewer_scene_data_path.stat().st_size
    )
    assert summary.byte_counts.total_run_bytes == sum(
        path.stat().st_size
        for path in result.layout.run_dir.rglob("*")
        if path.is_file()
    )


def test_model_free_observer_run_manifest_records_external_observer(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=1)

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            unigaze_device="mps",
            unigaze_batch_size=7,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["inference"] == {
        "schema_version": "inference-runtime-v1",
        "observer_source": "external_observer",
        "unigaze_model_id": None,
        "unigaze_device": "not_applicable",
        "unigaze_batch_size": None,
        "torch_version": None,
        "torch_mps_available": None,
        "mps_fallback_env": "not_applicable",
        "mps_fast_math_env": "not_applicable",
        "mps_prefer_metal_env": "not_applicable",
        "mps_preflight_passed": None,
    }


def test_default_mps_unavailable_fails_before_run_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "models"
    registry_path = tmp_path / "model_registry.json"
    make_tiny_video(video_path, frame_count=1)
    _write_model_registry_with_assets(models_root, registry_path)

    from chess_gaze import unigaze_runtime

    runtime_module = cast(Any, unigaze_runtime)
    monkeypatch.setattr(
        runtime_module.torch.backends.mps, "is_available", lambda: False
    )

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                models_root=models_root,
                model_registry_path=registry_path,
            )
        )

    assert exc_info.value.code is CliErrorCode.USAGE
    assert "MPS is unavailable" in str(exc_info.value)
    assert not output_root.exists()


@pytest.mark.parametrize(
    "env_name",
    [
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_MPS_PREFER_METAL",
    ],
)
def test_default_mps_rejects_unsafe_env_before_run_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "models"
    registry_path = tmp_path / "model_registry.json"
    make_tiny_video(video_path, frame_count=1)
    _write_model_registry_with_assets(models_root, registry_path)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_PREFER_METAL", raising=False)
    monkeypatch.setenv(env_name, "1")

    from chess_gaze import unigaze_runtime

    runtime_module = cast(Any, unigaze_runtime)
    monkeypatch.setattr(runtime_module.torch.backends.mps, "is_available", lambda: True)

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                models_root=models_root,
                model_registry_path=registry_path,
            )
        )

    assert exc_info.value.code is CliErrorCode.USAGE
    assert env_name in str(exc_info.value)
    assert not output_root.exists()


def test_default_model_observer_manifest_records_unigaze_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "models"
    registry_path = tmp_path / "model_registry.json"
    make_tiny_video(video_path, frame_count=1)
    _write_model_registry_with_assets(models_root, registry_path)

    from chess_gaze import pipeline
    from chess_gaze.frame_records import InferenceRuntimeRecord

    prepared_model = object()
    captured_gaze_models: list[object] = []

    def fake_prepare_unigaze_runtime(
        asset: object, *, device: str, batch_size: int, input_size_px: int
    ) -> object:
        del asset, input_size_px
        unigaze_device = cast(Any, device)
        return SimpleNamespace(
            model=prepared_model,
            inference=InferenceRuntimeRecord(
                observer_source="default_model_observer",
                unigaze_model_id="unigaze-h14-joint",
                unigaze_device=unigaze_device,
                unigaze_batch_size=batch_size,
                torch_version="test-torch",
                torch_mps_available=True,
                mps_fallback_env="unset",
                mps_fast_math_env="unset",
                mps_prefer_metal_env="unset",
                mps_preflight_passed=True,
            ),
        )

    def fake_default_observer_bundle_factory(
        resolved_assets: list[Any],
        calibration: object,
        run_layout: object,
        gaze_model: object,
        save_crop_images: bool,
    ) -> ObserverBundle:
        del resolved_assets, calibration, run_layout, save_crop_images
        captured_gaze_models.append(gaze_model)
        return ObserverBundle(frame_observer=_fake_record)

    monkeypatch.setattr(
        pipeline, "prepare_unigaze_runtime", fake_prepare_unigaze_runtime
    )
    monkeypatch.setattr(
        pipeline,
        "default_observer_bundle_factory",
        fake_default_observer_bundle_factory,
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=output_root,
            models_root=models_root,
            model_registry_path=registry_path,
        )
    )

    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["inference"]["observer_source"] == "default_model_observer"
    assert manifest["inference"]["unigaze_device"] == "mps"
    assert manifest["inference"]["unigaze_batch_size"] == 7
    assert captured_gaze_models == [prepared_model]


def test_explicit_cpu_batch_one_override_reaches_default_model_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "models"
    registry_path = tmp_path / "model_registry.json"
    make_tiny_video(video_path, frame_count=1)
    _write_model_registry_with_assets(models_root, registry_path)

    from chess_gaze import pipeline
    from chess_gaze.frame_records import InferenceRuntimeRecord

    captured_runtime_requests: list[tuple[str, int]] = []

    def fake_prepare_unigaze_runtime(
        asset: object, *, device: str, batch_size: int, input_size_px: int
    ) -> object:
        del asset, input_size_px
        captured_runtime_requests.append((device, batch_size))
        return SimpleNamespace(
            model=object(),
            inference=InferenceRuntimeRecord(
                observer_source="default_model_observer",
                unigaze_model_id="unigaze-h14-joint",
                unigaze_device=cast(Any, device),
                unigaze_batch_size=batch_size,
                torch_version="test-torch",
                torch_mps_available=True,
                mps_fallback_env="unset",
                mps_fast_math_env="unset",
                mps_prefer_metal_env="unset",
                mps_preflight_passed=None,
            ),
        )

    def fake_default_observer_bundle_factory(
        resolved_assets: list[Any],
        calibration: object,
        run_layout: object,
        gaze_model: object,
        save_crop_images: bool,
    ) -> ObserverBundle:
        del resolved_assets, calibration, run_layout, gaze_model, save_crop_images
        return ObserverBundle(frame_observer=_fake_record)

    monkeypatch.setattr(
        pipeline, "prepare_unigaze_runtime", fake_prepare_unigaze_runtime
    )
    monkeypatch.setattr(
        pipeline,
        "default_observer_bundle_factory",
        fake_default_observer_bundle_factory,
    )

    analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=output_root,
            models_root=models_root,
            model_registry_path=registry_path,
            unigaze_device="cpu",
            unigaze_batch_size=1,
        )
    )

    assert captured_runtime_requests == [("cpu", 1)]


def test_default_observer_bundle_factory_uses_prepared_gaze_model_and_batch_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chess_gaze import (
        face_observation,
        frame_observation,
        gaze_observation,
        pipeline,
    )
    from chess_gaze.calibration import default_calibration

    face_path = tmp_path / "mediapipe" / "face_landmarker.task"
    gaze_path = tmp_path / "unigaze" / "unigaze_h14_joint.safetensors"
    face_path.parent.mkdir(parents=True)
    gaze_path.parent.mkdir(parents=True)
    face_path.write_bytes(b"mediapipe")
    gaze_path.write_bytes(b"unigaze")
    prepared_model = object()
    captured: dict[str, object] = {}

    def fail_if_model_loaded(asset: object, *, device: str) -> object:
        del asset, device
        raise AssertionError("default factory must reuse prepared UniGaze model")

    class FakeFaceObserver:
        def __init__(self, *, model_asset_path: Path, calibration: object) -> None:
            captured["face_model_asset_path"] = model_asset_path
            captured["face_calibration"] = calibration

    class FakeModelBackedFrameObserver:
        def __init__(
            self,
            *,
            face_observer: object,
                gaze_model: object,
                calibration: object,
                run_layout: object,
                save_crop_images: bool,
            ) -> None:
                captured["face_observer"] = face_observer
                captured["gaze_model"] = gaze_model
                captured["calibration"] = calibration
                captured["run_layout"] = run_layout
                captured["save_crop_images"] = save_crop_images

        def __call__(self, frame: ObserverFrame) -> FrameRecord:
            return _fake_record(frame)

        def observe_batch(self, frames: list[ObserverFrame]) -> list[FrameRecord]:
            return [_fake_record(frame) for frame in frames]

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(
        gaze_observation.UniGazeModel,
        "from_local_asset",
        fail_if_model_loaded,
    )
    monkeypatch.setattr(face_observation, "MediaPipeFaceObserver", FakeFaceObserver)
    monkeypatch.setattr(
        frame_observation, "ModelBackedFrameObserver", FakeModelBackedFrameObserver
    )
    calibration = default_calibration()
    run_layout = cast(Any, object())
    resolved_assets = [
        ResolvedModelAsset(
            model_id="mediapipe-face-landmarker",
            task_name="face_landmarks",
            resolved_path=face_path,
            source_url="https://example.invalid/mediapipe",
            checksum_sha256=None,
            license="Google AI Edge Terms",
        ),
        ResolvedModelAsset(
            model_id="unigaze-h14-joint",
            task_name="gaze_estimation",
            resolved_path=gaze_path,
            source_url="https://example.invalid/unigaze",
            checksum_sha256=None,
            license="MG-NC-RAI-2.0",
        ),
    ]

    bundle = pipeline.default_observer_bundle_factory(
        resolved_assets, calibration, run_layout, prepared_model, False
    )

    assert captured["face_model_asset_path"] == face_path
    assert captured["gaze_model"] is prepared_model
    assert captured["save_crop_images"] is False
    assert bundle.frame_batch_observer is not None
    assert bundle.close is not None


def test_analyze_video_fails_when_scene_artifact_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=2)

    from chess_gaze import pipeline

    def build_then_corrupt_scene_artifacts(layout: object) -> object:
        scene_result = real_build_scene_artifacts(cast(Any, layout))
        scene_result.paths.scene_frames_jsonl_path.write_text(
            '{"frame_id":"f000000000","frame_index":99}\n',
            encoding="utf-8",
        )
        return scene_result

    monkeypatch.setattr(
        pipeline,
        "build_scene_artifacts",
        build_then_corrupt_scene_artifacts,
        raising=False,
    )

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
            observers=ObserverBundle(frame_observer=_fake_record),
        )

    assert exc_info.value.code is CliErrorCode.SCHEMA_VALIDATION_FAILED
    [run_dir] = (tmp_path / "output" / "tiny" / "runs").iterdir()
    summary = QASummary.model_validate_json(
        (run_dir / "qa_summary.json").read_text(encoding="utf-8")
    )
    assert summary.final_status == "failed"
    assert summary.artifact_validation.schema_validation_passed is False
    assert any(
        "Invalid scene frame record" in error
        for error in summary.artifact_validation.validation_errors
    )


def test_final_state_write_failure_does_not_leave_complete_qa_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=2)

    from chess_gaze import pipeline

    real_write_analysis_state = cast(Any, pipeline).write_analysis_state

    def fail_complete_state_write(layout: object, state: Any) -> Path:
        if state.status == "complete":
            raise RuntimeError("simulated final state write failure")
        return cast(Path, real_write_analysis_state(cast(Any, layout), state))

    monkeypatch.setattr(
        pipeline,
        "write_analysis_state",
        fail_complete_state_write,
    )

    with pytest.raises(RuntimeError, match="simulated final state write failure"):
        analyze_video(
            AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
            observers=ObserverBundle(frame_observer=_fake_record),
        )

    [run_dir] = (tmp_path / "output" / "tiny" / "runs").iterdir()
    assert not (run_dir / "qa_summary.json").exists()


def test_config_output_root_controls_run_layout_for_fake_observers(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    configured_output = tmp_path / "configured-output"
    config_path = tmp_path / "analysis.json"
    make_tiny_video(video_path, frame_count=1)
    config_path.write_text(
        json.dumps({"output_root": str(configured_output)}),
        encoding="utf-8",
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            config_path=config_path,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.layout.run_dir.is_relative_to(configured_output)


def test_config_models_root_controls_default_model_observer_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "configured-models"
    registry_path = tmp_path / "model_registry.json"
    config_path = tmp_path / "analysis.json"
    make_tiny_video(video_path, frame_count=1)
    _write_model_registry_with_assets(models_root, registry_path)
    config_path.write_text(
        json.dumps({"models_root": str(models_root), "output_root": str(output_root)}),
        encoding="utf-8",
    )
    captured_asset_paths: list[Path] = []
    captured_runtime_requests: list[tuple[str, int]] = []

    from chess_gaze import pipeline
    from chess_gaze.frame_records import InferenceRuntimeRecord

    def fake_default_observer_bundle_factory(
        resolved_assets: list[Any],
        calibration: object,
        run_layout: object,
        gaze_model: object,
        save_crop_images: bool,
    ) -> ObserverBundle:
        del calibration, run_layout, gaze_model, save_crop_images
        captured_asset_paths.extend(asset.resolved_path for asset in resolved_assets)
        return ObserverBundle(frame_observer=_fake_record)

    def fake_prepare_unigaze_runtime(
        asset: object, *, device: str, batch_size: int, input_size_px: int
    ) -> object:
        del asset, input_size_px
        unigaze_device = cast(Any, device)
        captured_runtime_requests.append((device, batch_size))
        return SimpleNamespace(
            model=object(),
            inference=InferenceRuntimeRecord(
                observer_source="default_model_observer",
                unigaze_model_id="unigaze-h14-joint",
                unigaze_device=unigaze_device,
                unigaze_batch_size=batch_size,
                torch_version="test-torch",
                torch_mps_available=True,
                mps_fallback_env="unset",
                mps_fast_math_env="unset",
                mps_prefer_metal_env="unset",
                mps_preflight_passed=True if device == "mps" else None,
            ),
        )

    monkeypatch.setattr(
        pipeline, "prepare_unigaze_runtime", fake_prepare_unigaze_runtime
    )
    monkeypatch.setattr(
        pipeline,
        "default_observer_bundle_factory",
        fake_default_observer_bundle_factory,
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            config_path=config_path,
            model_registry_path=registry_path,
        )
    )

    assert result.decoded_frame_count == 1
    assert sorted(path.relative_to(models_root) for path in captured_asset_paths) == [
        Path("mediapipe/face_landmarker.task"),
        Path("unigaze/unigaze_h14_joint.safetensors"),
    ]
    assert captured_runtime_requests == [("mps", 7)]
    assert result.layout.run_dir.is_relative_to(output_root)


@pytest.mark.parametrize(
    ("request_overrides", "expected_field"),
    [
        ({"unigaze_device": "cuda"}, "unigaze_device"),
        ({"unigaze_batch_size": 0}, "unigaze_batch_size"),
    ],
)
def test_invalid_runtime_request_overrides_fail_with_usage_before_io(
    tmp_path: Path,
    request_overrides: dict[str, Any],
    expected_field: str,
) -> None:
    output_root = tmp_path / "output"

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=tmp_path / "missing.mp4",
                output_root=output_root,
                **request_overrides,
            ),
            observers=ObserverBundle(frame_observer=_fake_record),
        )

    assert exc_info.value.code is CliErrorCode.USAGE
    assert expected_field in str(exc_info.value)
    assert not output_root.exists()


def test_no_face_fake_observer_records_face_not_found_and_processed_frames(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=2)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(
            frame_observer=lambda frame: _fake_record(frame, face_present=False)
        ),
    )

    records = _records_from(result.frames_jsonl_path)
    assert len(records) == 2
    assert list(result.layout.processed_frames_dir.glob("*.jpg")) == []
    assert all(record.status is FrameStatus.ERROR for record in records)
    assert all(
        record.face.reason_invalid is ErrorCode.FACE_NOT_FOUND for record in records
    )
    assert all(
        ErrorCode.FACE_NOT_FOUND in {error.code for error in record.errors}
        for record in records
    )


def test_one_eye_missing_fake_observer_preserves_the_other_eye(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=1)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(
            frame_observer=lambda frame: _fake_record(frame, right_eye_present=False)
        ),
    )

    [record] = _records_from(result.frames_jsonl_path)
    assert record.status is FrameStatus.ERROR
    assert record.left_eye.present is True
    assert record.left_eye.pupil_center == _point(38.0, 30.0)
    assert record.right_eye.present is False
    assert record.right_eye.reason_invalid is ErrorCode.RIGHT_EYE_NOT_FOUND


def test_missing_default_model_assets_fail_before_creating_run_layout(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=1)

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                models_root=tmp_path / "models",
            )
        )

    assert exc_info.value.code is CliErrorCode.MODEL_ASSET_MISSING
    assert not output_root.exists()


def test_raw_frame_write_failure_records_partial_status_and_error_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=3)

    from chess_gaze import pipeline

    real_raw_frame_writer = pipeline.raw_frame_writer

    def fail_second_raw_frame(path: Path, image: np.ndarray) -> str:
        if path.name == "f000000001.png":
            raise OSError("simulated raw write failure")
        return real_raw_frame_writer(path, image)

    monkeypatch.setattr(pipeline, "raw_frame_writer", fail_second_raw_frame)

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            save_frame_images=True,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    records = _records_from(result.frames_jsonl_path)
    assert len(records) == 3
    assert len(list(result.layout.raw_frames_dir.glob("*.png"))) == 2
    assert len(list(result.layout.processed_frames_dir.glob("*.jpg"))) == 3
    failed_record = records[1]
    assert failed_record.status is FrameStatus.ERROR
    assert ErrorCode.RAW_FRAME_WRITE_FAILED in {
        error.code for error in failed_record.errors
    }
    error_lines = _jsonl(result.errors_jsonl_path)
    assert any(
        line["frame_id"] == "f000000001"
        and line["code"] == ErrorCode.RAW_FRAME_WRITE_FAILED.value
        and "simulated raw write failure" in str(line["message"])
        for line in error_lines
    )


def test_processed_frame_write_failure_records_error_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=2)

    from chess_gaze import pipeline

    def fail_first_processed_frame(
        rgb_frame: np.ndarray,
        record: FrameRecord,
        output_path: Path,
        quality: int,
    ) -> str:
        del rgb_frame, record, quality
        if output_path.name == "f000000000.jpg":
            raise OSError("simulated processed write failure")
        output_path.write_bytes(b"jpg")
        return "digest"

    monkeypatch.setattr(
        pipeline,
        "render_processed_frame",
        fail_first_processed_frame,
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            save_frame_images=True,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    records = _records_from(result.frames_jsonl_path)
    assert records[0].status is FrameStatus.ERROR
    assert ErrorCode.PROCESSED_FRAME_WRITE_FAILED in {
        error.code for error in records[0].errors
    }
    error_lines = _jsonl(result.errors_jsonl_path)
    assert any(
        line["frame_id"] == "f000000000"
        and line["code"] == ErrorCode.PROCESSED_FRAME_WRITE_FAILED.value
        and "simulated processed write failure" in str(line["message"])
        for line in error_lines
    )


def test_malformed_errors_jsonl_fails_artifact_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=1)

    from chess_gaze import pipeline

    real_frame_error_writer = pipeline.frame_error_writer

    def append_malformed_error_json(errors_handle: TextIO, record: FrameRecord) -> None:
        real_frame_error_writer(errors_handle, record)
        errors_handle.write("{malformed-json\n")

    monkeypatch.setattr(pipeline, "frame_error_writer", append_malformed_error_json)

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
            observers=ObserverBundle(
                frame_observer=lambda frame: _fake_record(frame, face_present=False)
            ),
        )

    assert exc_info.value.code is CliErrorCode.SCHEMA_VALIDATION_FAILED
    [run_dir] = (tmp_path / "output" / "tiny" / "runs").iterdir()
    summary = QASummary.model_validate_json(
        (run_dir / "qa_summary.json").read_text(encoding="utf-8")
    )
    assert summary.final_status == "failed"
    assert summary.artifact_validation.schema_validation_passed is False
    assert summary.artifact_validation.counts_match is True


def test_invalid_utf8_errors_jsonl_writes_failed_qa_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=1)

    from chess_gaze import pipeline

    real_frame_error_writer = pipeline.frame_error_writer

    def append_invalid_utf8_error_jsonl(
        errors_handle: TextIO, record: FrameRecord
    ) -> None:
        real_frame_error_writer(errors_handle, record)
        errors_handle.flush()
        cast(Any, errors_handle).buffer.write(b"\xff\n")
        errors_handle.flush()

    monkeypatch.setattr(pipeline, "frame_error_writer", append_invalid_utf8_error_jsonl)

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
            observers=ObserverBundle(frame_observer=_fake_record),
        )

    assert exc_info.value.code is CliErrorCode.SCHEMA_VALIDATION_FAILED
    [run_dir] = (tmp_path / "output" / "tiny" / "runs").iterdir()
    summary = QASummary.model_validate_json(
        (run_dir / "qa_summary.json").read_text(encoding="utf-8")
    )
    assert summary.final_status == "failed"
    assert summary.status_transitions == [
        "created",
        "processing",
        "revalidating",
        "failed",
    ]
    assert summary.artifact_validation.schema_validation_passed is False
    assert summary.artifact_validation.counts_match is True
    assert any(
        CliErrorCode.SCHEMA_VALIDATION_FAILED.value in error
        for error in summary.artifact_validation.validation_errors
    )
