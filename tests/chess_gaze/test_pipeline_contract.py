from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import av
import numpy as np
import pytest

from chess_gaze.errors import CliErrorCode, ErrorCode, FrameStatus
from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.pipeline import (
    AnalyzeRequest,
    ObserverBundle,
    ObserverFrame,
    PipelineError,
    analyze_video,
)


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


def test_analyze_video_writes_one_artifact_set_per_decoded_frame(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=4)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.decoded_frame_count == 4
    assert len(list(result.layout.raw_frames_dir.glob("*.png"))) == 4
    assert len(list(result.layout.processed_frames_dir.glob("*.jpg"))) == 4
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
    assert not (result.layout.run_dir / "qa_summary.json").exists()


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


def test_config_models_root_controls_default_model_asset_gate(
    tmp_path: Path,
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

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                config_path=config_path,
                model_registry_path=registry_path,
            )
        )

    assert exc_info.value.code is CliErrorCode.PIPELINE_NOT_IMPLEMENTED
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
    assert len(list(result.layout.processed_frames_dir.glob("*.jpg"))) == 2
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
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
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
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
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
