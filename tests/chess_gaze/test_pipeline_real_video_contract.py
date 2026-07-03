from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.pipeline import (
    AnalyzeRequest,
    AnalyzeResult,
    ObserverBundle,
    ObserverFrame,
    analyze_video,
)

NAKAMURA_SHORT_VIDEO = Path("artifacts/input/nakamura_short.mp4")
NAKAMURA_SHORT_FRAME_COUNT = 180


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


def _deterministic_real_video_record(frame: ObserverFrame) -> FrameRecord:
    width = float(frame.rgb.shape[1])
    height = float(frame.rgb.shape[0])
    x_shift = float(frame.frame_index % 17)
    y_shift = float(frame.frame_index % 11)
    face_box = _box(
        width * 0.25 + x_shift,
        height * 0.20 + y_shift,
        width * 0.70 + x_shift,
        height * 0.82 + y_shift,
    )
    left_pupil = _point(width * 0.55 + x_shift, height * 0.42 + y_shift)
    right_pupil = _point(width * 0.40 + x_shift, height * 0.42 + y_shift)
    left_eye = {
        "present": True,
        "bounding_box": _box(
            left_pupil.x - 12.0,
            left_pupil.y - 8.0,
            left_pupil.x + 12.0,
            left_pupil.y + 8.0,
        ),
        "pupil_center": left_pupil,
        "iris_landmarks": [
            _point(left_pupil.x - 4.0, left_pupil.y),
            _point(left_pupil.x + 4.0, left_pupil.y),
            _point(left_pupil.x, left_pupil.y - 4.0),
            _point(left_pupil.x, left_pupil.y + 4.0),
        ],
        "reason_invalid": None,
    }
    right_eye = {
        "present": True,
        "bounding_box": _box(
            right_pupil.x - 12.0,
            right_pupil.y - 8.0,
            right_pupil.x + 12.0,
            right_pupil.y + 8.0,
        ),
        "pupil_center": right_pupil,
        "iris_landmarks": [
            _point(right_pupil.x - 4.0, right_pupil.y),
            _point(right_pupil.x + 4.0, right_pupil.y),
            _point(right_pupil.x, right_pupil.y - 4.0),
            _point(right_pupil.x, right_pupil.y + 4.0),
        ],
        "reason_invalid": None,
    }
    yaw = (frame.frame_index % 31) / 300.0
    pitch = -((frame.frame_index % 29) / 300.0)

    return FrameRecord.model_validate(
        {
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "status": FrameStatus.OK,
            "timestamp_seconds": frame.timestamp_seconds,
            "face": {
                "present": True,
                "bounding_box": face_box,
                "landmarks": [
                    _point(width * 0.55 + x_shift, height * 0.38 + y_shift),
                    _point(width * 0.40 + x_shift, height * 0.38 + y_shift),
                    _point(width * 0.48 + x_shift, height * 0.54 + y_shift),
                ],
                "reason_invalid": None,
            },
            "left_eye": left_eye,
            "right_eye": right_eye,
            "head_pose": {
                "valid": True,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "roll_radians": 0.0,
                "reason_invalid": None,
            },
            "geometric_gaze": {
                "valid": True,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "reason_invalid": None,
            },
            "appearance_gaze": {
                "valid": True,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "reason_invalid": None,
            },
            "recommended_gaze": {
                "valid": True,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "reason_invalid": None,
            },
            "errors": [],
        }
    )


def _records(path: Path) -> list[FrameRecord]:
    return [
        FrameRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _frame_record_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _assert_default_completed_artifact_contract(
    result: AnalyzeResult, *, expected_count: int
) -> list[str]:
    record_lines = _frame_record_lines(result.frames_jsonl_path)

    assert result.decoded_frame_count == expected_count
    assert result.qa_summary_path is None
    assert result.validated_record_count is None
    assert result.validated_error_count is None
    assert not (result.layout.run_dir / "qa_summary.json").exists()
    assert len(record_lines) == expected_count
    first_record = json.loads(record_lines[0])
    last_record = json.loads(record_lines[-1])
    assert first_record["frame_id"] == "f000000000"
    assert first_record["frame_index"] == 0
    assert last_record["frame_id"] == f"f{expected_count - 1:09d}"
    assert last_record["frame_index"] == expected_count - 1
    assert first_record["status"] == FrameStatus.OK.value
    assert last_record["status"] == FrameStatus.OK.value
    assert result.scene_manifest_path.is_file()
    assert result.scene_summary_path.is_file()
    assert result.scene_frames_jsonl_path.is_file()
    assert result.viewer_index_path.is_file()
    assert result.viewer_scene_data_path.is_file()
    state = json.loads(result.analysis_state_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["next_frame_index"] == expected_count
    return record_lines


@pytest.mark.parametrize(
    ("video_path", "expected_count"),
    [(NAKAMURA_SHORT_VIDEO, NAKAMURA_SHORT_FRAME_COUNT)],
)
def test_real_video_model_free_pipeline_writes_complete_artifact_contract(
    tmp_path: Path, video_path: Path, expected_count: int
) -> None:
    assert video_path.is_file(), f"missing mandatory real-data video: {video_path}"

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_deterministic_real_video_record),
    )
    raw_count = len(list(result.layout.raw_frames_dir.glob("*.png")))
    processed_count = len(list(result.layout.processed_frames_dir.glob("*.jpg")))
    crop_count = len(list(result.layout.crops_dir.rglob("*.png")))
    record_lines = _assert_default_completed_artifact_contract(
        result, expected_count=expected_count
    )
    print(
        f"{video_path}: decoded={result.decoded_frame_count} "
        f"raw={raw_count} processed={processed_count} crops={crop_count} "
        f"records={len(record_lines)}"
    )

    assert raw_count == 0
    assert processed_count == 0
    assert crop_count == 0
    assert not result.layout.crops_dir.exists()
    records = [FrameRecord.model_validate_json(line) for line in record_lines[:5]]
    assert all(record.status is FrameStatus.OK for record in records)
    assert all(
        ErrorCode.FACE_NOT_FOUND not in {error.code for error in record.errors}
        for record in records
    )


def test_default_completed_artifact_contract_avoids_full_record_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert NAKAMURA_SHORT_VIDEO.is_file(), (
        f"missing mandatory real-data video: {NAKAMURA_SHORT_VIDEO}"
    )

    result = analyze_video(
        AnalyzeRequest(video_path=NAKAMURA_SHORT_VIDEO, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_deterministic_real_video_record),
    )

    original_validator = FrameRecord.model_validate_json
    validated_lines: list[str] = []

    def _track_validation(line: str, *args: object, **kwargs: object) -> FrameRecord:
        validated_lines.append(line)
        return original_validator(line, *args, **kwargs)

    monkeypatch.setattr(FrameRecord, "model_validate_json", _track_validation)

    record_lines = _assert_default_completed_artifact_contract(
        result, expected_count=NAKAMURA_SHORT_FRAME_COUNT
    )

    assert len(record_lines) == NAKAMURA_SHORT_FRAME_COUNT
    assert validated_lines == []


@pytest.mark.native_mediapipe
def test_nakamura_short_default_model_pipeline_does_not_create_crop_directory(
    tmp_path: Path,
) -> None:
    assert NAKAMURA_SHORT_VIDEO.is_file(), (
        f"missing mandatory real-data video: {NAKAMURA_SHORT_VIDEO}"
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=NAKAMURA_SHORT_VIDEO,
            output_root=tmp_path / "output",
            unigaze_device="cpu",
            unigaze_batch_size=7,
        )
    )
    _assert_default_completed_artifact_contract(
        result, expected_count=NAKAMURA_SHORT_FRAME_COUNT
    )

    assert list(result.layout.raw_frames_dir.glob("*.png")) == []
    assert list(result.layout.processed_frames_dir.glob("*.jpg")) == []
    assert not result.layout.crops_dir.exists()
    assert list(result.layout.crops_dir.rglob("*.png")) == []


@pytest.mark.native_mediapipe
def test_nakamura_short_save_crops_retains_crop_images_only(
    tmp_path: Path,
) -> None:
    assert NAKAMURA_SHORT_VIDEO.is_file(), (
        f"missing mandatory real-data video: {NAKAMURA_SHORT_VIDEO}"
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=NAKAMURA_SHORT_VIDEO,
            output_root=tmp_path / "output",
            unigaze_device="cpu",
            unigaze_batch_size=7,
            save_crop_images=True,
        )
    )
    _assert_default_completed_artifact_contract(
        result, expected_count=NAKAMURA_SHORT_FRAME_COUNT
    )
    crop_paths = list(result.layout.crops_dir.rglob("*.png"))

    assert list(result.layout.raw_frames_dir.glob("*.png")) == []
    assert list(result.layout.processed_frames_dir.glob("*.jpg")) == []
    assert result.layout.crops_dir.is_dir()
    assert len(crop_paths) > 0
