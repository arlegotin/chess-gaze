from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.video_decode import iter_decoded_frames
from chess_gaze.visualization import render_processed_frame

MANDATORY_VIDEO_PATHS = (
    Path("artifacts/input/nakamura_short.mp4"),
)


def _point(x: float, y: float) -> dict[str, Any]:
    return Point2D(space=CoordinateSpace.IMAGE_PX, x=x, y=y).model_dump()


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> dict[str, Any]:
    return BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    ).model_dump()


def _failure_payload(frame_id: str, frame_index: int) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "frame_index": frame_index,
        "status": "ERROR",
        "timestamp_seconds": 0.0,
        "face": {
            "present": False,
            "bounding_box": None,
            "landmarks": None,
            "reason_invalid": "FACE_NOT_FOUND",
        },
        "left_eye": {
            "present": False,
            "bounding_box": None,
            "pupil_center": None,
            "iris_landmarks": None,
            "reason_invalid": "LEFT_EYE_NOT_FOUND",
        },
        "right_eye": {
            "present": False,
            "bounding_box": None,
            "pupil_center": None,
            "iris_landmarks": None,
            "reason_invalid": "RIGHT_EYE_NOT_FOUND",
        },
        "head_pose": {
            "valid": False,
            "yaw_radians": None,
            "pitch_radians": None,
            "roll_radians": None,
            "reason_invalid": "HEAD_POSE_INVALID",
        },
        "geometric_gaze": {
            "valid": False,
            "yaw_radians": None,
            "pitch_radians": None,
            "reason_invalid": "GAZE_ESTIMATORS_DISAGREE",
        },
        "appearance_gaze": {
            "valid": False,
            "yaw_radians": None,
            "pitch_radians": None,
            "reason_invalid": "GAZE_MODEL_FAILED",
        },
        "recommended_gaze": {
            "valid": False,
            "yaw_radians": None,
            "pitch_radians": None,
            "reason_invalid": "GAZE_ESTIMATORS_DISAGREE",
        },
        "errors": [
            {
                "code": "FACE_NOT_FOUND",
                "message": "No face detected in frame.",
            }
        ],
    }


def _valid_record(
    frame_id: str, frame_index: int, width: int, height: int
) -> FrameRecord:
    face_left = width * 0.28
    face_top = height * 0.18
    face_right = width * 0.72
    face_bottom = height * 0.78
    left_x = width * 0.58
    right_x = width * 0.42
    eye_y = height * 0.42
    iris_radius = max(3.0, min(width, height) * 0.018)

    payload = _failure_payload(frame_id, frame_index)
    payload["status"] = "OK"
    payload["face"] = {
        "present": True,
        "bounding_box": _box(face_left, face_top, face_right, face_bottom),
        "landmarks": [
            _point(left_x, eye_y),
            _point(right_x, eye_y),
            _point(width * 0.50, height * 0.55),
            _point(width * 0.43, height * 0.68),
            _point(width * 0.57, height * 0.68),
        ],
        "reason_invalid": None,
    }
    payload["left_eye"] = {
        "present": True,
        "bounding_box": _box(left_x - 18.0, eye_y - 12.0, left_x + 18.0, eye_y + 12.0),
        "pupil_center": _point(left_x, eye_y),
        "iris_landmarks": [
            _point(left_x - iris_radius, eye_y),
            _point(left_x + iris_radius, eye_y),
            _point(left_x, eye_y - iris_radius),
            _point(left_x, eye_y + iris_radius),
        ],
        "reason_invalid": None,
    }
    payload["right_eye"] = {
        "present": True,
        "bounding_box": _box(
            right_x - 18.0, eye_y - 12.0, right_x + 18.0, eye_y + 12.0
        ),
        "pupil_center": _point(right_x, eye_y),
        "iris_landmarks": [
            _point(right_x - iris_radius, eye_y),
            _point(right_x + iris_radius, eye_y),
            _point(right_x, eye_y - iris_radius),
            _point(right_x, eye_y + iris_radius),
        ],
        "reason_invalid": None,
    }
    payload["head_pose"] = {
        "valid": True,
        "yaw_radians": 0.08,
        "pitch_radians": -0.04,
        "roll_radians": 0.02,
        "reason_invalid": None,
    }
    payload["geometric_gaze"] = {
        "valid": True,
        "yaw_radians": 0.05,
        "pitch_radians": -0.03,
        "reason_invalid": None,
    }
    payload["appearance_gaze"] = {
        "valid": True,
        "yaw_radians": 0.04,
        "pitch_radians": -0.02,
        "reason_invalid": None,
    }
    payload["recommended_gaze"] = {
        "valid": True,
        "yaw_radians": 0.045,
        "pitch_radians": -0.025,
        "reason_invalid": None,
    }
    payload["errors"] = []
    return FrameRecord.model_validate(payload)


def _failure_record(frame_id: str, frame_index: int) -> FrameRecord:
    return FrameRecord.model_validate(_failure_payload(frame_id, frame_index))


def _jpeg_shape(path: Path) -> tuple[int, int, int]:
    image = np.asarray(Image.open(path).convert("RGB"))
    height, width, channels = image.shape
    return int(height), int(width), int(channels)


def test_visualization_renders_deterministic_frames_from_mandatory_videos(
    tmp_path: Path,
) -> None:
    missing_paths = [path for path in MANDATORY_VIDEO_PATHS if not path.is_file()]
    assert not missing_paths, f"missing mandatory real-data video(s): {missing_paths}"

    for video_path in MANDATORY_VIDEO_PATHS:
        decoded = next(iter_decoded_frames(video_path))
        frame_before = decoded.rgb.copy()
        height, width = decoded.rgb.shape[:2]
        records = (
            (_valid_record(decoded.frame_id, decoded.frame_index, width, height), 100),
            (_failure_record(decoded.frame_id, decoded.frame_index), 1),
        )

        for record, quality in records:
            record_before = record.model_dump(mode="json")
            output_path = (
                tmp_path / f"{video_path.stem}-{record.status.value}-q{quality}.jpg"
            )

            render_processed_frame(decoded.rgb, record, output_path, quality=quality)

            assert output_path.stat().st_size > 0
            assert _jpeg_shape(output_path) == decoded.rgb.shape
            assert np.array_equal(decoded.rgb, frame_before)
            assert record.model_dump(mode="json") == record_before
