from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from chess_gaze.errors import ErrorCode
from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.visualization import render_processed_frame


def _failure_payload(frame_id: str = "f000000001") -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "frame_index": 1,
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


def _observed_record() -> FrameRecord:
    payload = _failure_payload()
    payload["status"] = "OK"
    payload["face"] = {
        "present": True,
        "bounding_box": _box(40.0, 28.0, 170.0, 140.0),
        "landmarks": [
            _point(72.0, 58.0),
            _point(138.0, 58.0),
            _point(105.0, 102.0),
            _point(82.0, 122.0),
            _point(130.0, 122.0),
        ],
        "reason_invalid": None,
    }
    payload["left_eye"] = {
        "present": True,
        "bounding_box": _box(65.0, 55.0, 94.0, 80.0),
        "pupil_center": _point(80.0, 67.0),
        "iris_landmarks": [
            _point(74.0, 67.0),
            _point(86.0, 67.0),
            _point(80.0, 61.0),
            _point(80.0, 73.0),
        ],
        "reason_invalid": None,
    }
    payload["right_eye"] = {
        "present": True,
        "bounding_box": _box(120.0, 55.0, 149.0, 80.0),
        "pupil_center": _point(135.0, 68.0),
        "iris_landmarks": [
            _point(128.0, 68.0),
            _point(142.0, 68.0),
            _point(135.0, 61.0),
            _point(135.0, 75.0),
        ],
        "reason_invalid": None,
    }
    payload["head_pose"] = {
        "valid": True,
        "yaw_radians": 0.18,
        "pitch_radians": -0.08,
        "roll_radians": 0.04,
        "reason_invalid": None,
    }
    payload["geometric_gaze"] = {
        "valid": True,
        "yaw_radians": 0.12,
        "pitch_radians": -0.05,
        "reason_invalid": None,
    }
    payload["appearance_gaze"] = {
        "valid": True,
        "yaw_radians": 0.1,
        "pitch_radians": -0.04,
        "reason_invalid": None,
    }
    payload["recommended_gaze"] = {
        "valid": True,
        "yaw_radians": 0.11,
        "pitch_radians": -0.05,
        "reason_invalid": None,
    }
    payload["errors"] = []
    return FrameRecord.model_validate(payload)


def _rgb_jpeg(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _nonzero_near(image: np.ndarray, *, x: int, y: int, radius: int = 4) -> int:
    y_min = max(0, y - radius)
    y_max = min(image.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(image.shape[1], x + radius + 1)
    return int(np.count_nonzero(image[y_min:y_max, x_min:x_max]))


def test_render_processed_frame_writes_ok_jpeg_with_current_schema_overlays(
    tmp_path: Path,
) -> None:
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    record = _observed_record()
    record_before = record.model_dump(mode="json")
    output_path = tmp_path / "processed.jpg"

    digest = render_processed_frame(frame, record, output_path, quality=100)

    assert output_path.is_file()
    assert digest == sha256(output_path.read_bytes()).hexdigest()
    rendered = _rgb_jpeg(output_path)
    assert rendered.shape == frame.shape
    assert _nonzero_near(rendered, x=40, y=28) > 0
    assert _nonzero_near(rendered, x=80, y=67) > 0
    assert _nonzero_near(rendered, x=135, y=68) > 0
    assert _nonzero_near(rendered, x=105, y=102) > 0
    assert record.model_dump(mode="json") == record_before
    assert str(output_path) not in str(record.model_dump(mode="json"))


def test_render_processed_frame_writes_failure_jpeg_with_status_and_error_text(
    tmp_path: Path,
) -> None:
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    record = FrameRecord.model_validate(_failure_payload())
    output_path = tmp_path / "face-not-found.jpg"

    digest = render_processed_frame(frame, record, output_path, quality=95)

    assert output_path.is_file()
    assert digest == sha256(output_path.read_bytes()).hexdigest()
    rendered = _rgb_jpeg(output_path)
    assert rendered.shape == frame.shape
    assert int(np.count_nonzero(rendered[:45, :175])) > 0


def test_left_and_right_iris_centers_are_rendered_independently(
    tmp_path: Path,
) -> None:
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    record = _observed_record()
    output_path = tmp_path / "eyes.jpg"

    render_processed_frame(frame, record, output_path, quality=100)

    rendered = _rgb_jpeg(output_path)
    assert _nonzero_near(rendered, x=80, y=67, radius=2) > 0
    assert _nonzero_near(rendered, x=135, y=68, radius=2) > 0


def test_current_frame_record_rejects_unvalidated_candidate_overlay_fields() -> None:
    payload = _observed_record().model_dump(mode="json")
    payload["alternate_face_candidates"] = [
        {"bounding_box": _box(10.0, 10.0, 40.0, 60.0), "score": 0.31}
    ]

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(payload)


def test_render_processed_frame_rejects_jpeg_quality_outside_opencv_bounds(
    tmp_path: Path,
) -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    record = FrameRecord.model_validate(_failure_payload())

    with pytest.raises(ValueError, match="quality"):
        render_processed_frame(frame, record, tmp_path / "bad.jpg", quality=0)


def test_multiple_face_candidate_status_is_drawn_from_current_error_schema(
    tmp_path: Path,
) -> None:
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    record_payload = _observed_record().model_dump()
    record_payload["errors"] = [
        {
            "code": ErrorCode.MULTIPLE_FACE_CANDIDATES.value,
            "message": "Selected largest usable face; alternates unavailable.",
        }
    ]
    record = FrameRecord.model_validate(record_payload)
    output_path = tmp_path / "multiple-candidates.jpg"

    render_processed_frame(frame, record, output_path, quality=95)

    rendered = _rgb_jpeg(output_path)
    assert int(np.count_nonzero(rendered[:45, :210])) > 0
