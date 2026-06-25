import math
from typing import Any

import pytest
from pydantic import ValidationError

from chess_gaze.errors import ErrorCode
from chess_gaze.frame_records import FrameRecord, GazeAngles
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D


@pytest.fixture
def valid_frame_record_dict() -> dict[str, Any]:
    return {
        "frame_id": "f000000001",
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


def test_bbox_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValidationError):
        BBox(space=CoordinateSpace.IMAGE_PX, x_min=20, y_min=10, x_max=10, y_max=40)


def test_point_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        Point2D(space=CoordinateSpace.IMAGE_PX, x=math.nan, y=1.0)


def test_gaze_valid_requires_pitch_and_yaw() -> None:
    with pytest.raises(ValidationError):
        GazeAngles(valid=True, yaw_radians=None, pitch_radians=0.1, reason_invalid=None)


def test_gaze_angles_reject_enum_strings_in_direct_validation() -> None:
    invalid_reason: Any = "GAZE_MODEL_FAILED"

    with pytest.raises(ValidationError):
        GazeAngles(
            valid=False,
            yaw_radians=None,
            pitch_radians=None,
            reason_invalid=invalid_reason,
        )


def test_frame_record_accepts_valid_artifact_payload(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    record = FrameRecord.model_validate(valid_frame_record_dict)

    assert record.status.value == "ERROR"
    assert record.face.reason_invalid == ErrorCode.FACE_NOT_FOUND
    assert record.left_eye.reason_invalid == ErrorCode.LEFT_EYE_NOT_FOUND


def test_frame_record_rejects_unknown_fields(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["unknown"] = "rejected"

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_near_miss_enum_strings(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["status"] = "NOT_A_STATUS"
    valid_frame_record_dict["recommended_gaze"]["reason_invalid"] = "GAZE_MODEL_FAILED "

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_invalid_nested_enum_string(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["recommended_gaze"]["reason_invalid"] = "NOT_A_REASON"

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_present_face_without_landmarks(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["face"]["present"] = True
    valid_frame_record_dict["face"]["reason_invalid"] = None

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_present_eye_without_landmarks(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["left_eye"]["present"] = True
    valid_frame_record_dict["left_eye"]["reason_invalid"] = None

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_infinite_head_pose_angle(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["head_pose"]["yaw_radians"] = math.inf

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_error_code_names_are_stable() -> None:
    assert ErrorCode.FACE_NOT_FOUND.value == "FACE_NOT_FOUND"
    assert ErrorCode.GAZE_ESTIMATORS_DISAGREE.value == "GAZE_ESTIMATORS_DISAGREE"
