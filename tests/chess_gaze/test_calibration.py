from __future__ import annotations

from copy import deepcopy
from typing import Any

from chess_gaze.frame_records import FrameRecord


def _frame_record_payload(frame_id: str) -> dict[str, Any]:
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


def test_default_calibration_persists_named_constants() -> None:
    from chess_gaze.calibration import default_calibration

    calibration = default_calibration()

    assert calibration.model_dump() == {
        "raw_frame_image_format": "png",
        "processed_frame_image_format": "jpg",
        "processed_frame_jpeg_quality": 95,
        "max_face_candidates": 4,
        "candidate_face_score_min": 0.25,
        "usable_face_score_min": 0.5,
        "usable_eye_confidence_min": 0.5,
        "default_iris_diameter_mm": 11.7,
        "default_iris_diameter_uncertainty_mm": 0.5,
        "unigaze_input_size_px": 224,
        "unigaze_output_order": "pitch_yaw_radians",
        "face_landmarker_running_mode": "IMAGE",
        "camera_intrinsics_policy": "estimate_with_explicit_uncertainty",
        "metric_translation_allowed": False,
        "pnp_landmark_indices": {
            "nose_tip": 1,
            "chin": 152,
            "left_eye_outer": 33,
            "right_eye_outer": 263,
            "left_eye_inner": 133,
            "right_eye_inner": 362,
            "left_mouth_corner": 61,
            "right_mouth_corner": 291,
        },
    }


def test_derive_setup_constants_does_not_rewrite_frame_fields() -> None:
    from chess_gaze.calibration import derive_setup_constants

    records = [
        FrameRecord.model_validate(_frame_record_payload("f000000001")),
        FrameRecord.model_validate(_frame_record_payload("f000000002")),
    ]
    before = [record.model_dump() for record in records]

    derived = derive_setup_constants(records)

    assert [record.model_dump() for record in records] == before
    assert derived is not None
    assert derived.mirror_policy == "unknown"
    assert derived.measurement_usage == "qa_and_future_reconstruction_only"
