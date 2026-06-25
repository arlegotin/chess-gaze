from __future__ import annotations

from typing import Any

from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D


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


def _observed_frame_record(frame_id: str) -> FrameRecord:
    face_box = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=100.0,
        y_min=50.0,
        x_max=180.0,
        y_max=170.0,
    )
    left_eye_box = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=115.0,
        y_min=90.0,
        x_max=140.0,
        y_max=115.0,
    )
    right_eye_box = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=145.0,
        y_min=90.0,
        x_max=170.0,
        y_max=115.0,
    )
    iris_landmarks_left = [
        Point2D(space=CoordinateSpace.IMAGE_PX, x=120.0, y=100.0),
        Point2D(space=CoordinateSpace.IMAGE_PX, x=130.0, y=100.0),
        Point2D(space=CoordinateSpace.IMAGE_PX, x=125.0, y=95.0),
        Point2D(space=CoordinateSpace.IMAGE_PX, x=125.0, y=105.0),
    ]
    iris_landmarks_right = [
        Point2D(space=CoordinateSpace.IMAGE_PX, x=150.0, y=100.0),
        Point2D(space=CoordinateSpace.IMAGE_PX, x=160.0, y=100.0),
        Point2D(space=CoordinateSpace.IMAGE_PX, x=155.0, y=95.0),
        Point2D(space=CoordinateSpace.IMAGE_PX, x=155.0, y=105.0),
    ]

    payload = _frame_record_payload(frame_id)
    payload["status"] = "OK"
    payload["face"] = {
        "present": True,
        "bounding_box": face_box.model_dump(),
        "landmarks": [
            Point2D(space=CoordinateSpace.IMAGE_PX, x=110.0, y=70.0).model_dump(),
            Point2D(space=CoordinateSpace.IMAGE_PX, x=170.0, y=70.0).model_dump(),
            Point2D(space=CoordinateSpace.IMAGE_PX, x=140.0, y=155.0).model_dump(),
        ],
        "reason_invalid": None,
    }
    payload["left_eye"] = {
        "present": True,
        "bounding_box": left_eye_box.model_dump(),
        "pupil_center": Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=125.0,
            y=100.0,
        ).model_dump(),
        "iris_landmarks": [point.model_dump() for point in iris_landmarks_left],
        "reason_invalid": None,
    }
    payload["right_eye"] = {
        "present": True,
        "bounding_box": right_eye_box.model_dump(),
        "pupil_center": Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=155.0,
            y=100.0,
        ).model_dump(),
        "iris_landmarks": [point.model_dump() for point in iris_landmarks_right],
        "reason_invalid": None,
    }
    payload["errors"] = []
    return FrameRecord.model_validate(payload)


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
    assert derived.mirror_policy.value == "unknown"
    assert derived.mirror_policy.contributing_frame_count == 0
    assert derived.mirror_policy.usage == "future_use"


def test_derive_setup_constants_returns_provenance_records() -> None:
    from chess_gaze.calibration import derive_setup_constants

    derived = derive_setup_constants([_observed_frame_record("f000000003")])

    assert derived.selected_face_bbox_size_image_px.value == {
        "median_width_px": 80.0,
        "median_height_px": 120.0,
        "median_area_px2": 9600.0,
    }
    assert derived.selected_face_bbox_size_image_px.unit == "image_px"
    assert derived.selected_face_bbox_size_image_px.coordinate_space == "image_px"
    assert (
        derived.selected_face_bbox_size_image_px.derivation_method
        == "median selected-face bounding box width, height, and area from "
        "face.bounding_box where face.present is true"
    )
    assert derived.selected_face_bbox_size_image_px.contributing_frame_count == 1
    assert derived.selected_face_bbox_size_image_px.uncertainty == "low"
    assert derived.selected_face_bbox_size_image_px.usage == "measurement"

    assert derived.inter_pupil_distance_image_px.value == 30.0
    assert derived.inter_pupil_distance_image_px.unit == "image_px"
    assert derived.inter_pupil_distance_image_px.coordinate_space == "image_px"
    assert (
        derived.inter_pupil_distance_image_px.derivation_method
        == "median Euclidean distance between left and right pupil centers "
        "when both eyes are present"
    )
    assert derived.inter_pupil_distance_image_px.contributing_frame_count == 1
    assert derived.inter_pupil_distance_image_px.uncertainty == "low"
    assert derived.inter_pupil_distance_image_px.usage == "measurement"

    assert derived.left_iris_diameter_image_px.value == 10.0
    assert derived.left_iris_diameter_image_px.unit == "image_px"
    assert derived.left_iris_diameter_image_px.coordinate_space == "image_px"
    assert (
        derived.left_iris_diameter_image_px.derivation_method
        == "median maximum pairwise iris landmark distance per frame for the left eye"
    )
    assert derived.left_iris_diameter_image_px.contributing_frame_count == 1
    assert derived.left_iris_diameter_image_px.uncertainty == "medium"
    assert derived.left_iris_diameter_image_px.usage == "measurement"

    assert derived.right_iris_diameter_image_px.value == 10.0
    assert derived.facecam_roi_image_px.value == {
        "x_min": 100.0,
        "y_min": 50.0,
        "x_max": 180.0,
        "y_max": 170.0,
    }
    assert derived.facecam_roi_image_px.unit == "image_px"
    assert derived.facecam_roi_image_px.coordinate_space == "image_px"
    assert (
        derived.facecam_roi_image_px.derivation_method
        == "bounding box union over observed selected-face boxes; derived ROI "
        "for QA only"
    )
    assert derived.facecam_roi_image_px.contributing_frame_count == 1
    assert derived.facecam_roi_image_px.uncertainty == "medium"
    assert derived.facecam_roi_image_px.usage == "qa_only"

    assert (
        derived.estimated_camera_intrinsics_policy.value
        == "estimate_with_explicit_uncertainty"
    )
    assert derived.estimated_camera_intrinsics_policy.unit is None
    assert derived.estimated_camera_intrinsics_policy.coordinate_space is None
    assert (
        derived.estimated_camera_intrinsics_policy.derivation_method
        == "policy placeholder derived from calibration defaults; does not "
        "authorize metric camera_3d_m translation"
    )
    assert derived.estimated_camera_intrinsics_policy.contributing_frame_count == 0
    assert derived.estimated_camera_intrinsics_policy.uncertainty == "high"
    assert derived.estimated_camera_intrinsics_policy.usage == "future_use"


def test_derive_setup_constants_returns_null_provenance_without_evidence() -> None:
    from chess_gaze.calibration import derive_setup_constants

    derived = derive_setup_constants(
        [FrameRecord.model_validate(_frame_record_payload("f000000004"))]
    )

    assert derived.selected_face_bbox_size_image_px.value is None
    assert derived.selected_face_bbox_size_image_px.contributing_frame_count == 0
    assert derived.inter_pupil_distance_image_px.value is None
    assert derived.inter_pupil_distance_image_px.contributing_frame_count == 0
    assert derived.left_iris_diameter_image_px.value is None
    assert derived.left_iris_diameter_image_px.contributing_frame_count == 0
    assert derived.right_iris_diameter_image_px.value is None
    assert derived.right_iris_diameter_image_px.contributing_frame_count == 0
    assert derived.facecam_roi_image_px.value is None
    assert derived.facecam_roi_image_px.contributing_frame_count == 0
    assert derived.mirror_policy.value == "unknown"
    assert derived.mirror_policy.contributing_frame_count == 0
