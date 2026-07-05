from __future__ import annotations

import inspect
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


def _observed_frame_record(
    frame_id: str,
    *,
    face_x_min: float = 100.0,
    face_y_min: float = 50.0,
    face_width: float = 80.0,
    face_height: float = 120.0,
    left_pupil_x: float = 155.0,
    left_pupil_y: float = 100.0,
    right_pupil_x: float = 125.0,
    right_pupil_y: float = 100.0,
    left_iris_diameter: float = 10.0,
    right_iris_diameter: float = 10.0,
) -> FrameRecord:
    face_box = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=face_x_min,
        y_min=face_y_min,
        x_max=face_x_min + face_width,
        y_max=face_y_min + face_height,
    )
    left_eye_box = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=left_pupil_x - 10.0,
        y_min=left_pupil_y - 10.0,
        x_max=left_pupil_x + 15.0,
        y_max=left_pupil_y + 15.0,
    )
    right_eye_box = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=right_pupil_x - 10.0,
        y_min=right_pupil_y - 10.0,
        x_max=right_pupil_x + 15.0,
        y_max=right_pupil_y + 15.0,
    )
    iris_landmarks_left = [
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=left_pupil_x - (left_iris_diameter / 2.0),
            y=left_pupil_y,
        ),
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=left_pupil_x + (left_iris_diameter / 2.0),
            y=left_pupil_y,
        ),
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=left_pupil_x,
            y=left_pupil_y - (left_iris_diameter / 2.0),
        ),
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=left_pupil_x,
            y=left_pupil_y + (left_iris_diameter / 2.0),
        ),
    ]
    iris_landmarks_right = [
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=right_pupil_x - (right_iris_diameter / 2.0),
            y=right_pupil_y,
        ),
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=right_pupil_x + (right_iris_diameter / 2.0),
            y=right_pupil_y,
        ),
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=right_pupil_x,
            y=right_pupil_y - (right_iris_diameter / 2.0),
        ),
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=right_pupil_x,
            y=right_pupil_y + (right_iris_diameter / 2.0),
        ),
    ]

    payload = _frame_record_payload(frame_id)
    payload["status"] = "OK"
    payload["face"] = {
        "present": True,
        "bounding_box": face_box.model_dump(),
        "landmarks": [
            Point2D(
                space=CoordinateSpace.IMAGE_PX,
                x=face_x_min + 10.0,
                y=face_y_min + 20.0,
            ).model_dump(),
            Point2D(
                space=CoordinateSpace.IMAGE_PX,
                x=face_x_min + face_width - 10.0,
                y=face_y_min + 20.0,
            ).model_dump(),
            Point2D(
                space=CoordinateSpace.IMAGE_PX,
                x=face_x_min + (face_width / 2.0),
                y=face_y_min + face_height - 15.0,
            ).model_dump(),
        ],
        "reason_invalid": None,
    }
    payload["left_eye"] = {
        "present": True,
        "bounding_box": left_eye_box.model_dump(),
        "pupil_center": Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=left_pupil_x,
            y=left_pupil_y,
        ).model_dump(),
        "iris_landmarks": [point.model_dump() for point in iris_landmarks_left],
        "reason_invalid": None,
    }
    payload["right_eye"] = {
        "present": True,
        "bounding_box": right_eye_box.model_dump(),
        "pupil_center": Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=right_pupil_x,
            y=right_pupil_y,
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
        "unigaze_preprocessing_profile": "reference_face2x_imagenet",
        "unigaze_face_crop_scale": 2.0,
        "unigaze_image_mean_rgb": (0.485, 0.456, 0.406),
        "unigaze_image_std_rgb": (0.229, 0.224, 0.225),
        "face_landmarker_running_mode": "IMAGE",
        "camera_intrinsics_policy": "estimate_with_explicit_uncertainty",
        "metric_translation_allowed": False,
        "derived_percentile_lower": 0.05,
        "derived_percentile_upper": 0.95,
        "pnp_landmark_indices": {
            "nose_tip": 1,
            "chin": 152,
            "left_eye_outer": 263,
            "right_eye_outer": 33,
            "left_eye_inner": 362,
            "right_eye_inner": 133,
            "left_mouth_corner": 291,
            "right_mouth_corner": 61,
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

    records = [
        _observed_frame_record(
            "f000000003",
            face_width=80.0,
            face_height=120.0,
            left_pupil_x=155.0,
            right_pupil_x=125.0,
            left_iris_diameter=10.0,
            right_iris_diameter=14.0,
        ),
        _observed_frame_record(
            "f000000004",
            face_x_min=110.0,
            face_y_min=60.0,
            face_width=100.0,
            face_height=140.0,
            left_pupil_x=166.0,
            right_pupil_x=130.0,
            left_iris_diameter=12.0,
            right_iris_diameter=16.0,
        ),
        _observed_frame_record(
            "f000000005",
            face_x_min=90.0,
            face_y_min=40.0,
            face_width=120.0,
            face_height=160.0,
            left_pupil_x=162.0,
            right_pupil_x=120.0,
            left_iris_diameter=18.0,
            right_iris_diameter=20.0,
        ),
    ]

    derived = derive_setup_constants(records)

    assert derived.selected_face_bbox_size_image_px.value == {
        "width_px": {"median": 100.0, "p05": 82.0, "p95": 118.0},
        "height_px": {"median": 140.0, "p05": 122.0, "p95": 158.0},
        "area_px2": {"median": 14000.0, "p05": 10040.0, "p95": 18680.0},
    }
    assert derived.selected_face_bbox_size_image_px.unit == "image_px"
    assert derived.selected_face_bbox_size_image_px.coordinate_space == "image_px"
    assert (
        derived.selected_face_bbox_size_image_px.derivation_method
        == "median plus p05/p95 percentile range of selected-face bounding box "
        "width, height, and area from face.bounding_box where face.present is "
        "true; percentile policy lower=0.05 upper=0.95 using linear interpolation"
    )
    assert derived.selected_face_bbox_size_image_px.contributing_frame_count == 3
    assert derived.selected_face_bbox_size_image_px.uncertainty == "low"
    assert derived.selected_face_bbox_size_image_px.usage == "measurement"

    assert derived.inter_pupil_distance_image_px.value == {
        "median": 36.0,
        "p05": 30.6,
        "p95": 41.4,
    }
    assert derived.inter_pupil_distance_image_px.unit == "image_px"
    assert derived.inter_pupil_distance_image_px.coordinate_space == "image_px"
    assert (
        derived.inter_pupil_distance_image_px.derivation_method
        == "median plus p05/p95 percentile range of Euclidean distance between "
        "left and right pupil centers when both eyes are present; percentile "
        "policy lower=0.05 upper=0.95 using linear interpolation"
    )
    assert derived.inter_pupil_distance_image_px.contributing_frame_count == 3
    assert derived.inter_pupil_distance_image_px.uncertainty == "low"
    assert derived.inter_pupil_distance_image_px.usage == "measurement"

    assert derived.left_iris_diameter_image_px.value == {
        "median": 12.0,
        "p05": 10.2,
        "p95": 17.4,
    }
    assert derived.left_iris_diameter_image_px.unit == "image_px"
    assert derived.left_iris_diameter_image_px.coordinate_space == "image_px"
    assert (
        derived.left_iris_diameter_image_px.derivation_method
        == "median plus p05/p95 percentile range of maximum pairwise iris "
        "landmark distance per frame for the left eye; percentile policy "
        "lower=0.05 upper=0.95 using linear interpolation"
    )
    assert derived.left_iris_diameter_image_px.contributing_frame_count == 3
    assert derived.left_iris_diameter_image_px.uncertainty == "medium"
    assert derived.left_iris_diameter_image_px.usage == "measurement"

    assert derived.right_iris_diameter_image_px.value == {
        "median": 16.0,
        "p05": 14.2,
        "p95": 19.6,
    }
    assert derived.right_iris_diameter_image_px.unit == "image_px"
    assert derived.right_iris_diameter_image_px.coordinate_space == "image_px"
    assert (
        derived.right_iris_diameter_image_px.derivation_method
        == "median plus p05/p95 percentile range of maximum pairwise iris "
        "landmark distance per frame for the right eye; percentile policy "
        "lower=0.05 upper=0.95 using linear interpolation"
    )
    assert derived.right_iris_diameter_image_px.contributing_frame_count == 3
    assert derived.right_iris_diameter_image_px.uncertainty == "medium"
    assert derived.right_iris_diameter_image_px.usage == "measurement"
    assert derived.facecam_roi_image_px.value == {
        "x_min": 90.0,
        "y_min": 40.0,
        "x_max": 210.0,
        "y_max": 200.0,
    }
    assert derived.facecam_roi_image_px.unit == "image_px"
    assert derived.facecam_roi_image_px.coordinate_space == "image_px"
    assert (
        derived.facecam_roi_image_px.derivation_method
        == "bounding box union over observed selected-face boxes; derived ROI "
        "for QA only"
    )
    assert derived.facecam_roi_image_px.contributing_frame_count == 3
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


def test_percentile_policy_text_is_derived_from_named_constants() -> None:
    from chess_gaze import calibration

    source = inspect.getsource(calibration)

    assert calibration.PERCENTILE_POLICY_DESCRIPTION == (
        f"percentile policy lower={calibration.derived_percentile_lower} "
        f"upper={calibration.derived_percentile_upper} using linear interpolation"
    )
    assert (
        "percentile policy lower=0.05 upper=0.95 using linear interpolation"
        not in source
    )
    assert "derived_percentile_lower" in source
    assert "derived_percentile_upper" in source


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
