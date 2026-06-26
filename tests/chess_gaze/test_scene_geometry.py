from __future__ import annotations

import importlib
import math

import pytest

from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    EyeRecord,
    FaceRecord,
    FrameRecord,
    GazeAngles,
    HeadPoseRecord,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.scene_calibration import (
    DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M,
    default_scene_assumptions,
)
from chess_gaze.scene_records import CoordinateFrame3D, SceneInvalidReason


def _scene_geometry():
    return importlib.import_module("chess_gaze.scene_geometry")


def _point(x: float, y: float) -> Point2D:
    return Point2D(space=CoordinateSpace.IMAGE_PX, x=x, y=y)


def _normalized_point(x: float, y: float) -> Point2D:
    return Point2D(space=CoordinateSpace.NORMALIZED, x=x, y=y)


def _bbox(center_x: float, center_y: float) -> BBox:
    return BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=center_x - 2.0,
        y_min=center_y - 2.0,
        x_max=center_x + 2.0,
        y_max=center_y + 2.0,
    )


def _present_eye(center_x: float, center_y: float) -> EyeRecord:
    return EyeRecord(
        present=True,
        bounding_box=_bbox(center_x, center_y),
        pupil_center=_point(center_x, center_y),
        iris_landmarks=[
            _point(center_x - 1.0, center_y),
            _point(center_x + 1.0, center_y),
        ],
        reason_invalid=None,
    )


def _present_eye_with_pupil(point: Point2D) -> EyeRecord:
    return EyeRecord(
        present=True,
        bounding_box=_bbox(point.x, point.y),
        pupil_center=point,
        iris_landmarks=[
            point,
            point,
        ],
        reason_invalid=None,
    )


def _missing_eye(reason_invalid: ErrorCode) -> EyeRecord:
    return EyeRecord(
        present=False,
        bounding_box=None,
        pupil_center=None,
        iris_landmarks=None,
        reason_invalid=reason_invalid,
    )


def _invalid_angles(reason_invalid: ErrorCode) -> GazeAngles:
    return GazeAngles(
        valid=False,
        yaw_radians=None,
        pitch_radians=None,
        reason_invalid=reason_invalid,
    )


def _frame_record(
    *,
    left_eye: EyeRecord,
    right_eye: EyeRecord,
) -> FrameRecord:
    return FrameRecord(
        frame_id="frame-0001",
        frame_index=1,
        status=FrameStatus.OK,
        timestamp_seconds=0.0,
        face=FaceRecord(
            present=False,
            bounding_box=None,
            landmarks=None,
            reason_invalid=ErrorCode.FACE_NOT_FOUND,
        ),
        left_eye=left_eye,
        right_eye=right_eye,
        head_pose=HeadPoseRecord(
            valid=False,
            yaw_radians=None,
            pitch_radians=None,
            roll_radians=None,
            reason_invalid=ErrorCode.HEAD_POSE_INVALID,
        ),
        geometric_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
        appearance_gaze=_invalid_angles(ErrorCode.GAZE_MODEL_FAILED),
        recommended_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
        errors=[],
    )


def _frame_record_with_non_finite_left_eye() -> FrameRecord:
    non_finite_point = Point2D.model_construct(
        space=CoordinateSpace.IMAGE_PX,
        x=math.nan,
        y=540.0,
    )
    eye = EyeRecord.model_construct(
        present=True,
        bounding_box=BBox.model_construct(
            space=CoordinateSpace.IMAGE_PX,
            x_min=0.0,
            y_min=0.0,
            x_max=4.0,
            y_max=4.0,
        ),
        pupil_center=non_finite_point,
        iris_landmarks=[non_finite_point],
        reason_invalid=None,
    )
    return FrameRecord.model_construct(
        frame_id="frame-0001",
        frame_index=1,
        status=FrameStatus.OK,
        timestamp_seconds=0.0,
        face=FaceRecord(
            present=False,
            bounding_box=None,
            landmarks=None,
            reason_invalid=ErrorCode.FACE_NOT_FOUND,
        ),
        left_eye=eye,
        right_eye=_present_eye(1020.0, 540.0),
        head_pose=HeadPoseRecord(
            valid=False,
            yaw_radians=None,
            pitch_radians=None,
            roll_radians=None,
            reason_invalid=ErrorCode.HEAD_POSE_INVALID,
        ),
        geometric_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
        appearance_gaze=_invalid_angles(ErrorCode.GAZE_MODEL_FAILED),
        recommended_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
        errors=[],
    )


def test_estimated_camera_model_uses_landscape_image_size_for_intrinsics() -> None:
    scene_geometry = _scene_geometry()

    camera = scene_geometry.estimated_camera_model(1920, 1080)

    assert camera.policy == "estimated_pinhole_from_image_size"
    assert camera.frame_width_px == 1920
    assert camera.frame_height_px == 1080
    assert camera.fx_px == pytest.approx(1920.0)
    assert camera.fy_px == pytest.approx(1920.0)
    assert camera.cx_px == pytest.approx(960.0)
    assert camera.cy_px == pytest.approx(540.0)


def test_estimated_camera_model_uses_longest_side_for_portrait_intrinsics() -> None:
    scene_geometry = _scene_geometry()

    camera = scene_geometry.estimated_camera_model(1080, 1920)

    assert camera.fx_px == pytest.approx(1920.0)
    assert camera.fy_px == pytest.approx(1920.0)
    assert camera.cx_px == pytest.approx(540.0)
    assert camera.cy_px == pytest.approx(960.0)


def test_back_project_eye_points_projects_eyes_and_midpoint_in_camera_space() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    camera = scene_geometry.estimated_camera_model(1920, 1080)

    projection = scene_geometry.back_project_eye_points(
        _frame_record(
            left_eye=_present_eye(900.0, 540.0),
            right_eye=_present_eye(1020.0, 540.0),
        ),
        camera,
        assumptions,
    )

    expected_depth = DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M * 1920.0 / 120.0

    assert projection.left_eye.valid is True
    assert projection.left_eye.camera_point_m is not None
    assert projection.left_eye.camera_point_m.space == (
        CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M
    )
    assert projection.left_eye.camera_point_m.x == pytest.approx(-0.0315)
    assert projection.left_eye.camera_point_m.y == pytest.approx(0.0)
    assert projection.left_eye.camera_point_m.z == pytest.approx(expected_depth)

    assert projection.right_eye.valid is True
    assert projection.right_eye.camera_point_m is not None
    assert projection.right_eye.camera_point_m.x == pytest.approx(0.0315)
    assert projection.right_eye.camera_point_m.y == pytest.approx(0.0)
    assert projection.right_eye.camera_point_m.z == pytest.approx(expected_depth)

    assert projection.midpoint.valid is True
    assert projection.midpoint.origin_policy == "both_eyes_required"
    assert projection.midpoint.camera_point_m is not None
    assert projection.midpoint.camera_point_m.x == pytest.approx(0.0)
    assert projection.midpoint.camera_point_m.y == pytest.approx(0.0)
    assert projection.midpoint.camera_point_m.z == pytest.approx(expected_depth)
    assert projection.midpoint.pupil_distance_px == pytest.approx(120.0)
    assert projection.midpoint.estimated_depth_m == pytest.approx(expected_depth)
    assert (
        projection.diagnostics["depth_source"]
        == "interpupillary_distance_assumption"
    )
    assert projection.diagnostics["interpupillary_distance_m"] == pytest.approx(0.063)
    assert projection.diagnostics["pupil_distance_px"] == pytest.approx(120.0)


@pytest.mark.parametrize(
    ("left_x", "right_x", "expected_distance"),
    [
        (960.0, 960.0, 0.0),
        (1020.0, 900.0, -120.0),
    ],
)
def test_back_project_eye_points_rejects_non_positive_pupil_distance(
    left_x: float,
    right_x: float,
    expected_distance: float,
) -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    camera = scene_geometry.estimated_camera_model(1920, 1080)

    projection = scene_geometry.back_project_eye_points(
        _frame_record(
            left_eye=_present_eye(left_x, 540.0),
            right_eye=_present_eye(right_x, 540.0),
        ),
        camera,
        assumptions,
    )

    assert projection.left_eye.valid is False
    assert projection.left_eye.reason_invalid == SceneInvalidReason.LEFT_EYE_INVALID
    assert projection.right_eye.valid is False
    assert projection.right_eye.reason_invalid == SceneInvalidReason.RIGHT_EYE_INVALID
    assert projection.midpoint.valid is False
    assert projection.midpoint.reason_invalid == SceneInvalidReason.EYE_MIDPOINT_INVALID
    assert projection.midpoint.source_reason_invalid is not None
    assert "pupil_distance_px" in projection.midpoint.source_reason_invalid
    assert projection.diagnostics["pupil_distance_px"] == pytest.approx(
        expected_distance
    )


def test_back_project_eye_points_rejects_non_finite_pupil_input() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    camera = scene_geometry.estimated_camera_model(1920, 1080)

    projection = scene_geometry.back_project_eye_points(
        _frame_record_with_non_finite_left_eye(),
        camera,
        assumptions,
    )

    assert projection.left_eye.valid is False
    assert projection.left_eye.reason_invalid == SceneInvalidReason.NON_FINITE_INPUT
    assert projection.midpoint.valid is False
    assert projection.midpoint.reason_invalid == SceneInvalidReason.NON_FINITE_INPUT
    assert projection.diagnostics["non_finite_input"] is True


def test_back_project_eye_points_rejects_normalized_pupil_coordinates() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    camera = scene_geometry.estimated_camera_model(1920, 1080)

    projection = scene_geometry.back_project_eye_points(
        _frame_record(
            left_eye=_present_eye_with_pupil(_normalized_point(0.45, 0.50)),
            right_eye=_present_eye(1020.0, 540.0),
        ),
        camera,
        assumptions,
    )

    assert projection.left_eye.valid is False
    assert projection.left_eye.reason_invalid == SceneInvalidReason.LEFT_EYE_INVALID
    assert projection.left_eye.camera_point_m is None
    assert projection.right_eye.valid is False
    assert projection.right_eye.reason_invalid == SceneInvalidReason.RIGHT_EYE_INVALID
    assert projection.right_eye.camera_point_m is None
    assert projection.midpoint.valid is False
    assert projection.midpoint.reason_invalid == SceneInvalidReason.EYE_MIDPOINT_INVALID
    assert projection.midpoint.source_reason_invalid is not None
    assert "NORMALIZED" in projection.midpoint.source_reason_invalid
    assert projection.diagnostics["invalid_coordinate_space"] == "NORMALIZED"
    assert projection.diagnostics["invalid_coordinate_space_eyes"] == "left_eye"
    assert projection.diagnostics["left_eye_in_frame"] is None


def test_back_project_eye_points_marks_midpoint_invalid_when_one_eye_is_missing(
) -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    camera = scene_geometry.estimated_camera_model(1920, 1080)

    projection = scene_geometry.back_project_eye_points(
        _frame_record(
            left_eye=_present_eye(900.0, 540.0),
            right_eye=_missing_eye(ErrorCode.RIGHT_EYE_NOT_FOUND),
        ),
        camera,
        assumptions,
    )

    assert projection.midpoint.valid is False
    assert projection.midpoint.reason_invalid == SceneInvalidReason.EYE_MIDPOINT_INVALID
    assert projection.midpoint.source_reason_invalid == "RIGHT_EYE_NOT_FOUND"


def test_back_project_eye_points_preserves_out_of_frame_coordinates_in_diagnostics(
) -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    camera = scene_geometry.estimated_camera_model(1920, 1080)

    projection = scene_geometry.back_project_eye_points(
        _frame_record(
            left_eye=_present_eye(-60.0, 1100.0),
            right_eye=_present_eye(60.0, 1100.0),
        ),
        camera,
        assumptions,
    )

    assert projection.left_eye.image_px is not None
    assert projection.left_eye.image_px.x == pytest.approx(-60.0)
    assert projection.left_eye.image_px.y == pytest.approx(1100.0)
    assert projection.right_eye.image_px is not None
    assert projection.right_eye.image_px.x == pytest.approx(60.0)
    assert projection.right_eye.image_px.y == pytest.approx(1100.0)
    assert projection.diagnostics["left_eye_in_frame"] is False
    assert projection.diagnostics["right_eye_in_frame"] is False
