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
from chess_gaze.gaze_observation import pitch_yaw_to_unit_vector
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.scene_calibration import (
    DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M,
    default_scene_assumptions,
)
from chess_gaze.scene_records import (
    CoordinateFrame3D,
    SceneEyeMidpointRecord,
    SceneInvalidReason,
    SceneUniGazeRayRecord,
    UnitVector3D,
    Vector3D,
)


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


def _gaze_angles(
    *,
    valid: bool,
    pitch_radians: float | None,
    yaw_radians: float | None,
    reason_invalid: ErrorCode | None,
) -> GazeAngles:
    return GazeAngles(
        valid=valid,
        yaw_radians=yaw_radians,
        pitch_radians=pitch_radians,
        reason_invalid=reason_invalid,
    )


def _frame_record_with_gazes(
    *,
    appearance_gaze: GazeAngles,
    recommended_gaze: GazeAngles,
    frame_index: int = 1,
) -> FrameRecord:
    return FrameRecord(
        frame_id=f"frame-{frame_index:04d}",
        frame_index=frame_index,
        status=FrameStatus.OK,
        timestamp_seconds=float(frame_index) / 30.0,
        face=FaceRecord(
            present=False,
            bounding_box=None,
            landmarks=None,
            reason_invalid=ErrorCode.FACE_NOT_FOUND,
        ),
        left_eye=_present_eye(900.0, 540.0),
        right_eye=_present_eye(1020.0, 540.0),
        head_pose=HeadPoseRecord(
            valid=False,
            yaw_radians=None,
            pitch_radians=None,
            roll_radians=None,
            reason_invalid=ErrorCode.HEAD_POSE_INVALID,
        ),
        geometric_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
        appearance_gaze=appearance_gaze,
        recommended_gaze=recommended_gaze,
        errors=[],
    )


def _camera_point(x: float, y: float, z: float) -> Vector3D:
    return Vector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=x,
        y=y,
        z=z,
    )


def _scene_point(x: float, y: float, z: float) -> Vector3D:
    return Vector3D(
        space=CoordinateFrame3D.SCENE_PSEUDO_M,
        x=x,
        y=y,
        z=z,
    )


def _camera_point_non_finite(x: float, y: float, z: float) -> Vector3D:
    return Vector3D.model_construct(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=x,
        y=y,
        z=z,
    )


def _camera_unit_vector(x: float, y: float, z: float) -> UnitVector3D:
    return UnitVector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=x,
        y=y,
        z=z,
    )


def _direction_from_pitch_yaw(
    *,
    pitch_radians: float,
    yaw_radians: float,
) -> UnitVector3D:
    x, y, z = pitch_yaw_to_unit_vector(
        pitch_radians=pitch_radians,
        yaw_radians=yaw_radians,
    )
    return _camera_unit_vector(x, y, z)


def _midpoint_record(
    *,
    valid: bool,
    camera_point: Vector3D | None,
    scene_point: Vector3D | None,
    reason_invalid: SceneInvalidReason | None,
) -> SceneEyeMidpointRecord:
    return SceneEyeMidpointRecord(
        valid=valid,
        origin_policy="both_eyes_required" if valid else None,
        camera_point_m=camera_point,
        scene_point_m=scene_point,
        pupil_distance_px=120.0 if valid else None,
        estimated_depth_m=0.7 if valid else None,
        source_reason_invalid=None if valid else "midpoint invalid",
        reason_invalid=reason_invalid,
    )


def _ray_record(
    *,
    valid: bool,
    direction_camera: UnitVector3D | None,
    reason_invalid: SceneInvalidReason | None = None,
) -> SceneUniGazeRayRecord:
    if valid:
        assert direction_camera is not None
        return SceneUniGazeRayRecord(
            valid=True,
            source="appearance_gaze",
            origin_camera_m=_camera_point(0.0, 0.0, 0.7),
            origin_scene_m=_scene_point(0.0, 0.0, 0.0),
            direction_camera=direction_camera,
            direction_scene=UnitVector3D(
                space=CoordinateFrame3D.SCENE_PSEUDO_M,
                x=direction_camera.x,
                y=direction_camera.y,
                z=direction_camera.z,
            ),
            direction_source="appearance_gaze_unigaze_pitch_yaw",
            pitch_radians=0.0,
            yaw_radians=0.0,
            source_reason_invalid=None,
            reason_invalid=None,
        )
    return SceneUniGazeRayRecord(
        valid=False,
        source="appearance_gaze",
        origin_camera_m=None,
        origin_scene_m=None,
        direction_camera=None,
        direction_scene=None,
        direction_source=None,
        pitch_radians=None,
        yaw_radians=None,
        source_reason_invalid="ray invalid",
        reason_invalid=reason_invalid,
    )


def _direction_angles_in_degrees(
    yaw_degrees: float,
    pitch_degrees: float = 0.0,
) -> UnitVector3D:
    return _direction_from_pitch_yaw(
        pitch_radians=math.radians(pitch_degrees),
        yaw_radians=math.radians(yaw_degrees),
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


def test_back_project_eye_points_uses_euclidean_pupil_distance() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    camera = scene_geometry.estimated_camera_model(1920, 1080)

    projection = scene_geometry.back_project_eye_points(
        _frame_record(
            left_eye=_present_eye(900.0, 500.0),
            right_eye=_present_eye(1020.0, 580.0),
        ),
        camera,
        assumptions,
    )

    expected_pupil_distance = math.sqrt((120.0**2) + (80.0**2))
    expected_depth = (
        DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M
        * 1920.0
        / expected_pupil_distance
    )

    assert projection.left_eye.valid is True
    assert projection.left_eye.camera_point_m is not None
    assert projection.left_eye.camera_point_m.x == pytest.approx(
        ((900.0 - 960.0) * expected_depth) / 1920.0
    )
    assert projection.left_eye.camera_point_m.y == pytest.approx(
        ((500.0 - 540.0) * expected_depth) / 1920.0
    )
    assert projection.left_eye.camera_point_m.z == pytest.approx(expected_depth)

    assert projection.right_eye.valid is True
    assert projection.right_eye.camera_point_m is not None
    assert projection.right_eye.camera_point_m.x == pytest.approx(
        ((1020.0 - 960.0) * expected_depth) / 1920.0
    )
    assert projection.right_eye.camera_point_m.y == pytest.approx(
        ((580.0 - 540.0) * expected_depth) / 1920.0
    )
    assert projection.right_eye.camera_point_m.z == pytest.approx(expected_depth)

    assert projection.midpoint.valid is True
    assert projection.midpoint.pupil_distance_px == pytest.approx(
        expected_pupil_distance
    )
    assert projection.midpoint.estimated_depth_m == pytest.approx(expected_depth)
    assert projection.midpoint.camera_point_m is not None
    assert projection.midpoint.camera_point_m.x == pytest.approx(0.0)
    assert projection.midpoint.camera_point_m.y == pytest.approx(0.0)
    assert projection.midpoint.camera_point_m.z == pytest.approx(expected_depth)
    assert projection.diagnostics["pupil_distance_px"] == pytest.approx(
        expected_pupil_distance
    )


def test_back_project_eye_points_rejects_overlapping_pupil_distance() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    camera = scene_geometry.estimated_camera_model(1920, 1080)

    projection = scene_geometry.back_project_eye_points(
        _frame_record(
            left_eye=_present_eye(960.0, 540.0),
            right_eye=_present_eye(960.0, 540.0),
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
    assert projection.diagnostics["pupil_distance_px"] == pytest.approx(0.0)


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


def test_robust_scene_center_uses_geometric_median_after_mad_screening() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()

    estimate = scene_geometry.robust_scene_center(
        [
            _camera_point(0.0, 0.0, 0.70),
            _camera_point(0.01, 0.0, 0.70),
            _camera_point(-0.01, 0.0, 0.70),
            _camera_point(0.0, 0.01, 0.70),
            _camera_point(0.0, -0.01, 0.70),
            _camera_point(1.0, 2.0, 3.0),
        ],
        assumptions,
    )

    assert estimate.point_camera_m.x == pytest.approx(0.0, abs=1e-6)
    assert estimate.point_camera_m.y == pytest.approx(0.0, abs=1e-6)
    assert estimate.point_camera_m.z == pytest.approx(0.70, abs=1e-6)
    assert estimate.candidate_count == 6
    assert estimate.finite_candidate_count == 6
    assert estimate.dropped_non_finite_count == 0
    assert estimate.inlier_count == 5
    assert estimate.mad_m == pytest.approx((0.005, 0.005, 0.0))
    assert estimate.thresholds_m == pytest.approx((0.0175, 0.0175, 0.015))
    assert estimate.iteration_count >= 1
    assert estimate.convergence_tolerance_m == pytest.approx(1e-6)
    assert estimate.fallback_used is False
    assert estimate.uncertainty == "medium"


def test_robust_scene_center_falls_back_when_inliers_are_insufficient() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()

    estimate = scene_geometry.robust_scene_center(
        [
            _camera_point(0.01, 0.0, 0.70),
            _camera_point(-0.01, 0.0, 0.70),
            _camera_point(0.0, 0.01, 0.70),
            _camera_point(0.0, -0.01, 0.70),
        ],
        assumptions,
    )

    assert estimate.point_camera_m.x == pytest.approx(
        assumptions.default_scene_center_camera_m[0]
    )
    assert estimate.point_camera_m.y == pytest.approx(
        assumptions.default_scene_center_camera_m[1]
    )
    assert estimate.point_camera_m.z == pytest.approx(
        assumptions.default_scene_center_camera_m[2]
    )
    assert estimate.candidate_count == 4
    assert estimate.finite_candidate_count == 4
    assert estimate.inlier_count == 4
    assert estimate.fallback_used is True
    assert estimate.uncertainty == "high"


def test_robust_scene_center_uses_min_axis_tolerance_when_mad_is_zero() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()

    estimate = scene_geometry.robust_scene_center(
        [
            _camera_point(0.0, 0.0, 0.70),
            _camera_point(0.0, 0.0, 0.70),
            _camera_point(0.0, 0.0, 0.70),
            _camera_point(0.0, 0.0, 0.70),
            _camera_point(0.014, 0.0, 0.70),
        ],
        assumptions,
    )

    assert estimate.inlier_count == 5
    assert estimate.mad_m == pytest.approx((0.0, 0.0, 0.0))
    assert estimate.thresholds_m == pytest.approx((0.015, 0.015, 0.015))
    assert estimate.point_camera_m.x == pytest.approx(0.0, abs=1e-6)
    assert estimate.fallback_used is False


def test_robust_scene_center_drops_non_finite_candidates() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()

    estimate = scene_geometry.robust_scene_center(
        [
            _camera_point(0.0, 0.0, 0.70),
            _camera_point(0.01, 0.0, 0.70),
            _camera_point(-0.01, 0.0, 0.70),
            _camera_point(0.0, 0.01, 0.70),
            _camera_point(0.0, -0.01, 0.70),
            _camera_point_non_finite(math.nan, 0.0, 0.70),
        ],
        assumptions,
    )

    assert estimate.candidate_count == 6
    assert estimate.finite_candidate_count == 5
    assert estimate.dropped_non_finite_count == 1
    assert estimate.inlier_count == 5
    assert estimate.fallback_used is False


def test_robust_scene_center_does_not_stop_at_coincident_sample_point() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    inliers = [
        _camera_point(0.0, 0.0, 1.0),
        _camera_point(0.0, 0.01, 1.0),
        _camera_point(0.01, 0.0, 1.0),
        _camera_point(0.01, 0.01, 1.0),
        _camera_point(0.0, 0.0, 1.0),
    ]

    estimate = scene_geometry.robust_scene_center(inliers, assumptions)

    returned_point = estimate.point_camera_m
    sample_point = (0.0, 0.0, 1.0)
    assert (
        returned_point.x,
        returned_point.y,
        returned_point.z,
    ) != pytest.approx(sample_point, abs=1e-9)

    returned_sum_distance = sum(
        math.dist(
            (returned_point.x, returned_point.y, returned_point.z),
            (point.x, point.y, point.z),
        )
        for point in inliers
    )
    sample_sum_distance = sum(
        math.dist(sample_point, (point.x, point.y, point.z))
        for point in inliers
    )
    assert returned_sum_distance < sample_sum_distance


def test_unigaze_ray_from_frame_uses_appearance_gaze_not_recommended_gaze() -> None:
    scene_geometry = _scene_geometry()
    midpoint = _midpoint_record(
        valid=True,
        camera_point=_camera_point(0.1, -0.02, 0.7),
        scene_point=_scene_point(0.0, 0.0, 0.0),
        reason_invalid=None,
    )

    ray = scene_geometry.unigaze_ray_from_frame(
        _frame_record_with_gazes(
            appearance_gaze=_gaze_angles(
                valid=True,
                pitch_radians=0.05,
                yaw_radians=0.10,
                reason_invalid=None,
            ),
            recommended_gaze=_gaze_angles(
                valid=True,
                pitch_radians=-0.30,
                yaw_radians=0.75,
                reason_invalid=None,
            ),
        ),
        midpoint,
    )

    appearance_vector = pitch_yaw_to_unit_vector(
        pitch_radians=0.05,
        yaw_radians=0.10,
    )
    recommended_vector = pitch_yaw_to_unit_vector(
        pitch_radians=-0.30,
        yaw_radians=0.75,
    )

    assert ray.valid is True
    assert ray.source == "appearance_gaze"
    assert ray.direction_source == "appearance_gaze_unigaze_pitch_yaw"
    assert ray.pitch_radians == pytest.approx(0.05)
    assert ray.yaw_radians == pytest.approx(0.10)
    assert ray.direction_camera is not None
    assert ray.direction_camera.x == pytest.approx(appearance_vector[0])
    assert ray.direction_camera.y == pytest.approx(appearance_vector[1])
    assert ray.direction_camera.z == pytest.approx(appearance_vector[2])
    assert ray.direction_camera.x != pytest.approx(recommended_vector[0])
    assert ray.direction_camera.y != pytest.approx(recommended_vector[1])
    assert ray.direction_camera.z != pytest.approx(recommended_vector[2])


def test_unigaze_ray_from_frame_matches_pitch_yaw_sign_convention() -> None:
    scene_geometry = _scene_geometry()
    midpoint = _midpoint_record(
        valid=True,
        camera_point=_camera_point(0.0, 0.0, 0.7),
        scene_point=_scene_point(0.0, 0.0, 0.0),
        reason_invalid=None,
    )
    pitch_radians = 0.2
    yaw_radians = 0.1

    ray = scene_geometry.unigaze_ray_from_frame(
        _frame_record_with_gazes(
            appearance_gaze=_gaze_angles(
                valid=True,
                pitch_radians=pitch_radians,
                yaw_radians=yaw_radians,
                reason_invalid=None,
            ),
            recommended_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
        ),
        midpoint,
    )

    expected_x, expected_y, expected_z = pitch_yaw_to_unit_vector(
        pitch_radians=pitch_radians,
        yaw_radians=yaw_radians,
    )
    assert ray.valid is True
    assert ray.direction_camera is not None
    assert ray.direction_camera.x == pytest.approx(expected_x)
    assert ray.direction_camera.y == pytest.approx(expected_y)
    assert ray.direction_camera.z == pytest.approx(expected_z)


def test_unigaze_ray_from_frame_requires_valid_midpoint() -> None:
    scene_geometry = _scene_geometry()

    ray = scene_geometry.unigaze_ray_from_frame(
        _frame_record_with_gazes(
            appearance_gaze=_gaze_angles(
                valid=True,
                pitch_radians=0.05,
                yaw_radians=0.10,
                reason_invalid=None,
            ),
            recommended_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
        ),
        _midpoint_record(
            valid=False,
            camera_point=None,
            scene_point=None,
            reason_invalid=SceneInvalidReason.EYE_MIDPOINT_INVALID,
        ),
    )

    assert ray.valid is False
    assert ray.reason_invalid == SceneInvalidReason.EYE_MIDPOINT_INVALID


def test_robust_main_direction_selects_dominant_cluster_with_outliers() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    inlier_directions = [
        _direction_angles_in_degrees(0.0, 0.0),
        _direction_angles_in_degrees(3.0, 0.0),
        _direction_angles_in_degrees(-3.0, 0.0),
        _direction_angles_in_degrees(2.0, -1.0),
        _direction_angles_in_degrees(-2.0, 1.0),
    ]
    rays = [
        _ray_record(valid=True, direction_camera=direction)
        for direction in inlier_directions
    ]
    rays.extend(
        [
            _ray_record(
                valid=True,
                direction_camera=_direction_angles_in_degrees(70.0, 0.0),
            ),
            _ray_record(
                valid=True,
                direction_camera=_direction_angles_in_degrees(-75.0, 0.0),
            ),
        ]
    )

    estimate = scene_geometry.robust_main_direction(rays, assumptions)

    expected_x = sum(direction.x for direction in inlier_directions)
    expected_y = sum(direction.y for direction in inlier_directions)
    expected_z = sum(direction.z for direction in inlier_directions)
    expected_norm = math.sqrt(
        (expected_x * expected_x)
        + (expected_y * expected_y)
        + (expected_z * expected_z)
    )

    assert estimate.candidate_count == 7
    assert estimate.finite_candidate_count == 7
    assert estimate.inlier_count == 5
    assert estimate.fallback_used is False
    assert estimate.direction_camera.x == pytest.approx(expected_x / expected_norm)
    assert estimate.direction_camera.y == pytest.approx(expected_y / expected_norm)
    assert estimate.direction_camera.z == pytest.approx(expected_z / expected_norm)


def test_robust_main_direction_prefers_lower_median_residual_on_inlier_ties() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    cluster_a = [0.0, -9.0, -6.0, 6.0, 9.0]
    cluster_b = [50.0, 47.0, 48.0, 52.0, 53.0]
    rays = [
        _ray_record(valid=True, direction_camera=_direction_angles_in_degrees(yaw))
        for yaw in [*cluster_a, *cluster_b]
    ]

    estimate = scene_geometry.robust_main_direction(rays, assumptions)

    assert estimate.inlier_count == 5
    assert estimate.fallback_used is False
    assert estimate.direction_camera.x > 0.70
    assert estimate.direction_camera.z > 0.60


def test_robust_main_direction_prefers_earlier_seed_index_when_residuals_tie() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    cluster_a = [0.0, -3.0, -6.0, 3.0, 6.0]
    cluster_b = [50.0, 47.0, 44.0, 53.0, 56.0]
    rays = [
        _ray_record(valid=True, direction_camera=_direction_angles_in_degrees(yaw))
        for yaw in [*cluster_a, *cluster_b]
    ]

    estimate = scene_geometry.robust_main_direction(rays, assumptions)

    assert estimate.inlier_count == 5
    assert estimate.fallback_used is False
    assert estimate.direction_camera.x == pytest.approx(0.0, abs=0.08)
    assert estimate.direction_camera.z > 0.99


def test_robust_main_direction_falls_back_when_too_few_rays_are_valid() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    rays = [
        _ray_record(
            valid=True,
            direction_camera=_direction_angles_in_degrees(0.0),
        ),
        _ray_record(
            valid=True,
            direction_camera=_direction_angles_in_degrees(2.0),
        ),
        _ray_record(
            valid=True,
            direction_camera=_direction_angles_in_degrees(-2.0),
        ),
        _ray_record(
            valid=True,
            direction_camera=_direction_angles_in_degrees(1.0),
        ),
        _ray_record(
            valid=False,
            direction_camera=None,
            reason_invalid=SceneInvalidReason.UNIGAZE_INVALID,
        ),
    ]

    estimate = scene_geometry.robust_main_direction(rays, assumptions)

    assert estimate.candidate_count == 5
    assert estimate.finite_candidate_count == 4
    assert estimate.inlier_count == 4
    assert estimate.fallback_used is True
    assert estimate.uncertainty == "high"
    assert estimate.direction_camera.x == pytest.approx(0.0)
    assert estimate.direction_camera.y == pytest.approx(0.0)
    assert estimate.direction_camera.z == pytest.approx(1.0)


def test_robust_main_direction_treats_opposite_rays_as_outliers() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    forward_cluster = [
        _direction_angles_in_degrees(0.0),
        _direction_angles_in_degrees(2.0),
        _direction_angles_in_degrees(-2.0),
        _direction_angles_in_degrees(3.0),
        _direction_angles_in_degrees(-3.0),
    ]
    rays = [
        _ray_record(valid=True, direction_camera=direction)
        for direction in forward_cluster
    ]
    rays.extend(
        [
            _ray_record(
                valid=True,
                direction_camera=_direction_angles_in_degrees(180.0),
            ),
            _ray_record(
                valid=True,
                direction_camera=_direction_angles_in_degrees(177.0),
            ),
        ]
    )

    estimate = scene_geometry.robust_main_direction(rays, assumptions)

    assert estimate.inlier_count == 5
    assert estimate.fallback_used is False
    assert estimate.direction_camera.z > 0.99


def test_build_scene_axis_basis_returns_right_handed_orthonormal_columns() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    direction = scene_geometry.RobustDirectionEstimate(
        direction_camera=_direction_angles_in_degrees(8.0, -4.0),
        candidate_count=10,
        finite_candidate_count=10,
        inlier_count=8,
        angle_threshold_radians=assumptions.direction_inlier_angle_radians,
        median_angular_residual_radians=0.05,
        fallback_used=False,
        uncertainty="medium",
    )

    axes = scene_geometry.build_scene_axis_basis(
        direction,
        [
            _camera_unit_vector(1.0, 0.0, 0.0),
            _camera_unit_vector(1.0, 0.0, 0.0),
            _camera_unit_vector(1.0, 0.0, 0.0),
        ],
        assumptions,
    )

    right = axes.right_camera
    up = axes.up_camera
    back = axes.back_camera
    forward = axes.forward_camera

    assert math.sqrt((right.x**2) + (right.y**2) + (right.z**2)) == pytest.approx(1.0)
    assert math.sqrt((up.x**2) + (up.y**2) + (up.z**2)) == pytest.approx(1.0)
    assert math.sqrt((back.x**2) + (back.y**2) + (back.z**2)) == pytest.approx(1.0)
    assert (
        (right.x * up.x) + (right.y * up.y) + (right.z * up.z)
    ) == pytest.approx(0.0, abs=1e-6)
    assert (
        (right.x * back.x) + (right.y * back.y) + (right.z * back.z)
    ) == pytest.approx(0.0, abs=1e-6)
    assert (
        (up.x * back.x) + (up.y * back.y) + (up.z * back.z)
    ) == pytest.approx(0.0, abs=1e-6)
    assert (
        (back.x * forward.x) + (back.y * forward.y) + (back.z * forward.z)
    ) == pytest.approx(-1.0, abs=1e-6)
    assert axes.determinant_right_up_back == pytest.approx(1.0, abs=1e-6)
    assert axes.fallbacks == []


def test_build_scene_axis_basis_records_fallbacks_for_degenerate_right_and_up() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    direction = scene_geometry.RobustDirectionEstimate(
        direction_camera=_camera_unit_vector(0.0, -1.0, 0.0),
        candidate_count=8,
        finite_candidate_count=8,
        inlier_count=8,
        angle_threshold_radians=assumptions.direction_inlier_angle_radians,
        median_angular_residual_radians=0.01,
        fallback_used=False,
        uncertainty="medium",
    )

    axes = scene_geometry.build_scene_axis_basis(
        direction,
        [
            _camera_unit_vector(0.0, -1.0, 0.0),
            _camera_unit_vector(0.0, -1.0, 0.0),
        ],
        assumptions,
    )

    assert axes.determinant_right_up_back == pytest.approx(1.0, abs=1e-6)
    assert any(
        "right_axis" in fallback and "parallel" in fallback
        for fallback in axes.fallbacks
    )
    assert any(
        "up_axis" in fallback and "parallel" in fallback
        for fallback in axes.fallbacks
    )
