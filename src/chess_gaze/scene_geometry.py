from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from chess_gaze.frame_records import EyeRecord, FrameRecord
from chess_gaze.geometry import CoordinateSpace, Point2D
from chess_gaze.scene_calibration import SceneAssumptions
from chess_gaze.scene_records import (
    CoordinateFrame3D,
    SceneAxisBasisRecord,
    SceneCameraModel,
    SceneEyeMidpointRecord,
    SceneEyeRecord,
    SceneInvalidReason,
    SceneMonitorHitRecord,
    SceneMonitorPlaneRecord,
    SceneUniGazeRayRecord,
    UnitVector3D,
    Vector3D,
)


@dataclass(frozen=True)
class SceneEyePairProjection:
    left_eye_valid: bool
    right_eye_valid: bool
    left_eye: SceneEyeRecord
    right_eye: SceneEyeRecord
    midpoint: SceneEyeMidpointRecord
    diagnostics: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class RobustPointEstimate:
    point_camera_m: Vector3D
    candidate_count: int
    finite_candidate_count: int
    inlier_count: int
    mad_m: tuple[float, float, float]
    thresholds_m: tuple[float, float, float]
    iteration_count: int
    fallback_used: bool
    uncertainty: str


@dataclass(frozen=True)
class RobustDirectionEstimate:
    direction_camera: UnitVector3D
    candidate_count: int
    finite_candidate_count: int
    inlier_count: int
    angle_threshold_radians: float
    median_angular_residual_radians: float | None
    fallback_used: bool
    uncertainty: str


def estimated_camera_model(frame_width: int, frame_height: int) -> SceneCameraModel:
    focal_length_px = float(max(frame_width, frame_height))
    return SceneCameraModel(
        policy="estimated_pinhole_from_image_size",
        frame_width_px=frame_width,
        frame_height_px=frame_height,
        fx_px=focal_length_px,
        fy_px=focal_length_px,
        cx_px=frame_width / 2.0,
        cy_px=frame_height / 2.0,
        metric_translation_allowed=False,
        uncertainty="high",
    )


def back_project_eye_points(
    frame_record: FrameRecord,
    camera: SceneCameraModel,
    assumptions: SceneAssumptions,
) -> SceneEyePairProjection:
    left_source_reason = _source_reason(frame_record.left_eye)
    right_source_reason = _source_reason(frame_record.right_eye)
    left_point = _pupil_center(frame_record.left_eye)
    right_point = _pupil_center(frame_record.right_eye)

    diagnostics: dict[str, str | int | float | bool | None] = {
        "depth_source": "interpupillary_distance_assumption",
        "interpupillary_distance_m": assumptions.adult_male_interpupillary_distance_m,
        "pupil_distance_px": None,
        "non_finite_input": False,
        "invalid_coordinate_space": None,
        "invalid_coordinate_space_eyes": None,
        "left_eye_in_frame": _point_in_frame(left_point, camera),
        "right_eye_in_frame": _point_in_frame(right_point, camera),
        "left_source_reason_invalid": left_source_reason,
        "right_source_reason_invalid": right_source_reason,
    }

    left_state = _eye_projection_state(
        eye=frame_record.left_eye,
        point=left_point,
        invalid_reason=SceneInvalidReason.LEFT_EYE_INVALID,
        source_reason=left_source_reason,
    )
    right_state = _eye_projection_state(
        eye=frame_record.right_eye,
        point=right_point,
        invalid_reason=SceneInvalidReason.RIGHT_EYE_INVALID,
        source_reason=right_source_reason,
    )
    _persist_invalid_coordinate_space_diagnostics(
        diagnostics=diagnostics,
        left_state=left_state,
        right_state=right_state,
    )

    if left_state.kind == "non_finite" or right_state.kind == "non_finite":
        diagnostics["non_finite_input"] = True
        left_eye = _finalize_invalid_eye(
            state=left_state,
            point=left_point,
            fallback_reason=(
                "paired pupil distance unavailable because "
                "an eye input is non-finite"
            ),
        )
        right_eye = _finalize_invalid_eye(
            state=right_state,
            point=right_point,
            fallback_reason=(
                "paired pupil distance unavailable because "
                "an eye input is non-finite"
            ),
        )
        midpoint = SceneEyeMidpointRecord(
            valid=False,
            origin_policy=None,
            camera_point_m=None,
            scene_point_m=None,
            pupil_distance_px=None,
            estimated_depth_m=None,
            source_reason_invalid=_first_reason(
                left_eye.source_reason_invalid,
                right_eye.source_reason_invalid,
                "non-finite pupil input",
            ),
            reason_invalid=SceneInvalidReason.NON_FINITE_INPUT,
        )
        return _eye_pair_projection(
            left_eye=left_eye,
            right_eye=right_eye,
            midpoint=midpoint,
            diagnostics=diagnostics,
        )

    if left_state.kind != "projectable" or right_state.kind != "projectable":
        left_eye = _finalize_invalid_eye(
            state=left_state,
            point=left_point,
            fallback_reason=_first_reason(
                left_source_reason,
                right_source_reason,
                "paired pupil distance unavailable",
            ),
        )
        right_eye = _finalize_invalid_eye(
            state=right_state,
            point=right_point,
            fallback_reason=_first_reason(
                left_source_reason,
                right_source_reason,
                "paired pupil distance unavailable",
            ),
        )
        midpoint = SceneEyeMidpointRecord(
            valid=False,
            origin_policy=None,
            camera_point_m=None,
            scene_point_m=None,
            pupil_distance_px=None,
            estimated_depth_m=None,
            source_reason_invalid=_first_reason(
                left_eye.source_reason_invalid,
                right_eye.source_reason_invalid,
                "missing eye pair input",
            ),
            reason_invalid=SceneInvalidReason.EYE_MIDPOINT_INVALID,
        )
        return _eye_pair_projection(
            left_eye=left_eye,
            right_eye=right_eye,
            midpoint=midpoint,
            diagnostics=diagnostics,
        )

    pupil_distance_px = right_state.point.x - left_state.point.x
    diagnostics["pupil_distance_px"] = pupil_distance_px

    if not math.isfinite(pupil_distance_px):
        diagnostics["non_finite_input"] = True
        left_eye = _invalid_eye_record(
            image_px=left_state.point,
            reason_invalid=SceneInvalidReason.NON_FINITE_INPUT,
            source_reason_invalid="pupil_distance_px is non-finite",
        )
        right_eye = _invalid_eye_record(
            image_px=right_state.point,
            reason_invalid=SceneInvalidReason.NON_FINITE_INPUT,
            source_reason_invalid="pupil_distance_px is non-finite",
        )
        midpoint = SceneEyeMidpointRecord(
            valid=False,
            origin_policy=None,
            camera_point_m=None,
            scene_point_m=None,
            pupil_distance_px=None,
            estimated_depth_m=None,
            source_reason_invalid="pupil_distance_px is non-finite",
            reason_invalid=SceneInvalidReason.NON_FINITE_INPUT,
        )
        return _eye_pair_projection(
            left_eye=left_eye,
            right_eye=right_eye,
            midpoint=midpoint,
            diagnostics=diagnostics,
        )

    if pupil_distance_px <= 0.0:
        reason = f"pupil_distance_px must be > 0, got {pupil_distance_px}"
        left_eye = _invalid_eye_record(
            image_px=left_state.point,
            reason_invalid=SceneInvalidReason.LEFT_EYE_INVALID,
            source_reason_invalid=reason,
        )
        right_eye = _invalid_eye_record(
            image_px=right_state.point,
            reason_invalid=SceneInvalidReason.RIGHT_EYE_INVALID,
            source_reason_invalid=reason,
        )
        midpoint = SceneEyeMidpointRecord(
            valid=False,
            origin_policy=None,
            camera_point_m=None,
            scene_point_m=None,
            pupil_distance_px=pupil_distance_px,
            estimated_depth_m=None,
            source_reason_invalid=reason,
            reason_invalid=SceneInvalidReason.EYE_MIDPOINT_INVALID,
        )
        return _eye_pair_projection(
            left_eye=left_eye,
            right_eye=right_eye,
            midpoint=midpoint,
            diagnostics=diagnostics,
        )

    depth_m = (
        assumptions.adult_male_interpupillary_distance_m
        * camera.fx_px
        / pupil_distance_px
    )
    if not math.isfinite(depth_m):
        diagnostics["non_finite_input"] = True
        reason = "estimated_depth_m is non-finite"
        left_eye = _invalid_eye_record(
            image_px=left_state.point,
            reason_invalid=SceneInvalidReason.NON_FINITE_INPUT,
            source_reason_invalid=reason,
        )
        right_eye = _invalid_eye_record(
            image_px=right_state.point,
            reason_invalid=SceneInvalidReason.NON_FINITE_INPUT,
            source_reason_invalid=reason,
        )
        midpoint = SceneEyeMidpointRecord(
            valid=False,
            origin_policy=None,
            camera_point_m=None,
            scene_point_m=None,
            pupil_distance_px=pupil_distance_px,
            estimated_depth_m=None,
            source_reason_invalid=reason,
            reason_invalid=SceneInvalidReason.NON_FINITE_INPUT,
        )
        return _eye_pair_projection(
            left_eye=left_eye,
            right_eye=right_eye,
            midpoint=midpoint,
            diagnostics=diagnostics,
        )

    left_camera = _back_project_point(left_state.point, camera, depth_m)
    right_camera = _back_project_point(right_state.point, camera, depth_m)
    midpoint_camera = _camera_vector(
        x=(left_camera.x + right_camera.x) / 2.0,
        y=(left_camera.y + right_camera.y) / 2.0,
        z=(left_camera.z + right_camera.z) / 2.0,
    )

    left_eye = SceneEyeRecord(
        valid=True,
        image_px=left_state.point,
        camera_point_m=left_camera,
        scene_point_m=_scene_vector(
            x=left_camera.x - midpoint_camera.x,
            y=left_camera.y - midpoint_camera.y,
            z=left_camera.z - midpoint_camera.z,
        ),
        source_reason_invalid=None,
        reason_invalid=None,
    )
    right_eye = SceneEyeRecord(
        valid=True,
        image_px=right_state.point,
        camera_point_m=right_camera,
        scene_point_m=_scene_vector(
            x=right_camera.x - midpoint_camera.x,
            y=right_camera.y - midpoint_camera.y,
            z=right_camera.z - midpoint_camera.z,
        ),
        source_reason_invalid=None,
        reason_invalid=None,
    )
    midpoint = SceneEyeMidpointRecord(
        valid=True,
        origin_policy="both_eyes_required",
        camera_point_m=midpoint_camera,
        scene_point_m=_scene_vector(x=0.0, y=0.0, z=0.0),
        pupil_distance_px=pupil_distance_px,
        estimated_depth_m=depth_m,
        source_reason_invalid=None,
        reason_invalid=None,
    )
    return _eye_pair_projection(
        left_eye=left_eye,
        right_eye=right_eye,
        midpoint=midpoint,
        diagnostics=diagnostics,
    )


def robust_scene_center(
    points: Sequence[Vector3D],
    assumptions: SceneAssumptions,
) -> RobustPointEstimate:
    raise NotImplementedError("Task 3 implements robust_scene_center")


def unigaze_ray_from_frame(
    frame_record: FrameRecord,
    midpoint: SceneEyeMidpointRecord,
) -> SceneUniGazeRayRecord:
    raise NotImplementedError("Task 3 implements unigaze_ray_from_frame")


def robust_main_direction(
    rays: Sequence[SceneUniGazeRayRecord],
    assumptions: SceneAssumptions,
) -> RobustDirectionEstimate:
    raise NotImplementedError("Task 3 implements robust_main_direction")


def build_scene_axis_basis(
    direction: RobustDirectionEstimate,
    eye_pair_right_vectors: Sequence[UnitVector3D],
    assumptions: SceneAssumptions,
) -> SceneAxisBasisRecord:
    raise NotImplementedError("Task 4 implements build_scene_axis_basis")


def build_monitor_plane(
    center: RobustPointEstimate,
    direction: RobustDirectionEstimate,
    axes: SceneAxisBasisRecord,
    assumptions: SceneAssumptions,
) -> SceneMonitorPlaneRecord:
    raise NotImplementedError("Task 4 implements build_monitor_plane")


def camera_point_to_scene(
    point: Vector3D,
    center: Vector3D,
    axes: SceneAxisBasisRecord,
) -> Vector3D:
    raise NotImplementedError("Task 4 implements camera_point_to_scene")


def intersect_ray_with_monitor(
    ray: SceneUniGazeRayRecord,
    monitor: SceneMonitorPlaneRecord,
    assumptions: SceneAssumptions,
) -> SceneMonitorHitRecord:
    raise NotImplementedError("Task 4 implements intersect_ray_with_monitor")


@dataclass(frozen=True)
class _EyeProjectionState:
    kind: str
    point: Point2D | None
    reason_invalid: SceneInvalidReason
    source_reason_invalid: str
    coordinate_space: str | None


def _eye_projection_state(
    *,
    eye: EyeRecord,
    point: Point2D | None,
    invalid_reason: SceneInvalidReason,
    source_reason: str | None,
) -> _EyeProjectionState:
    if not eye.present or point is None:
        return _EyeProjectionState(
            kind="missing",
            point=point,
            reason_invalid=invalid_reason,
            source_reason_invalid=_first_reason(
                source_reason,
                "pupil center missing",
            ),
            coordinate_space=None,
        )
    if point.space != CoordinateSpace.IMAGE_PX:
        return _EyeProjectionState(
            kind="wrong_space",
            point=point,
            reason_invalid=invalid_reason,
            source_reason_invalid=(
                "pupil center must use IMAGE_PX, got "
                f"{point.space.value}"
            ),
            coordinate_space=point.space.value,
        )
    if not math.isfinite(point.x) or not math.isfinite(point.y):
        return _EyeProjectionState(
            kind="non_finite",
            point=point,
            reason_invalid=SceneInvalidReason.NON_FINITE_INPUT,
            source_reason_invalid="pupil center contains non-finite coordinates",
            coordinate_space=point.space.value,
        )
    return _EyeProjectionState(
        kind="projectable",
        point=point,
        reason_invalid=invalid_reason,
        source_reason_invalid=_first_reason(source_reason, "eye present"),
        coordinate_space=point.space.value,
    )


def _finalize_invalid_eye(
    *,
    state: _EyeProjectionState,
    point: Point2D | None,
    fallback_reason: str,
) -> SceneEyeRecord:
    if state.kind == "projectable":
        return _invalid_eye_record(
            image_px=point,
            reason_invalid=state.reason_invalid,
            source_reason_invalid=fallback_reason,
        )
    return _invalid_eye_record(
        image_px=point,
        reason_invalid=state.reason_invalid,
        source_reason_invalid=state.source_reason_invalid,
    )


def _invalid_eye_record(
    *,
    image_px: Point2D | None,
    reason_invalid: SceneInvalidReason,
    source_reason_invalid: str,
) -> SceneEyeRecord:
    persisted_image = image_px
    if image_px is not None and (
        not math.isfinite(image_px.x) or not math.isfinite(image_px.y)
    ):
        persisted_image = None
    return SceneEyeRecord(
        valid=False,
        image_px=persisted_image,
        camera_point_m=None,
        scene_point_m=None,
        source_reason_invalid=source_reason_invalid,
        reason_invalid=reason_invalid,
    )


def _back_project_point(
    point: Point2D,
    camera: SceneCameraModel,
    depth_m: float,
) -> Vector3D:
    return _camera_vector(
        x=((point.x - camera.cx_px) * depth_m) / camera.fx_px,
        y=((point.y - camera.cy_px) * depth_m) / camera.fy_px,
        z=depth_m,
    )


def _camera_vector(*, x: float, y: float, z: float) -> Vector3D:
    return Vector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=x,
        y=y,
        z=z,
    )


def _scene_vector(*, x: float, y: float, z: float) -> Vector3D:
    return Vector3D(
        space=CoordinateFrame3D.SCENE_PSEUDO_M,
        x=x,
        y=y,
        z=z,
    )


def _point_in_frame(
    point: Point2D | None,
    camera: SceneCameraModel,
) -> bool | None:
    if (
        point is None
        or point.space != CoordinateSpace.IMAGE_PX
        or not math.isfinite(point.x)
        or not math.isfinite(point.y)
    ):
        return None
    return (
        0.0 <= point.x < camera.frame_width_px
        and 0.0 <= point.y < camera.frame_height_px
    )


def _pupil_center(eye: EyeRecord) -> Point2D | None:
    return eye.pupil_center


def _source_reason(eye: EyeRecord) -> str | None:
    if eye.reason_invalid is None:
        return None
    return eye.reason_invalid.value


def _persist_invalid_coordinate_space_diagnostics(
    *,
    diagnostics: dict[str, str | int | float | bool | None],
    left_state: _EyeProjectionState,
    right_state: _EyeProjectionState,
) -> None:
    invalid_states: list[tuple[str, _EyeProjectionState]] = []
    if left_state.kind == "wrong_space":
        invalid_states.append(("left_eye", left_state))
    if right_state.kind == "wrong_space":
        invalid_states.append(("right_eye", right_state))
    if not invalid_states:
        return

    eye_names = ",".join(name for name, _ in invalid_states)
    coordinate_spaces = sorted(
        {
            state.coordinate_space
            for _, state in invalid_states
            if state.coordinate_space is not None
        }
    )
    diagnostics["invalid_coordinate_space_eyes"] = eye_names
    diagnostics["invalid_coordinate_space"] = ",".join(coordinate_spaces)


def _first_reason(*reasons: str | None) -> str:
    for reason in reasons:
        if reason:
            return reason
    raise ValueError("expected at least one non-empty reason")


def _eye_pair_projection(
    *,
    left_eye: SceneEyeRecord,
    right_eye: SceneEyeRecord,
    midpoint: SceneEyeMidpointRecord,
    diagnostics: dict[str, str | int | float | bool | None],
) -> SceneEyePairProjection:
    return SceneEyePairProjection(
        left_eye_valid=left_eye.valid,
        right_eye_valid=right_eye.valid,
        left_eye=left_eye,
        right_eye=right_eye,
        midpoint=midpoint,
        diagnostics=diagnostics,
    )
