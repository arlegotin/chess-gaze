from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from chess_gaze.frame_records import EyeRecord, FrameRecord
from chess_gaze.gaze_observation import pitch_yaw_to_unit_vector
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
    dropped_non_finite_count: int
    inlier_count: int
    mad_m: tuple[float, float, float]
    thresholds_m: tuple[float, float, float]
    iteration_count: int
    convergence_tolerance_m: float
    fallback_used: bool
    uncertainty: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RobustDirectionEstimate:
    direction_camera: UnitVector3D
    candidate_count: int
    finite_candidate_count: int
    inlier_count: int
    angle_threshold_radians: float
    median_angular_residual_radians: float | None
    angular_residual_percentiles_radians: dict[str, float | None]
    fallback_used: bool
    uncertainty: Literal["low", "medium", "high"]


_GEOMETRIC_MEDIAN_MAX_ITERATIONS = 128
_GEOMETRIC_MEDIAN_CONVERGENCE_TOLERANCE_M = 1e-6
_VECTOR_EPSILON = 1e-8
_RANSAC_SEED_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


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
                "paired pupil distance unavailable because an eye input is non-finite"
            ),
        )
        right_eye = _finalize_invalid_eye(
            state=right_state,
            point=right_point,
            fallback_reason=(
                "paired pupil distance unavailable because an eye input is non-finite"
            ),
        )
        midpoint = SceneEyeMidpointRecord(
            valid=False,
            origin_policy=None,
            camera_m=None,
            scene_m=None,
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
            camera_m=None,
            scene_m=None,
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

    assert left_state.point is not None
    assert right_state.point is not None
    pupil_distance_px = math.hypot(
        right_state.point.x - left_state.point.x,
        right_state.point.y - left_state.point.y,
    )
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
            camera_m=None,
            scene_m=None,
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
            camera_m=None,
            scene_m=None,
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
            camera_m=None,
            scene_m=None,
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
        camera_m=left_camera,
        scene_m=_scene_vector(
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
        camera_m=right_camera,
        scene_m=_scene_vector(
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
        camera_m=midpoint_camera,
        scene_m=_scene_vector(x=0.0, y=0.0, z=0.0),
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
    candidate_count = len(points)
    finite_points = [point for point in points if _is_finite_vector(point)]
    finite_candidate_count = len(finite_points)
    dropped_non_finite_count = candidate_count - finite_candidate_count
    thresholds = (
        assumptions.scene_center_min_axis_tolerance_m,
        assumptions.scene_center_min_axis_tolerance_m,
        assumptions.scene_center_min_axis_tolerance_m,
    )
    mad_m = (0.0, 0.0, 0.0)

    if finite_points:
        medians = (
            _median(point.x for point in finite_points),
            _median(point.y for point in finite_points),
            _median(point.z for point in finite_points),
        )
        mad_m = (
            _median(abs(point.x - medians[0]) for point in finite_points),
            _median(abs(point.y - medians[1]) for point in finite_points),
            _median(abs(point.z - medians[2]) for point in finite_points),
        )
        thresholds = (
            max(3.5 * mad_m[0], assumptions.scene_center_min_axis_tolerance_m),
            max(3.5 * mad_m[1], assumptions.scene_center_min_axis_tolerance_m),
            max(3.5 * mad_m[2], assumptions.scene_center_min_axis_tolerance_m),
        )
        inliers = [
            point
            for point in finite_points
            if abs(point.x - medians[0]) <= thresholds[0]
            and abs(point.y - medians[1]) <= thresholds[1]
            and abs(point.z - medians[2]) <= thresholds[2]
        ]
    else:
        inliers = []

    if len(inliers) < assumptions.min_scene_center_inlier_frames:
        return RobustPointEstimate(
            point_camera_m=_camera_vector(
                x=assumptions.default_scene_center_camera_m[0],
                y=assumptions.default_scene_center_camera_m[1],
                z=assumptions.default_scene_center_camera_m[2],
            ),
            candidate_count=candidate_count,
            finite_candidate_count=finite_candidate_count,
            dropped_non_finite_count=dropped_non_finite_count,
            inlier_count=len(inliers),
            mad_m=mad_m,
            thresholds_m=thresholds,
            iteration_count=0,
            convergence_tolerance_m=_GEOMETRIC_MEDIAN_CONVERGENCE_TOLERANCE_M,
            fallback_used=True,
            uncertainty="high",
        )

    estimate, iteration_count = _geometric_median_camera_point(inliers)
    return RobustPointEstimate(
        point_camera_m=estimate,
        candidate_count=candidate_count,
        finite_candidate_count=finite_candidate_count,
        dropped_non_finite_count=dropped_non_finite_count,
        inlier_count=len(inliers),
        mad_m=mad_m,
        thresholds_m=thresholds,
        iteration_count=iteration_count,
        convergence_tolerance_m=_GEOMETRIC_MEDIAN_CONVERGENCE_TOLERANCE_M,
        fallback_used=False,
        uncertainty="medium",
    )


def unigaze_ray_from_frame(
    frame_record: FrameRecord,
    midpoint: SceneEyeMidpointRecord,
) -> SceneUniGazeRayRecord:
    if (
        not midpoint.valid
        or midpoint.camera_point_m is None
        or midpoint.scene_point_m is None
        or not _is_finite_vector(midpoint.camera_point_m)
        or not _is_finite_vector(midpoint.scene_point_m)
    ):
        return SceneUniGazeRayRecord(
            valid=False,
            source="appearance_gaze",
            origin_camera_m=None,
            scene_m=None,
            direction_camera=None,
            direction_scene=None,
            direction_source=None,
            pitch_radians=None,
            yaw_radians=None,
            source_reason_invalid=_first_reason(
                midpoint.source_reason_invalid,
                "eye midpoint unavailable",
            ),
            reason_invalid=SceneInvalidReason.EYE_MIDPOINT_INVALID,
        )

    appearance_gaze = frame_record.appearance_gaze
    if (
        not appearance_gaze.valid
        or appearance_gaze.pitch_radians is None
        or appearance_gaze.yaw_radians is None
    ):
        source_reason = None
        if appearance_gaze.reason_invalid is not None:
            source_reason = appearance_gaze.reason_invalid.value
        return SceneUniGazeRayRecord(
            valid=False,
            source="appearance_gaze",
            origin_camera_m=None,
            scene_m=None,
            direction_camera=None,
            direction_scene=None,
            direction_source=None,
            pitch_radians=None,
            yaw_radians=None,
            source_reason_invalid=_first_reason(
                source_reason,
                "appearance gaze unavailable",
            ),
            reason_invalid=SceneInvalidReason.UNIGAZE_INVALID,
        )

    direction_xyz = pitch_yaw_to_unit_vector(
        pitch_radians=appearance_gaze.pitch_radians,
        yaw_radians=appearance_gaze.yaw_radians,
    )
    normalized_direction = _normalize_tuple(direction_xyz)
    if normalized_direction is None:
        return SceneUniGazeRayRecord(
            valid=False,
            source="appearance_gaze",
            origin_camera_m=None,
            scene_m=None,
            direction_camera=None,
            direction_scene=None,
            direction_source=None,
            pitch_radians=None,
            yaw_radians=None,
            source_reason_invalid="appearance gaze direction is non-finite",
            reason_invalid=SceneInvalidReason.UNIGAZE_INVALID,
        )

    return SceneUniGazeRayRecord(
        valid=True,
        source="appearance_gaze",
        origin_camera_m=midpoint.camera_point_m,
        scene_m=midpoint.scene_point_m,
        direction_camera=_camera_unit_vector(normalized_direction),
        direction_scene=_scene_unit_vector(normalized_direction),
        direction_source="appearance_gaze_unigaze_pitch_yaw",
        pitch_radians=appearance_gaze.pitch_radians,
        yaw_radians=appearance_gaze.yaw_radians,
        source_reason_invalid=None,
        reason_invalid=None,
    )


def robust_main_direction(
    rays: Sequence[SceneUniGazeRayRecord],
    assumptions: SceneAssumptions,
) -> RobustDirectionEstimate:
    candidate_count = len(rays)
    candidate_directions: list[tuple[int, tuple[float, float, float]]] = []
    for frame_order, ray in enumerate(rays):
        if not ray.valid or ray.direction_camera is None:
            continue
        normalized_direction = _normalized_direction(ray.direction_camera)
        if normalized_direction is None:
            continue
        candidate_directions.append((frame_order, normalized_direction))

    finite_candidate_count = len(candidate_directions)
    best_inlier_indices: list[int] = []
    best_seed_frame_index = candidate_count + 1
    best_median_residual: float | None = None
    best_seed_direction: tuple[float, float, float] | None = None
    best_inlier_residuals: list[float] = []

    for seed_frame_index, seed_direction in _direction_ransac_seeds(
        candidate_directions
    ):
        residuals = [
            _angular_distance(seed_direction, candidate_direction)
            for _frame_order, candidate_direction in candidate_directions
        ]
        inlier_indices = [
            index
            for index, residual in enumerate(residuals)
            if residual <= assumptions.direction_inlier_angle_radians
        ]
        if not inlier_indices:
            continue
        median_residual = _median(residuals[index] for index in inlier_indices)
        if _seed_is_better(
            candidate_inlier_count=len(inlier_indices),
            candidate_median_residual=median_residual,
            candidate_seed_frame_index=seed_frame_index,
            best_inlier_count=len(best_inlier_indices),
            best_median_residual=best_median_residual,
            best_seed_frame_index=best_seed_frame_index,
        ):
            best_inlier_indices = inlier_indices
            best_seed_frame_index = seed_frame_index
            best_median_residual = median_residual
            best_seed_direction = seed_direction
            best_inlier_residuals = [residuals[index] for index in inlier_indices]

    residual_percentiles = _angular_residual_percentiles(best_inlier_residuals)
    fallback_direction = _camera_unit_vector((0.0, 0.0, 1.0))
    if (
        len(best_inlier_indices) < assumptions.min_main_direction_inlier_frames
        or best_seed_direction is None
    ):
        return RobustDirectionEstimate(
            direction_camera=fallback_direction,
            candidate_count=candidate_count,
            finite_candidate_count=finite_candidate_count,
            inlier_count=len(best_inlier_indices),
            angle_threshold_radians=assumptions.direction_inlier_angle_radians,
            median_angular_residual_radians=best_median_residual,
            angular_residual_percentiles_radians=residual_percentiles,
            fallback_used=True,
            uncertainty="high",
        )

    inlier_directions = [
        candidate_directions[index][1] for index in best_inlier_indices
    ]
    mean_direction = _normalize_tuple(
        (
            sum(direction[0] for direction in inlier_directions),
            sum(direction[1] for direction in inlier_directions),
            sum(direction[2] for direction in inlier_directions),
        )
    )
    if mean_direction is None:
        return RobustDirectionEstimate(
            direction_camera=fallback_direction,
            candidate_count=candidate_count,
            finite_candidate_count=finite_candidate_count,
            inlier_count=len(best_inlier_indices),
            angle_threshold_radians=assumptions.direction_inlier_angle_radians,
            median_angular_residual_radians=best_median_residual,
            angular_residual_percentiles_radians=residual_percentiles,
            fallback_used=True,
            uncertainty="high",
        )

    return RobustDirectionEstimate(
        direction_camera=_camera_unit_vector(mean_direction),
        candidate_count=candidate_count,
        finite_candidate_count=finite_candidate_count,
        inlier_count=len(best_inlier_indices),
        angle_threshold_radians=assumptions.direction_inlier_angle_radians,
        median_angular_residual_radians=best_median_residual,
        angular_residual_percentiles_radians=residual_percentiles,
        fallback_used=False,
        uncertainty="medium",
    )


def build_scene_axis_basis(
    direction: RobustDirectionEstimate,
    eye_pair_right_vectors: Sequence[UnitVector3D],
    assumptions: SceneAssumptions,
) -> SceneAxisBasisRecord:
    del assumptions
    fallbacks: list[str] = []
    forward_direction = _normalized_direction(direction.direction_camera)
    if forward_direction is None:
        forward_direction = (0.0, 0.0, 1.0)
        _append_fallback(
            fallbacks,
            "forward_axis_invalid_used_default_forward",
        )
    back_direction = _negate_tuple(forward_direction)
    preferred_right = _preferred_right_direction(eye_pair_right_vectors, fallbacks)

    right_direction = _project_onto_plane(preferred_right, back_direction)
    if right_direction is None:
        _append_fallback(
            fallbacks,
            "right_axis_parallel_to_forward_used_camera_right",
        )
        right_direction = _project_onto_plane((1.0, 0.0, 0.0), back_direction)
    if right_direction is None:
        _append_fallback(
            fallbacks,
            "right_axis_camera_right_parallel_used_camera_forward",
        )
        right_direction = _project_onto_plane((0.0, 0.0, 1.0), back_direction)
    if right_direction is None:
        right_direction = _project_onto_plane((0.0, -1.0, 0.0), back_direction)
    if right_direction is None:
        right_direction = (1.0, 0.0, 0.0)

    preferred_up = (0.0, -1.0, 0.0)
    up_direction = _project_onto_plane(preferred_up, back_direction)
    if up_direction is None:
        _append_fallback(
            fallbacks,
            "up_axis_camera_up_parallel_to_back_used_cross_product",
        )
        up_direction = _normalize_tuple(_cross(back_direction, right_direction))
    if up_direction is None:
        up_direction = (0.0, 0.0, -1.0)

    right_direction = _normalize_tuple(_cross(up_direction, back_direction))
    if right_direction is None:
        right_direction = (1.0, 0.0, 0.0)
        _append_fallback(
            fallbacks,
            "right_axis_cross_product_invalid_used_camera_right",
        )
    up_direction = _normalize_tuple(_cross(back_direction, right_direction))
    if up_direction is None:
        up_direction = (0.0, 0.0, -1.0)
        _append_fallback(
            fallbacks,
            "up_axis_cross_product_invalid_used_camera_forward",
        )

    determinant = _determinant(right_direction, up_direction, back_direction)
    return SceneAxisBasisRecord(
        right_camera=_camera_unit_vector(right_direction),
        up_camera=_camera_unit_vector(up_direction),
        back_camera=_camera_unit_vector(back_direction),
        forward_camera=_camera_unit_vector(forward_direction),
        determinant_right_up_back=determinant,
        convention="right_up_back_columns_right_handed",
        fallbacks=fallbacks,
    )


def build_monitor_plane(
    center: RobustPointEstimate,
    direction: RobustDirectionEstimate,
    axes: SceneAxisBasisRecord,
    assumptions: SceneAssumptions,
) -> SceneMonitorPlaneRecord:
    forward_direction = _normalized_direction(direction.direction_camera)
    if forward_direction is None:
        forward_direction = _vector_tuple(axes.forward_camera)

    center_camera = center.point_camera_m
    monitor_center_camera = _camera_vector(
        x=center_camera.x
        + (forward_direction[0] * assumptions.monitor_distance_from_eyes_m),
        y=center_camera.y
        + (forward_direction[1] * assumptions.monitor_distance_from_eyes_m),
        z=center_camera.z
        + (forward_direction[2] * assumptions.monitor_distance_from_eyes_m),
    )

    return SceneMonitorPlaneRecord(
        center_camera_m=monitor_center_camera,
        center_scene_m=camera_point_to_scene(
            monitor_center_camera,
            center_camera,
            axes,
        ),
        normal_camera=_camera_unit_vector(_negate_tuple(forward_direction)),
        right_camera=axes.right_camera,
        up_camera=axes.up_camera,
        physical_width_m=assumptions.monitor_width_m,
        physical_height_m=assumptions.monitor_height_m,
        extended_width_m=assumptions.monitor_width_m * assumptions.extended_plane_scale,
        extended_height_m=(
            assumptions.monitor_height_m * assumptions.extended_plane_scale
        ),
        distance_from_scene_center_m=assumptions.monitor_distance_from_eyes_m,
        distance_source="DEFAULT_MONITOR_DISTANCE_FROM_EYES_M",
    )


def camera_point_to_scene(
    point: Vector3D,
    center: Vector3D,
    axes: SceneAxisBasisRecord,
) -> Vector3D:
    relative = _subtract(_vector_tuple(point), _vector_tuple(center))
    return _scene_vector(
        x=_dot(relative, _vector_tuple(axes.right_camera)),
        y=_dot(relative, _vector_tuple(axes.up_camera)),
        z=_dot(relative, _vector_tuple(axes.back_camera)),
    )


def intersect_ray_with_monitor(
    ray: SceneUniGazeRayRecord,
    monitor: SceneMonitorPlaneRecord,
    assumptions: SceneAssumptions,
) -> SceneMonitorHitRecord:
    if not ray.valid or ray.origin_camera_m is None or ray.direction_camera is None:
        return _invalid_monitor_hit(
            reason_invalid=ray.reason_invalid or SceneInvalidReason.UNIGAZE_INVALID,
            source_reason_invalid=_first_reason(
                ray.source_reason_invalid,
                "unigaze ray unavailable",
            ),
        )

    origin = _vector_tuple(ray.origin_camera_m)
    direction = _vector_tuple(ray.direction_camera)
    plane_center = _vector_tuple(monitor.center_camera_m)
    normal = _vector_tuple(monitor.normal_camera)
    origin_to_plane = _subtract(origin, plane_center)

    denominator = _dot(direction, normal)
    signed_distance = _dot(origin_to_plane, normal)
    persisted_denominator = _finite_or_none(denominator)
    persisted_signed_distance = _finite_or_none(signed_distance)

    if persisted_denominator is None or persisted_signed_distance is None:
        return _invalid_monitor_hit(
            reason_invalid=SceneInvalidReason.RAY_INTERSECTION_NON_FINITE,
            denominator=persisted_denominator,
            signed_distance_m=persisted_signed_distance,
            source_reason_invalid="ray-plane denominator or distance is non-finite",
        )

    if abs(denominator) < assumptions.ray_plane_parallel_epsilon:
        if abs(signed_distance) < assumptions.ray_plane_parallel_epsilon:
            return _invalid_monitor_hit(
                reason_invalid=SceneInvalidReason.RAY_COPLANAR_WITH_MONITOR,
                denominator=denominator,
                signed_distance_m=signed_distance,
                source_reason_invalid="ray origin lies on monitor plane",
            )
        return _invalid_monitor_hit(
            reason_invalid=SceneInvalidReason.RAY_PARALLEL_TO_MONITOR,
            denominator=denominator,
            signed_distance_m=signed_distance,
            source_reason_invalid="ray direction is parallel to monitor plane",
        )

    ray_t = -signed_distance / denominator
    if not math.isfinite(ray_t):
        return _invalid_monitor_hit(
            reason_invalid=SceneInvalidReason.RAY_INTERSECTION_NON_FINITE,
            denominator=denominator,
            signed_distance_m=signed_distance,
            source_reason_invalid="ray-plane intersection t is non-finite",
        )
    if ray_t < 0.0:
        return _invalid_monitor_hit(
            reason_invalid=SceneInvalidReason.RAY_INTERSECTION_BEHIND_ORIGIN,
            denominator=denominator,
            signed_distance_m=signed_distance,
            t=ray_t,
            source_reason_invalid="ray-plane intersection is behind ray origin",
        )

    hit_xyz = _add(origin, _scale(direction, ray_t))
    if not all(math.isfinite(value) for value in hit_xyz):
        return _invalid_monitor_hit(
            reason_invalid=SceneInvalidReason.RAY_INTERSECTION_NON_FINITE,
            denominator=denominator,
            signed_distance_m=signed_distance,
            t=ray_t,
            source_reason_invalid="ray-plane intersection point is non-finite",
        )

    hit_from_center = _subtract(hit_xyz, plane_center)
    u_m = _dot(hit_from_center, _vector_tuple(monitor.right_camera))
    v_m = _dot(hit_from_center, _vector_tuple(monitor.up_camera))
    if not math.isfinite(u_m) or not math.isfinite(v_m):
        return _invalid_monitor_hit(
            reason_invalid=SceneInvalidReason.RAY_INTERSECTION_NON_FINITE,
            denominator=denominator,
            signed_distance_m=signed_distance,
            t=ray_t,
            source_reason_invalid="monitor plane coordinates are non-finite",
        )

    scene_center_camera = _monitor_scene_center_camera(monitor)
    hit_camera = _camera_vector(x=hit_xyz[0], y=hit_xyz[1], z=hit_xyz[2])
    return SceneMonitorHitRecord(
        valid=True,
        point_camera_m=hit_camera,
        point_scene_m=_camera_point_to_scene_from_monitor(
            hit_camera,
            scene_center_camera,
            monitor,
        ),
        plane_uv_m=(u_m, v_m),
        ray_t_m=ray_t,
        denominator=denominator,
        signed_origin_distance_m=signed_distance,
        within_physical_monitor=_within_rectangle(
            u_m,
            v_m,
            width_m=monitor.width_m,
            height_m=monitor.height_m,
        ),
        within_extended_plane=_within_rectangle(
            u_m,
            v_m,
            width_m=monitor.extended_width_m,
            height_m=monitor.extended_height_m,
        ),
        source_reason_invalid=None,
        reason_invalid=None,
    )


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
                f"pupil center must use IMAGE_PX, got {point.space.value}"
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
        camera_m=None,
        scene_m=None,
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


def _camera_unit_vector(
    xyz: tuple[float, float, float],
) -> UnitVector3D:
    return UnitVector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
    )


def _scene_unit_vector(
    xyz: tuple[float, float, float],
) -> UnitVector3D:
    return UnitVector3D(
        space=CoordinateFrame3D.SCENE_PSEUDO_M,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
    )


def _vector_tuple(vector: Vector3D | UnitVector3D) -> tuple[float, float, float]:
    return (vector.x, vector.y, vector.z)


def _finite_or_none(value: float) -> float | None:
    if math.isfinite(value):
        return value
    return None


def _invalid_monitor_hit(
    *,
    reason_invalid: SceneInvalidReason,
    denominator: float | None = None,
    signed_distance_m: float | None = None,
    t: float | None = None,
    source_reason_invalid: str,
) -> SceneMonitorHitRecord:
    return SceneMonitorHitRecord(
        valid=False,
        point_camera_m=None,
        point_scene_m=None,
        plane_uv_m=None,
        ray_t_m=_finite_or_none(t) if t is not None else None,
        denominator=(_finite_or_none(denominator) if denominator is not None else None),
        signed_origin_distance_m=(
            _finite_or_none(signed_distance_m)
            if signed_distance_m is not None
            else None
        ),
        within_physical_monitor=None,
        within_extended_plane=None,
        source_reason_invalid=source_reason_invalid,
        reason_invalid=reason_invalid,
    )


def _within_rectangle(
    u_m: float,
    v_m: float,
    *,
    width_m: float,
    height_m: float,
) -> bool:
    return abs(u_m) <= (width_m / 2.0) and abs(v_m) <= (height_m / 2.0)


def _monitor_scene_center_camera(
    monitor: SceneMonitorPlaneRecord,
) -> tuple[float, float, float]:
    center_scene = _vector_tuple(monitor.center_scene_m)
    scene_center_offset = _add(
        _add(
            _scale(_vector_tuple(monitor.right_camera), center_scene[0]),
            _scale(_vector_tuple(monitor.up_camera), center_scene[1]),
        ),
        _scale(_vector_tuple(monitor.normal_camera), center_scene[2]),
    )
    return _subtract(_vector_tuple(monitor.center_camera_m), scene_center_offset)


def _camera_point_to_scene_from_monitor(
    point: Vector3D,
    scene_center_camera: tuple[float, float, float],
    monitor: SceneMonitorPlaneRecord,
) -> Vector3D:
    relative = _subtract(_vector_tuple(point), scene_center_camera)
    return _scene_vector(
        x=_dot(relative, _vector_tuple(monitor.right_camera)),
        y=_dot(relative, _vector_tuple(monitor.up_camera)),
        z=_dot(relative, _vector_tuple(monitor.normal_camera)),
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


def _is_finite_vector(vector: Vector3D) -> bool:
    return (
        math.isfinite(vector.x) and math.isfinite(vector.y) and math.isfinite(vector.z)
    )


def _normalized_direction(
    vector: UnitVector3D | Vector3D,
) -> tuple[float, float, float] | None:
    return _normalize_tuple((vector.x, vector.y, vector.z))


def _normalize_tuple(
    xyz: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    if not all(math.isfinite(value) for value in xyz):
        return None
    norm = math.sqrt((xyz[0] * xyz[0]) + (xyz[1] * xyz[1]) + (xyz[2] * xyz[2]))
    if norm <= _VECTOR_EPSILON:
        return None
    return (xyz[0] / norm, xyz[1] / norm, xyz[2] / norm)


def _median(values: Iterable[float]) -> float:
    sorted_values: list[float] = sorted(values)
    if not sorted_values:
        raise ValueError("median requires at least one value")
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0)


def _angular_residual_percentiles(
    residuals: Sequence[float],
) -> dict[str, float | None]:
    if not residuals:
        return {"p50": None, "p75": None, "p90": None, "p95": None}

    sorted_residuals = sorted(residuals)
    return {
        "p50": _percentile(sorted_residuals, 50.0),
        "p75": _percentile(sorted_residuals, 75.0),
        "p90": _percentile(sorted_residuals, 90.0),
        "p95": _percentile(sorted_residuals, 95.0),
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return float(lower_value + ((upper_value - lower_value) * weight))


def _geometric_median_camera_point(
    points: Sequence[Vector3D],
) -> tuple[Vector3D, int]:
    current = (
        _median(point.x for point in points),
        _median(point.y for point in points),
        _median(point.z for point in points),
    )
    for iteration_count in range(1, _GEOMETRIC_MEDIAN_MAX_ITERATIONS + 1):
        numerator = [0.0, 0.0, 0.0]
        denominator = 0.0
        coincident_count = 0
        for point in points:
            point_xyz = (point.x, point.y, point.z)
            distance = _distance(current, point_xyz)
            if distance <= _GEOMETRIC_MEDIAN_CONVERGENCE_TOLERANCE_M:
                coincident_count += 1
                continue
            weight = 1.0 / distance
            numerator[0] += point.x * weight
            numerator[1] += point.y * weight
            numerator[2] += point.z * weight
            denominator += weight
        if denominator <= _VECTOR_EPSILON:
            return (
                _camera_vector(x=current[0], y=current[1], z=current[2]),
                iteration_count,
            )
        next_point = (
            numerator[0] / denominator,
            numerator[1] / denominator,
            numerator[2] / denominator,
        )
        if coincident_count > 0:
            update_distance = _distance(current, next_point)
            if update_distance <= _GEOMETRIC_MEDIAN_CONVERGENCE_TOLERANCE_M:
                return (
                    _camera_vector(x=current[0], y=current[1], z=current[2]),
                    iteration_count,
                )
            resultant_norm = denominator * update_distance
            if resultant_norm <= (
                coincident_count + _GEOMETRIC_MEDIAN_CONVERGENCE_TOLERANCE_M
            ):
                return (
                    _camera_vector(x=current[0], y=current[1], z=current[2]),
                    iteration_count,
                )
            coincidence_weight = coincident_count / resultant_norm
            next_point = (
                ((1.0 - coincidence_weight) * next_point[0])
                + (coincidence_weight * current[0]),
                ((1.0 - coincidence_weight) * next_point[1])
                + (coincidence_weight * current[1]),
                ((1.0 - coincidence_weight) * next_point[2])
                + (coincidence_weight * current[2]),
            )
        if _distance(current, next_point) <= _GEOMETRIC_MEDIAN_CONVERGENCE_TOLERANCE_M:
            return (
                _camera_vector(
                    x=next_point[0],
                    y=next_point[1],
                    z=next_point[2],
                ),
                iteration_count,
            )
        current = next_point
    return (
        _camera_vector(x=current[0], y=current[1], z=current[2]),
        _GEOMETRIC_MEDIAN_MAX_ITERATIONS,
    )


def _distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return math.sqrt(
        ((left[0] - right[0]) ** 2)
        + ((left[1] - right[1]) ** 2)
        + ((left[2] - right[2]) ** 2)
    )


def _direction_ransac_seeds(
    candidate_directions: Sequence[tuple[int, tuple[float, float, float]]],
) -> list[tuple[int, tuple[float, float, float]]]:
    if not candidate_directions:
        return []
    seeds: list[tuple[int, tuple[float, float, float]]] = []
    used_candidate_indices: set[int] = set()
    candidate_count = len(candidate_directions)
    for quantile in _RANSAC_SEED_QUANTILES:
        candidate_index = round((candidate_count - 1) * quantile)
        if candidate_index in used_candidate_indices:
            continue
        used_candidate_indices.add(candidate_index)
        seeds.append(candidate_directions[candidate_index])

    median_seed = _normalize_tuple(
        (
            _median(direction[0] for _frame_order, direction in candidate_directions),
            _median(direction[1] for _frame_order, direction in candidate_directions),
            _median(direction[2] for _frame_order, direction in candidate_directions),
        )
    )
    if median_seed is not None:
        seeds.append((candidate_directions[-1][0] + 1, median_seed))
    return seeds


def _angular_distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    dot_product = max(-1.0, min(1.0, _dot(left, right)))
    return math.acos(dot_product)


def _seed_is_better(
    *,
    candidate_inlier_count: int,
    candidate_median_residual: float,
    candidate_seed_frame_index: int,
    best_inlier_count: int,
    best_median_residual: float | None,
    best_seed_frame_index: int,
) -> bool:
    if candidate_inlier_count != best_inlier_count:
        return candidate_inlier_count > best_inlier_count
    if best_median_residual is None:
        return True
    if not math.isclose(candidate_median_residual, best_median_residual):
        return candidate_median_residual < best_median_residual
    return candidate_seed_frame_index < best_seed_frame_index


def _preferred_right_direction(
    eye_pair_right_vectors: Sequence[UnitVector3D],
    fallbacks: list[str],
) -> tuple[float, float, float]:
    valid_vectors = [
        normalized_direction
        for vector in eye_pair_right_vectors
        if (normalized_direction := _normalized_direction(vector)) is not None
    ]
    if not valid_vectors:
        _append_fallback(
            fallbacks,
            "right_axis_missing_eye_pair_evidence_used_camera_right",
        )
        return (1.0, 0.0, 0.0)
    mean_direction = _normalize_tuple(
        (
            sum(vector[0] for vector in valid_vectors),
            sum(vector[1] for vector in valid_vectors),
            sum(vector[2] for vector in valid_vectors),
        )
    )
    if mean_direction is None:
        _append_fallback(
            fallbacks,
            "right_axis_eye_pair_mean_degenerate_used_camera_right",
        )
        return (1.0, 0.0, 0.0)
    return mean_direction


def _project_onto_plane(
    vector: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    projected = _subtract(
        vector,
        _scale(plane_normal, _dot(vector, plane_normal)),
    )
    return _normalize_tuple(projected)


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return (left[0] * right[0]) + (left[1] * right[1]) + (left[2] * right[2])


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        (left[1] * right[2]) - (left[2] * right[1]),
        (left[2] * right[0]) - (left[0] * right[2]),
        (left[0] * right[1]) - (left[1] * right[0]),
    )


def _scale(
    vector: tuple[float, float, float],
    scalar: float,
) -> tuple[float, float, float]:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _add(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _negate_tuple(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (-vector[0], -vector[1], -vector[2])


def _determinant(
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    back: tuple[float, float, float],
) -> float:
    return (
        right[0] * ((up[1] * back[2]) - (up[2] * back[1]))
        - right[1] * ((up[0] * back[2]) - (up[2] * back[0]))
        + right[2] * ((up[0] * back[1]) - (up[1] * back[0]))
    )


def _append_fallback(fallbacks: list[str], reason: str) -> None:
    if reason not in fallbacks:
        fallbacks.append(reason)


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
