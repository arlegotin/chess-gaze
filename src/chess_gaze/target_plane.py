from __future__ import annotations

import math
from dataclasses import dataclass

from chess_gaze.scene_records import (
    CoordinateFrame3D,
    SceneInvalidReason,
    SceneTargetPlaneHitRecord,
    UnitVector3D,
    Vector3D,
)

_EPSILON = 1e-9
_ORTHOGONAL_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ConfiguredTargetPlane:
    origin_camera_m: tuple[float, float, float]
    x_axis_camera: tuple[float, float, float]
    y_axis_camera: tuple[float, float, float]
    normal_camera: tuple[float, float, float]
    width_m: float
    height_m: float
    mirror_horizontal: bool


def build_configured_target_plane(
    *,
    origin_camera_m: tuple[float, float, float],
    x_axis_camera: tuple[float, float, float],
    y_axis_camera: tuple[float, float, float],
    width_m: float,
    height_m: float,
    mirror_horizontal: bool,
) -> ConfiguredTargetPlane:
    origin = _finite_triplet(origin_camera_m, "origin_camera_m")
    x_axis = _normalize_triplet(x_axis_camera, "x_axis_camera")
    y_axis = _normalize_triplet(y_axis_camera, "y_axis_camera")
    if not math.isfinite(width_m) or width_m <= 0.0:
        raise ValueError("target plane width_m must be positive and finite")
    if not math.isfinite(height_m) or height_m <= 0.0:
        raise ValueError("target plane height_m must be positive and finite")
    axis_dot = _dot(x_axis, y_axis)
    if abs(axis_dot) > _ORTHOGONAL_TOLERANCE:
        raise ValueError("target plane axes must be finite, non-zero, and orthogonal")
    normal = _cross(x_axis, y_axis)
    normal = _normalize_triplet(normal, "target plane axes normal")
    return ConfiguredTargetPlane(
        origin_camera_m=origin,
        x_axis_camera=x_axis,
        y_axis_camera=y_axis,
        normal_camera=normal,
        width_m=width_m,
        height_m=height_m,
        mirror_horizontal=mirror_horizontal,
    )


def intersect_ray_with_target_plane(
    *,
    origin_camera_m: Vector3D,
    direction_camera: tuple[float, float, float],
    plane: ConfiguredTargetPlane,
) -> SceneTargetPlaneHitRecord:
    if origin_camera_m.space != CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M:
        raise ValueError("origin_camera_m must use camera_opencv_pseudo_m")
    origin = (origin_camera_m.x, origin_camera_m.y, origin_camera_m.z)
    direction = _normalize_triplet(direction_camera, "direction_camera")
    denominator = _dot(plane.normal_camera, direction)
    if abs(denominator) <= _EPSILON:
        return _invalid_hit(
            SceneInvalidReason.RAY_TARGET_PLANE_PARALLEL,
            "gaze ray is parallel to target plane",
        )
    origin_to_plane = _subtract(plane.origin_camera_m, origin)
    ray_t_m = _dot(plane.normal_camera, origin_to_plane) / denominator
    if not math.isfinite(ray_t_m):
        return _invalid_hit(
            SceneInvalidReason.RAY_TARGET_PLANE_INTERSECTION_NON_FINITE,
            "ray-target-plane intersection distance is non-finite",
        )
    if ray_t_m < 0.0:
        return _invalid_hit(
            SceneInvalidReason.RAY_TARGET_PLANE_INTERSECTION_BEHIND_ORIGIN,
            "target plane intersection is behind the gaze origin",
        )

    point = _add(origin, _scale(direction, ray_t_m))
    if not all(math.isfinite(coordinate) for coordinate in point):
        return _invalid_hit(
            SceneInvalidReason.RAY_TARGET_PLANE_INTERSECTION_NON_FINITE,
            "ray-target-plane intersection point is non-finite",
        )

    plane_delta = _subtract(point, plane.origin_camera_m)
    target_x = _dot(plane_delta, plane.x_axis_camera) / plane.width_m
    target_y = _dot(plane_delta, plane.y_axis_camera) / plane.height_m
    if plane.mirror_horizontal:
        target_x = 1.0 - target_x
    if not math.isfinite(target_x) or not math.isfinite(target_y):
        return _invalid_hit(
            SceneInvalidReason.RAY_TARGET_PLANE_INTERSECTION_NON_FINITE,
            "target-plane normalized coordinates are non-finite",
        )
    return SceneTargetPlaneHitRecord(
        valid=True,
        point_camera_m=_vector(point),
        target_x_normalized=target_x,
        target_y_normalized=target_y,
        inside_bounds=(0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0),
        ray_t_m=ray_t_m,
        source_reason_invalid=None,
        reason_invalid=None,
    )


def target_plane_unit_vector(
    value: tuple[float, float, float],
) -> UnitVector3D:
    return UnitVector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=value[0],
        y=value[1],
        z=value[2],
    )


def target_plane_vector(value: tuple[float, float, float]) -> Vector3D:
    return _vector(value)


def _invalid_hit(
    reason: SceneInvalidReason, source_reason_invalid: str
) -> SceneTargetPlaneHitRecord:
    return SceneTargetPlaneHitRecord(
        valid=False,
        point_camera_m=None,
        target_x_normalized=None,
        target_y_normalized=None,
        inside_bounds=None,
        ray_t_m=None,
        source_reason_invalid=source_reason_invalid,
        reason_invalid=reason,
    )


def _vector(value: tuple[float, float, float]) -> Vector3D:
    return Vector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=value[0],
        y=value[1],
        z=value[2],
    )


def _finite_triplet(
    value: tuple[float, float, float], field_name: str
) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    triplet = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(coordinate) for coordinate in triplet):
        raise ValueError(f"{field_name} must contain only finite values")
    return triplet


def _normalize_triplet(
    value: tuple[float, float, float], field_name: str
) -> tuple[float, float, float]:
    triplet = _finite_triplet(value, field_name)
    norm = math.sqrt(_dot(triplet, triplet))
    if norm <= _EPSILON:
        raise ValueError(f"{field_name} must be non-zero")
    return (triplet[0] / norm, triplet[1] / norm, triplet[2] / norm)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return (left[0] * right[0]) + (left[1] * right[1]) + (left[2] * right[2])


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        (left[1] * right[2]) - (left[2] * right[1]),
        (left[2] * right[0]) - (left[0] * right[2]),
        (left[0] * right[1]) - (left[1] * right[0]),
    )


def _subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _add(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale(
    value: tuple[float, float, float], scalar: float
) -> tuple[float, float, float]:
    return (value[0] * scalar, value[1] * scalar, value[2] * scalar)
