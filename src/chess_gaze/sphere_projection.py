from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict

from chess_gaze.geometry import StrictSchemaModel
from chess_gaze.scene_calibration import SceneAssumptions
from chess_gaze.scene_records import (
    CoordinateFrame3D,
    SceneInvalidReason,
    UnitVector3D,
    Vector3D,
)

_INTERSECTION_EPSILON = 1e-9
_EQUATOR_EPSILON = 1e-9


class GazeSphereSurface(StrictSchemaModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    center_scene_m: Vector3D
    radius_m: float
    radius_source: Literal["DEFAULT_GAZE_SPHERE_RADIUS_M"]
    center_source: Literal["robust_scene_center"]


class SphereHitResult(StrictSchemaModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    valid: bool
    point_scene_m: Vector3D | None = None
    ray_t_m: float | None = None
    radius_m: float | None = None
    theta_radians: float | None = None
    phi_radians: float | None = None
    hemisphere: Literal["front", "rear", "equator"] | None = None
    source_reason_invalid: str | None = None
    reason_invalid: SceneInvalidReason | None = None


@dataclass(frozen=True)
class _QuadraticRoots:
    near: float
    far: float


def build_gaze_sphere(assumptions: SceneAssumptions) -> GazeSphereSurface:
    return GazeSphereSurface(
        center_scene_m=_scene_vector(0.0, 0.0, 0.0),
        radius_m=assumptions.gaze_sphere_radius_m,
        radius_source="DEFAULT_GAZE_SPHERE_RADIUS_M",
        center_source="robust_scene_center",
    )


def intersect_ray_with_sphere(
    *,
    origin_scene_m: Vector3D | None,
    direction_scene: UnitVector3D | None,
    sphere: GazeSphereSurface,
    source_reason_invalid: str | None = None,
    invalid_reason: SceneInvalidReason = SceneInvalidReason.UNIGAZE_INVALID,
) -> SphereHitResult:
    if not _valid_radius(sphere.radius_m):
        return _invalid_hit(
            SceneInvalidReason.SPHERE_RADIUS_INVALID,
            "sphere radius must be finite and > 0",
        )
    if origin_scene_m is None or direction_scene is None:
        return _invalid_hit(
            invalid_reason,
            source_reason_invalid or "ray origin or direction unavailable",
        )
    if (
        origin_scene_m.space != CoordinateFrame3D.SCENE_PSEUDO_M
        or direction_scene.space != CoordinateFrame3D.SCENE_PSEUDO_M
    ):
        return _invalid_hit(
            SceneInvalidReason.NON_FINITE_INPUT,
            "ray origin and direction must use scene_pseudo_m",
        )

    origin = _tuple(origin_scene_m)
    direction = _normalize(_tuple(direction_scene))
    center = _tuple(sphere.center_scene_m)
    if direction is None or not _finite_tuple(origin) or not _finite_tuple(center):
        return _invalid_hit(
            SceneInvalidReason.NON_FINITE_INPUT,
            "ray-sphere input contains non-finite values",
        )

    roots = _ray_sphere_roots(
        origin=origin,
        direction=direction,
        center=center,
        radius=sphere.radius_m,
    )
    if roots is None:
        return _invalid_hit(
            SceneInvalidReason.RAY_SPHERE_DISCRIMINANT_NEGATIVE,
            "ray does not intersect gaze sphere",
        )

    candidates = [
        root
        for root in (roots.near, roots.far)
        if math.isfinite(root) and root >= -_INTERSECTION_EPSILON
    ]
    if not candidates:
        return _invalid_hit(
            SceneInvalidReason.RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN,
            "ray-sphere intersections are behind ray origin",
        )
    ray_t = max(0.0, min(candidates))
    point_tuple = (
        origin[0] + (direction[0] * ray_t),
        origin[1] + (direction[1] * ray_t),
        origin[2] + (direction[2] * ray_t),
    )
    if not _finite_tuple(point_tuple):
        return _invalid_hit(
            SceneInvalidReason.RAY_SPHERE_INTERSECTION_NON_FINITE,
            "ray-sphere intersection point is non-finite",
        )

    point = _scene_vector(point_tuple[0], point_tuple[1], point_tuple[2])
    relative = (
        point_tuple[0] - center[0],
        point_tuple[1] - center[1],
        point_tuple[2] - center[2],
    )
    theta = math.atan2(relative[0], -relative[2])
    phi = math.asin(max(-1.0, min(1.0, relative[1] / sphere.radius_m)))
    return SphereHitResult(
        valid=True,
        point_scene_m=point,
        ray_t_m=ray_t,
        radius_m=sphere.radius_m,
        theta_radians=theta,
        phi_radians=phi,
        hemisphere=_hemisphere(relative[2]),
        source_reason_invalid=None,
        reason_invalid=None,
    )


def _ray_sphere_roots(
    *,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    center: tuple[float, float, float],
    radius: float,
) -> _QuadraticRoots | None:
    oc = (
        origin[0] - center[0],
        origin[1] - center[1],
        origin[2] - center[2],
    )
    a = _dot(direction, direction)
    b = 2.0 * _dot(oc, direction)
    c = _dot(oc, oc) - (radius * radius)
    discriminant = (b * b) - (4.0 * a * c)
    if not math.isfinite(discriminant):
        return None
    if discriminant < -_INTERSECTION_EPSILON:
        return None
    sqrt_discriminant = math.sqrt(max(0.0, discriminant))
    denominator = 2.0 * a
    if abs(denominator) <= _INTERSECTION_EPSILON:
        return None
    near = (-b - sqrt_discriminant) / denominator
    far = (-b + sqrt_discriminant) / denominator
    return _QuadraticRoots(near=min(near, far), far=max(near, far))


def _valid_radius(radius_m: float) -> bool:
    return math.isfinite(radius_m) and radius_m > 0.0


def _invalid_hit(
    reason: SceneInvalidReason,
    source_reason: str,
) -> SphereHitResult:
    return SphereHitResult(
        valid=False,
        source_reason_invalid=source_reason,
        reason_invalid=reason,
    )


def _hemisphere(z: float) -> Literal["front", "rear", "equator"]:
    if z < -_EQUATOR_EPSILON:
        return "front"
    if z > _EQUATOR_EPSILON:
        return "rear"
    return "equator"


def _scene_vector(x: float, y: float, z: float) -> Vector3D:
    return Vector3D(space=CoordinateFrame3D.SCENE_PSEUDO_M, x=x, y=y, z=z)


def _tuple(vector: Vector3D | UnitVector3D) -> tuple[float, float, float]:
    return (vector.x, vector.y, vector.z)


def _finite_tuple(values: tuple[float, float, float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _normalize(
    values: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    if not _finite_tuple(values):
        return None
    norm = math.sqrt(_dot(values, values))
    if norm <= _INTERSECTION_EPSILON:
        return None
    return (values[0] / norm, values[1] / norm, values[2] / norm)


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return (left[0] * right[0]) + (left[1] * right[1]) + (left[2] * right[2])
