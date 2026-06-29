from __future__ import annotations

import math

import pytest

from chess_gaze.scene_calibration import (
    DEFAULT_GAZE_SPHERE_RADIUS_M,
    default_scene_assumptions,
)
from chess_gaze.scene_records import (
    CoordinateFrame3D,
    SceneInvalidReason,
    UnitVector3D,
    Vector3D,
)
from chess_gaze.sphere_projection import build_gaze_sphere, intersect_ray_with_sphere


def _scene_point(x: float, y: float, z: float) -> Vector3D:
    return Vector3D(space=CoordinateFrame3D.SCENE_PSEUDO_M, x=x, y=y, z=z)


def _scene_unit(x: float, y: float, z: float) -> UnitVector3D:
    norm = math.sqrt((x * x) + (y * y) + (z * z))
    return UnitVector3D(
        space=CoordinateFrame3D.SCENE_PSEUDO_M,
        x=x / norm,
        y=y / norm,
        z=z / norm,
    )


def test_build_gaze_sphere_uses_scene_origin_and_default_radius() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    assert sphere.center_scene_m == _scene_point(0.0, 0.0, 0.0)
    assert sphere.radius_m == pytest.approx(DEFAULT_GAZE_SPHERE_RADIUS_M)
    assert sphere.radius_source == "DEFAULT_GAZE_SPHERE_RADIUS_M"
    assert sphere.center_source == "robust_scene_center"


def test_origin_inside_sphere_hits_front_surface() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, 0.0),
        direction_scene=_scene_unit(0.0, 0.0, -1.0),
        sphere=sphere,
    )

    assert hit.valid is True
    assert hit.reason_invalid is None
    assert hit.point_scene_m == _scene_point(0.0, 0.0, -sphere.radius_m)
    assert hit.ray_t_m == pytest.approx(sphere.radius_m)
    assert hit.radius_m == pytest.approx(sphere.radius_m)
    assert hit.theta_radians == pytest.approx(0.0)
    assert hit.phi_radians == pytest.approx(0.0)
    assert hit.hemisphere == "front"


def test_rear_hemisphere_hit_is_valid_direction_evidence() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, 0.0),
        direction_scene=_scene_unit(0.0, 0.0, 1.0),
        sphere=sphere,
    )

    assert hit.valid is True
    assert hit.point_scene_m == _scene_point(0.0, 0.0, sphere.radius_m)
    assert hit.hemisphere == "rear"
    assert hit.theta_radians == pytest.approx(math.pi)


def test_origin_outside_sphere_selects_nearest_forward_root() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, -1.0),
        direction_scene=_scene_unit(0.0, 0.0, 1.0),
        sphere=sphere,
    )

    assert hit.valid is True
    assert hit.point_scene_m == _scene_point(0.0, 0.0, -sphere.radius_m)
    assert hit.ray_t_m == pytest.approx(1.0 - sphere.radius_m)


def test_tangent_ray_has_one_forward_intersection() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(-1.0, sphere.radius_m, 0.0),
        direction_scene=_scene_unit(1.0, 0.0, 0.0),
        sphere=sphere,
    )

    assert hit.valid is True
    assert hit.point_scene_m == _scene_point(0.0, sphere.radius_m, 0.0)
    assert hit.hemisphere == "equator"
    assert hit.phi_radians == pytest.approx(math.pi / 2.0)


def test_sphere_miss_and_behind_origin_are_invalid_with_reasons() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    miss = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(-1.0, sphere.radius_m + 0.1, 0.0),
        direction_scene=_scene_unit(1.0, 0.0, 0.0),
        sphere=sphere,
    )
    behind = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, -1.0),
        direction_scene=_scene_unit(0.0, 0.0, -1.0),
        sphere=sphere,
    )

    assert miss.valid is False
    assert miss.reason_invalid == SceneInvalidReason.RAY_SPHERE_DISCRIMINANT_NEGATIVE
    assert behind.valid is False
    assert (
        behind.reason_invalid
        == SceneInvalidReason.RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN
    )


def test_invalid_radius_and_missing_ray_are_invalid() -> None:
    invalid_sphere = build_gaze_sphere(default_scene_assumptions()).model_copy(
        update={"radius_m": 0.0}
    )

    radius_hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, 0.0),
        direction_scene=_scene_unit(0.0, 0.0, -1.0),
        sphere=invalid_sphere,
    )
    missing_ray_hit = intersect_ray_with_sphere(
        origin_scene_m=None,
        direction_scene=None,
        sphere=build_gaze_sphere(default_scene_assumptions()),
        source_reason_invalid="appearance gaze unavailable",
    )

    assert radius_hit.valid is False
    assert radius_hit.reason_invalid == SceneInvalidReason.SPHERE_RADIUS_INVALID
    assert missing_ray_hit.valid is False
    assert missing_ray_hit.reason_invalid == SceneInvalidReason.UNIGAZE_INVALID
    assert missing_ray_hit.source_reason_invalid == "appearance gaze unavailable"
