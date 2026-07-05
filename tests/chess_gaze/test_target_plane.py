from __future__ import annotations

import pytest

from chess_gaze.scene_records import CoordinateFrame3D, SceneInvalidReason, Vector3D
from chess_gaze.target_plane import (
    ConfiguredTargetPlane,
    build_configured_target_plane,
    intersect_ray_with_target_plane,
)


def _camera_point(x: float, y: float, z: float) -> Vector3D:
    return Vector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=x,
        y=y,
        z=z,
    )


def test_intersect_ray_with_target_plane_maps_hit_to_normalized_coordinates() -> None:
    plane = build_configured_target_plane(
        origin_camera_m=(-1.0, -0.5, 2.0),
        x_axis_camera=(1.0, 0.0, 0.0),
        y_axis_camera=(0.0, 1.0, 0.0),
        width_m=2.0,
        height_m=1.0,
        mirror_horizontal=False,
    )

    hit = intersect_ray_with_target_plane(
        origin_camera_m=_camera_point(0.0, 0.0, 0.0),
        direction_camera=(0.0, 0.0, 1.0),
        plane=plane,
    )

    assert hit.valid is True
    assert hit.reason_invalid is None
    assert hit.point_camera_m == _camera_point(0.0, 0.0, 2.0)
    assert hit.ray_t_m == pytest.approx(2.0)
    assert hit.target_x_normalized == pytest.approx(0.5)
    assert hit.target_y_normalized == pytest.approx(0.5)
    assert hit.inside_bounds is True


def test_intersect_ray_with_target_plane_applies_horizontal_mirror() -> None:
    plane = build_configured_target_plane(
        origin_camera_m=(-1.0, -0.5, 2.0),
        x_axis_camera=(1.0, 0.0, 0.0),
        y_axis_camera=(0.0, 1.0, 0.0),
        width_m=2.0,
        height_m=1.0,
        mirror_horizontal=True,
    )

    hit = intersect_ray_with_target_plane(
        origin_camera_m=_camera_point(-0.5, 0.0, 0.0),
        direction_camera=(0.0, 0.0, 1.0),
        plane=plane,
    )

    assert hit.valid is True
    assert hit.target_x_normalized == pytest.approx(0.75)
    assert hit.target_y_normalized == pytest.approx(0.5)


def test_intersect_ray_with_target_plane_rejects_parallel_ray() -> None:
    plane = build_configured_target_plane(
        origin_camera_m=(-1.0, -0.5, 2.0),
        x_axis_camera=(1.0, 0.0, 0.0),
        y_axis_camera=(0.0, 1.0, 0.0),
        width_m=2.0,
        height_m=1.0,
        mirror_horizontal=False,
    )

    hit = intersect_ray_with_target_plane(
        origin_camera_m=_camera_point(0.0, 0.0, 0.0),
        direction_camera=(1.0, 0.0, 0.0),
        plane=plane,
    )

    assert hit.valid is False
    assert hit.reason_invalid == SceneInvalidReason.RAY_TARGET_PLANE_PARALLEL


def test_build_configured_target_plane_rejects_degenerate_axes() -> None:
    with pytest.raises(ValueError, match="target plane axes"):
        build_configured_target_plane(
            origin_camera_m=(0.0, 0.0, 1.0),
            x_axis_camera=(1.0, 0.0, 0.0),
            y_axis_camera=(2.0, 0.0, 0.0),
            width_m=2.0,
            height_m=1.0,
            mirror_horizontal=False,
        )


def test_configured_target_plane_can_roundtrip_from_tuple_fields() -> None:
    plane = ConfiguredTargetPlane(
        origin_camera_m=(0.0, 0.0, 1.0),
        x_axis_camera=(1.0, 0.0, 0.0),
        y_axis_camera=(0.0, 1.0, 0.0),
        normal_camera=(0.0, 0.0, 1.0),
        width_m=2.0,
        height_m=1.0,
        mirror_horizontal=False,
    )

    assert plane.normal_camera == (0.0, 0.0, 1.0)
