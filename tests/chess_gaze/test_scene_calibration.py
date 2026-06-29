from __future__ import annotations

import pytest
from pydantic import ValidationError

from chess_gaze.scene_calibration import (
    DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M,
    DEFAULT_EYE_SPHERE_RADIUS_M,
    DEFAULT_GAZE_SPHERE_RADIUS_M,
    DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M,
    DEFAULT_SCENE_CENTER_CAMERA_M,
    DIRECTION_INLIER_ANGLE_RADIANS,
    MIN_MAIN_DIRECTION_INLIER_FRAMES,
    MIN_SCENE_CENTER_INLIER_FRAMES,
    SCENE_CENTER_MIN_AXIS_TOLERANCE_M,
    SceneAssumptions,
    default_scene_assumptions,
)


def test_scene_constant_values_are_exact() -> None:
    assert DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M == 0.063
    assert DEFAULT_GAZE_SPHERE_RADIUS_M == 0.700
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M == 0.090
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M == 0.120
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M == 0.100
    assert DEFAULT_EYE_SPHERE_RADIUS_M == 0.012
    assert DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M == (0.0, 0.035, 0.020)
    assert DEFAULT_SCENE_CENTER_CAMERA_M == (0.0, 0.0, 0.650)
    assert SCENE_CENTER_MIN_AXIS_TOLERANCE_M == 0.015
    assert MIN_SCENE_CENTER_INLIER_FRAMES == 5
    assert MIN_MAIN_DIRECTION_INLIER_FRAMES == 5
    assert DIRECTION_INLIER_ANGLE_RADIANS == 0.35


def test_default_scene_assumptions_is_strict_and_frozen() -> None:
    assumptions = default_scene_assumptions()

    assert SceneAssumptions.model_config["strict"] is True
    assert SceneAssumptions.model_config["frozen"] is True

    with pytest.raises(ValidationError):
        SceneAssumptions.model_validate(
            {
                **assumptions.model_dump(),
                "gaze_sphere_radius_m": "0.7",
            }
        )

    with pytest.raises(ValidationError):
        assumptions.gaze_sphere_radius_m = 0.5


def test_default_scene_assumptions_persists_metadata_for_every_record() -> None:
    assumptions = default_scene_assumptions()
    names = {record.name for record in assumptions.records}

    assert assumptions.records
    assert assumptions.model_dump()["adult_male_interpupillary_distance_m"] == 0.063
    assert "DEFAULT_GAZE_SPHERE_RADIUS_M" in names
    assert "DEFAULT_MONITOR_DISTANCE_FROM_EYES_M" not in names
    assert "DEFAULT_MONITOR_WIDTH_M" not in names
    assert "DEFAULT_MONITOR_HEIGHT_M" not in names
    assert "DEFAULT_EXTENDED_PLANE_SCALE" not in names
    assert "RAY_PLANE_PARALLEL_EPSILON" not in names

    for record in assumptions.records:
        assert set(record.model_dump()) == {
            "name",
            "value",
            "unit",
            "source",
            "uncertainty",
        }


def test_default_scene_assumptions_persists_head_radius_tuple_constant() -> None:
    assumptions = default_scene_assumptions()
    names = {record.name: record.value for record in assumptions.records}

    assert names["DEFAULT_HEAD_ELLIPSOID_RADIUS_M"] == (0.090, 0.120, 0.100)
    assert "DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M" not in names
    assert "DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M" not in names
    assert "DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M" not in names
