from __future__ import annotations

import pytest
from pydantic import ValidationError

from chess_gaze.scene_calibration import (
    DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M,
    DEFAULT_EXTENDED_PLANE_SCALE,
    DEFAULT_EYE_SPHERE_RADIUS_M,
    DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M,
    DEFAULT_MONITOR_DISTANCE_FROM_EYES_M,
    DEFAULT_MONITOR_HEIGHT_M,
    DEFAULT_MONITOR_WIDTH_M,
    DEFAULT_SCENE_CENTER_CAMERA_M,
    DIRECTION_INLIER_ANGLE_RADIANS,
    MIN_MAIN_DIRECTION_INLIER_FRAMES,
    MIN_SCENE_CENTER_INLIER_FRAMES,
    RAY_PLANE_PARALLEL_EPSILON,
    SCENE_CENTER_MIN_AXIS_TOLERANCE_M,
    SceneAssumptions,
    default_scene_assumptions,
)


def test_scene_constant_values_are_exact() -> None:
    assert DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M == 0.063
    assert DEFAULT_MONITOR_DISTANCE_FROM_EYES_M == 0.700
    assert DEFAULT_MONITOR_WIDTH_M == 0.600
    assert DEFAULT_MONITOR_HEIGHT_M == 0.340
    assert DEFAULT_EXTENDED_PLANE_SCALE == 3.0
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M == 0.090
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M == 0.120
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M == 0.100
    assert DEFAULT_EYE_SPHERE_RADIUS_M == 0.012
    assert DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M == (0.0, 0.035, 0.020)
    assert RAY_PLANE_PARALLEL_EPSILON == 1e-6
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
                "monitor_width_m": "0.6",
            }
        )

    with pytest.raises(ValidationError):
        assumptions.monitor_width_m = 0.5


def test_default_scene_assumptions_persists_metadata_for_every_record() -> None:
    assumptions = default_scene_assumptions()

    assert assumptions.records
    assert assumptions.model_dump()["adult_male_interpupillary_distance_m"] == 0.063

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
