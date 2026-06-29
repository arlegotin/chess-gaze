from __future__ import annotations

import math
from typing import Literal

from pydantic import ConfigDict, field_validator

from chess_gaze.geometry import StrictSchemaModel

DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M = 0.063
DEFAULT_GAZE_SPHERE_RADIUS_M = 0.700
DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M = 0.090
DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M = 0.120
DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M = 0.100
DEFAULT_EYE_SPHERE_RADIUS_M = 0.012
DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M = (0.0, 0.035, 0.020)
DEFAULT_SCENE_CENTER_CAMERA_M = (0.0, 0.0, 0.650)
SCENE_CENTER_MIN_AXIS_TOLERANCE_M = 0.015
MIN_SCENE_CENTER_INLIER_FRAMES = 5
MIN_MAIN_DIRECTION_INLIER_FRAMES = 5
DIRECTION_INLIER_ANGLE_RADIANS = 0.35


def _validate_finite_triplet(
    value: tuple[float, float, float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    for coordinate in value:
        if not math.isfinite(coordinate):
            raise ValueError(f"{field_name} must contain only finite values")
    return value


class SceneAssumptionRecord(StrictSchemaModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str
    value: float | int | tuple[float, float, float]
    unit: str
    source: str
    uncertainty: Literal["low", "medium", "high"]

    @field_validator("value")
    @classmethod
    def validate_value(
        cls,
        value: float | int | tuple[float, float, float],
    ) -> float | int | tuple[float, float, float]:
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("value must be finite")
            return value
        if isinstance(value, tuple):
            return _validate_finite_triplet(value, field_name="value")
        return value


class SceneAssumptions(StrictSchemaModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    adult_male_interpupillary_distance_m: float
    gaze_sphere_radius_m: float
    head_ellipsoid_radius_m: tuple[float, float, float]
    eye_sphere_radius_m: float
    head_center_from_eye_midpoint_m: tuple[float, float, float]
    default_scene_center_camera_m: tuple[float, float, float]
    scene_center_min_axis_tolerance_m: float
    min_scene_center_inlier_frames: int
    min_main_direction_inlier_frames: int
    direction_inlier_angle_radians: float
    records: list[SceneAssumptionRecord]

    @field_validator(
        "head_ellipsoid_radius_m",
        "head_center_from_eye_midpoint_m",
        "default_scene_center_camera_m",
    )
    @classmethod
    def validate_triplets(
        cls,
        value: tuple[float, float, float],
        info: object,
    ) -> tuple[float, float, float]:
        field_name = getattr(info, "field_name", "triplet")
        return _validate_finite_triplet(value, field_name=field_name)


def default_scene_assumptions() -> SceneAssumptions:
    records = [
        SceneAssumptionRecord(
            name="DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M",
            value=DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M,
            unit="m",
            source="adult_male_default",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_HEAD_ELLIPSOID_RADIUS_M",
            value=(
                DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M,
                DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M,
                DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M,
            ),
            unit="m",
            source="adult_male_default",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_GAZE_SPHERE_RADIUS_M",
            value=DEFAULT_GAZE_SPHERE_RADIUS_M,
            unit="m",
            source="hypothetical_gaze_sphere_default",
            uncertainty="high",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_EYE_SPHERE_RADIUS_M",
            value=DEFAULT_EYE_SPHERE_RADIUS_M,
            unit="m",
            source="adult_male_default",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M",
            value=DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M,
            unit="m_in_head_local_axes",
            source="adult_male_default",
            uncertainty="high",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_SCENE_CENTER_CAMERA_M",
            value=DEFAULT_SCENE_CENTER_CAMERA_M,
            unit="camera_opencv_pseudo_m",
            source="fallback_default",
            uncertainty="high",
        ),
        SceneAssumptionRecord(
            name="SCENE_CENTER_MIN_AXIS_TOLERANCE_M",
            value=SCENE_CENTER_MIN_AXIS_TOLERANCE_M,
            unit="camera_opencv_pseudo_m",
            source="algorithm_constant",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="MIN_SCENE_CENTER_INLIER_FRAMES",
            value=MIN_SCENE_CENTER_INLIER_FRAMES,
            unit="frames",
            source="algorithm_constant",
            uncertainty="low",
        ),
        SceneAssumptionRecord(
            name="MIN_MAIN_DIRECTION_INLIER_FRAMES",
            value=MIN_MAIN_DIRECTION_INLIER_FRAMES,
            unit="frames",
            source="algorithm_constant",
            uncertainty="low",
        ),
        SceneAssumptionRecord(
            name="DIRECTION_INLIER_ANGLE_RADIANS",
            value=DIRECTION_INLIER_ANGLE_RADIANS,
            unit="radians",
            source="algorithm_constant",
            uncertainty="medium",
        ),
    ]
    return SceneAssumptions(
        adult_male_interpupillary_distance_m=(
            DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M
        ),
        gaze_sphere_radius_m=DEFAULT_GAZE_SPHERE_RADIUS_M,
        head_ellipsoid_radius_m=(
            DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M,
            DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M,
            DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M,
        ),
        eye_sphere_radius_m=DEFAULT_EYE_SPHERE_RADIUS_M,
        head_center_from_eye_midpoint_m=DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M,
        default_scene_center_camera_m=DEFAULT_SCENE_CENTER_CAMERA_M,
        scene_center_min_axis_tolerance_m=SCENE_CENTER_MIN_AXIS_TOLERANCE_M,
        min_scene_center_inlier_frames=MIN_SCENE_CENTER_INLIER_FRAMES,
        min_main_direction_inlier_frames=MIN_MAIN_DIRECTION_INLIER_FRAMES,
        direction_inlier_angle_radians=DIRECTION_INLIER_ANGLE_RADIANS,
        records=records,
    )
