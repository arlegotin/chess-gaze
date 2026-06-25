from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import field_validator, model_validator

from chess_gaze.geometry import Point2D, StrictSchemaModel
from chess_gaze.scene_calibration import SceneAssumptionRecord, _validate_finite_triplet


def _coerce_enum_field(
    payload: dict[str, Any],
    *,
    field_name: str,
    enum_type: type[StrEnum],
) -> None:
    value = payload.get(field_name)
    if isinstance(value, str):
        try:
            payload[field_name] = enum_type(value)
        except ValueError:
            return


class CoordinateFrame3D(StrEnum):
    IMAGE_PX = "image_px"
    CAMERA_OPENCV_PSEUDO_M = "camera_opencv_pseudo_m"
    SCENE_PSEUDO_M = "scene_pseudo_m"
    MONITOR_PLANE_PSEUDO_M = "monitor_plane_pseudo_m"
    THREE_VIEW = "three_view"


class SceneInvalidReason(StrEnum):
    LEFT_EYE_INVALID = "LEFT_EYE_INVALID"
    RIGHT_EYE_INVALID = "RIGHT_EYE_INVALID"
    EYE_MIDPOINT_INVALID = "EYE_MIDPOINT_INVALID"
    UNIGAZE_INVALID = "UNIGAZE_INVALID"
    RAY_PARALLEL_TO_MONITOR = "RAY_PARALLEL_TO_MONITOR"
    RAY_COPLANAR_WITH_MONITOR = "RAY_COPLANAR_WITH_MONITOR"
    RAY_INTERSECTION_NON_FINITE = "RAY_INTERSECTION_NON_FINITE"
    RAY_INTERSECTION_BEHIND_ORIGIN = "RAY_INTERSECTION_BEHIND_ORIGIN"
    SCENE_CENTER_INSUFFICIENT_INLIERS = "SCENE_CENTER_INSUFFICIENT_INLIERS"
    MAIN_DIRECTION_INSUFFICIENT_INLIERS = "MAIN_DIRECTION_INSUFFICIENT_INLIERS"
    SCENE_AXIS_DEGENERATE = "SCENE_AXIS_DEGENERATE"
    MONITOR_PLANE_DEGENERATE = "MONITOR_PLANE_DEGENERATE"
    NON_FINITE_INPUT = "NON_FINITE_INPUT"


class Vector3D(StrictSchemaModel):
    space: CoordinateFrame3D
    x: float
    y: float
    z: float

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        _coerce_enum_field(
            coerced,
            field_name="space",
            enum_type=CoordinateFrame3D,
        )
        return coerced


class UnitVector3D(Vector3D):
    @model_validator(mode="after")
    def validate_unit_norm(self) -> UnitVector3D:
        norm = math.sqrt((self.x * self.x) + (self.y * self.y) + (self.z * self.z))
        if not 0.999 <= norm <= 1.001:
            raise ValueError("unit vector norm must be within [0.999, 1.001]")
        return self


class SceneCameraModel(StrictSchemaModel):
    frame_width_px: int
    frame_height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    model: Literal["estimated_pinhole_from_frame_size"]


class SceneEyeRecord(StrictSchemaModel):
    valid: bool
    image_px: Point2D | None
    camera_point_m: Vector3D | None
    scene_point_m: Vector3D | None
    source_reason_invalid: str | None
    reason_invalid: SceneInvalidReason | None

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        _coerce_enum_field(
            coerced,
            field_name="reason_invalid",
            enum_type=SceneInvalidReason,
        )
        return coerced

    @model_validator(mode="after")
    def validate_eye(self) -> SceneEyeRecord:
        if self.valid:
            if self.image_px is None or self.camera_point_m is None:
                raise ValueError("valid eye requires image_px and camera_point_m")
            if self.reason_invalid is not None:
                raise ValueError("valid eye cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid eye requires reason_invalid")
        return self


class SceneEyeMidpointRecord(StrictSchemaModel):
    valid: bool
    camera_point_m: Vector3D | None
    scene_point_m: Vector3D | None
    pupil_distance_px: float | None
    estimated_depth_m: float | None
    reason_invalid: SceneInvalidReason | None

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        _coerce_enum_field(
            coerced,
            field_name="reason_invalid",
            enum_type=SceneInvalidReason,
        )
        return coerced

    @model_validator(mode="after")
    def validate_midpoint(self) -> SceneEyeMidpointRecord:
        if self.valid:
            if self.camera_point_m is None:
                raise ValueError("valid eye midpoint requires camera_point_m")
            if self.reason_invalid is not None:
                raise ValueError("valid eye midpoint cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid eye midpoint requires reason_invalid")
        return self


class SceneHeadRecord(StrictSchemaModel):
    valid: bool
    ellipsoid_center_scene_m: Vector3D | None
    radii_m: tuple[float, float, float]
    reason_invalid: SceneInvalidReason | None

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        _coerce_enum_field(
            coerced,
            field_name="reason_invalid",
            enum_type=SceneInvalidReason,
        )
        return coerced

    @field_validator("radii_m")
    @classmethod
    def validate_radii(
        cls,
        value: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return _validate_finite_triplet(value, field_name="radii_m")

    @model_validator(mode="after")
    def validate_head(self) -> SceneHeadRecord:
        if self.valid:
            if self.ellipsoid_center_scene_m is None:
                raise ValueError("valid head requires ellipsoid_center_scene_m")
            if self.reason_invalid is not None:
                raise ValueError("valid head cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid head requires reason_invalid")
        return self


class SceneUniGazeRayRecord(StrictSchemaModel):
    valid: bool
    source: Literal["appearance_gaze"]
    origin_camera_m: Vector3D | None
    origin_scene_m: Vector3D | None
    direction_camera: UnitVector3D | None
    direction_scene: UnitVector3D | None
    pitch_radians: float | None
    yaw_radians: float | None
    reason_invalid: SceneInvalidReason | None

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        _coerce_enum_field(
            coerced,
            field_name="reason_invalid",
            enum_type=SceneInvalidReason,
        )
        return coerced

    @model_validator(mode="after")
    def validate_ray(self) -> SceneUniGazeRayRecord:
        if self.valid:
            required_values = (
                self.origin_camera_m,
                self.origin_scene_m,
                self.direction_camera,
                self.direction_scene,
                self.pitch_radians,
                self.yaw_radians,
            )
            if any(value is None for value in required_values):
                raise ValueError(
                    "valid unigaze ray requires origins, directions, and angles"
                )
            if self.reason_invalid is not None:
                raise ValueError("valid unigaze ray cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid unigaze ray requires reason_invalid")
        return self


class SceneMonitorHitRecord(StrictSchemaModel):
    valid: bool
    point_camera_m: Vector3D | None
    point_scene_m: Vector3D | None
    u_m: float | None
    v_m: float | None
    t: float | None
    denominator: float | None
    signed_distance_m: float | None
    within_physical_monitor: bool | None
    within_extended_plane: bool | None
    reason_invalid: SceneInvalidReason | None

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        _coerce_enum_field(
            coerced,
            field_name="reason_invalid",
            enum_type=SceneInvalidReason,
        )
        return coerced

    @model_validator(mode="after")
    def validate_hit(self) -> SceneMonitorHitRecord:
        if self.valid:
            required_values = (
                self.point_camera_m,
                self.point_scene_m,
                self.u_m,
                self.v_m,
                self.t,
                self.denominator,
                self.signed_distance_m,
                self.within_physical_monitor,
                self.within_extended_plane,
            )
            if any(value is None for value in required_values):
                raise ValueError(
                    "valid monitor hit requires point, uv, t, denominator, "
                    "distance, and bounds flags"
                )
            if self.t < 0:
                raise ValueError("valid monitor hit requires t >= 0")
            if self.reason_invalid is not None:
                raise ValueError("valid monitor hit cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid monitor hit requires reason_invalid")
        return self


class SceneAxisBasisRecord(StrictSchemaModel):
    right_camera: UnitVector3D
    up_camera: UnitVector3D
    back_camera: UnitVector3D
    forward_camera: UnitVector3D
    determinant_right_up_back: float
    convention: Literal["right_up_back_columns_right_handed"]
    fallbacks: list[str]


class SceneMonitorPlaneRecord(StrictSchemaModel):
    center_camera_m: Vector3D
    center_scene_m: Vector3D
    normal_camera: UnitVector3D
    right_camera: UnitVector3D
    up_camera: UnitVector3D
    width_m: float
    height_m: float
    extended_width_m: float
    extended_height_m: float
    distance_from_scene_center_m: float


class SceneFrameRecord(StrictSchemaModel):
    schema_version: Literal["gaze-scene-frame-v1"] = "gaze-scene-frame-v1"
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    left_eye: SceneEyeRecord
    right_eye: SceneEyeRecord
    eye_midpoint: SceneEyeMidpointRecord
    head: SceneHeadRecord
    unigaze_ray: SceneUniGazeRayRecord
    main_monitor_hit: SceneMonitorHitRecord
    diagnostics: dict[str, str | int | float | bool | None]

    @model_validator(mode="after")
    def validate_frame_dependencies(self) -> SceneFrameRecord:
        if self.eye_midpoint.valid and (
            not self.left_eye.valid or not self.right_eye.valid
        ):
            raise ValueError("valid eye midpoint requires both eyes to be valid")
        if self.main_monitor_hit.valid and not self.unigaze_ray.valid:
            raise ValueError("valid monitor hit requires a valid unigaze ray")
        return self


class SceneManifest(StrictSchemaModel):
    schema_version: Literal["gaze-scene-manifest-v1"] = "gaze-scene-manifest-v1"
    run_id: str
    source_video_path: str
    source_video_sha256: str
    camera_model: SceneCameraModel
    assumptions: list[SceneAssumptionRecord]
    scene_center_camera_m: Vector3D
    axis_basis: SceneAxisBasisRecord
    monitor_plane: SceneMonitorPlaneRecord
    robust_estimators: dict[str, object]
    viewer_dependency: dict[str, object]


class SceneSummary(StrictSchemaModel):
    schema_version: Literal["gaze-scene-summary-v1"] = "gaze-scene-summary-v1"
    run_id: str
    decoded_frames: int
    scene_frame_records: int
    valid_eye_midpoint_frames: int
    valid_unigaze_ray_frames: int
    valid_monitor_hit_frames: int
    invalid_reason_counts: dict[str, int]
    representative_invalid_frame_ids: list[str]
    count_validation_passed: bool


class ViewerHitPoint(StrictSchemaModel):
    frame_id: str
    frame_index: int
    point_scene_m: Vector3D
    u_m: float
    v_m: float
    within_physical_monitor: bool
    within_extended_plane: bool


class ViewerSceneData(StrictSchemaModel):
    schema_version: Literal["gaze-scene-viewer-data-v1"] = (
        "gaze-scene-viewer-data-v1"
    )
    run_id: str
    source_video_stem: str
    frame_count: int
    frames: list[SceneFrameRecord]
    valid_hit_points: list[ViewerHitPoint]
    monitor_plane: SceneMonitorPlaneRecord
    axis_basis: SceneAxisBasisRecord
    assumptions: list[SceneAssumptionRecord]
    summary: SceneSummary
