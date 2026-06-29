from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from chess_gaze.errors import FrameStatus
from chess_gaze.geometry import Point2D, StrictSchemaModel
from chess_gaze.scene_calibration import (
    SceneAssumptionRecord as SceneAssumptionRecord,
)
from chess_gaze.scene_calibration import (
    _validate_finite_triplet,
)


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


def _require_vector_space(
    value: Vector3D | UnitVector3D | None,
    *,
    expected: CoordinateFrame3D,
    field_name: str,
) -> None:
    if value is not None and value.space != expected:
        raise ValueError(f"{field_name} must use {expected.value}")


class SceneSchemaModel(StrictSchemaModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=True,
        use_enum_values=True,
    )


class CoordinateFrame3D(StrEnum):
    IMAGE_PX = "image_px"
    CAMERA_OPENCV_PSEUDO_M = "camera_opencv_pseudo_m"
    SCENE_PSEUDO_M = "scene_pseudo_m"
    GAZE_SPHERE_PSEUDO_M = "gaze_sphere_pseudo_m"
    THREE_VIEW = "three_view"


class SceneInvalidReason(StrEnum):
    LEFT_EYE_INVALID = "LEFT_EYE_INVALID"
    RIGHT_EYE_INVALID = "RIGHT_EYE_INVALID"
    EYE_MIDPOINT_INVALID = "EYE_MIDPOINT_INVALID"
    UNIGAZE_INVALID = "UNIGAZE_INVALID"
    SPHERE_RADIUS_INVALID = "SPHERE_RADIUS_INVALID"
    RAY_SPHERE_DISCRIMINANT_NEGATIVE = "RAY_SPHERE_DISCRIMINANT_NEGATIVE"
    RAY_SPHERE_INTERSECTION_NON_FINITE = "RAY_SPHERE_INTERSECTION_NON_FINITE"
    RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN = (
        "RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN"
    )
    RAY_PARALLEL_TO_MONITOR = "RAY_PARALLEL_TO_MONITOR"
    RAY_COPLANAR_WITH_MONITOR = "RAY_COPLANAR_WITH_MONITOR"
    RAY_INTERSECTION_NON_FINITE = "RAY_INTERSECTION_NON_FINITE"
    RAY_INTERSECTION_BEHIND_ORIGIN = "RAY_INTERSECTION_BEHIND_ORIGIN"
    SCENE_CENTER_INSUFFICIENT_INLIERS = "SCENE_CENTER_INSUFFICIENT_INLIERS"
    MAIN_DIRECTION_INSUFFICIENT_INLIERS = "MAIN_DIRECTION_INSUFFICIENT_INLIERS"
    SCENE_AXIS_DEGENERATE = "SCENE_AXIS_DEGENERATE"
    MONITOR_PLANE_DEGENERATE = "MONITOR_PLANE_DEGENERATE"
    NON_FINITE_INPUT = "NON_FINITE_INPUT"


class Vector3D(SceneSchemaModel):
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


class SceneCameraModel(SceneSchemaModel):
    policy: Literal["estimated_pinhole_from_image_size"]
    frame_width_px: int
    frame_height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    metric_translation_allowed: bool
    uncertainty: Literal["low", "medium", "high"]


class SceneFrameCameraRecord(SceneSchemaModel):
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    depth_source: Literal["interpupillary_distance_assumption"]


class SceneFrameDiagnosticsRecord(SceneSchemaModel):
    warnings: list[str]
    source_error_codes: list[str]


class SceneEyeRecord(SceneSchemaModel):
    valid: bool
    image_px: Point2D | None = None
    camera_point_m: Vector3D | None = Field(default=None, alias="camera_m")
    scene_point_m: Vector3D | None = Field(default=None, alias="scene_m")
    source_reason_invalid: str | None = None
    reason_invalid: SceneInvalidReason | None = None

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
        _require_vector_space(
            self.camera_point_m,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="camera_point_m",
        )
        _require_vector_space(
            self.scene_point_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="scene_point_m",
        )
        if self.valid:
            if self.image_px is None or self.camera_point_m is None:
                raise ValueError("valid eye requires image_px and camera_point_m")
            if self.reason_invalid is not None:
                raise ValueError("valid eye cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid eye requires reason_invalid")
        return self


class SceneEyeMidpointRecord(SceneSchemaModel):
    valid: bool
    origin_policy: Literal["both_eyes_required"] | None = None
    camera_point_m: Vector3D | None = Field(default=None, alias="camera_m")
    scene_point_m: Vector3D | None = Field(default=None, alias="scene_m")
    pupil_distance_px: float | None = None
    estimated_depth_m: float | None = None
    source_reason_invalid: str | None = None
    reason_invalid: SceneInvalidReason | None = None

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
        _require_vector_space(
            self.camera_point_m,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="camera_point_m",
        )
        _require_vector_space(
            self.scene_point_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="scene_point_m",
        )
        if self.valid:
            if (
                self.camera_point_m is None
                or self.scene_point_m is None
                or self.origin_policy is None
            ):
                raise ValueError(
                    "valid eye midpoint requires origin_policy, camera_m, and scene_m"
                )
            if self.reason_invalid is not None:
                raise ValueError("valid eye midpoint cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid eye midpoint requires reason_invalid")
        return self


class SceneHeadRecord(SceneSchemaModel):
    valid: bool
    ellipsoid_center_camera_m: Vector3D | None = None
    ellipsoid_center_scene_m: Vector3D | None = Field(default=None, alias="scene_m")
    radii_m: tuple[float, float, float] = Field(alias="ellipsoid_radii_m")
    yaw_radians: float | None = None
    pitch_radians: float | None = None
    roll_radians: float | None = None
    orientation_source: str | None = None
    source_reason_invalid: str | None = None
    reason_invalid: SceneInvalidReason | None = None

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

    @field_validator("radii_m", mode="before")
    @classmethod
    def coerce_radii_sequence(
        cls,
        value: Any,
    ) -> tuple[float, float, float] | Any:
        if isinstance(value, list):
            if len(value) != 3:
                raise ValueError("radii_m must contain exactly 3 values")
            return (value[0], value[1], value[2])
        return value

    @field_validator("radii_m")
    @classmethod
    def validate_radii(
        cls,
        value: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError("radii_m must contain exactly 3 values")
        return _validate_finite_triplet(value, field_name="radii_m")

    @model_validator(mode="after")
    def validate_head(self) -> SceneHeadRecord:
        _require_vector_space(
            self.ellipsoid_center_camera_m,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="ellipsoid_center_camera_m",
        )
        _require_vector_space(
            self.ellipsoid_center_scene_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="ellipsoid_center_scene_m",
        )
        if self.valid:
            if (
                self.ellipsoid_center_scene_m is None
                and self.ellipsoid_center_camera_m is None
            ):
                raise ValueError("valid head requires a persisted ellipsoid center")
            if self.reason_invalid is not None:
                raise ValueError("valid head cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid head requires reason_invalid")
        return self


class SceneUniGazeRayRecord(SceneSchemaModel):
    valid: bool
    source: Literal["appearance_gaze"]
    origin_camera_m: Vector3D | None = None
    origin_scene_m: Vector3D | None = Field(default=None, alias="scene_m")
    direction_camera: UnitVector3D | None = None
    direction_scene: UnitVector3D | None = None
    direction_source: str | None = None
    pitch_radians: float | None = None
    yaw_radians: float | None = None
    source_reason_invalid: str | None = None
    reason_invalid: SceneInvalidReason | None = None

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
        _require_vector_space(
            self.origin_camera_m,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="origin_camera_m",
        )
        _require_vector_space(
            self.origin_scene_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="origin_scene_m",
        )
        _require_vector_space(
            self.direction_camera,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="direction_camera",
        )
        _require_vector_space(
            self.direction_scene,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="direction_scene",
        )
        if self.valid:
            required_values = (
                self.origin_camera_m,
                self.origin_scene_m,
                self.direction_camera,
                self.direction_scene,
                self.direction_source,
                self.pitch_radians,
                self.yaw_radians,
            )
            if any(value is None for value in required_values):
                raise ValueError(
                    "valid unigaze ray requires origins, directions, source, and angles"
                )
            if self.reason_invalid is not None:
                raise ValueError("valid unigaze ray cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid unigaze ray requires reason_invalid")
        return self


class SceneSphereHitRecord(SceneSchemaModel):
    valid: bool
    point_scene_m: Vector3D | None = None
    ray_t_m: float | None = None
    radius_m: float | None = None
    theta_radians: float | None = None
    phi_radians: float | None = None
    hemisphere: Literal["front", "rear", "equator"] | None = None
    source_reason_invalid: str | None = None
    reason_invalid: SceneInvalidReason | None = None

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
    def validate_hit(self) -> SceneSphereHitRecord:
        _require_vector_space(
            self.point_scene_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="point_scene_m",
        )
        if self.valid:
            required_values = (
                self.point_scene_m,
                self.ray_t_m,
                self.radius_m,
                self.theta_radians,
                self.phi_radians,
                self.hemisphere,
            )
            if any(value is None for value in required_values):
                raise ValueError(
                    "valid sphere hit requires point, t, radius, angles, and hemisphere"
                )
            if self.ray_t_m is not None and self.ray_t_m < 0:
                raise ValueError("valid sphere hit requires ray_t_m >= 0")
            if self.radius_m is not None and self.radius_m <= 0:
                raise ValueError("valid sphere hit requires radius_m > 0")
            if self.reason_invalid is not None:
                raise ValueError("valid sphere hit cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid sphere hit requires reason_invalid")
        return self


class SceneGazeSphereRecord(SceneSchemaModel):
    center_scene_m: Vector3D
    radius_m: float
    radius_source: Literal["DEFAULT_GAZE_SPHERE_RADIUS_M"]
    center_source: Literal["robust_scene_center"]

    @model_validator(mode="after")
    def validate_sphere(self) -> SceneGazeSphereRecord:
        _require_vector_space(
            self.center_scene_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="center_scene_m",
        )
        if self.radius_m <= 0:
            raise ValueError("gaze sphere radius_m must be > 0")
        return self


class SceneAxisBasisRecord(SceneSchemaModel):
    right_camera: UnitVector3D
    up_camera: UnitVector3D
    back_camera: UnitVector3D
    forward_camera: UnitVector3D
    determinant_right_up_back: float
    convention: Literal["right_up_back_columns_right_handed"]
    fallbacks: list[str]

    @model_validator(mode="after")
    def validate_axis_basis(self) -> SceneAxisBasisRecord:
        _require_vector_space(
            self.right_camera,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="right_camera",
        )
        _require_vector_space(
            self.up_camera,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="up_camera",
        )
        _require_vector_space(
            self.back_camera,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="back_camera",
        )
        _require_vector_space(
            self.forward_camera,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="forward_camera",
        )
        computed_determinant = (
            self.right_camera.x
            * (
                (self.up_camera.y * self.back_camera.z)
                - (self.up_camera.z * self.back_camera.y)
            )
            - self.right_camera.y
            * (
                (self.up_camera.x * self.back_camera.z)
                - (self.up_camera.z * self.back_camera.x)
            )
            + self.right_camera.z
            * (
                (self.up_camera.x * self.back_camera.y)
                - (self.up_camera.y * self.back_camera.x)
            )
        )
        dot_product = (
            (self.back_camera.x * self.forward_camera.x)
            + (self.back_camera.y * self.forward_camera.y)
            + (self.back_camera.z * self.forward_camera.z)
        )
        if abs(dot_product + 1.0) > 0.001:
            raise ValueError("back_camera must be anti-parallel to forward_camera")
        if not 0.99 <= computed_determinant <= 1.01:
            raise ValueError("computed determinant must be near +1")
        if abs(self.determinant_right_up_back - computed_determinant) > 0.001:
            raise ValueError(
                "determinant_right_up_back must match the computed determinant"
            )
        return self


class SceneMonitorPlaneRecord(SceneSchemaModel):
    center_camera_m: Vector3D
    center_scene_m: Vector3D
    normal_camera: UnitVector3D
    right_camera: UnitVector3D
    up_camera: UnitVector3D
    width_m: float = Field(alias="physical_width_m")
    height_m: float = Field(alias="physical_height_m")
    extended_width_m: float
    extended_height_m: float
    distance_from_scene_center_m: float
    distance_source: str | None = None

    @model_validator(mode="after")
    def validate_spaces(self) -> SceneMonitorPlaneRecord:
        _require_vector_space(
            self.center_camera_m,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="center_camera_m",
        )
        _require_vector_space(
            self.center_scene_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="center_scene_m",
        )
        _require_vector_space(
            self.normal_camera,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="normal_camera",
        )
        _require_vector_space(
            self.right_camera,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="right_camera",
        )
        _require_vector_space(
            self.up_camera,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="up_camera",
        )
        return self


class SceneFrameRecord(SceneSchemaModel):
    schema_version: Literal["gaze-scene-frame-v2"] = "gaze-scene-frame-v2"
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    source_frame_status: FrameStatus
    valid_for_scene_center: bool
    valid_for_sphere_projection: bool
    camera: SceneFrameCameraRecord
    left_eye: SceneEyeRecord
    right_eye: SceneEyeRecord
    eye_midpoint: SceneEyeMidpointRecord
    head: SceneHeadRecord
    unigaze_ray: SceneUniGazeRayRecord
    sphere_hit: SceneSphereHitRecord
    diagnostics: SceneFrameDiagnosticsRecord

    @model_validator(mode="before")
    @classmethod
    def coerce_frame_status(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        _coerce_enum_field(
            coerced,
            field_name="source_frame_status",
            enum_type=FrameStatus,
        )
        return coerced

    @model_validator(mode="after")
    def validate_frame_dependencies(self) -> SceneFrameRecord:
        if self.eye_midpoint.valid and (
            not self.left_eye.valid or not self.right_eye.valid
        ):
            raise ValueError("valid eye midpoint requires both eyes to be valid")
        if self.valid_for_scene_center and not self.eye_midpoint.valid:
            raise ValueError(
                "valid_for_scene_center requires a valid eye_midpoint record"
            )
        if self.valid_for_sphere_projection and not self.unigaze_ray.valid:
            raise ValueError(
                "valid_for_sphere_projection requires a valid unigaze_ray record"
            )
        if self.sphere_hit.valid and not self.unigaze_ray.valid:
            raise ValueError("valid sphere hit requires a valid unigaze ray")
        return self


class SceneSourceArtifactsRecord(SceneSchemaModel):
    frame_records: str
    scene_frame_records: str
    scene_summary: str
    viewer: str


class SceneCoordinateFramesRecord(SceneSchemaModel):
    math_frame: CoordinateFrame3D
    scene_frame: CoordinateFrame3D
    projection_frame: CoordinateFrame3D
    viewer_frame: CoordinateFrame3D

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        for field_name in (
            "math_frame",
            "scene_frame",
            "projection_frame",
            "viewer_frame",
        ):
            _coerce_enum_field(
                coerced,
                field_name=field_name,
                enum_type=CoordinateFrame3D,
            )
        return coerced

    @model_validator(mode="after")
    def validate_semantic_mapping(self) -> SceneCoordinateFramesRecord:
        if self.math_frame != CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M:
            raise ValueError("math_frame must be camera_opencv_pseudo_m")
        if self.scene_frame != CoordinateFrame3D.SCENE_PSEUDO_M:
            raise ValueError("scene_frame must be scene_pseudo_m")
        if self.projection_frame != CoordinateFrame3D.GAZE_SPHERE_PSEUDO_M:
            raise ValueError("projection_frame must be gaze_sphere_pseudo_m")
        if self.viewer_frame != CoordinateFrame3D.THREE_VIEW:
            raise ValueError("viewer_frame must be three_view")
        return self


class SceneCenterEstimatorRecord(SceneSchemaModel):
    method: Literal["geometric_median_after_mad_screen"]
    candidate_frame_count: int
    finite_candidate_frame_count: int
    dropped_non_finite_frame_count: int
    inlier_frame_count: int
    mad_m: tuple[float, float, float]
    thresholds_m: tuple[float, float, float]
    iteration_count: int
    convergence_tolerance_m: float
    fallback_used: bool
    uncertainty: Literal["low", "medium", "high"]

    @field_validator("mad_m", "thresholds_m")
    @classmethod
    def validate_finite_triplet(
        cls,
        value: tuple[float, float, float],
        info: object,
    ) -> tuple[float, float, float]:
        field_name = getattr(info, "field_name", "estimator_triplet")
        return _validate_finite_triplet(value, field_name=field_name)


class SceneDirectionEstimatorRecord(SceneSchemaModel):
    method: Literal["angular_ransac_then_normalized_inlier_mean"]
    candidate_frame_count: int
    finite_candidate_frame_count: int
    inlier_frame_count: int
    inlier_angle_radians: float
    median_angular_residual_radians: float | None
    angular_residual_percentiles_radians: dict[str, float | None]
    fallback_used: bool
    uncertainty: Literal["low", "medium", "high"]

    @field_validator("angular_residual_percentiles_radians")
    @classmethod
    def validate_angular_residual_percentiles(
        cls,
        value: dict[str, float | None],
    ) -> dict[str, float | None]:
        expected_keys = {"p50", "p75", "p90", "p95"}
        actual_keys = set(value)
        if actual_keys != expected_keys:
            raise ValueError(
                "angular_residual_percentiles_radians must contain exactly "
                "p50, p75, p90, and p95"
            )
        for percentile_name, residual in value.items():
            if residual is not None and not math.isfinite(residual):
                raise ValueError(
                    "angular_residual_percentiles_radians values must be "
                    f"finite or null, got {percentile_name}={residual!r}"
                )
        return value


class SceneOrientationEstimatorRecord(SceneSchemaModel):
    method: Literal[
        "camera_stable_right_up_back_axes",
        "anatomical_frontal_webcam_right_up_back_axes",
    ]
    candidate_frame_count: int
    fallbacks: list[str]


class SceneRobustEstimatorsRecord(SceneSchemaModel):
    scene_center: SceneCenterEstimatorRecord
    main_unigaze_direction: SceneDirectionEstimatorRecord
    scene_orientation: SceneOrientationEstimatorRecord


class SceneViewerDependencyRecord(SceneSchemaModel):
    library: str
    version: str
    source: str
    license: str
    dist_integrity: str
    cdn_provider: str | None = None
    module_urls: dict[str, str] = Field(default_factory=dict)


class SceneManifest(SceneSchemaModel):
    schema_version: Literal["gaze-scene-manifest-v2"] = "gaze-scene-manifest-v2"
    run_id: str
    source_video_path: str
    source_video_sha256: str
    source_artifacts: SceneSourceArtifactsRecord
    coordinate_frames: SceneCoordinateFramesRecord
    camera_model: SceneCameraModel
    assumptions: list[SceneAssumptionRecord]
    robust_estimators: SceneRobustEstimatorsRecord
    scene_center_camera_m: Vector3D
    axis_basis: SceneAxisBasisRecord = Field(alias="scene_axes_camera")
    gaze_sphere: SceneGazeSphereRecord
    viewer_dependency: SceneViewerDependencyRecord = Field(alias="viewer")
    generated_at_utc: str

    @model_validator(mode="after")
    def validate_spaces(self) -> SceneManifest:
        _require_vector_space(
            self.scene_center_camera_m,
            expected=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            field_name="scene_center_camera_m",
        )
        return self


def _coerce_invalid_reason_counts(data: dict[str, int]) -> dict[str, int]:
    coerced: dict[str, int] = {}
    for key, value in data.items():
        coerced[str(SceneInvalidReason(key))] = value
    return coerced


class SceneSphereHitAngleBoundsRecord(SceneSchemaModel):
    theta_min_radians: float
    theta_max_radians: float
    phi_min_radians: float
    phi_max_radians: float
    front_hemisphere_frames: int
    rear_hemisphere_frames: int
    equator_frames: int


class SceneArtifactValidationRecord(SceneSchemaModel):
    scene_frame_count_matches_decoded: bool
    viewer_exists: bool
    scene_manifest_valid: bool
    scene_summary_valid: bool


class SceneSummary(SceneSchemaModel):
    schema_version: Literal["gaze-scene-summary-v2"] = "gaze-scene-summary-v2"
    run_id: str
    decoded_frames: int
    scene_frame_records: int
    valid_eye_midpoint_frames: int
    valid_unigaze_ray_frames: int
    valid_sphere_hit_frames: int
    invalid_sphere_hit_reasons: dict[str, int]
    sphere_hit_angle_bounds: SceneSphereHitAngleBoundsRecord
    representative_scene_warning_frame_ids: list[str]
    artifact_validation: SceneArtifactValidationRecord

    @field_validator("invalid_sphere_hit_reasons", mode="before")
    @classmethod
    def validate_invalid_reason_counts(
        cls,
        value: Any,
    ) -> dict[str, int]:
        if not isinstance(value, dict):
            raise TypeError("invalid_sphere_hit_reasons must be a dict")
        return _coerce_invalid_reason_counts(value)


class ViewerHitPoint(SceneSchemaModel):
    frame_id: str
    frame_index: int
    point_scene_m: Vector3D
    radius_m: float
    theta_radians: float
    phi_radians: float
    hemisphere: Literal["front", "rear", "equator"]

    @model_validator(mode="after")
    def validate_spaces(self) -> ViewerHitPoint:
        _require_vector_space(
            self.point_scene_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="point_scene_m",
        )
        if self.radius_m <= 0:
            raise ValueError("viewer hit point radius_m must be > 0")
        return self


class ViewerSceneData(SceneSchemaModel):
    schema_version: Literal["gaze-scene-viewer-data-v2"] = "gaze-scene-viewer-data-v2"
    run_id: str
    source_video_stem: str
    frame_count: int
    frames: list[SceneFrameRecord]
    valid_hit_points: list[ViewerHitPoint]
    gaze_sphere: SceneGazeSphereRecord
    axis_basis: SceneAxisBasisRecord
    assumptions: list[SceneAssumptionRecord]
    summary: SceneSummary
