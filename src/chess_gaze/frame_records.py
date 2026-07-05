from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.geometry import BBox, Point2D, RotationRadians, StrictSchemaModel
from chess_gaze.unigaze_preprocessing import (
    LEGACY_UNIGAZE_FACE_CROP_SCALE,
    LEGACY_UNIGAZE_PREPROCESSING_PROFILE,
    UniGazePreprocessingProfile,
)


def _coerce_error_code_field(payload: dict[str, Any], field_name: str) -> None:
    value = payload.get(field_name)
    if isinstance(value, str):
        try:
            payload[field_name] = ErrorCode(value)
        except ValueError:
            return


def _coerce_error_code_record(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    coerced = dict(payload)
    _coerce_error_code_field(coerced, "reason_invalid")
    return coerced


def _coerce_error_list(payload: Any) -> Any:
    if not isinstance(payload, list):
        return payload

    coerced_errors: list[Any] = []
    for error_payload in payload:
        if not isinstance(error_payload, dict):
            coerced_errors.append(error_payload)
            continue

        coerced_error = dict(error_payload)
        value = coerced_error.get("code")
        if isinstance(value, str):
            try:
                coerced_error["code"] = ErrorCode(value)
            except ValueError:
                pass
        coerced_errors.append(coerced_error)

    return coerced_errors


class GazeAngles(StrictSchemaModel):
    valid: bool
    yaw_radians: RotationRadians | None
    pitch_radians: RotationRadians | None
    reason_invalid: ErrorCode | None

    @model_validator(mode="after")
    def validate_complete_angles(self) -> GazeAngles:
        if self.valid and (self.yaw_radians is None or self.pitch_radians is None):
            raise ValueError("valid gaze requires both yaw and pitch")
        return self


class HeadPoseRecord(StrictSchemaModel):
    valid: bool
    yaw_radians: RotationRadians | None
    pitch_radians: RotationRadians | None
    roll_radians: RotationRadians | None
    reason_invalid: ErrorCode | None


class FaceRecord(StrictSchemaModel):
    present: bool
    bounding_box: BBox | None
    landmarks: list[Point2D] | None
    reason_invalid: ErrorCode | None

    @model_validator(mode="after")
    def validate_face_landmarks(self) -> FaceRecord:
        if self.present and (self.bounding_box is None or not self.landmarks):
            raise ValueError("present face requires bounding box and landmarks")
        return self


class EyeRecord(StrictSchemaModel):
    present: bool
    bounding_box: BBox | None
    pupil_center: Point2D | None
    iris_landmarks: list[Point2D] | None
    reason_invalid: ErrorCode | None

    @model_validator(mode="after")
    def validate_eye_landmarks(self) -> EyeRecord:
        if self.present and (
            self.bounding_box is None
            or self.pupil_center is None
            or not self.iris_landmarks
        ):
            raise ValueError(
                "present eye requires bounding box, pupil center, and iris landmarks"
            )
        return self


class ErrorRecord(StrictSchemaModel):
    code: ErrorCode
    message: str


class FrameErrorRecord(StrictSchemaModel):
    frame_id: str
    frame_index: int
    code: ErrorCode
    message: str

    @model_validator(mode="before")
    @classmethod
    def coerce_artifact_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        coerced = dict(data)
        _coerce_error_code_field(coerced, "code")
        return coerced


class PnPLandmarkIndices(StrictSchemaModel):
    nose_tip: int
    chin: int
    left_eye_outer: int
    right_eye_outer: int
    left_eye_inner: int
    right_eye_inner: int
    left_mouth_corner: int
    right_mouth_corner: int


class FrameRecord(StrictSchemaModel):
    frame_id: str
    frame_index: int
    status: FrameStatus
    timestamp_seconds: float
    face: FaceRecord
    left_eye: EyeRecord
    right_eye: EyeRecord
    head_pose: HeadPoseRecord
    geometric_gaze: GazeAngles
    appearance_gaze: GazeAngles
    recommended_gaze: GazeAngles
    errors: list[ErrorRecord]

    @model_validator(mode="before")
    @classmethod
    def coerce_artifact_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        coerced = dict(data)

        status = coerced.get("status")
        if isinstance(status, str):
            try:
                coerced["status"] = FrameStatus(status)
            except ValueError:
                pass

        for field_name in (
            "face",
            "left_eye",
            "right_eye",
            "head_pose",
            "geometric_gaze",
            "appearance_gaze",
            "recommended_gaze",
        ):
            coerced[field_name] = _coerce_error_code_record(coerced.get(field_name))

        coerced["errors"] = _coerce_error_list(coerced.get("errors"))
        return coerced


class VideoManifest(StrictSchemaModel):
    source_path: str
    source_sha256: str
    frame_width: int
    frame_height: int
    frame_count_decoded: int


class InferenceRuntimeRecord(StrictSchemaModel):
    schema_version: Literal["inference-runtime-v1"] = "inference-runtime-v1"
    observer_source: Literal[
        "default_model_observer",
        "external_observer",
        "legacy_manifest_without_inference",
    ]
    unigaze_model_id: str | None
    unigaze_device: Literal["cpu", "mps", "not_applicable"]
    unigaze_batch_size: int | None
    torch_version: str | None
    torch_mps_available: bool | None
    mps_fallback_env: str
    mps_fast_math_env: str
    mps_prefer_metal_env: str
    mps_preflight_passed: bool | None

    @model_validator(mode="after")
    def validate_runtime_semantics(self) -> InferenceRuntimeRecord:
        issues: list[str] = []

        if self.observer_source in {
            "external_observer",
            "legacy_manifest_without_inference",
        }:
            if self.unigaze_model_id is not None:
                issues.append(
                    f"{self.observer_source} cannot declare a UniGaze model identifier"
                )
            if self.unigaze_device != "not_applicable":
                issues.append(
                    f"{self.observer_source} must use unigaze_device=not_applicable"
                )
            if self.unigaze_batch_size is not None:
                issues.append(
                    f"{self.observer_source} cannot declare unigaze_batch_size"
                )
            if self.torch_version is not None:
                issues.append(f"{self.observer_source} cannot declare torch_version")
            if self.torch_mps_available is not None:
                issues.append(
                    f"{self.observer_source} cannot declare torch_mps_available"
                )
            if self.mps_fallback_env != "not_applicable":
                issues.append(
                    f"{self.observer_source} must use mps_fallback_env=not_applicable"
                )
            if self.mps_fast_math_env != "not_applicable":
                issues.append(
                    f"{self.observer_source} must use mps_fast_math_env=not_applicable"
                )
            if self.mps_prefer_metal_env != "not_applicable":
                issues.append(
                    f"{self.observer_source} must use "
                    "mps_prefer_metal_env=not_applicable"
                )
            if self.mps_preflight_passed is not None:
                issues.append(
                    f"{self.observer_source} cannot declare mps_preflight_passed"
                )

        if self.observer_source == "default_model_observer":
            if self.unigaze_model_id is None or not self.unigaze_model_id.strip():
                issues.append(
                    "default_model_observer requires a UniGaze model identifier"
                )
            if self.unigaze_device == "not_applicable":
                issues.append(
                    "default_model_observer cannot use unigaze_device=not_applicable"
                )
            if self.unigaze_batch_size is None:
                issues.append("default_model_observer requires unigaze_batch_size")
            elif self.unigaze_batch_size <= 0:
                issues.append("default_model_observer requires unigaze_batch_size >= 1")
            if self.torch_version is None or not self.torch_version.strip():
                issues.append("default_model_observer requires torch_version")
            if self.torch_mps_available is None:
                issues.append("default_model_observer requires torch_mps_available")
            if self.mps_fallback_env == "not_applicable":
                issues.append(
                    "default_model_observer cannot use mps_fallback_env=not_applicable"
                )
            if self.mps_fast_math_env == "not_applicable":
                issues.append(
                    "default_model_observer cannot use mps_fast_math_env=not_applicable"
                )
            if self.mps_prefer_metal_env == "not_applicable":
                issues.append(
                    "default_model_observer cannot use "
                    "mps_prefer_metal_env=not_applicable"
                )
            if self.unigaze_device == "cpu" and self.mps_preflight_passed is not None:
                issues.append(
                    "default_model_observer with unigaze_device=cpu requires "
                    "mps_preflight_passed=None"
                )
            elif self.unigaze_device == "mps" and self.mps_preflight_passed is not True:
                issues.append(
                    "default_model_observer with unigaze_device=mps requires "
                    "mps_preflight_passed=True"
                )
            if self.unigaze_device == "mps" and self.torch_mps_available is not True:
                issues.append(
                    "default_model_observer with unigaze_device=mps requires "
                    "torch_mps_available=True"
                )

        if issues:
            raise ValueError("; ".join(issues))
        return self


class FrameImageRetentionPolicy(StrictSchemaModel):
    schema_version: Literal["frame-image-retention-v1"] = "frame-image-retention-v1"
    save_frame_images: bool


class CropImageRetentionPolicy(StrictSchemaModel):
    schema_version: Literal["crop-image-retention-v1"] = "crop-image-retention-v1"
    save_crop_images: bool


class QASummaryPolicy(StrictSchemaModel):
    schema_version: Literal["qa-summary-policy-v1"] = "qa-summary-policy-v1"
    generate_qa_summary: bool


class RunManifest(StrictSchemaModel):
    run_id: str
    created_at_utc: str
    input_path: str
    video: VideoManifest
    inference: InferenceRuntimeRecord
    frame_image_retention: FrameImageRetentionPolicy = Field(
        default_factory=lambda: FrameImageRetentionPolicy(save_frame_images=True)
    )
    crop_image_retention: CropImageRetentionPolicy = Field(
        default_factory=lambda: CropImageRetentionPolicy(save_crop_images=True)
    )
    qa_summary_policy: QASummaryPolicy = Field(
        default_factory=lambda: QASummaryPolicy(generate_qa_summary=True)
    )


def _legacy_artifact_inference_record() -> InferenceRuntimeRecord:
    # Legacy run manifests predate inference metadata. Use the least-claiming valid
    # record so artifact readers stay strict without inventing model-runtime facts.
    return InferenceRuntimeRecord(
        observer_source="legacy_manifest_without_inference",
        unigaze_model_id=None,
        unigaze_device="not_applicable",
        unigaze_batch_size=None,
        torch_version=None,
        torch_mps_available=None,
        mps_fallback_env="not_applicable",
        mps_fast_math_env="not_applicable",
        mps_prefer_metal_env="not_applicable",
        mps_preflight_passed=None,
    )


def read_run_manifest_artifact_json(payload: str) -> RunManifest:
    raw_manifest = json.loads(payload)
    if not isinstance(raw_manifest, dict) or "inference" in raw_manifest:
        return RunManifest.model_validate(raw_manifest)

    legacy_manifest = dict(raw_manifest)
    legacy_manifest["inference"] = _legacy_artifact_inference_record().model_dump()
    return RunManifest.model_validate(legacy_manifest)


class CalibrationRecord(StrictSchemaModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    raw_frame_image_format: str
    processed_frame_image_format: str
    processed_frame_jpeg_quality: int
    max_face_candidates: int
    candidate_face_score_min: float
    usable_face_score_min: float
    usable_eye_confidence_min: float
    default_iris_diameter_mm: float
    default_iris_diameter_uncertainty_mm: float
    unigaze_input_size_px: int
    unigaze_output_order: str
    unigaze_preprocessing_profile: UniGazePreprocessingProfile = (
        LEGACY_UNIGAZE_PREPROCESSING_PROFILE
    )
    unigaze_face_crop_scale: float = LEGACY_UNIGAZE_FACE_CROP_SCALE
    unigaze_image_mean_rgb: tuple[float, float, float] | None = None
    unigaze_image_std_rgb: tuple[float, float, float] | None = None
    target_plane_origin_camera_m: tuple[float, float, float] | None = None
    target_plane_x_axis_camera: tuple[float, float, float] | None = None
    target_plane_y_axis_camera: tuple[float, float, float] | None = None
    target_plane_width_m: float | None = None
    target_plane_height_m: float | None = None
    target_plane_mirror_horizontal: bool = False
    face_landmarker_running_mode: str
    camera_intrinsics_policy: str
    metric_translation_allowed: bool
    derived_percentile_lower: float
    derived_percentile_upper: float
    pnp_landmark_indices: PnPLandmarkIndices

    @field_validator("unigaze_image_mean_rgb", "unigaze_image_std_rgb", mode="before")
    @classmethod
    def coerce_rgb_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator(
        "target_plane_origin_camera_m",
        "target_plane_x_axis_camera",
        "target_plane_y_axis_camera",
        mode="before",
    )
    @classmethod
    def coerce_target_plane_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_unigaze_preprocessing(self) -> CalibrationRecord:
        if self.unigaze_face_crop_scale <= 0.0:
            raise ValueError("unigaze_face_crop_scale must be positive")
        if (self.unigaze_image_mean_rgb is None) != (
            self.unigaze_image_std_rgb is None
        ):
            raise ValueError(
                "unigaze_image_mean_rgb and unigaze_image_std_rgb must both be set"
            )
        return self

    @model_validator(mode="after")
    def validate_target_plane(self) -> CalibrationRecord:
        fields = (
            self.target_plane_origin_camera_m,
            self.target_plane_x_axis_camera,
            self.target_plane_y_axis_camera,
            self.target_plane_width_m,
            self.target_plane_height_m,
        )
        configured = [field is not None for field in fields]
        if any(configured) and not all(configured):
            raise ValueError(
                "target plane calibration fields must be all set or all null"
            )
        for field_name in (
            "target_plane_origin_camera_m",
            "target_plane_x_axis_camera",
            "target_plane_y_axis_camera",
        ):
            value = getattr(self, field_name)
            if value is not None:
                if len(value) != 3:
                    raise ValueError(f"{field_name} must contain exactly three values")
                for coordinate in value:
                    if not math.isfinite(coordinate):
                        raise ValueError(f"{field_name} must contain finite values")
        if self.target_plane_width_m is not None and self.target_plane_width_m <= 0.0:
            raise ValueError("target_plane_width_m must be positive")
        if self.target_plane_height_m is not None and self.target_plane_height_m <= 0.0:
            raise ValueError("target_plane_height_m must be positive")
        return self
