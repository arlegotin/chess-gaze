from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, model_validator

from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.geometry import BBox, Point2D, RotationRadians, StrictSchemaModel


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


class RunManifest(StrictSchemaModel):
    run_id: str
    created_at_utc: str
    input_path: str
    video: VideoManifest


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
    face_landmarker_running_mode: str
    camera_intrinsics_policy: str
    metric_translation_allowed: bool
    derived_percentile_lower: float
    derived_percentile_upper: float
    pnp_landmark_indices: PnPLandmarkIndices
