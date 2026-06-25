from __future__ import annotations

from pydantic import model_validator

from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.geometry import BBox, Point2D, StrictSchemaModel


class GazeAngles(StrictSchemaModel):
    valid: bool
    yaw_radians: float | None
    pitch_radians: float | None
    reason_invalid: ErrorCode | None

    @model_validator(mode="after")
    def validate_complete_angles(self) -> GazeAngles:
        if self.valid and (
            self.yaw_radians is None or self.pitch_radians is None
        ):
            raise ValueError("valid gaze requires both yaw and pitch")
        return self


class HeadPoseRecord(StrictSchemaModel):
    valid: bool
    yaw_radians: float | None
    pitch_radians: float | None
    roll_radians: float | None
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
    raw_frame_image_format: str
    processed_frame_image_format: str
    processed_frame_jpeg_quality: int
