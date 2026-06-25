from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chess_gaze.frame_records import (
    CalibrationRecord,
    FrameRecord,
    PnPLandmarkIndices,
)

raw_frame_image_format = "png"
processed_frame_image_format = "jpg"
processed_frame_jpeg_quality = 95
max_face_candidates = 4
candidate_face_score_min = 0.25
usable_face_score_min = 0.50
usable_eye_confidence_min = 0.50
default_iris_diameter_mm = 11.7
default_iris_diameter_uncertainty_mm = 0.5
unigaze_input_size_px = 224
unigaze_output_order = "pitch_yaw_radians"
face_landmarker_running_mode = "IMAGE"
camera_intrinsics_policy = "estimate_with_explicit_uncertainty"
metric_translation_allowed = False

PNP_LANDMARK_INDICES = PnPLandmarkIndices(
    nose_tip=1,
    chin=152,
    left_eye_outer=33,
    right_eye_outer=263,
    left_eye_inner=133,
    right_eye_inner=362,
    left_mouth_corner=61,
    right_mouth_corner=291,
)


@dataclass(frozen=True)
class DerivedSetupConstants:
    inter_pupil_distance_px: float | None = None
    iris_diameter_left_px: float | None = None
    iris_diameter_right_px: float | None = None
    derived_facecam_roi_image_px: tuple[float, float, float, float] | None = None
    mirror_policy: str = "unknown"
    measurement_usage: str = "qa_and_future_reconstruction_only"


def default_calibration() -> CalibrationRecord:
    return CalibrationRecord(
        raw_frame_image_format=raw_frame_image_format,
        processed_frame_image_format=processed_frame_image_format,
        processed_frame_jpeg_quality=processed_frame_jpeg_quality,
        max_face_candidates=max_face_candidates,
        candidate_face_score_min=candidate_face_score_min,
        usable_face_score_min=usable_face_score_min,
        usable_eye_confidence_min=usable_eye_confidence_min,
        default_iris_diameter_mm=default_iris_diameter_mm,
        default_iris_diameter_uncertainty_mm=default_iris_diameter_uncertainty_mm,
        unigaze_input_size_px=unigaze_input_size_px,
        unigaze_output_order=unigaze_output_order,
        face_landmarker_running_mode=face_landmarker_running_mode,
        camera_intrinsics_policy=camera_intrinsics_policy,
        metric_translation_allowed=metric_translation_allowed,
        pnp_landmark_indices=PNP_LANDMARK_INDICES,
    )


def derive_setup_constants(records: Iterable[FrameRecord]) -> DerivedSetupConstants:
    tuple(records)
    return DerivedSetupConstants()
