from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import dist
from statistics import median
from typing import Any

from chess_gaze.face_landmark_indices import (
    MEDIAPIPE_ANATOMICAL_LEFT_EYE_INNER_INDEX,
    MEDIAPIPE_ANATOMICAL_LEFT_EYE_OUTER_INDEX,
    MEDIAPIPE_ANATOMICAL_LEFT_MOUTH_CORNER_INDEX,
    MEDIAPIPE_ANATOMICAL_RIGHT_EYE_INNER_INDEX,
    MEDIAPIPE_ANATOMICAL_RIGHT_EYE_OUTER_INDEX,
    MEDIAPIPE_ANATOMICAL_RIGHT_MOUTH_CORNER_INDEX,
    MEDIAPIPE_CHIN_INDEX,
    MEDIAPIPE_NOSE_TIP_INDEX,
)
from chess_gaze.frame_records import (
    CalibrationRecord,
    EyeRecord,
    FrameRecord,
    PnPLandmarkIndices,
)
from chess_gaze.unigaze_preprocessing import (
    DEFAULT_UNIGAZE_PREPROCESSING_PROFILE,
    resolve_unigaze_preprocessing_profile,
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
derived_percentile_lower = 0.05
derived_percentile_upper = 0.95

PNP_LANDMARK_INDICES = PnPLandmarkIndices(
    nose_tip=MEDIAPIPE_NOSE_TIP_INDEX,
    chin=MEDIAPIPE_CHIN_INDEX,
    left_eye_outer=MEDIAPIPE_ANATOMICAL_LEFT_EYE_OUTER_INDEX,
    right_eye_outer=MEDIAPIPE_ANATOMICAL_RIGHT_EYE_OUTER_INDEX,
    left_eye_inner=MEDIAPIPE_ANATOMICAL_LEFT_EYE_INNER_INDEX,
    right_eye_inner=MEDIAPIPE_ANATOMICAL_RIGHT_EYE_INNER_INDEX,
    left_mouth_corner=MEDIAPIPE_ANATOMICAL_LEFT_MOUTH_CORNER_INDEX,
    right_mouth_corner=MEDIAPIPE_ANATOMICAL_RIGHT_MOUTH_CORNER_INDEX,
)

PERCENTILE_POLICY_DESCRIPTION = (
    f"percentile policy lower={derived_percentile_lower} "
    f"upper={derived_percentile_upper} using linear interpolation"
)
FACE_BBOX_DERIVATION_METHOD = (
    "median plus p05/p95 percentile range of selected-face bounding box width, "
    "height, and area from face.bounding_box where face.present is true; "
    f"{PERCENTILE_POLICY_DESCRIPTION}"
)
INTER_PUPIL_DERIVATION_METHOD = (
    "median plus p05/p95 percentile range of Euclidean distance between left "
    "and right pupil centers when both eyes are present; "
    f"{PERCENTILE_POLICY_DESCRIPTION}"
)
LEFT_IRIS_DERIVATION_METHOD = (
    "median plus p05/p95 percentile range of maximum pairwise iris landmark "
    "distance per frame for the left eye; "
    f"{PERCENTILE_POLICY_DESCRIPTION}"
)
RIGHT_IRIS_DERIVATION_METHOD = (
    "median plus p05/p95 percentile range of maximum pairwise iris landmark "
    "distance per frame for the right eye; "
    f"{PERCENTILE_POLICY_DESCRIPTION}"
)
FACECAM_ROI_DERIVATION_METHOD = (
    "bounding box union over observed selected-face boxes; derived ROI for QA only"
)
CAMERA_INTRINSICS_DERIVATION_METHOD = (
    "policy placeholder derived from calibration defaults; does not authorize "
    "metric camera_3d_m translation"
)
MIRROR_POLICY_DERIVATION_METHOD = (
    "no mirror-evidence signal is available in current FrameRecord schemas"
)


@dataclass(frozen=True)
class DerivedConstantRecord:
    value: Any
    unit: str | None
    coordinate_space: str | None
    derivation_method: str
    contributing_frame_count: int
    uncertainty: str
    usage: str


@dataclass(frozen=True)
class DerivedSetupConstants:
    selected_face_bbox_size_image_px: DerivedConstantRecord
    inter_pupil_distance_image_px: DerivedConstantRecord
    left_iris_diameter_image_px: DerivedConstantRecord
    right_iris_diameter_image_px: DerivedConstantRecord
    estimated_camera_intrinsics_policy: DerivedConstantRecord
    facecam_roi_image_px: DerivedConstantRecord
    mirror_policy: DerivedConstantRecord


def _null_record(
    *,
    unit: str | None,
    coordinate_space: str | None,
    derivation_method: str,
    uncertainty: str,
    usage: str,
    value: Any = None,
) -> DerivedConstantRecord:
    return DerivedConstantRecord(
        value=value,
        unit=unit,
        coordinate_space=coordinate_space,
        derivation_method=derivation_method,
        contributing_frame_count=0,
        uncertainty=uncertainty,
        usage=usage,
    )


def _median_measurement(
    values: Sequence[float],
    *,
    derivation_method: str,
    uncertainty: str,
    usage: str,
) -> DerivedConstantRecord:
    if not values:
        return _null_record(
            unit="image_px",
            coordinate_space="image_px",
            derivation_method=derivation_method,
            uncertainty=uncertainty,
            usage=usage,
        )
    return DerivedConstantRecord(
        value=_measurement_summary(values),
        unit="image_px",
        coordinate_space="image_px",
        derivation_method=derivation_method,
        contributing_frame_count=len(values),
        uncertainty=uncertainty,
        usage=usage,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * percentile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + ((upper_value - lower_value) * fraction)


def _measurement_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": median(values),
        "p05": _percentile(values, derived_percentile_lower),
        "p95": _percentile(values, derived_percentile_upper),
    }


def _iris_diameter(eye: EyeRecord) -> float | None:
    if not eye.present or not eye.iris_landmarks:
        return None

    max_distance = 0.0
    for index, point in enumerate(eye.iris_landmarks):
        for other_point in eye.iris_landmarks[index + 1 :]:
            max_distance = max(
                max_distance,
                dist((point.x, point.y), (other_point.x, other_point.y)),
            )
    return max_distance if max_distance > 0.0 else None


def default_calibration(
    *,
    unigaze_preprocessing_profile: str = DEFAULT_UNIGAZE_PREPROCESSING_PROFILE,
    target_plane_origin_camera_m: tuple[float, float, float] | None = None,
    target_plane_x_axis_camera: tuple[float, float, float] | None = None,
    target_plane_y_axis_camera: tuple[float, float, float] | None = None,
    target_plane_width_m: float | None = None,
    target_plane_height_m: float | None = None,
    target_plane_mirror_horizontal: bool = False,
) -> CalibrationRecord:
    unigaze_preprocessing = resolve_unigaze_preprocessing_profile(
        unigaze_preprocessing_profile
    )
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
        unigaze_preprocessing_profile=unigaze_preprocessing.profile,
        unigaze_face_crop_scale=unigaze_preprocessing.crop_scale,
        unigaze_image_mean_rgb=unigaze_preprocessing.image_mean_rgb,
        unigaze_image_std_rgb=unigaze_preprocessing.image_std_rgb,
        target_plane_origin_camera_m=target_plane_origin_camera_m,
        target_plane_x_axis_camera=target_plane_x_axis_camera,
        target_plane_y_axis_camera=target_plane_y_axis_camera,
        target_plane_width_m=target_plane_width_m,
        target_plane_height_m=target_plane_height_m,
        target_plane_mirror_horizontal=target_plane_mirror_horizontal,
        face_landmarker_running_mode=face_landmarker_running_mode,
        camera_intrinsics_policy=camera_intrinsics_policy,
        metric_translation_allowed=metric_translation_allowed,
        derived_percentile_lower=derived_percentile_lower,
        derived_percentile_upper=derived_percentile_upper,
        pnp_landmark_indices=PNP_LANDMARK_INDICES,
    )


def derive_setup_constants(records: Iterable[FrameRecord]) -> DerivedSetupConstants:
    frame_records = tuple(records)
    face_widths: list[float] = []
    face_heights: list[float] = []
    face_areas: list[float] = []
    inter_pupil_distances: list[float] = []
    left_iris_diameters: list[float] = []
    right_iris_diameters: list[float] = []
    roi_boxes: list[tuple[float, float, float, float]] = []

    for record in frame_records:
        if record.face.present and record.face.bounding_box is not None:
            bbox = record.face.bounding_box
            width = bbox.x_max - bbox.x_min
            height = bbox.y_max - bbox.y_min
            face_widths.append(width)
            face_heights.append(height)
            face_areas.append(width * height)
            roi_boxes.append((bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max))

        if record.left_eye.present and record.right_eye.present:
            left_pupil = record.left_eye.pupil_center
            right_pupil = record.right_eye.pupil_center
            if left_pupil is not None and right_pupil is not None:
                inter_pupil_distances.append(
                    dist((left_pupil.x, left_pupil.y), (right_pupil.x, right_pupil.y))
                )

        left_iris_diameter = _iris_diameter(record.left_eye)
        if left_iris_diameter is not None:
            left_iris_diameters.append(left_iris_diameter)

        right_iris_diameter = _iris_diameter(record.right_eye)
        if right_iris_diameter is not None:
            right_iris_diameters.append(right_iris_diameter)

    if face_areas:
        selected_face_bbox_size_image_px = DerivedConstantRecord(
            value={
                "width_px": _measurement_summary(face_widths),
                "height_px": _measurement_summary(face_heights),
                "area_px2": _measurement_summary(face_areas),
            },
            unit="image_px",
            coordinate_space="image_px",
            derivation_method=FACE_BBOX_DERIVATION_METHOD,
            contributing_frame_count=len(face_areas),
            uncertainty="low",
            usage="measurement",
        )
    else:
        selected_face_bbox_size_image_px = _null_record(
            unit="image_px",
            coordinate_space="image_px",
            derivation_method=FACE_BBOX_DERIVATION_METHOD,
            uncertainty="low",
            usage="measurement",
        )

    if roi_boxes:
        x_mins, y_mins, x_maxs, y_maxs = zip(*roi_boxes, strict=True)
        facecam_roi_image_px = DerivedConstantRecord(
            value={
                "x_min": min(x_mins),
                "y_min": min(y_mins),
                "x_max": max(x_maxs),
                "y_max": max(y_maxs),
            },
            unit="image_px",
            coordinate_space="image_px",
            derivation_method=FACECAM_ROI_DERIVATION_METHOD,
            contributing_frame_count=len(roi_boxes),
            uncertainty="medium",
            usage="qa_only",
        )
    else:
        facecam_roi_image_px = _null_record(
            unit="image_px",
            coordinate_space="image_px",
            derivation_method=FACECAM_ROI_DERIVATION_METHOD,
            uncertainty="medium",
            usage="qa_only",
        )

    return DerivedSetupConstants(
        selected_face_bbox_size_image_px=selected_face_bbox_size_image_px,
        inter_pupil_distance_image_px=_median_measurement(
            inter_pupil_distances,
            derivation_method=INTER_PUPIL_DERIVATION_METHOD,
            uncertainty="low",
            usage="measurement",
        ),
        left_iris_diameter_image_px=_median_measurement(
            left_iris_diameters,
            derivation_method=LEFT_IRIS_DERIVATION_METHOD,
            uncertainty="medium",
            usage="measurement",
        ),
        right_iris_diameter_image_px=_median_measurement(
            right_iris_diameters,
            derivation_method=RIGHT_IRIS_DERIVATION_METHOD,
            uncertainty="medium",
            usage="measurement",
        ),
        estimated_camera_intrinsics_policy=_null_record(
            unit=None,
            coordinate_space=None,
            derivation_method=CAMERA_INTRINSICS_DERIVATION_METHOD,
            uncertainty="high",
            usage="future_use",
            value=camera_intrinsics_policy,
        ),
        facecam_roi_image_px=facecam_roi_image_px,
        mirror_policy=_null_record(
            unit=None,
            coordinate_space=None,
            derivation_method=MIRROR_POLICY_DERIVATION_METHOD,
            uncertainty="high",
            usage="future_use",
            value="unknown",
        ),
    )
