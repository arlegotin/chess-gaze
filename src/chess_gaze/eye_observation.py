from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, dist, floor, isfinite
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.errors import ErrorCode
from chess_gaze.face_landmark_indices import (
    MEDIAPIPE_ANATOMICAL_LEFT_EYE_CONTOUR_INDICES,
    MEDIAPIPE_ANATOMICAL_LEFT_IRIS_INDICES,
    MEDIAPIPE_ANATOMICAL_RIGHT_EYE_CONTOUR_INDICES,
    MEDIAPIPE_ANATOMICAL_RIGHT_IRIS_INDICES,
)
from chess_gaze.face_observation import FaceCandidate
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.image_io import save_rgb_png

LEFT_EYE_CONTOUR_INDICES = MEDIAPIPE_ANATOMICAL_LEFT_EYE_CONTOUR_INDICES
RIGHT_EYE_CONTOUR_INDICES = MEDIAPIPE_ANATOMICAL_RIGHT_EYE_CONTOUR_INDICES
LEFT_IRIS_INDICES = MEDIAPIPE_ANATOMICAL_LEFT_IRIS_INDICES
RIGHT_IRIS_INDICES = MEDIAPIPE_ANATOMICAL_RIGHT_IRIS_INDICES

CONFIDENCE_SOURCE_DERIVED_GEOMETRY = "derived_landmark_geometry"
LEFT_EYE_CROP_SPACE = "left_eye_crop_px"
RIGHT_EYE_CROP_SPACE = "right_eye_crop_px"

MIN_EYE_CONTOUR_LANDMARKS = 4
MIN_IRIS_LANDMARKS = 4
CLOSED_EYE_OPEN_METRIC_MAX = 0.03
EYE_CROP_PADDING_FRACTION = 0.25

SelectedFace = FaceCandidate
EyeSide = Literal["left", "right"]
OcclusionState = Literal["none", "partial", "severe", "unknown"]


@dataclass(frozen=True)
class CropTransformToImagePx:
    source_space: str
    target_space: CoordinateSpace
    m00: float
    m01: float
    m02: float
    m10: float
    m11: float
    m12: float


@dataclass(frozen=True)
class EyeObservation:
    present: bool
    confidence: float
    confidence_source: str
    reason_missing: ErrorCode | None
    eye_landmarks_image_px: tuple[Point2D, ...]
    eye_landmarks_image_norm: tuple[Point2D, ...]
    iris_present: bool
    iris_landmarks_image_px: tuple[Point2D, ...]
    iris_landmarks_image_norm: tuple[Point2D, ...]
    iris_center_image_px: Point2D | None
    iris_center_image_norm: Point2D | None
    iris_diameter_px: float | None
    bounding_box_image_px: BBox | None
    bounding_box_image_norm: BBox | None
    crop_bbox_image_px: BBox | None
    eye_crop_path: Path | None
    eye_crop_sha256: str | None
    eye_crop_transform_to_image_px: CropTransformToImagePx | None
    normalized_iris_offset_xy: tuple[float, float] | None
    eye_open_metric: float | None
    occlusion: OcclusionState


@dataclass(frozen=True)
class EyePairObservation:
    frame_id: str
    image_width_px: int
    image_height_px: int
    left: EyeObservation
    right: EyeObservation


@dataclass(frozen=True)
class _LandmarkPair:
    image_px: Point2D
    image_norm: Point2D


@dataclass(frozen=True)
class _CropRecord:
    path: Path | None
    sha256: str | None
    bbox_image_px: BBox
    transform_to_image_px: CropTransformToImagePx


def observe_eyes(
    face: SelectedFace,
    rgb_frame: npt.NDArray[np.uint8],
    run_layout: RunLayout,
    frame_id: str,
    *,
    save_crop_images: bool = False,
) -> EyePairObservation:
    frame = _validate_rgb_frame(rgb_frame)
    image_height_px, image_width_px, _channels = frame.shape

    return EyePairObservation(
        frame_id=frame_id,
        image_width_px=int(image_width_px),
        image_height_px=int(image_height_px),
        left=_observe_eye(
            side="left",
            face=face,
            rgb_frame=frame,
            run_layout=run_layout,
            frame_id=frame_id,
            eye_indices=LEFT_EYE_CONTOUR_INDICES,
            iris_indices=LEFT_IRIS_INDICES,
            eye_missing_code=ErrorCode.LEFT_EYE_NOT_FOUND,
            iris_missing_code=ErrorCode.LEFT_IRIS_NOT_FOUND,
            save_crop_images=save_crop_images,
        ),
        right=_observe_eye(
            side="right",
            face=face,
            rgb_frame=frame,
            run_layout=run_layout,
            frame_id=frame_id,
            eye_indices=RIGHT_EYE_CONTOUR_INDICES,
            iris_indices=RIGHT_IRIS_INDICES,
            eye_missing_code=ErrorCode.RIGHT_EYE_NOT_FOUND,
            iris_missing_code=ErrorCode.RIGHT_IRIS_NOT_FOUND,
            save_crop_images=save_crop_images,
        ),
    )


def _observe_eye(
    *,
    side: EyeSide,
    face: SelectedFace,
    rgb_frame: npt.NDArray[np.uint8],
    run_layout: RunLayout,
    frame_id: str,
    eye_indices: tuple[int, ...],
    iris_indices: tuple[int, ...],
    eye_missing_code: ErrorCode,
    iris_missing_code: ErrorCode,
    save_crop_images: bool,
) -> EyeObservation:
    eye_pairs = _landmarks_at_indices(face, eye_indices)
    eye_px = tuple(pair.image_px for pair in eye_pairs)
    eye_norm = tuple(pair.image_norm for pair in eye_pairs)
    eye_bbox_px = _bbox_from_points(eye_px)
    eye_bbox_norm = _bbox_from_points(eye_norm)

    if len(eye_pairs) < MIN_EYE_CONTOUR_LANDMARKS or eye_bbox_px is None:
        return _missing_eye(
            reason_missing=eye_missing_code,
            eye_landmarks_image_px=eye_px,
            eye_landmarks_image_norm=eye_norm,
            occlusion="unknown",
        )

    eye_open_metric = _eye_open_metric(eye_bbox_px)
    if (
        eye_open_metric is None
        or eye_open_metric <= CLOSED_EYE_OPEN_METRIC_MAX
        or eye_bbox_norm is None
    ):
        return _missing_eye(
            reason_missing=eye_missing_code,
            eye_landmarks_image_px=eye_px,
            eye_landmarks_image_norm=eye_norm,
            eye_open_metric=eye_open_metric,
            bounding_box_image_px=eye_bbox_px,
            bounding_box_image_norm=eye_bbox_norm,
            occlusion="severe",
        )

    crop_record = _eye_crop_record(
        side=side,
        rgb_frame=rgb_frame,
        bbox=eye_bbox_px,
        run_layout=run_layout,
        frame_id=frame_id,
        save_crop_images=save_crop_images,
    )
    if crop_record is None:
        return _missing_eye(
            reason_missing=eye_missing_code,
            eye_landmarks_image_px=eye_px,
            eye_landmarks_image_norm=eye_norm,
            eye_open_metric=eye_open_metric,
            bounding_box_image_px=eye_bbox_px,
            bounding_box_image_norm=eye_bbox_norm,
            occlusion="severe",
        )

    iris_pairs = _landmarks_at_indices(face, iris_indices)
    iris_px = tuple(pair.image_px for pair in iris_pairs)
    iris_norm = tuple(pair.image_norm for pair in iris_pairs)
    iris_center_px = _center_point(iris_px, CoordinateSpace.IMAGE_PX)
    iris_center_norm = _center_point(iris_norm, CoordinateSpace.NORMALIZED)
    iris_diameter_px = _diameter_px(iris_px)

    if (
        len(iris_pairs) < MIN_IRIS_LANDMARKS
        or iris_center_px is None
        or iris_center_norm is None
        or iris_diameter_px is None
    ):
        return EyeObservation(
            present=True,
            confidence=0.5,
            confidence_source=CONFIDENCE_SOURCE_DERIVED_GEOMETRY,
            reason_missing=iris_missing_code,
            eye_landmarks_image_px=eye_px,
            eye_landmarks_image_norm=eye_norm,
            iris_present=False,
            iris_landmarks_image_px=iris_px,
            iris_landmarks_image_norm=iris_norm,
            iris_center_image_px=None,
            iris_center_image_norm=None,
            iris_diameter_px=None,
            bounding_box_image_px=eye_bbox_px,
            bounding_box_image_norm=eye_bbox_norm,
            crop_bbox_image_px=crop_record.bbox_image_px,
            eye_crop_path=crop_record.path,
            eye_crop_sha256=crop_record.sha256,
            eye_crop_transform_to_image_px=crop_record.transform_to_image_px,
            normalized_iris_offset_xy=None,
            eye_open_metric=eye_open_metric,
            occlusion="partial",
        )

    return EyeObservation(
        present=True,
        confidence=1.0,
        confidence_source=CONFIDENCE_SOURCE_DERIVED_GEOMETRY,
        reason_missing=None,
        eye_landmarks_image_px=eye_px,
        eye_landmarks_image_norm=eye_norm,
        iris_present=True,
        iris_landmarks_image_px=iris_px,
        iris_landmarks_image_norm=iris_norm,
        iris_center_image_px=iris_center_px,
        iris_center_image_norm=iris_center_norm,
        iris_diameter_px=iris_diameter_px,
        bounding_box_image_px=eye_bbox_px,
        bounding_box_image_norm=eye_bbox_norm,
        crop_bbox_image_px=crop_record.bbox_image_px,
        eye_crop_path=crop_record.path,
        eye_crop_sha256=crop_record.sha256,
        eye_crop_transform_to_image_px=crop_record.transform_to_image_px,
        normalized_iris_offset_xy=_normalized_iris_offset(
            iris_center_px,
            eye_bbox_px,
        ),
        eye_open_metric=eye_open_metric,
        occlusion="none",
    )


def _missing_eye(
    *,
    reason_missing: ErrorCode,
    eye_landmarks_image_px: tuple[Point2D, ...],
    eye_landmarks_image_norm: tuple[Point2D, ...],
    eye_open_metric: float | None = None,
    bounding_box_image_px: BBox | None = None,
    bounding_box_image_norm: BBox | None = None,
    occlusion: OcclusionState,
) -> EyeObservation:
    return EyeObservation(
        present=False,
        confidence=0.0,
        confidence_source=CONFIDENCE_SOURCE_DERIVED_GEOMETRY,
        reason_missing=reason_missing,
        eye_landmarks_image_px=eye_landmarks_image_px,
        eye_landmarks_image_norm=eye_landmarks_image_norm,
        iris_present=False,
        iris_landmarks_image_px=(),
        iris_landmarks_image_norm=(),
        iris_center_image_px=None,
        iris_center_image_norm=None,
        iris_diameter_px=None,
        bounding_box_image_px=bounding_box_image_px,
        bounding_box_image_norm=bounding_box_image_norm,
        crop_bbox_image_px=None,
        eye_crop_path=None,
        eye_crop_sha256=None,
        eye_crop_transform_to_image_px=None,
        normalized_iris_offset_xy=None,
        eye_open_metric=eye_open_metric,
        occlusion=occlusion,
    )


def _landmarks_at_indices(
    face: SelectedFace,
    indices: tuple[int, ...],
) -> tuple[_LandmarkPair, ...]:
    pairs: list[_LandmarkPair] = []
    for index in indices:
        if index >= len(face.landmarks_image_px) or index >= len(
            face.landmarks_image_norm
        ):
            continue

        point_px = face.landmarks_image_px[index]
        point_norm = face.landmarks_image_norm[index]
        if not _valid_point(point_px) or not _valid_point(point_norm):
            continue
        if _is_absent_placeholder(point_px, point_norm):
            continue

        pairs.append(_LandmarkPair(image_px=point_px, image_norm=point_norm))
    return tuple(pairs)


def _valid_point(point: Point2D) -> bool:
    return isfinite(point.x) and isfinite(point.y)


def _is_absent_placeholder(point_px: Point2D, point_norm: Point2D) -> bool:
    return (
        point_px.x == 0.0
        and point_px.y == 0.0
        and point_norm.x == 0.0
        and point_norm.y == 0.0
    )


def _bbox_from_points(points: Sequence[Point2D]) -> BBox | None:
    if len(points) < 2:
        return None

    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    x_min = min(x_values)
    y_min = min(y_values)
    x_max = max(x_values)
    y_max = max(y_values)
    if x_max <= x_min or y_max <= y_min:
        return None

    return BBox(
        space=points[0].space,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def _eye_open_metric(bbox: BBox) -> float | None:
    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min
    if width <= 0.0:
        return None
    return height / width


def _eye_crop_record(
    *,
    side: EyeSide,
    rgb_frame: npt.NDArray[np.uint8],
    bbox: BBox,
    run_layout: RunLayout,
    frame_id: str,
    save_crop_images: bool,
) -> _CropRecord | None:
    frame_height, frame_width, _channels = rgb_frame.shape
    crop_bounds = _crop_bounds(
        bbox,
        image_width_px=int(frame_width),
        image_height_px=int(frame_height),
    )
    if crop_bounds is None:
        return None

    x_min, y_min, x_max, y_max = crop_bounds
    relative_path: Path | None = None
    crop_sha256: str | None = None
    if save_crop_images:
        crop = rgb_frame[y_min:y_max, x_min:x_max]
        crop_path = _absolute_crop_path(run_layout, side, frame_id)
        crop_sha256 = save_rgb_png(crop_path, crop)
        relative_path = run_layout.relative_artifact_path(crop_path)
    crop_bbox = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=float(x_min),
        y_min=float(y_min),
        x_max=float(x_max),
        y_max=float(y_max),
    )

    return _CropRecord(
        path=relative_path,
        sha256=crop_sha256,
        bbox_image_px=crop_bbox,
        transform_to_image_px=CropTransformToImagePx(
            source_space=(
                LEFT_EYE_CROP_SPACE if side == "left" else RIGHT_EYE_CROP_SPACE
            ),
            target_space=CoordinateSpace.IMAGE_PX,
            m00=1.0,
            m01=0.0,
            m02=float(x_min),
            m10=0.0,
            m11=1.0,
            m12=float(y_min),
        ),
    )


def _absolute_crop_path(run_layout: RunLayout, side: EyeSide, frame_id: str) -> Path:
    if side == "left":
        return run_layout.left_eye_crops_dir / f"{frame_id}.png"
    return run_layout.right_eye_crops_dir / f"{frame_id}.png"


def _crop_bounds(
    bbox: BBox,
    *,
    image_width_px: int,
    image_height_px: int,
) -> tuple[int, int, int, int] | None:
    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min
    padding_x = width * EYE_CROP_PADDING_FRACTION
    padding_y = height * EYE_CROP_PADDING_FRACTION

    x_min = _clamp_int(floor(bbox.x_min - padding_x), 0, image_width_px)
    y_min = _clamp_int(floor(bbox.y_min - padding_y), 0, image_height_px)
    x_max = _clamp_int(ceil(bbox.x_max + padding_x), 0, image_width_px)
    y_max = _clamp_int(ceil(bbox.y_max + padding_y), 0, image_height_px)
    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return min(max(value, lower), upper)


def _center_point(
    points: Sequence[Point2D],
    space: CoordinateSpace,
) -> Point2D | None:
    if not points:
        return None
    return Point2D(
        space=space,
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
    )


def _diameter_px(points: Sequence[Point2D]) -> float | None:
    if len(points) < 2:
        return None

    max_distance = 0.0
    for index, point in enumerate(points):
        for other_point in points[index + 1 :]:
            max_distance = max(
                max_distance,
                dist((point.x, point.y), (other_point.x, other_point.y)),
            )

    return max_distance if max_distance > 0.0 else None


def _normalized_iris_offset(
    iris_center: Point2D,
    eye_bbox: BBox,
) -> tuple[float, float] | None:
    width = eye_bbox.x_max - eye_bbox.x_min
    height = eye_bbox.y_max - eye_bbox.y_min
    if width <= 0.0 or height <= 0.0:
        return None

    center_x = eye_bbox.x_min + (width / 2.0)
    center_y = eye_bbox.y_min + (height / 2.0)
    return (
        (iris_center.x - center_x) / (width / 2.0),
        (iris_center.y - center_y) / (height / 2.0),
    )


def _validate_rgb_frame(
    rgb_frame: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError("rgb_frame must have shape (height, width, 3)")
    if rgb_frame.dtype != np.uint8:
        raise ValueError("rgb_frame must have dtype uint8")
    return np.ascontiguousarray(rgb_frame)
