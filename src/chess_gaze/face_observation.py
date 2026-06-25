from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from chess_gaze.errors import ErrorCode
from chess_gaze.frame_records import CalibrationRecord, ErrorRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D

MODEL_SCORE_SELECTION_SOURCE = "model_score_times_area_fraction"
AREA_ONLY_SELECTION_SOURCE = "area_only_no_model_score"
MEDIAPIPE_SCORE_SOURCE_UNAVAILABLE = "not_exposed_by_mediapipe_face_landmarker"
MEDIAPIPE_IMAGE_RUNNING_MODE = "IMAGE"
DEFAULT_FRAME_ID = "unknown"
DETECTION_REGION_FULL_FRAME = "full_frame"
DETECTION_REGION_LEFT_HALF = "left_half"
DETECTION_REGION_RIGHT_HALF = "right_half"
REGION_REFINEMENT_MIN_IOU = 0.25
REGION_REFINEMENT_TOP_SHIFT_MIN_PX = 8.0
REGION_REFINEMENT_TOP_SHIFT_FRACTION = 0.15
LOW_FULL_FRAME_FACE_TOP_FRACTION = 0.60
REGION_SEAM_MARGIN_FRACTION = 0.02
REGION_SEAM_MARGIN_MIN_PX = 4.0


@dataclass(frozen=True)
class BlendshapeScore:
    category_name: str
    score: float


@dataclass(frozen=True)
class FaceCandidate:
    candidate_id: str
    frame_id: str
    image_width_px: int
    image_height_px: int
    candidate_score: float | None
    score_source: str
    bounding_box_image_px: BBox
    bounding_box_image_norm: BBox
    landmarks_image_px: Sequence[Point2D]
    landmarks_image_norm: Sequence[Point2D]
    selection_score: float | None = None
    selection_score_source: str | None = None
    blendshapes: Sequence[BlendshapeScore] = ()
    facial_transformation_matrix: tuple[tuple[float, ...], ...] | None = None

    @property
    def area_fraction(self) -> float:
        return (
            self.bounding_box_image_norm.x_max - self.bounding_box_image_norm.x_min
        ) * (self.bounding_box_image_norm.y_max - self.bounding_box_image_norm.y_min)

    @property
    def has_valid_landmarks(self) -> bool:
        return bool(self.landmarks_image_px) and bool(self.landmarks_image_norm)


@dataclass(frozen=True)
class FaceSelection:
    present: bool
    primary_candidate_id: str | None
    selection_reason: str
    selection_score_source: str | None
    reason_invalid: ErrorCode | None
    candidates: tuple[FaceCandidate, ...]
    errors: tuple[ErrorRecord, ...]


@dataclass(frozen=True)
class FaceLandmarkerOptionsRecord:
    running_mode: str
    num_faces: int
    min_face_detection_confidence: float
    min_face_presence_confidence: float
    min_tracking_confidence_source: str
    output_face_blendshapes: bool
    output_facial_transformation_matrixes: bool


@dataclass(frozen=True)
class FaceObservation:
    frame_id: str
    image_width_px: int
    image_height_px: int
    face_landmarker_options: FaceLandmarkerOptionsRecord
    selection: FaceSelection


@dataclass(frozen=True)
class _DetectionRegion:
    name: str
    x_min_px: int
    y_min_px: int
    x_max_px: int
    y_max_px: int

    @property
    def width_px(self) -> int:
        return self.x_max_px - self.x_min_px

    @property
    def height_px(self) -> int:
        return self.y_max_px - self.y_min_px


@dataclass(frozen=True)
class _RegionSelection:
    region: _DetectionRegion
    selection: FaceSelection


class FaceObserver(Protocol):
    def observe(
        self,
        rgb_frame: npt.NDArray[np.uint8],
        *,
        frame_id: str | None = None,
    ) -> FaceObservation:
        """Observe face candidates in one RGB frame."""


class MediaPipeFaceObserver:
    def __init__(
        self,
        *,
        model_asset_path: Path | str,
        calibration: CalibrationRecord,
    ) -> None:
        if calibration.face_landmarker_running_mode != MEDIAPIPE_IMAGE_RUNNING_MODE:
            raise ValueError(
                "MediaPipeFaceObserver requires IMAGE running mode, got "
                f"{calibration.face_landmarker_running_mode!r}"
            )

        self._model_asset_path = Path(model_asset_path)
        self._calibration = calibration
        self.face_landmarker_options = FaceLandmarkerOptionsRecord(
            running_mode=MEDIAPIPE_IMAGE_RUNNING_MODE,
            num_faces=calibration.max_face_candidates,
            min_face_detection_confidence=calibration.candidate_face_score_min,
            min_face_presence_confidence=calibration.usable_face_score_min,
            min_tracking_confidence_source="ignored_for_image_mode",
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._mediapipe: Any | None = None
        self._landmarker: Any | None = None

    def observe(
        self,
        rgb_frame: npt.NDArray[np.uint8],
        *,
        frame_id: str | None = None,
    ) -> FaceObservation:
        frame = _validate_rgb_frame(rgb_frame)
        image_height_px, image_width_px, _channels = frame.shape
        mp = self._mediapipe_module()
        observation_frame_id = frame_id or DEFAULT_FRAME_ID
        full_frame_selection: FaceSelection | None = None
        region_selections: list[_RegionSelection] = []

        for region in _detection_regions(
            image_width_px=int(image_width_px), image_height_px=int(image_height_px)
        ):
            region_frame = np.ascontiguousarray(
                frame[
                    region.y_min_px : region.y_max_px,
                    region.x_min_px : region.x_max_px,
                ]
            )
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=region_frame)
            result = self._landmarker_instance().detect(mp_image)
            candidates = _candidates_from_mediapipe_result(
                result,
                frame_id=observation_frame_id,
                image_width_px=int(image_width_px),
                image_height_px=int(image_height_px),
                detection_region=region,
                max_face_candidates=self._calibration.max_face_candidates,
            )
            selection = select_primary_face(candidates, self._calibration)
            region_selections.append(
                _RegionSelection(region=region, selection=selection)
            )
            if region.name == DETECTION_REGION_FULL_FRAME:
                full_frame_selection = selection
                if selection.present and not _full_frame_needs_region_refinement(
                    selection, image_height_px=int(image_height_px)
                ):
                    return FaceObservation(
                        frame_id=observation_frame_id,
                        image_width_px=int(image_width_px),
                        image_height_px=int(image_height_px),
                        face_landmarker_options=self.face_landmarker_options,
                        selection=selection,
                    )
                continue
            if (
                full_frame_selection is not None
                and not full_frame_selection.present
                and selection.present
            ):
                return FaceObservation(
                    frame_id=observation_frame_id,
                    image_width_px=int(image_width_px),
                    image_height_px=int(image_height_px),
                    face_landmarker_options=self.face_landmarker_options,
                    selection=selection,
                )

        if full_frame_selection is None:
            raise AssertionError("full-frame face detection was not attempted")

        selection = _select_region_refined_face(region_selections, full_frame_selection)
        return FaceObservation(
            frame_id=observation_frame_id,
            image_width_px=int(image_width_px),
            image_height_px=int(image_height_px),
            face_landmarker_options=self.face_landmarker_options,
            selection=selection,
        )

    def close(self) -> None:
        if self._landmarker is not None and hasattr(self._landmarker, "close"):
            self._landmarker.close()
        self._landmarker = None

    def __enter__(self) -> MediaPipeFaceObserver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _mediapipe_module(self) -> Any:
        if self._mediapipe is None:
            self._mediapipe = _import_mediapipe()
        return self._mediapipe

    def _landmarker_instance(self) -> Any:
        if self._landmarker is None:
            mp = self._mediapipe_module()
            base_options = mp.tasks.BaseOptions(
                model_asset_path=str(self._model_asset_path)
            )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=self.face_landmarker_options.num_faces,
                min_face_detection_confidence=(
                    self.face_landmarker_options.min_face_detection_confidence
                ),
                min_face_presence_confidence=(
                    self.face_landmarker_options.min_face_presence_confidence
                ),
                output_face_blendshapes=(
                    self.face_landmarker_options.output_face_blendshapes
                ),
                output_facial_transformation_matrixes=(
                    self.face_landmarker_options.output_facial_transformation_matrixes
                ),
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        return self._landmarker


def select_primary_face(
    candidates: Sequence[FaceCandidate],
    calibration: CalibrationRecord,
) -> FaceSelection:
    candidate_records = tuple(candidates)
    errors = _initial_selection_errors(candidate_records)

    valid_candidates = tuple(
        candidate for candidate in candidate_records if candidate.has_valid_landmarks
    )
    if not valid_candidates:
        return _face_not_found_selection(
            candidates=_score_candidates_by_area(candidate_records),
            errors=errors,
            selection_reason="no_valid_landmarks",
        )

    if _all_valid_candidates_have_model_scores(valid_candidates):
        scored_candidates = _score_candidates_by_model_score(candidate_records)
        candidates_above_threshold = tuple(
            candidate
            for candidate in scored_candidates
            if candidate.has_valid_landmarks
            and candidate.candidate_score is not None
            and candidate.candidate_score >= calibration.candidate_face_score_min
        )
        if not candidates_above_threshold:
            return _face_not_found_selection(
                candidates=scored_candidates,
                errors=errors,
                selection_reason="no_candidate_passed_score_threshold",
            )

        selected = _highest_selection_score(candidates_above_threshold)
        return FaceSelection(
            present=True,
            primary_candidate_id=selected.candidate_id,
            selection_reason=_selection_reason(scored_candidates),
            selection_score_source=MODEL_SCORE_SELECTION_SOURCE,
            reason_invalid=None,
            candidates=scored_candidates,
            errors=tuple(errors),
        )

    scored_candidates = _score_candidates_by_area(candidate_records)
    valid_scored_candidates = tuple(
        candidate for candidate in scored_candidates if candidate.has_valid_landmarks
    )
    selected = _highest_selection_score(valid_scored_candidates)
    return FaceSelection(
        present=True,
        primary_candidate_id=selected.candidate_id,
        selection_reason=_selection_reason(scored_candidates),
        selection_score_source=AREA_ONLY_SELECTION_SOURCE,
        reason_invalid=None,
        candidates=scored_candidates,
        errors=tuple(errors),
    )


def _initial_selection_errors(
    candidates: Sequence[FaceCandidate],
) -> list[ErrorRecord]:
    if len(candidates) <= 1:
        return []

    return [
        ErrorRecord(
            code=ErrorCode.MULTIPLE_FACE_CANDIDATES,
            message="Multiple face candidates observed in frame.",
        )
    ]


def _face_not_found_selection(
    *,
    candidates: tuple[FaceCandidate, ...],
    errors: list[ErrorRecord],
    selection_reason: str,
) -> FaceSelection:
    return FaceSelection(
        present=False,
        primary_candidate_id=None,
        selection_reason=selection_reason,
        selection_score_source=_shared_selection_score_source(candidates),
        reason_invalid=ErrorCode.FACE_NOT_FOUND,
        candidates=candidates,
        errors=(
            *errors,
            ErrorRecord(
                code=ErrorCode.FACE_NOT_FOUND,
                message="No valid face candidate was selected.",
            ),
        ),
    )


def _score_candidates_by_area(
    candidates: Sequence[FaceCandidate],
) -> tuple[FaceCandidate, ...]:
    return tuple(
        replace(
            candidate,
            selection_score=_finite_selection_score(candidate.area_fraction),
            selection_score_source=AREA_ONLY_SELECTION_SOURCE,
        )
        for candidate in candidates
    )


def _score_candidates_by_model_score(
    candidates: Sequence[FaceCandidate],
) -> tuple[FaceCandidate, ...]:
    return tuple(
        replace(
            candidate,
            selection_score=_candidate_model_selection_score(candidate),
            selection_score_source=(
                MODEL_SCORE_SELECTION_SOURCE
                if candidate.candidate_score is not None
                else None
            ),
        )
        for candidate in candidates
    )


def _candidate_model_selection_score(candidate: FaceCandidate) -> float | None:
    if candidate.candidate_score is None:
        return None
    return _finite_selection_score(candidate.candidate_score * candidate.area_fraction)


def _finite_selection_score(value: float) -> float:
    if not isfinite(value):
        raise ValueError("selection score must be finite")
    return value


def _all_valid_candidates_have_model_scores(
    candidates: Sequence[FaceCandidate],
) -> bool:
    return all(candidate.candidate_score is not None for candidate in candidates)


def _highest_selection_score(candidates: Sequence[FaceCandidate]) -> FaceCandidate:
    return sorted(
        candidates,
        key=lambda candidate: (
            -(
                candidate.selection_score
                if candidate.selection_score is not None
                else 0.0
            ),
            candidate.candidate_id,
        ),
    )[0]


def _selection_reason(candidates: Sequence[FaceCandidate]) -> str:
    if len(candidates) == 1:
        return "single_candidate"
    return "highest_selection_score"


def _shared_selection_score_source(
    candidates: Sequence[FaceCandidate],
) -> str | None:
    sources = {
        candidate.selection_score_source
        for candidate in candidates
        if candidate.selection_score_source is not None
    }
    if len(sources) == 1:
        return next(iter(sources))
    return None


def _full_frame_needs_region_refinement(
    selection: FaceSelection, *, image_height_px: int
) -> bool:
    if _selection_has_multiple_candidates(selection):
        return True

    selected = _primary_candidate(selection)
    if selected is None:
        return False

    bbox = selected.bounding_box_image_px
    if bbox.y_min >= image_height_px * LOW_FULL_FRAME_FACE_TOP_FRACTION:
        return True

    return _bbox_height(bbox) < _bbox_width(bbox)


def _select_region_refined_face(
    region_selections: Sequence[_RegionSelection],
    full_frame_selection: FaceSelection,
) -> FaceSelection:
    if not full_frame_selection.present:
        return full_frame_selection

    scored_refinements = [
        (score, region_selection.selection)
        for region_selection in region_selections
        if region_selection.region.name != DETECTION_REGION_FULL_FRAME
        for score in (_region_refinement_score(region_selection, full_frame_selection),)
        if score is not None
    ]
    if scored_refinements:
        return max(
            scored_refinements,
            key=lambda item: (item[0], _primary_selection_area(item[1])),
        )[1]

    return full_frame_selection


def _region_refinement_score(
    region_selection: _RegionSelection, full_frame_selection: FaceSelection
) -> float | None:
    fallback = _primary_candidate(region_selection.selection)
    full_primary = _primary_candidate(full_frame_selection)
    if fallback is None or full_primary is None:
        return None
    if _candidate_is_near_region_seam(fallback, region_selection.region):
        return None

    max_iou = _max_iou_with_full_frame_candidates(fallback, full_frame_selection)
    if max_iou < REGION_REFINEMENT_MIN_IOU:
        return None

    area = _bbox_area(fallback.bounding_box_image_px)
    if _selection_has_multiple_candidates(full_frame_selection):
        return area * max_iou
    if _bbox_area(fallback.bounding_box_image_px) > _bbox_area(
        full_primary.bounding_box_image_px
    ):
        return area * max_iou

    top_shift_px = (
        full_primary.bounding_box_image_px.y_min - fallback.bounding_box_image_px.y_min
    )
    top_shift_threshold_px = max(
        REGION_REFINEMENT_TOP_SHIFT_MIN_PX,
        _bbox_height(full_primary.bounding_box_image_px)
        * REGION_REFINEMENT_TOP_SHIFT_FRACTION,
    )
    if top_shift_px < top_shift_threshold_px:
        return None
    return area * max_iou


def _primary_selection_area(selection: FaceSelection) -> float:
    candidate = _primary_candidate(selection)
    if candidate is None:
        return 0.0
    return _bbox_area(candidate.bounding_box_image_px)


def _selection_has_multiple_candidates(selection: FaceSelection) -> bool:
    if len(selection.candidates) > 1:
        return True
    return ErrorCode.MULTIPLE_FACE_CANDIDATES in {
        error.code for error in selection.errors
    }


def _primary_candidate(selection: FaceSelection) -> FaceCandidate | None:
    if selection.primary_candidate_id is None:
        return None
    for candidate in selection.candidates:
        if candidate.candidate_id == selection.primary_candidate_id:
            return candidate
    return None


def _candidate_is_near_region_seam(
    candidate: FaceCandidate, region: _DetectionRegion
) -> bool:
    seam_margin_px = max(
        REGION_SEAM_MARGIN_MIN_PX,
        region.width_px * REGION_SEAM_MARGIN_FRACTION,
    )
    bbox = candidate.bounding_box_image_px
    if region.name == DETECTION_REGION_LEFT_HALF:
        return bbox.x_max >= region.x_max_px - seam_margin_px
    if region.name == DETECTION_REGION_RIGHT_HALF:
        return bbox.x_min <= region.x_min_px + seam_margin_px
    return False


def _max_iou_with_full_frame_candidates(
    fallback: FaceCandidate, full_frame_selection: FaceSelection
) -> float:
    return max(
        (
            _bbox_iou(fallback.bounding_box_image_px, candidate.bounding_box_image_px)
            for candidate in full_frame_selection.candidates
        ),
        default=0.0,
    )


def _bbox_iou(left: BBox, right: BBox) -> float:
    intersection_x_min = max(left.x_min, right.x_min)
    intersection_y_min = max(left.y_min, right.y_min)
    intersection_x_max = min(left.x_max, right.x_max)
    intersection_y_max = min(left.y_max, right.y_max)
    intersection_width = max(0.0, intersection_x_max - intersection_x_min)
    intersection_height = max(0.0, intersection_y_max - intersection_y_min)
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0.0:
        return 0.0

    union_area = _bbox_area(left) + _bbox_area(right) - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def _bbox_area(bbox: BBox) -> float:
    return _bbox_width(bbox) * _bbox_height(bbox)


def _bbox_width(bbox: BBox) -> float:
    return bbox.x_max - bbox.x_min


def _bbox_height(bbox: BBox) -> float:
    return bbox.y_max - bbox.y_min


def _import_mediapipe() -> Any:
    return importlib.import_module("mediapipe")


def _validate_rgb_frame(
    rgb_frame: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError("rgb_frame must have shape (height, width, 3)")
    if rgb_frame.dtype != np.uint8:
        raise ValueError("rgb_frame must have dtype uint8")
    return np.ascontiguousarray(rgb_frame)


def _detection_regions(
    *, image_width_px: int, image_height_px: int
) -> tuple[_DetectionRegion, ...]:
    midpoint_x = image_width_px // 2
    return (
        _DetectionRegion(
            name=DETECTION_REGION_FULL_FRAME,
            x_min_px=0,
            y_min_px=0,
            x_max_px=image_width_px,
            y_max_px=image_height_px,
        ),
        _DetectionRegion(
            name=DETECTION_REGION_LEFT_HALF,
            x_min_px=0,
            y_min_px=0,
            x_max_px=midpoint_x,
            y_max_px=image_height_px,
        ),
        _DetectionRegion(
            name=DETECTION_REGION_RIGHT_HALF,
            x_min_px=midpoint_x,
            y_min_px=0,
            x_max_px=image_width_px,
            y_max_px=image_height_px,
        ),
    )


def _candidates_from_mediapipe_result(
    result: Any,
    *,
    frame_id: str,
    image_width_px: int,
    image_height_px: int,
    detection_region: _DetectionRegion,
    max_face_candidates: int,
) -> tuple[FaceCandidate, ...]:
    face_landmarks = tuple(getattr(result, "face_landmarks", ()) or ())
    face_blendshapes = tuple(getattr(result, "face_blendshapes", ()) or ())
    facial_transformation_matrixes = tuple(
        getattr(result, "facial_transformation_matrixes", ()) or ()
    )

    candidates: list[FaceCandidate] = []
    for index, landmarks in enumerate(face_landmarks[:max_face_candidates]):
        normalized_landmarks = _normalized_points(landmarks)
        if not normalized_landmarks:
            continue

        region_pixel_landmarks = _pixel_points(
            normalized_landmarks,
            image_width_px=detection_region.width_px,
            image_height_px=detection_region.height_px,
        )
        pixel_landmarks = _translate_pixel_points(
            region_pixel_landmarks,
            offset_x_px=detection_region.x_min_px,
            offset_y_px=detection_region.y_min_px,
        )
        source_normalized_landmarks = _normalize_pixel_points(
            pixel_landmarks,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )
        candidates.append(
            FaceCandidate(
                candidate_id=f"face_{index}",
                frame_id=frame_id,
                image_width_px=image_width_px,
                image_height_px=image_height_px,
                candidate_score=None,
                score_source=MEDIAPIPE_SCORE_SOURCE_UNAVAILABLE,
                bounding_box_image_px=_bbox_from_points(pixel_landmarks),
                bounding_box_image_norm=_bbox_from_points(source_normalized_landmarks),
                landmarks_image_px=pixel_landmarks,
                landmarks_image_norm=source_normalized_landmarks,
                blendshapes=_blendshapes_for_candidate(face_blendshapes, index),
                facial_transformation_matrix=_matrix_for_candidate(
                    facial_transformation_matrixes, index
                ),
            )
        )

    return tuple(candidates)


def _normalized_points(landmarks: Sequence[Any]) -> tuple[Point2D, ...]:
    return tuple(
        Point2D(
            space=CoordinateSpace.NORMALIZED,
            x=float(landmark.x),
            y=float(landmark.y),
        )
        for landmark in landmarks
    )


def _pixel_points(
    normalized_landmarks: Sequence[Point2D],
    *,
    image_width_px: int,
    image_height_px: int,
) -> tuple[Point2D, ...]:
    return tuple(
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=point.x * image_width_px,
            y=point.y * image_height_px,
        )
        for point in normalized_landmarks
    )


def _translate_pixel_points(
    pixel_landmarks: Sequence[Point2D],
    *,
    offset_x_px: int,
    offset_y_px: int,
) -> tuple[Point2D, ...]:
    return tuple(
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=point.x + offset_x_px,
            y=point.y + offset_y_px,
        )
        for point in pixel_landmarks
    )


def _normalize_pixel_points(
    pixel_landmarks: Sequence[Point2D],
    *,
    image_width_px: int,
    image_height_px: int,
) -> tuple[Point2D, ...]:
    return tuple(
        Point2D(
            space=CoordinateSpace.NORMALIZED,
            x=point.x / image_width_px,
            y=point.y / image_height_px,
        )
        for point in pixel_landmarks
    )


def _bbox_from_points(points: Sequence[Point2D]) -> BBox:
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    return BBox(
        space=points[0].space,
        x_min=min(x_values),
        y_min=min(y_values),
        x_max=max(x_values),
        y_max=max(y_values),
    )


def _blendshapes_for_candidate(
    face_blendshapes: Sequence[Sequence[Any]],
    candidate_index: int,
) -> tuple[BlendshapeScore, ...]:
    if candidate_index >= len(face_blendshapes):
        return ()

    return tuple(
        BlendshapeScore(
            category_name=str(category.category_name),
            score=float(category.score),
        )
        for category in face_blendshapes[candidate_index]
    )


def _matrix_for_candidate(
    facial_transformation_matrixes: Sequence[Any],
    candidate_index: int,
) -> tuple[tuple[float, ...], ...] | None:
    if candidate_index >= len(facial_transformation_matrixes):
        return None

    matrix = np.asarray(facial_transformation_matrixes[candidate_index], dtype=float)
    if matrix.ndim != 2:
        return None
    return tuple(tuple(float(value) for value in row) for row in matrix)
