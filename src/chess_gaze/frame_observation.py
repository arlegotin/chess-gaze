from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.eye_observation import EyeObservation, EyePairObservation, observe_eyes
from chess_gaze.face_observation import FaceCandidate, FaceObservation, FaceObserver
from chess_gaze.frame_records import (
    CalibrationRecord,
    ErrorRecord,
    EyeRecord,
    FaceRecord,
    FrameRecord,
    GazeAngles,
    HeadPoseRecord,
)
from chess_gaze.gaze_observation import (
    FaceModelGaze,
    NormalizedFaceCrop,
    normalize_face_crop,
)
from chess_gaze.head_pose import HeadPoseObservation, ImageSize, estimate_head_pose

FRAME_WARNING_ERROR_CODES = frozenset({ErrorCode.MULTIPLE_FACE_CANDIDATES})


class EyeObserver(Protocol):
    def __call__(
        self,
        face: FaceCandidate,
        rgb_frame: npt.NDArray[np.uint8],
        run_layout: RunLayout,
        frame_id: str,
        *,
        save_crop_images: bool = False,
    ) -> EyePairObservation: ...


HeadPoseEstimator = Callable[
    [FaceCandidate, CalibrationRecord, ImageSize], HeadPoseObservation
]


class FaceCropNormalizer(Protocol):
    def __call__(
        self,
        rgb_frame: npt.NDArray[np.uint8],
        bbox: Any,
        *,
        input_size_px: int,
    ) -> NormalizedFaceCrop: ...


class FaceGazeModel(Protocol):
    def predict(self, normalized_batch: Any) -> FaceModelGaze: ...

    def predict_batch(self, normalized_batch: Any) -> tuple[FaceModelGaze, ...]: ...


class ModelInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class _FrameEvidence:
    frame: Any
    face_observation: FaceObservation | None
    errors: list[ErrorRecord]
    face_record: FaceRecord
    selected_face: FaceCandidate | None
    left_eye: EyeRecord
    right_eye: EyeRecord
    head_pose_record: HeadPoseRecord
    normalized_face_crop: NormalizedFaceCrop | None


@dataclass(frozen=True)
class ModelBackedFrameObserver:
    face_observer: FaceObserver
    gaze_model: FaceGazeModel
    calibration: CalibrationRecord
    run_layout: RunLayout
    eye_observer: EyeObserver = observe_eyes
    head_pose_estimator: HeadPoseEstimator = estimate_head_pose
    face_crop_normalizer: FaceCropNormalizer = normalize_face_crop
    save_crop_images: bool = False

    def __call__(self, frame: Any) -> FrameRecord:
        return self.observe_batch([frame])[0]

    def observe_batch(self, frames: Sequence[Any]) -> list[FrameRecord]:
        evidence_items = [self._collect_frame_evidence(frame) for frame in frames]
        crop_items = [
            (index, evidence.normalized_face_crop.tensor)
            for index, evidence in enumerate(evidence_items)
            if evidence.selected_face is not None
            and evidence.normalized_face_crop is not None
        ]
        appearance_by_index: dict[int, FaceModelGaze] = {}
        if crop_items:
            batch = torch.cat([tensor for _index, tensor in crop_items], dim=0)
            try:
                gazes = self.gaze_model.predict_batch(batch)
                if len(gazes) != len(crop_items):
                    raise ValueError(
                        "Appearance gaze model returned a different number of rows"
                    )
            except Exception as exc:
                raise ModelInferenceError(
                    f"UniGaze batch inference failed: {exc}"
                ) from exc
            for (index, _tensor), gaze in zip(crop_items, gazes, strict=True):
                appearance_by_index[index] = gaze

        records: list[FrameRecord] = []
        for index, evidence in enumerate(evidence_items):
            if evidence.selected_face is None:
                if evidence.face_observation is None:
                    raise AssertionError(
                        "missing face evidence requires face observation"
                    )
                records.append(
                    _missing_face_record(
                        evidence.frame,
                        evidence.face_observation,
                        evidence.errors,
                    )
                )
                continue

            appearance_gaze = appearance_by_index.get(index, _invalid_face_model_gaze())
            if (
                index in appearance_by_index
                and not appearance_gaze.valid
                and appearance_gaze.reason_invalid is not None
                and not any(
                    error.code is appearance_gaze.reason_invalid
                    for error in evidence.errors
                )
            ):
                _append_error_once(
                    evidence.errors,
                    ErrorRecord(
                        code=appearance_gaze.reason_invalid,
                        message=(
                            "Appearance gaze model failed: non-finite UniGaze output."
                        ),
                    ),
                )
            records.append(self._record_from_evidence(evidence, appearance_gaze))
        return records

    def _collect_frame_evidence(self, frame: Any) -> _FrameEvidence:
        face_observation = self.face_observer.observe(
            frame.rgb, frame_id=frame.frame_id
        )
        errors = list(face_observation.selection.errors)
        selected_face = _selected_face(face_observation)
        if selected_face is None:
            _append_error_once(
                errors,
                ErrorRecord(
                    code=ErrorCode.FACE_NOT_FOUND,
                    message="No selected face candidate is available for frame.",
                ),
            )
            return _FrameEvidence(
                frame=frame,
                face_observation=face_observation,
                errors=errors,
                face_record=FaceRecord(
                    present=False,
                    bounding_box=None,
                    landmarks=None,
                    reason_invalid=(
                        face_observation.selection.reason_invalid
                        or ErrorCode.FACE_NOT_FOUND
                    ),
                ),
                selected_face=None,
                left_eye=_missing_eye_record(ErrorCode.LEFT_EYE_NOT_FOUND),
                right_eye=_missing_eye_record(ErrorCode.RIGHT_EYE_NOT_FOUND),
                head_pose_record=HeadPoseRecord(
                    valid=False,
                    yaw_radians=None,
                    pitch_radians=None,
                    roll_radians=None,
                    reason_invalid=ErrorCode.HEAD_POSE_INVALID,
                ),
                normalized_face_crop=None,
            )

        face_record = FaceRecord(
            present=True,
            bounding_box=selected_face.bounding_box_image_px,
            landmarks=list(selected_face.landmarks_image_px),
            reason_invalid=None,
        )
        eye_pair = self.eye_observer(
            selected_face,
            frame.rgb,
            self.run_layout,
            frame.frame_id,
            save_crop_images=self.save_crop_images,
        )
        left_eye = _eye_record(eye_pair.left, ErrorCode.LEFT_EYE_NOT_FOUND)
        right_eye = _eye_record(eye_pair.right, ErrorCode.RIGHT_EYE_NOT_FOUND)
        _append_eye_errors(errors, eye_pair.left, ErrorCode.LEFT_EYE_NOT_FOUND)
        _append_eye_errors(errors, eye_pair.right, ErrorCode.RIGHT_EYE_NOT_FOUND)

        head_pose = self.head_pose_estimator(
            selected_face,
            self.calibration,
            ImageSize(
                width_px=face_observation.image_width_px,
                height_px=face_observation.image_height_px,
            ),
        )
        head_pose_record = _head_pose_record(head_pose)
        for error in head_pose.errors:
            _append_error_once(errors, error)
        if not head_pose.valid:
            _append_error_once(
                errors,
                ErrorRecord(
                    code=ErrorCode.HEAD_POSE_INVALID,
                    message="Head pose observation is invalid.",
                ),
            )

        normalized_face_crop = self._normalized_face_crop(
            frame.rgb,
            selected_face,
            errors,
        )

        return _FrameEvidence(
            frame=frame,
            face_observation=face_observation,
            errors=errors,
            face_record=face_record,
            selected_face=selected_face,
            left_eye=left_eye,
            right_eye=right_eye,
            head_pose_record=head_pose_record,
            normalized_face_crop=normalized_face_crop,
        )

    def _record_from_evidence(
        self,
        evidence: _FrameEvidence,
        appearance_gaze: FaceModelGaze,
    ) -> FrameRecord:
        appearance_gaze_record = _face_model_gaze_record(appearance_gaze)
        return FrameRecord(
            frame_id=evidence.frame.frame_id,
            frame_index=evidence.frame.frame_index,
            status=_frame_status(
                errors=evidence.errors,
                face=evidence.face_record,
                left_eye=evidence.left_eye,
                right_eye=evidence.right_eye,
                head_pose=evidence.head_pose_record,
                appearance_gaze=appearance_gaze,
            ),
            timestamp_seconds=evidence.frame.timestamp_seconds,
            face=evidence.face_record,
            left_eye=evidence.left_eye,
            right_eye=evidence.right_eye,
            head_pose=evidence.head_pose_record,
            geometric_gaze=_invalid_gaze(ErrorCode.GAZE_MODEL_FAILED),
            appearance_gaze=appearance_gaze_record,
            recommended_gaze=appearance_gaze_record,
            errors=evidence.errors,
        )

    def close(self) -> None:
        close = getattr(self.face_observer, "close", None)
        if callable(close):
            close()

    def _normalized_face_crop(
        self,
        rgb_frame: npt.NDArray[np.uint8],
        selected_face: FaceCandidate,
        errors: list[ErrorRecord],
    ) -> NormalizedFaceCrop | None:
        try:
            return self.face_crop_normalizer(
                rgb_frame,
                selected_face.bounding_box_image_px,
                input_size_px=self.calibration.unigaze_input_size_px,
            )
        except Exception as exc:
            _append_error_once(
                errors,
                ErrorRecord(
                    code=ErrorCode.GAZE_MODEL_FAILED,
                    message=f"Appearance gaze model failed: {exc}",
                ),
            )
            return None


def _selected_face(observation: FaceObservation) -> FaceCandidate | None:
    selection = observation.selection
    if not selection.present or selection.primary_candidate_id is None:
        return None

    for candidate in selection.candidates:
        if candidate.candidate_id == selection.primary_candidate_id:
            return candidate
    return None


def _missing_face_record(
    frame: Any,
    observation: FaceObservation,
    errors: list[ErrorRecord],
) -> FrameRecord:
    reason = observation.selection.reason_invalid or ErrorCode.FACE_NOT_FOUND
    invalid_gaze = _invalid_gaze(reason)
    return FrameRecord(
        frame_id=frame.frame_id,
        frame_index=frame.frame_index,
        status=FrameStatus.ERROR,
        timestamp_seconds=frame.timestamp_seconds,
        face=FaceRecord(
            present=False,
            bounding_box=None,
            landmarks=None,
            reason_invalid=reason,
        ),
        left_eye=_missing_eye_record(ErrorCode.LEFT_EYE_NOT_FOUND),
        right_eye=_missing_eye_record(ErrorCode.RIGHT_EYE_NOT_FOUND),
        head_pose=HeadPoseRecord(
            valid=False,
            yaw_radians=None,
            pitch_radians=None,
            roll_radians=None,
            reason_invalid=ErrorCode.HEAD_POSE_INVALID,
        ),
        geometric_gaze=invalid_gaze,
        appearance_gaze=_invalid_gaze(ErrorCode.GAZE_MODEL_FAILED),
        recommended_gaze=invalid_gaze,
        errors=errors,
    )


def _eye_record(eye: EyeObservation, default_reason: ErrorCode) -> EyeRecord:
    present = (
        eye.present
        and eye.iris_center_image_px is not None
        and eye.bounding_box_image_px is not None
        and bool(eye.iris_landmarks_image_px)
    )
    if not present:
        return _missing_eye_record(eye.reason_missing or default_reason)

    return EyeRecord(
        present=True,
        bounding_box=eye.bounding_box_image_px,
        pupil_center=eye.iris_center_image_px,
        iris_landmarks=list(eye.iris_landmarks_image_px),
        reason_invalid=None,
    )


def _missing_eye_record(reason: ErrorCode) -> EyeRecord:
    return EyeRecord(
        present=False,
        bounding_box=None,
        pupil_center=None,
        iris_landmarks=None,
        reason_invalid=reason,
    )


def _append_eye_errors(
    errors: list[ErrorRecord],
    eye: EyeObservation,
    default_reason: ErrorCode,
) -> None:
    reason = eye.reason_missing
    if reason is None and not eye.present:
        reason = default_reason
    if reason is None:
        return

    _append_error_once(
        errors,
        ErrorRecord(
            code=reason,
            message=f"Eye observation is invalid: {reason.value}.",
        ),
    )


def _head_pose_record(head_pose: HeadPoseObservation) -> HeadPoseRecord:
    return HeadPoseRecord(
        valid=head_pose.valid,
        yaw_radians=head_pose.yaw_radians,
        pitch_radians=head_pose.pitch_radians,
        roll_radians=head_pose.roll_radians,
        reason_invalid=head_pose.reason_invalid,
    )


def _face_model_gaze_record(face_gaze: FaceModelGaze) -> GazeAngles:
    if (
        face_gaze.valid
        and face_gaze.pitch_radians is not None
        and face_gaze.yaw_radians is not None
    ):
        return GazeAngles(
            valid=True,
            yaw_radians=face_gaze.yaw_radians,
            pitch_radians=face_gaze.pitch_radians,
            reason_invalid=None,
        )
    return _invalid_gaze(face_gaze.reason_invalid or ErrorCode.GAZE_MODEL_FAILED)


def _invalid_face_model_gaze() -> FaceModelGaze:
    return FaceModelGaze(
        valid=False,
        method="unigaze_h14_joint",
        pitch_radians=None,
        yaw_radians=None,
        unit_vector=None,
        confidence=None,
        confidence_source="not_provided_by_unigaze",
        reason_invalid=ErrorCode.GAZE_MODEL_FAILED,
    )


def _invalid_gaze(reason: ErrorCode) -> GazeAngles:
    return GazeAngles(
        valid=False,
        yaw_radians=None,
        pitch_radians=None,
        reason_invalid=reason,
    )


def _frame_status(
    *,
    errors: list[ErrorRecord],
    face: FaceRecord,
    left_eye: EyeRecord,
    right_eye: EyeRecord,
    head_pose: HeadPoseRecord,
    appearance_gaze: FaceModelGaze,
) -> FrameStatus:
    if not (
        face.present
        and left_eye.present
        and right_eye.present
        and head_pose.valid
        and appearance_gaze.valid
    ):
        return FrameStatus.ERROR

    if errors:
        if _only_warning_errors(errors):
            return FrameStatus.WARNING
        return FrameStatus.ERROR

    return FrameStatus.OK


def _only_warning_errors(errors: list[ErrorRecord]) -> bool:
    return bool(errors) and (
        {error.code for error in errors} <= FRAME_WARNING_ERROR_CODES
    )


def _append_error_once(errors: list[ErrorRecord], error: ErrorRecord) -> None:
    key = (error.code, error.message)
    if key in {(item.code, item.message) for item in errors}:
        return
    errors.append(error)
