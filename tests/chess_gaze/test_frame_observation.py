from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.calibration import default_calibration
from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.eye_observation import EyeObservation, EyePairObservation
from chess_gaze.face_observation import (
    FaceCandidate,
    FaceLandmarkerOptionsRecord,
    FaceObservation,
    FaceSelection,
)
from chess_gaze.frame_observation import ModelBackedFrameObserver, ModelInferenceError
from chess_gaze.frame_records import ErrorRecord
from chess_gaze.gaze_observation import (
    CropTransformRecord,
    FaceModelGaze,
    NormalizedFaceCrop,
    pitch_yaw_to_unit_vector,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D, Transform2D
from chess_gaze.head_pose import HeadPoseObservation, ImageSize
from chess_gaze.pipeline import ObserverFrame


def _run_layout(tmp_path: Path) -> RunLayout:
    return RunLayout(
        run_dir=tmp_path,
        raw_frames_dir=tmp_path / "raw_frames",
        processed_frames_dir=tmp_path / "processed_frames",
        crops_dir=tmp_path / "crops",
        face_crops_dir=tmp_path / "crops" / "face",
        eyes_crops_dir=tmp_path / "crops" / "eyes",
        left_eye_crops_dir=tmp_path / "crops" / "eyes" / "left",
        right_eye_crops_dir=tmp_path / "crops" / "eyes" / "right",
        records_dir=tmp_path / "records",
    )


def _point(x: float, y: float) -> Point2D:
    return Point2D(space=CoordinateSpace.IMAGE_PX, x=x, y=y)


def _norm_point(x: float, y: float) -> Point2D:
    return Point2D(space=CoordinateSpace.NORMALIZED, x=x, y=y)


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> BBox:
    return BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def _candidate() -> FaceCandidate:
    return FaceCandidate(
        candidate_id="face_0",
        frame_id="f000000000",
        image_width_px=64,
        image_height_px=48,
        candidate_score=None,
        score_source="test",
        bounding_box_image_px=_box(10.0, 8.0, 54.0, 44.0),
        bounding_box_image_norm=BBox(
            space=CoordinateSpace.NORMALIZED,
            x_min=0.1,
            y_min=0.1,
            x_max=0.9,
            y_max=0.9,
        ),
        landmarks_image_px=(_point(20.0, 20.0), _point(44.0, 20.0)),
        landmarks_image_norm=(_norm_point(0.3, 0.4), _norm_point(0.7, 0.4)),
    )


class _FakeFaceObserver:
    def __init__(self, observation: FaceObservation) -> None:
        self.observation = observation
        self.observed_frame_ids: list[str | None] = []

    def observe(
        self, rgb_frame: np.ndarray, *, frame_id: str | None = None
    ) -> FaceObservation:
        del rgb_frame
        self.observed_frame_ids.append(frame_id)
        return self.observation


class _FakeGazeModel:
    def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze:
        return self.predict_batch(normalized_batch)[0]

    def predict_batch(
        self, normalized_batch: torch.Tensor
    ) -> tuple[FaceModelGaze, ...]:
        assert tuple(normalized_batch.shape[1:]) == (3, 224, 224)
        return tuple(
            FaceModelGaze(
                valid=True,
                method="fake_unigaze",
                pitch_radians=0.02 + index,
                yaw_radians=0.01 + index,
                unit_vector=pitch_yaw_to_unit_vector(
                    pitch_radians=0.02 + index,
                    yaw_radians=0.01 + index,
                ),
                confidence=None,
                confidence_source="not_provided_by_unigaze",
                reason_invalid=None,
            )
            for index in range(normalized_batch.shape[0])
        )


class _DisagreeingGazeModel:
    def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze:
        return self.predict_batch(normalized_batch)[0]

    def predict_batch(
        self, normalized_batch: torch.Tensor
    ) -> tuple[FaceModelGaze, ...]:
        assert tuple(normalized_batch.shape[1:]) == (3, 224, 224)
        return tuple(
            FaceModelGaze(
                valid=True,
                method="fake_unigaze",
                pitch_radians=1.0 + index,
                yaw_radians=1.0 + index,
                unit_vector=pitch_yaw_to_unit_vector(
                    pitch_radians=1.0 + index,
                    yaw_radians=1.0 + index,
                ),
                confidence=None,
                confidence_source="not_provided_by_unigaze",
                reason_invalid=None,
            )
            for index in range(normalized_batch.shape[0])
        )


class _OneInvalidRowGazeModel:
    def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze:
        return self.predict_batch(normalized_batch)[0]

    def predict_batch(
        self, normalized_batch: torch.Tensor
    ) -> tuple[FaceModelGaze, ...]:
        assert tuple(normalized_batch.shape[1:]) == (3, 224, 224)
        return (
            FaceModelGaze(
                valid=True,
                method="fake_unigaze",
                pitch_radians=0.02,
                yaw_radians=0.01,
                unit_vector=pitch_yaw_to_unit_vector(
                    pitch_radians=0.02,
                    yaw_radians=0.01,
                ),
                confidence=None,
                confidence_source="not_provided_by_unigaze",
                reason_invalid=None,
            ),
            FaceModelGaze(
                valid=False,
                method="fake_unigaze",
                pitch_radians=None,
                yaw_radians=None,
                unit_vector=None,
                confidence=None,
                confidence_source="not_provided_by_unigaze",
                reason_invalid=ErrorCode.GAZE_MODEL_FAILED,
            ),
        )


class _RaisingBatchGazeModel:
    def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze:
        return self.predict_batch(normalized_batch)[0]

    def predict_batch(
        self, normalized_batch: torch.Tensor
    ) -> tuple[FaceModelGaze, ...]:
        del normalized_batch
        raise ValueError(
            "UniGaze pred_gaze must have shape (batch, 2) matching input batch"
        )


def _face_observation(
    candidate: FaceCandidate, *, errors: tuple[ErrorRecord, ...] = ()
) -> FaceObservation:
    return FaceObservation(
        frame_id="f000000000",
        image_width_px=64,
        image_height_px=48,
        face_landmarker_options=FaceLandmarkerOptionsRecord(
            running_mode="IMAGE",
            num_faces=4,
            min_face_detection_confidence=0.25,
            min_face_presence_confidence=0.5,
            min_tracking_confidence_source="ignored_for_image_mode",
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        ),
        selection=FaceSelection(
            present=True,
            primary_candidate_id=candidate.candidate_id,
            selection_reason="single_candidate",
            selection_score_source="test",
            reason_invalid=None,
            candidates=(candidate,),
            errors=errors,
        ),
    )


def _missing_face_observation() -> FaceObservation:
    return FaceObservation(
        frame_id="f000000000",
        image_width_px=64,
        image_height_px=48,
        face_landmarker_options=FaceLandmarkerOptionsRecord(
            running_mode="IMAGE",
            num_faces=4,
            min_face_detection_confidence=0.25,
            min_face_presence_confidence=0.5,
            min_tracking_confidence_source="ignored_for_image_mode",
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        ),
        selection=FaceSelection(
            present=False,
            primary_candidate_id=None,
            selection_reason="no_valid_landmarks",
            selection_score_source=None,
            reason_invalid=ErrorCode.FACE_NOT_FOUND,
            candidates=(),
            errors=(),
        ),
    )


def _eye(
    *,
    center_x: float,
    center_y: float,
    offset: tuple[float, float],
) -> EyeObservation:
    return EyeObservation(
        present=True,
        confidence=1.0,
        confidence_source="test",
        reason_missing=None,
        eye_landmarks_image_px=(_point(center_x - 4.0, center_y),),
        eye_landmarks_image_norm=(_norm_point(0.4, 0.4),),
        iris_present=True,
        iris_landmarks_image_px=(
            _point(center_x - 1.0, center_y),
            _point(center_x + 1.0, center_y),
        ),
        iris_landmarks_image_norm=(_norm_point(0.4, 0.4), _norm_point(0.5, 0.4)),
        iris_center_image_px=_point(center_x, center_y),
        iris_center_image_norm=_norm_point(0.5, 0.5),
        iris_diameter_px=2.0,
        bounding_box_image_px=_box(
            center_x - 5.0, center_y - 3.0, center_x + 5.0, center_y + 3.0
        ),
        bounding_box_image_norm=BBox(
            space=CoordinateSpace.NORMALIZED,
            x_min=0.4,
            y_min=0.4,
            x_max=0.6,
            y_max=0.5,
        ),
        crop_bbox_image_px=_box(
            center_x - 6.0, center_y - 4.0, center_x + 6.0, center_y + 4.0
        ),
        eye_crop_path=Path("crops/eyes/test.png"),
        eye_crop_sha256="0" * 64,
        eye_crop_transform_to_image_px=None,
        normalized_iris_offset_xy=offset,
        eye_open_metric=0.6,
        occlusion="none",
    )


def _missing_eye(reason: ErrorCode) -> EyeObservation:
    return EyeObservation(
        present=False,
        confidence=0.0,
        confidence_source="test",
        reason_missing=reason,
        eye_landmarks_image_px=(),
        eye_landmarks_image_norm=(),
        iris_present=False,
        iris_landmarks_image_px=(),
        iris_landmarks_image_norm=(),
        iris_center_image_px=None,
        iris_center_image_norm=None,
        iris_diameter_px=None,
        bounding_box_image_px=None,
        bounding_box_image_norm=None,
        crop_bbox_image_px=None,
        eye_crop_path=None,
        eye_crop_sha256=None,
        eye_crop_transform_to_image_px=None,
        normalized_iris_offset_xy=None,
        eye_open_metric=None,
        occlusion="unknown",
    )


def _observe_eyes(
    face: FaceCandidate,
    rgb_frame: np.ndarray,
    run_layout: RunLayout,
    frame_id: str,
    *,
    save_crop_images: bool = False,
) -> EyePairObservation:
    del face, rgb_frame, run_layout
    assert save_crop_images is False
    return EyePairObservation(
        frame_id=frame_id,
        image_width_px=64,
        image_height_px=48,
        left=_eye(center_x=40.0, center_y=24.0, offset=(0.0, 0.0)),
        right=_eye(center_x=24.0, center_y=24.0, offset=(0.0, 0.0)),
    )


def _observe_eyes_missing_right(
    face: FaceCandidate,
    rgb_frame: np.ndarray,
    run_layout: RunLayout,
    frame_id: str,
    *,
    save_crop_images: bool = False,
) -> EyePairObservation:
    del face, rgb_frame, run_layout
    assert save_crop_images is False
    return EyePairObservation(
        frame_id=frame_id,
        image_width_px=64,
        image_height_px=48,
        left=_eye(center_x=40.0, center_y=24.0, offset=(0.0, 0.0)),
        right=_missing_eye(ErrorCode.RIGHT_EYE_NOT_FOUND),
    )


def _estimate_head_pose(
    face: FaceCandidate,
    calibration: object,
    image_size: ImageSize,
) -> HeadPoseObservation:
    del face, calibration
    assert image_size == ImageSize(width_px=64, height_px=48)
    return HeadPoseObservation(
        valid=True,
        method="fake_head_pose",
        reason_invalid=None,
        facial_transformation_matrix=None,
        pnp_method="fake",
        pnp_landmarks=(),
        pnp_point_count=8,
        pnp_min_point_count=6,
        canonical_points_source="test",
        camera_intrinsics_policy="test",
        metric_translation_allowed=False,
        reprojection_error_px=0.0,
        reprojection_error_max_px=8.0,
        reprojection_error_threshold_name="test",
        reprojection_error_threshold_source="test",
        rotation_matrix=None,
        quaternion_wxyz=None,
        yaw_radians=0.01,
        pitch_radians=0.02,
        roll_radians=0.03,
        translation_camera_3d_m=None,
        errors=(),
    )


def _normalize_face_crop(
    rgb_frame: np.ndarray,
    bbox: BBox,
    *,
    input_size_px: int,
) -> NormalizedFaceCrop:
    assert rgb_frame.shape == (48, 64, 3)
    assert bbox == _box(10.0, 8.0, 54.0, 44.0)
    assert input_size_px == 224
    return NormalizedFaceCrop(
        tensor=torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        transform=CropTransformRecord(
            source_bbox_image_px=bbox,
            output_size_px=input_size_px,
            image_px_from_crop_px=Transform2D(
                source_space=CoordinateSpace.IMAGE_PX,
                target_space=CoordinateSpace.IMAGE_PX,
                m00=1.0,
                m01=0.0,
                m02=0.0,
                m10=0.0,
                m11=1.0,
                m12=0.0,
            ),
        ),
    )


def _observer_frame() -> ObserverFrame:
    return ObserverFrame(
        frame_id="f000000000",
        frame_index=0,
        timestamp_seconds=0.0,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        pts=None,
        pts_seconds=None,
        duration_seconds=None,
    )


def test_model_backed_frame_observer_maps_model_outputs_to_frame_record(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)
    candidate = _candidate()
    face_observer = _FakeFaceObserver(_face_observation(candidate))
    observer = ModelBackedFrameObserver(
        face_observer=face_observer,
        gaze_model=_FakeGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    record = observer(_observer_frame())

    assert face_observer.observed_frame_ids == ["f000000000"]
    assert record.status is FrameStatus.OK
    assert record.face.present is True
    assert record.face.bounding_box == candidate.bounding_box_image_px
    assert record.left_eye.present is True
    assert record.left_eye.pupil_center == _point(40.0, 24.0)
    assert record.right_eye.present is True
    assert record.right_eye.pupil_center == _point(24.0, 24.0)
    assert record.head_pose.valid is True
    assert record.head_pose.yaw_radians == 0.01
    assert record.geometric_gaze.valid is False
    assert record.geometric_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
    assert record.appearance_gaze.valid is True
    assert record.recommended_gaze.valid is True
    assert record.recommended_gaze.yaw_radians == record.appearance_gaze.yaw_radians
    assert record.recommended_gaze.pitch_radians == record.appearance_gaze.pitch_radians
    assert record.errors == []


def test_model_backed_frame_observer_preserves_missing_right_eye_reason(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_face_observation(candidate)),
        gaze_model=_FakeGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=_observe_eyes_missing_right,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    record = observer(_observer_frame())

    assert record.status is FrameStatus.ERROR
    assert record.left_eye.present is True
    assert record.left_eye.pupil_center == _point(40.0, 24.0)
    assert record.right_eye.present is False
    assert record.right_eye.reason_invalid is ErrorCode.RIGHT_EYE_NOT_FOUND
    assert record.geometric_gaze.valid is False
    assert record.geometric_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
    assert ErrorCode.RIGHT_EYE_NOT_FOUND in {error.code for error in record.errors}


def test_model_backed_frame_observer_records_missing_face_without_later_models(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)

    def fail_eye_observer(*args: object, **kwargs: object) -> EyePairObservation:
        raise AssertionError("eye observer must not run without a selected face")

    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_missing_face_observation()),
        gaze_model=_FakeGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=fail_eye_observer,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    record = observer(_observer_frame())

    assert record.status is FrameStatus.ERROR
    assert record.face.present is False
    assert record.face.reason_invalid is ErrorCode.FACE_NOT_FOUND
    assert record.left_eye.present is False
    assert record.right_eye.present is False
    assert record.geometric_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
    assert record.appearance_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
    assert record.recommended_gaze.valid is False
    assert record.recommended_gaze == record.appearance_gaze
    assert ErrorCode.FACE_NOT_FOUND in {error.code for error in record.errors}


def test_model_backed_frame_observer_uses_unigaze_without_disagreement_status(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_face_observation(candidate)),
        gaze_model=_DisagreeingGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    record = observer(_observer_frame())

    assert record.face.present is True
    assert record.left_eye.present is True
    assert record.right_eye.present is True
    assert record.head_pose.valid is True
    assert record.appearance_gaze.valid is True
    assert record.recommended_gaze == record.appearance_gaze
    assert record.status is FrameStatus.OK
    assert record.errors == []


def test_model_backed_frame_observer_marks_multiple_face_candidates_as_warning(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(
            _face_observation(
                candidate,
                errors=(
                    ErrorRecord(
                        code=ErrorCode.MULTIPLE_FACE_CANDIDATES,
                        message="Multiple face candidates were detected.",
                    ),
                ),
            )
        ),
        gaze_model=_FakeGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    record = observer(_observer_frame())

    assert record.face.present is True
    assert record.left_eye.present is True
    assert record.right_eye.present is True
    assert record.head_pose.valid is True
    assert record.appearance_gaze.valid is True
    assert record.recommended_gaze.valid is True
    assert record.status is FrameStatus.WARNING
    assert [error.code for error in record.errors] == [
        ErrorCode.MULTIPLE_FACE_CANDIDATES
    ]


def test_model_backed_observer_marks_multiple_candidates_without_gaze_warning(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(
            _face_observation(
                candidate,
                errors=(
                    ErrorRecord(
                        code=ErrorCode.MULTIPLE_FACE_CANDIDATES,
                        message="Multiple face candidates were detected.",
                    ),
                ),
            )
        ),
        gaze_model=_DisagreeingGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    record = observer(_observer_frame())

    assert record.face.present is True
    assert record.left_eye.present is True
    assert record.right_eye.present is True
    assert record.head_pose.valid is True
    assert record.appearance_gaze.valid is True
    assert record.recommended_gaze == record.appearance_gaze
    assert record.status is FrameStatus.WARNING
    assert [error.code for error in record.errors] == [
        ErrorCode.MULTIPLE_FACE_CANDIDATES
    ]


def test_model_backed_frame_observer_batch_maps_model_rows_to_frames(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_face_observation(candidate)),
        gaze_model=_FakeGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )
    frames = [
        _observer_frame(),
        ObserverFrame(
            frame_id="f000000001",
            frame_index=1,
            timestamp_seconds=1.0,
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            pts=None,
            pts_seconds=None,
            duration_seconds=None,
        ),
    ]

    records = observer.observe_batch(frames)

    assert [record.frame_id for record in records] == ["f000000000", "f000000001"]
    assert records[0].appearance_gaze.pitch_radians == 0.02
    assert records[1].appearance_gaze.pitch_radians == 1.02
    assert records[0].recommended_gaze == records[0].appearance_gaze
    assert records[1].recommended_gaze == records[1].appearance_gaze
    assert records[1].recommended_gaze.valid is True


def test_model_backed_frame_observer_batch_preserves_missing_face_record(
    tmp_path: Path,
) -> None:
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_missing_face_observation()),
        gaze_model=_FakeGazeModel(),
        calibration=default_calibration(),
        run_layout=_run_layout(tmp_path),
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    [record] = observer.observe_batch([_observer_frame()])

    assert record.face.present is False
    assert record.appearance_gaze.valid is False
    assert record.appearance_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
    assert ErrorCode.FACE_NOT_FOUND in {error.code for error in record.errors}


def test_model_backed_frame_observer_batch_marks_only_invalid_model_row(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_face_observation(candidate)),
        gaze_model=_OneInvalidRowGazeModel(),
        calibration=default_calibration(),
        run_layout=_run_layout(tmp_path),
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )
    frames = [
        _observer_frame(),
        ObserverFrame(
            frame_id="f000000001",
            frame_index=1,
            timestamp_seconds=1.0,
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            pts=None,
            pts_seconds=None,
            duration_seconds=None,
        ),
    ]

    first, second = observer.observe_batch(frames)

    assert first.appearance_gaze.valid is True
    assert second.appearance_gaze.valid is False
    assert second.status is FrameStatus.ERROR
    assert ErrorCode.GAZE_MODEL_FAILED in {error.code for error in second.errors}


def test_model_backed_frame_observer_batch_propagates_model_contract_errors(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_face_observation(candidate)),
        gaze_model=_RaisingBatchGazeModel(),
        calibration=default_calibration(),
        run_layout=_run_layout(tmp_path),
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    with pytest.raises(ModelInferenceError, match="pred_gaze must have shape"):
        observer.observe_batch([_observer_frame()])
