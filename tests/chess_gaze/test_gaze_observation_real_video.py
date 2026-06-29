from __future__ import annotations

import importlib
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chess_gaze.artifact_runs import create_run_layout
from chess_gaze.calibration import default_calibration
from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.eye_observation import observe_eyes
from chess_gaze.face_observation import FaceCandidate, MediaPipeFaceObserver
from chess_gaze.frame_observation import ModelBackedFrameObserver
from chess_gaze.gaze_observation import (
    GazeThresholds,
    UniGazeModel,
    compute_per_eye_geometric_gaze,
    normalize_face_crop,
    synthesize_recommended_gaze,
)
from chess_gaze.head_pose import ImageSize, estimate_head_pose
from chess_gaze.model_assets import (
    ResolvedModelAsset,
    load_model_registry,
    sha256_file,
)
from chess_gaze.pipeline import ObserverFrame
from chess_gaze.video_decode import DecodedFrame, iter_decoded_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / "src" / "chess_gaze" / "model_registry.json"
MODELS_ROOT = REPO_ROOT / "models"
MEDIAPIPE_MODEL_ID = "mediapipe-face-landmarker"
UNIGAZE_MODEL_ID = "unigaze-h14-joint"
NAKAMURA_SHORT_VIDEO = Path("artifacts/input/nakamura_short.mp4")
NAKAMURA_SHORT_FRAME_INDICES = (0, 30, 60, 90, 120, 150, 179)
SAMPLED_FRAME_INDICES = {
    NAKAMURA_SHORT_VIDEO: NAKAMURA_SHORT_FRAME_INDICES,
}
NAKAMURA_SHORT_RECOMMENDED_FRAME_INDICES = (0, 90, 170)

pytestmark = pytest.mark.native_mediapipe


def test_default_model_observer_recommends_gaze_on_nakamura_short_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = REPO_ROOT / NAKAMURA_SHORT_VIDEO
    if not video_path.is_file():
        pytest.skip(f"BLOCKED: missing mandatory real-data video: {video_path}")

    registry = load_model_registry(MODEL_REGISTRY_PATH)
    mediapipe_entry = registry.by_id(MEDIAPIPE_MODEL_ID)
    unigaze_entry = registry.by_id(UNIGAZE_MODEL_ID)
    mediapipe_path = MODELS_ROOT / mediapipe_entry.expected_relative_path
    unigaze_path = MODELS_ROOT / unigaze_entry.expected_relative_path
    if not mediapipe_path.is_file():
        pytest.skip(
            "BLOCKED: missing mandatory MediaPipe Face Landmarker task asset: "
            f"{mediapipe_path}"
        )
    if not unigaze_path.is_file():
        pytest.skip(
            f"BLOCKED: missing mandatory UniGaze checkpoint asset: {unigaze_path}"
        )
    assert mediapipe_entry.checksum_sha256 is not None
    assert unigaze_entry.checksum_sha256 is not None
    assert sha256_file(mediapipe_path) == mediapipe_entry.checksum_sha256
    assert sha256_file(unigaze_path) == unigaze_entry.checksum_sha256

    _disable_network_helpers(monkeypatch)
    calibration = default_calibration()
    run_layout = create_run_layout(
        input_path=video_path,
        output_root=tmp_path / "nakamura-short-observer",
        clock=lambda: datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
        run_suffix="abcdef12",
    )
    face_observer = MediaPipeFaceObserver(
        model_asset_path=mediapipe_path,
        calibration=calibration,
    )
    observer = ModelBackedFrameObserver(
        face_observer=face_observer,
        gaze_model=UniGazeModel.from_local_asset(
            ResolvedModelAsset(
                model_id=unigaze_entry.model_id,
                task_name=unigaze_entry.task_name,
                resolved_path=unigaze_path,
                source_url=unigaze_entry.source_url,
                checksum_sha256=unigaze_entry.checksum_sha256,
                license=unigaze_entry.license,
            ),
            device="cpu",
        ),
        calibration=calibration,
        run_layout=run_layout,
    )

    try:
        sampled_frames = _sample_frames(
            video_path,
            NAKAMURA_SHORT_RECOMMENDED_FRAME_INDICES,
        )
        records = [observer(_observer_frame(frame)) for frame in sampled_frames]
    finally:
        observer.close()

    assert [record.frame_index for record in records] == list(
        NAKAMURA_SHORT_RECOMMENDED_FRAME_INDICES
    )
    assert all(record.status is FrameStatus.OK for record in records)
    assert all(record.face.present for record in records)
    assert all(
        record.left_eye.present and record.right_eye.present for record in records
    )
    assert all(record.head_pose.valid for record in records)
    assert all(record.geometric_gaze.valid for record in records)
    assert all(record.appearance_gaze.valid for record in records)
    assert all(record.recommended_gaze.valid for record in records)
    assert all(not record.errors for record in records)


def _observer_frame(frame: DecodedFrame) -> ObserverFrame:
    timestamp_seconds = frame.pts_seconds if frame.pts_seconds is not None else 0.0
    return ObserverFrame(
        frame_id=frame.frame_id,
        frame_index=frame.frame_index,
        timestamp_seconds=timestamp_seconds,
        rgb=frame.rgb,
        pts=frame.pts,
        pts_seconds=frame.pts_seconds,
        duration_seconds=frame.duration_seconds,
    )


def test_unigaze_predicts_from_real_video_face_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for video_path in SAMPLED_FRAME_INDICES:
        absolute_video_path = REPO_ROOT / video_path
        if not absolute_video_path.is_file():
            pytest.skip(
                f"BLOCKED: missing mandatory real-data video: {absolute_video_path}"
            )

    registry = load_model_registry(MODEL_REGISTRY_PATH)
    mediapipe_entry = registry.by_id(MEDIAPIPE_MODEL_ID)
    unigaze_entry = registry.by_id(UNIGAZE_MODEL_ID)
    mediapipe_path = MODELS_ROOT / mediapipe_entry.expected_relative_path
    unigaze_path = MODELS_ROOT / unigaze_entry.expected_relative_path
    if not mediapipe_path.is_file():
        pytest.skip(
            "BLOCKED: missing mandatory MediaPipe Face Landmarker task asset: "
            f"{mediapipe_path}"
        )
    if not unigaze_path.is_file():
        pytest.skip(
            f"BLOCKED: missing mandatory UniGaze checkpoint asset: {unigaze_path}"
        )
    assert mediapipe_entry.checksum_sha256 is not None, (
        f"missing committed registry checksum for {MEDIAPIPE_MODEL_ID}"
    )
    assert unigaze_entry.checksum_sha256 is not None, (
        f"missing committed registry checksum for {UNIGAZE_MODEL_ID}"
    )
    assert sha256_file(mediapipe_path) == mediapipe_entry.checksum_sha256
    assert sha256_file(unigaze_path) == unigaze_entry.checksum_sha256

    _disable_network_helpers(monkeypatch)
    calibration = default_calibration()
    face_observer = MediaPipeFaceObserver(
        model_asset_path=mediapipe_path,
        calibration=calibration,
    )
    model = UniGazeModel.from_local_asset(
        ResolvedModelAsset(
            model_id=unigaze_entry.model_id,
            task_name=unigaze_entry.task_name,
            resolved_path=unigaze_path,
            source_url=unigaze_entry.source_url,
            checksum_sha256=unigaze_entry.checksum_sha256,
            license=unigaze_entry.license,
        ),
        device="cpu",
    )

    prediction_counts: dict[str, int] = {}
    representative_failures: dict[str, list[str]] = {}
    try:
        for video_path, frame_indices in SAMPLED_FRAME_INDICES.items():
            absolute_video_path = REPO_ROOT / video_path
            run_layout = create_run_layout(
                input_path=absolute_video_path,
                output_root=tmp_path / video_path.stem,
                clock=lambda: datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
                run_suffix="abcdef12",
            )
            sampled_frames = _sample_frames(absolute_video_path, frame_indices)
            video_prediction_count = 0
            video_failures: list[str] = []

            for frame in sampled_frames:
                face_observation = face_observer.observe(
                    frame.rgb, frame_id=frame.frame_id
                )
                if not face_observation.selection.present:
                    video_failures.append(f"{frame.frame_id}:face_missing")
                    continue

                selected_face = _selected_face(face_observation.selection.candidates)
                eye_observation = observe_eyes(
                    selected_face,
                    frame.rgb,
                    run_layout,
                    frame_id=frame.frame_id,
                )
                head_pose = estimate_head_pose(
                    selected_face,
                    calibration,
                    ImageSize(
                        width_px=frame.rgb.shape[1],
                        height_px=frame.rgb.shape[0],
                    ),
                )
                if (
                    not eye_observation.left.present
                    or not eye_observation.right.present
                    or not head_pose.valid
                ):
                    video_failures.append(f"{frame.frame_id}:incomplete_evidence")
                    continue

                left_gaze = compute_per_eye_geometric_gaze(
                    eye_observation.left,
                    head_pose,
                    missing_reason=ErrorCode.LEFT_EYE_NOT_FOUND,
                )
                right_gaze = compute_per_eye_geometric_gaze(
                    eye_observation.right,
                    head_pose,
                    missing_reason=ErrorCode.RIGHT_EYE_NOT_FOUND,
                )
                normalized_crop = normalize_face_crop(
                    frame.rgb,
                    selected_face.bounding_box_image_px,
                    input_size_px=calibration.unigaze_input_size_px,
                )
                assert normalized_crop.tensor.shape == (1, 3, 224, 224)
                face_gaze = model.predict(normalized_crop.tensor)
                recommended = synthesize_recommended_gaze(
                    left_gaze,
                    right_gaze,
                    face_gaze,
                    thresholds=GazeThresholds(max_pairwise_angle_delta_radians=math.pi),
                )

                assert face_gaze.valid is True
                assert face_gaze.method == "unigaze_h14_joint"
                assert face_gaze.pitch_radians is not None
                assert face_gaze.yaw_radians is not None
                assert math.isfinite(face_gaze.pitch_radians)
                assert math.isfinite(face_gaze.yaw_radians)
                assert face_gaze.confidence is None
                assert face_gaze.confidence_source == "not_provided_by_unigaze"
                assert recommended.target_image_px is None
                assert recommended.target_board_norm is None
                assert recommended.target_square is None
                video_prediction_count += 1
                break

            prediction_counts[str(video_path)] = video_prediction_count
            representative_failures[str(video_path)] = video_failures[:5]
            assert video_prediction_count > 0, (
                f"{video_path} produced no UniGaze predictions from sampled real "
                f"frames; representative failures={video_failures[:5]}"
            )
    finally:
        face_observer.close()

    print(f"unigaze_prediction_counts={prediction_counts}")
    print(f"representative_unigaze_failures={representative_failures}")


def _disable_network_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    huggingface_hub = importlib.import_module("huggingface_hub")
    unigaze = importlib.import_module("unigaze")

    def fail_network_helper(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network helper must not be used during analysis")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_network_helper)
    monkeypatch.setattr(unigaze, "load", fail_network_helper, raising=False)


def _selected_face(candidates: tuple[FaceCandidate, ...]) -> FaceCandidate:
    selected = tuple(
        candidate for candidate in candidates if candidate.selection_score is not None
    )
    assert selected, "detected face selection did not preserve selected candidate score"
    return sorted(
        selected,
        key=lambda candidate: (
            -(candidate.selection_score or 0.0),
            candidate.candidate_id,
        ),
    )[0]


def _sample_frames(
    video_path: Path,
    frame_indices: tuple[int, ...],
) -> tuple[DecodedFrame, ...]:
    wanted_indices = set(frame_indices)
    sampled: dict[int, DecodedFrame] = {}
    for frame in iter_decoded_frames(video_path):
        if frame.frame_index in wanted_indices:
            sampled[frame.frame_index] = frame
        if len(sampled) == len(wanted_indices):
            break

    missing_indices = sorted(wanted_indices.difference(sampled))
    assert not missing_indices, (
        f"{video_path} did not decode expected sample frames: {missing_indices}"
    )

    ordered_frames = tuple(sampled[index] for index in frame_indices)
    assert tuple(frame.frame_id for frame in ordered_frames) == tuple(
        f"f{index:09d}" for index in frame_indices
    )
    return ordered_frames
