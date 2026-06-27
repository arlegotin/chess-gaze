from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chess_gaze.calibration import default_calibration
from chess_gaze.face_observation import (
    FaceCandidate,
    FaceObservation,
    MediaPipeFaceObserver,
)
from chess_gaze.head_pose import PNP_METHOD_NAME, ImageSize, estimate_head_pose
from chess_gaze.model_assets import load_model_registry, sha256_file
from chess_gaze.video_decode import DecodedFrame, iter_decoded_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / "src" / "chess_gaze" / "model_registry.json"
MODELS_ROOT = REPO_ROOT / "models"
MEDIAPIPE_MODEL_ID = "mediapipe-face-landmarker"
NAKAMURA_SHORT_VIDEO = Path("artifacts/input/nakamura_short.mp4")
NAKAMURA_SHORT_FRAME_INDICES = (0, 30, 60, 90, 120, 150, 179)
SAMPLED_FRAME_INDICES = {
    NAKAMURA_SHORT_VIDEO: NAKAMURA_SHORT_FRAME_INDICES,
}
NAKAMURA_SHORT_TRANSFORM_POSE_FRAME_INDICES = (0, 30, 60, 90, 120, 150, 179)
NAKAMURA_SHORT_DOWN_LOOKING_FRAME_INDICES = frozenset(
    NAKAMURA_SHORT_TRANSFORM_POSE_FRAME_INDICES
)


def test_head_pose_matches_real_video_evidence() -> None:
    for video_path in SAMPLED_FRAME_INDICES:
        absolute_video_path = REPO_ROOT / video_path
        if not absolute_video_path.is_file():
            pytest.skip(
                f"BLOCKED: missing mandatory real-data video: {absolute_video_path}"
            )

    registry = load_model_registry(MODEL_REGISTRY_PATH)
    model_entry = registry.by_id(MEDIAPIPE_MODEL_ID)
    model_path = MODELS_ROOT / model_entry.expected_relative_path
    if not model_path.is_file():
        pytest.skip(
            "BLOCKED: missing mandatory MediaPipe Face Landmarker task asset: "
            f"{model_path}"
        )
    assert model_entry.checksum_sha256 is not None, (
        "missing committed registry checksum for "
        f"{MEDIAPIPE_MODEL_ID}; real MediaPipe verification cannot pass without it"
    )
    assert sha256_file(model_path) == model_entry.checksum_sha256

    calibration = default_calibration()
    observer = MediaPipeFaceObserver(
        model_asset_path=model_path,
        calibration=calibration,
    )
    valid_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    representative_failures: dict[str, list[str]] = {}
    try:
        for video_path, frame_indices in SAMPLED_FRAME_INDICES.items():
            absolute_video_path = REPO_ROOT / video_path
            sampled_frames = _sample_frames(absolute_video_path, frame_indices)
            video_face_present_count = 0
            video_valid_count = 0
            video_invalid_count = 0
            video_preserved_transform_count = 0
            video_failures: list[str] = []

            for frame in sampled_frames:
                face_observation = observer.observe(frame.rgb, frame_id=frame.frame_id)
                if not face_observation.selection.present:
                    video_failures.append(f"{frame.frame_id}:face_missing")
                    continue

                selected_face = _selected_face(face_observation)
                video_face_present_count += 1
                if selected_face.facial_transformation_matrix is not None:
                    video_preserved_transform_count += 1

                observation = estimate_head_pose(
                    selected_face,
                    calibration,
                    ImageSize(
                        width_px=frame.rgb.shape[1],
                        height_px=frame.rgb.shape[0],
                    ),
                )

                assert (
                    observation.facial_transformation_matrix
                    == selected_face.facial_transformation_matrix
                )
                assert observation.metric_translation_allowed is False
                assert observation.translation_camera_3d_m is None
                assert observation.pnp_point_count >= 0
                assert observation.pnp_min_point_count > 0

                if observation.valid:
                    video_valid_count += 1
                    assert observation.pnp_method == PNP_METHOD_NAME
                    assert observation.rotation_matrix is not None
                    assert np.asarray(observation.rotation_matrix).shape == (3, 3)
                    assert observation.quaternion_wxyz is not None
                    assert all(
                        math.isfinite(value) for value in observation.quaternion_wxyz
                    )
                    assert observation.yaw_radians is not None
                    assert observation.pitch_radians is not None
                    assert observation.roll_radians is not None
                    assert math.isfinite(observation.yaw_radians)
                    assert math.isfinite(observation.pitch_radians)
                    assert math.isfinite(observation.roll_radians)
                    if observation.reprojection_error_px is not None:
                        assert math.isfinite(observation.reprojection_error_px)
                else:
                    video_invalid_count += 1
                    video_failures.append(
                        f"{frame.frame_id}:{observation.reason_invalid}:"
                        f"{observation.reprojection_error_px}"
                    )

            valid_counts[str(video_path)] = video_valid_count
            invalid_counts[str(video_path)] = video_invalid_count
            representative_failures[str(video_path)] = video_failures[:5]

            assert video_face_present_count > 0, (
                f"{video_path} produced no face candidates in sampled frames "
                f"{tuple(frame.frame_id for frame in sampled_frames)}; attach manual "
                "frame evidence proving the face is absent or fully hidden before "
                "accepting this result"
            )
            assert video_preserved_transform_count > 0, (
                f"{video_path} had no sampled detected-face frame with a MediaPipe "
                "facial transformation matrix"
            )
            assert video_valid_count > 0, (
                f"{video_path} produced no valid head-pose evidence in sampled "
                f"detected-face frames; representative failures={video_failures[:5]}"
            )
    finally:
        observer.close()

    print(f"head_pose_valid_counts={valid_counts}")
    print(f"head_pose_invalid_counts={invalid_counts}")
    print(f"representative_head_pose_failures={representative_failures}")


def test_head_pose_uses_mediapipe_transform_on_nakamura_short_frames() -> None:
    video_path = REPO_ROOT / NAKAMURA_SHORT_VIDEO
    if not video_path.is_file():
        pytest.skip(f"BLOCKED: missing repair verification video: {video_path}")

    registry = load_model_registry(MODEL_REGISTRY_PATH)
    model_entry = registry.by_id(MEDIAPIPE_MODEL_ID)
    model_path = MODELS_ROOT / model_entry.expected_relative_path
    if not model_path.is_file():
        pytest.skip(
            "BLOCKED: missing mandatory MediaPipe Face Landmarker task asset: "
            f"{model_path}"
        )
    assert model_entry.checksum_sha256 is not None
    assert sha256_file(model_path) == model_entry.checksum_sha256

    calibration = default_calibration()
    observer = MediaPipeFaceObserver(
        model_asset_path=model_path,
        calibration=calibration,
    )
    try:
        sampled_frames = _sample_frames(
            video_path,
            NAKAMURA_SHORT_TRANSFORM_POSE_FRAME_INDICES,
        )
        valid_frame_ids: list[str] = []
        for frame in sampled_frames:
            face_observation = observer.observe(frame.rgb, frame_id=frame.frame_id)
            assert face_observation.selection.present is True
            selected_face = _selected_face(face_observation)
            assert selected_face.facial_transformation_matrix is not None

            observation = estimate_head_pose(
                selected_face,
                calibration,
                ImageSize(
                    width_px=frame.rgb.shape[1],
                    height_px=frame.rgb.shape[0],
                ),
            )

            assert observation.valid is True
            assert observation.rotation_matrix is not None
            assert observation.quaternion_wxyz is not None
            assert observation.yaw_radians is not None
            assert observation.pitch_radians is not None
            assert observation.roll_radians is not None
            assert abs(observation.pitch_radians) < 1.0
            if frame.frame_index in NAKAMURA_SHORT_DOWN_LOOKING_FRAME_INDICES:
                assert observation.pitch_radians < 0.0
            valid_frame_ids.append(frame.frame_id)

        assert valid_frame_ids == [
            "f000000000",
            "f000000030",
            "f000000060",
            "f000000090",
            "f000000120",
            "f000000150",
            "f000000179",
        ]
    finally:
        observer.close()


def _selected_face(face_observation: FaceObservation) -> FaceCandidate:
    selected_id = face_observation.selection.primary_candidate_id
    for candidate in face_observation.selection.candidates:
        if candidate.candidate_id == selected_id:
            return candidate
    raise AssertionError(f"selected face candidate {selected_id!r} was not preserved")


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
