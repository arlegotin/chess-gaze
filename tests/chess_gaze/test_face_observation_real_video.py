from __future__ import annotations

from pathlib import Path

import pytest

from chess_gaze.calibration import default_calibration
from chess_gaze.errors import ErrorCode
from chess_gaze.face_observation import (
    MEDIAPIPE_SCORE_SOURCE_UNAVAILABLE,
    MediaPipeFaceObserver,
)
from chess_gaze.model_assets import load_model_registry, sha256_file
from chess_gaze.video_decode import DecodedFrame, iter_decoded_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / "src" / "chess_gaze" / "model_registry.json"
MODELS_ROOT = REPO_ROOT / "models"
MEDIAPIPE_MODEL_ID = "mediapipe-face-landmarker"
SAMPLED_FRAME_INDICES = {
    Path("artifacts/input/test_1.mp4"): (0, 300, 900, 1800, 2700, 3600),
    Path("artifacts/input/test_2.mp4"): (0, 300, 900, 1500, 1972),
}
TEST_0_RECOVERED_FRAME_INDICES = (80, 217, 247, 258)


def test_mediapipe_face_observer_matches_real_video_evidence() -> None:
    for video_path in SAMPLED_FRAME_INDICES:
        absolute_video_path = REPO_ROOT / video_path
        if not absolute_video_path.is_file():
            pytest.skip(
                f"BLOCKED: missing mandatory real-data video: {absolute_video_path}"
            )

    registry = load_model_registry(MODEL_REGISTRY_PATH)
    model_entry = registry.by_id(MEDIAPIPE_MODEL_ID)
    model_path = MODELS_ROOT / model_entry.expected_relative_path
    assert model_path.relative_to(REPO_ROOT) == Path(
        "models/mediapipe/face_landmarker.task"
    )
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

    observer = MediaPipeFaceObserver(
        model_asset_path=model_path,
        calibration=default_calibration(),
    )
    face_present_counts: dict[str, int] = {}
    face_not_found_frames: dict[str, list[str]] = {}
    try:
        for video_path, frame_indices in SAMPLED_FRAME_INDICES.items():
            absolute_video_path = REPO_ROOT / video_path
            sampled_frames = _sample_frames(absolute_video_path, frame_indices)
            present_count = 0
            not_found_frame_ids: list[str] = []

            for frame in sampled_frames:
                observation = observer.observe(frame.rgb, frame_id=frame.frame_id)

                assert observation.frame_id == frame.frame_id
                assert observation.image_width_px == frame.rgb.shape[1]
                assert observation.image_height_px == frame.rgb.shape[0]
                assert observation.selection.candidates == tuple(
                    observation.selection.candidates
                )

                for candidate in observation.selection.candidates:
                    assert candidate.frame_id == frame.frame_id
                    assert candidate.image_width_px == frame.rgb.shape[1]
                    assert candidate.image_height_px == frame.rgb.shape[0]
                    assert candidate.candidate_score is None
                    assert candidate.score_source == MEDIAPIPE_SCORE_SOURCE_UNAVAILABLE
                    assert candidate.selection_score_source is not None

                if observation.selection.present:
                    present_count += 1
                    assert observation.selection.primary_candidate_id is not None
                elif observation.selection.reason_invalid == ErrorCode.FACE_NOT_FOUND:
                    not_found_frame_ids.append(frame.frame_id)

            face_present_counts[str(video_path)] = present_count
            face_not_found_frames[str(video_path)] = not_found_frame_ids[:3]
            assert present_count > 0, (
                f"{video_path} produced no face candidates in sampled frames "
                f"{tuple(frame.frame_id for frame in sampled_frames)}; attach manual "
                "frame evidence proving the face is absent or fully hidden before "
                "accepting this result"
            )
    finally:
        observer.close()

    print(f"face_present_counts={face_present_counts}")
    print(f"representative_face_not_found_frames={face_not_found_frames}")


def test_mediapipe_face_observer_recovers_test0_visible_split_frame_faces() -> None:
    video_path = REPO_ROOT / "artifacts/input/test_0.mp4"
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

    observer = MediaPipeFaceObserver(
        model_asset_path=model_path,
        calibration=default_calibration(),
    )
    try:
        sampled_frames = _sample_frames(video_path, TEST_0_RECOVERED_FRAME_INDICES)
        recovered_frames: list[str] = []
        for frame in sampled_frames:
            observation = observer.observe(frame.rgb, frame_id=frame.frame_id)
            if observation.selection.present:
                recovered_frames.append(frame.frame_id)
                assert observation.selection.primary_candidate_id is not None

        assert recovered_frames == [
            "f000000080",
            "f000000217",
            "f000000247",
            "f000000258",
        ]
    finally:
        observer.close()


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
