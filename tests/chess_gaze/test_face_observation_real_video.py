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
NAKAMURA_SHORT_VIDEO = Path("artifacts/input/nakamura_short.mp4")
NAKAMURA_SHORT_FRAME_INDICES = (0, 30, 60, 90, 120, 150, 179)
CARLSEN_VIDEO = Path("artifacts/input/carlsen_1.mp4")
CARLSEN_REPORTED_VISIBLE_FRAME_INDICES = (
    2036,
    2037,
    2042,
    2050,
    2062,
    5694,
    5695,
    5697,
    9029,
    9030,
    9031,
    15079,
    15080,
    15081,
    15082,
    15083,
)
CARLSEN_ADJACENT_CONTROL_FRAME_INDICES = (
    2035,
    2063,
    5693,
    5698,
    9028,
    9032,
    15078,
    15084,
)
CARLSEN_FACE_CENTER_X_BOUNDS = (450.0, 660.0)
CARLSEN_FACE_CENTER_Y_BOUNDS = (240.0, 430.0)
SAMPLED_FRAME_INDICES = {
    NAKAMURA_SHORT_VIDEO: NAKAMURA_SHORT_FRAME_INDICES,
}
NAKAMURA_SHORT_EXPECTED_FACE_BOXES = {
    "f000000000": (368.3, 678.9, 532.2, 871.1),
    "f000000030": (348.9, 705.1, 513.4, 884.7),
    "f000000060": (353.7, 720.1, 519.3, 893.4),
    "f000000090": (363.7, 695.2, 527.9, 894.5),
    "f000000120": (390.8, 650.9, 547.8, 852.3),
    "f000000150": (388.0, 635.1, 541.8, 823.0),
    "f000000179": (389.6, 685.2, 551.0, 869.4),
}

pytestmark = pytest.mark.native_mediapipe


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


def test_mediapipe_face_observer_recovers_nakamura_short_visible_faces() -> None:
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

    observer = MediaPipeFaceObserver(
        model_asset_path=model_path,
        calibration=default_calibration(),
    )
    try:
        sampled_frames = _sample_frames(video_path, NAKAMURA_SHORT_FRAME_INDICES)
        recovered_boxes: dict[str, tuple[float, float, float, float]] = {}
        for frame in sampled_frames:
            observation = observer.observe(frame.rgb, frame_id=frame.frame_id)
            assert observation.selection.present, (
                f"{frame.frame_id} should recover the visible face"
            )
            assert observation.selection.primary_candidate_id is not None
            candidate = next(
                item
                for item in observation.selection.candidates
                if item.candidate_id == observation.selection.primary_candidate_id
            )
            bbox = candidate.bounding_box_image_px
            recovered_boxes[frame.frame_id] = (
                round(bbox.x_min, 1),
                round(bbox.y_min, 1),
                round(bbox.x_max, 1),
                round(bbox.y_max, 1),
            )

        assert recovered_boxes == NAKAMURA_SHORT_EXPECTED_FACE_BOXES
    finally:
        observer.close()


def test_mediapipe_observer_keeps_nakamura_short_faces_bounded() -> None:
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

    observer = MediaPipeFaceObserver(
        model_asset_path=model_path,
        calibration=default_calibration(),
    )
    try:
        sampled_frames = _sample_frames(video_path, NAKAMURA_SHORT_FRAME_INDICES)
        recovered_boxes: dict[str, tuple[float, float, float, float]] = {}
        for frame in sampled_frames:
            observation = observer.observe(frame.rgb, frame_id=frame.frame_id)

            assert observation.selection.present, (
                f"{frame.frame_id} should recover face"
            )
            assert observation.selection.primary_candidate_id is not None
            candidate = next(
                item
                for item in observation.selection.candidates
                if item.candidate_id == observation.selection.primary_candidate_id
            )
            bbox = candidate.bounding_box_image_px
            width = bbox.x_max - bbox.x_min
            height = bbox.y_max - bbox.y_min
            assert width <= 230.0, (
                f"{frame.frame_id} selected face width {width:.1f} is overexpanded"
            )
            assert height <= 285.0, (
                f"{frame.frame_id} selected face height {height:.1f} is overexpanded"
            )
            recovered_boxes[frame.frame_id] = (
                round(bbox.x_min, 1),
                round(bbox.y_min, 1),
                round(bbox.x_max, 1),
                round(bbox.y_max, 1),
            )

        assert recovered_boxes == NAKAMURA_SHORT_EXPECTED_FACE_BOXES
    finally:
        observer.close()


def test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region() -> (
    None
):
    video_path = REPO_ROOT / CARLSEN_VIDEO
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

    sampled_indices = (
        CARLSEN_ADJACENT_CONTROL_FRAME_INDICES + CARLSEN_REPORTED_VISIBLE_FRAME_INDICES
    )
    observer = MediaPipeFaceObserver(
        model_asset_path=model_path,
        calibration=default_calibration(),
    )
    try:
        sampled_frames = _sample_frames(video_path, sampled_indices)
        recovered_centers: dict[str, tuple[float, float]] = {}
        for frame in sampled_frames:
            observation = observer.observe(frame.rgb, frame_id=frame.frame_id)
            assert observation.selection.present, (
                f"{frame.frame_id} should recover the visible player face"
            )
            assert observation.selection.primary_candidate_id is not None
            candidate = next(
                item
                for item in observation.selection.candidates
                if item.candidate_id == observation.selection.primary_candidate_id
            )
            bbox = candidate.bounding_box_image_px
            center_x = round((bbox.x_min + bbox.x_max) / 2, 1)
            center_y = round((bbox.y_min + bbox.y_max) / 2, 1)
            recovered_centers[frame.frame_id] = (center_x, center_y)
            assert (
                CARLSEN_FACE_CENTER_X_BOUNDS[0]
                <= center_x
                <= (CARLSEN_FACE_CENTER_X_BOUNDS[1])
            ), (
                f"{frame.frame_id} selected center x={center_x:.1f} outside "
                "visible player-face bounds"
            )
            assert (
                CARLSEN_FACE_CENTER_Y_BOUNDS[0]
                <= center_y
                <= (CARLSEN_FACE_CENTER_Y_BOUNDS[1])
            ), (
                f"{frame.frame_id} selected center y={center_y:.1f} outside "
                "visible player-face bounds"
            )

        assert set(recovered_centers) == {f"f{index:09d}" for index in sampled_indices}
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
