from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from chess_gaze.artifact_runs import create_run_layout
from chess_gaze.calibration import default_calibration
from chess_gaze.eye_observation import CropTransformToImagePx, observe_eyes
from chess_gaze.face_observation import FaceCandidate, MediaPipeFaceObserver
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

pytestmark = pytest.mark.native_mediapipe


def test_eye_observation_matches_real_video_evidence(tmp_path: Path) -> None:
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
    eye_present_counts: dict[str, dict[str, int]] = {}
    iris_present_counts: dict[str, dict[str, int]] = {}
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
            video_eye_counts = {"left": 0, "right": 0}
            video_iris_counts = {"left": 0, "right": 0}
            video_face_present_count = 0
            video_independent_pair_count = 0
            video_failures: list[str] = []

            for frame in sampled_frames:
                face_observation = observer.observe(frame.rgb, frame_id=frame.frame_id)
                if not face_observation.selection.present:
                    video_failures.append(f"{frame.frame_id}:face_missing")
                    continue

                selected_face = _selected_face(face_observation.selection.candidates)
                video_face_present_count += 1
                eye_observation = observe_eyes(
                    selected_face,
                    frame.rgb,
                    run_layout,
                    frame_id=frame.frame_id,
                )

                assert eye_observation.frame_id == frame.frame_id
                for side_name, eye in (
                    ("left", eye_observation.left),
                    ("right", eye_observation.right),
                ):
                    if eye.present:
                        video_eye_counts[side_name] += 1
                        assert eye.eye_crop_path is None
                        assert eye.eye_crop_sha256 is None
                        assert eye.crop_bbox_image_px is not None
                        assert eye.eye_crop_transform_to_image_px is not None
                        _assert_crop_transform_maps_to_image_px(
                            eye.eye_crop_transform_to_image_px,
                            crop_x=0.0,
                            crop_y=0.0,
                            expected_x=eye.crop_bbox_image_px.x_min,
                            expected_y=eye.crop_bbox_image_px.y_min,
                        )

                    if eye.iris_present:
                        video_iris_counts[side_name] += 1
                        assert eye.iris_center_image_px is not None
                        assert eye.crop_bbox_image_px is not None
                        assert eye.eye_crop_transform_to_image_px is not None
                        _assert_crop_transform_maps_to_image_px(
                            eye.eye_crop_transform_to_image_px,
                            crop_x=(
                                eye.iris_center_image_px.x
                                - eye.crop_bbox_image_px.x_min
                            ),
                            crop_y=(
                                eye.iris_center_image_px.y
                                - eye.crop_bbox_image_px.y_min
                            ),
                            expected_x=eye.iris_center_image_px.x,
                            expected_y=eye.iris_center_image_px.y,
                        )

                    if not eye.present or not eye.iris_present:
                        video_failures.append(
                            f"{frame.frame_id}:{side_name}:{eye.reason_missing}"
                        )

                if eye_observation.left.present and eye_observation.right.present:
                    video_independent_pair_count += 1
                    assert (
                        eye_observation.left.eye_landmarks_image_px
                        != eye_observation.right.eye_landmarks_image_px
                    )

            assert list(run_layout.crops_dir.rglob("*.png")) == []

            eye_present_counts[str(video_path)] = video_eye_counts
            iris_present_counts[str(video_path)] = video_iris_counts
            representative_failures[str(video_path)] = video_failures[:5]

            assert video_face_present_count > 0, (
                f"{video_path} produced no face candidates in sampled frames "
                f"{tuple(frame.frame_id for frame in sampled_frames)}; attach manual "
                "frame evidence proving the face is absent or fully hidden before "
                "accepting this result"
            )
            assert sum(video_eye_counts.values()) > 0, (
                f"{video_path} produced no eye evidence in sampled frames with a "
                f"detected face; representative failures={video_failures[:5]}"
            )
            assert sum(video_iris_counts.values()) > 0, (
                f"{video_path} produced no iris evidence in sampled frames with a "
                f"detected face; representative failures={video_failures[:5]}"
            )
            assert video_independent_pair_count > 0, (
                f"{video_path} had no sampled detected-face frame where both eyes "
                "were independently observable"
            )
    finally:
        observer.close()

    print(f"eye_present_counts={eye_present_counts}")
    print(f"iris_present_counts={iris_present_counts}")
    print(f"representative_eye_failures={representative_failures}")


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


def _assert_crop_transform_maps_to_image_px(
    transform: CropTransformToImagePx,
    *,
    crop_x: float,
    crop_y: float,
    expected_x: float,
    expected_y: float,
) -> None:
    mapped_x = (transform.m00 * crop_x) + (transform.m01 * crop_y) + transform.m02
    mapped_y = (transform.m10 * crop_x) + (transform.m11 * crop_y) + transform.m12

    assert mapped_x == pytest.approx(expected_x)
    assert mapped_y == pytest.approx(expected_y)
