from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_gaze.calibration import default_calibration
from chess_gaze.face_observation import FaceCandidate, MediaPipeFaceObserver
from chess_gaze.gaze_observation import normalize_face_crop
from chess_gaze.model_assets import load_model_registry, sha256_file
from chess_gaze.unigaze_preprocessing import (
    UNIGAZE_FACE_MODEL_ID,
    load_unigaze_face_model_points,
)
from chess_gaze.video_decode import iter_decoded_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / "src" / "chess_gaze" / "model_registry.json"
MODELS_ROOT = REPO_ROOT / "models"
INPUT_ROOT = REPO_ROOT / "artifacts" / "input"
MEDIAPIPE_MODEL_ID = "mediapipe-face-landmarker"
SAMPLE_COORDINATES = np.asarray((0, 56, 112, 168, 223))

EXPECTED_ORACLES = {
    "carlsen_short.mp4": {
        "source_sha256": (
            "48505b38898a843c5b03d9cfa717efda2a915f0c5399c81369be20d316f6fc01"
        ),
        "source_bbox": (405.0, 160.0, 749.0, 530.0),
        "normalized_from_camera_rotation": (
            (0.9993842009041407, -0.031217592645080932, -0.016021263765325974),
            (0.0313715116220637, 0.9994631430295481, 0.009447432669448754),
            (0.01571773653358143, -0.009944226211366787, 0.9998270176002039),
        ),
        "homography": (
            (0.6978695086328435, -0.02217495910759597, -76.6305527939751),
            (0.02251181843604569, 0.696904623921215, -69.04880436307214),
            (5.577833074761944e-06, -3.5289581134133957e-06, 0.4879171170217803),
        ),
        "warped_rgb_samples": (
            (106, 66, 60),
            (93, 53, 48),
            (177, 142, 130),
            (173, 146, 134),
            (59, 52, 54),
        ),
    },
    "nakamura_short.mp4": {
        "source_sha256": (
            "6524928897505e614a0eae419a1b7bd0e2a8dff25ffed22db2706d02bbf909bc"
        ),
        "source_bbox": (267.0, 521.0, 609.0, 905.0),
        "normalized_from_camera_rotation": (
            (0.9823086780572287, -0.18624948702763974, -0.019513830874557662),
            (0.18637288318533532, 0.9824678783484496, 0.004692166523187148),
            (0.018297798409785265, -0.008246004816698673, 0.9997985767032866),
        ),
        "homography": (
            (0.690083660613677, -0.13103678914392727, -55.948236634283944),
            (0.13153222561585712, 0.6891157479621004, -94.66747087217902),
            (6.644976925194498e-06, -2.9945958800542067e-06, 0.4961376307707991),
        ),
        "warped_rgb_samples": (
            (114, 88, 76),
            (218, 180, 171),
            (255, 229, 189),
            (237, 186, 139),
            (97, 82, 70),
        ),
    },
    "nepo_short.mp4": {
        "source_sha256": (
            "aa24fb658a3a3723d8b953d01c5ddf174d60978b6a5a2312c5c79f4b23c36b8c"
        ),
        "source_bbox": (849.0, 111.0, 1205.0, 535.0),
        "normalized_from_camera_rotation": (
            (0.9918895291372973, -0.12709993068973963, -0.0008772721663882301),
            (0.12710287155480313, 0.9918571665707189, 0.00801381088050207),
            (-0.00014842612220730824, -0.008060318912339597, 0.9999675040863666),
        ),
        "homography": (
            (0.6686830470975332, -0.08603958556426182, -39.04672245442225),
            (0.08568080603068329, 0.6683135137124517, -86.65974086617153),
            (-5.824203023416893e-08, -3.1628484986884036e-06, 0.5594366862224597),
        ),
        "warped_rgb_samples": (
            (117, 109, 101),
            (64, 40, 47),
            (163, 140, 144),
            (138, 118, 132),
            (45, 56, 60),
        ),
    },
}

pytestmark = pytest.mark.native_mediapipe


def test_official_geometric_normalization_matches_pinned_short_video_frames() -> None:
    video_paths = sorted(INPUT_ROOT.glob("*_short.mp4"))
    assert [path.name for path in video_paths] == sorted(EXPECTED_ORACLES)
    for video_path in video_paths:
        assert (
            sha256_file(video_path)
            == EXPECTED_ORACLES[video_path.name]["source_sha256"]
        )

    registry = load_model_registry(MODEL_REGISTRY_PATH)
    mediapipe_entry = registry.by_id(MEDIAPIPE_MODEL_ID)
    face_model_entry = registry.by_id(UNIGAZE_FACE_MODEL_ID)
    mediapipe_path = MODELS_ROOT / mediapipe_entry.expected_relative_path
    face_model_path = MODELS_ROOT / face_model_entry.expected_relative_path
    assert mediapipe_entry.checksum_sha256 is not None
    assert face_model_entry.checksum_sha256 is not None
    assert sha256_file(mediapipe_path) == mediapipe_entry.checksum_sha256
    assert sha256_file(face_model_path) == face_model_entry.checksum_sha256

    calibration = default_calibration(
        unigaze_preprocessing_profile="official_geometric_v1"
    )
    face_model_points = load_unigaze_face_model_points(face_model_path)
    observer = MediaPipeFaceObserver(
        model_asset_path=mediapipe_path,
        calibration=calibration,
    )
    try:
        for video_path in video_paths:
            frame = next(iter_decoded_frames(video_path))
            observation = observer.observe(frame.rgb, frame_id=frame.frame_id)
            face = _selected_face(
                observation.selection.candidates,
                observation.selection.primary_candidate_id,
            )
            normalized = normalize_face_crop(
                frame.rgb,
                face.bounding_box_image_px,
                input_size_px=calibration.unigaze_input_size_px,
                profile=calibration.unigaze_preprocessing_profile,
                crop_scale=calibration.unigaze_face_crop_scale,
                image_mean_rgb=calibration.unigaze_image_mean_rgb,
                image_std_rgb=calibration.unigaze_image_std_rgb,
                landmarks_image_px=face.landmarks_image_px,
                face_model_points=face_model_points,
            )
            expected = EXPECTED_ORACLES[video_path.name]

            bbox = normalized.transform.source_bbox_image_px
            assert (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) == expected[
                "source_bbox"
            ]
            rotation = np.asarray(
                expected["normalized_from_camera_rotation"], dtype=np.float64
            )
            assert normalized.camera_from_normalized_rotation is not None
            np.testing.assert_allclose(
                normalized.camera_from_normalized_rotation,
                np.linalg.inv(rotation),
                rtol=0.0,
                atol=1e-10,
            )
            assert normalized.normalized_image_from_cropped_image_homography is not None
            np.testing.assert_allclose(
                normalized.normalized_image_from_cropped_image_homography,
                np.asarray(expected["homography"], dtype=np.float64),
                rtol=0.0,
                atol=1e-10,
            )

            normalized_hwc = normalized.tensor[0].permute(1, 2, 0).numpy()
            mean = np.asarray(calibration.unigaze_image_mean_rgb)
            std = np.asarray(calibration.unigaze_image_std_rgb)
            warped_rgb = (normalized_hwc * std + mean) * 255.0
            sampled_rgb = warped_rgb[SAMPLE_COORDINATES, SAMPLE_COORDINATES]
            np.testing.assert_allclose(
                sampled_rgb,
                np.asarray(expected["warped_rgb_samples"]),
                rtol=0.0,
                atol=1.1,
            )
    finally:
        observer.close()


def _selected_face(
    candidates: tuple[FaceCandidate, ...], selected_id: str | None
) -> FaceCandidate:
    assert selected_id is not None
    return next(
        candidate for candidate in candidates if candidate.candidate_id == selected_id
    )
