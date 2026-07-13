from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

import chess_gaze.unigaze_preprocessing as unigaze_preprocessing
from chess_gaze.calibration import default_calibration
from chess_gaze.frame_records import CalibrationRecord, FaceRecord
from chess_gaze.gaze_observation import normalize_face_crop
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.model_assets import sha256_file

FIXED_FACE_MODEL_POINTS = np.asarray(
    [
        [-46.509498596191406, -38.32709503173828, 36.4160041809082],
        [-17.76072120666504, -32.58519744873047, 29.076156616210938],
        [18.04391098022461, -30.956823348999023, 29.06296730041504],
        [44.737586975097656, -34.107872009277344, 36.73243713378906],
        [-10.611664772033691, 12.958349227905273, 21.622764587402344],
        [11.289629936218262, 13.86424446105957, 21.837900161743164],
    ],
    dtype=np.float64,
)
FIXED_LANDMARKS_CROP_PX = np.asarray(
    [
        [16.613196859672737, 6.124515930812084],
        [26.840454238044952, 8.633409198615917],
        [39.32757877182398, 9.738644620713943],
        [48.12964125174405, 8.961837996842675],
        [28.927089425750758, 24.97960425499225],
        [36.59800894884669, 25.55988987337322],
    ],
    dtype=np.float64,
)
EXPECTED_NORMALIZED_FROM_CAMERA_ROTATION = np.asarray(
    [
        [0.9992439432689036, 0.038828770502339856, -0.001966830365993612],
        [-0.03875766751308453, 0.9988477511771979, 0.028302176191941107],
        [0.003063502792093412, -0.028204548383746697, 0.9995974781886515],
    ],
    dtype=np.float64,
)
EXPECTED_NORMALIZED_IMAGE_FROM_IMAGE_HOMOGRAPHY = np.asarray(
    [
        [3.748271714989965, 0.13541681110464907, -32.620254843067436],
        [-0.1442343254424911, 3.735487988635367, 34.59647648565123],
        [9.883283317641872e-06, -9.099177034933384e-05, 0.8274264718371915],
    ],
    dtype=np.float64,
)
EXPECTED_WARPED_RGB_SHA256 = (
    "d00fe00685d12a6873a9ca499f0c2b36e79bd8df8188e7b28ecb64b0f7d87a5f"
)
WARP_SAMPLE_INDICES = np.asarray((32, 56, 80, 104, 128, 152, 176, 200))
EXPECTED_WARPED_RGB_SAMPLES = np.asarray(
    [
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [6, 0, 5],
            [36, 0, 27],
            [68, 0, 51],
            [113, 0, 85],
            [168, 0, 126],
        ],
        [
            [48, 0, 36],
            [48, 0, 36],
            [96, 0, 72],
            [96, 0, 72],
            [144, 0, 108],
            [144, 0, 108],
            [162, 0, 122],
            [192, 0, 144],
        ],
        [
            [48, 0, 36],
            [48, 0, 36],
            [96, 0, 72],
            [96, 0, 72],
            [144, 0, 108],
            [144, 0, 108],
            [152, 0, 114],
            [192, 0, 144],
        ],
        [
            [48, 60, 72],
            [48, 60, 72],
            [96, 60, 108],
            [96, 60, 108],
            [144, 60, 144],
            [144, 60, 144],
            [144, 60, 144],
            [192, 60, 180],
        ],
        [
            [48, 60, 72],
            [48, 60, 72],
            [96, 60, 108],
            [96, 60, 108],
            [144, 60, 144],
            [144, 60, 144],
            [144, 60, 144],
            [192, 60, 180],
        ],
        [
            [48, 120, 108],
            [48, 120, 108],
            [96, 120, 144],
            [96, 120, 144],
            [143, 120, 179],
            [144, 120, 180],
            [144, 120, 180],
            [192, 120, 216],
        ],
        [
            [48, 120, 108],
            [48, 120, 108],
            [96, 120, 144],
            [96, 120, 144],
            [132, 120, 171],
            [144, 120, 180],
            [144, 120, 180],
            [192, 120, 216],
        ],
        [
            [48, 146, 124],
            [48, 158, 131],
            [96, 171, 174],
            [96, 180, 180],
            [123, 180, 200],
            [144, 180, 216],
            [144, 180, 216],
            [192, 180, 252],
        ],
    ],
    dtype=np.uint8,
)


def _coordinate_rgb_crop() -> np.ndarray:
    y, x = np.indices((48, 64))
    step = 12
    return np.stack(
        (
            ((x // step) * step * 4) % 256,
            ((y // step) * step * 5) % 256,
            (((x // step) + (y // step)) * step * 3) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_official_geometric_equation_matches_pinned_oracle() -> None:
    actual = unigaze_preprocessing.normalize_unigaze_face_geometry(
        _coordinate_rgb_crop(),
        FIXED_LANDMARKS_CROP_PX,
        FIXED_FACE_MODEL_POINTS,
    )

    np.testing.assert_allclose(
        actual.normalized_from_camera_rotation,
        EXPECTED_NORMALIZED_FROM_CAMERA_ROTATION,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual.camera_from_normalized_rotation,
        np.linalg.inv(EXPECTED_NORMALIZED_FROM_CAMERA_ROTATION),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual.normalized_image_from_image_homography,
        EXPECTED_NORMALIZED_IMAGE_FROM_IMAGE_HOMOGRAPHY,
        atol=1e-10,
    )
    assert hashlib.sha256(actual.warped_rgb.tobytes()).hexdigest() == (
        EXPECTED_WARPED_RGB_SHA256
    )
    np.testing.assert_allclose(
        actual.warped_rgb[np.ix_(WARP_SAMPLE_INDICES, WARP_SAMPLE_INDICES)],
        EXPECTED_WARPED_RGB_SAMPLES,
        atol=1,
    )


@pytest.mark.parametrize(
    ("landmarks", "face_model_points", "message"),
    [
        (np.zeros((5, 2)), FIXED_FACE_MODEL_POINTS, "landmarks_crop_px"),
        (FIXED_LANDMARKS_CROP_PX, np.zeros((6, 2)), "face_model_points"),
        (
            FIXED_LANDMARKS_CROP_PX.copy(),
            np.full((6, 3), np.nan),
            "face_model_points",
        ),
    ],
)
def test_official_geometric_equation_rejects_invalid_fixture_shapes(
    landmarks: np.ndarray,
    face_model_points: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        unigaze_preprocessing.normalize_unigaze_face_geometry(
            _coordinate_rgb_crop(), landmarks, face_model_points
        )


def test_official_geometric_equation_rejects_degenerate_center_and_basis() -> None:
    camera = np.asarray(
        [[256.0, 0.0, 32.0], [0.0, 256.0, 24.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="face_center_camera"):
        unigaze_preprocessing.warp_unigaze_face_geometry(
            _coordinate_rgb_crop(),
            camera_matrix=camera,
            head_rotation=np.eye(3),
            face_center_camera=np.zeros(3),
        )

    with pytest.raises(ValueError, match="normalization basis"):
        unigaze_preprocessing.warp_unigaze_face_geometry(
            _coordinate_rgb_crop(),
            camera_matrix=camera,
            head_rotation=np.eye(3),
            face_center_camera=np.asarray([1.0, 0.0, 0.0]),
        )

    with pytest.raises(ValueError, match="camera_matrix must be invertible"):
        unigaze_preprocessing.warp_unigaze_face_geometry(
            _coordinate_rgb_crop(),
            camera_matrix=np.zeros((3, 3)),
            head_rotation=np.eye(3),
            face_center_camera=np.asarray([0.0, 0.0, 600.0]),
        )


@pytest.mark.parametrize(
    ("failed_call", "message"),
    [(1, "EPNP"), (2, "refinement")],
)
def test_official_geometric_equation_rejects_failed_pose_solves(
    monkeypatch: pytest.MonkeyPatch,
    failed_call: int,
    message: str,
) -> None:
    call_count = 0

    def solve_pnp(
        *args: object, **kwargs: object
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        nonlocal call_count
        del args, kwargs
        call_count += 1
        return (
            call_count != failed_call,
            np.zeros((3, 1), dtype=np.float64),
            np.asarray([[0.0], [0.0], [600.0]], dtype=np.float64),
        )

    monkeypatch.setattr(cv2, "solvePnP", solve_pnp)

    with pytest.raises(ValueError, match=message):
        unigaze_preprocessing.normalize_unigaze_face_geometry(
            _coordinate_rgb_crop(),
            FIXED_LANDMARKS_CROP_PX,
            FIXED_FACE_MODEL_POINTS,
        )

    assert call_count == failed_call


def test_face_model_loader_selects_the_pinned_six_rows(tmp_path: Path) -> None:
    all_points = np.arange(50 * 3, dtype=np.float64).reshape(50, 3)
    path = tmp_path / "face_model.txt"
    np.savetxt(path, all_points)

    selected = unigaze_preprocessing.load_unigaze_face_model_points(path)

    np.testing.assert_array_equal(
        selected,
        all_points[np.asarray(unigaze_preprocessing.UNIGAZE_FACE_MODEL_ROW_INDICES)],
    )


@pytest.mark.parametrize(
    "all_points",
    [
        np.arange(49 * 3, dtype=np.float64).reshape(49, 3),
        np.full((50, 3), np.nan, dtype=np.float64),
    ],
)
def test_face_model_loader_rejects_wrong_shape_or_non_finite_points(
    tmp_path: Path,
    all_points: np.ndarray,
) -> None:
    path = tmp_path / "face_model.txt"
    np.savetxt(path, all_points)

    with pytest.raises(ValueError, match=r"finite with shape \(50, 3\)"):
        unigaze_preprocessing.load_unigaze_face_model_points(path)


def test_persisted_landmarks_profile_and_asset_reproduce_geometry() -> None:
    asset_path = Path("models/unigaze/face_model.txt")
    if not asset_path.is_file():
        pytest.skip(f"BLOCKED: missing pinned UniGaze face model: {asset_path}")
    assert (
        sha256_file(asset_path)
        == unigaze_preprocessing.UNIGAZE_FACE_MODEL_CHECKSUM_SHA256
    )

    calibration = default_calibration(
        unigaze_preprocessing_profile="official_geometric_v1"
    )
    persisted_calibration = CalibrationRecord.model_validate_json(
        calibration.model_dump_json()
    )
    assert (
        persisted_calibration.unigaze_face_model_checksum_sha256
        == unigaze_preprocessing.UNIGAZE_FACE_MODEL_CHECKSUM_SHA256
    )
    assert (
        persisted_calibration.unigaze_face_model_id
        == unigaze_preprocessing.UNIGAZE_FACE_MODEL_ID
    )

    landmarks = [
        Point2D(space=CoordinateSpace.IMAGE_PX, x=1.0, y=1.0) for _ in range(478)
    ]
    for index, (x, y) in zip(
        (33, 133, 362, 263, 98, 327),
        FIXED_LANDMARKS_CROP_PX,
        strict=True,
    ):
        landmarks[index] = Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=float(x),
            y=float(y),
        )
    face = FaceRecord(
        present=True,
        bounding_box=BBox(
            space=CoordinateSpace.IMAGE_PX,
            x_min=0.0,
            y_min=0.0,
            x_max=64.0,
            y_max=48.0,
        ),
        landmarks=landmarks,
        reason_invalid=None,
    )
    persisted_face = FaceRecord.model_validate_json(face.model_dump_json())
    assert persisted_face.bounding_box is not None
    face_model_points = unigaze_preprocessing.load_unigaze_face_model_points(asset_path)

    first = normalize_face_crop(
        _coordinate_rgb_crop(),
        persisted_face.bounding_box,
        input_size_px=persisted_calibration.unigaze_input_size_px,
        profile=persisted_calibration.unigaze_preprocessing_profile,
        crop_scale=persisted_calibration.unigaze_face_crop_scale,
        image_mean_rgb=persisted_calibration.unigaze_image_mean_rgb,
        image_std_rgb=persisted_calibration.unigaze_image_std_rgb,
        landmarks_image_px=persisted_face.landmarks,
        face_model_points=face_model_points,
    )
    second = normalize_face_crop(
        _coordinate_rgb_crop(),
        persisted_face.bounding_box,
        input_size_px=persisted_calibration.unigaze_input_size_px,
        profile=persisted_calibration.unigaze_preprocessing_profile,
        crop_scale=persisted_calibration.unigaze_face_crop_scale,
        image_mean_rgb=persisted_calibration.unigaze_image_mean_rgb,
        image_std_rgb=persisted_calibration.unigaze_image_std_rgb,
        landmarks_image_px=persisted_face.landmarks,
        face_model_points=face_model_points,
    )

    np.testing.assert_array_equal(first.tensor.numpy(), second.tensor.numpy())
    np.testing.assert_array_equal(
        first.camera_from_normalized_rotation,
        second.camera_from_normalized_rotation,
    )
    np.testing.assert_array_equal(
        first.normalized_image_from_cropped_image_homography,
        second.normalized_image_from_cropped_image_homography,
    )
    assert first.camera_from_normalized_rotation is not None
    assert first.normalized_image_from_cropped_image_homography is not None
    np.testing.assert_allclose(
        first.camera_from_normalized_rotation,
        np.linalg.inv(EXPECTED_NORMALIZED_FROM_CAMERA_ROTATION),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        first.normalized_image_from_cropped_image_homography,
        EXPECTED_NORMALIZED_IMAGE_FROM_IMAGE_HOMOGRAPHY,
        atol=1e-10,
    )
