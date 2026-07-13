from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

UniGazePreprocessingProfile = Literal[
    "legacy_bbox_rgb01",
    "reference_face2x_imagenet",
    "official_geometric_v1",
]

LEGACY_UNIGAZE_PREPROCESSING_PROFILE: UniGazePreprocessingProfile = "legacy_bbox_rgb01"
REFERENCE_UNIGAZE_PREPROCESSING_PROFILE: UniGazePreprocessingProfile = (
    "reference_face2x_imagenet"
)
OFFICIAL_UNIGAZE_PREPROCESSING_PROFILE: UniGazePreprocessingProfile = (
    "official_geometric_v1"
)
DEFAULT_UNIGAZE_PREPROCESSING_PROFILE = OFFICIAL_UNIGAZE_PREPROCESSING_PROFILE

LEGACY_UNIGAZE_FACE_CROP_SCALE = 1.0
REFERENCE_UNIGAZE_FACE_CROP_SCALE = 2.0
OFFICIAL_UNIGAZE_FACE_CROP_SCALE = 2.0
UNIGAZE_IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
UNIGAZE_IMAGENET_STD_RGB = (0.229, 0.224, 0.225)
UNIGAZE_FACE_MODEL_ID = "unigaze-face-model-v1"
UNIGAZE_FACE_MODEL_CHECKSUM_SHA256 = (
    "0c943d1d48627d97038b64f9a73816b9ab80a002ce81a8f04d532da2f4c337d7"
)
UNIGAZE_FACE_MODEL_ROW_INDICES = (20, 23, 26, 29, 15, 19)
UNIGAZE_MEDIAPIPE_LANDMARK_INDICES = (33, 133, 362, 263, 98, 327)
UNIGAZE_FACE_MODEL_ROW_COUNT = 50
UNIGAZE_NORMALIZED_FOCAL_LENGTH_PX = 960.0
UNIGAZE_NORMALIZED_DISTANCE = 600.0
UNIGAZE_NORMALIZED_SIZE_PX = 224


@dataclass(frozen=True)
class UniGazePreprocessingConfig:
    profile: UniGazePreprocessingProfile
    crop_scale: float
    image_mean_rgb: tuple[float, float, float] | None
    image_std_rgb: tuple[float, float, float] | None
    face_model_id: str | None
    face_model_checksum_sha256: str | None


@dataclass(frozen=True)
class UniGazeGeometricNormalization:
    warped_rgb: npt.NDArray[np.uint8]
    normalized_from_camera_rotation: npt.NDArray[np.float64]
    camera_from_normalized_rotation: npt.NDArray[np.float64]
    normalized_image_from_image_homography: npt.NDArray[np.float64]


def resolve_unigaze_preprocessing_profile(
    profile: str,
) -> UniGazePreprocessingConfig:
    if profile == LEGACY_UNIGAZE_PREPROCESSING_PROFILE:
        return UniGazePreprocessingConfig(
            profile=LEGACY_UNIGAZE_PREPROCESSING_PROFILE,
            crop_scale=LEGACY_UNIGAZE_FACE_CROP_SCALE,
            image_mean_rgb=None,
            image_std_rgb=None,
            face_model_id=None,
            face_model_checksum_sha256=None,
        )
    if profile == REFERENCE_UNIGAZE_PREPROCESSING_PROFILE:
        return UniGazePreprocessingConfig(
            profile=REFERENCE_UNIGAZE_PREPROCESSING_PROFILE,
            crop_scale=REFERENCE_UNIGAZE_FACE_CROP_SCALE,
            image_mean_rgb=UNIGAZE_IMAGENET_MEAN_RGB,
            image_std_rgb=UNIGAZE_IMAGENET_STD_RGB,
            face_model_id=None,
            face_model_checksum_sha256=None,
        )
    if profile == OFFICIAL_UNIGAZE_PREPROCESSING_PROFILE:
        return UniGazePreprocessingConfig(
            profile=OFFICIAL_UNIGAZE_PREPROCESSING_PROFILE,
            crop_scale=OFFICIAL_UNIGAZE_FACE_CROP_SCALE,
            image_mean_rgb=UNIGAZE_IMAGENET_MEAN_RGB,
            image_std_rgb=UNIGAZE_IMAGENET_STD_RGB,
            face_model_id=UNIGAZE_FACE_MODEL_ID,
            face_model_checksum_sha256=UNIGAZE_FACE_MODEL_CHECKSUM_SHA256,
        )
    raise ValueError(f"unknown unigaze preprocessing profile: {profile!r}")


def load_unigaze_face_model_points(
    path: Path,
) -> npt.NDArray[np.float64]:
    all_points = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64)
    expected_shape = (UNIGAZE_FACE_MODEL_ROW_COUNT, 3)
    if all_points.shape != expected_shape or not np.isfinite(all_points).all():
        raise ValueError(
            f"UniGaze face model must be finite with shape {expected_shape}"
        )
    return np.ascontiguousarray(
        all_points[np.asarray(UNIGAZE_FACE_MODEL_ROW_INDICES)],
        dtype=np.float64,
    )


def normalize_unigaze_face_geometry(
    rgb_crop: npt.NDArray[np.uint8],
    landmarks_crop_px: npt.ArrayLike,
    face_model_points: npt.ArrayLike,
) -> UniGazeGeometricNormalization:
    crop = _validate_rgb_crop(rgb_crop)
    landmarks = _finite_array(landmarks_crop_px, shape=(6, 2), name="landmarks_crop_px")
    model_points = _finite_array(
        face_model_points, shape=(6, 3), name="face_model_points"
    )
    height_px, width_px, _channels = crop.shape
    focal_length_px = float(width_px * 4)
    camera_matrix = np.asarray(
        [
            [focal_length_px, 0.0, float(width_px // 2)],
            [0.0, focal_length_px, float(height_px // 2)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((1, 5), dtype=np.float64)
    success, rotation_vector, translation = cv2.solvePnP(
        model_points.reshape(6, 1, 3),
        landmarks.reshape(6, 1, 2),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        raise ValueError("UniGaze face pose solvePnP EPNP failed")
    success, rotation_vector, translation = cv2.solvePnP(
        model_points.reshape(6, 1, 3),
        landmarks.reshape(6, 1, 2),
        camera_matrix,
        distortion,
        rotation_vector,
        translation,
        True,
    )
    if not success:
        raise ValueError("UniGaze face pose solvePnP refinement failed")
    head_rotation = np.asarray(cv2.Rodrigues(rotation_vector)[0], dtype=np.float64)
    transformed_points = head_rotation @ model_points.T + translation.reshape(3, 1)
    two_eye_center = np.mean(transformed_points[:, :4], axis=1)
    nose_center = np.mean(transformed_points[:, 4:6], axis=1)
    face_center = np.mean(np.column_stack((two_eye_center, nose_center)), axis=1)
    return warp_unigaze_face_geometry(
        crop,
        camera_matrix=camera_matrix,
        head_rotation=head_rotation,
        face_center_camera=face_center,
    )


def warp_unigaze_face_geometry(
    rgb_crop: npt.NDArray[np.uint8],
    *,
    camera_matrix: npt.ArrayLike,
    head_rotation: npt.ArrayLike,
    face_center_camera: npt.ArrayLike,
) -> UniGazeGeometricNormalization:
    """Apply the pinned UniGaze camera-normalization equation.

    Equation source: UniGaze revision 9c240fbe33f3d6146970a77b7c8fa06a7e60019e,
    ``unigaze/gazelib/gaze/normalize.py``. This is an independent implementation.
    """

    crop = _validate_rgb_crop(rgb_crop)
    camera = _finite_array(camera_matrix, shape=(3, 3), name="camera_matrix")
    rotation = _finite_array(head_rotation, shape=(3, 3), name="head_rotation")
    center_raw = np.asarray(face_center_camera, dtype=np.float64)
    if center_raw.shape not in ((3,), (3, 1)) or not np.isfinite(center_raw).all():
        raise ValueError("face_center_camera must be finite with shape (3,)")
    center = center_raw.reshape(3)
    distance = float(np.linalg.norm(center))
    if not np.isfinite(distance) or distance <= np.finfo(np.float64).eps:
        raise ValueError("face_center_camera must have non-zero finite distance")
    try:
        camera_inverse = np.linalg.inv(camera)
    except np.linalg.LinAlgError as exc:
        raise ValueError("camera_matrix must be invertible") from exc

    forward = center / distance
    down = np.cross(forward, rotation[:, 0])
    down_norm = float(np.linalg.norm(down))
    if not np.isfinite(down_norm) or down_norm <= np.finfo(np.float64).eps:
        raise ValueError("UniGaze normalization basis is degenerate")
    down /= down_norm
    right = np.cross(down, forward)
    right_norm = float(np.linalg.norm(right))
    if not np.isfinite(right_norm) or right_norm <= np.finfo(np.float64).eps:
        raise ValueError("UniGaze normalization basis is degenerate")
    right /= right_norm
    normalized_from_camera = np.column_stack((right, down, forward)).T
    normalized_camera = np.asarray(
        [
            [
                UNIGAZE_NORMALIZED_FOCAL_LENGTH_PX,
                0.0,
                UNIGAZE_NORMALIZED_SIZE_PX / 2.0,
            ],
            [
                0.0,
                UNIGAZE_NORMALIZED_FOCAL_LENGTH_PX,
                UNIGAZE_NORMALIZED_SIZE_PX / 2.0,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = np.diag((1.0, 1.0, UNIGAZE_NORMALIZED_DISTANCE / distance))
    homography = normalized_camera @ scale @ normalized_from_camera @ camera_inverse
    if not np.isfinite(homography).all():
        raise ValueError("UniGaze normalization homography must be finite")
    warped = cv2.warpPerspective(
        crop,
        homography,
        (UNIGAZE_NORMALIZED_SIZE_PX, UNIGAZE_NORMALIZED_SIZE_PX),
    )
    return UniGazeGeometricNormalization(
        warped_rgb=np.ascontiguousarray(warped),
        normalized_from_camera_rotation=normalized_from_camera,
        camera_from_normalized_rotation=np.linalg.inv(normalized_from_camera),
        normalized_image_from_image_homography=homography,
    )


def _validate_rgb_crop(
    rgb_crop: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    if rgb_crop.ndim != 3 or rgb_crop.shape[2] != 3 or rgb_crop.dtype != np.uint8:
        raise ValueError("rgb_crop must be uint8 with shape (height, width, 3)")
    if rgb_crop.shape[0] < 1 or rgb_crop.shape[1] < 1:
        raise ValueError("rgb_crop must be non-empty")
    return np.ascontiguousarray(rgb_crop)


def _finite_array(
    value: npt.ArrayLike,
    *,
    shape: tuple[int, ...],
    name: str,
) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return np.ascontiguousarray(array)
