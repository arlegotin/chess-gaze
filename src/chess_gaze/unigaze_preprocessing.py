from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

UniGazePreprocessingProfile = Literal[
    "legacy_bbox_rgb01",
    "reference_face2x_imagenet",
]

LEGACY_UNIGAZE_PREPROCESSING_PROFILE: UniGazePreprocessingProfile = (
    "legacy_bbox_rgb01"
)
REFERENCE_UNIGAZE_PREPROCESSING_PROFILE: UniGazePreprocessingProfile = (
    "reference_face2x_imagenet"
)
DEFAULT_UNIGAZE_PREPROCESSING_PROFILE = REFERENCE_UNIGAZE_PREPROCESSING_PROFILE

LEGACY_UNIGAZE_FACE_CROP_SCALE = 1.0
REFERENCE_UNIGAZE_FACE_CROP_SCALE = 2.0
UNIGAZE_IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
UNIGAZE_IMAGENET_STD_RGB = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class UniGazePreprocessingConfig:
    profile: UniGazePreprocessingProfile
    crop_scale: float
    image_mean_rgb: tuple[float, float, float] | None
    image_std_rgb: tuple[float, float, float] | None


def resolve_unigaze_preprocessing_profile(
    profile: str,
) -> UniGazePreprocessingConfig:
    if profile == LEGACY_UNIGAZE_PREPROCESSING_PROFILE:
        return UniGazePreprocessingConfig(
            profile=LEGACY_UNIGAZE_PREPROCESSING_PROFILE,
            crop_scale=LEGACY_UNIGAZE_FACE_CROP_SCALE,
            image_mean_rgb=None,
            image_std_rgb=None,
        )
    if profile == REFERENCE_UNIGAZE_PREPROCESSING_PROFILE:
        return UniGazePreprocessingConfig(
            profile=REFERENCE_UNIGAZE_PREPROCESSING_PROFILE,
            crop_scale=REFERENCE_UNIGAZE_FACE_CROP_SCALE,
            image_mean_rgb=UNIGAZE_IMAGENET_MEAN_RGB,
            image_std_rgb=UNIGAZE_IMAGENET_STD_RGB,
        )
    raise ValueError(f"unknown unigaze preprocessing profile: {profile!r}")
