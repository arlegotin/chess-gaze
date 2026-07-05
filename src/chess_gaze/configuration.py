from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from chess_gaze.unigaze_preprocessing import (
    DEFAULT_UNIGAZE_PREPROCESSING_PROFILE,
    UniGazePreprocessingProfile,
)


class ConfigurationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TargetPlaneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin_camera_m: tuple[float, float, float]
    x_axis_camera: tuple[float, float, float]
    y_axis_camera: tuple[float, float, float]
    width_m: float
    height_m: float
    mirror_horizontal: bool = False

    @field_validator("origin_camera_m", "x_axis_camera", "y_axis_camera", mode="before")
    @classmethod
    def coerce_triplet(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("origin_camera_m", "x_axis_camera", "y_axis_camera")
    @classmethod
    def validate_triplet(
        cls, value: tuple[float, float, float], info: object
    ) -> tuple[float, float, float]:
        field_name = getattr(info, "field_name", "target_plane_triplet")
        if len(value) != 3:
            raise ValueError(f"{field_name} must contain exactly three values")
        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError(f"{field_name} must contain only finite values")
        return value

    @field_validator("width_m", "height_m")
    @classmethod
    def validate_positive_meters(cls, value: float, info: object) -> float:
        field_name = getattr(info, "field_name", "target_plane_size")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name} must be positive and finite")
        return value


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: Path = Path("artifacts/output")
    models_root: Path = Path("models")
    raw_frame_image_format: str = "png"
    processed_frame_image_format: str = "jpg"
    processed_frame_jpeg_quality: int = 95
    save_frame_images: bool = False
    save_crop_images: bool = False
    unigaze_device: Literal["cpu", "mps"] = "mps"
    unigaze_batch_size: int = 7
    unigaze_preprocessing_profile: UniGazePreprocessingProfile = (
        DEFAULT_UNIGAZE_PREPROCESSING_PROFILE
    )
    target_plane: TargetPlaneConfig | None = None

    @field_validator("unigaze_batch_size")
    @classmethod
    def validate_unigaze_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("unigaze_batch_size must be at least 1")
        return value


def apply_analysis_overrides(
    config: AnalysisConfig,
    *,
    output_root: Path | None = None,
    models_root: Path | None = None,
    unigaze_device: str | None = None,
    unigaze_batch_size: int | None = None,
    unigaze_preprocessing_profile: str | None = None,
    save_frame_images: bool | None = None,
    save_crop_images: bool | None = None,
) -> AnalysisConfig:
    payload = config.model_dump(mode="python")

    if output_root is not None:
        payload["output_root"] = output_root
    if models_root is not None:
        payload["models_root"] = models_root
    if unigaze_device is not None:
        payload["unigaze_device"] = unigaze_device
    if unigaze_batch_size is not None:
        payload["unigaze_batch_size"] = unigaze_batch_size
    if unigaze_preprocessing_profile is not None:
        payload["unigaze_preprocessing_profile"] = unigaze_preprocessing_profile
    if save_frame_images is not None:
        payload["save_frame_images"] = save_frame_images
    if save_crop_images is not None:
        payload["save_crop_images"] = save_crop_images

    return AnalysisConfig.model_validate(payload)


def load_config(path: Path | None) -> AnalysisConfig:
    if path is None:
        return AnalysisConfig()

    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(
            "CONFIG_PATH_MISSING",
            f"Configuration file is missing: {path}",
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            "CONFIG_LOAD_UNREADABLE",
            f"Configuration file is unreadable: {path}",
        ) from exc

    try:
        payload = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            "CONFIG_LOAD_INVALID",
            f"Configuration file must contain valid JSON: {path}",
        ) from exc

    if not isinstance(payload, dict):
        raise ConfigurationError(
            "CONFIG_LOAD_UNSUPPORTED_SHAPE",
            f"Configuration file must contain a JSON object: {path}",
        )

    try:
        return AnalysisConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigurationError("CONFIG_LOAD_INVALID", str(exc)) from exc


def load_env_file(path: Path = Path(".env")) -> dict[str, str]:
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        values[key.strip()] = value.strip()
    return values
