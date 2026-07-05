from __future__ import annotations

import importlib
import io
import math
import os
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from chess_gaze.errors import ErrorCode
from chess_gaze.geometry import BBox, CoordinateSpace, Transform2D
from chess_gaze.model_assets import ResolvedModelAsset
from chess_gaze.unigaze_preprocessing import (
    DEFAULT_UNIGAZE_PREPROCESSING_PROFILE,
    resolve_unigaze_preprocessing_profile,
)

UNIGAZE_MODEL_ID = "unigaze-h14-joint"
UNIGAZE_BUILDER_KEY = "unigaze_h14_joint"
UNIGAZE_METHOD = "unigaze_h14_joint"
UNIGAZE_CONFIDENCE_SOURCE = "not_provided_by_unigaze"


@dataclass(frozen=True)
class CropTransformRecord:
    source_bbox_image_px: BBox
    output_size_px: int
    image_px_from_crop_px: Transform2D


@dataclass(frozen=True)
class NormalizedFaceCrop:
    tensor: torch.Tensor
    transform: CropTransformRecord


@dataclass(frozen=True)
class FaceModelGaze:
    valid: bool
    method: str
    pitch_radians: float | None
    yaw_radians: float | None
    unit_vector: tuple[float, float, float] | None
    confidence: float | None
    confidence_source: str
    reason_invalid: ErrorCode | None


class UniGazeModel:
    def __init__(self, backend: Any, *, device: str) -> None:
        self._backend = backend
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def from_local_asset(
        cls, asset: ResolvedModelAsset, *, device: str
    ) -> UniGazeModel:
        if asset.model_id != UNIGAZE_MODEL_ID:
            raise ValueError(
                f"Expected {UNIGAZE_MODEL_ID} asset, got {asset.model_id!r}"
            )
        if not asset.resolved_path.is_file():
            raise FileNotFoundError(asset.resolved_path)

        with _offline_huggingface_boundary():
            backend = _build_unigaze_backend(UNIGAZE_BUILDER_KEY)
            with redirect_stdout(io.StringIO()):
                backend.load_unigaze_weights(str(asset.resolved_path))
            backend = backend.to(device).eval()
        return cls(backend, device=device)

    def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze:
        gazes = self.predict_batch(normalized_batch)
        if len(gazes) != 1:
            raise ValueError("UniGaze predict() requires exactly one batch row")
        return gazes[0]

    def predict_batch(
        self, normalized_batch: torch.Tensor
    ) -> tuple[FaceModelGaze, ...]:
        if normalized_batch.ndim != 4 or normalized_batch.shape[1] != 3:
            raise ValueError("normalized_batch must have shape (batch, 3, H, W)")
        if normalized_batch.shape[0] < 1:
            raise ValueError("normalized_batch must contain a non-empty batch")

        normalized_batch = normalized_batch.to(self._device)
        with torch.inference_mode():
            output = self._backend(normalized_batch)

        pred_gaze = output.get("pred_gaze") if isinstance(output, dict) else None
        if not isinstance(pred_gaze, torch.Tensor):
            raise ValueError("UniGaze output must contain tensor pred_gaze")
        if (
            pred_gaze.ndim != 2
            or pred_gaze.shape[0] != normalized_batch.shape[0]
            or pred_gaze.shape[1] != 2
        ):
            raise ValueError(
                "UniGaze pred_gaze must have shape (batch, 2) matching input batch"
            )

        pred_gaze_cpu = pred_gaze.detach().cpu()
        return tuple(
            _face_model_gaze_from_pred_row(pred_gaze_cpu[index])
            for index in range(pred_gaze_cpu.shape[0])
        )


def normalize_face_crop(
    rgb_frame: npt.NDArray[np.uint8],
    bbox: BBox,
    *,
    input_size_px: int,
    profile: str = DEFAULT_UNIGAZE_PREPROCESSING_PROFILE,
    crop_scale: float | None = None,
    image_mean_rgb: tuple[float, float, float] | None = None,
    image_std_rgb: tuple[float, float, float] | None = None,
) -> NormalizedFaceCrop:
    frame = _validate_rgb_frame(rgb_frame)
    if bbox.space is not CoordinateSpace.IMAGE_PX:
        raise ValueError("face crop bbox must be in image_px")
    if input_size_px <= 0:
        raise ValueError("input_size_px must be positive")
    try:
        preprocessing = resolve_unigaze_preprocessing_profile(profile)
    except ValueError as exc:
        raise ValueError(f"unknown unigaze preprocessing profile: {profile!r}") from exc
    resolved_crop_scale = (
        preprocessing.crop_scale if crop_scale is None else crop_scale
    )
    resolved_image_mean_rgb = (
        preprocessing.image_mean_rgb
        if image_mean_rgb is None and preprocessing.image_mean_rgb is not None
        else image_mean_rgb
    )
    resolved_image_std_rgb = (
        preprocessing.image_std_rgb
        if image_std_rgb is None and preprocessing.image_std_rgb is not None
        else image_std_rgb
    )
    if resolved_crop_scale <= 0.0:
        raise ValueError("crop_scale must be positive")
    if (resolved_image_mean_rgb is None) != (resolved_image_std_rgb is None):
        raise ValueError("image_mean_rgb and image_std_rgb must both be set")
    if resolved_image_mean_rgb is not None:
        _validate_rgb_stats(resolved_image_mean_rgb, "image_mean_rgb")
    if resolved_image_std_rgb is not None:
        _validate_rgb_stats(resolved_image_std_rgb, "image_std_rgb")

    height_px, width_px, _channels = frame.shape
    scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max = _scaled_bbox_bounds(
        bbox, scale=resolved_crop_scale
    )
    x_min = _clamp_int(math.floor(scaled_x_min), 0, width_px - 1)
    y_min = _clamp_int(math.floor(scaled_y_min), 0, height_px - 1)
    x_max = _clamp_int(math.ceil(scaled_x_max), x_min + 1, width_px)
    y_max = _clamp_int(math.ceil(scaled_y_max), y_min + 1, height_px)

    crop = frame[y_min:y_max, x_min:x_max]
    resized = cv2.resize(
        crop, (input_size_px, input_size_px), interpolation=cv2.INTER_AREA
    )
    normalized = resized.astype(np.float32) / 255.0
    if resolved_image_mean_rgb is not None and resolved_image_std_rgb is not None:
        mean = np.asarray(resolved_image_mean_rgb, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(resolved_image_std_rgb, dtype=np.float32).reshape(1, 1, 3)
        normalized = (normalized - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)

    scale_x = (x_max - x_min) / float(input_size_px)
    scale_y = (y_max - y_min) / float(input_size_px)
    crop_bbox = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=float(x_min),
        y_min=float(y_min),
        x_max=float(x_max),
        y_max=float(y_max),
    )
    return NormalizedFaceCrop(
        tensor=tensor,
        transform=CropTransformRecord(
            source_bbox_image_px=crop_bbox,
            output_size_px=input_size_px,
            image_px_from_crop_px=Transform2D(
                source_space=CoordinateSpace.IMAGE_PX,
                target_space=CoordinateSpace.IMAGE_PX,
                m00=scale_x,
                m01=0.0,
                m02=float(x_min),
                m10=0.0,
                m11=scale_y,
                m12=float(y_min),
            ),
        ),
    )


def pitch_yaw_to_unit_vector(
    *, pitch_radians: float, yaw_radians: float
) -> tuple[float, float, float]:
    _require_finite(pitch_radians, "pitch_radians")
    _require_finite(yaw_radians, "yaw_radians")
    cos_pitch = math.cos(pitch_radians)
    return (
        cos_pitch * math.sin(yaw_radians),
        math.sin(pitch_radians),
        cos_pitch * math.cos(yaw_radians),
    )


def _face_model_gaze_from_pred_row(pred_row: torch.Tensor) -> FaceModelGaze:
    pitch_radians = float(pred_row[0])
    # UniGaze's reference drawing treats positive yaw as image-left. Frame
    # records and overlays use positive yaw as image-right.
    yaw_radians = -float(pred_row[1])
    if not math.isfinite(pitch_radians) or not math.isfinite(yaw_radians):
        return FaceModelGaze(
            valid=False,
            method=UNIGAZE_METHOD,
            pitch_radians=None,
            yaw_radians=None,
            unit_vector=None,
            confidence=None,
            confidence_source=UNIGAZE_CONFIDENCE_SOURCE,
            reason_invalid=ErrorCode.GAZE_MODEL_FAILED,
        )
    return FaceModelGaze(
        valid=True,
        method=UNIGAZE_METHOD,
        pitch_radians=pitch_radians,
        yaw_radians=yaw_radians,
        unit_vector=pitch_yaw_to_unit_vector(
            pitch_radians=pitch_radians, yaw_radians=yaw_radians
        ),
        confidence=None,
        confidence_source=UNIGAZE_CONFIDENCE_SOURCE,
        reason_invalid=None,
    )


def _build_unigaze_backend(builder_key: str) -> Any:
    loader = importlib.import_module("unigaze.loader")
    return loader.build_unigaze_model(builder_key)


@contextmanager
def _offline_huggingface_boundary() -> Any:
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def _validate_rgb_frame(
    rgb_frame: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError("rgb_frame must have shape (height, width, 3)")
    if rgb_frame.dtype != np.uint8:
        raise ValueError("rgb_frame must have dtype uint8")
    return np.ascontiguousarray(rgb_frame)


def _scaled_bbox_bounds(
    bbox: BBox, *, scale: float
) -> tuple[float, float, float, float]:
    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min
    center_x = bbox.x_min + width / 2.0
    center_y = bbox.y_min + height / 2.0
    scaled_width = width * scale
    scaled_height = height * scale
    return (
        center_x - (scaled_width / 2.0),
        center_y - (scaled_height / 2.0),
        center_x + (scaled_width / 2.0),
        center_y + (scaled_height / 2.0),
    )


def _validate_rgb_stats(
    values: tuple[float, float, float], field_name: str
) -> None:
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain three RGB values")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} values must be finite")
        if field_name == "image_std_rgb" and value <= 0.0:
            raise ValueError("image_std_rgb values must be positive")


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return min(max(value, lower), upper)


def _require_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
