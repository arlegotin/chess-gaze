from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch

from chess_gaze.frame_records import InferenceRuntimeRecord
from chess_gaze.gaze_observation import UNIGAZE_MODEL_ID, UniGazeModel
from chess_gaze.model_assets import ResolvedModelAsset

UniGazeDevice = Literal["cpu", "mps"]


class UniGazeRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedUniGazeRuntime:
    model: UniGazeModel
    inference: InferenceRuntimeRecord


def prepare_unigaze_runtime(
    asset: ResolvedModelAsset,
    *,
    device: UniGazeDevice,
    batch_size: int,
    input_size_px: int,
) -> PreparedUniGazeRuntime:
    if batch_size < 1:
        raise UniGazeRuntimeError("unigaze_batch_size must be >= 1")
    if device not in {"cpu", "mps"}:
        raise UniGazeRuntimeError(f"Unsupported UniGaze device: {device}")
    if asset.model_id != UNIGAZE_MODEL_ID:
        raise UniGazeRuntimeError(
            f"Expected {UNIGAZE_MODEL_ID} asset, got {asset.model_id!r}"
        )

    torch_mps_available = _mps_available()
    if device == "mps":
        _validate_mps_request(torch_mps_available)

    try:
        model = UniGazeModel.from_local_asset(asset, device=device)
    except Exception as exc:
        raise UniGazeRuntimeError(str(exc)) from exc

    mps_preflight_passed: bool | None = None
    if device == "mps":
        try:
            dummy_batch = torch.zeros(
                (batch_size, 3, input_size_px, input_size_px),
                dtype=torch.float32,
            )
            model.predict_batch(dummy_batch)
            synchronize_if_needed(device)
        except Exception as exc:
            raise UniGazeRuntimeError(str(exc)) from exc
        mps_preflight_passed = True

    return PreparedUniGazeRuntime(
        model=model,
        inference=InferenceRuntimeRecord(
            observer_source="default_model_observer",
            unigaze_model_id=asset.model_id,
            unigaze_model_checksum_sha256=asset.checksum_sha256,
            unigaze_device=device,
            unigaze_batch_size=batch_size,
            torch_version=torch.__version__,
            torch_mps_available=torch_mps_available,
            mps_fallback_env=_env_state("PYTORCH_ENABLE_MPS_FALLBACK"),
            mps_fast_math_env=_env_state("PYTORCH_MPS_FAST_MATH"),
            mps_prefer_metal_env=_env_state("PYTORCH_MPS_PREFER_METAL"),
            mps_preflight_passed=mps_preflight_passed,
        ),
    )


def external_observer_inference_record() -> InferenceRuntimeRecord:
    return InferenceRuntimeRecord(
        observer_source="external_observer",
        unigaze_model_id=None,
        unigaze_model_checksum_sha256=None,
        unigaze_device="not_applicable",
        unigaze_batch_size=None,
        torch_version=None,
        torch_mps_available=None,
        mps_fallback_env="not_applicable",
        mps_fast_math_env="not_applicable",
        mps_prefer_metal_env="not_applicable",
        mps_preflight_passed=None,
    )


def synchronize_if_needed(device: str) -> None:
    if torch.device(device).type == "mps":
        torch.mps.synchronize()


def _validate_mps_request(torch_mps_available: bool) -> None:
    if not torch_mps_available:
        raise UniGazeRuntimeError(
            "UniGaze device=mps requested, but MPS is unavailable"
        )

    for env_name in (
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_MPS_PREFER_METAL",
    ):
        if _env_enabled(env_name):
            raise UniGazeRuntimeError(
                f"UniGaze device=mps rejects enabled {env_name}={_env_state(env_name)}"
            )


def _mps_available() -> bool:
    torch_mps_backend = getattr(torch.backends, "mps", None)
    return bool(torch_mps_backend is not None and torch_mps_backend.is_available())


def _env_state(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return "unset"
    return value


def _env_enabled(name: str) -> bool:
    return _env_state(name).lower() not in {"unset", "0", "false", "no"}
