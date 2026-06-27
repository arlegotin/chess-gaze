from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from chess_gaze.gaze_observation import UNIGAZE_MODEL_ID
from chess_gaze.model_assets import ResolvedModelAsset
from chess_gaze.unigaze_runtime import prepare_unigaze_runtime


class RecordingUniGazeModel:
    def __init__(self) -> None:
        self.predict_batch_input_shapes: list[tuple[int, ...]] = []
        self.predict_batch_input_dtypes: list[torch.dtype] = []

    def predict_batch(self, normalized_batch: torch.Tensor) -> tuple[object, ...]:
        self.predict_batch_input_shapes.append(tuple(normalized_batch.shape))
        self.predict_batch_input_dtypes.append(normalized_batch.dtype)
        return ()


def _asset(tmp_path: Path) -> ResolvedModelAsset:
    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"fake unigaze weights")
    return ResolvedModelAsset(
        model_id=UNIGAZE_MODEL_ID,
        task_name="gaze_estimation",
        resolved_path=asset_path,
        source_url="https://huggingface.co/UniGaze/UniGaze-models",
        checksum_sha256="abc123",
        license="MG-NC-RAI-2.0",
    )


def test_prepare_unigaze_runtime_cpu_loads_model_without_dummy_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chess_gaze import unigaze_runtime

    fake_model = RecordingUniGazeModel()
    loaded_devices: list[str] = []

    def fake_from_local_asset(
        asset: ResolvedModelAsset, *, device: str
    ) -> RecordingUniGazeModel:
        assert asset.model_id == UNIGAZE_MODEL_ID
        loaded_devices.append(device)
        return fake_model

    runtime_module = cast(Any, unigaze_runtime)
    monkeypatch.setattr(
        runtime_module.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(
        runtime_module.UniGazeModel,
        "from_local_asset",
        staticmethod(fake_from_local_asset),
    )

    prepared = prepare_unigaze_runtime(
        _asset(tmp_path),
        device="cpu",
        batch_size=5,
        input_size_px=224,
    )

    assert loaded_devices == ["cpu"]
    assert cast(object, prepared.model) is fake_model
    assert fake_model.predict_batch_input_shapes == []
    assert prepared.inference.unigaze_device == "cpu"
    assert prepared.inference.unigaze_batch_size == 5
    assert prepared.inference.torch_mps_available is True
    assert prepared.inference.mps_preflight_passed is None


def test_prepare_unigaze_runtime_mps_preflights_requested_batch_and_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chess_gaze import unigaze_runtime

    fake_model = RecordingUniGazeModel()
    loaded_devices: list[str] = []
    sync_calls: list[None] = []

    def fake_from_local_asset(
        asset: ResolvedModelAsset, *, device: str
    ) -> RecordingUniGazeModel:
        assert asset.model_id == UNIGAZE_MODEL_ID
        loaded_devices.append(device)
        return fake_model

    def fake_synchronize() -> None:
        sync_calls.append(None)

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_PREFER_METAL", raising=False)
    runtime_module = cast(Any, unigaze_runtime)
    monkeypatch.setattr(
        runtime_module.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(
        runtime_module.torch,
        "mps",
        SimpleNamespace(synchronize=fake_synchronize),
    )
    monkeypatch.setattr(
        runtime_module.UniGazeModel,
        "from_local_asset",
        staticmethod(fake_from_local_asset),
    )

    prepared = prepare_unigaze_runtime(
        _asset(tmp_path),
        device="mps",
        batch_size=7,
        input_size_px=96,
    )

    assert loaded_devices == ["mps"]
    assert cast(object, prepared.model) is fake_model
    assert fake_model.predict_batch_input_shapes == [(7, 3, 96, 96)]
    assert fake_model.predict_batch_input_dtypes == [torch.float32]
    assert sync_calls == [None]
    assert prepared.inference.unigaze_device == "mps"
    assert prepared.inference.unigaze_batch_size == 7
    assert prepared.inference.torch_mps_available is True
    assert prepared.inference.mps_fallback_env == "unset"
    assert prepared.inference.mps_fast_math_env == "unset"
    assert prepared.inference.mps_preflight_passed is True
