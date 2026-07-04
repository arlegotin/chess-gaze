from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_gaze.errors import ErrorCode
from chess_gaze.gaze_observation import (
    UniGazeModel,
    normalize_face_crop,
    pitch_yaw_to_unit_vector,
)
from chess_gaze.geometry import BBox, CoordinateSpace
from chess_gaze.model_assets import ResolvedModelAsset


class FakeUniGazeBackend:
    def __init__(self) -> None:
        self.loaded_path: str | None = None
        self.device: str | None = None
        self.eval_called = False

    def load_unigaze_weights(self, path: str) -> None:
        self.loaded_path = path

    def to(self, device: str) -> FakeUniGazeBackend:
        self.device = device
        return self

    def eval(self) -> FakeUniGazeBackend:
        self.eval_called = True
        return self

    def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        assert batch.ndim == 4
        assert batch.shape[1:] == (3, 224, 224)
        rows = []
        for index in range(batch.shape[0]):
            rows.append([0.125 + index, -0.25 - index])
        return {
            "pred_gaze": torch.tensor(rows, dtype=torch.float32, device=batch.device)
        }


def _asset(path: Path) -> ResolvedModelAsset:
    return ResolvedModelAsset(
        model_id="unigaze-h14-joint",
        task_name="gaze_estimation",
        resolved_path=path,
        source_url="https://huggingface.co/UniGaze/UniGaze-models",
        checksum_sha256="abc123",
        license="MG-NC-RAI-2.0",
    )


def test_unigaze_model_loads_local_asset_without_download_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    fake_backend = FakeUniGazeBackend()
    observed_offline_env: list[str | None] = []

    huggingface_hub = importlib.import_module("huggingface_hub")
    unigaze = importlib.import_module("unigaze")
    unigaze_loader = importlib.import_module("unigaze.loader")

    def fail_network_helper(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network helper must not be used")

    def fake_build(builder_key: str) -> FakeUniGazeBackend:
        observed_offline_env.append(__import__("os").environ.get("HF_HUB_OFFLINE"))
        assert builder_key == "unigaze_h14_joint"
        return fake_backend

    monkeypatch.setattr(unigaze, "load", fail_network_helper, raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_network_helper)
    monkeypatch.setattr(unigaze_loader, "build_unigaze_model", fake_build)

    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")
    gaze = model.predict(torch.zeros((1, 3, 224, 224), dtype=torch.float32))

    assert fake_backend.loaded_path == str(asset_path)
    assert fake_backend.device == "cpu"
    assert fake_backend.eval_called is True
    assert observed_offline_env == ["1"]
    assert gaze.method == "unigaze_h14_joint"
    assert gaze.pitch_radians == pytest.approx(0.125)
    assert gaze.yaw_radians == pytest.approx(0.25)
    assert gaze.confidence is None
    assert gaze.confidence_source == "not_provided_by_unigaze"
    assert gaze.unit_vector == pytest.approx(
        pitch_yaw_to_unit_vector(pitch_radians=0.125, yaw_radians=0.25)
    )


def test_from_local_asset_suppresses_backend_weight_load_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class NoisyBackend(FakeUniGazeBackend):
        def load_unigaze_weights(self, path: str) -> None:
            super().load_unigaze_weights(path)
            print(f"Loaded UniGaze pretrained weights from {path}")

    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: NoisyBackend()
    )

    UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    captured = capsys.readouterr()
    assert captured.out == ""


def test_unigaze_prediction_requires_documented_output_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadBackend(FakeUniGazeBackend):
        def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            del batch
            return {"pred_gaze": torch.zeros((1, 3), dtype=torch.float32)}

    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")

    unigaze_loader = importlib.import_module("unigaze.loader")

    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: BadBackend()
    )

    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    with pytest.raises(ValueError, match="pred_gaze"):
        model.predict(torch.zeros((1, 3, 224, 224), dtype=torch.float32))


def test_unigaze_predict_batch_maps_each_output_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    fake_backend = FakeUniGazeBackend()
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: fake_backend
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    gazes = model.predict_batch(torch.zeros((3, 3, 224, 224), dtype=torch.float32))

    assert [gaze.pitch_radians for gaze in gazes] == pytest.approx(
        [0.125, 1.125, 2.125]
    )
    assert [gaze.yaw_radians for gaze in gazes] == pytest.approx([0.25, 1.25, 2.25])
    assert all(gaze.valid for gaze in gazes)


def test_unigaze_predict_batch_rejects_empty_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: FakeUniGazeBackend()
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    with pytest.raises(ValueError, match="non-empty"):
        model.predict_batch(torch.zeros((0, 3, 224, 224), dtype=torch.float32))


def test_unigaze_predict_batch_rejects_output_row_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadBackend(FakeUniGazeBackend):
        def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            del batch
            return {"pred_gaze": torch.zeros((1, 2), dtype=torch.float32)}

    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: BadBackend()
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    with pytest.raises(ValueError, match="shape"):
        model.predict_batch(torch.zeros((2, 3, 224, 224), dtype=torch.float32))


def test_unigaze_predict_batch_marks_non_finite_row_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NonFiniteBackend(FakeUniGazeBackend):
        def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            del batch
            return {
                "pred_gaze": torch.tensor(
                    [[0.1, -0.2], [float("nan"), -0.3]], dtype=torch.float32
                )
            }

    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: NonFiniteBackend()
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    valid_gaze, invalid_gaze = model.predict_batch(
        torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    )

    assert valid_gaze.valid is True
    assert invalid_gaze.valid is False
    assert invalid_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED


def test_unigaze_predict_batch_moves_input_to_model_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DeviceSpyBackend(FakeUniGazeBackend):
        observed_device: torch.device | None = None

        def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            self.observed_device = batch.device
            return {"pred_gaze": torch.tensor([[0.1, -0.2]], dtype=torch.float32)}

    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    backend = DeviceSpyBackend()
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(unigaze_loader, "build_unigaze_model", lambda _key: backend)

    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")
    model.predict_batch(torch.zeros((1, 3, 224, 224), dtype=torch.float32))

    assert backend.observed_device == torch.device("cpu")


def test_normalize_face_crop_records_transform_and_returns_chw_tensor() -> None:
    rgb_frame = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb_frame[10:30, 20:50] = 128
    bbox = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=20.0,
        y_min=10.0,
        x_max=50.0,
        y_max=30.0,
    )

    normalized = normalize_face_crop(rgb_frame, bbox, input_size_px=224)

    assert normalized.tensor.shape == (1, 3, 224, 224)
    assert normalized.tensor.dtype == torch.float32
    assert normalized.transform.source_bbox_image_px == bbox
    assert normalized.transform.output_size_px == 224
    assert normalized.transform.image_px_from_crop_px.m00 == pytest.approx(30.0 / 224.0)
    assert normalized.transform.image_px_from_crop_px.m11 == pytest.approx(20.0 / 224.0)


def test_gaze_observation_no_longer_exports_pupil_geometric_helpers() -> None:
    import chess_gaze.gaze_observation as gaze_observation

    assert not hasattr(gaze_observation, "compute_per_eye_geometric_gaze")
    assert not hasattr(gaze_observation, "synthesize_recommended_gaze")
    assert not hasattr(gaze_observation, "GazeThresholds")
