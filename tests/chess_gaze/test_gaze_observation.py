from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

import chess_gaze.gaze_observation as gaze_observation
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


def test_camera_gaze_conversion_composes_model_and_physical_ray_signs() -> None:
    identity = np.eye(3, dtype=np.float64)

    centre = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=0.0,
        yaw_radians=0.0,
        camera_from_normalized_rotation=identity,
    )
    positive_physical_x = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=0.0,
        yaw_radians=-0.2,
        camera_from_normalized_rotation=identity,
    )
    positive_model_x = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=0.0,
        yaw_radians=0.2,
        camera_from_normalized_rotation=identity,
    )
    positive_physical_image_up = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=0.15,
        yaw_radians=0.0,
        camera_from_normalized_rotation=identity,
    )

    assert centre.valid is True
    assert centre.pitch_radians == pytest.approx(0.0)
    assert centre.yaw_radians == pytest.approx(0.0)
    assert positive_physical_x.yaw_radians == pytest.approx(0.2)
    assert positive_physical_x.unit_vector is not None
    assert positive_physical_x.unit_vector[0] > 0.0
    assert positive_model_x.yaw_radians == pytest.approx(-0.2)
    assert positive_physical_image_up.pitch_radians == pytest.approx(0.15)
    assert positive_physical_image_up.unit_vector is not None
    assert positive_physical_image_up.unit_vector[1] > 0.0


def test_camera_gaze_conversion_applies_known_yaw_rotation() -> None:
    yaw = np.pi / 6.0
    camera_from_normalized = np.asarray(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )

    gaze = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=0.0,
        yaw_radians=0.0,
        camera_from_normalized_rotation=camera_from_normalized,
    )

    assert gaze.pitch_radians == pytest.approx(0.0)
    assert gaze.yaw_radians == pytest.approx(-yaw)


def test_camera_gaze_conversion_matches_pinned_inverse_rotation_oracle() -> None:
    normalized_from_camera = np.asarray(
        [
            [0.9992439432689036, 0.038828770502339856, -0.001966830365993612],
            [-0.03875766751308453, 0.9988477511771979, 0.028302176191941107],
            [0.003063502792093412, -0.028204548383746697, 0.9995974781886515],
        ],
        dtype=np.float64,
    )

    gaze = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=0.125,
        yaw_radians=-0.25,
        camera_from_normalized_rotation=np.linalg.inv(normalized_from_camera),
    )

    assert gaze.pitch_radians == pytest.approx(0.08799865007484307)
    assert gaze.yaw_radians == pytest.approx(0.250754738938695)
    assert gaze.unit_vector == pytest.approx(
        (0.2471750345974333, 0.08788512060118626, 0.9649770504259014)
    )


def test_camera_gaze_conversion_reconstructs_negative_model_vector_as_scene_ray() -> (
    None
):
    camera_from_normalized = np.asarray(
        [
            [0.98, 0.0, 0.2],
            [0.0, 1.0, 0.0],
            [-0.2, 0.0, 0.98],
        ],
        dtype=np.float64,
    )
    raw_pitch = 0.12
    raw_yaw = -0.22

    gaze = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=raw_pitch,
        yaw_radians=raw_yaw,
        camera_from_normalized_rotation=camera_from_normalized,
    )

    model_vector_camera = camera_from_normalized @ np.asarray(
        pitch_yaw_to_unit_vector(
            pitch_radians=raw_pitch,
            yaw_radians=raw_yaw,
        )
    )
    model_vector_camera /= np.linalg.norm(model_vector_camera)
    assert gaze.unit_vector is not None
    repository_x, repository_image_up_y, repository_z = gaze.unit_vector
    scene_camera_ray = np.asarray((repository_x, -repository_image_up_y, -repository_z))
    np.testing.assert_allclose(scene_camera_ray, -model_vector_camera, atol=1e-12)


def test_camera_gaze_conversion_is_horizontally_flip_equivariant() -> None:
    positive = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=0.1,
        yaw_radians=0.35,
        camera_from_normalized_rotation=np.eye(3),
    )
    negative = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=0.1,
        yaw_radians=-0.35,
        camera_from_normalized_rotation=np.eye(3),
    )

    assert positive.yaw_radians is not None
    assert negative.yaw_radians is not None
    assert positive.pitch_radians is not None
    assert negative.pitch_radians is not None
    assert positive.yaw_radians == pytest.approx(-negative.yaw_radians)
    assert positive.pitch_radians == pytest.approx(negative.pitch_radians)
    assert positive.unit_vector is not None
    assert negative.unit_vector is not None
    assert positive.unit_vector[0] == pytest.approx(-negative.unit_vector[0])
    assert positive.unit_vector[1:] == pytest.approx(negative.unit_vector[1:])


def test_unigaze_predict_batch_uses_row_aligned_inverse_rotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: FakeUniGazeBackend()
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")
    yaw = 0.3
    yaw_rotation = np.asarray(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )

    first, second = model.predict_batch(
        torch.zeros((2, 3, 224, 224), dtype=torch.float32),
        camera_from_normalized_rotations=(np.eye(3), yaw_rotation),
    )

    assert first.yaw_radians == pytest.approx(0.25)
    expected_second = gaze_observation.camera_gaze_from_unigaze_prediction(
        pitch_radians=1.125,
        yaw_radians=-1.25,
        camera_from_normalized_rotation=yaw_rotation,
    )
    assert second.pitch_radians == pytest.approx(expected_second.pitch_radians)
    assert second.yaw_radians == pytest.approx(expected_second.yaw_radians)


def test_unigaze_predict_batch_rejects_misaligned_inverse_rotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: FakeUniGazeBackend()
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    with pytest.raises(ValueError, match="one inverse rotation per batch row"):
        model.predict_batch(
            torch.zeros((2, 3, 224, 224), dtype=torch.float32),
            camera_from_normalized_rotations=(np.eye(3),),
        )


def test_camera_gaze_conversion_rejects_invalid_inverse_rotation() -> None:
    with pytest.raises(ValueError, match="camera_from_normalized_rotation"):
        gaze_observation.camera_gaze_from_unigaze_prediction(
            pitch_radians=0.0,
            yaw_radians=0.0,
            camera_from_normalized_rotation=np.zeros((2, 2)),
        )

    invalid = np.eye(3)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="camera_from_normalized_rotation"):
        gaze_observation.camera_gaze_from_unigaze_prediction(
            pitch_radians=0.0,
            yaw_radians=0.0,
            camera_from_normalized_rotation=invalid,
        )


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

    normalized = normalize_face_crop(
        rgb_frame,
        bbox,
        input_size_px=224,
        profile="legacy_bbox_rgb01",
        crop_scale=1.0,
        image_mean_rgb=None,
        image_std_rgb=None,
    )

    assert normalized.tensor.shape == (1, 3, 224, 224)
    assert normalized.tensor.dtype == torch.float32
    assert normalized.transform.source_bbox_image_px == bbox
    assert normalized.transform.output_size_px == 224
    crop_transform = normalized.transform.image_px_from_crop_px
    assert crop_transform is not None
    assert crop_transform.m00 == pytest.approx(30.0 / 224.0)
    assert crop_transform.m11 == pytest.approx(20.0 / 224.0)


def test_reference_unigaze_preprocessing_expands_crop_and_uses_imagenet_transform() -> (
    None
):
    rgb_frame = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb_frame[:, :] = np.array([255, 128, 0], dtype=np.uint8)
    bbox = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=30.0,
        y_min=20.0,
        x_max=50.0,
        y_max=40.0,
    )

    normalized = normalize_face_crop(
        rgb_frame,
        bbox,
        input_size_px=224,
        profile="reference_face2x_imagenet",
        crop_scale=2.0,
        image_mean_rgb=(0.485, 0.456, 0.406),
        image_std_rgb=(0.229, 0.224, 0.225),
    )

    assert normalized.transform.source_bbox_image_px.x_min == pytest.approx(20.0)
    assert normalized.transform.source_bbox_image_px.y_min == pytest.approx(10.0)
    assert normalized.transform.source_bbox_image_px.x_max == pytest.approx(60.0)
    assert normalized.transform.source_bbox_image_px.y_max == pytest.approx(50.0)
    crop_transform = normalized.transform.image_px_from_crop_px
    assert crop_transform is not None
    assert crop_transform.m00 == pytest.approx(40.0 / 224.0)
    assert crop_transform.m11 == pytest.approx(40.0 / 224.0)
    assert float(normalized.tensor[0, 0, 0, 0]) == pytest.approx((1.0 - 0.485) / 0.229)
    assert float(normalized.tensor[0, 1, 0, 0]) == pytest.approx(
        ((128.0 / 255.0) - 0.456) / 0.224
    )
    assert float(normalized.tensor[0, 2, 0, 0]) == pytest.approx((0.0 - 0.406) / 0.225)


def test_default_unigaze_preprocessing_requires_official_geometry_inputs() -> None:
    rgb_frame = np.zeros((80, 100, 3), dtype=np.uint8)
    bbox = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=30.0,
        y_min=20.0,
        x_max=50.0,
        y_max=40.0,
    )

    with pytest.raises(
        ValueError,
        match="official_geometric_v1 requires face landmarks and face model points",
    ):
        normalize_face_crop(rgb_frame, bbox, input_size_px=224)


def test_normalize_face_crop_rejects_unknown_preprocessing_profile() -> None:
    rgb_frame = np.zeros((40, 60, 3), dtype=np.uint8)
    bbox = BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=20.0,
        y_min=10.0,
        x_max=50.0,
        y_max=30.0,
    )

    with pytest.raises(ValueError, match="unigaze preprocessing profile"):
        normalize_face_crop(
            rgb_frame,
            bbox,
            input_size_px=224,
            profile="unknown",
            crop_scale=1.0,
            image_mean_rgb=None,
            image_std_rgb=None,
        )


def test_gaze_observation_no_longer_exports_pupil_geometric_helpers() -> None:
    import chess_gaze.gaze_observation as gaze_observation

    assert not hasattr(gaze_observation, "compute_per_eye_geometric_gaze")
    assert not hasattr(gaze_observation, "synthesize_recommended_gaze")
    assert not hasattr(gaze_observation, "GazeThresholds")
