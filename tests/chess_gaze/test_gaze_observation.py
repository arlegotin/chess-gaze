from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chess_gaze.errors import ErrorCode
from chess_gaze.frame_records import GazeAngles
from chess_gaze.gaze_observation import (
    FaceModelGaze,
    GazeThresholds,
    UniGazeModel,
    compute_per_eye_geometric_gaze,
    normalize_face_crop,
    pitch_yaw_to_unit_vector,
    synthesize_recommended_gaze,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
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
        assert batch.shape == (1, 3, 224, 224)
        return {"pred_gaze": torch.tensor([[0.125, -0.25]], dtype=torch.float32)}


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


def test_per_eye_geometric_gaze_uses_independent_eye_offsets() -> None:
    head_pose = SimpleNamespace(
        valid=True,
        yaw_radians=0.05,
        pitch_radians=-0.02,
    )
    left_eye = SimpleNamespace(
        present=True,
        normalized_iris_offset=Point2D(
            space=CoordinateSpace.NORMALIZED, x=-0.20, y=0.10
        ),
    )
    right_eye = SimpleNamespace(
        present=True,
        normalized_iris_offset=Point2D(
            space=CoordinateSpace.NORMALIZED, x=0.15, y=-0.05
        ),
    )

    left = compute_per_eye_geometric_gaze(left_eye, head_pose)
    right = compute_per_eye_geometric_gaze(right_eye, head_pose)

    assert left.valid is True
    assert right.valid is True
    assert left.yaw_radians != right.yaw_radians
    assert left.pitch_radians != right.pitch_radians
    assert left.yaw_radians == pytest.approx(-0.15)
    assert right.yaw_radians == pytest.approx(0.20)


def test_per_eye_geometric_gaze_accepts_eye_observation_offset_tuple() -> None:
    head_pose = SimpleNamespace(
        valid=True,
        yaw_radians=0.05,
        pitch_radians=-0.02,
    )
    eye = SimpleNamespace(
        present=True,
        normalized_iris_offset_xy=(-0.20, 0.10),
    )

    gaze = compute_per_eye_geometric_gaze(eye, head_pose)

    assert gaze.valid is True
    assert gaze.yaw_radians == pytest.approx(-0.15)
    assert gaze.pitch_radians == pytest.approx(-0.12)


def test_recommended_gaze_invalid_when_estimators_disagree() -> None:
    left = GazeAngles(
        valid=True, yaw_radians=0.0, pitch_radians=0.0, reason_invalid=None
    )
    right = GazeAngles(
        valid=True, yaw_radians=0.02, pitch_radians=0.0, reason_invalid=None
    )
    face = FaceModelGaze(
        valid=True,
        method="unigaze_h14_joint",
        pitch_radians=0.0,
        yaw_radians=0.40,
        unit_vector=pitch_yaw_to_unit_vector(pitch_radians=0.0, yaw_radians=0.40),
        confidence=None,
        confidence_source="not_provided_by_unigaze",
        reason_invalid=None,
    )

    recommended = synthesize_recommended_gaze(
        left,
        right,
        face,
        thresholds=GazeThresholds(max_pairwise_angle_delta_radians=0.10),
    )

    assert recommended.gaze.valid is False
    assert recommended.gaze.reason_invalid is ErrorCode.GAZE_ESTIMATORS_DISAGREE
    assert recommended.target_image_px is None
    assert recommended.target_board_norm is None
    assert recommended.target_square is None


def test_recommended_gaze_uses_unigaze_when_head_pose_blocks_geometric_gaze() -> None:
    left = GazeAngles(
        valid=False,
        yaw_radians=None,
        pitch_radians=None,
        reason_invalid=ErrorCode.HEAD_POSE_INVALID,
    )
    right = GazeAngles(
        valid=False,
        yaw_radians=None,
        pitch_radians=None,
        reason_invalid=ErrorCode.HEAD_POSE_INVALID,
    )
    face = FaceModelGaze(
        valid=True,
        method="unigaze_h14_joint",
        pitch_radians=-0.12,
        yaw_radians=0.03,
        unit_vector=pitch_yaw_to_unit_vector(pitch_radians=-0.12, yaw_radians=0.03),
        confidence=None,
        confidence_source="not_provided_by_unigaze",
        reason_invalid=None,
    )

    recommended = synthesize_recommended_gaze(
        left,
        right,
        face,
        thresholds=GazeThresholds(max_pairwise_angle_delta_radians=0.35),
    )

    assert recommended.gaze.valid is True
    assert recommended.gaze.reason_invalid is None
    assert recommended.gaze.pitch_radians == pytest.approx(-0.12)
    assert recommended.gaze.yaw_radians == pytest.approx(0.03)
    assert recommended.method == "appearance_only_unigaze_h14_joint"


def test_recommended_gaze_rejects_single_geometric_eye_without_appearance() -> None:
    left = GazeAngles(
        valid=True, yaw_radians=0.08, pitch_radians=-0.04, reason_invalid=None
    )
    right = GazeAngles(
        valid=False,
        yaw_radians=None,
        pitch_radians=None,
        reason_invalid=ErrorCode.RIGHT_EYE_NOT_FOUND,
    )
    face = FaceModelGaze(
        valid=False,
        method="unigaze_h14_joint",
        pitch_radians=None,
        yaw_radians=None,
        unit_vector=None,
        confidence=None,
        confidence_source="not_provided_by_unigaze",
        reason_invalid=ErrorCode.GAZE_MODEL_FAILED,
    )

    recommended = synthesize_recommended_gaze(
        left,
        right,
        face,
        thresholds=GazeThresholds(max_pairwise_angle_delta_radians=0.35),
    )

    assert recommended.gaze.valid is False
    assert recommended.gaze.reason_invalid is ErrorCode.RIGHT_EYE_NOT_FOUND


def test_recommended_gaze_prefers_non_model_reason_when_no_sources_are_valid() -> None:
    left = GazeAngles(
        valid=False,
        yaw_radians=None,
        pitch_radians=None,
        reason_invalid=ErrorCode.HEAD_POSE_INVALID,
    )
    right = GazeAngles(
        valid=False,
        yaw_radians=None,
        pitch_radians=None,
        reason_invalid=ErrorCode.HEAD_POSE_INVALID,
    )
    face = FaceModelGaze(
        valid=False,
        method="unigaze_h14_joint",
        pitch_radians=None,
        yaw_radians=None,
        unit_vector=None,
        confidence=None,
        confidence_source="not_provided_by_unigaze",
        reason_invalid=ErrorCode.GAZE_MODEL_FAILED,
    )

    recommended = synthesize_recommended_gaze(
        left,
        right,
        face,
        thresholds=GazeThresholds(max_pairwise_angle_delta_radians=0.35),
    )

    assert recommended.gaze.valid is False
    assert recommended.gaze.reason_invalid is ErrorCode.HEAD_POSE_INVALID


def test_recommended_gaze_rejects_large_unigaze_geometric_disagreement() -> None:
    left = GazeAngles(
        valid=True, yaw_radians=0.08, pitch_radians=-0.04, reason_invalid=None
    )
    right = GazeAngles(
        valid=True, yaw_radians=0.10, pitch_radians=-0.05, reason_invalid=None
    )
    face = FaceModelGaze(
        valid=True,
        method="unigaze_h14_joint",
        pitch_radians=0.65,
        yaw_radians=-0.80,
        unit_vector=pitch_yaw_to_unit_vector(pitch_radians=0.65, yaw_radians=-0.80),
        confidence=None,
        confidence_source="not_provided_by_unigaze",
        reason_invalid=None,
    )

    recommended = synthesize_recommended_gaze(
        left,
        right,
        face,
        thresholds=GazeThresholds(max_pairwise_angle_delta_radians=0.35),
    )

    assert recommended.gaze.valid is False
    assert recommended.gaze.reason_invalid is ErrorCode.GAZE_ESTIMATORS_DISAGREE


def test_recommended_gaze_averages_agreeing_estimators() -> None:
    left = GazeAngles(
        valid=True, yaw_radians=0.10, pitch_radians=-0.01, reason_invalid=None
    )
    right = GazeAngles(
        valid=True, yaw_radians=0.12, pitch_radians=0.00, reason_invalid=None
    )
    face = FaceModelGaze(
        valid=True,
        method="unigaze_h14_joint",
        pitch_radians=0.02,
        yaw_radians=0.11,
        unit_vector=pitch_yaw_to_unit_vector(pitch_radians=0.02, yaw_radians=0.11),
        confidence=None,
        confidence_source="not_provided_by_unigaze",
        reason_invalid=None,
    )

    recommended = synthesize_recommended_gaze(
        left,
        right,
        face,
        thresholds=GazeThresholds(max_pairwise_angle_delta_radians=0.10),
    )

    assert recommended.gaze.valid is True
    assert recommended.gaze.yaw_radians == pytest.approx(0.11)
    assert recommended.gaze.pitch_radians == pytest.approx(0.0033333333)
    assert recommended.target_image_px is None
    assert recommended.target_board_norm is None
    assert recommended.target_square is None
