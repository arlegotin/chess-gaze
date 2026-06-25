from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chess_gaze.errors import CliErrorCode
from chess_gaze.model_assets import (
    ModelAssetError,
    load_model_registry,
    prefetch_model_asset,
    validate_required_assets,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_registry(tmp_path: Path, *, unigaze_checksum: str | None) -> Path:
    registry_path = tmp_path / "model_registry.json"
    write_json(
        registry_path,
        {
            "models": [
                {
                    "model_id": "mediapipe-face-landmarker",
                    "task_name": "face_landmarks",
                    "expected_relative_path": "mediapipe/face_landmarker.task",
                    "checksum_sha256": None,
                    "source_url": "https://example.invalid/mediapipe",
                    "license": "Google AI Edge Terms",
                    "license_approved": True,
                    "license_approved_by": "repo_owner",
                    "license_approved_at": "2026-06-25",
                    "input_contract": {"image_mode": "IMAGE"},
                    "output_contract": {"landmarks": "face mesh"},
                },
                {
                    "model_id": "unigaze-h14-joint",
                    "task_name": "gaze_estimation",
                    "expected_relative_path": "unigaze/unigaze_h14_joint.safetensors",
                    "checksum_sha256": unigaze_checksum,
                    "source_url": "https://example.invalid/unigaze",
                    "license": "MG-NC-RAI-2.0",
                    "license_approved": True,
                    "license_approved_by": "repo_owner",
                    "license_approved_at": "2026-06-25",
                    "input_contract": {"input_size_px": 224},
                    "output_contract": {"order": "pitch_yaw_radians"},
                },
            ]
        },
    )
    return registry_path


def test_registry_file_records_unigaze_license_approval_metadata() -> None:
    registry_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "chess_gaze"
        / "model_registry.json"
    )
    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))

    unigaze_entry = next(
        model
        for model in registry_data["models"]
        if model["model_id"] == "unigaze-h14-joint"
    )

    assert unigaze_entry["license"] == "MG-NC-RAI-2.0"
    assert unigaze_entry["license_approved"] is True
    assert unigaze_entry["license_approved_by"] == "repo_owner"
    assert unigaze_entry["license_approved_at"] == "2026-06-25"


def test_manifest_cannot_add_uncommitted_registry_entry(tmp_path: Path) -> None:
    registry_path = build_registry(tmp_path, unigaze_checksum=None)
    models_root = tmp_path / "models"
    manifest_path = models_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    write_json(
        manifest_path,
        {
            "models": [
                {
                    "model_id": "mediapipe-face-landmarker",
                    "installed_relative_path": "mediapipe/face_landmarker.task",
                    "checksum_sha256": None,
                },
                {
                    "model_id": "surprise-model",
                    "installed_relative_path": "surprise/model.bin",
                    "checksum_sha256": "abc123",
                },
            ]
        },
    )

    registry = load_model_registry(registry_path)

    with pytest.raises(ModelAssetError) as exc_info:
        validate_required_assets(registry, models_root, {"MG-NC-RAI-2.0"})

    assert exc_info.value.code == CliErrorCode.MODEL_ASSET_MISSING
    assert "surprise-model" not in {
        asset.model_id for asset in exc_info.value.resolved_assets
    }


def test_validate_required_assets_raises_for_missing_asset(tmp_path: Path) -> None:
    registry_path = build_registry(tmp_path, unigaze_checksum=None)
    registry = load_model_registry(registry_path)

    with pytest.raises(ModelAssetError) as exc_info:
        validate_required_assets(registry, tmp_path / "models", {"MG-NC-RAI-2.0"})

    assert exc_info.value.code == CliErrorCode.MODEL_ASSET_MISSING
    assert "mediapipe-face-landmarker" in str(exc_info.value)


def test_validate_required_assets_raises_for_checksum_mismatch(tmp_path: Path) -> None:
    asset_bytes = b"fixture-unigaze-weights"
    registry_path = build_registry(tmp_path, unigaze_checksum=sha256_bytes(asset_bytes))
    models_root = tmp_path / "models"
    mediapipe_path = models_root / "mediapipe" / "face_landmarker.task"
    asset_path = models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    mediapipe_path.parent.mkdir(parents=True)
    asset_path.parent.mkdir(parents=True)
    mediapipe_path.write_bytes(b"mediapipe")
    asset_path.write_bytes(b"wrong-bytes")

    registry = load_model_registry(registry_path)

    with pytest.raises(ModelAssetError) as exc_info:
        validate_required_assets(registry, models_root, {"MG-NC-RAI-2.0"})

    assert exc_info.value.code == CliErrorCode.MODEL_ASSET_CHECKSUM_MISMATCH
    assert "unigaze-h14-joint" in str(exc_info.value)


def test_validate_required_assets_raises_for_unapproved_license(tmp_path: Path) -> None:
    asset_bytes = b"fixture-unigaze-weights"
    registry_path = build_registry(tmp_path, unigaze_checksum=sha256_bytes(asset_bytes))
    models_root = tmp_path / "models"
    asset_path = models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(asset_bytes)
    mediapipe_path = models_root / "mediapipe" / "face_landmarker.task"
    mediapipe_path.parent.mkdir(parents=True)
    mediapipe_path.write_bytes(b"mediapipe")

    registry = load_model_registry(registry_path)

    with pytest.raises(ModelAssetError) as exc_info:
        validate_required_assets(registry, models_root, set())

    assert exc_info.value.code == CliErrorCode.MODEL_LICENSE_NOT_APPROVED
    assert "MG-NC-RAI-2.0" in str(exc_info.value)


def test_validate_required_assets_resolves_matching_fixture_assets(
    tmp_path: Path,
) -> None:
    unigaze_bytes = b"fixture-unigaze-weights"
    registry_path = build_registry(
        tmp_path, unigaze_checksum=sha256_bytes(unigaze_bytes)
    )
    models_root = tmp_path / "models"
    mediapipe_path = models_root / "mediapipe" / "face_landmarker.task"
    unigaze_path = models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    mediapipe_path.parent.mkdir(parents=True)
    unigaze_path.parent.mkdir(parents=True)
    mediapipe_path.write_bytes(b"mediapipe")
    unigaze_path.write_bytes(unigaze_bytes)

    registry = load_model_registry(registry_path)

    resolved_assets = validate_required_assets(registry, models_root, {"MG-NC-RAI-2.0"})

    assert {asset.model_id for asset in resolved_assets} == {
        "mediapipe-face-landmarker",
        "unigaze-h14-joint",
    }
    assert {asset.resolved_path for asset in resolved_assets} == {
        mediapipe_path,
        unigaze_path,
    }


def test_prefetch_model_asset_accepts_hf_token_without_analysis_dependency(
    tmp_path: Path,
) -> None:
    registry_path = build_registry(tmp_path, unigaze_checksum=None)
    registry = load_model_registry(registry_path)
    models_root = tmp_path / "models"

    with pytest.raises(ModelAssetError) as exc_info:
        validate_required_assets(registry, models_root, {"MG-NC-RAI-2.0"})

    assert exc_info.value.code == CliErrorCode.MODEL_ASSET_MISSING

    resolved = prefetch_model_asset(
        "unigaze-h14-joint",
        registry,
        models_root,
        hf_token="test-token",
    )

    assert resolved.model_id == "unigaze-h14-joint"
    assert resolved.resolved_path == (
        models_root / "unigaze" / "unigaze_h14_joint.safetensors"
    )
