from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from chess_gaze.errors import CliErrorCode


class ModelRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    task_name: str
    expected_relative_path: Path
    checksum_sha256: str | None
    source_url: str
    license: str
    requires_license_approval: bool
    license_approved: bool
    license_approved_by: str | None
    license_approved_at: str | None
    input_contract: dict[str, object]
    output_contract: dict[str, object]


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelRegistryEntry]

    def by_id(self, model_id: str) -> ModelRegistryEntry:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise ModelAssetError(
            CliErrorCode.MODEL_ASSET_MISSING,
            f"Model registry entry not found: {model_id}",
        )


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    installed_relative_path: Path
    checksum_sha256: str | None = None


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ManifestEntry] = Field(default_factory=list)


class ResolvedModelAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    task_name: str
    resolved_path: Path
    source_url: str
    checksum_sha256: str | None
    license: str


class ModelAssetError(RuntimeError):
    def __init__(
        self,
        code: CliErrorCode,
        message: str,
        *,
        resolved_assets: list[ResolvedModelAsset] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.resolved_assets = resolved_assets or []


def load_model_registry(path: Path) -> ModelRegistry:
    return ModelRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def load_model_manifest(path: Path) -> ModelManifest:
    if not path.is_file():
        return ModelManifest()
    return ModelManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_required_assets(
    registry: ModelRegistry,
    models_root: Path,
    approved_licenses: set[str],
) -> list[ResolvedModelAsset]:
    manifest = load_model_manifest(models_root / "manifest.json")
    manifest_paths = {
        entry.model_id: entry.installed_relative_path
        for entry in manifest.models
        if any(model.model_id == entry.model_id for model in registry.models)
    }

    resolved_assets: list[ResolvedModelAsset] = []
    for entry in registry.models:
        if entry.requires_license_approval and (
            not entry.license_approved
            or entry.license_approved_by is None
            or entry.license_approved_at is None
            or entry.license not in approved_licenses
        ):
            raise ModelAssetError(
                CliErrorCode.MODEL_LICENSE_NOT_APPROVED,
                f"Model license not approved for {entry.model_id}: {entry.license}",
                resolved_assets=resolved_assets,
            )

        relative_path = manifest_paths.get(entry.model_id, entry.expected_relative_path)
        resolved_path = models_root / relative_path
        if not resolved_path.is_file():
            raise ModelAssetError(
                CliErrorCode.MODEL_ASSET_MISSING,
                f"Required model asset missing for {entry.model_id}: {resolved_path}",
                resolved_assets=resolved_assets,
            )

        if entry.checksum_sha256 is not None:
            actual_checksum = sha256_file(resolved_path)
            if actual_checksum != entry.checksum_sha256:
                raise ModelAssetError(
                    CliErrorCode.MODEL_ASSET_CHECKSUM_MISMATCH,
                    (
                        "Model asset checksum mismatch for "
                        f"{entry.model_id}: expected "
                        f"{entry.checksum_sha256}, got {actual_checksum}"
                    ),
                    resolved_assets=resolved_assets,
                )

        resolved_assets.append(
            ResolvedModelAsset(
                model_id=entry.model_id,
                task_name=entry.task_name,
                resolved_path=resolved_path,
                source_url=entry.source_url,
                checksum_sha256=entry.checksum_sha256,
                license=entry.license,
            )
        )

    return resolved_assets


def prefetch_model_asset(
    model_id: str,
    registry: ModelRegistry,
    models_root: Path,
    hf_token: str | None,
) -> ResolvedModelAsset:
    del hf_token

    entry = registry.by_id(model_id)
    return ResolvedModelAsset(
        model_id=entry.model_id,
        task_name=entry.task_name,
        resolved_path=models_root / entry.expected_relative_path,
        source_url=entry.source_url,
        checksum_sha256=entry.checksum_sha256,
        license=entry.license,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8192):
            digest.update(chunk)
    return digest.hexdigest()
