from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigurationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: Path = Path("artifacts/output")
    models_root: Path = Path("models")
    raw_frame_image_format: str = "png"
    processed_frame_image_format: str = "jpg"
    processed_frame_jpeg_quality: int = 95


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
