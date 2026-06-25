from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError


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
        return AnalysisConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


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
