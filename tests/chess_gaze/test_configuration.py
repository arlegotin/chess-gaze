from __future__ import annotations

from pathlib import Path

import pytest

from chess_gaze.configuration import load_config, load_env_file


def test_load_config_uses_task_4_defaults(tmp_path: Path) -> None:
    config = load_config(None)

    assert config.output_root == Path("artifacts/output")
    assert config.models_root == Path("models")
    assert config.raw_frame_image_format == "png"
    assert config.processed_frame_image_format == "jpg"
    assert config.processed_frame_jpeg_quality == 95


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"output_root": "custom-output", "unexpected_key": "value"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected_key"):
        load_config(config_path)


def test_load_env_file_reads_simple_key_value_pairs(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# ignored comment",
                "HF_TOKEN=test-token",
                "MODELS_ROOT=models",
                "",
            ]
        ),
        encoding="utf-8",
    )

    values = load_env_file(env_path)

    assert values == {"HF_TOKEN": "test-token", "MODELS_ROOT": "models"}


def test_load_env_file_returns_empty_dict_for_missing_file(tmp_path: Path) -> None:
    values = load_env_file(tmp_path / ".env")

    assert values == {}
