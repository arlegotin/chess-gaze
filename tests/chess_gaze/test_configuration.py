from __future__ import annotations

from pathlib import Path

import pytest

from chess_gaze.configuration import ConfigurationError, load_config, load_env_file


def test_load_config_uses_task_4_defaults(tmp_path: Path) -> None:
    config = load_config(None)

    assert config.output_root == Path("artifacts/output")
    assert config.models_root == Path("models")
    assert config.raw_frame_image_format == "png"
    assert config.processed_frame_image_format == "jpg"
    assert config.processed_frame_jpeg_quality == 95
    assert config.save_frame_images is False
    assert config.save_crop_images is False


def test_load_config_uses_unigaze_runtime_defaults() -> None:
    config = load_config(None)

    assert config.unigaze_device == "mps"
    assert config.unigaze_batch_size == 7
    assert config.unigaze_preprocessing_profile == "official_geometric_v1"
    assert config.target_plane is None


def test_load_config_accepts_legacy_unigaze_preprocessing_profile(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"unigaze_preprocessing_profile": "legacy_bbox_rgb01"}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.unigaze_preprocessing_profile == "legacy_bbox_rgb01"


def test_load_config_accepts_official_unigaze_preprocessing_profile(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"unigaze_preprocessing_profile": "official_geometric_v1"}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.unigaze_preprocessing_profile == "official_geometric_v1"


def test_load_config_rejects_unknown_unigaze_preprocessing_profile(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"unigaze_preprocessing_profile": "crop_it_somehow"}',
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError, match="unigaze_preprocessing_profile"
    ) as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_INVALID"


def test_load_config_accepts_target_plane(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "target_plane": {
            "origin_camera_m": [-0.30, -0.20, 0.70],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 1.0, 0.0],
            "width_m": 0.60,
            "height_m": 0.40,
            "mirror_horizontal": true
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.target_plane is not None
    assert config.target_plane.origin_camera_m == (-0.30, -0.20, 0.70)
    assert config.target_plane.x_axis_camera == (1.0, 0.0, 0.0)
    assert config.target_plane.y_axis_camera == (0.0, 1.0, 0.0)
    assert config.target_plane.width_m == 0.60
    assert config.target_plane.height_m == 0.40
    assert config.target_plane.mirror_horizontal is True


def test_load_config_rejects_incomplete_target_plane(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "target_plane": {
            "origin_camera_m": [-0.30, -0.20, 0.70],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "width_m": 0.60,
            "height_m": 0.40
          }
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="target_plane") as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_INVALID"


def test_load_config_accepts_save_frame_images(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"save_frame_images": true}', encoding="utf-8")

    config = load_config(config_path)

    assert config.save_frame_images is True


def test_load_config_accepts_save_crop_images(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"save_crop_images": true}', encoding="utf-8")

    config = load_config(config_path)

    assert config.save_crop_images is True


def test_load_config_accepts_explicit_cpu_batch_one_runtime(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"unigaze_device": "cpu", "unigaze_batch_size": 1}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.unigaze_device == "cpu"
    assert config.unigaze_batch_size == 1


def test_load_config_accepts_unigaze_runtime_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"unigaze_device": "mps", "unigaze_batch_size": 7}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.unigaze_device == "mps"
    assert config.unigaze_batch_size == 7


@pytest.mark.parametrize("batch_size", [0, -1])
def test_load_config_rejects_invalid_unigaze_batch_size(
    tmp_path: Path, batch_size: int
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        f'{{"unigaze_batch_size": {batch_size}}}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unigaze_batch_size") as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_INVALID"


def test_load_config_rejects_unsupported_unigaze_device(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"unigaze_device": "cuda"}', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unigaze_device") as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_INVALID"


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"output_root": "custom-output", "unexpected_key": "value"}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unexpected_key") as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_INVALID"


def test_load_config_raises_stable_error_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="missing") as exc_info:
        load_config(tmp_path / "missing.json")

    assert exc_info.value.code == "CONFIG_PATH_MISSING"


def test_load_config_raises_stable_error_for_unreadable_input(tmp_path: Path) -> None:
    config_dir = tmp_path / "config-dir"
    config_dir.mkdir()

    with pytest.raises(ConfigurationError, match="unreadable") as exc_info:
        load_config(config_dir)

    assert exc_info.value.code == "CONFIG_LOAD_UNREADABLE"


def test_load_config_raises_stable_error_for_invalid_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"output_root":', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="valid JSON") as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_INVALID"


def test_load_config_raises_stable_error_for_unsupported_shape(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('["not", "an", "object"]', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="JSON object") as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_UNSUPPORTED_SHAPE"


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
