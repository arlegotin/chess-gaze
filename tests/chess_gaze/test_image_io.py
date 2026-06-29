from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pytest
from PIL import Image

import chess_gaze.image_io as image_io
from chess_gaze.image_io import atomic_write_bytes, save_bgr_jpeg, save_rgb_png


def test_atomic_write_replaces_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"

    atomic_write_bytes(target, b"abc")

    assert target.read_bytes() == b"abc"
    assert not list(tmp_path.glob("*.tmp"))


def test_save_rgb_png_returns_sha256(tmp_path: Path) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    digest = save_rgb_png(tmp_path / "frame.png", image)

    assert digest == sha256((tmp_path / "frame.png").read_bytes()).hexdigest()
    assert (tmp_path / "frame.png").is_file()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("shape", [(0, 3, 3), (3, 0, 3)])
def test_save_rgb_png_rejects_empty_image_dimensions(
    tmp_path: Path, shape: tuple[int, int, int]
) -> None:
    image = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="image must have positive height and width"):
        save_rgb_png(tmp_path / "frame.png", image)

    assert not (tmp_path / "frame.png").exists()


def test_save_bgr_jpeg_returns_sha256_of_written_bytes(tmp_path: Path) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = np.array([255, 0, 0], dtype=np.uint8)

    digest = save_bgr_jpeg(tmp_path / "frame.jpg", image, quality=90)

    assert digest == sha256((tmp_path / "frame.jpg").read_bytes()).hexdigest()

    loaded = cv2.imread(str(tmp_path / "frame.jpg"))
    assert loaded is not None
    assert loaded.shape == image.shape


def test_save_bgr_jpeg_converts_rgb_input_before_encoding(tmp_path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :] = np.array([255, 0, 0], dtype=np.uint8)

    save_bgr_jpeg(tmp_path / "frame.jpg", image, quality=95)

    loaded = np.asarray(Image.open(tmp_path / "frame.jpg").convert("RGB"))
    pixel = loaded[8, 8]
    assert pixel[0] >= 200
    assert pixel[1] <= 40
    assert pixel[2] <= 40


@pytest.mark.parametrize("shape", [(0, 3, 3), (3, 0, 3)])
def test_save_bgr_jpeg_rejects_empty_image_dimensions(
    tmp_path: Path, shape: tuple[int, int, int]
) -> None:
    image = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="image must have positive height and width"):
        save_bgr_jpeg(tmp_path / "frame.jpg", image, quality=90)

    assert not (tmp_path / "frame.jpg").exists()


def test_atomic_write_bytes_removes_temp_file_when_flush_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp_path = tmp_path / "artifact.bin.fake.tmp"

    class BrokenTempFile:
        def __init__(self) -> None:
            self.name = str(temp_path)

        def __enter__(self) -> BrokenTempFile:
            temp_path.write_bytes(b"partial")
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
            return False

        def write(self, data: bytes) -> int:
            return len(data)

        def flush(self) -> None:
            raise OSError("flush failed")

    monkeypatch.setattr(image_io, "NamedTemporaryFile", lambda **_: BrokenTempFile())

    with pytest.raises(OSError, match="flush failed"):
        atomic_write_bytes(tmp_path / "artifact.bin", b"abc")

    assert not temp_path.exists()
