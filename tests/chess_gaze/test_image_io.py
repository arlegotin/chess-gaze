from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

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


def test_save_bgr_jpeg_returns_sha256_of_written_bytes(tmp_path: Path) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = np.array([255, 0, 0], dtype=np.uint8)

    digest = save_bgr_jpeg(tmp_path / "frame.jpg", image, quality=90)

    assert digest == sha256((tmp_path / "frame.jpg").read_bytes()).hexdigest()

    loaded = cv2.imread(str(tmp_path / "frame.jpg"))
    assert loaded is not None
    assert loaded.shape == image.shape
