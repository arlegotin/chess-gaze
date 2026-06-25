from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from PIL import Image


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(data)
            temp_file.flush()
            temp_path = Path(temp_file.name)

        temp_path.replace(path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def save_rgb_png(path: Path, image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    data = buffer.getvalue()
    atomic_write_bytes(path, data)
    return sha256(data).hexdigest()


def save_bgr_jpeg(path: Path, image: np.ndarray, quality: int) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("failed to encode JPEG image")

    data = encoded.tobytes()
    atomic_write_bytes(path, data)
    return sha256(data).hexdigest()
