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
            temp_path = Path(temp_file.name)
            temp_file.write(data)
            temp_file.flush()

        temp_path.replace(path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def save_rgb_png(path: Path, image: np.ndarray) -> str:
    image = _validate_rgb_image(image)
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    data = buffer.getvalue()
    atomic_write_bytes(path, data)
    return sha256(data).hexdigest()


def save_bgr_jpeg(path: Path, image: np.ndarray, quality: int) -> str:
    """Persist an RGB image as JPEG, converting to BGR only at the OpenCV boundary."""

    image = _validate_rgb_image(image)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise ValueError("failed to encode JPEG image")

    data = encoded.tobytes()
    atomic_write_bytes(path, data)
    return sha256(data).hexdigest()


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image must have positive height and width")
    return np.ascontiguousarray(image)
