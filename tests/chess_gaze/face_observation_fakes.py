from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


def build_fake_mediapipe(
    captured: dict[str, Any],
    *,
    face_landmarks: list[list[SimpleNamespace]],
    face_blendshapes: list[list[SimpleNamespace]],
    facial_transformation_matrixes: list[np.ndarray],
) -> SimpleNamespace:
    class FakeBaseOptions:
        def __init__(self, *, model_asset_path: str) -> None:
            self.model_asset_path = model_asset_path

    class FakeFaceLandmarkerOptions:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class FakeFaceLandmarker:
        @classmethod
        def create_from_options(
            cls, options: FakeFaceLandmarkerOptions
        ) -> FakeFaceLandmarker:
            captured["options"] = options
            return cls()

        def detect(self, image: Any) -> SimpleNamespace:
            captured["image"] = image
            return SimpleNamespace(
                face_landmarks=face_landmarks,
                face_blendshapes=face_blendshapes,
                facial_transformation_matrixes=facial_transformation_matrixes,
            )

        def close(self) -> None:
            captured["closed"] = True

    class FakeImage:
        def __init__(self, *, image_format: str, data: np.ndarray) -> None:
            self.image_format = image_format
            self.data = data

    return SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=FakeBaseOptions,
            vision=SimpleNamespace(
                FaceLandmarker=FakeFaceLandmarker,
                FaceLandmarkerOptions=FakeFaceLandmarkerOptions,
                RunningMode=SimpleNamespace(IMAGE="IMAGE"),
            ),
        ),
        Image=FakeImage,
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )


def fake_detection_result(
    *,
    face_landmarks: list[list[SimpleNamespace]],
    face_blendshapes: list[list[SimpleNamespace]],
    facial_transformation_matrixes: list[np.ndarray],
) -> SimpleNamespace:
    return SimpleNamespace(
        face_landmarks=face_landmarks,
        face_blendshapes=face_blendshapes,
        facial_transformation_matrixes=facial_transformation_matrixes,
    )


def empty_detection_result() -> SimpleNamespace:
    return fake_detection_result(
        face_landmarks=[],
        face_blendshapes=[],
        facial_transformation_matrixes=[],
    )


def build_sequence_fake_mediapipe(
    captured: dict[str, Any],
    *,
    results: list[SimpleNamespace],
) -> SimpleNamespace:
    class FakeBaseOptions:
        def __init__(self, *, model_asset_path: str) -> None:
            self.model_asset_path = model_asset_path

    class FakeFaceLandmarkerOptions:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class FakeFaceLandmarker:
        @classmethod
        def create_from_options(
            cls, options: FakeFaceLandmarkerOptions
        ) -> FakeFaceLandmarker:
            captured["options"] = options
            return cls()

        def detect(self, image: Any) -> SimpleNamespace:
            captured.setdefault("images", []).append(image)
            captured["detect_shapes"].append(image.data.shape)
            captured.setdefault("detect_contiguous", []).append(
                bool(image.data.flags.c_contiguous)
            )
            index = len(captured["detect_shapes"]) - 1
            return results[index]

        def close(self) -> None:
            captured["closed"] = True

    class FakeImage:
        def __init__(self, *, image_format: str, data: np.ndarray) -> None:
            self.image_format = image_format
            self.data = data

    return SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=FakeBaseOptions,
            vision=SimpleNamespace(
                FaceLandmarker=FakeFaceLandmarker,
                FaceLandmarkerOptions=FakeFaceLandmarkerOptions,
                RunningMode=SimpleNamespace(IMAGE="IMAGE"),
            ),
        ),
        Image=FakeImage,
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
