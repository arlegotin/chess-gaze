from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from chess_gaze.calibration import default_calibration
from chess_gaze.errors import ErrorCode
from chess_gaze.face_observation import (
    MEDIAPIPE_IMAGE_RUNNING_MODE,
    MEDIAPIPE_SCORE_SOURCE_UNAVAILABLE,
    FaceCandidate,
    MediaPipeFaceObserver,
    select_primary_face,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D


def make_candidate(
    candidate_id: str,
    *,
    bbox_norm: tuple[float, float, float, float],
    candidate_score: float | None,
    landmarks_present: bool = True,
) -> FaceCandidate:
    x_min, y_min, x_max, y_max = bbox_norm
    width = 200
    height = 100
    landmarks_norm = [
        Point2D(space=CoordinateSpace.NORMALIZED, x=x_min, y=y_min),
        Point2D(space=CoordinateSpace.NORMALIZED, x=x_max, y=y_max),
    ]
    landmarks_px = [
        Point2D(space=CoordinateSpace.IMAGE_PX, x=x_min * width, y=y_min * height),
        Point2D(space=CoordinateSpace.IMAGE_PX, x=x_max * width, y=y_max * height),
    ]
    if not landmarks_present:
        landmarks_norm = []
        landmarks_px = []

    return FaceCandidate(
        candidate_id=candidate_id,
        frame_id="f000000042",
        image_width_px=width,
        image_height_px=height,
        candidate_score=candidate_score,
        score_source=(
            "mediapipe_face_landmarker"
            if candidate_score is not None
            else "not_exposed_by_mediapipe_face_landmarker"
        ),
        bounding_box_image_px=BBox(
            space=CoordinateSpace.IMAGE_PX,
            x_min=x_min * width,
            y_min=y_min * height,
            x_max=x_max * width,
            y_max=y_max * height,
        ),
        bounding_box_image_norm=BBox(
            space=CoordinateSpace.NORMALIZED,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        ),
        landmarks_image_px=landmarks_px,
        landmarks_image_norm=landmarks_norm,
    )


def test_select_primary_face_selects_single_candidate() -> None:
    calibration = default_calibration()
    candidate = make_candidate(
        "face_0",
        bbox_norm=(0.10, 0.20, 0.50, 0.70),
        candidate_score=0.80,
    )

    selection = select_primary_face([candidate], calibration)

    assert selection.present is True
    assert selection.primary_candidate_id == "face_0"
    assert selection.selection_reason == "single_candidate"
    assert selection.reason_invalid is None
    assert [item.candidate_id for item in selection.candidates] == ["face_0"]


def test_select_primary_face_uses_score_times_area_for_multiple_scored_faces() -> None:
    calibration = default_calibration()
    high_score_small_area = make_candidate(
        "face_0",
        bbox_norm=(0.00, 0.00, 0.20, 0.20),
        candidate_score=0.90,
    )
    lower_score_larger_area = make_candidate(
        "face_1",
        bbox_norm=(0.00, 0.00, 0.60, 0.60),
        candidate_score=0.50,
    )

    selection = select_primary_face(
        [high_score_small_area, lower_score_larger_area], calibration
    )

    assert selection.present is True
    assert selection.primary_candidate_id == "face_1"
    assert selection.selection_score_source == "model_score_times_area_fraction"
    scores_by_id = {
        candidate.candidate_id: candidate.selection_score
        for candidate in selection.candidates
    }
    assert scores_by_id["face_0"] == pytest.approx(0.90 * 0.04)
    assert scores_by_id["face_1"] == pytest.approx(0.50 * 0.36)


def test_select_primary_face_uses_area_only_when_model_scores_are_nullable() -> None:
    calibration = default_calibration()
    smaller_candidate = make_candidate(
        "face_0",
        bbox_norm=(0.00, 0.00, 0.30, 0.30),
        candidate_score=None,
    )
    larger_candidate = make_candidate(
        "face_1",
        bbox_norm=(0.10, 0.10, 0.70, 0.70),
        candidate_score=None,
    )

    selection = select_primary_face([smaller_candidate, larger_candidate], calibration)

    assert selection.present is True
    assert selection.primary_candidate_id == "face_1"
    assert selection.selection_score_source == "area_only_no_model_score"
    assert all(candidate.candidate_score is None for candidate in selection.candidates)
    assert {candidate.selection_score_source for candidate in selection.candidates} == {
        "area_only_no_model_score"
    }


def test_select_primary_face_preserves_all_candidates_and_multiple_error() -> None:
    calibration = default_calibration()
    candidates = [
        make_candidate("face_0", bbox_norm=(0.0, 0.0, 0.3, 0.3), candidate_score=None),
        make_candidate("face_1", bbox_norm=(0.1, 0.1, 0.5, 0.5), candidate_score=None),
        make_candidate("face_2", bbox_norm=(0.2, 0.2, 0.4, 0.4), candidate_score=None),
    ]

    selection = select_primary_face(candidates, calibration)

    assert [candidate.candidate_id for candidate in selection.candidates] == [
        "face_0",
        "face_1",
        "face_2",
    ]
    assert ErrorCode.MULTIPLE_FACE_CANDIDATES in {
        error.code for error in selection.errors
    }


def test_select_primary_face_reports_face_not_found_for_invalid_landmarks() -> None:
    calibration = default_calibration()
    candidate = make_candidate(
        "face_0",
        bbox_norm=(0.10, 0.20, 0.50, 0.70),
        candidate_score=None,
        landmarks_present=False,
    )

    selection = select_primary_face([candidate], calibration)

    assert selection.present is False
    assert selection.primary_candidate_id is None
    assert selection.reason_invalid == ErrorCode.FACE_NOT_FOUND
    assert ErrorCode.FACE_NOT_FOUND in {error.code for error in selection.errors}
    assert [item.candidate_id for item in selection.candidates] == ["face_0"]


def test_mediapipe_observer_persists_options_without_importing_mediapipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def fail_import() -> object:
        imports.append("mediapipe")
        raise AssertionError("MediaPipe should not import during observer setup")

    monkeypatch.setattr("chess_gaze.face_observation._import_mediapipe", fail_import)

    calibration = default_calibration()
    observer = MediaPipeFaceObserver(
        model_asset_path=Path("models/mediapipe/face_landmarker.task"),
        calibration=calibration,
    )

    assert imports == []
    assert observer.face_landmarker_options.running_mode == "IMAGE"
    assert observer.face_landmarker_options.num_faces == calibration.max_face_candidates
    assert (
        observer.face_landmarker_options.min_face_detection_confidence
        == calibration.candidate_face_score_min
    )
    assert (
        observer.face_landmarker_options.min_face_presence_confidence
        == calibration.usable_face_score_min
    )
    assert observer.face_landmarker_options.output_face_blendshapes is True
    assert (
        observer.face_landmarker_options.output_facial_transformation_matrixes is True
    )
    assert (
        observer.face_landmarker_options.min_tracking_confidence_source
        == "ignored_for_image_mode"
    )


def test_mediapipe_observer_rejects_non_image_running_mode() -> None:
    calibration = default_calibration().model_copy(
        update={"face_landmarker_running_mode": "VIDEO"}
    )

    with pytest.raises(ValueError, match="requires IMAGE running mode"):
        MediaPipeFaceObserver(
            model_asset_path=Path("models/mediapipe/face_landmarker.task"),
            calibration=calibration,
        )


def test_mediapipe_observer_configures_image_mode_and_maps_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": []}
    fake_mediapipe = build_fake_mediapipe(
        captured,
        face_landmarks=[
            [
                SimpleNamespace(x=0.10, y=0.20, z=0.0),
                SimpleNamespace(x=0.40, y=0.80, z=0.0),
            ]
        ],
        face_blendshapes=[[SimpleNamespace(category_name="browDownLeft", score=0.25)]],
        facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
    )

    def fake_import() -> object:
        captured["imports"].append("mediapipe")
        return fake_mediapipe

    monkeypatch.setattr("chess_gaze.face_observation._import_mediapipe", fake_import)
    calibration = default_calibration()
    observer = MediaPipeFaceObserver(
        model_asset_path=Path("models/mediapipe/face_landmarker.task"),
        calibration=calibration,
    )

    observation = observer.observe(
        np.zeros((10, 20, 3), dtype=np.uint8),
        frame_id="f000000005",
    )

    options = captured["options"]
    assert options.running_mode == MEDIAPIPE_IMAGE_RUNNING_MODE
    assert options.num_faces == calibration.max_face_candidates
    assert options.min_face_detection_confidence == calibration.candidate_face_score_min
    assert options.min_face_presence_confidence == calibration.usable_face_score_min
    assert options.output_face_blendshapes is True
    assert options.output_facial_transformation_matrixes is True

    image = captured["image"]
    assert image.image_format == "SRGB"
    assert image.data.shape == (10, 20, 3)
    assert observation.frame_id == "f000000005"
    assert observation.image_width_px == 20
    assert observation.image_height_px == 10
    assert observation.selection.present is True
    assert observation.selection.primary_candidate_id == "face_0"
    candidate = observation.selection.candidates[0]
    assert candidate.frame_id == "f000000005"
    assert candidate.image_width_px == 20
    assert candidate.image_height_px == 10
    assert candidate.candidate_score is None
    assert candidate.score_source == MEDIAPIPE_SCORE_SOURCE_UNAVAILABLE
    assert candidate.bounding_box_image_px.x_min == pytest.approx(2.0)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(2.0)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(8.0)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(8.0)
    assert candidate.blendshapes[0].category_name == "browDownLeft"
    assert candidate.blendshapes[0].score == pytest.approx(0.25)
    assert candidate.facial_transformation_matrix == (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_mediapipe_observer_preserves_multiple_adapter_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": []}
    fake_mediapipe = build_fake_mediapipe(
        captured,
        face_landmarks=[
            [
                SimpleNamespace(x=0.10, y=0.10, z=0.0),
                SimpleNamespace(x=0.20, y=0.20, z=0.0),
            ],
            [
                SimpleNamespace(x=0.30, y=0.30, z=0.0),
                SimpleNamespace(x=0.80, y=0.80, z=0.0),
            ],
        ],
        face_blendshapes=[
            [SimpleNamespace(category_name="face0", score=0.10)],
            [SimpleNamespace(category_name="face1", score=0.20)],
        ],
        facial_transformation_matrixes=[
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.float64) * 2.0,
        ],
    )

    def fake_import() -> object:
        captured["imports"].append("mediapipe")
        return fake_mediapipe

    monkeypatch.setattr("chess_gaze.face_observation._import_mediapipe", fake_import)
    observer = MediaPipeFaceObserver(
        model_asset_path=Path("models/mediapipe/face_landmarker.task"),
        calibration=default_calibration(),
    )

    observation = observer.observe(
        np.zeros((100, 200, 3), dtype=np.uint8),
        frame_id="f000000009",
    )

    assert observation.selection.present is True
    assert observation.selection.primary_candidate_id == "face_1"
    candidate_ids = [
        candidate.candidate_id for candidate in observation.selection.candidates
    ]
    assert candidate_ids == [
        "face_0",
        "face_1",
    ]
    assert ErrorCode.MULTIPLE_FACE_CANDIDATES in {
        error.code for error in observation.selection.errors
    }
    assert [
        candidate.blendshapes[0].category_name
        for candidate in observation.selection.candidates
    ] == ["face0", "face1"]


def test_mediapipe_observer_reports_face_not_found_for_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": []}
    fake_mediapipe = build_fake_mediapipe(
        captured,
        face_landmarks=[],
        face_blendshapes=[],
        facial_transformation_matrixes=[],
    )

    def fake_import() -> object:
        captured["imports"].append("mediapipe")
        return fake_mediapipe

    monkeypatch.setattr("chess_gaze.face_observation._import_mediapipe", fake_import)

    observer = MediaPipeFaceObserver(
        model_asset_path=Path("models/mediapipe/face_landmarker.task"),
        calibration=default_calibration(),
    )
    observation = observer.observe(np.zeros((10, 20, 3), dtype=np.uint8))

    assert observation.selection.present is False
    assert observation.selection.reason_invalid == ErrorCode.FACE_NOT_FOUND
    assert ErrorCode.FACE_NOT_FOUND in {
        error.code for error in observation.selection.errors
    }


def test_mediapipe_observer_recovers_full_frame_miss_from_right_half_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            _fake_detection_result(
                face_landmarks=[],
                face_blendshapes=[],
                facial_transformation_matrixes=[],
            ),
            _fake_detection_result(
                face_landmarks=[],
                face_blendshapes=[],
                facial_transformation_matrixes=[],
            ),
            _fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.20, y=0.30, z=0.0),
                        SimpleNamespace(x=0.60, y=0.70, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="eyeBlinkLeft", score=0.12)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
        ],
    )

    def fake_import() -> object:
        captured["imports"].append("mediapipe")
        return fake_mediapipe

    monkeypatch.setattr("chess_gaze.face_observation._import_mediapipe", fake_import)

    observer = MediaPipeFaceObserver(
        model_asset_path=Path("models/mediapipe/face_landmarker.task"),
        calibration=default_calibration(),
    )

    observation = observer.observe(
        np.zeros((100, 200, 3), dtype=np.uint8),
        frame_id="f000000080",
    )

    assert captured["detect_shapes"] == [(100, 200, 3), (100, 100, 3), (100, 100, 3)]
    assert captured["detect_contiguous"] == [True, True, True]
    assert observation.selection.present is True
    assert observation.image_width_px == 200
    assert observation.image_height_px == 100
    candidate = observation.selection.candidates[0]
    assert candidate.frame_id == "f000000080"
    assert candidate.image_width_px == 200
    assert candidate.image_height_px == 100
    assert candidate.bounding_box_image_px.x_min == pytest.approx(120.0)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(30.0)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(160.0)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(70.0)
    assert candidate.bounding_box_image_norm.x_min == pytest.approx(0.60)
    assert candidate.bounding_box_image_norm.x_max == pytest.approx(0.80)
    assert candidate.landmarks_image_px[0].x == pytest.approx(120.0)
    assert candidate.landmarks_image_px[1].x == pytest.approx(160.0)
    assert candidate.blendshapes[0].category_name == "eyeBlinkLeft"


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


def _fake_detection_result(
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
