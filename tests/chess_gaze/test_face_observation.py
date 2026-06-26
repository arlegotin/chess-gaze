from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from face_observation_fakes import (
    build_fake_mediapipe,
    build_sequence_fake_mediapipe,
    empty_detection_result,
    fake_detection_result,
)

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
            fake_detection_result(
                face_landmarks=[],
                face_blendshapes=[],
                facial_transformation_matrixes=[],
            ),
            fake_detection_result(
                face_landmarks=[],
                face_blendshapes=[],
                facial_transformation_matrixes=[],
            ),
            fake_detection_result(
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
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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

    assert captured["detect_shapes"] == [
        (100, 200, 3),
        (100, 100, 3),
        (100, 100, 3),
        (50, 100, 3),
        (50, 100, 3),
        (45, 100, 3),
        (45, 100, 3),
        (49, 100, 3),
    ]
    assert captured["detect_contiguous"] == [True] * 8
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


def test_mediapipe_observer_prefers_focused_right_half_over_ambiguous_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.76, y=0.29, z=0.0),
                        SimpleNamespace(x=0.90, y=0.53, z=0.0),
                    ],
                    [
                        SimpleNamespace(x=0.77, y=0.32, z=0.0),
                        SimpleNamespace(x=0.91, y=0.63, z=0.0),
                    ],
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="full_good", score=0.10)],
                    [SimpleNamespace(category_name="full_bad", score=0.20)],
                ],
                facial_transformation_matrixes=[
                    np.eye(4, dtype=np.float64),
                    np.eye(4, dtype=np.float64) * 2.0,
                ],
            ),
            fake_detection_result(
                face_landmarks=[],
                face_blendshapes=[],
                facial_transformation_matrixes=[],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.54, y=0.29, z=0.0),
                        SimpleNamespace(x=0.79, y=0.53, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="right_refined", score=0.30)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 3.0],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000040",
    )

    assert captured["detect_shapes"] == [
        (720, 1280, 3),
        (720, 640, 3),
        (720, 640, 3),
        (360, 640, 3),
        (360, 640, 3),
        (324, 640, 3),
        (324, 640, 3),
        (350, 640, 3),
    ]
    assert observation.selection.present is True
    assert observation.selection.primary_candidate_id == "face_0"
    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "right_refined"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(985.6)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(208.8)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(1145.6)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(381.6)
    assert ErrorCode.MULTIPLE_FACE_CANDIDATES not in {
        error.code for error in observation.selection.errors
    }


def test_mediapipe_observer_prefers_focused_left_half_over_low_partial_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.21, y=0.69, z=0.0),
                        SimpleNamespace(x=0.31, y=0.82, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="full_partial", score=0.10)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.39, y=0.62, z=0.0),
                        SimpleNamespace(x=0.56, y=0.79, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="left_refined", score=0.30)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 4.0],
            ),
            fake_detection_result(
                face_landmarks=[],
                face_blendshapes=[],
                facial_transformation_matrixes=[],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000180",
    )

    assert captured["detect_shapes"] == [
        (720, 1280, 3),
        (720, 640, 3),
        (720, 640, 3),
        (360, 640, 3),
        (360, 640, 3),
        (324, 640, 3),
        (324, 640, 3),
        (350, 640, 3),
    ]
    assert observation.selection.present is True
    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "left_refined"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(249.6)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(446.4)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(358.4)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(568.8)


def test_mediapipe_observer_scores_all_focused_half_frame_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.10, y=0.20, z=0.0),
                        SimpleNamespace(x=0.20, y=0.40, z=0.0),
                    ],
                    [
                        SimpleNamespace(x=0.76, y=0.29, z=0.0),
                        SimpleNamespace(x=0.90, y=0.53, z=0.0),
                    ],
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="left_small_full", score=0.10)],
                    [SimpleNamespace(category_name="right_large_full", score=0.20)],
                ],
                facial_transformation_matrixes=[
                    np.eye(4, dtype=np.float64),
                    np.eye(4, dtype=np.float64) * 2.0,
                ],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.21, y=0.20, z=0.0),
                        SimpleNamespace(x=0.39, y=0.40, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="left_small_refined", score=0.30)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 3.0],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.54, y=0.29, z=0.0),
                        SimpleNamespace(x=0.79, y=0.53, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="right_large_refined", score=0.40)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 4.0],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000040",
    )

    assert captured["detect_shapes"] == [
        (720, 1280, 3),
        (720, 640, 3),
        (720, 640, 3),
        (360, 640, 3),
        (360, 640, 3),
        (324, 640, 3),
        (324, 640, 3),
        (350, 640, 3),
    ]
    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "right_large_refined"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(985.6)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(1145.6)


def test_mediapipe_observer_prefers_tighter_top_left_face_over_wrong_half_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.50, y=0.43, z=0.0),
                        SimpleNamespace(x=0.675, y=0.79, z=0.0),
                    ],
                    [
                        SimpleNamespace(x=0.71, y=0.32, z=0.0),
                        SimpleNamespace(x=0.92, y=0.74, z=0.0),
                    ],
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="real_face", score=0.30)],
                    [SimpleNamespace(category_name="board_false_positive", score=0.10)],
                ],
                facial_transformation_matrixes=[
                    np.eye(4, dtype=np.float64),
                    np.eye(4, dtype=np.float64) * 2.0,
                ],
            ),
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.50, y=0.43, z=0.0),
                        SimpleNamespace(x=0.675, y=0.79, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="top_left_real_face", score=0.40)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 3.0],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000237",
    )

    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "top_left_real_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(320.0)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(154.8)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(432.0)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(284.4)


def test_mediapipe_observer_prefers_focused_left_face_over_tall_full_false_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.295, y=0.10, z=0.0),
                        SimpleNamespace(x=0.467, y=0.49, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="full_frame_board", score=0.10)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.45, y=0.217, z=0.0),
                        SimpleNamespace(x=0.623, y=0.401, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="left_half_real_face", score=0.30)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 2.0],
            ),
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.45, y=0.433, z=0.0),
                        SimpleNamespace(x=0.623, y=0.802, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="top_left_real_face", score=0.40)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 3.0],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000265",
    )

    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "top_left_real_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(288.0)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(155.88)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(398.72)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(288.72)


def test_mediapipe_observer_recovers_small_face_from_right_top_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.523, y=0.638, z=0.0),
                        SimpleNamespace(x=0.671, y=0.936, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="right_top_real_face", score=0.40)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000422",
    )

    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "right_top_real_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(974.72)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(229.68)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(1069.44)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(336.96)


def test_mediapipe_observer_recovers_small_face_from_right_upper_middle_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.435, y=0.342, z=0.0),
                        SimpleNamespace(x=0.575, y=0.636, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="right_upper_middle_real_face", score=0.40
                        )
                    ]
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000510",
    )

    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "right_upper_middle_real_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(918.4)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(199.7)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(1008.0)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(302.6)


def test_mediapipe_observer_scans_focused_regions_before_accepting_half_frame_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.50, y=0.43, z=0.0),
                        SimpleNamespace(x=0.675, y=0.79, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="left_half_face", score=0.30)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.50, y=0.43, z=0.0),
                        SimpleNamespace(x=0.675, y=0.79, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="top_left_face", score=0.40)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 2.0],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000238",
    )

    assert captured["detect_shapes"] == [
        (720, 1280, 3),
        (720, 640, 3),
        (720, 640, 3),
        (360, 640, 3),
        (360, 640, 3),
        (324, 640, 3),
        (324, 640, 3),
        (350, 640, 3),
    ]
    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "top_left_face"


def test_mediapipe_observer_keeps_large_full_face_without_focused_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.30, y=0.18, z=0.0),
                        SimpleNamespace(x=0.50, y=0.56, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="large_full_face", score=0.50)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.10, y=0.20, z=0.0),
                        SimpleNamespace(x=0.27, y=0.39, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [SimpleNamespace(category_name="unrelated_small_face", score=0.20)]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64) * 2.0],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000300",
    )

    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "large_full_face"


@pytest.mark.parametrize(
    ("result_index", "landmarks", "category_name"),
    [
        (
            3,
            [
                SimpleNamespace(x=0.80, y=0.30, z=0.0),
                SimpleNamespace(x=0.995, y=0.62, z=0.0),
            ],
            "right_edge_clipped_face",
        ),
        (
            4,
            [
                SimpleNamespace(x=0.005, y=0.30, z=0.0),
                SimpleNamespace(x=0.20, y=0.62, z=0.0),
            ],
            "left_edge_clipped_face",
        ),
        (
            7,
            [
                SimpleNamespace(x=0.35, y=0.005, z=0.0),
                SimpleNamespace(x=0.55, y=0.32, z=0.0),
            ],
            "top_edge_clipped_face",
        ),
        (
            3,
            [
                SimpleNamespace(x=0.35, y=0.60, z=0.0),
                SimpleNamespace(x=0.50, y=0.995, z=0.0),
            ],
            "bottom_edge_clipped_face",
        ),
    ],
)
def test_mediapipe_observer_rejects_candidates_clipped_by_focused_region_seams(
    monkeypatch: pytest.MonkeyPatch,
    result_index: int,
    landmarks: list[SimpleNamespace],
    category_name: str,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    results = [empty_detection_result() for _index in range(8)]
    results[result_index] = fake_detection_result(
        face_landmarks=[landmarks],
        face_blendshapes=[[SimpleNamespace(category_name=category_name, score=0.20)]],
        facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
    )
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=results,
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
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_id="f000000301",
    )

    assert observation.selection.present is False
    assert observation.selection.reason_invalid is ErrorCode.FACE_NOT_FOUND
