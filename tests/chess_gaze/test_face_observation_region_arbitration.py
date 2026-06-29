from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from face_observation_fakes import (
    build_sequence_fake_mediapipe,
    empty_detection_result,
    fake_detection_result,
)

from chess_gaze.calibration import default_calibration
from chess_gaze.face_observation import MediaPipeFaceObserver


def test_mediapipe_observer_prefers_compact_left_half_over_overexpanded_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.155, y=0.503, z=0.0),
                        SimpleNamespace(x=0.281, y=0.775, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="full_frame_overexpanded_face", score=0.20
                        )
                    ]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.321, y=0.568, z=0.0),
                        SimpleNamespace(x=0.500, y=0.750, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="left_half_compact_face", score=0.40
                        )
                    ]
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
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        frame_id="f000001429",
    )

    assert captured["detect_shapes"] == [
        (1080, 1920, 3),
        (1080, 960, 3),
        (1080, 960, 3),
        (540, 960, 3),
        (540, 960, 3),
        (486, 960, 3),
        (486, 960, 3),
        (525, 960, 3),
    ]
    assert observation.selection.present is True
    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "left_half_compact_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(308.16)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(613.44)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(480.0)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(810.0)


def test_mediapipe_observer_rejects_larger_focused_overexpanded_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.20, y=0.50, z=0.0),
                        SimpleNamespace(x=0.35, y=0.78, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="full_frame_overexpanded_face", score=0.20
                        )
                    ]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.35, y=0.45, z=0.0),
                        SimpleNamespace(x=0.75, y=0.80, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="left_half_larger_false_positive",
                            score=0.40,
                        )
                    ]
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
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        frame_id="f000001500",
    )

    assert observation.selection.present is True
    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "full_frame_overexpanded_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(384.0)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(540.0)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(672.0)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(842.4)


def test_mediapipe_observer_rejects_top_shift_candidate_without_compact_geometry_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.20, y=0.48, z=0.0),
                        SimpleNamespace(x=0.36, y=0.76, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="full_frame_overexpanded_face", score=0.20
                        )
                    ]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.42, y=0.40, z=0.0),
                        SimpleNamespace(x=0.72, y=0.66, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="left_half_top_shift_false_positive",
                            score=0.40,
                        )
                    ]
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
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        frame_id="f000001501",
    )

    assert observation.selection.present is True
    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "full_frame_overexpanded_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(384.0)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(518.4)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(691.2)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(820.8)


def test_mediapipe_observer_rejects_large_full_frame_refinement_without_compact_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"imports": [], "detect_shapes": []}
    fake_mediapipe = build_sequence_fake_mediapipe(
        captured,
        results=[
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.17, y=0.45, z=0.0),
                        SimpleNamespace(x=0.42, y=0.75, z=0.0),
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
                        SimpleNamespace(x=0.42, y=0.40, z=0.0),
                        SimpleNamespace(x=0.78, y=0.75, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="left_half_large_false_positive",
                            score=0.40,
                        )
                    ]
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
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        frame_id="f000001502",
    )

    assert observation.selection.present is True
    candidate = observation.selection.candidates[0]
    assert candidate.blendshapes[0].category_name == "large_full_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(326.4)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(486.0)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(806.4)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(810.0)


def test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate(  # noqa: E501
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
                        SimpleNamespace(x=0.375, y=0.20, z=0.0),
                        SimpleNamespace(x=0.5625, y=0.40, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="cross_region_real_face", score=0.80
                        )
                    ]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.375, y=0.4444444444, z=0.0),
                        SimpleNamespace(x=0.5625, y=0.8888888889, z=0.0),
                    ],
                    [
                        SimpleNamespace(x=0.2604166667, y=0.3703703704, z=0.0),
                        SimpleNamespace(x=0.6354166667, y=0.8641975309, z=0.0),
                    ],
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="cross_region_real_face", score=0.80
                        )
                    ],
                    [
                        SimpleNamespace(
                            category_name="larger_single_region_false_positive",
                            score=0.30,
                        )
                    ],
                ],
                facial_transformation_matrixes=[
                    np.eye(4, dtype=np.float64) * 2.0,
                    np.eye(4, dtype=np.float64) * 3.0,
                ],
            ),
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
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        frame_id="f000001503",
    )

    assert captured["detect_shapes"] == [
        (1080, 1920, 3),
        (1080, 960, 3),
        (1080, 960, 3),
        (540, 960, 3),
        (540, 960, 3),
        (486, 960, 3),
        (486, 960, 3),
        (525, 960, 3),
    ]
    assert observation.selection.present is True
    assert observation.selection.primary_candidate_id is not None
    candidate = next(
        candidate
        for candidate in observation.selection.candidates
        if candidate.candidate_id == observation.selection.primary_candidate_id
    )
    assert candidate.blendshapes[0].category_name == "cross_region_real_face"
    assert candidate.bounding_box_image_px.x_min == pytest.approx(360.0)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(216.0)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(540.0)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(432.0)


def test_mediapipe_observer_requires_consensus_fallback_precedence_over_stronger_single_region_candidate(  # noqa: E501
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
                        SimpleNamespace(x=0.375, y=0.1851851852, z=0.0),
                        SimpleNamespace(x=0.46875, y=0.2425925926, z=0.0),
                    ]
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="cross_region_supported_small_face",
                            score=0.80,
                        )
                    ]
                ],
                facial_transformation_matrixes=[np.eye(4, dtype=np.float64)],
            ),
            empty_detection_result(),
            empty_detection_result(),
            empty_detection_result(),
            fake_detection_result(
                face_landmarks=[
                    [
                        SimpleNamespace(x=0.375, y=0.4115226337, z=0.0),
                        SimpleNamespace(x=0.46875, y=0.5390946502, z=0.0),
                    ],
                    [
                        SimpleNamespace(x=0.2604166667, y=0.3292181070, z=0.0),
                        SimpleNamespace(x=0.6770833333, y=0.7818930041, z=0.0),
                    ],
                ],
                face_blendshapes=[
                    [
                        SimpleNamespace(
                            category_name="cross_region_supported_small_face",
                            score=0.80,
                        )
                    ],
                    [
                        SimpleNamespace(
                            category_name="stronger_single_region_false_positive",
                            score=0.30,
                        )
                    ],
                ],
                facial_transformation_matrixes=[
                    np.eye(4, dtype=np.float64) * 2.0,
                    np.eye(4, dtype=np.float64) * 3.0,
                ],
            ),
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
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        frame_id="f000001504",
    )

    assert captured["detect_shapes"] == [
        (1080, 1920, 3),
        (1080, 960, 3),
        (1080, 960, 3),
        (540, 960, 3),
        (540, 960, 3),
        (486, 960, 3),
        (486, 960, 3),
        (525, 960, 3),
    ]
    assert observation.selection.present is True
    assert observation.selection.primary_candidate_id is not None
    candidate = next(
        candidate
        for candidate in observation.selection.candidates
        if candidate.candidate_id == observation.selection.primary_candidate_id
    )
    assert candidate.blendshapes[0].category_name == (
        "cross_region_supported_small_face"
    )
    assert candidate.bounding_box_image_px.x_min == pytest.approx(360.0)
    assert candidate.bounding_box_image_px.y_min == pytest.approx(200.0)
    assert candidate.bounding_box_image_px.x_max == pytest.approx(450.0)
    assert candidate.bounding_box_image_px.y_max == pytest.approx(262.0)
