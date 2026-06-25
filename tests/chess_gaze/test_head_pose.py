from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from chess_gaze.calibration import default_calibration
from chess_gaze.errors import ErrorCode
from chess_gaze.face_observation import FaceCandidate
from chess_gaze.frame_records import CalibrationRecord, PnPLandmarkIndices
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.head_pose import (
    CANONICAL_FACE_MODEL_POINTS_MM,
    PNP_LANDMARK_NAMES,
    PNP_METHOD_NAME,
    PNP_REPROJECTION_ERROR_MAX_PX,
    ImageSize,
    estimate_head_pose,
)

IMAGE_SIZE = ImageSize(width_px=640, height_px=480)


def test_preserves_mediapipe_facial_transformation_matrix() -> None:
    calibration = default_calibration()
    facial_transformation_matrix = (
        (1.0, 0.0, 0.0, 12.0),
        (0.0, 1.0, 0.0, -3.0),
        (0.0, 0.0, 1.0, 42.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    face = _face_with_landmarks(
        _projected_landmarks(calibration, IMAGE_SIZE),
        image_size=IMAGE_SIZE,
        facial_transformation_matrix=facial_transformation_matrix,
    )

    observation = estimate_head_pose(face, calibration, IMAGE_SIZE)

    assert observation.valid is True
    assert observation.facial_transformation_matrix == facial_transformation_matrix


def test_pnp_uses_named_landmark_indices_from_calibration() -> None:
    calibration = default_calibration().model_copy(
        update={
            "pnp_landmark_indices": PnPLandmarkIndices(
                nose_tip=10,
                chin=20,
                left_eye_outer=30,
                right_eye_outer=40,
                left_eye_inner=50,
                right_eye_inner=60,
                left_mouth_corner=70,
                right_mouth_corner=80,
            )
        }
    )
    landmarks = _projected_landmarks(calibration, IMAGE_SIZE)
    face = _face_with_landmarks(landmarks, image_size=IMAGE_SIZE)

    observation = estimate_head_pose(face, calibration, IMAGE_SIZE)

    assert observation.valid is True
    assert observation.pnp_method == PNP_METHOD_NAME
    assert tuple(item.name for item in observation.pnp_landmarks) == PNP_LANDMARK_NAMES

    for pnp_landmark in observation.pnp_landmarks:
        expected_index = getattr(calibration.pnp_landmark_indices, pnp_landmark.name)
        expected_point = landmarks[expected_index]
        assert pnp_landmark.landmark_index == expected_index
        assert pnp_landmark.image_point.x == pytest.approx(expected_point.x)
        assert pnp_landmark.image_point.y == pytest.approx(expected_point.y)


def test_metric_translation_remains_null_when_intrinsics_are_unavailable() -> None:
    calibration = default_calibration()
    face = _face_with_landmarks(
        _projected_landmarks(calibration, IMAGE_SIZE),
        image_size=IMAGE_SIZE,
    )

    observation = estimate_head_pose(face, calibration, IMAGE_SIZE)

    assert observation.valid is True
    assert observation.metric_translation_allowed is False
    assert observation.camera_intrinsics_policy == calibration.camera_intrinsics_policy
    assert observation.translation_camera_3d_m is None


def test_invalid_point_count_produces_head_pose_invalid() -> None:
    calibration = default_calibration()
    landmarks = [
        Point2D(space=CoordinateSpace.IMAGE_PX, x=float(index), y=float(index))
        for index in range(62)
    ]
    face = _face_with_landmarks(landmarks, image_size=IMAGE_SIZE)

    observation = estimate_head_pose(face, calibration, IMAGE_SIZE)

    assert observation.valid is False
    assert observation.reason_invalid == ErrorCode.HEAD_POSE_INVALID
    assert observation.pnp_point_count < observation.pnp_min_point_count
    assert observation.rotation_matrix is None
    assert ErrorCode.HEAD_POSE_INVALID in {error.code for error in observation.errors}


def test_invalid_reprojection_error_produces_head_pose_invalid() -> None:
    calibration = default_calibration()
    landmarks = _projected_landmarks(calibration, IMAGE_SIZE)
    landmarks[calibration.pnp_landmark_indices.chin] = Point2D(
        space=CoordinateSpace.IMAGE_PX,
        x=4000.0,
        y=-3000.0,
    )
    face = _face_with_landmarks(landmarks, image_size=IMAGE_SIZE)

    observation = estimate_head_pose(face, calibration, IMAGE_SIZE)

    assert observation.valid is False
    assert observation.reason_invalid == ErrorCode.HEAD_POSE_INVALID
    assert observation.reprojection_error_px is not None
    assert observation.reprojection_error_px > observation.reprojection_error_max_px
    assert observation.reprojection_error_max_px == PNP_REPROJECTION_ERROR_MAX_PX
    assert observation.rotation_matrix is None


def test_mediapipe_transform_keeps_pose_valid_when_pnp_reprojection_is_high() -> None:
    calibration = default_calibration()
    landmarks = _projected_landmarks(calibration, IMAGE_SIZE)
    landmarks[calibration.pnp_landmark_indices.chin] = Point2D(
        space=CoordinateSpace.IMAGE_PX,
        x=4000.0,
        y=-3000.0,
    )
    expected_yaw = 0.21
    expected_pitch = -0.12
    expected_roll = 0.08
    face = _face_with_landmarks(
        landmarks,
        image_size=IMAGE_SIZE,
        facial_transformation_matrix=_facial_transform_matrix(
            expected_yaw,
            expected_pitch,
            expected_roll,
        ),
    )

    observation = estimate_head_pose(face, calibration, IMAGE_SIZE)

    assert observation.valid is True
    assert observation.reason_invalid is None
    assert observation.method == "mediapipe_transform_and_solvepnp_iterative"
    assert observation.pnp_method == PNP_METHOD_NAME
    assert observation.reprojection_error_px is not None
    assert observation.reprojection_error_px > observation.reprojection_error_max_px
    assert observation.yaw_radians == pytest.approx(expected_yaw)
    assert observation.pitch_radians == pytest.approx(expected_pitch)
    assert observation.roll_radians == pytest.approx(expected_roll)
    assert abs(observation.pitch_radians) < 1.0
    assert observation.rotation_matrix is not None
    assert observation.quaternion_wxyz is not None


def test_valid_rotations_are_stored_as_matrix_quaternion_and_angles() -> None:
    calibration = default_calibration()
    face = _face_with_landmarks(
        _projected_landmarks(calibration, IMAGE_SIZE),
        image_size=IMAGE_SIZE,
    )

    observation = estimate_head_pose(face, calibration, IMAGE_SIZE)

    assert observation.valid is True
    assert observation.reason_invalid is None
    assert observation.rotation_matrix is not None
    assert np.asarray(observation.rotation_matrix).shape == (3, 3)
    assert all(
        math.isfinite(value) for row in observation.rotation_matrix for value in row
    )

    assert observation.quaternion_wxyz is not None
    assert len(observation.quaternion_wxyz) == 4
    assert all(math.isfinite(value) for value in observation.quaternion_wxyz)
    assert sum(value * value for value in observation.quaternion_wxyz) == pytest.approx(
        1.0
    )

    assert observation.yaw_radians is not None
    assert observation.pitch_radians is not None
    assert observation.roll_radians is not None
    assert math.isfinite(observation.yaw_radians)
    assert math.isfinite(observation.pitch_radians)
    assert math.isfinite(observation.roll_radians)


def _projected_landmarks(
    calibration: CalibrationRecord,
    image_size: ImageSize,
) -> list[Point2D]:
    indices = calibration.pnp_landmark_indices.model_dump()
    landmarks = [
        Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=10.0 + float(index % 17),
            y=12.0 + float(index % 19),
        )
        for index in range(max(indices.values()) + 1)
    ]

    object_points = np.asarray(
        [CANONICAL_FACE_MODEL_POINTS_MM[name] for name in PNP_LANDMARK_NAMES],
        dtype=np.float64,
    )
    image_points, _jacobian = cv2.projectPoints(
        object_points,
        np.asarray([[0.12], [-0.18], [0.05]], dtype=np.float64),
        np.asarray([[0.0], [0.0], [620.0]], dtype=np.float64),
        _camera_matrix(image_size),
        np.zeros((4, 1), dtype=np.float64),
    )

    for name, projected in zip(
        PNP_LANDMARK_NAMES,
        image_points.reshape(-1, 2),
        strict=True,
    ):
        landmark_index = indices[name]
        landmarks[landmark_index] = Point2D(
            space=CoordinateSpace.IMAGE_PX,
            x=float(projected[0]),
            y=float(projected[1]),
        )

    return landmarks


def _camera_matrix(image_size: ImageSize) -> np.ndarray:
    focal_length_px = float(max(image_size.width_px, image_size.height_px))
    return np.asarray(
        [
            [focal_length_px, 0.0, image_size.width_px / 2.0],
            [0.0, focal_length_px, image_size.height_px / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _facial_transform_matrix(
    yaw_radians: float, pitch_radians: float, roll_radians: float
) -> tuple[tuple[float, ...], ...]:
    cy = math.cos(yaw_radians)
    sy = math.sin(yaw_radians)
    cp = math.cos(pitch_radians)
    sp = math.sin(pitch_radians)
    cr = math.cos(roll_radians)
    sr = math.sin(roll_radians)
    rotation = np.asarray(
        [
            [cy * cr, (sp * sy * cr) - (cp * sr), (cp * sy * cr) + (sp * sr)],
            [cy * sr, (sp * sy * sr) + (cp * cr), (cp * sy * sr) - (sp * cr)],
            [-sy, sp * cy, cp * cy],
        ],
        dtype=np.float64,
    )
    return (
        (float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2]), 0.0),
        (float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2]), 0.0),
        (float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2]), 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _face_with_landmarks(
    landmarks_image_px: list[Point2D],
    *,
    image_size: ImageSize,
    facial_transformation_matrix: tuple[tuple[float, ...], ...] | None = None,
) -> FaceCandidate:
    return FaceCandidate(
        candidate_id="face_0",
        frame_id="f000000042",
        image_width_px=image_size.width_px,
        image_height_px=image_size.height_px,
        candidate_score=None,
        score_source="synthetic",
        bounding_box_image_px=BBox(
            space=CoordinateSpace.IMAGE_PX,
            x_min=0.0,
            y_min=0.0,
            x_max=float(image_size.width_px),
            y_max=float(image_size.height_px),
        ),
        bounding_box_image_norm=BBox(
            space=CoordinateSpace.NORMALIZED,
            x_min=0.0,
            y_min=0.0,
            x_max=1.0,
            y_max=1.0,
        ),
        landmarks_image_px=landmarks_image_px,
        landmarks_image_norm=[
            Point2D(
                space=CoordinateSpace.NORMALIZED,
                x=point.x / image_size.width_px,
                y=point.y / image_size.height_px,
            )
            for point in landmarks_image_px
        ],
        facial_transformation_matrix=facial_transformation_matrix,
    )
