from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from chess_gaze.errors import ErrorCode
from chess_gaze.face_observation import FaceCandidate
from chess_gaze.frame_records import CalibrationRecord, ErrorRecord, PnPLandmarkIndices
from chess_gaze.geometry import CoordinateSpace, Point2D

type SelectedFace = FaceCandidate
type MatrixTuple = tuple[tuple[float, ...], ...]

PNP_LANDMARK_NAMES = (
    "nose_tip",
    "chin",
    "left_eye_outer",
    "right_eye_outer",
    "left_eye_inner",
    "right_eye_inner",
    "left_mouth_corner",
    "right_mouth_corner",
)
PNP_METHOD_NAME = "solvepnp_iterative"
HEAD_POSE_METHOD_WITH_MEDIAPIPE_TRANSFORM = "mediapipe_transform_and_solvepnp_iterative"
PNP_MIN_POINT_COUNT = 6
PNP_REPROJECTION_ERROR_MAX_PX = 8.0
PNP_REPROJECTION_ERROR_THRESHOLD_NAME = "PNP_REPROJECTION_ERROR_MAX_PX"
PNP_REPROJECTION_ERROR_THRESHOLD_SOURCE = "chess_gaze.head_pose"
CANONICAL_FACE_MODEL_POINTS_SOURCE = "task_9_named_canonical_face_model_mm"

CANONICAL_FACE_MODEL_POINTS_MM = {
    "nose_tip": (0.0, 0.0, 0.0),
    "chin": (0.0, -63.6, -12.5),
    "left_eye_outer": (-43.3, 32.7, -26.0),
    "right_eye_outer": (43.3, 32.7, -26.0),
    "left_eye_inner": (-20.0, 32.0, -20.0),
    "right_eye_inner": (20.0, 32.0, -20.0),
    "left_mouth_corner": (-28.9, -28.9, -24.1),
    "right_mouth_corner": (28.9, -28.9, -24.1),
}


@dataclass(frozen=True)
class ImageSize:
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        if self.width_px <= 0:
            raise ValueError("width_px must be positive")
        if self.height_px <= 0:
            raise ValueError("height_px must be positive")


@dataclass(frozen=True)
class PnPLandmarkEvidence:
    name: str
    landmark_index: int
    image_point: Point2D
    object_point_mm: tuple[float, float, float]


@dataclass(frozen=True)
class HeadPoseObservation:
    valid: bool
    method: str | None
    reason_invalid: ErrorCode | None
    facial_transformation_matrix: MatrixTuple | None
    pnp_method: str | None
    pnp_landmarks: tuple[PnPLandmarkEvidence, ...]
    pnp_point_count: int
    pnp_min_point_count: int
    canonical_points_source: str
    camera_intrinsics_policy: str
    metric_translation_allowed: bool
    reprojection_error_px: float | None
    reprojection_error_max_px: float
    reprojection_error_threshold_name: str
    reprojection_error_threshold_source: str
    rotation_matrix: MatrixTuple | None
    quaternion_wxyz: tuple[float, float, float, float] | None
    yaw_radians: float | None
    pitch_radians: float | None
    roll_radians: float | None
    translation_camera_3d_m: tuple[float, float, float] | None
    errors: tuple[ErrorRecord, ...]


def estimate_head_pose(
    face: SelectedFace,
    calibration: CalibrationRecord,
    image_size: ImageSize,
) -> HeadPoseObservation:
    facial_transform, facial_transform_error = _facial_transformation_matrix(
        face.facial_transformation_matrix
    )
    pnp_landmarks = _pnp_landmarks(
        face.landmarks_image_px, calibration.pnp_landmark_indices
    )
    reprojection_error_max_px = _reprojection_error_max_px(calibration)
    transform_rotation_matrix = _transform_rotation_matrix(facial_transform)

    if facial_transform_error is not None:
        return _invalid_observation(
            facial_transform=facial_transform,
            pnp_landmarks=pnp_landmarks,
            calibration=calibration,
            reprojection_error_max_px=reprojection_error_max_px,
            pnp_method=None,
            reprojection_error_px=None,
            message=facial_transform_error,
        )

    if len(pnp_landmarks) < PNP_MIN_POINT_COUNT:
        if transform_rotation_matrix is not None:
            return _valid_transform_observation(
                facial_transform=facial_transform,
                transform_rotation_matrix=transform_rotation_matrix,
                pnp_landmarks=pnp_landmarks,
                calibration=calibration,
                reprojection_error_max_px=reprojection_error_max_px,
                pnp_method=None,
                reprojection_error_px=None,
            )
        return _invalid_observation(
            facial_transform=facial_transform,
            pnp_landmarks=pnp_landmarks,
            calibration=calibration,
            reprojection_error_max_px=reprojection_error_max_px,
            pnp_method=None,
            reprojection_error_px=None,
            message=(
                f"Head pose PnP requires at least {PNP_MIN_POINT_COUNT} named "
                f"landmarks; got {len(pnp_landmarks)}."
            ),
        )

    camera_matrix = _estimated_camera_matrix(image_size)
    image_points = np.asarray(
        [[item.image_point.x, item.image_point.y] for item in pnp_landmarks],
        dtype=np.float64,
    )
    object_points = np.asarray(
        [item.object_point_mm for item in pnp_landmarks],
        dtype=np.float64,
    )
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not bool(success):
        if transform_rotation_matrix is not None:
            return _valid_transform_observation(
                facial_transform=facial_transform,
                transform_rotation_matrix=transform_rotation_matrix,
                pnp_landmarks=pnp_landmarks,
                calibration=calibration,
                reprojection_error_max_px=reprojection_error_max_px,
                pnp_method=PNP_METHOD_NAME,
                reprojection_error_px=None,
            )
        return _invalid_observation(
            facial_transform=facial_transform,
            pnp_landmarks=pnp_landmarks,
            calibration=calibration,
            reprojection_error_max_px=reprojection_error_max_px,
            pnp_method=PNP_METHOD_NAME,
            reprojection_error_px=None,
            message="OpenCV solvePnP did not return a pose solution.",
        )

    rotation_matrix_array = np.asarray(
        cv2.Rodrigues(rotation_vector)[0], dtype=np.float64
    )
    if not _finite_matrix(rotation_matrix_array, expected_shape=(3, 3)):
        if transform_rotation_matrix is not None:
            return _valid_transform_observation(
                facial_transform=facial_transform,
                transform_rotation_matrix=transform_rotation_matrix,
                pnp_landmarks=pnp_landmarks,
                calibration=calibration,
                reprojection_error_max_px=reprojection_error_max_px,
                pnp_method=PNP_METHOD_NAME,
                reprojection_error_px=None,
            )
        return _invalid_observation(
            facial_transform=facial_transform,
            pnp_landmarks=pnp_landmarks,
            calibration=calibration,
            reprojection_error_max_px=reprojection_error_max_px,
            pnp_method=PNP_METHOD_NAME,
            reprojection_error_px=None,
            message="OpenCV solvePnP produced a non-finite rotation matrix.",
        )

    projected_points = np.asarray(
        cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
        )[0],
        dtype=np.float64,
    )
    reprojection_error_px = _reprojection_error_px(
        image_points,
        projected_points.reshape(-1, 2),
    )
    if (
        reprojection_error_px is None
        or reprojection_error_px > reprojection_error_max_px
    ):
        if transform_rotation_matrix is not None:
            return _valid_transform_observation(
                facial_transform=facial_transform,
                transform_rotation_matrix=transform_rotation_matrix,
                pnp_landmarks=pnp_landmarks,
                calibration=calibration,
                reprojection_error_max_px=reprojection_error_max_px,
                pnp_method=PNP_METHOD_NAME,
                reprojection_error_px=reprojection_error_px,
            )
        return _invalid_observation(
            facial_transform=facial_transform,
            pnp_landmarks=pnp_landmarks,
            calibration=calibration,
            reprojection_error_max_px=reprojection_error_max_px,
            pnp_method=PNP_METHOD_NAME,
            reprojection_error_px=reprojection_error_px,
            message=(
                "Head pose PnP reprojection error exceeded "
                f"{PNP_REPROJECTION_ERROR_THRESHOLD_NAME}."
            ),
        )

    pose_rotation_matrix_array = (
        transform_rotation_matrix
        if transform_rotation_matrix is not None
        else rotation_matrix_array
    )
    rotation_matrix = _matrix_tuple(pose_rotation_matrix_array)
    quaternion = _quaternion_wxyz(pose_rotation_matrix_array)
    yaw_radians, pitch_radians, roll_radians = _yaw_pitch_roll(
        pose_rotation_matrix_array
    )

    if quaternion is None or not all(
        math.isfinite(value)
        for value in (*quaternion, yaw_radians, pitch_radians, roll_radians)
    ):
        return _invalid_observation(
            facial_transform=facial_transform,
            pnp_landmarks=pnp_landmarks,
            calibration=calibration,
            reprojection_error_max_px=reprojection_error_max_px,
            pnp_method=PNP_METHOD_NAME,
            reprojection_error_px=reprojection_error_px,
            message="Head pose rotation conversion produced non-finite values.",
        )

    return HeadPoseObservation(
        valid=True,
        method=_method_name(facial_transform),
        reason_invalid=None,
        facial_transformation_matrix=facial_transform,
        pnp_method=PNP_METHOD_NAME,
        pnp_landmarks=pnp_landmarks,
        pnp_point_count=len(pnp_landmarks),
        pnp_min_point_count=PNP_MIN_POINT_COUNT,
        canonical_points_source=CANONICAL_FACE_MODEL_POINTS_SOURCE,
        camera_intrinsics_policy=calibration.camera_intrinsics_policy,
        metric_translation_allowed=calibration.metric_translation_allowed,
        reprojection_error_px=reprojection_error_px,
        reprojection_error_max_px=reprojection_error_max_px,
        reprojection_error_threshold_name=PNP_REPROJECTION_ERROR_THRESHOLD_NAME,
        reprojection_error_threshold_source=PNP_REPROJECTION_ERROR_THRESHOLD_SOURCE,
        rotation_matrix=rotation_matrix,
        quaternion_wxyz=quaternion,
        yaw_radians=yaw_radians,
        pitch_radians=pitch_radians,
        roll_radians=roll_radians,
        translation_camera_3d_m=None,
        errors=(),
    )


def _facial_transformation_matrix(
    matrix: Sequence[Sequence[float]] | None,
) -> tuple[MatrixTuple | None, str | None]:
    if matrix is None:
        return None, None

    matrix_array = np.asarray(matrix, dtype=np.float64)
    if not _finite_matrix(matrix_array, expected_shape=(4, 4)):
        return None, "MediaPipe facial transformation matrix must be finite 4x4."

    return _matrix_tuple(matrix_array), None


def _transform_rotation_matrix(
    facial_transform: MatrixTuple | None,
) -> npt.NDArray[np.float64] | None:
    if facial_transform is None:
        return None

    rotation_matrix = np.asarray(facial_transform, dtype=np.float64)[:3, :3]
    if not _finite_matrix(rotation_matrix, expected_shape=(3, 3)):
        return None
    return rotation_matrix


def _pnp_landmarks(
    landmarks_image_px: Sequence[Point2D],
    indices: PnPLandmarkIndices,
) -> tuple[PnPLandmarkEvidence, ...]:
    index_by_name = _indices_by_name(indices)
    landmarks: list[PnPLandmarkEvidence] = []
    for name in PNP_LANDMARK_NAMES:
        landmark_index = index_by_name[name]
        if landmark_index < 0 or landmark_index >= len(landmarks_image_px):
            continue

        point = landmarks_image_px[landmark_index]
        if point.space != CoordinateSpace.IMAGE_PX:
            continue
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            continue

        landmarks.append(
            PnPLandmarkEvidence(
                name=name,
                landmark_index=landmark_index,
                image_point=point,
                object_point_mm=CANONICAL_FACE_MODEL_POINTS_MM[name],
            )
        )

    return tuple(landmarks)


def _indices_by_name(indices: PnPLandmarkIndices) -> dict[str, int]:
    return {name: int(getattr(indices, name)) for name in PNP_LANDMARK_NAMES}


def _estimated_camera_matrix(image_size: ImageSize) -> npt.NDArray[np.float64]:
    focal_length_px = float(max(image_size.width_px, image_size.height_px))
    return np.asarray(
        [
            [focal_length_px, 0.0, image_size.width_px / 2.0],
            [0.0, focal_length_px, image_size.height_px / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _reprojection_error_max_px(calibration: CalibrationRecord) -> float:
    configured = getattr(calibration, "head_pose_reprojection_error_max_px", None)
    if configured is None:
        return PNP_REPROJECTION_ERROR_MAX_PX

    value = float(configured)
    if not math.isfinite(value) or value <= 0.0:
        return PNP_REPROJECTION_ERROR_MAX_PX
    return value


def _reprojection_error_px(
    expected_points: npt.NDArray[np.float64],
    projected_points: npt.NDArray[np.float64],
) -> float | None:
    if expected_points.shape != projected_points.shape:
        return None
    if not _finite_matrix(expected_points) or not _finite_matrix(projected_points):
        return None

    deltas = expected_points - projected_points
    squared_errors = np.sum(deltas * deltas, axis=1)
    error = float(np.sqrt(np.mean(squared_errors)))
    return error if math.isfinite(error) else None


def _finite_matrix(
    matrix: npt.NDArray[np.float64],
    *,
    expected_shape: tuple[int, int] | None = None,
) -> bool:
    if expected_shape is not None and matrix.shape != expected_shape:
        return False
    return matrix.ndim == 2 and bool(np.isfinite(matrix).all())


def _matrix_tuple(
    matrix: npt.NDArray[np.float64],
) -> MatrixTuple:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _quaternion_wxyz(
    rotation_matrix: npt.NDArray[np.float64],
) -> tuple[float, float, float, float] | None:
    trace = float(np.trace(rotation_matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(rotation_matrix)))
        if diagonal_index == 0:
            scale = (
                math.sqrt(
                    1.0
                    + rotation_matrix[0, 0]
                    - rotation_matrix[1, 1]
                    - rotation_matrix[2, 2]
                )
                * 2.0
            )
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        elif diagonal_index == 1:
            scale = (
                math.sqrt(
                    1.0
                    + rotation_matrix[1, 1]
                    - rotation_matrix[0, 0]
                    - rotation_matrix[2, 2]
                )
                * 2.0
            )
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        else:
            scale = (
                math.sqrt(
                    1.0
                    + rotation_matrix[2, 2]
                    - rotation_matrix[0, 0]
                    - rotation_matrix[1, 1]
                )
                * 2.0
            )
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
            z = 0.25 * scale

    quaternion = (float(w), float(x), float(y), float(z))
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0 or not math.isfinite(norm):
        return None

    normalized = tuple(value / norm for value in quaternion)
    if not all(math.isfinite(value) for value in normalized):
        return None
    return (normalized[0], normalized[1], normalized[2], normalized[3])


def _yaw_pitch_roll(
    rotation_matrix: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    sy = math.sqrt(
        float(rotation_matrix[0, 0] * rotation_matrix[0, 0])
        + float(rotation_matrix[1, 0] * rotation_matrix[1, 0])
    )
    singular = sy < 1e-6
    if singular:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return float(yaw), float(pitch), float(roll)


def _method_name(
    facial_transform: MatrixTuple | None,
) -> str:
    if facial_transform is None:
        return PNP_METHOD_NAME
    return HEAD_POSE_METHOD_WITH_MEDIAPIPE_TRANSFORM


def _invalid_observation(
    *,
    facial_transform: MatrixTuple | None,
    pnp_landmarks: tuple[PnPLandmarkEvidence, ...],
    calibration: CalibrationRecord,
    reprojection_error_max_px: float,
    pnp_method: str | None,
    reprojection_error_px: float | None,
    message: str,
) -> HeadPoseObservation:
    return HeadPoseObservation(
        valid=False,
        method=_method_name(facial_transform),
        reason_invalid=ErrorCode.HEAD_POSE_INVALID,
        facial_transformation_matrix=facial_transform,
        pnp_method=pnp_method,
        pnp_landmarks=pnp_landmarks,
        pnp_point_count=len(pnp_landmarks),
        pnp_min_point_count=PNP_MIN_POINT_COUNT,
        canonical_points_source=CANONICAL_FACE_MODEL_POINTS_SOURCE,
        camera_intrinsics_policy=calibration.camera_intrinsics_policy,
        metric_translation_allowed=calibration.metric_translation_allowed,
        reprojection_error_px=reprojection_error_px,
        reprojection_error_max_px=reprojection_error_max_px,
        reprojection_error_threshold_name=PNP_REPROJECTION_ERROR_THRESHOLD_NAME,
        reprojection_error_threshold_source=PNP_REPROJECTION_ERROR_THRESHOLD_SOURCE,
        rotation_matrix=None,
        quaternion_wxyz=None,
        yaw_radians=None,
        pitch_radians=None,
        roll_radians=None,
        translation_camera_3d_m=None,
        errors=(
            ErrorRecord(
                code=ErrorCode.HEAD_POSE_INVALID,
                message=message,
            ),
        ),
    )


def _valid_transform_observation(
    *,
    facial_transform: MatrixTuple | None,
    transform_rotation_matrix: npt.NDArray[np.float64],
    pnp_landmarks: tuple[PnPLandmarkEvidence, ...],
    calibration: CalibrationRecord,
    reprojection_error_max_px: float,
    pnp_method: str | None,
    reprojection_error_px: float | None,
) -> HeadPoseObservation:
    rotation_matrix = _matrix_tuple(transform_rotation_matrix)
    quaternion = _quaternion_wxyz(transform_rotation_matrix)
    yaw_radians, pitch_radians, roll_radians = _yaw_pitch_roll(
        transform_rotation_matrix
    )
    if quaternion is None or not all(
        math.isfinite(value)
        for value in (*quaternion, yaw_radians, pitch_radians, roll_radians)
    ):
        return _invalid_observation(
            facial_transform=facial_transform,
            pnp_landmarks=pnp_landmarks,
            calibration=calibration,
            reprojection_error_max_px=reprojection_error_max_px,
            pnp_method=pnp_method,
            reprojection_error_px=reprojection_error_px,
            message="MediaPipe facial transform rotation conversion failed.",
        )

    return HeadPoseObservation(
        valid=True,
        method=_method_name(facial_transform),
        reason_invalid=None,
        facial_transformation_matrix=facial_transform,
        pnp_method=pnp_method,
        pnp_landmarks=pnp_landmarks,
        pnp_point_count=len(pnp_landmarks),
        pnp_min_point_count=PNP_MIN_POINT_COUNT,
        canonical_points_source=CANONICAL_FACE_MODEL_POINTS_SOURCE,
        camera_intrinsics_policy=calibration.camera_intrinsics_policy,
        metric_translation_allowed=calibration.metric_translation_allowed,
        reprojection_error_px=reprojection_error_px,
        reprojection_error_max_px=reprojection_error_max_px,
        reprojection_error_threshold_name=PNP_REPROJECTION_ERROR_THRESHOLD_NAME,
        reprojection_error_threshold_source=PNP_REPROJECTION_ERROR_THRESHOLD_SOURCE,
        rotation_matrix=rotation_matrix,
        quaternion_wxyz=quaternion,
        yaw_radians=yaw_radians,
        pitch_radians=pitch_radians,
        roll_radians=roll_radians,
        translation_camera_3d_m=None,
        errors=(),
    )
