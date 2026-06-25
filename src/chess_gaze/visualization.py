from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from chess_gaze.errors import FrameStatus
from chess_gaze.frame_records import EyeRecord, FrameRecord, GazeAngles, HeadPoseRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.image_io import save_bgr_jpeg

Color = tuple[int, int, int]
Pixel = tuple[int, int]

_FACE_COLOR: Color = (0, 255, 120)
_LANDMARK_COLOR: Color = (255, 220, 0)
_LEFT_EYE_COLOR: Color = (80, 180, 255)
_RIGHT_EYE_COLOR: Color = (255, 120, 80)
_IRIS_CENTER_COLOR: Color = (255, 255, 255)
_IRIS_LANDMARK_COLOR: Color = (160, 255, 255)
_GEOMETRIC_GAZE_COLOR: Color = (255, 0, 255)
_APPEARANCE_GAZE_COLOR: Color = (0, 220, 255)
_RECOMMENDED_GAZE_COLOR: Color = (255, 255, 255)
_HEAD_X_COLOR: Color = (255, 80, 80)
_HEAD_Y_COLOR: Color = (80, 255, 80)
_HEAD_Z_COLOR: Color = (80, 160, 255)
_TEXT_COLOR: Color = (255, 255, 255)
_TEXT_SHADOW_COLOR: Color = (0, 0, 0)
_ERROR_COLOR: Color = (255, 90, 90)
_OK_COLOR: Color = (90, 255, 140)


def render_processed_frame(
    rgb_frame: np.ndarray, record: FrameRecord, output_path: Path, quality: int
) -> str:
    """Draw validated frame-record overlays on an RGB frame and persist a JPEG."""

    _validate_rgb_frame(rgb_frame)
    _validate_quality(quality)

    canvas = rgb_frame.copy()

    if record.face.bounding_box is not None:
        _draw_box(canvas, record.face.bounding_box, _FACE_COLOR)

    if record.face.landmarks:
        for landmark in record.face.landmarks:
            _draw_circle(canvas, landmark, _LANDMARK_COLOR, radius=2, filled=True)

    _draw_eye(canvas, record.left_eye, _LEFT_EYE_COLOR, "L")
    _draw_eye(canvas, record.right_eye, _RIGHT_EYE_COLOR, "R")

    _draw_eye_gaze_vectors(canvas, record)
    _draw_face_gaze_vectors(canvas, record)
    _draw_head_pose(canvas, record)
    _draw_status_text(canvas, record)

    # save_bgr_jpeg owns the OpenCV BGR conversion boundary; canvas remains RGB here.
    return save_bgr_jpeg(output_path, canvas, quality)


def _validate_rgb_frame(rgb_frame: np.ndarray) -> None:
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError("rgb_frame must have shape (height, width, 3)")
    if rgb_frame.dtype != np.uint8:
        raise ValueError("rgb_frame must have dtype uint8")
    if rgb_frame.shape[0] <= 0 or rgb_frame.shape[1] <= 0:
        raise ValueError("rgb_frame must have nonzero width and height")


def _validate_quality(quality: int) -> None:
    if quality < 1 or quality > 100:
        raise ValueError("quality must be between 1 and 100")


def _draw_box(
    image: np.ndarray, bbox: BBox, color: Color, *, thickness: int = 2
) -> None:
    top_left, bottom_right = _bbox_pixels(bbox, image)
    cv2.rectangle(image, top_left, bottom_right, color, thickness, lineType=cv2.LINE_AA)


def _draw_circle(
    image: np.ndarray,
    point: Point2D,
    color: Color,
    *,
    radius: int,
    filled: bool = False,
) -> None:
    center = _point_pixel(point, image)
    thickness = -1 if filled else 1
    cv2.circle(image, center, radius, color, thickness, lineType=cv2.LINE_AA)


def _draw_eye(image: np.ndarray, eye: EyeRecord, color: Color, label: str) -> None:
    if eye.bounding_box is not None:
        _draw_box(image, eye.bounding_box, color, thickness=1)

    if eye.iris_landmarks:
        iris_pixels = [_point_pixel(point, image) for point in eye.iris_landmarks]
        if len(iris_pixels) >= 3:
            contour = np.array(iris_pixels, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [contour], isClosed=True, color=color, thickness=1)
        for point in eye.iris_landmarks:
            _draw_circle(image, point, _IRIS_LANDMARK_COLOR, radius=1, filled=True)

    if eye.pupil_center is not None:
        center = _point_pixel(eye.pupil_center, image)
        cv2.circle(
            image,
            center,
            4,
            _IRIS_CENTER_COLOR,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(image, center, 6, color, 1, lineType=cv2.LINE_AA)
        _draw_text(
            image,
            label,
            (center[0] + 7, center[1] + 4),
            color=color,
            scale=0.35,
        )


def _draw_eye_gaze_vectors(image: np.ndarray, record: FrameRecord) -> None:
    for eye in (record.left_eye, record.right_eye):
        if eye.pupil_center is None:
            continue
        _draw_gaze_vector(
            image,
            eye.pupil_center,
            record.geometric_gaze,
            _GEOMETRIC_GAZE_COLOR,
            label=None,
        )


def _draw_face_gaze_vectors(image: np.ndarray, record: FrameRecord) -> None:
    origin = _face_center(record)
    if origin is None:
        return

    _draw_gaze_vector(
        image,
        origin,
        record.appearance_gaze,
        _APPEARANCE_GAZE_COLOR,
        label="UniGaze",
    )
    _draw_gaze_vector(
        image,
        origin,
        record.recommended_gaze,
        _RECOMMENDED_GAZE_COLOR,
        label="rec",
    )


def _draw_gaze_vector(
    image: np.ndarray,
    origin: Point2D,
    gaze: GazeAngles,
    color: Color,
    *,
    label: str | None,
) -> None:
    if not gaze.valid or gaze.yaw_radians is None or gaze.pitch_radians is None:
        return

    start = _point_pixel(origin, image)
    length = max(24.0, min(image.shape[:2]) * 0.25)
    end = _clip_pixel(
        start[0] + round(gaze.yaw_radians * length),
        start[1] - round(gaze.pitch_radians * length),
        image,
    )
    cv2.arrowedLine(image, start, end, color, 2, line_type=cv2.LINE_AA, tipLength=0.25)
    if label is not None:
        _draw_text(image, label, (end[0] + 4, end[1] - 4), color=color, scale=0.35)


def _draw_head_pose(image: np.ndarray, record: FrameRecord) -> None:
    if not _has_complete_head_pose(record.head_pose):
        return

    origin = _nose_or_face_center(record)
    if origin is None:
        return

    start = _point_pixel(origin, image)
    face_box = record.face.bounding_box
    if face_box is None:
        length = max(24.0, min(image.shape[:2]) * 0.18)
    else:
        (x_min, y_min), (x_max, y_max) = _bbox_pixels(face_box, image)
        length = max(24.0, min(x_max - x_min, y_max - y_min) * 0.35)

    roll = record.head_pose.roll_radians
    yaw = record.head_pose.yaw_radians
    pitch = record.head_pose.pitch_radians
    assert roll is not None
    assert yaw is not None
    assert pitch is not None
    x_end = _clip_pixel(
        start[0] + round(math.cos(roll) * length),
        start[1] + round(math.sin(roll) * length),
        image,
    )
    y_end = _clip_pixel(
        start[0] - round(math.sin(roll) * length),
        start[1] + round(math.cos(roll) * length),
        image,
    )
    z_end = _clip_pixel(
        start[0] + round(yaw * length),
        start[1] - round(pitch * length),
        image,
    )
    cv2.arrowedLine(image, start, x_end, _HEAD_X_COLOR, 2, line_type=cv2.LINE_AA)
    cv2.arrowedLine(image, start, y_end, _HEAD_Y_COLOR, 2, line_type=cv2.LINE_AA)
    cv2.arrowedLine(image, start, z_end, _HEAD_Z_COLOR, 2, line_type=cv2.LINE_AA)


def _draw_status_text(image: np.ndarray, record: FrameRecord) -> None:
    status_color = _OK_COLOR if record.status is FrameStatus.OK else _ERROR_COLOR
    lines = [
        f"{record.frame_id} status={record.status.value}",
        f"idx={record.frame_index} t={record.timestamp_seconds:.3f}s",
    ]

    if _has_complete_head_pose(record.head_pose):
        lines.append(
            "head "
            f"yaw={record.head_pose.yaw_radians:.3f} "
            f"pitch={record.head_pose.pitch_radians:.3f} "
            f"roll={record.head_pose.roll_radians:.3f}"
        )
    elif record.head_pose.reason_invalid is not None:
        lines.append(f"head={record.head_pose.reason_invalid.value}")

    error_codes = [error.code.value for error in record.errors]
    if not error_codes:
        error_codes = [
            reason.value
            for reason in (
                record.face.reason_invalid,
                record.left_eye.reason_invalid,
                record.right_eye.reason_invalid,
                record.recommended_gaze.reason_invalid,
            )
            if reason is not None
        ]
    if error_codes:
        lines.append("errors=" + ",".join(error_codes[:3]))

    for index, line in enumerate(lines):
        _draw_text(
            image,
            line,
            (8, 18 + index * 18),
            color=status_color if index == 0 else _TEXT_COLOR,
            scale=0.45,
        )


def _has_complete_head_pose(head_pose: HeadPoseRecord) -> bool:
    return (
        head_pose.valid
        and head_pose.yaw_radians is not None
        and head_pose.pitch_radians is not None
        and head_pose.roll_radians is not None
    )


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: Pixel,
    *,
    color: Color,
    scale: float,
) -> None:
    clipped_origin = _clip_pixel(origin[0], origin[1], image)
    cv2.putText(
        image,
        text,
        (clipped_origin[0] + 1, clipped_origin[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _TEXT_SHADOW_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        clipped_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _face_center(record: FrameRecord) -> Point2D | None:
    bbox = record.face.bounding_box
    if bbox is None:
        return None
    if bbox.space is CoordinateSpace.NORMALIZED:
        return Point2D(
            space=CoordinateSpace.NORMALIZED,
            x=(bbox.x_min + bbox.x_max) / 2.0,
            y=(bbox.y_min + bbox.y_max) / 2.0,
        )
    return Point2D(
        space=CoordinateSpace.IMAGE_PX,
        x=(bbox.x_min + bbox.x_max) / 2.0,
        y=(bbox.y_min + bbox.y_max) / 2.0,
    )


def _nose_or_face_center(record: FrameRecord) -> Point2D | None:
    if record.face.landmarks and len(record.face.landmarks) >= 3:
        return record.face.landmarks[2]
    return _face_center(record)


def _bbox_pixels(bbox: BBox, image: np.ndarray) -> tuple[Pixel, Pixel]:
    height, width = image.shape[:2]
    if bbox.space is CoordinateSpace.NORMALIZED:
        x_min = bbox.x_min * width
        y_min = bbox.y_min * height
        x_max = bbox.x_max * width
        y_max = bbox.y_max * height
    else:
        x_min = bbox.x_min
        y_min = bbox.y_min
        x_max = bbox.x_max
        y_max = bbox.y_max

    top_left = _clip_pixel(round(x_min), round(y_min), image)
    bottom_right = _clip_pixel(round(x_max), round(y_max), image)
    return top_left, bottom_right


def _point_pixel(point: Point2D, image: np.ndarray) -> Pixel:
    height, width = image.shape[:2]
    if point.space is CoordinateSpace.NORMALIZED:
        return _clip_pixel(
            round(point.x * (width - 1)),
            round(point.y * (height - 1)),
            image,
        )
    return _clip_pixel(round(point.x), round(point.y), image)


def _clip_pixel(x: int, y: int, image: np.ndarray) -> Pixel:
    height, width = image.shape[:2]
    clipped_x = min(max(x, 0), width - 1)
    clipped_y = min(max(y, 0), height - 1)
    return clipped_x, clipped_y
