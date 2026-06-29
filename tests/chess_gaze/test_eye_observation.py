from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from chess_gaze.artifact_runs import RunLayout, create_run_layout
from chess_gaze.errors import ErrorCode
from chess_gaze.eye_observation import (
    LEFT_EYE_CONTOUR_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYE_CONTOUR_INDICES,
    RIGHT_IRIS_INDICES,
    CropTransformToImagePx,
    observe_eyes,
)
from chess_gaze.face_observation import FaceCandidate
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D

IMAGE_WIDTH_PX = 200
IMAGE_HEIGHT_PX = 100
MEDIAPIPE_ANATOMICAL_LEFT_EYE_CONTOUR_INDICES = (
    263,
    387,
    385,
    362,
    380,
    373,
    374,
    386,
)
MEDIAPIPE_ANATOMICAL_RIGHT_EYE_CONTOUR_INDICES = (
    33,
    160,
    158,
    133,
    153,
    144,
    145,
    159,
)
MEDIAPIPE_ANATOMICAL_LEFT_IRIS_INDICES = (473, 474, 475, 476, 477)
MEDIAPIPE_ANATOMICAL_RIGHT_IRIS_INDICES = (468, 469, 470, 471, 472)
LANDMARK_COUNT = (
    max(
        *MEDIAPIPE_ANATOMICAL_LEFT_EYE_CONTOUR_INDICES,
        *MEDIAPIPE_ANATOMICAL_RIGHT_EYE_CONTOUR_INDICES,
        *MEDIAPIPE_ANATOMICAL_LEFT_IRIS_INDICES,
        *MEDIAPIPE_ANATOMICAL_RIGHT_IRIS_INDICES,
    )
    + 1
)


def test_eye_landmark_constants_follow_streamer_anatomy() -> None:
    assert LEFT_EYE_CONTOUR_INDICES == MEDIAPIPE_ANATOMICAL_LEFT_EYE_CONTOUR_INDICES
    assert RIGHT_EYE_CONTOUR_INDICES == MEDIAPIPE_ANATOMICAL_RIGHT_EYE_CONTOUR_INDICES
    assert LEFT_IRIS_INDICES == MEDIAPIPE_ANATOMICAL_LEFT_IRIS_INDICES
    assert RIGHT_IRIS_INDICES == MEDIAPIPE_ANATOMICAL_RIGHT_IRIS_INDICES


def test_observe_eyes_records_independent_eye_and_iris_measurements(
    tmp_path: Path,
) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(
        left_eye=EyeFixture(
            bbox=(125.0, 32.0, 170.0, 56.0),
            iris_center=(149.0, 42.0),
            iris_radius_x=9.0,
            iris_radius_y=6.0,
        ),
        right_eye=EyeFixture(
            bbox=(40.0, 30.0, 80.0, 50.0),
            iris_center=(57.0, 40.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
    )
    rgb_frame = gradient_rgb_frame()

    observation = observe_eyes(face, rgb_frame, run_layout, frame_id="f000000042")

    assert observation.frame_id == "f000000042"
    assert observation.left.present is True
    assert observation.right.present is True
    assert observation.left.reason_missing is None
    assert observation.right.reason_missing is None
    assert observation.left.iris_present is True
    assert observation.right.iris_present is True

    assert observation.left.crop_bbox_image_px is not None
    assert observation.right.crop_bbox_image_px is not None
    assert observation.left.eye_crop_transform_to_image_px is not None
    assert observation.right.eye_crop_transform_to_image_px is not None
    assert observation.left.eye_crop_path is None
    assert observation.right.eye_crop_path is None
    assert observation.left.eye_crop_sha256 is None
    assert observation.right.eye_crop_sha256 is None
    assert not run_layout.crops_dir.exists()
    assert list(run_layout.crops_dir.rglob("*.png")) == []

    assert observation.left.bounding_box_image_px is not None
    assert observation.right.bounding_box_image_px is not None
    assert observation.left.bounding_box_image_px.x_min == pytest.approx(125.0)
    assert observation.left.bounding_box_image_px.y_min == pytest.approx(32.0)
    assert observation.left.bounding_box_image_px.x_max == pytest.approx(170.0)
    assert observation.left.bounding_box_image_px.y_max == pytest.approx(56.0)
    assert observation.right.bounding_box_image_px.x_min == pytest.approx(40.0)
    assert observation.right.bounding_box_image_px.y_max == pytest.approx(50.0)

    assert observation.left.iris_center_image_px is not None
    assert observation.right.iris_center_image_px is not None
    assert observation.left.iris_center_image_px.x == pytest.approx(149.0)
    assert observation.left.iris_center_image_px.y == pytest.approx(42.0)
    assert observation.right.iris_center_image_px.x == pytest.approx(57.0)
    assert observation.right.iris_center_image_px.y == pytest.approx(40.0)
    assert observation.left.iris_diameter_px == pytest.approx(18.0)
    assert observation.right.iris_diameter_px == pytest.approx(12.0)

    assert observation.left.normalized_iris_offset_xy is not None
    assert observation.right.normalized_iris_offset_xy is not None
    assert observation.left.normalized_iris_offset_xy == pytest.approx(
        (1.0 / 15.0, -1.0 / 6.0)
    )
    assert observation.right.normalized_iris_offset_xy == pytest.approx((-0.15, 0.0))
    assert observation.left.eye_open_metric == pytest.approx(24.0 / 45.0)
    assert observation.right.eye_open_metric == pytest.approx(0.5)
    assert observation.left.occlusion == "none"
    assert observation.right.occlusion == "none"
    assert (
        observation.left.eye_landmarks_image_px
        != observation.right.eye_landmarks_image_px
    )
    assert (
        observation.left.iris_landmarks_image_px
        != observation.right.iris_landmarks_image_px
    )


def test_one_missing_eye_does_not_invalidate_the_other(tmp_path: Path) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(
        left_eye=EyeFixture.degenerate(),
        right_eye=EyeFixture(
            bbox=(40.0, 30.0, 80.0, 50.0),
            iris_center=(57.0, 40.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
    )

    observation = observe_eyes(
        face,
        gradient_rgb_frame(),
        run_layout,
        frame_id="f000000043",
    )

    assert observation.left.present is False
    assert observation.left.iris_present is False
    assert observation.left.reason_missing == ErrorCode.LEFT_EYE_NOT_FOUND
    assert observation.left.eye_crop_path is None
    assert observation.right.present is True
    assert observation.right.iris_present is True
    assert observation.right.reason_missing is None
    assert observation.right.crop_bbox_image_px is not None
    assert observation.right.eye_crop_transform_to_image_px is not None
    assert observation.right.eye_crop_path is None
    assert observation.right.eye_crop_sha256 is None


def test_missing_iris_keeps_eye_contour_with_explicit_reason(
    tmp_path: Path,
) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(
        left_eye=EyeFixture(
            bbox=(125.0, 32.0, 170.0, 56.0),
            iris_center=(0.0, 0.0),
            iris_radius_x=0.0,
            iris_radius_y=0.0,
        ),
        right_eye=EyeFixture(
            bbox=(40.0, 30.0, 80.0, 50.0),
            iris_center=(57.0, 40.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
    )

    observation = observe_eyes(
        face,
        gradient_rgb_frame(),
        run_layout,
        frame_id="f000000044",
    )

    assert observation.left.present is True
    assert observation.left.iris_present is False
    assert observation.left.reason_missing == ErrorCode.LEFT_IRIS_NOT_FOUND
    assert observation.left.iris_center_image_px is None
    assert observation.left.iris_diameter_px is None
    assert observation.left.crop_bbox_image_px is not None
    assert observation.left.eye_crop_transform_to_image_px is not None
    assert observation.left.eye_crop_path is None
    assert observation.left.eye_crop_sha256 is None
    assert observation.left.occlusion == "partial"
    assert observation.right.present is True
    assert observation.right.iris_present is True
    assert observation.right.reason_missing is None


def test_observe_eyes_retains_crop_files_when_requested(tmp_path: Path) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(
        left_eye=EyeFixture(
            bbox=(125.0, 32.0, 170.0, 56.0),
            iris_center=(149.0, 42.0),
            iris_radius_x=9.0,
            iris_radius_y=6.0,
        ),
        right_eye=EyeFixture(
            bbox=(40.0, 30.0, 80.0, 50.0),
            iris_center=(57.0, 40.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
    )

    observation = observe_eyes(
        face,
        gradient_rgb_frame(),
        run_layout,
        frame_id="f000000047",
        save_crop_images=True,
    )

    assert observation.left.eye_crop_path == Path("crops/eyes/left/f000000047.png")
    assert observation.right.eye_crop_path == Path("crops/eyes/right/f000000047.png")
    assert observation.left.eye_crop_sha256 is not None
    assert observation.right.eye_crop_sha256 is not None
    assert run_layout.crops_dir.is_dir()
    assert (run_layout.run_dir / observation.left.eye_crop_path).is_file()
    assert (run_layout.run_dir / observation.right.eye_crop_path).is_file()


@pytest.mark.parametrize("save_crop_images", [False, True])
@pytest.mark.parametrize(
    ("left_eye_bbox", "left_iris_center"),
    [
        ((240.0, 32.0, 280.0, 56.0), (260.0, 42.0)),
        ((125.0, 140.0, 170.0, 170.0), (149.0, 155.0)),
    ],
)
def test_fully_off_frame_eye_is_missing_without_empty_crop_write(
    tmp_path: Path,
    save_crop_images: bool,
    left_eye_bbox: tuple[float, float, float, float],
    left_iris_center: tuple[float, float],
) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(
        left_eye=EyeFixture(
            bbox=left_eye_bbox,
            iris_center=left_iris_center,
            iris_radius_x=7.0,
            iris_radius_y=5.0,
        ),
        right_eye=EyeFixture(
            bbox=(40.0, 30.0, 80.0, 50.0),
            iris_center=(57.0, 40.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
    )

    observation = observe_eyes(
        face,
        gradient_rgb_frame(),
        run_layout,
        frame_id="f000000048",
        save_crop_images=save_crop_images,
    )

    assert observation.left.present is False
    assert observation.left.reason_missing == ErrorCode.LEFT_EYE_NOT_FOUND
    assert observation.left.bounding_box_image_px is not None
    assert observation.left.crop_bbox_image_px is None
    assert observation.left.eye_crop_transform_to_image_px is None
    assert observation.left.eye_crop_path is None
    assert observation.left.eye_crop_sha256 is None
    assert observation.left.occlusion == "severe"

    assert observation.right.present is True
    assert observation.right.reason_missing is None
    if save_crop_images:
        assert observation.right.eye_crop_path == Path(
            "crops/eyes/right/f000000048.png"
        )
        assert (run_layout.run_dir / observation.right.eye_crop_path).is_file()
        assert not list(run_layout.left_eye_crops_dir.glob("*.png"))
    else:
        assert observation.right.eye_crop_path is None
        assert not run_layout.crops_dir.exists()


def test_partially_off_frame_eye_keeps_clipped_positive_crop(
    tmp_path: Path,
) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(
        left_eye=EyeFixture(
            bbox=(180.0, 32.0, 220.0, 56.0),
            iris_center=(192.0, 42.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
        right_eye=EyeFixture(
            bbox=(40.0, 30.0, 80.0, 50.0),
            iris_center=(57.0, 40.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
    )

    observation = observe_eyes(
        face,
        gradient_rgb_frame(),
        run_layout,
        frame_id="f000000049",
        save_crop_images=True,
    )

    assert observation.left.present is True
    assert observation.left.reason_missing is None
    assert observation.left.crop_bbox_image_px is not None
    assert observation.left.crop_bbox_image_px.x_min == pytest.approx(170.0)
    assert observation.left.crop_bbox_image_px.x_max == pytest.approx(
        float(IMAGE_WIDTH_PX)
    )
    assert observation.left.crop_bbox_image_px.x_max > (
        observation.left.crop_bbox_image_px.x_min
    )
    assert observation.left.eye_crop_path == Path("crops/eyes/left/f000000049.png")
    assert (run_layout.run_dir / observation.left.eye_crop_path).is_file()


def test_crop_transform_maps_crop_coordinates_back_to_image_px(
    tmp_path: Path,
) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(
        left_eye=EyeFixture(
            bbox=(125.0, 32.0, 170.0, 56.0),
            iris_center=(149.0, 42.0),
            iris_radius_x=9.0,
            iris_radius_y=6.0,
        ),
        right_eye=EyeFixture(
            bbox=(40.0, 30.0, 80.0, 50.0),
            iris_center=(57.0, 40.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
    )

    observation = observe_eyes(
        face,
        gradient_rgb_frame(),
        run_layout,
        frame_id="f000000045",
    )

    left = observation.left
    assert left.crop_bbox_image_px is not None
    assert left.eye_crop_transform_to_image_px is not None
    assert left.eye_crop_transform_to_image_px.source_space == "left_eye_crop_px"
    assert left.eye_crop_transform_to_image_px.target_space == CoordinateSpace.IMAGE_PX

    mapped_origin = map_crop_point(left.eye_crop_transform_to_image_px, 0.0, 0.0)
    mapped_iris = map_crop_point(
        left.eye_crop_transform_to_image_px,
        left.iris_center_image_px.x - left.crop_bbox_image_px.x_min,  # type: ignore[union-attr]
        left.iris_center_image_px.y - left.crop_bbox_image_px.y_min,  # type: ignore[union-attr]
    )

    assert mapped_origin == pytest.approx(
        (left.crop_bbox_image_px.x_min, left.crop_bbox_image_px.y_min)
    )
    assert mapped_iris == pytest.approx((149.0, 42.0))


def test_closed_or_insufficient_landmarks_produce_explicit_reason_codes(
    tmp_path: Path,
) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(
        left_eye=EyeFixture(
            bbox=(125.0, 41.7, 170.0, 42.3),
            iris_center=(149.0, 42.0),
            iris_radius_x=9.0,
            iris_radius_y=0.2,
        ),
        right_eye=EyeFixture.insufficient_contour(
            bbox=(40.0, 30.0, 80.0, 50.0),
            iris_center=(57.0, 40.0),
            iris_radius_x=6.0,
            iris_radius_y=5.0,
        ),
    )

    observation = observe_eyes(
        face,
        gradient_rgb_frame(),
        run_layout,
        frame_id="f000000046",
    )

    assert observation.left.present is False
    assert observation.left.reason_missing == ErrorCode.LEFT_EYE_NOT_FOUND
    assert observation.left.eye_open_metric is not None
    assert observation.left.occlusion == "severe"
    assert observation.right.present is False
    assert observation.right.reason_missing == ErrorCode.RIGHT_EYE_NOT_FOUND
    assert observation.right.occlusion == "unknown"


class EyeFixture:
    def __init__(
        self,
        *,
        bbox: tuple[float, float, float, float],
        iris_center: tuple[float, float],
        iris_radius_x: float,
        iris_radius_y: float,
        contour_points: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        self.bbox = bbox
        self.iris_center = iris_center
        self.iris_radius_x = iris_radius_x
        self.iris_radius_y = iris_radius_y
        self.contour_points = contour_points

    @classmethod
    def degenerate(cls) -> EyeFixture:
        return cls(
            bbox=(0.0, 0.0, 0.0, 0.0),
            iris_center=(0.0, 0.0),
            iris_radius_x=0.0,
            iris_radius_y=0.0,
        )

    @classmethod
    def insufficient_contour(
        cls,
        *,
        bbox: tuple[float, float, float, float],
        iris_center: tuple[float, float],
        iris_radius_x: float,
        iris_radius_y: float,
    ) -> EyeFixture:
        return cls(
            bbox=bbox,
            iris_center=iris_center,
            iris_radius_x=iris_radius_x,
            iris_radius_y=iris_radius_y,
            contour_points=((125.0, 44.0), (170.0, 44.0)),
        )


def make_run_layout(tmp_path: Path) -> RunLayout:
    source = tmp_path / "input" / "clip.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    return create_run_layout(
        input_path=source,
        output_root=tmp_path / "output",
        clock=lambda: datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
        run_suffix="abcdef12",
    )


def make_face_candidate(
    *,
    left_eye: EyeFixture,
    right_eye: EyeFixture,
) -> FaceCandidate:
    landmark_px = [
        Point2D(space=CoordinateSpace.IMAGE_PX, x=0.0, y=0.0)
        for _ in range(LANDMARK_COUNT)
    ]
    landmark_norm = [
        Point2D(space=CoordinateSpace.NORMALIZED, x=0.0, y=0.0)
        for _ in range(LANDMARK_COUNT)
    ]

    set_eye_landmarks(
        landmark_px,
        landmark_norm,
        eye_indices=MEDIAPIPE_ANATOMICAL_LEFT_EYE_CONTOUR_INDICES,
        iris_indices=MEDIAPIPE_ANATOMICAL_LEFT_IRIS_INDICES,
        fixture=left_eye,
    )
    set_eye_landmarks(
        landmark_px,
        landmark_norm,
        eye_indices=MEDIAPIPE_ANATOMICAL_RIGHT_EYE_CONTOUR_INDICES,
        iris_indices=MEDIAPIPE_ANATOMICAL_RIGHT_IRIS_INDICES,
        fixture=right_eye,
    )

    return FaceCandidate(
        candidate_id="face_0",
        frame_id="f000000042",
        image_width_px=IMAGE_WIDTH_PX,
        image_height_px=IMAGE_HEIGHT_PX,
        candidate_score=None,
        score_source="not_exposed_by_mediapipe_face_landmarker",
        bounding_box_image_px=BBox(
            space=CoordinateSpace.IMAGE_PX,
            x_min=20.0,
            y_min=10.0,
            x_max=180.0,
            y_max=90.0,
        ),
        bounding_box_image_norm=BBox(
            space=CoordinateSpace.NORMALIZED,
            x_min=0.1,
            y_min=0.1,
            x_max=0.9,
            y_max=0.9,
        ),
        landmarks_image_px=landmark_px,
        landmarks_image_norm=landmark_norm,
    )


def set_eye_landmarks(
    landmark_px: list[Point2D],
    landmark_norm: list[Point2D],
    *,
    eye_indices: tuple[int, ...],
    iris_indices: tuple[int, ...],
    fixture: EyeFixture,
) -> None:
    contour_points = fixture.contour_points or contour_from_bbox(fixture.bbox)
    for index, point in zip(eye_indices, contour_points, strict=False):
        set_landmark(landmark_px, landmark_norm, index, point)

    center_x, center_y = fixture.iris_center
    iris_points = (
        (center_x, center_y),
        (center_x - fixture.iris_radius_x, center_y),
        (center_x + fixture.iris_radius_x, center_y),
        (center_x, center_y - fixture.iris_radius_y),
        (center_x, center_y + fixture.iris_radius_y),
    )
    for index, point in zip(iris_indices, iris_points, strict=False):
        set_landmark(landmark_px, landmark_norm, index, point)


def set_landmark(
    landmark_px: list[Point2D],
    landmark_norm: list[Point2D],
    index: int,
    point: tuple[float, float],
) -> None:
    x, y = point
    landmark_px[index] = Point2D(space=CoordinateSpace.IMAGE_PX, x=x, y=y)
    landmark_norm[index] = Point2D(
        space=CoordinateSpace.NORMALIZED,
        x=x / IMAGE_WIDTH_PX,
        y=y / IMAGE_HEIGHT_PX,
    )


def contour_from_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float], ...]:
    x_min, y_min, x_max, y_max = bbox
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    return (
        (x_min, y_mid),
        (x_min + ((x_mid - x_min) * 0.5), y_min),
        (x_mid, y_min),
        (x_mid + ((x_max - x_mid) * 0.5), y_min),
        (x_max, y_mid),
        (x_mid + ((x_max - x_mid) * 0.5), y_max),
        (x_mid, y_max),
        (x_min + ((x_mid - x_min) * 0.5), y_max),
    )


def gradient_rgb_frame() -> np.ndarray:
    frame = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(IMAGE_WIDTH_PX, dtype=np.uint8)
    frame[:, :, 1] = np.arange(IMAGE_HEIGHT_PX, dtype=np.uint8).reshape(-1, 1)
    frame[:, :, 2] = 64
    return frame


def map_crop_point(
    transform: CropTransformToImagePx,
    x: float,
    y: float,
) -> tuple[float, float]:
    return (
        (transform.m00 * x) + (transform.m01 * y) + transform.m02,
        (transform.m10 * x) + (transform.m11 * y) + transform.m12,
    )
