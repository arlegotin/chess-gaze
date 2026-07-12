from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from chess_gaze import video_decode
from chess_gaze.errors import CliErrorCode
from chess_gaze.video_decode import (
    VideoDecodeError,
    inspect_video,
    iter_decoded_frames,
)


def make_tiny_video(path: Path, frame_count: int = 3) -> None:
    import av
    import numpy as np

    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=3)
    stream.width = 8
    stream.height = 6
    stream.pix_fmt = "yuv420p"
    for index in range(frame_count):
        image = np.full((6, 8, 3), index * 40, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


ORIENTATION_MARKER_RGB = np.asarray(
    (
        ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
        ((255, 255, 0), (255, 0, 255), (0, 255, 255)),
    ),
    dtype=np.uint8,
)


def make_oriented_marker_video(path: Path, rotation_degrees: int) -> None:
    import av

    upright = np.repeat(np.repeat(ORIENTATION_MARKER_RGB, 64, axis=0), 64, axis=1)
    stored = np.rot90(upright, k=-(rotation_degrees // 90))
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=1)
    stream.width = stored.shape[1]
    stream.height = stored.shape[0]
    stream.pix_fmt = "yuv420p"
    stream.set_display_rotation(rotation_degrees)
    for packet in stream.encode(av.VideoFrame.from_ndarray(stored, format="rgb24")):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_inspect_video_reports_expected_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    make_tiny_video(path)

    inspection = inspect_video(path)

    assert inspection.source_path == path
    assert inspection.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inspection.video_manifest.source_path == str(path)
    assert inspection.video_manifest.source_sha256 == inspection.source_sha256
    assert inspection.video_manifest.frame_width == 8
    assert inspection.video_manifest.frame_height == 6
    assert inspection.video_manifest.frame_count_decoded == 3
    assert "mp4" in inspection.container_name
    assert inspection.container_long_name
    assert inspection.codec_name == "mpeg4"
    assert inspection.frame_width == 8
    assert inspection.frame_height == 6
    assert inspection.nominal_fps == pytest.approx(3.0)
    assert inspection.time_base is not None
    assert Fraction(inspection.time_base) > 0
    assert inspection.rotation_degrees is None
    assert inspection.pixel_format == "yuv420p"
    assert inspection.color_range is None or inspection.color_range >= 0
    assert inspection.color_space is None or inspection.color_space >= 0
    assert inspection.frame_count_expected == 3
    assert inspection.frame_count_decoded == 3
    assert inspection.pyav_version
    assert inspection.ffmpeg_versions["libavcodec"]
    assert inspection.ffmpeg_versions["libavformat"]


def test_inspect_video_records_stable_usable_pts_identity(tmp_path: Path) -> None:
    path = tmp_path / "tiny_short.mp4"
    make_tiny_video(path)

    inspection = inspect_video(path)
    second = inspect_video(path)

    assert inspection.video_manifest.pts_sequence_usable is True
    assert len(inspection.video_manifest.pts_sequence_sha256 or "") == 64
    assert second.video_manifest.pts_sequence_sha256 == (
        inspection.video_manifest.pts_sequence_sha256
    )


@pytest.mark.parametrize(
    "frames",
    [
        [SimpleNamespace(pts=None, time_base=Fraction(1, 30))],
        [
            SimpleNamespace(pts=1, time_base=Fraction(1, 30)),
            SimpleNamespace(pts=1, time_base=Fraction(1, 30)),
        ],
        [
            SimpleNamespace(pts=2, time_base=Fraction(1, 30)),
            SimpleNamespace(pts=1, time_base=Fraction(1, 30)),
        ],
        [SimpleNamespace(pts=1, time_base=Fraction(0, 1))],
    ],
    ids=["missing-pts", "duplicate-pts", "decreasing-pts", "non-positive-time-base"],
)
def test_decoded_pts_identity_marks_untrustworthy_sequences_unusable(
    frames: list[SimpleNamespace],
) -> None:
    count, digest, usable = video_decode._decoded_pts_identity(cast(Any, frames))

    assert count == len(frames)
    assert len(digest) == 64
    assert usable is False


def test_iter_decoded_frames_yields_full_rgb_frames_in_order(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    make_tiny_video(path, frame_count=4)

    frames = list(iter_decoded_frames(path))

    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert [frame.frame_id for frame in frames] == [
        "f000000000",
        "f000000001",
        "f000000002",
        "f000000003",
    ]
    assert all(frame.rgb.shape == (6, 8, 3) for frame in frames)
    assert all(frame.rgb.dtype == np.uint8 for frame in frames)
    assert all(frame.pts is not None for frame in frames)
    pts_values = [frame.pts for frame in frames if frame.pts is not None]
    assert pts_values == sorted(pts_values)
    assert len(set(pts_values)) == len(frames)
    assert all(frame.pts_seconds is None or frame.pts_seconds >= 0 for frame in frames)
    pts_seconds = [
        frame.pts_seconds for frame in frames if frame.pts_seconds is not None
    ]
    assert pts_seconds == sorted(pts_seconds)
    assert all(
        frame.duration_seconds is None or frame.duration_seconds > 0 for frame in frames
    )
    pixel_values = [int(frame.rgb[0, 0, 0]) for frame in frames]
    assert pixel_values == sorted(pixel_values)
    assert pixel_values[0] == pytest.approx(0, abs=2)
    assert pixel_values[-1] == pytest.approx(120, abs=4)


@pytest.mark.parametrize("rotation_degrees", [0, 90, 180, 270])
def test_video_decode_applies_display_orientation(
    tmp_path: Path, rotation_degrees: int
) -> None:
    import av

    path = tmp_path / f"rotation_{rotation_degrees}_short.mp4"
    make_oriented_marker_video(path, rotation_degrees)
    with av.open(str(path)) as container:
        stored_rgb = next(container.decode(container.streams.video[0])).to_ndarray(
            format="rgb24"
        )

    inspection = inspect_video(path)
    decoded = next(iter_decoded_frames(path)).rgb

    assert inspection.rotation_degrees == (
        None if rotation_degrees == 0 else rotation_degrees
    )
    assert inspection.video_manifest.frame_width == 192
    assert inspection.video_manifest.frame_height == 128
    assert decoded.shape == (128, 192, 3)
    samples = np.asarray(
        [
            [decoded[row * 64 + 32, column * 64 + 32] for column in range(3)]
            for row in range(2)
        ]
    )
    np.testing.assert_allclose(samples, ORIENTATION_MARKER_RGB, atol=2)
    if rotation_degrees == 0:
        np.testing.assert_array_equal(decoded, stored_rgb)


def test_video_decode_rejects_non_right_angle_display_rotation() -> None:
    with pytest.raises(VideoDecodeError, match="45 degrees"):
        video_decode._right_angle_rotation(45)


def test_unsupported_video_raises_stable_error(tmp_path: Path) -> None:
    path = tmp_path / "not-a-video.mp4"
    path.write_text("not a video", encoding="utf-8")

    with pytest.raises(VideoDecodeError, match="Unsupported video") as exc_info:
        inspect_video(path)

    assert exc_info.value.code is CliErrorCode.UNSUPPORTED_VIDEO
