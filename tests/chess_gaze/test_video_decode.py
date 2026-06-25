from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

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


def test_unsupported_video_raises_stable_error(tmp_path: Path) -> None:
    path = tmp_path / "not-a-video.mp4"
    path.write_text("not a video", encoding="utf-8")

    with pytest.raises(VideoDecodeError, match="Unsupported video") as exc_info:
        inspect_video(path)

    assert exc_info.value.code is CliErrorCode.UNSUPPORTED_VIDEO
