from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
from av.sidedata.sidedata import Type as SideDataType

from chess_gaze.artifact_runs import frame_id
from chess_gaze.errors import CliErrorCode
from chess_gaze.frame_records import VideoManifest
from chess_gaze.model_assets import sha256_file


@dataclass(frozen=True)
class VideoInspection:
    source_path: Path
    source_sha256: str
    video_manifest: VideoManifest
    container_name: str
    container_long_name: str | None
    stream_index: int
    codec_name: str
    codec_profile: str | None
    frame_width: int
    frame_height: int
    nominal_fps: float | None
    time_base: str | None
    rotation_degrees: int | None
    pixel_format: str | None
    color_range: int | None
    color_space: int | None
    pyav_version: str
    ffmpeg_versions: dict[str, str]
    frame_count_expected: int | None
    frame_count_decoded: int


@dataclass(frozen=True)
class DecodedFrame:
    frame_index: int
    frame_id: str
    rgb: np.ndarray
    pts: int | None
    pts_seconds: float | None
    duration_seconds: float | None


class VideoDecodeError(RuntimeError):
    def __init__(self, code: CliErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def inspect_video(path: Path) -> VideoInspection:
    source_sha256 = sha256_file(path)

    with _open_video_container(path) as container:
        stream = _video_stream_or_raise(container, path)
        stream_rotation = _rotation_from_stream(stream)
        rotation_degrees = stream_rotation

        def inspected_frames() -> Iterator[av.VideoFrame]:
            nonlocal rotation_degrees
            for frame, resolved_rotation in _frames_with_rotation(
                container.decode(stream), stream_rotation
            ):
                rotation_degrees = resolved_rotation
                yield frame

        (
            frame_count_decoded,
            pts_sequence_sha256,
            pts_sequence_usable,
        ) = _decoded_pts_identity(inspected_frames())
        frame_width, frame_height = _oriented_dimensions(
            stream.width, stream.height, rotation_degrees
        )

        return VideoInspection(
            source_path=path,
            source_sha256=source_sha256,
            video_manifest=VideoManifest(
                source_path=str(path),
                source_sha256=source_sha256,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_count_decoded=frame_count_decoded,
                pts_sequence_sha256=pts_sequence_sha256,
                pts_sequence_usable=pts_sequence_usable,
            ),
            container_name=container.format.name,
            container_long_name=container.format.long_name,
            stream_index=stream.index,
            codec_name=stream.name or stream.codec_context.name,
            codec_profile=stream.profile,
            frame_width=frame_width,
            frame_height=frame_height,
            nominal_fps=_fraction_to_float(stream.average_rate),
            time_base=_fraction_to_string(stream.time_base),
            rotation_degrees=rotation_degrees,
            pixel_format=stream.pix_fmt,
            color_range=_codec_value(stream.codec_context.color_range),
            color_space=_codec_value(stream.codec_context.colorspace),
            pyav_version=av.__version__,
            ffmpeg_versions=_ffmpeg_versions(),
            frame_count_expected=_frame_count_hint(stream.frames),
            frame_count_decoded=frame_count_decoded,
        )


def iter_decoded_frames(path: Path) -> Iterator[DecodedFrame]:
    with _open_video_container(path) as container:
        stream = _video_stream_or_raise(container, path)
        stream_rotation = _rotation_from_stream(stream)

        for index, (frame, rotation_degrees) in enumerate(
            _frames_with_rotation(container.decode(stream), stream_rotation)
        ):
            yield DecodedFrame(
                frame_index=index,
                frame_id=frame_id(index),
                rgb=_oriented_rgb(frame, rotation_degrees),
                pts=frame.pts,
                pts_seconds=_frame_pts_seconds(frame),
                duration_seconds=_frame_duration_seconds(frame),
            )


def _open_video_container(path: Path) -> av.container.InputContainer:
    try:
        return av.open(str(path))
    except (FileNotFoundError, av.FFmpegError) as exc:
        raise VideoDecodeError(
            CliErrorCode.UNSUPPORTED_VIDEO,
            f"Unsupported video input: {path}",
        ) from exc


def _video_stream_or_raise(
    container: av.container.InputContainer, path: Path
) -> av.video.stream.VideoStream:
    if not container.streams.video:
        raise VideoDecodeError(
            CliErrorCode.UNSUPPORTED_VIDEO,
            f"Unsupported video input: {path}",
        )
    return container.streams.video[0]


def _ffmpeg_versions() -> dict[str, str]:
    return {
        library: ".".join(str(part) for part in version)
        for library, version in av.library_versions.items()
    }


def _rotation_from_stream(stream: av.video.stream.VideoStream) -> int | None:
    raw_rotation = stream.metadata.get("rotate")
    if raw_rotation is None:
        return None

    try:
        return _right_angle_rotation(int(raw_rotation))
    except ValueError:
        return None


def _rotation_from_frame(frame: av.VideoFrame) -> int | None:
    side_data = cast(Any, frame.side_data)
    if SideDataType.DISPLAYMATRIX not in side_data:
        return None
    return _right_angle_rotation(frame.rotation)


def _frames_with_rotation(
    frames: Iterable[av.VideoFrame], stream_rotation: int | None
) -> Iterator[tuple[av.VideoFrame, int | None]]:
    previous_rotation: int | None = None
    for frame in frames:
        frame_rotation = _rotation_from_frame(frame)
        rotation = stream_rotation if frame_rotation is None else frame_rotation
        numeric_rotation = rotation or 0
        if previous_rotation is not None and numeric_rotation != previous_rotation:
            raise VideoDecodeError(
                CliErrorCode.UNSUPPORTED_VIDEO,
                "Video display rotation changes during decode",
            )
        previous_rotation = numeric_rotation
        yield frame, rotation


def _right_angle_rotation(rotation_degrees: int) -> int:
    normalized = rotation_degrees % 360
    if normalized % 90:
        raise VideoDecodeError(
            CliErrorCode.UNSUPPORTED_VIDEO,
            f"Unsupported video display rotation: {rotation_degrees} degrees",
        )
    return normalized


def _oriented_dimensions(
    width: int, height: int, rotation_degrees: int | None
) -> tuple[int, int]:
    return (height, width) if rotation_degrees in (90, 270) else (width, height)


def _oriented_rgb(frame: av.VideoFrame, rotation_degrees: int | None) -> np.ndarray:
    rgb = frame.to_ndarray(format="rgb24")
    if not rotation_degrees:
        return rgb
    return np.ascontiguousarray(np.rot90(rgb, k=rotation_degrees // 90))


def _frame_count_hint(frame_count: int) -> int | None:
    if frame_count <= 0:
        return None
    return frame_count


def _fraction_to_float(value: Fraction | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _fraction_to_string(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _codec_value(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _decoded_pts_identity(
    frames: Iterable[av.VideoFrame],
) -> tuple[int, str, bool]:
    digest = hashlib.sha256()
    count = 0
    usable = True
    previous_seconds: float | None = None
    for frame in frames:
        count += 1
        time_base = frame.time_base
        time_base_text = (
            "null"
            if time_base is None
            else f"{time_base.numerator}/{time_base.denominator}"
        )
        digest.update(f"{frame.pts}\t{time_base_text}\n".encode())
        seconds = _frame_pts_seconds(frame)
        if (
            seconds is None
            or not math.isfinite(seconds)
            or (previous_seconds is not None and seconds <= previous_seconds)
        ):
            usable = False
        previous_seconds = seconds
    return count, digest.hexdigest(), usable and count > 0


def _frame_pts_seconds(frame: av.VideoFrame) -> float | None:
    if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
        return None
    return float(frame.pts * frame.time_base)


def _frame_duration_seconds(frame: av.VideoFrame) -> float | None:
    if frame.duration is None or frame.time_base is None:
        return None
    return float(frame.duration * frame.time_base)
