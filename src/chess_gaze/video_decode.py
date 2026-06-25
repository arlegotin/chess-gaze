from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

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
    def __init__(self, code: CliErrorCode | str, message: str) -> None:
        super().__init__(message)
        self.code = code


def inspect_video(path: Path) -> VideoInspection:
    source_sha256 = sha256_file(path)

    with _open_video_container(path) as container:
        stream = _video_stream_or_raise(container, path)
        frame_count_decoded = sum(1 for _ in container.decode(stream))

        return VideoInspection(
            source_path=path,
            source_sha256=source_sha256,
            video_manifest=VideoManifest(
                source_path=str(path),
                source_sha256=source_sha256,
                frame_width=stream.width,
                frame_height=stream.height,
            ),
            container_name=container.format.name,
            container_long_name=container.format.long_name,
            stream_index=stream.index,
            codec_name=stream.name or stream.codec_context.name,
            codec_profile=stream.profile,
            frame_width=stream.width,
            frame_height=stream.height,
            nominal_fps=_fraction_to_float(stream.average_rate),
            time_base=_fraction_to_string(stream.time_base),
            rotation_degrees=_rotation_from_stream(stream),
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

        for index, frame in enumerate(container.decode(stream)):
            yield DecodedFrame(
                frame_index=index,
                frame_id=frame_id(index),
                rgb=frame.to_ndarray(format="rgb24"),
                pts=frame.pts,
                pts_seconds=_frame_pts_seconds(frame),
                duration_seconds=_frame_duration_seconds(frame),
            )


def _open_video_container(path: Path) -> av.container.InputContainer:
    try:
        return av.open(str(path))
    except (FileNotFoundError, av.FFmpegError) as exc:
        raise VideoDecodeError(
            _unsupported_video_code(),
            f"Unsupported video input: {path}",
        ) from exc


def _video_stream_or_raise(
    container: av.container.InputContainer, path: Path
) -> av.video.stream.VideoStream:
    if not container.streams.video:
        raise VideoDecodeError(
            _unsupported_video_code(),
            f"Unsupported video input: {path}",
        )
    return container.streams.video[0]


def _unsupported_video_code() -> CliErrorCode | str:
    unsupported: object = getattr(CliErrorCode, "UNSUPPORTED_VIDEO", None)
    if isinstance(unsupported, str):
        return unsupported
    return "UNSUPPORTED_VIDEO"


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
        return int(raw_rotation)
    except ValueError:
        return None


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


def _frame_pts_seconds(frame: av.VideoFrame) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return float(frame.pts * frame.time_base)


def _frame_duration_seconds(frame: av.VideoFrame) -> float | None:
    if frame.duration is None or frame.time_base is None:
        return None
    return float(frame.duration * frame.time_base)
