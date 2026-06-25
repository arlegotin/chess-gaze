from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from chess_gaze.artifact_runs import create_run_layout, frame_id


def test_frame_id_is_zero_padded() -> None:
    assert frame_id(0) == "f000000000"
    assert frame_id(42) == "f000000042"


def test_run_layout_is_immutable_and_complete(tmp_path: Path) -> None:
    source = tmp_path / "input" / "clip.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")

    layout = create_run_layout(
        input_path=source,
        output_root=tmp_path / "output",
        clock=lambda: datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
        run_suffix="abcdef12",
    )

    assert layout.run_dir.name == "20260625T120000Z-abcdef12"
    assert layout.raw_frames_dir.is_dir()
    assert layout.processed_frames_dir.is_dir()
    assert layout.records_dir.is_dir()
    assert layout.face_crops_dir.is_dir()
    assert layout.left_eye_crops_dir.is_dir()
    assert layout.right_eye_crops_dir.is_dir()

    with pytest.raises(FrozenInstanceError):
        cast(Any, layout).run_dir = tmp_path


def test_create_run_layout_rejects_existing_run_dir(tmp_path: Path) -> None:
    source = tmp_path / "input" / "clip.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    output_root = tmp_path / "output"

    create_run_layout(
        input_path=source,
        output_root=output_root,
        clock=lambda: datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
        run_suffix="abcdef12",
    )

    with pytest.raises(FileExistsError):
        create_run_layout(
            input_path=source,
            output_root=output_root,
            clock=lambda: datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
            run_suffix="abcdef12",
        )


def test_relative_artifact_path_is_from_run_root(tmp_path: Path) -> None:
    source = tmp_path / "input" / "clip.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")

    layout = create_run_layout(
        input_path=source,
        output_root=tmp_path / "output",
        clock=lambda: datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
        run_suffix="abcdef12",
    )

    artifact_path = layout.raw_frames_dir / "f000000042.png"

    assert layout.relative_artifact_path(artifact_path) == Path(
        "raw_frames/f000000042.png"
    )
