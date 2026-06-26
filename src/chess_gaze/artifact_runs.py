from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex


def frame_id(frame_index: int) -> str:
    return f"f{frame_index:09d}"


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    raw_frames_dir: Path
    processed_frames_dir: Path
    crops_dir: Path
    face_crops_dir: Path
    eyes_crops_dir: Path
    left_eye_crops_dir: Path
    right_eye_crops_dir: Path
    records_dir: Path

    @property
    def scene_dir(self) -> Path:
        return self.run_dir / "scene"

    @property
    def viewer_dir(self) -> Path:
        return self.run_dir / "viewer"

    def relative_artifact_path(self, artifact_path: Path) -> Path:
        return artifact_path.relative_to(self.run_dir)


def _run_id(created_at: datetime, run_suffix: str) -> str:
    created_at_utc = created_at.astimezone(UTC)
    return f"{created_at_utc.strftime('%Y%m%dT%H%M%SZ')}-{run_suffix}"


def create_run_layout(
    input_path: Path,
    output_root: Path,
    clock: Callable[[], datetime],
    run_suffix: str | None = None,
) -> RunLayout:
    del input_path

    output_root.mkdir(parents=True, exist_ok=True)
    created_at = clock()
    resolved_run_suffix = run_suffix or token_hex(4)
    run_dir = output_root / _run_id(created_at, resolved_run_suffix)
    run_dir.mkdir()

    raw_frames_dir = run_dir / "raw_frames"
    processed_frames_dir = run_dir / "processed_frames"
    crops_dir = run_dir / "crops"
    face_crops_dir = crops_dir / "face"
    eyes_crops_dir = crops_dir / "eyes"
    left_eye_crops_dir = eyes_crops_dir / "left"
    right_eye_crops_dir = eyes_crops_dir / "right"
    records_dir = run_dir / "records"
    scene_dir = run_dir / "scene"
    viewer_dir = run_dir / "viewer"

    for directory in (
        raw_frames_dir,
        processed_frames_dir,
        face_crops_dir,
        left_eye_crops_dir,
        right_eye_crops_dir,
        records_dir,
        scene_dir,
        viewer_dir,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    return RunLayout(
        run_dir=run_dir,
        raw_frames_dir=raw_frames_dir,
        processed_frames_dir=processed_frames_dir,
        crops_dir=crops_dir,
        face_crops_dir=face_crops_dir,
        eyes_crops_dir=eyes_crops_dir,
        left_eye_crops_dir=left_eye_crops_dir,
        right_eye_crops_dir=right_eye_crops_dir,
        records_dir=records_dir,
    )
