# Frame-Level Gaze Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local `chess-gaze analyze <video_path>` pipeline that decodes every frame, preserves raw evidence, records strict per-frame face/eye/head/gaze observations, and writes QA artifacts without temporal smoothing.

**Architecture:** Implement deep domain modules under `src/chess_gaze/` only where they own real invariants: CLI/preflight, strict record schemas, artifact runs, model assets, video decode, image IO, calibration, observations, visualization, pipeline orchestration, and QA summary. Heavy ML integrations are isolated behind small protocol-style wrappers so most tests run with deterministic fakes while real model smoke checks remain opt-in when ignored local assets exist.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, mypy, PyAV, MediaPipe Face Landmarker, OpenCV headless, NumPy, Pydantic v2, Pillow, PyTorch, torchvision, timm, UniGaze `unigaze_h14_joint`, safetensors, huggingface_hub.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Use installed Superpowers skills for development flow.
- Use `uv` for dependency resolution, locking, environment management, and command execution.
- The public command is `uv run chess-gaze analyze <video_path>`.
- Default output root is `artifacts/output/`.
- Default model root is `models/`.
- Accept initial optional arguments `--output-root`, `--models-root`, and `--config`.
- Analyze one input video per command invocation; no batch mode.
- Reject missing, unreadable, unsupported, model-missing, checksum-mismatched, or license-unapproved inputs before creating a run directory.
- Repo owner granted license/use approval for UniGaze `unigaze_h14_joint` under MG-NC-RAI-2.0 on 2026-06-25; implementation must record that approval in registry or config metadata.
- Never download model assets during analysis.
- `HF_TOKEN` may be read from `.env` only by explicit setup/prefetch tooling for Hugging Face model asset downloads.
- `HF_TOKEN` must never be required by `chess-gaze analyze`, logged, persisted into artifacts, or transmitted during analysis.
- No other secret or credential is required to start implementation.
- Do not send video frames, crops, metadata, or model inputs to a remote service.
- Preserve every decoded source frame; no sampling, skipping, dropping, deduplication, tracking, temporal smoothing, interpolation, or across-frame averaging.
- Raw frames are lossless PNG by default.
- Processed visualization frames are JPEG by default with `processed_frame_jpeg_quality=95`.
- Frame IDs are zero-padded decoder-emission IDs in presentation order: `f000000000`, `f000000001`.
- Every decoded frame has exactly one raw frame, one processed frame, and one `records/frames.jsonl` line.
- Failed observations still produce a frame record and processed frame.
- Every coordinate-bearing field uses a named coordinate space.
- Do not populate metric `camera_3d_m` translation unless calibrated intrinsics and a scale source exist.
- UniGaze `unigaze_h14_joint` is the selected learned face-level gaze model.
- Store UniGaze `pred_gaze[:, 0]` as pitch radians and `pred_gaze[:, 1]` as yaw radians.
- Store UniGaze confidence as null with `confidence_source="not_provided_by_unigaze"`.
- Do not copy face-level UniGaze gaze into independent left/right eye fields.
- Keep `target_image_px`, `target_board_norm`, and `target_square` null in this implementation.
- Use MediaPipe Face Landmarker in `IMAGE` mode for measurement output.
- Do not assume MediaPipe exposes per-candidate confidence; scores are nullable with explicit score provenance.
- Keep left and right eye observations independent.
- Keep local videos, model binaries, and generated artifacts ignored.
- Standard gates after implementation are `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy`.

---

## File Structure

- Modify `pyproject.toml` to add the `chess-gaze` console script and runtime dependencies.
- Modify `uv.lock` through `uv lock` or `uv sync`.
- Modify `src/chess_gaze/__init__.py` only if package exports change.
- Create `src/chess_gaze/cli.py` for argument parsing, command dispatch, and stable CLI exit codes.
- Create `src/chess_gaze/configuration.py` for named defaults and JSON config loading.
- Create `src/chess_gaze/errors.py` for pre-run CLI errors and frame-time error codes.
- Create `src/chess_gaze/geometry.py` for typed coordinate, bbox, transform, and angle helpers.
- Create `src/chess_gaze/frame_records.py` for strict Pydantic schemas and JSONL validation.
- Create `src/chess_gaze/artifact_runs.py` for immutable run directory creation and artifact path ownership.
- Create `src/chess_gaze/image_io.py` for RGB/BGR boundaries, image encoding, hashing, and atomic writes.
- Create `src/chess_gaze/model_assets.py` for committed registry loading and local asset validation.
- Create `src/chess_gaze/model_registry.json` as the committed model registry authority.
- Create `.env.example` documenting `HF_TOKEN` as the only supported secret-like setup variable.
- Create `src/chess_gaze/video_decode.py` for PyAV source inspection and frame iteration.
- Create `src/chess_gaze/calibration.py` for constants, camera assumptions, landmark indices, and derived setup summaries.
- Create `src/chess_gaze/face_observation.py` for MediaPipe adapter, candidate capture, and selection.
- Create `src/chess_gaze/eye_observation.py` for per-eye/iris measurement and crop transforms.
- Create `src/chess_gaze/head_pose.py` for MediaPipe transform preservation and PnP pose evidence.
- Create `src/chess_gaze/gaze_observation.py` for per-eye geometry, UniGaze, and recommended gaze synthesis.
- Create `src/chess_gaze/visualization.py` for processed frame overlays.
- Create `src/chess_gaze/qa_summary.py` for run aggregation, validation, and deterministic QA samples.
- Create `src/chess_gaze/pipeline.py` for end-to-end orchestration.
- Create tests mirroring the package modules under `tests/chess_gaze/`.
- Create `docs/superpowers/closeouts/2026-06-25-frame-gaze-analysis-pipeline.md` after implementation verification.

Dependency sequencing rule: do not install the full ML stack at the start. Add
runtime dependencies in the task that first consumes them, and keep the
model-free artifact contract testable before MediaPipe and UniGaze are wired.
If execution is parallelized, the fake-observer parts of Task 12 may be pulled
forward immediately after Task 5; they must not require MediaPipe, UniGaze, or
real model assets.

### Task 1: CLI Skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/chess_gaze/cli.py`
- Modify: `src/chess_gaze/__init__.py`
- Create: `tests/chess_gaze/test_cli.py`

**Interfaces:**
- Produces: `chess_gaze.cli.main(argv: list[str] | None = None) -> int`
- Produces: console script `chess-gaze = "chess_gaze.cli:main"`
- Consumes: no domain modules yet

- [ ] **Step 1: Add console script**

Do not add runtime dependencies in this task. Add this script entry to `pyproject.toml`:

```toml
[project.scripts]
chess-gaze = "chess_gaze.cli:main"
```

Run:

```sh
UV_CACHE_DIR=.uv-cache uv lock
```

Expected: `uv.lock` resolves for Python 3.12 with no new runtime dependency.

- [ ] **Step 2: Write failing CLI tests**

Create `tests/chess_gaze/test_cli.py`:

```python
from pathlib import Path

from chess_gaze.cli import main


def test_analyze_requires_video_path(capsys) -> None:
    exit_code = main(["analyze"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "video_path" in captured.err


def test_unknown_command_returns_usage(capsys) -> None:
    exit_code = main(["unknown"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "usage:" in captured.err


def test_missing_input_returns_stable_error_without_output_dir(
    tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "missing.mp4"
    output_root = tmp_path / "output"

    exit_code = main(
        [
            "analyze",
            str(missing),
            "--output-root",
            str(output_root),
            "--models-root",
            str(tmp_path / "models"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 10
    assert "INPUT_NOT_FOUND" in captured.err
    assert not output_root.exists()
```

- [ ] **Step 3: Run CLI tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py -q
```

Expected: tests fail because `chess_gaze.cli` does not exist.

- [ ] **Step 4: Implement minimal CLI**

Create `src/chess_gaze/cli.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

INPUT_NOT_FOUND_EXIT = 10
USAGE_EXIT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-gaze")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("video_path")
    analyze.add_argument("--output-root", default="artifacts/output")
    analyze.add_argument("--models-root", default="models")
    analyze.add_argument("--config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    if args.command != "analyze":
        parser.print_usage(sys.stderr)
        return USAGE_EXIT

    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"INPUT_NOT_FOUND: {video_path}", file=sys.stderr)
        return INPUT_NOT_FOUND_EXIT

    print("Pipeline implementation is not wired yet", file=sys.stderr)
    return 1
```

- [ ] **Step 5: Run focused CLI tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py -q
```

Expected: all tests in `tests/chess_gaze/test_cli.py` pass.

- [ ] **Step 6: Commit**

Run:

```sh
git add pyproject.toml uv.lock src/chess_gaze/cli.py src/chess_gaze/__init__.py tests/chess_gaze/test_cli.py
git commit -m "feat: add analyze cli skeleton"
```

Expected: commit succeeds.

### Task 2: Strict Schema, Errors, and Geometry Foundations

**Files:**
- Create: `src/chess_gaze/errors.py`
- Create: `src/chess_gaze/geometry.py`
- Create: `src/chess_gaze/frame_records.py`
- Create: `tests/chess_gaze/test_frame_records.py`

**Interfaces:**
- Produces: `ErrorCode`, `CliErrorCode`, `FrameStatus`
- Produces: `Point2D`, `BBox`, `Transform2D`, `RotationRadians`
- Produces: `FrameRecord`, `RunManifest`, `VideoManifest`, `CalibrationRecord`, `ErrorRecord`
- Consumes: none from later tasks

- [ ] **Step 1: Add schema dependency**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv add pydantic
```

Expected: `pydantic` is added to `pyproject.toml` and `uv.lock`.

- [ ] **Step 2: Write failing strict-schema tests**

Create `tests/chess_gaze/test_frame_records.py`:

```python
import math

import pytest
from pydantic import ValidationError

from chess_gaze.errors import ErrorCode
from chess_gaze.frame_records import FrameRecord, GazeAngles
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D


def test_bbox_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValidationError):
        BBox(space=CoordinateSpace.IMAGE_PX, x_min=20, y_min=10, x_max=10, y_max=40)


def test_point_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        Point2D(space=CoordinateSpace.IMAGE_PX, x=math.nan, y=1.0)


def test_gaze_valid_requires_pitch_and_yaw() -> None:
    with pytest.raises(ValidationError):
        GazeAngles(valid=True, yaw_radians=None, pitch_radians=0.1, reason_invalid=None)


def test_frame_record_rejects_unknown_fields(valid_frame_record_dict: dict) -> None:
    valid_frame_record_dict["unknown"] = "rejected"

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_error_code_names_are_stable() -> None:
    assert ErrorCode.FACE_NOT_FOUND.value == "FACE_NOT_FOUND"
    assert ErrorCode.GAZE_ESTIMATORS_DISAGREE.value == "GAZE_ESTIMATORS_DISAGREE"
```

Add a `valid_frame_record_dict` fixture in the same file with one valid minimal frame record containing `face.present=False`, independent invalid eyes, invalid gaze layers, and explicit reason codes.

- [ ] **Step 3: Run schema tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py -q
```

Expected: tests fail because the schema modules do not exist.

- [ ] **Step 4: Implement strict schema foundations**

Create the modules with these exact invariants:

```python
model_config = ConfigDict(extra="forbid", strict=True)
```

Use `@field_validator` or `@model_validator` to reject:

- NaN and infinity in every float field.
- `BBox.x_max <= BBox.x_min` or `BBox.y_max <= BBox.y_min`.
- `valid=True` gaze without both yaw and pitch.
- `present=True` face or eye records without required landmarks.
- unknown fields.

Use enums for all stable codes and statuses. Include these enum values:

```python
class ErrorCode(StrEnum):
    FRAME_DECODE_FAILED = "FRAME_DECODE_FAILED"
    RAW_FRAME_WRITE_FAILED = "RAW_FRAME_WRITE_FAILED"
    PROCESSED_FRAME_WRITE_FAILED = "PROCESSED_FRAME_WRITE_FAILED"
    FACE_NOT_FOUND = "FACE_NOT_FOUND"
    MULTIPLE_FACE_CANDIDATES = "MULTIPLE_FACE_CANDIDATES"
    PRIMARY_FACE_LOW_SCORE = "PRIMARY_FACE_LOW_SCORE"
    LEFT_EYE_NOT_FOUND = "LEFT_EYE_NOT_FOUND"
    RIGHT_EYE_NOT_FOUND = "RIGHT_EYE_NOT_FOUND"
    LEFT_IRIS_NOT_FOUND = "LEFT_IRIS_NOT_FOUND"
    RIGHT_IRIS_NOT_FOUND = "RIGHT_IRIS_NOT_FOUND"
    HEAD_POSE_INVALID = "HEAD_POSE_INVALID"
    GAZE_MODEL_FAILED = "GAZE_MODEL_FAILED"
    GAZE_ESTIMATORS_DISAGREE = "GAZE_ESTIMATORS_DISAGREE"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
```

- [ ] **Step 5: Run schema tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py -q
```

Expected: all schema tests pass.

- [ ] **Step 6: Commit**

Run:

```sh
git add src/chess_gaze/errors.py src/chess_gaze/geometry.py src/chess_gaze/frame_records.py tests/chess_gaze/test_frame_records.py
git commit -m "feat: add strict artifact schemas"
```

Expected: commit succeeds.

### Task 3: Artifact Runs and Atomic Image IO

**Files:**
- Create: `src/chess_gaze/artifact_runs.py`
- Create: `src/chess_gaze/image_io.py`
- Create: `tests/chess_gaze/test_artifact_runs.py`
- Create: `tests/chess_gaze/test_image_io.py`

**Interfaces:**
- Consumes: `RunManifest` from `frame_records`
- Produces: `RunLayout`, `create_run_layout(input_path: Path, output_root: Path, clock: Callable[[], datetime]) -> RunLayout`
- Produces: `atomic_write_bytes(path: Path, data: bytes) -> None`
- Produces: `save_rgb_png(path: Path, image: np.ndarray) -> str`
- Produces: `save_bgr_jpeg(path: Path, image: np.ndarray, quality: int) -> str`

- [ ] **Step 1: Add image and array dependencies**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv add numpy pillow opencv-python-headless
```

Expected: `numpy`, `pillow`, and `opencv-python-headless` are added. No `opencv-python` GUI provider is added.

- [ ] **Step 2: Write failing artifact tests**

Create `tests/chess_gaze/test_artifact_runs.py`:

```python
from datetime import UTC, datetime
from pathlib import Path

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
```

Create `tests/chess_gaze/test_image_io.py`:

```python
from pathlib import Path

import numpy as np

from chess_gaze.image_io import atomic_write_bytes, save_rgb_png


def test_atomic_write_replaces_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"

    atomic_write_bytes(target, b"abc")

    assert target.read_bytes() == b"abc"
    assert not list(tmp_path.glob("*.tmp"))


def test_save_rgb_png_returns_sha256(tmp_path: Path) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    digest = save_rgb_png(tmp_path / "frame.png", image)

    assert len(digest) == 64
    assert (tmp_path / "frame.png").is_file()
```

- [ ] **Step 3: Run artifact tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_image_io.py -q
```

Expected: tests fail because artifact and image IO modules do not exist.

- [ ] **Step 4: Implement run layout and atomic writes**

Implement:

- all directories in the spec output tree;
- no overwrite when run directory already exists;
- relative artifact path helper;
- `.tmp` write in the target directory followed by `Path.replace`;
- RGB PNG via Pillow;
- BGR JPEG via OpenCV boundary conversion and explicit quality;
- SHA-256 of bytes actually written.

- [ ] **Step 5: Run artifact tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_image_io.py -q
```

Expected: all artifact tests pass.

- [ ] **Step 6: Commit**

Run:

```sh
git add src/chess_gaze/artifact_runs.py src/chess_gaze/image_io.py tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_image_io.py
git commit -m "feat: add artifact run layout"
```

Expected: commit succeeds.

### Task 4: Configuration and Model Asset Gate

**Files:**
- Create: `.env.example`
- Create: `src/chess_gaze/configuration.py`
- Create: `src/chess_gaze/model_assets.py`
- Create: `src/chess_gaze/model_registry.json`
- Create: `tests/chess_gaze/test_configuration.py`
- Create: `tests/chess_gaze/test_model_assets.py`

**Interfaces:**
- Consumes: `CliErrorCode` from `errors`
- Produces: `AnalysisConfig`
- Produces: `load_config(path: Path | None) -> AnalysisConfig`
- Produces: `load_env_file(path: Path = Path(".env")) -> dict[str, str]`
- Produces: `load_model_registry(path: Path) -> ModelRegistry`
- Produces: `validate_required_assets(registry: ModelRegistry, models_root: Path, approved_licenses: set[str]) -> list[ResolvedModelAsset]`
- Produces: `prefetch_model_asset(model_id: str, registry: ModelRegistry, models_root: Path, hf_token: str | None) -> ResolvedModelAsset`

- [ ] **Step 1: Write failing config and model asset tests**

Create tests that assert:

- defaults match the spec;
- unknown config keys are rejected;
- ignored `models/manifest.json` cannot add a model absent from `model_registry.json`;
- missing required assets raise `MODEL_ASSET_MISSING`;
- checksum mismatch raises `MODEL_ASSET_CHECKSUM_MISMATCH`;
- unapproved `MG-NC-RAI-2.0` raises `MODEL_LICENSE_NOT_APPROVED`;
- UniGaze `unigaze_h14_joint` registry metadata records `license_approved=true`, `license_approved_by="repo_owner"`, and `license_approved_at="2026-06-25"`;
- fixture assets with matching checksums resolve to local paths.
- `.env` parsing reads `HF_TOKEN` without logging it;
- `prefetch_model_asset` accepts `HF_TOKEN` for setup-time downloads but the analysis validation path does not require it.

Use tiny fixture files in `tmp_path` with SHA-256 computed inside the test:

```python
import hashlib


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
```

- [ ] **Step 2: Run asset tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_model_assets.py -q
```

Expected: tests fail because configuration and model asset modules do not exist.

- [ ] **Step 3: Implement config and registry validation**

Create `src/chess_gaze/model_registry.json` with registry entries for:

- `mediapipe-face-landmarker`
- `unigaze-h14-joint`

The committed registry must include model IDs, task names, expected relative paths, license names, approval requirement, approval metadata, input/output contract, source URLs, and a checksum field. For UniGaze `unigaze_h14_joint`, record the repo owner's 2026-06-25 MG-NC-RAI-2.0 intended-use approval explicitly. If the real local model files are not present during implementation, keep analysis preflight failing for real runs and use test fixture registries for passing automated tests. Do not add guessed production checksums.

Implement a separate local `models/manifest.json` reader only as installed-path evidence. It must never create or override committed registry entries.

Create `.env.example` with:

```dotenv
# Optional setup-time token for explicit Hugging Face model prefetch only.
# chess-gaze analyze must run from local verified model files and must not use this.
HF_TOKEN=
```

Do not add any other required secret variable.

- [ ] **Step 4: Run asset tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_model_assets.py -q
```

Expected: all asset tests pass.

- [ ] **Step 5: Commit**

Run:

```sh
git add .env.example src/chess_gaze/configuration.py src/chess_gaze/model_assets.py src/chess_gaze/model_registry.json tests/chess_gaze/test_configuration.py tests/chess_gaze/test_model_assets.py
git commit -m "feat: add model asset gate"
```

Expected: commit succeeds.

### Task 5: PyAV Video Inspection and Decode

**Files:**
- Create: `src/chess_gaze/video_decode.py`
- Create: `tests/chess_gaze/test_video_decode.py`

**Interfaces:**
- Consumes: `VideoManifest` from `frame_records`
- Produces: `inspect_video(path: Path) -> VideoInspection`
- Produces: `iter_decoded_frames(path: Path) -> Iterator[DecodedFrame]`
- Produces: `DecodedFrame.frame_index: int`, `DecodedFrame.frame_id: str`, `DecodedFrame.rgb: np.ndarray`

- [ ] **Step 1: Add video dependency**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv add av
```

Expected: `av` is added to `pyproject.toml` and `uv.lock`.

- [ ] **Step 2: Write failing decode tests**

Create tests that:

- write a tiny synthetic video with PyAV into `tmp_path`;
- inspect it and assert dimensions, codec/container fields, nullable timing fields, PyAV version, and FFmpeg versions;
- decode all frames and assert frame IDs and RGB array shape;
- assert unsupported input raises `UNSUPPORTED_VIDEO`.

Use this helper in the test:

```python
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
```

- [ ] **Step 3: Run decode tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_video_decode.py -q
```

Expected: tests fail because `video_decode.py` does not exist.

- [ ] **Step 4: Implement PyAV inspection and frame iteration**

Implement:

- source SHA-256;
- container, stream, codec, dimensions, nominal FPS, time base, rotation, pixel format, color range, and color space;
- nullable `pts`, `pts_seconds`, and `duration_seconds`;
- `frame_count_expected` as metadata hint only;
- `frame_count_decoded` as actual decode count;
- conversion to full-frame RGB arrays;
- no sampling, dropping, dedupe, or duplicate suppression.

- [ ] **Step 5: Run decode tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_video_decode.py -q
```

Expected: all decode tests pass.

- [ ] **Step 6: Commit**

Run:

```sh
git add src/chess_gaze/video_decode.py tests/chess_gaze/test_video_decode.py
git commit -m "feat: add faithful video decoding"
```

Expected: commit succeeds.

### Task 6: Calibration Defaults and Derived Setup Summary

**Files:**
- Create: `src/chess_gaze/calibration.py`
- Create: `tests/chess_gaze/test_calibration.py`

**Interfaces:**
- Consumes: `CalibrationRecord` from `frame_records`
- Produces: `default_calibration() -> CalibrationRecord`
- Produces: `derive_setup_constants(records: Iterable[FrameRecord]) -> DerivedSetupConstants`

- [ ] **Step 1: Write failing calibration tests**

Create tests asserting:

- all constants from the spec are named and persisted;
- PnP landmark indices exactly match `nose_tip=1`, `chin=152`, `left_eye_outer=33`, `right_eye_outer=263`, `left_eye_inner=133`, `right_eye_inner=362`, `left_mouth_corner=61`, and `right_mouth_corner=291`;
- `metric_translation_allowed` defaults to false;
- derived setup constants do not rewrite any per-frame gaze or eye fields.

- [ ] **Step 2: Run calibration tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_calibration.py -q
```

Expected: tests fail because `calibration.py` does not exist.

- [ ] **Step 3: Implement calibration module**

Implement immutable named defaults:

- `raw_frame_image_format="png"`
- `processed_frame_image_format="jpg"`
- `processed_frame_jpeg_quality=95`
- `max_face_candidates=4`
- `candidate_face_score_min=0.25`
- `usable_face_score_min=0.50`
- `usable_eye_confidence_min=0.50`
- `default_iris_diameter_mm=11.7`
- `default_iris_diameter_uncertainty_mm=0.5`
- `unigaze_input_size_px=224`
- `unigaze_output_order="pitch_yaw_radians"`
- `face_landmarker_running_mode="IMAGE"`

- [ ] **Step 4: Run calibration tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_calibration.py -q
```

Expected: all calibration tests pass.

- [ ] **Step 5: Commit**

Run:

```sh
git add src/chess_gaze/calibration.py tests/chess_gaze/test_calibration.py
git commit -m "feat: add calibration defaults"
```

Expected: commit succeeds.

### Task 7: Face Observation and Candidate Selection

**Files:**
- Create: `src/chess_gaze/face_observation.py`
- Create: `tests/chess_gaze/test_face_observation.py`

**Interfaces:**
- Consumes: geometry types and calibration defaults
- Produces: `FaceObserver`
- Produces: `select_primary_face(candidates: Sequence[FaceCandidate], calibration: CalibrationRecord) -> FaceSelection`
- Produces: `MediaPipeFaceObserver.observe(rgb_frame: np.ndarray) -> FaceObservation`

- [ ] **Step 1: Add MediaPipe dependency**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv add mediapipe
```

Expected: `mediapipe` is added only when the Face Landmarker adapter is implemented.

- [ ] **Step 2: Write failing face selection tests**

Create tests that cover:

- single candidate selection;
- multiple candidates with real scores use `candidate_score * candidate_area_fraction`;
- nullable scores use area-only selection and `selection_score_source="area_only_no_model_score"`;
- all candidates are preserved;
- no valid landmarks produces `FACE_NOT_FOUND`;
- `MULTIPLE_FACE_CANDIDATES` is emitted when more than one candidate exists.

- [ ] **Step 3: Run face tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py -q
```

Expected: tests fail because `face_observation.py` does not exist.

- [ ] **Step 4: Implement face candidate data and MediaPipe adapter**

Implement:

- a pure `select_primary_face` function independent of MediaPipe;
- a lazy MediaPipe import inside `MediaPipeFaceObserver`;
- Face Landmarker options persisted as data:
  - running mode `IMAGE`;
  - `num_faces=max_face_candidates`;
  - `min_face_detection_confidence`;
  - `min_face_presence_confidence`;
  - blendshapes enabled;
  - facial transformation matrices enabled;
- nullable score handling with explicit score source.

- [ ] **Step 5: Run face tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py -q
```

Expected: all face tests pass without requiring the real `.task` model.

- [ ] **Step 6: Commit**

Run:

```sh
git add src/chess_gaze/face_observation.py tests/chess_gaze/test_face_observation.py
git commit -m "feat: add face observation selection"
```

Expected: commit succeeds.

### Task 8: Eye and Iris Observation

**Files:**
- Create: `src/chess_gaze/eye_observation.py`
- Create: `tests/chess_gaze/test_eye_observation.py`

**Interfaces:**
- Consumes: selected face landmarks from `face_observation`
- Consumes: `save_rgb_png` from `image_io`
- Produces: `observe_eyes(face: SelectedFace, rgb_frame: np.ndarray, run_layout: RunLayout, frame_id: str) -> EyePairObservation`

- [ ] **Step 1: Write failing eye observation tests**

Create tests with synthetic MediaPipe-style landmarks that assert:

- left and right eye records are independent;
- one missing eye does not invalidate the other;
- iris center and diameter are computed from iris landmarks;
- eye crop path is relative to the run directory;
- eye crop transform maps crop coordinates back to `image_px`;
- closed or insufficient landmarks produce explicit reason codes.

- [ ] **Step 2: Run eye tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_eye_observation.py -q
```

Expected: tests fail because `eye_observation.py` does not exist.

- [ ] **Step 3: Implement eye and iris measurements**

Implement named landmark groups for each eye and iris. Store:

- `present`;
- `confidence` and `confidence_source`;
- `reason_missing`;
- eye and iris landmarks in `image_px` and `image_norm`;
- iris center and diameter in pixels;
- eyelid/eye contour bbox;
- crop path;
- crop transform back to `image_px`;
- normalized iris offset;
- eye open metric;
- occlusion state.

- [ ] **Step 4: Run eye tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_eye_observation.py -q
```

Expected: all eye observation tests pass.

- [ ] **Step 5: Commit**

Run:

```sh
git add src/chess_gaze/eye_observation.py tests/chess_gaze/test_eye_observation.py
git commit -m "feat: add eye and iris observation"
```

Expected: commit succeeds.

### Task 9: Head Pose Evidence

**Files:**
- Create: `src/chess_gaze/head_pose.py`
- Create: `tests/chess_gaze/test_head_pose.py`

**Interfaces:**
- Consumes: selected face landmarks and calibration PnP indices
- Produces: `estimate_head_pose(face: SelectedFace, calibration: CalibrationRecord, image_size: ImageSize) -> HeadPoseObservation`

- [ ] **Step 1: Write failing head-pose tests**

Create tests that assert:

- MediaPipe facial transformation matrix is preserved when provided;
- PnP uses the exact named landmark indices from calibration;
- metric translation remains null when intrinsics are unavailable;
- invalid point count produces `HEAD_POSE_INVALID`;
- invalid reprojection error produces `HEAD_POSE_INVALID`;
- valid rotations are stored as matrix, quaternion, and yaw/pitch/roll radians.

- [ ] **Step 2: Run head-pose tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_head_pose.py -q
```

Expected: tests fail because `head_pose.py` does not exist.

- [ ] **Step 3: Implement head pose module**

Implement:

- primary preservation of MediaPipe transform matrix;
- PnP only when enough named landmarks are present;
- OpenCV `solvePnP` with explicit method name stored;
- finite matrix checks;
- named reprojection threshold from calibration;
- no metric translation when `metric_translation_allowed` is false.

- [ ] **Step 4: Run head-pose tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_head_pose.py -q
```

Expected: all head-pose tests pass.

- [ ] **Step 5: Commit**

Run:

```sh
git add src/chess_gaze/head_pose.py tests/chess_gaze/test_head_pose.py
git commit -m "feat: add head pose evidence"
```

Expected: commit succeeds.

### Task 10: Gaze Observation and UniGaze Local Wrapper

**Files:**
- Create: `src/chess_gaze/gaze_observation.py`
- Create: `tests/chess_gaze/test_gaze_observation.py`

**Interfaces:**
- Consumes: eye observations, head pose, model assets, and image crops
- Produces: `compute_per_eye_geometric_gaze(eye: EyeObservation, head_pose: HeadPoseObservation) -> GazeAngles`
- Produces: `UniGazeModel.from_local_asset(asset: ResolvedModelAsset, device: str) -> UniGazeModel`
- Produces: `UniGazeModel.predict(normalized_batch: torch.Tensor) -> FaceModelGaze`
- Produces: `synthesize_recommended_gaze(left: GazeAngles, right: GazeAngles, face: FaceModelGaze, thresholds: GazeThresholds) -> RecommendedGaze`

- [ ] **Step 1: Add UniGaze runtime dependencies**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv add torch torchvision timm "unigaze==0.1.3" safetensors huggingface_hub
```

Expected: UniGaze runtime dependencies are added only after the local-only model wrapper is under test. Dependency resolution must remain Python 3.12 compatible.

- [ ] **Step 2: Write failing gaze tests**

Create tests that assert:

- `pred_gaze[:, 0]` maps to pitch radians;
- `pred_gaze[:, 1]` maps to yaw radians;
- UniGaze confidence is null with `confidence_source="not_provided_by_unigaze"`;
- the wrapper never calls `unigaze.load`;
- the wrapper never calls `hf_hub_download`;
- local asset path is passed into the weight-loading path;
- per-eye gaze fields remain independent;
- recommended gaze becomes invalid with `GAZE_ESTIMATORS_DISAGREE` when disagreement exceeds a named threshold;
- board and screen targets remain null.

Use monkeypatch fakes for the UniGaze backend so no real model file is required.

- [ ] **Step 3: Run gaze tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py -q
```

Expected: tests fail because `gaze_observation.py` does not exist.

- [ ] **Step 4: Implement gaze module**

Implement:

- per-eye geometric apparent-gaze proxy from iris offset and head pose;
- `FaceModelGaze` with method `unigaze_h14_joint`;
- local-only UniGaze loading through explicit asset path;
- `HF_HUB_OFFLINE=1` inside the UniGaze loading boundary;
- rejection of helper code paths that call `unigaze.load`;
- normalization transform recording for the 224x224 UniGaze input;
- pitch/yaw to unit vector conversion using radians;
- recommended-gaze synthesis with named disagreement threshold.

- [ ] **Step 5: Run gaze tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py -q
```

Expected: all gaze tests pass without downloading or loading the real H14 weights.

- [ ] **Step 6: Commit**

Run:

```sh
git add src/chess_gaze/gaze_observation.py tests/chess_gaze/test_gaze_observation.py
git commit -m "feat: add gaze observation"
```

Expected: commit succeeds.

### Task 11: Visualization Frames

**Files:**
- Create: `src/chess_gaze/visualization.py`
- Create: `tests/chess_gaze/test_visualization.py`

**Interfaces:**
- Consumes: `FrameRecord`, RGB frames, and `save_bgr_jpeg`
- Produces: `render_processed_frame(rgb_frame: np.ndarray, record: FrameRecord, output_path: Path, quality: int) -> str`

- [ ] **Step 1: Write failing visualization tests**

Create tests that assert:

- a processed JPEG is written for an OK frame;
- a processed JPEG is written for `FACE_NOT_FOUND`;
- alternate face candidate boxes are drawn when present;
- left and right iris centers are drawn independently;
- status and error code text are drawn for failure frames;
- visualization output path is never used as source evidence in the frame record.

- [ ] **Step 2: Run visualization tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_visualization.py -q
```

Expected: tests fail because `visualization.py` does not exist.

- [ ] **Step 3: Implement visualization overlays**

Implement OpenCV drawing with a clear RGB-to-BGR boundary. Draw:

- selected face bbox;
- alternate face bboxes;
- score and score provenance when available;
- major face landmarks;
- left and right eye contours;
- iris centers;
- per-eye gaze vectors;
- UniGaze face-level vector;
- head pose axes when valid;
- frame status and error summary.

- [ ] **Step 4: Run visualization tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_visualization.py -q
```

Expected: all visualization tests pass.

- [ ] **Step 5: Commit**

Run:

```sh
git add src/chess_gaze/visualization.py tests/chess_gaze/test_visualization.py
git commit -m "feat: add processed frame visualization"
```

Expected: commit succeeds.

### Task 12: Pipeline Orchestration With Fake Observers

**Files:**
- Create: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/cli.py`
- Create: `tests/chess_gaze/test_pipeline_contract.py`
- Modify: `tests/chess_gaze/test_cli.py`

**Interfaces:**
- Consumes: all prior modules
- Produces: `analyze_video(request: AnalyzeRequest, observers: ObserverBundle | None = None) -> AnalyzeResult`
- Produces: CLI analyze command wired to `analyze_video`

- [ ] **Step 1: Write failing pipeline contract tests**

Create tests with a tiny synthetic video and fake observers that assert:

- every decoded frame writes one raw PNG;
- every decoded frame writes one processed JPEG;
- every decoded frame writes one `frames.jsonl` line;
- no-face fake observer produces `FACE_NOT_FOUND` records and processed frames;
- one-eye-missing fake observer preserves the other eye;
- no smoothing or across-frame mutation occurs by comparing per-frame fake values;
- pre-run missing model failure creates no run directory;
- frame-time write failure records partial status and error evidence.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_cli.py -q
```

Expected: tests fail because pipeline orchestration is not wired.

- [ ] **Step 3: Implement orchestration**

Implement this order:

1. Resolve config.
2. Validate input video.
3. Inspect video.
4. Validate model assets and license approval.
5. Estimate disk space.
6. Create immutable run layout.
7. Write run/calibration/video manifests.
8. Decode frames one by one.
9. Save raw PNG.
10. Run face, eye, head, gaze observers for the same frame only.
11. Validate and append one frame record.
12. Append frame-time errors.
13. Render processed JPEG.
14. Finalize QA summary and run status.
15. Re-read and validate artifact records from disk.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_cli.py -q
```

Expected: all pipeline and CLI tests pass with fake observers.

- [ ] **Step 5: Commit**

Run:

```sh
git add src/chess_gaze/pipeline.py src/chess_gaze/cli.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_cli.py
git commit -m "feat: orchestrate frame analysis pipeline"
```

Expected: commit succeeds.

### Task 13: QA Summary and Artifact Revalidation

**Files:**
- Create: `src/chess_gaze/qa_summary.py`
- Create: `tests/chess_gaze/test_qa_summary.py`
- Modify: `src/chess_gaze/pipeline.py`

**Interfaces:**
- Consumes: manifests, JSONL records, artifact paths, and error records
- Produces: `build_qa_summary(run_layout: RunLayout) -> QASummary`
- Produces: `validate_run_artifacts(run_layout: RunLayout) -> ArtifactValidationResult`

- [ ] **Step 1: Write failing QA summary tests**

Create tests that assert:

- record count equals decoded frame count;
- raw frame count equals decoded frame count;
- processed frame count equals decoded frame count;
- byte counts include raw, processed, crops, JSONL, and total run size;
- face/eye/iris/head/gaze rates are computed;
- errors are counted by code and severity;
- worst blur and exposure frame IDs are sorted deterministically;
- 30 QA sample IDs are deterministic when at least 30 frames exist;
- disk-space estimate and closeout free-space measurement are present;
- malformed JSONL produces `SCHEMA_VALIDATION_FAILED`.

- [ ] **Step 2: Run QA tests and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_qa_summary.py -q
```

Expected: tests fail because `qa_summary.py` does not exist.

- [ ] **Step 3: Implement QA summary and disk revalidation**

Implement:

- manifest and JSONL re-read validation from disk;
- file count checks;
- status transition capture;
- byte count aggregation;
- deterministic sample selection;
- representative failure frame IDs;
- final status update to `complete` or `failed`.

- [ ] **Step 4: Run QA tests and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_qa_summary.py -q
```

Expected: all QA summary tests pass.

- [ ] **Step 5: Commit**

Run:

```sh
git add src/chess_gaze/qa_summary.py src/chess_gaze/pipeline.py tests/chess_gaze/test_qa_summary.py
git commit -m "feat: add qa summary validation"
```

Expected: commit succeeds.

### Task 14: Full Gates, Real-Asset Smoke Checks, and Closeout

**Files:**
- Create: `docs/superpowers/closeouts/2026-06-25-frame-gaze-analysis-pipeline.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: complete implementation from Tasks 1-13
- Produces: verified local gates, documented smoke status, and closeout

- [ ] **Step 1: Run full automated tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
```

Expected: all automated tests pass. Tests requiring ignored videos or real model binaries skip with explicit messages when assets are absent.

- [ ] **Step 2: Run lint**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check .
```

Expected: all checks pass.

- [ ] **Step 3: Run format check**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
```

Expected: all files are formatted.

- [ ] **Step 4: Run type check**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run mypy
```

Expected: no type errors.

- [ ] **Step 5: Verify OpenCV provider**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run python -c 'import importlib.metadata as m; providers=[d.metadata["Name"] for d in m.distributions() if d.metadata["Name"].lower().startswith("opencv-python")]; print(providers); raise SystemExit(0 if providers == ["opencv-python-headless"] else 1)'
```

Expected: prints `['opencv-python-headless']` and exits 0.

- [ ] **Step 6: Run real-video smoke when assets exist**

If `artifacts/input/test_1.mp4`, `artifacts/input/test_2.mp4`, `models/mediapipe/face_landmarker.task`, and `models/unigaze/unigaze_h14_joint.safetensors` exist with approved checksums in `src/chess_gaze/model_registry.json`, run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/test_1.mp4
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/test_2.mp4
```

Expected:

- each command exits 0;
- `test_1` decoded frame count is 3613 unless PyAV evidence proves a different count;
- `test_2` decoded frame count is 1973 unless PyAV evidence proves a different count;
- artifact counts match decoded frame count;
- `FACE_NOT_FOUND` over 1 percent is reported as a smoke warning with representative frames, not hidden.

If the ignored videos or model files are absent, record the exact missing paths and do not claim real-model smoke passed.

- [ ] **Step 7: Update README with usage and model policy**

Add concise usage:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4
```

Document:

- model binaries stay under ignored `models/`;
- optional setup-time `HF_TOKEN` can live in ignored `.env`;
- `src/chess_gaze/model_registry.json` is the trust root;
- UniGaze `unigaze_h14_joint` license/use approval was granted by the repo owner on 2026-06-25 and is recorded as metadata, not as a secret;
- analysis does not download models;
- analysis does not require or read `HF_TOKEN` for network access;
- real-model smoke requires local assets and license approval.

- [ ] **Step 8: Write closeout**

Create `docs/superpowers/closeouts/2026-06-25-frame-gaze-analysis-pipeline.md` with:

- request summary;
- spec and plan followed;
- task summary;
- dependency and model-license status;
- TDD evidence;
- full gate outputs;
- real-video smoke status;
- manual QA sample notes;
- remaining limitations.

- [ ] **Step 9: Commit final documentation**

Run:

```sh
git add README.md docs/superpowers/closeouts/2026-06-25-frame-gaze-analysis-pipeline.md
git commit -m "docs: close out frame gaze pipeline"
```

Expected: commit succeeds.

## Self-Review

Spec coverage: this plan covers CLI interface, pre-run validation, model registry authority, no downloads, immutable artifacts, atomic writes, PyAV decode, strict schemas, MediaPipe `IMAGE` mode, nullable scores, face selection, independent eyes, head pose, UniGaze pitch/yaw mapping, per-eye and recommended gaze separation, visualization, QA summary, real-video smoke, manual QA, and standard gates.

Forbidden-token scan: no known placeholder tokens appear in executable steps, and no hidden "add tests" step lacks concrete test behavior.

Type consistency: public interfaces are named once and reused consistently across dependent tasks: `AnalyzeRequest`, `AnalyzeResult`, `RunLayout`, `FrameRecord`, `CalibrationRecord`, `ResolvedModelAsset`, `DecodedFrame`, observer outputs, and QA summary validators.

Execution handoff: implement this plan task-by-task. Use subagent-driven development unless there is a specific reason to keep execution inline.
