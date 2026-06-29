# Default Crop Image Retention Closeout

Date: 2026-06-29

## Summary

`chess-gaze analyze` no longer creates or retains eye crop PNGs by default. The analyzer
still computes eye crop bounds, crop-to-image transforms, pupil and iris
evidence, frame records, scene records, viewer artifacts, model runtime
metadata, and QA summaries from in-memory image data and committed JSONL
records.

Crop image retention is now explicit:

- CLI: `--save-crops`
- Programmatic API: `AnalyzeRequest(save_crop_images=True)`
- Config: `"save_crop_images": true`

This policy is intentionally separate from `--save-frames`, which remains
scoped to `raw_frames/*.png` and `processed_frames/*.jpg`.

## Root Cause

Eye crop PNG writes were embedded in `eye_observation.observe_eyes()` as an
unconditional side effect. Raw and processed frame image retention already had a
policy, but crops were only counted by QA rather than governed by a persisted
retention contract.

That made `crops/**/*.png` default durable output even though downstream scene
and viewer artifacts consume `records/frames.jsonl`, and current frame records
do not persist crop paths or crop hashes.

## Durable Surface Changed

- `RunManifest` now persists `crop_image_retention` with schema version
  `crop-image-retention-v1`.
- Legacy run manifests without that field read as `save_crop_images=true`, so
  historical completed runs keep their old crop-retaining semantics.
- `AnalysisConfig`, `AnalyzeRequest`, and `--save-crops` resolve one
  `save_crop_images` boolean for the run.
- Resume compatibility requires matching crop-image retention policy.
- `ModelBackedFrameObserver` carries the resolved policy into
  `observe_eyes()`.
- `observe_eyes(save_crop_images=False)` computes crop geometry and transforms
  but returns `eye_crop_path=None` and `eye_crop_sha256=None` without calling
  the PNG writer.
- `observe_eyes(save_crop_images=True)` preserves the old crop path and hash
  behavior.
- `create_run_layout()` keeps stable in-memory crop paths but no longer creates
  the on-disk `crops/` directory tree up front.
- `qa_summary.py` rejects stray crop files when the manifest disables crop
  saving. When saving is enabled, QA reports crop counts and bytes without
  enforcing an exact count because current durable records do not encode every
  write-eligible eye crop.

## Artifact And Visual Evidence

The real test video is `artifacts/input/nakamura_short.mp4`, SHA-256
`6364e160934c7a8de4318095172edeaf457f008f07a57f4266b2882225b5cb88`, with
180 decoded frames at 1920x1080.

Pre-fix reproduction:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --no-resume --output-root /private/tmp/chess-gaze-crop-repro-before --progress off
```

Run `20260629T171227Z-5b4ddce6` completed with `360` crop PNGs, `0` raw frame
files, and `0` processed frame files.

Post-fix default verification after lazy crop-directory creation:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --no-resume --output-root /private/tmp/chess-gaze-crop-repro-after-lazy --progress off
```

Run `20260629T180027Z-c7af0830` completed with no on-disk `crops/`
directory and:

```json
{
  "crop_files": 0,
  "decoded_frames": 180,
  "frame_records": 180,
  "processed_frames": 0,
  "raw_frames": 0,
  "scene_frame_records": 180,
  "crops_bytes": 0,
  "final_status": "complete",
  "counts_match": true,
  "validation_errors": []
}
```

Post-fix explicit crop-saving verification after lazy crop-directory creation:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --no-resume --save-crops --output-root /private/tmp/chess-gaze-crop-repro-save-crops-lazy --progress off
```

Run `20260629T180125Z-2abb6d65` completed with `360` crop PNGs, `0` raw frame
files, `0` processed frame files, `403163` crop bytes, and no validation
errors. Its manifest persisted `save_frame_images=false` and
`save_crop_images=true`.

Visual inspection confirmed the source frame is the Hikaru/chess-board video
layout and the retained crop sample
`crops/eyes/left/f000000000.png` is a 43x17 eye-region debug PNG, not an
independent downstream analysis input.

## Library Research

No new third-party dependency was added. The relevant storage boundary is local
code calling the image writer.

- PyAV video frame APIs decode frames into in-memory video frames before the
  observer pipeline runs: https://pyav.org/docs/stable/api/video.html
- MediaPipe Face Landmarker Python `IMAGE` mode consumes image data and returns
  detection results without requiring persisted crop files:
  https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
- NumPy array slicing supports in-memory crop views/copies independent of disk
  persistence: https://numpy.org/doc/stable/user/basics.copies.html
- Pillow image persistence is explicit image save behavior, which remains gated
  behind the crop-retention flag: https://pillow.readthedocs.io/en/stable/reference/Image.html
- Pydantic v2 `default_factory` is the existing-compatible mechanism used for
  legacy manifest defaults: https://docs.pydantic.dev/latest/concepts/fields/

## Tests

Added or updated coverage for:

- config `save_crop_images`;
- CLI `--save-crops`;
- default and explicit manifest crop retention policy;
- resume compatibility rejecting mismatched crop-retention policy;
- `observe_eyes()` preserving crop geometry while omitting the `crops/` tree and crop files by
  default;
- `observe_eyes(save_crop_images=True)` preserving old crop path/hash behavior;
- model-backed observer propagation of the crop-retention policy;
- QA accepting zero crop files when saving is disabled;
- QA rejecting stray crop files when saving is disabled;
- model-free and model-backed real-video contracts expecting no default
  `crops/` tree or crop files.

## Verification

Focused local gate results:

```sh
uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_analysis_resume.py -q
# 47 passed in 1.85s
```

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_does_not_retain_raw_or_processed_frame_images_by_default tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_retains_raw_and_processed_frame_images_when_requested -q
# 2 passed in 0.89s
```

```sh
uv run pytest tests/chess_gaze/test_eye_observation.py -q
# 7 passed in 0.75s
```

```sh
uv run pytest tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py -q
# 45 passed in 1.26s
```

```sh
uv run pytest tests/chess_gaze/test_qa_summary.py -k 'crop_images or crop_files' -q
# RED before QA fix: stray crop was accepted when crop saving was disabled.
```

```sh
uv run pytest tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_qa_summary_real_video_contract.py -q
# 14 passed in 1.01s
```

```sh
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py -q
# 3 passed, 18 warnings in 195.88s
```

Final local gate results:

```sh
uv run pytest
# 403 passed, 18 warnings in 161.92s
```

Warnings were existing Torch JIT deprecation warnings from
`tests/chess_gaze/test_gaze_observation.py`.

```sh
uv run ruff check .
# All checks passed!

uv run ruff format --check .
# 69 files already formatted

uv run mypy
# Success: no issues found in 69 source files
```

## Source Layout

`pipeline.py` is 839 lines after crop-retention plumbing, crossing the
800-line source-layout review threshold. The source-layout document records the
review and rationale: the file still owns one cohesive analysis orchestration
boundary, and this change only threads artifact policy through that boundary.

## Residual Risk

The `RunLayout` object still exposes crop paths for callers and resume cleanup,
but the on-disk `crops/` tree is absent until explicit crop-saving writes create
it. QA does not assert an exact crop count when crop saving is enabled because
the current durable frame records do not encode every write-eligible eye crop.
Exact opt-in crop-count validation would require adding explicit per-eye
crop-write fields to the frame record contract.
