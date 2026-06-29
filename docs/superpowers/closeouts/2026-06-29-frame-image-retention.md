# Default Frame Image Retention Closeout

Date: 2026-06-29

## Summary

`chess-gaze analyze` no longer retains raw decoded PNGs or processed overlay
JPEGs by default. The analyzer still passes decoded RGB frames through the
analysis pipeline in memory, writes strict frame records, builds scene/viewer
artifacts, and writes QA summaries.

Frame image retention is now explicit:

- CLI: `--save-frames`
- Programmatic API: `AnalyzeRequest(save_frame_images=True)`
- Config: `"save_frame_images": true`

## Root Cause

Raw and processed frame image retention was an unconditional pipeline side
effect and a QA invariant. `_prepare_decoded_frame()` always wrote raw PNGs,
`_render_processed_frame_and_collect_errors()` always wrote processed JPEGs, and
QA validation required both image counts to match decoded-frame count.

That made large frame-image artifacts the default durable output even when later
analysis stages only needed in-memory frames and committed JSONL records.

## Durable Surface Changed

- `RunManifest` now persists `frame_image_retention` with schema version
  `frame-image-retention-v1`.
- Legacy run manifests without that field read as `save_frame_images=true`, so
  historical saved-frame runs keep their old validation semantics.
- `AnalysisConfig`, `AnalyzeRequest`, and `--save-frames` resolve one
  `save_frame_images` boolean for the run.
- `pipeline.py` gates raw and processed image writes on that boolean. Default
  runs do not call image writers.
- `qa_summary.py` validates raw and processed image counts against the persisted
  retention policy: `0` by default, decoded-frame count when saving is enabled.
- Resume compatibility requires matching frame-image retention policy.

## Tests

Added or updated coverage for:

- default fake-video runs retaining zero raw and processed frame image files;
- explicit `save_frame_images=True` retaining one raw and one processed image
  per decoded frame;
- persisted manifest policy for both default and explicit-save runs;
- CLI `--save-frames` forwarding;
- config `save_frame_images`;
- QA accepting zero image files when policy disables saving;
- QA rejecting stray image files when policy disables saving;
- raw and processed image write failure paths under explicit saving;
- resume compatibility rejecting mismatched frame-image retention policies;
- legacy modern run manifests without `frame_image_retention` reading as
  saved-frame runs;
- real-video model-free contracts expecting zero default image files.

## Verification

Fresh local gate results:

```sh
uv run pytest -q
# 379 passed, 18 warnings in 39.06s
```

Warnings were existing Torch JIT deprecation warnings from
`tests/chess_gaze/test_gaze_observation.py`.

```sh
uv run ruff check .
# All checks passed!

uv run ruff format --check .
# 67 files already formatted

uv run mypy
# Success: no issues found in 67 source files
```

## Source Layout

`pipeline.py` is 772 lines after the change, below the 800-line source-layout
review trigger. `qa_summary.py` is 710 lines. No new implementation module was
needed because the policy is small and belongs at existing run-manifest,
pipeline, and QA validation boundaries.

## Residual Risk

Default runs no longer have raw-frame blur and exposure rankings because those
QA samples require retained raw image files. That is intentional under the
disk-saving default. Run with `--save-frames` when visual frame-level debugging,
processed overlay inspection, blur/exposure ranking, or external frame-image QA
is required.
