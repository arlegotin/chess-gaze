# Empty Eye Crop Guard Closeout

Date: 2026-06-29

## Result

Off-frame eye landmarks no longer crash analysis with `cannot write empty image`
or invalid crop `BBox` errors. Eye crop bounds now represent the positive
intersection between the padded eye bbox and the decoded frame. If the
intersection is empty, that eye becomes an explicit missing-eye observation
instead of a fabricated crop or schema failure.

## Root Cause

The failed `nepo_2.mp4` run stopped with `next_frame_index=25921`, but the
failure was inside the uncommitted batch beginning there. Direct MediaPipe
replay of frames `25921..25927` showed:

- frames `25921..25923`: left eye still has a tiny positive in-frame crop;
- frames `25924..25927`: left-eye landmarks are fully beyond the right frame
  edge.

Before this fix, `_crop_bounds()` clamped `x_max/y_max` but allowed
`x_min/y_min` to remain outside the frame. That produced empty NumPy slices when
crop images were saved and invalid `BBox` values even when crop saving was
disabled.

## Changes

- `src/chess_gaze/eye_observation.py`
  - `_crop_bounds()` now returns `None` for empty image intersections.
  - `_eye_crop_record()` returns no crop record for empty intersections.
  - `_observe_eye()` converts that condition into `LEFT_EYE_NOT_FOUND` or
    `RIGHT_EYE_NOT_FOUND`, preserving landmarks, original eye bbox, normalized
    bbox, and open metric.
  - Partial intersections remain valid and clipped to the image.

- `src/chess_gaze/image_io.py`
  - PNG and JPEG writers now validate RGB shape, dtype, and positive dimensions
    before entering Pillow/OpenCV.

- Tests
  - Added synthetic off-right and off-bottom eye regressions in both crop modes.
  - Added partial-edge crop regression to protect valid clipped eyes.
  - Added explicit empty-image writer guard tests.
  - Added a native MediaPipe regression over `nepo_2.mp4` frames `25921..25927`
    when the optional large artifact is present.

## Verification

Fresh local evidence:

```sh
uv run pytest tests/chess_gaze/test_eye_observation.py tests/chess_gaze/test_image_io.py -q
# 21 passed

uv run pytest tests/chess_gaze/test_eye_observation_real_video.py::test_nepo_edge_window_marks_off_frame_eye_missing_without_empty_crop -q
# 1 passed

uv run pytest tests/chess_gaze/test_eye_observation.py tests/chess_gaze/test_eye_observation_real_video.py tests/chess_gaze/test_image_io.py -q
# 23 passed

uv run pytest tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py -q
# 45 passed

uv run ruff check .
# All checks passed

uv run ruff format --check .
# 69 files already formatted

uv run mypy
# Success: no issues found in 69 source files

uv run pytest -q
# 413 passed, 18 warnings
```

Real-video checks:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 \
  --no-resume --output-root /private/tmp/chess-gaze-empty-crop-nakamura-20260629 \
  --models-root models --progress off
# complete, 180 frame records, crop_files=0, schema_validation_passed=true
```

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 \
  --no-resume --save-crops \
  --output-root /private/tmp/chess-gaze-empty-crop-nakamura-save-crops-20260629 \
  --models-root models --progress off
# complete, 180 frame records, crop_files=360, schema_validation_passed=true
```

Focused `nepo_2.mp4` replay over frames `25921..25927` with
`save_crop_images=False` and `True` produced no exceptions. Frames `25924..25927`
recorded the left eye as `LEFT_EYE_NOT_FOUND`; frames `25921..25923` retained a
positive clipped left-eye crop bbox.

## Remaining Limitations

- The full `nepo_2.mp4` analysis was not rerun end-to-end because the original
  failure happened after roughly seventy minutes. The exact failing window is
  now covered by a native MediaPipe regression and by focused replay evidence.
- The native `nepo_2` regression is skipped if the optional large video artifact
  is absent from a test environment.
- Importing both PyAV and OpenCV on this macOS environment emits an existing
  duplicate AVFoundation class warning. It did not affect test or CLI outcomes.
