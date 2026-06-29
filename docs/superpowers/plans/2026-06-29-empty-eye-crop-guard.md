# Empty Eye Crop Guard Plan

Date: 2026-06-29

## Goal

Prevent off-frame eye landmarks from causing empty crop images or invalid crop
schemas while preserving valid partial-edge eye observations.

## Step 1: Regression Tests

Add tests in `tests/chess_gaze/test_eye_observation.py`:

- fully off-right left eye, `save_crop_images=True`: observation returns without
  exception, left eye is missing with `LEFT_EYE_NOT_FOUND`, no left crop path or
  file is recorded, right eye remains valid;
- fully off-right left eye, `save_crop_images=False`: same missing-eye result,
  proving analysis quality does not depend on artifact retention;
- partially off-right left eye: eye remains present, crop bbox has
  `x_max == IMAGE_WIDTH_PX` and `x_min < x_max`.

Add tests in `tests/chess_gaze/test_image_io.py`:

- `save_rgb_png()` rejects `(0, w, 3)` and `(h, 0, 3)` arrays with a local error;
- `save_bgr_jpeg()` rejects the same empty dimensions before OpenCV encoding.

Run the new focused tests and confirm they fail before code changes.

## Step 2: Eye Observation Fix

Modify `src/chess_gaze/eye_observation.py`:

- change `_crop_bounds()` to return `tuple[int, int, int, int] | None`;
- compute padded bounds, intersect with `[0, image_width] x [0, image_height]`;
- return `None` when the intersection is empty;
- have `_eye_crop_record()` return `_CropRecord | None`;
- in `_observe_eye()`, before iris processing, convert `None` crop bounds to
  `_missing_eye(...)` with the eye-specific missing code, preserved landmarks,
  bbox, normalized bbox, open metric, and `occlusion="severe"`.

Do not create placeholder crops or fake 1-pixel boxes outside the image.

## Step 3: Image IO Defense

Modify `src/chess_gaze/image_io.py`:

- add a small RGB image validator for shape `(height, width, 3)`, dtype
  `uint8`, and positive height/width;
- use it in both `save_rgb_png()` and `save_bgr_jpeg()`.

Keep the writer error message specific enough to identify bad artifact input.

## Step 4: Verification

Run:

```sh
uv run pytest tests/chess_gaze/test_eye_observation.py tests/chess_gaze/test_image_io.py -q
uv run pytest tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Run real-video checks:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --no-resume --output-root /private/tmp/chess-gaze-empty-crop-nakamura --models-root models --progress off
find /private/tmp/chess-gaze-empty-crop-nakamura/nakamura_short/runs -path '*/crops/*' -type f | wc -l
```

Also run a focused script over `nepo_2.mp4` frames `25921..25927` with
`save_crop_images=True` and `False` to verify the original failing window no
longer raises.

## Step 5: Closeout and Commit

Write `docs/superpowers/closeouts/2026-06-29-empty-eye-crop-guard.md` with root
cause, changed boundary, verification evidence, and remaining limitations.

Commit the code, tests, and docs with a meaningful message.
