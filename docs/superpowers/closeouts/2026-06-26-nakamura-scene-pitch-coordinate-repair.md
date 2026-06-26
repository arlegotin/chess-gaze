# Nakamura Scene Pitch Coordinate Repair Closeout

> **Superseded 2026-06-26:** This closeout only repaired pitch/up-down
> semantics. Its scene-axis wording predates the anatomical front/back and
> left/right repair. Current coordinate guidance is in
> `docs/superpowers/plans/2026-06-26-anatomical-scene-coordinate-repair.md` and
> `docs/superpowers/closeouts/2026-06-26-anatomical-scene-coordinate-repair.md`.

Date: 2026-06-26

## Summary

Fixed a systematic sign-convention bug in scene gaze-ray construction. A frame
gaze pitch that visually means image-up was being used directly as OpenCV
camera-space +Y, where +Y means image-down. This made upward-looking source
frames, including `f000001651`, render as downward 3D scene rays.

The durable boundary is now explicit:

- frame-record gaze angles keep the existing overlay convention: positive pitch
  is image-up;
- `camera_opencv_pseudo_m` keeps OpenCV convention: +X image-right, +Y
  image-down, +Z camera-forward;
- scene ray conversion preserves `pitch_yaw_to_unit_vector()` X/Z semantics and
  negates the Y component when entering `camera_opencv_pseudo_m`;
- scene space remains `right_up_back_columns_right_handed`, so semantic forward
  gaze toward the monitor is negative scene Z.

Implementation commits:

- `24ec76f docs: plan nakamura scene pitch repair`
- `9169d64 fix: map scene gaze pitch into opencv camera space`
- `90d4e0b docs: clarify scene gaze coordinate signs`
- `528ec5e test: lock scene upward gaze direction`

## Root Cause

`src/chess_gaze/scene_geometry.py` used `pitch_yaw_to_unit_vector()` directly
when building `SceneUniGazeRayRecord.direction_camera`.

That helper is correct for frame/overlay semantics: positive pitch produces a
positive image-up component. It was wrong to persist that Y component directly
as OpenCV camera-space Y, because OpenCV camera +Y points down in the image.

Observed faulty artifact evidence from the reported run
`artifacts/output/nakamura_1/runs/20260626T042921Z-d0f9cfa2`:

- `processed_frames/f000001651.jpg` visibly shows the streamer looking
  upward/up-left.
- `frames.jsonl` frame `1651` had `appearance_gaze.pitch_radians =
  0.8304274082183838`.
- old `scene_frames.jsonl` frame `1651` had
  `unigaze_ray.direction_camera.y = +0.7382197511455898` and
  `unigaze_ray.direction_scene.y = -0.7861916476000175`, which inverted the
  visual upward direction.

## Code And Docs Changed

Production fix:

- Added `_frame_gaze_angles_to_camera_direction()` in
  `src/chess_gaze/scene_geometry.py`.
- `unigaze_ray_from_frame()` now converts frame pitch/yaw to camera direction
  with `(x, -image_up_y, z)`.

Regression tests:

- `tests/chess_gaze/test_scene_geometry.py::test_unigaze_ray_from_frame_maps_positive_pitch_to_camera_up`
- `tests/chess_gaze/test_scene_geometry.py::test_unigaze_ray_from_frame_preserves_nakamura_upward_gaze_direction`
- `tests/chess_gaze/test_scene_artifacts.py::test_scene_frame_direction_maps_positive_pitch_to_scene_up`

Documentation repair:

- `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`
  now defines `scene_pseudo_m` as +X scene-right, +Y scene-up, +Z scene-back
  under `right_up_back_columns_right_handed`.
- `docs/superpowers/plans/2026-06-26-3d-scene-artifact-viewer.md` marks the
  old direct `pitch_yaw_to_unit_vector()` scene-ray instruction as superseded.

## Real Nakamura Verification

Sandboxed real analysis failed with native MediaPipe service initialization:

```text
Check failed: service_ Service is unavailable.
```

The same analysis completed unsandboxed:

```sh
MPLCONFIGDIR=/Volumes/git/legotin/chess-gaze/.cache/matplotlib \
UV_CACHE_DIR=.uv-cache \
uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 \
  --output-root artifacts/output \
  --models-root models
```

Fresh run:

```text
artifacts/output/nakamura_1/runs/20260626T104848Z-21353a29
```

Fresh run validation:

- source video sha256:
  `eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45`
- decoded frames: `1973`
- `records/frames.jsonl`: `1973`
- `records/scene_frames.jsonl`: `1973`
- processed frames: `1973`
- valid eye midpoint frames: `1973`
- valid UniGaze ray frames: `1973`
- valid monitor hit frames: `1958`
- invalid monitor hit reasons: `RAY_INTERSECTION_BEHIND_ORIGIN: 15`
- `qa_summary.final_status`: `complete`
- `qa_summary.artifact_validation.counts_match`: `true`
- `qa_summary.artifact_validation.schema_validation_passed`: `true`

Whole-run numeric audit on the fresh run:

```json
{
  "valid_rays_checked": 1973,
  "positive_pitch_frames": 181,
  "negative_pitch_frames": 1792,
  "formula_mismatch_count_sampled": 0,
  "pitch_camera_y_sign_mismatch_count_sampled": 0,
  "max_formula_delta": 2.220446049250313e-16,
  "axis_convention": "right_up_back_columns_right_handed",
  "axis_determinant_right_up_back": 1.0000000000000002,
  "monitor_center_scene_z": -0.7
}
```

Fresh frame `f000001651`:

```json
{
  "pitch": 0.8304274082183838,
  "yaw": -0.42343080043792725,
  "direction_camera": {
    "x": -0.277170535923229,
    "y": -0.7382197511455898,
    "z": 0.614986254346041
  },
  "direction_scene": {
    "x": -0.5926519291171297,
    "y": 0.7861916476000175,
    "z": -0.17511820053242522
  },
  "monitor_hit_valid": true,
  "hit_point_scene_y": 3.031831810018465
}
```

The viewer-facing payload at
`viewer/scene-data.json` has the same frame `1651` values:

- `direction_camera.y = -0.7382197511455898`
- `direction_scene.y = 0.7861916476000175`

Visual inspection:

- the reported old processed frame and the fresh processed frame both show the
  streamer looking upward/up-left;
- the fresh scene records and viewer payload now map that visual direction to
  camera-up and scene-up.

## Viewer Check

The local viewer server started successfully:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view \
  artifacts/output/nakamura_1/runs/20260626T104848Z-21353a29 \
  --host 127.0.0.1 \
  --port 0
```

Browser-helper URL:

```text
http://127.0.0.1:51193/
```

The gstack browser helper loaded the page and DOM payload, but its Chromium
instance could not create a WebGL context:

```text
THREE.WebGLRenderer: Error creating WebGL context.
```

Because of that headless-browser WebGL limitation, this closeout does not claim
a fresh rendered canvas screenshot. The served viewer data path was still
verified through the browser DOM by parsing `#scene-data-json`, which returned
the corrected frame `1651` values above.

## Verification

Red tests before production change:

```text
test_unigaze_ray_from_frame_maps_positive_pitch_to_camera_up
test_unigaze_ray_from_frame_preserves_nakamura_upward_gaze_direction
```

Both failed because `direction_camera.y` was positive for positive pitch.

Focused post-fix gates:

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
39 passed
```

```text
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_scene_artifacts.py \
  tests/chess_gaze/test_scene_records.py \
  tests/chess_gaze/test_scene_viewer.py -q
51 passed
```

```text
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_gaze_observation.py \
  tests/chess_gaze/test_head_pose.py \
  tests/chess_gaze/test_frame_observation.py -q
23 passed
```

Final focused scene/viewer gate:

```text
91 passed in 1.35s
```

Adjacent gaze/head/frame/pipeline gate:

```text
43 passed, 18 warnings in 2.45s
```

Isolated Nakamura real-video scene contract:

```text
1 passed in 559.62s (0:09:19)
```

Broad non-real-video suite, excluding absent real-video fixture files:

```text
233 passed, 18 warnings in 3.68s
```

Static gates:

```text
UV_CACHE_DIR=.uv-cache uv run ruff check .
All checks passed!
```

```text
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
56 files already formatted
```

```text
UV_CACHE_DIR=.uv-cache uv run mypy
Success: no issues found in 56 source files
```

Documented full pytest gate:

```text
7 failed, 235 passed, 7 skipped, 18 warnings in 644.26s (0:10:44)
```

All 7 failures are absent mandatory real-data fixtures in this checkout:

- `artifacts/input/test_1.mp4`
- `artifacts/input/test_2.mp4`

The skipped tests also cite missing `test_0.mp4`, `test_1.mp4`, or
`test_2.mp4`. These failures are not introduced by the coordinate fix, but they
remain a full-gate limitation for this checkout.

## Residual Risk

- Gaze quality remains bounded by the current UniGaze/head/face estimators and
  the pseudo-metric assumptions; this repair fixes sign interpretation, not
  estimator accuracy.
- The head ellipsoid is still rendered as an unrotated ellipsoid in the viewer;
  this is pre-existing viewer scope and not the ray-sign defect.
- Browser-rendered Three.js screenshot verification was blocked by the
  headless helper's WebGL context failure. The JSON payload and geometric data
  path were verified directly and through the browser DOM.
- Full pytest cannot pass in this checkout until the legacy mandatory fixtures
  `test_0.mp4`, `test_1.mp4`, and `test_2.mp4` are restored or those tests are
  intentionally re-baselined.
