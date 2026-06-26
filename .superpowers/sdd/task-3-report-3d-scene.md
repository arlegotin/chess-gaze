# Task 3 Report: Robust Scene Center, Main Direction, And Axis Basis

## What I implemented

- Added RED coverage in `tests/chess_gaze/test_scene_geometry.py` for:
  - robust scene-center MAD screening, fallback, zero-MAD tolerance, and non-finite candidate dropping;
  - UniGaze ray construction from `appearance_gaze` only, including sign-locking against `pitch_yaw_to_unit_vector()` and midpoint invalidation;
  - dominant-direction angular RANSAC with outliers, tie-breaks, fallback, and opposite-ray rejection;
  - right-handed axis-basis construction with orthogonality, determinant `+1`, and degenerate-axis fallback diagnostics.
- Implemented `robust_scene_center()` in `src/chess_gaze/scene_geometry.py` with:
  - finite-candidate filtering;
  - component median and MAD screening using `max(3.5 * MAD, SCENE_CENTER_MIN_AXIS_TOLERANCE_M)`;
  - Weiszfeld geometric median on surviving inliers;
  - fallback to `DEFAULT_SCENE_CENTER_CAMERA_M` when inliers are insufficient.
- Implemented `unigaze_ray_from_frame()` with:
  - `appearance_gaze` as the sole source;
  - direction conversion through existing `pitch_yaw_to_unit_vector()`;
  - midpoint-required origin handling;
  - `EYE_MIDPOINT_INVALID` and `UNIGAZE_INVALID` invalidation paths.
- Implemented `robust_main_direction()` with:
  - valid finite direction filtering and normalization;
  - deterministic angular RANSAC seeded by fixed frame-order quantiles plus a coordinate-wise median seed;
  - tie-breaks by inlier count, then lower median angular residual, then lower seed frame index;
  - fallback to `[0.0, 0.0, 1.0]` when valid inliers are insufficient.
- Implemented `build_scene_axis_basis()` with:
  - semantic `forward_camera` from dominant UniGaze direction;
  - right-handed `[right, up, back]` basis with `back = -forward`;
  - preferred eye-pair right evidence, camera-up preference, and explicit fallback diagnostics for degenerate projections.
- Extended `RobustPointEstimate` conservatively to persist:
  - `dropped_non_finite_count`;
  - `convergence_tolerance_m`.

## RED command and failing output summary

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Summary:

- Result: failed as expected.
- Output: `14 failed, 9 passed in 1.46s`.
- Failure mode: all new Task 3 tests hit the Task 2 placeholders with `NotImplementedError` in:
  - `robust_scene_center()`
  - `unigaze_ray_from_frame()`
  - `robust_main_direction()`
  - `build_scene_axis_basis()`

## GREEN command and passing output summary

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Summary:

- Result: passed.
- Output: `23 passed in 0.71s`.

## Files changed

- `src/chess_gaze/scene_geometry.py`
- `tests/chess_gaze/test_scene_geometry.py`
- `.superpowers/sdd/task-3-report-3d-scene.md`

## Self-review findings

- Kept Task 2 eye back-projection behavior unchanged and confined new logic to the Task 3 placeholders plus private math helpers.
- Reused `pitch_yaw_to_unit_vector()` exactly as directed, so yaw/pitch sign handling stays anchored to the existing overlay convention.
- Kept monitor-plane and ray-plane work out of this task.
- Used explicit fallback diagnostics in axis construction because the public axis record has no richer structured fallback schema yet.
- The deterministic RANSAC seed policy is implemented as fixed quantiles `(0.0, 0.25, 0.5, 0.75, 1.0)` plus a coordinate-wise median seed. That matches the task brief and tests, but the specific quantile list is still an implementation choice rather than a separately documented constant.

## Concerns

- No blocking concerns from the scoped Task 3 work.
