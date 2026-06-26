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

## Review Fixes

- Review issue fixed: `_geometric_median_camera_point()` no longer returns immediately when the iterate coincides with one or more samples.
- Added regression coverage for the asymmetric all-inlier set where the component-wise median is a duplicated sample point but not the geometric median.
- Updated the helper to use modified Weiszfeld coincidence handling:
  - skip zero-distance terms for the weighted update;
  - compute the non-zero Weiszfeld update;
  - shrink the step using the coincidence multiplicity when the iterate sits on one or more samples;
  - return the current iterate only when the coincidence optimality condition holds or the update is converged.

RED evidence:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

- Result: failed as expected.
- Output: `1 failed, 23 passed in 1.02s`.
- Failing test: `test_robust_scene_center_does_not_stop_at_coincident_sample_point`
- Failure mode: `robust_scene_center()` returned exactly `(0.0, 0.0, 1.0)`.

GREEN evidence:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

- Result: passed.
- Output: `24 passed in 0.93s`.

Focused Ruff evidence:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_geometry.py tests/chess_gaze/test_scene_geometry.py
```

- Result: passed.
- Output: `All checks passed!`

## Review Fixes Round 2

- Strengthened `test_unigaze_ray_from_frame_uses_appearance_gaze_not_recommended_gaze` so it now asserts:
  - `direction_camera` equals `pitch_yaw_to_unit_vector()` for `appearance_gaze`;
  - `direction_camera` does not match the intentionally different `recommended_gaze` vector.
- Kept the existing sign-convention test unchanged.

Mutation-style RED evidence:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q -k uses_appearance_gaze_not_recommended_gaze
```

- Current correct implementation passed immediately after adding the stronger test, so a temporary mutation was used as allowed.
- Temporary mutation: changed `unigaze_ray_from_frame()` to read `frame_record.recommended_gaze`.
- Result under mutation: failed as expected.
- Output: `1 failed, 23 deselected in 0.65s`.
- Failure mode: the strengthened mixed-gaze test detected `recommended_gaze` values in the returned ray.

GREEN evidence:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

- Result: passed.
- Output: `24 passed in 0.65s`.

Focused Ruff evidence:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_geometry.py tests/chess_gaze/test_scene_geometry.py
```

- Result: passed.
- Output: `All checks passed!`
