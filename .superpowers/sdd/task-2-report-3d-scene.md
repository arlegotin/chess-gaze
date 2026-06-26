# Task 2 Report: Camera Model And Eye Back-Projection

## What I Implemented

- Added `src/chess_gaze/scene_geometry.py`.
- Implemented `estimated_camera_model()` with the required
  `estimated_pinhole_from_image_size` policy and `fx = fy = max(width, height)`.
- Implemented `back_project_eye_points()` with:
  - OpenCV-style pseudo-metric camera coordinates
  - depth from the interpupillary-distance assumption
  - persisted left/right `SceneEyeRecord` values
  - midpoint `SceneEyeMidpointRecord`
  - flat diagnostics for pupil distance, source assumption, frame inclusion, and
    non-finite input
- Added conservative public dataclasses for `SceneEyePairProjection`,
  `RobustPointEstimate`, and `RobustDirectionEstimate`.
- Added explicit `NotImplementedError` placeholders for later-task public
  geometry functions instead of inventing behavior outside Task 2.

## RED Command And Failing Output Summary

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Summary:

- `8` tests failed as expected.
- Failure mode was `ModuleNotFoundError: No module named 'chess_gaze.scene_geometry'`
  from the new RED tests, which confirmed the required Task 2 module and
  behavior were not implemented yet.

## GREEN Command And Passing Output Summary

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Summary:

- `8 passed in 0.08s`

Additional verification:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_geometry.py tests/chess_gaze/test_scene_geometry.py
```

- `All checks passed!`

## Files Changed

- `src/chess_gaze/scene_geometry.py`
- `tests/chess_gaze/test_scene_geometry.py`
- `.superpowers/sdd/task-2-report-3d-scene.md`

## Self-Review Findings

- The tests lock the controller-required behavior that the projection result must
  expose real left/right `SceneEyeRecord` values, not just booleans and
  diagnostics.
- Non-finite pupil inputs cannot be persisted as `Point2D` inside
  `SceneEyeRecord` because the shared schema correctly rejects non-finite
  floats. The implementation preserves that condition in invalid reasons and
  diagnostics instead of silently coercing it.
- Later-task public functions are present but intentionally unimplemented, which
  keeps the Task 2 surface honest and avoids speculative geometry behavior.

## Concerns

- None.
