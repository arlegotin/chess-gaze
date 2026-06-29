# Task 1 Report: Calibration And Sphere Projection Math

## What I implemented

- Replaced the active monitor-projection calibration constant and `SceneAssumptions` field set in [src/chess_gaze/scene_calibration.py](/Volumes/git/legotin/chess-gaze/src/chess_gaze/scene_calibration.py) with `DEFAULT_GAZE_SPHERE_RADIUS_M` and `gaze_sphere_radius_m`.
- Removed the old monitor-projection assumption records from the default calibration metadata and persisted `DEFAULT_GAZE_SPHERE_RADIUS_M` instead.
- Added the sphere-specific invalid reasons to `SceneInvalidReason` in [src/chess_gaze/scene_records.py](/Volumes/git/legotin/chess-gaze/src/chess_gaze/scene_records.py).
- Added [src/chess_gaze/sphere_projection.py](/Volumes/git/legotin/chess-gaze/src/chess_gaze/sphere_projection.py) with:
  - `GazeSphereSurface`
  - `SphereHitResult`
  - `build_gaze_sphere(...)`
  - `intersect_ray_with_sphere(...)`
- Replaced the calibration assertions in [tests/chess_gaze/test_scene_calibration.py](/Volumes/git/legotin/chess-gaze/tests/chess_gaze/test_scene_calibration.py) per the brief.
- Added [tests/chess_gaze/test_sphere_projection.py](/Volumes/git/legotin/chess-gaze/tests/chess_gaze/test_sphere_projection.py) with the exact required sphere hit and invalid-path coverage.

## Tests run and output summary

- Focused task test command:
  - `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py -q`
- Final result:
  - `11 passed in 0.10s`
- Diff hygiene:
  - `git diff --check -- src/chess_gaze/scene_calibration.py src/chess_gaze/scene_records.py src/chess_gaze/sphere_projection.py tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py`
  - Result: no output

## TDD evidence

### RED command

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py -q
```

Relevant failing output:

```text
ImportError: cannot import name 'DEFAULT_GAZE_SPHERE_RADIUS_M' from 'chess_gaze.scene_calibration'
ERROR tests/chess_gaze/test_scene_calibration.py
ERROR tests/chess_gaze/test_sphere_projection.py
2 errors in 0.14s
```

### GREEN command

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py -q
```

Relevant passing output:

```text
...........                                                              [100%]
11 passed in 0.10s
```

## Files changed

- [src/chess_gaze/scene_calibration.py](/Volumes/git/legotin/chess-gaze/src/chess_gaze/scene_calibration.py)
- [src/chess_gaze/scene_records.py](/Volumes/git/legotin/chess-gaze/src/chess_gaze/scene_records.py)
- [src/chess_gaze/sphere_projection.py](/Volumes/git/legotin/chess-gaze/src/chess_gaze/sphere_projection.py)
- [tests/chess_gaze/test_scene_calibration.py](/Volumes/git/legotin/chess-gaze/tests/chess_gaze/test_scene_calibration.py)
- [tests/chess_gaze/test_sphere_projection.py](/Volumes/git/legotin/chess-gaze/tests/chess_gaze/test_sphere_projection.py)

## Self-review findings

- The sphere intersection logic covers the required cases from the brief: origin-inside forward hit, rear hit, outside-origin nearest root, tangent hit, discriminant miss, behind-origin invalidation, invalid radius, and missing-ray invalidation.
- The new models remain strict/frozen Pydantic artifacts.
- Calibration metadata now persists the sphere-radius record and excludes the removed monitor-plane records named in the brief.

## Concerns if any

- This task intentionally removes active monitor-plane calibration fields from `SceneAssumptions`. Other repo modules and tests still reference those removed fields/constants, so broader repo migration work remains for later tasks. I did not run the full test suite because the brief scoped verification to the two Task 1 test files.
