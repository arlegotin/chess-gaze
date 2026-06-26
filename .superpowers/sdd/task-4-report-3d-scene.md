# Task 4 Report: Monitor Plane And Ray-Plane Intersection

## Scope

Changed files:

- `src/chess_gaze/scene_geometry.py`
- `tests/chess_gaze/test_scene_geometry.py`
- `.superpowers/sdd/task-4-report-3d-scene.md`

## RED Evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Result before implementation:

```text
11 failed, 25 passed in 1.16s
```

Expected failure mode: the new monitor-plane, scene-transform, and ray-plane
intersection tests failed on the Task 4 `NotImplementedError` placeholders.

## GREEN Evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Result after implementation:

```text
36 passed in 0.96s
```

Focused lint:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_geometry.py tests/chess_gaze/test_scene_geometry.py
```

Result:

```text
All checks passed!
```

## Design Notes

- `build_monitor_plane` uses `assumptions.monitor_distance_from_eyes_m`, width,
  height, and extended-plane scale. No measured monitor distance is inferred.
- Monitor center is `scene_center_camera + dominant_direction * distance`.
- Monitor normal is the opposite of the dominant UniGaze direction.
- Monitor right/up axes come from `SceneAxisBasisRecord`.
- `camera_point_to_scene` projects camera-relative points onto the right/up/back
  basis centered at the scene center.
- `intersect_ray_with_monitor` uses the signed origin distance convention
  `dot(origin - plane_center, normal)` and computes `t = -distance / denom`.
- Hit `u/v` and camera/scene hit points are never clamped to physical or
  extended bounds.
- Invalid hit records persist denominator, signed distance, and `t` when those
  values are finite; non-finite values are omitted to satisfy strict schemas.

## Edge Cases Covered

- Valid center hit with persisted denominator, signed distance, `t`, camera
  point, scene point, monitor `u/v`, and bounds flags.
- Parallel non-coplanar ray: `RAY_PARALLEL_TO_MONITOR`.
- Coplanar ray: `RAY_COPLANAR_WITH_MONITOR`.
- Behind-origin intersection: `RAY_INTERSECTION_BEHIND_ORIGIN`.
- Non-finite `t`: `RAY_INTERSECTION_NON_FINITE` without persisting Inf.
- Physical out-of-bounds but extended in-bounds valid hit.
- Extended out-of-bounds valid hit with `within_extended_plane == False`.
- Parallel denominator boundary: `abs(denom) < 1e-6` invalid,
  `abs(denom) == 1e-6` computable.

## Residual Risk

The intersection API reconstructs the scene transform from the persisted monitor
record because it does not receive the original scene center or axis basis.
This is consistent with the current monitor schema where `normal_camera` is the
scene back axis, but future schema changes should preserve that invariant or pass
the axis basis explicitly.
