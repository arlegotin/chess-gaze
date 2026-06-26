# Task 5 Report: Scene Artifact Writer And Summary

## Scope

Implemented Task 5 only:

- `src/chess_gaze/scene_artifacts.py`
- `tests/chess_gaze/test_scene_artifacts.py`
- `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`
- `.superpowers/sdd/task-5-report-3d-scene.md`

No pipeline, `RunLayout`, QA summary, CLI, or viewer integration files were modified.

## RED Evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Result before implementation:

```text
ModuleNotFoundError: No module named 'chess_gaze.scene_artifacts'
2 errors during collection
```

This failed for the expected reason: the public Task 5 writer module and API did not exist.

## Design Notes

- `build_scene_artifacts()` reads only:
  - `run_manifest.json`
  - `video_manifest.json`
  - `records/frames.jsonl`
- Source JSON is validated through `RunManifest`, `VideoManifest`, and `FrameRecord`.
- Camera calibration is estimated once with `estimated_camera_model()`.
- The first pass collects only valid robust-estimator candidates:
  - midpoint camera points from `back_project_eye_points()`;
  - appearance-gaze rays from `unigaze_ray_from_frame()`;
  - eye-pair right vectors from valid left/right camera points.
- The second pass rebuilds strict `SceneFrameRecord`s in the final robust basis:
  - final eye and midpoint `scene_m`;
  - final UniGaze `origin_scene_m` and `direction_scene`;
  - monitor intersections, diagnostics, and invalid monitor-hit reasons.
- JSON and JSONL writes use strict models, aliases, atomic writes where applicable, and `allow_nan=False`.
- The manifest records the approved Three.js placeholder dependency metadata from the spec. Task 5 does not generate viewer files, so summary `artifact_validation.viewer_exists` is honestly `false`.

## Edge Cases Covered

- Every source frame produces exactly one scene frame record.
- Frame IDs and contiguous frame indices are preserved.
- Valid forward intersections produce one monitor hit per valid frame.
- Invalid appearance gaze produces explicit UniGaze and monitor-hit invalid reasons.
- Duplicate source eye/gaze frames produce duplicate persisted hits and duplicate viewer hit identities.
- Real-video model-free run preserves one valid viewer hit per decoded frame.

## GREEN Evidence

Synthetic command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts.py -q
```

Result:

```text
2 passed in 1.12s
```

Real-video command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Result:

```text
1 passed in 484.36s (0:08:04)
```

Focused Ruff command:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_artifacts.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py
```

Result:

```text
All checks passed!
```

## Real-Video Checkpoint

- Required input existed: `artifacts/input/nakamura_1.mp4`.
- `analyze_video()` used deterministic fake observers only.
- Decoded frame count asserted: `1973`.
- Scene frame count asserted: `1973`.
- Viewer data valid hit count asserted: `1973`.
- `scene_summary.json` validation and count checks passed.

## Residual Risks

- Task 5 intentionally leaves viewer file generation and `viewer_exists=true` to later viewer integration tasks.
- Task 5 intentionally leaves pipeline, QA summary, CLI, and `RunLayout.scene_dir`/`viewer_dir` integration to Task 6+.
- Head ellipsoid placement uses the existing default eye-midpoint offset without per-frame head-local rotation refinement.
