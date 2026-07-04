# Hit-Area-Only Viewer Data Design

Date: 2026-07-04

## Status

Approved by direct user correction on 2026-07-04: Hit Points must be removed
from viewer data as well as from visualization. This spec supersedes the
schema-preservation clause in
`docs/superpowers/specs/2026-07-04-visualization-redo-design.md`.

## Goal

Make Hit Area the only gaze-hit viewer artifact surface. The generated
`viewer/scene-data.json` must no longer contain the duplicated
`valid_hit_points` collection or the `ViewerHitPoint` schema branch.

## Required Behavior

- Remove top-level `valid_hit_points` from generated `viewer/scene-data.json`.
- Remove the `ViewerHitPoint` record type and any validation path dedicated to
  viewer hit-point data.
- Keep per-frame `frames[*].sphere_hit` records. Hit Area derives from these
  records and still needs point, radius, theta, phi, and hemisphere fields for
  valid sphere hits.
- Keep summary fields such as `valid_sphere_hit_frames`; they count valid
  sphere hits and are not Hit Points visualization data.
- Keep the viewer's visible `Hits` count derived from `frames[*].sphere_hit`.
- Update QA streaming validation so `viewer/scene-data.json` validates only the
  `frames` array plus the small envelope; it should still reject unexpected
  top-level keys.
- Update run equivalence fixtures and active docs to use the slimmer viewer
  data contract.

## Implementation Surface

- `src/chess_gaze/scene_records.py` owns the `ViewerSceneData` schema.
- `src/chess_gaze/scene_artifacts.py` owns `ViewerSceneData` construction.
- `src/chess_gaze/qa_summary.py` owns streaming validation for large
  `viewer/scene-data.json` files.
- `tests/chess_gaze/test_scene_records.py`,
  `tests/chess_gaze/test_scene_artifacts.py`,
  `tests/chess_gaze/test_scene_viewer.py`,
  `tests/chess_gaze/test_qa_summary.py`, and
  `tests/chess_gaze/test_run_equivalence.py` pin the data contract.
- `README.md`, ADR-0006, and the 2026-07-04 closeout/spec/plan must not claim
  that viewer hit-point data remains active.

No new dependency, model, checkpoint, or frontend build step is required.

## Testing

Use test-first development.

Focused RED tests should require:

- Generated scene data lacks top-level `valid_hit_points`.
- `ViewerSceneData` accepts payloads without `valid_hit_points` and rejects
  payloads that still include it as an unexpected key.
- `build_viewer_scene_data()` preserves frames and valid sphere-hit summaries
  without emitting duplicated hit points.
- QA summary rejects unexpected `valid_hit_points` top-level data.
- Run-equivalence fixtures validate with the slimmer viewer payload.

Required gates:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_run_equivalence.py -q
UV_CACHE_DIR=.uv-cache uv run pytest -q -m "not native_mediapipe and not local_socket" --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

## Acceptance Criteria

1. New generated viewer scene data has no `valid_hit_points` key.
2. The Python viewer schema has no `valid_hit_points` field.
3. The Python codebase has no active `ViewerHitPoint` type or viewer
   hit-point validator path.
4. Hit Area still renders from `frames[*].sphere_hit`.
5. QA summary still validates viewer scene data without materializing the whole
   file.
6. Active docs describe Hit Area and sphere hits, not viewer hit-point data.
7. Focused and broad non-native verification pass, or blockers are recorded
   with exact output.

