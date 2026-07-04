# Hit-Area-Only Viewer Data Closeout

Date: 2026-07-04

## Summary

Hit Points were removed from the viewer data contract, not only from the
browser visualization. Generated `viewer/scene-data.json` no longer contains
top-level `valid_hit_points`, `ViewerSceneData` no longer has a
`valid_hit_points` field, and the dedicated `ViewerHitPoint` schema branch was
deleted. The breaking viewer data contract change bumps the schema version to
`gaze-scene-viewer-data-v3`. Hit Area remains backed by per-frame
`frames[*].sphere_hit` records.

## Durable Surface Changed

- `src/chess_gaze/scene_records.py`: removed `ViewerHitPoint` and
  `ViewerSceneData.valid_hit_points`.
- `src/chess_gaze/scene_artifacts.py`: stopped deriving duplicated viewer
  hit-point records from valid sphere hits.
- `src/chess_gaze/qa_summary.py`: kept streaming validation for
  `viewer/scene-data.json`, but now validates only `frames` plus the small
  envelope, counts valid `frames[*].sphere_hit` records, cross-checks that
  count against `SceneSummary.valid_sphere_hit_frames`, and rejects
  `valid_hit_points` as an unexpected top-level key.
- Tests now pin the slim viewer data contract across scene records, scene
  artifacts, generated viewer files, QA summary, run equivalence, and the
  real-video contract test source.
- README and ADR-0006 now describe per-frame `sphere_hit` as the persisted
  source for Hit Area instead of a viewer hit-point collection.

## Root Cause

The previous implementation duplicated every valid `frames[*].sphere_hit` into
top-level viewer hit-point data. After the visualization layer stopped drawing
Hit Points, that duplicate data branch no longer had an active consumer. The
durable fix was to remove the duplicate schema/producer/validator path and keep
`sphere_hit` as the single source for Hit Area.

## Test-First Evidence

- Focused RED command:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_run_equivalence.py -q`
  -> `15 failed, 84 passed in 1.90s`.
- RED failures were the expected old contract failures: `ViewerSceneData`
  required missing `valid_hit_points`, generated `scene-data.json` still
  emitted it, and QA still treated `valid_hit_points` as a known array.
- Final focused command:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_run_equivalence.py tests/chess_gaze/test_pipeline_contract.py -q`
  -> `141 passed in 2.49s`.
- Review-fix focused command:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_records.py::test_viewer_scene_data_serializes_schema_version_without_hit_points tests/chess_gaze/test_scene_viewer.py::test_scene_data_is_strict_schema_versioned_viewer_scene_data tests/chess_gaze/test_run_equivalence.py::test_compare_runs_rejects_non_v3_viewer_data tests/chess_gaze/test_qa_summary.py::test_qa_summary_rejects_viewer_sphere_hit_count_mismatch -q`
  -> `4 passed in 0.74s`.

## Verification

- Broad non-native, non-socket suite:
  `UV_CACHE_DIR=.uv-cache uv run pytest -q -m "not native_mediapipe and not local_socket" --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py`
  -> `419 passed, 12 deselected, 18 warnings in 5.12s`.
- Lint:
  `UV_CACHE_DIR=.uv-cache uv run ruff check .` -> `All checks passed!`.
- Format:
  `UV_CACHE_DIR=.uv-cache uv run ruff format --check .` -> `71 files already formatted`.
- Type check:
  `UV_CACHE_DIR=.uv-cache uv run mypy` -> `Success: no issues found in 71 source files`.
- JavaScript syntax:
  `node --check src/chess_gaze/viewer_assets/scene_viewer.js` -> passed.
- Stale-reference scan over active source and canonical docs found no active
  production references to `valid_hit_points`, `ViewerHitPoint`, or viewer
  hit-point data. Remaining matches are the new spec/plan, supersession notes,
  or negative contract assertions.

## Review

A subagent review found no critical defects and two important issues. Both were
addressed before final commit:

- QA now counts valid `frames[*].sphere_hit` records while streaming viewer
  data and rejects mismatches against `SceneSummary.valid_sphere_hit_frames`.
- The breaking viewer data contract change now uses
  `gaze-scene-viewer-data-v3` instead of redefining v2.

## Residual Risk

The full native/real-video gate was not rerun for this follow-up because the
previous visualization closeout already captured an environment-level native
MediaPipe abort in `tests/chess_gaze/test_eye_observation_real_video.py`.
The real-video contract test source was updated to the new viewer data contract,
but native real-video execution remains outside the verified subset here.

Browser canvas pixel verification was also not repeated. This follow-up changes
Python-generated JSON and streaming validation, while prior browser smoke was
blocked by headless Chromium failing to create a WebGL context. Generated viewer
tests verify both `scene-data.json` and embedded file-url HTML omit
`valid_hit_points`.
