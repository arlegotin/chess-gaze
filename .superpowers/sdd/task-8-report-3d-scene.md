# Task 8 Report: Viewer Data Generation And Static Viewer

Date: 2026-06-26

## RED

- Command: `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q`
- Result: failed during collection with `ModuleNotFoundError: No module named 'chess_gaze.scene_viewer'`.
- Interpretation: expected RED for Task 8 because the public viewer generator module did not exist yet.

## GREEN

- Command: `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q`
- Result: `12 passed in 1.32s`.
- Command: `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py -q`
- Result: `32 passed in 2.03s`.

## Real-Video Checkpoint

- Command: `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q`
- Result: `1 passed in 427.55s (0:07:07)`.
- Coverage: confirms the model-free real-video scene artifact contract, generated `viewer/index.html`, generated `viewer/scene-data.json`, and 1973-frame integrated viewer data.

## Ruff

- Command: `UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_viewer.py src/chess_gaze/pipeline.py tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py`
- Result: `All checks passed!`

## Design Notes

- Added `src/chess_gaze/scene_viewer.py` as the Task 8 generation boundary. It copies packaged `viewer_assets` recursively with `importlib.resources.files("chess_gaze").joinpath("viewer_assets")` and writes strict `viewer/scene-data.json` using `ViewerSceneData.model_dump(mode="json", by_alias=True)` plus `allow_nan=False`.
- Replaced the pipeline placeholder viewer writer. `analyze_video()` now builds scene artifacts, marks the scene result as having a viewer, calls `build_scene_viewer()`, rewrites the scene summary with `viewer_exists=True`, then builds QA.
- Implemented the static viewer with local ESM imports for vendored Three.js and OrbitControls. The viewer renders a light 3D scene, monitor plane/rectangle, axes, head ellipsoid, eyes, UniGaze ray or warning segment, current hit, and accumulated hit points through local controls.
- Kept the viewer build-tool-free: no package metadata, package manager files, remote assets, CDN references, or remote calls were added.

## Changed Files

- `src/chess_gaze/scene_viewer.py`
- `src/chess_gaze/viewer_assets/index.html`
- `src/chess_gaze/viewer_assets/scene_viewer.js`
- `src/chess_gaze/viewer_assets/styles.css`
- `src/chess_gaze/pipeline.py`
- `tests/chess_gaze/test_scene_viewer.py`
- `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`
- `.superpowers/sdd/task-8-report-3d-scene.md`

## Residual Risks

- The viewer behavior is covered by static and schema tests, plus local ESM import checks when Node is available. It was not browser-smoked in this task because Task 8 explicitly forbids adding Playwright and Task 9 owns the localhost viewer command/browser smoke.
- The monitor plane rendering uses the scene-space center and dimensions directly for a readable first viewer. It does not yet derive a rotated Three.js plane from the persisted camera basis; that may matter if future scene axes stop being near the viewer's default plane orientation.

## Review Fixes Round 1

Reviewer finding: standalone `build_scene_viewer(layout, build_scene_artifacts(layout))` wrote `viewer/scene-data.json` with `summary.artifact_validation.viewer_exists=True` while leaving persisted `scene/scene_summary.json` with `viewer_exists=False`.

RED evidence:

- Command: `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q`
- Result: `test_build_scene_viewer_updates_persisted_scene_summary` failed because `after_summary.artifact_validation.viewer_exists` was `False`.

Fix:

- Moved the viewer-exists update into `build_scene_viewer()`.
- `build_scene_viewer()` now calls `scene_result_with_viewer_exists(..., viewer_exists=True)`, rewrites `scene/scene_summary.json`, and writes `viewer/scene-data.json` from the same updated scene result.
- Removed duplicate pre-update and scene-summary rewrite logic from `pipeline.analyze_video()`.

Validation:

- Command: `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q`
- Result: `13 passed in 0.83s`.
- Command: `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py -q`
- Result: `33 passed in 1.21s`.
- Real-video checkpoint intentionally skipped for this review fix because the change is metadata consistency only; it does not change frame decoding, scene geometry, hit generation, or viewer frame/hit content.
