# Visualization Redo Closeout

Date: 2026-07-04

## Summary

The scene viewer visualization was simplified around hit areas only. Hit Points
UI, materials, cached point geometry, current-frame point rendering, and
accumulated point rendering were removed. Hit Area remains available with a
minimum and default angular error of `0.5 deg` and a default opacity of `4%`.
Scene loads now initialize the frame controls to the last available frame, and
successful loads clear and hide both status surfaces instead of showing the
former accumulated/frame sentence.

## Durable Surface Changed

The durable runtime surface is the generated viewer asset bundle under
`src/chess_gaze/viewer_assets/`:

- `index.html` owns the available controls and default form values.
- `scene_viewer.js` owns frame initialization, status rendering, hit-area
  aggregation, and the Three.js scene objects.
- `styles.css` owns fallback status visibility.
- `tests/chess_gaze/test_scene_viewer.py` now pins the viewer contract with
  generated HTML assertions, JS source assertions, and executable Node probes.

No persisted scene schema or run artifact format changed.

## Root Cause

The previous viewer contract exposed two visualization concepts, Hit Points and
Hit Area. The requested product behavior only needs Hit Area, but the old code
kept separate point controls, point materials, point geometry builders, and
status text tied to per-frame sphere-hit validity. The fix removed that
obsolete presentation branch at the viewer boundary instead of hiding it with a
temporary toggle default.

## Review

A subagent review found no critical defects. The important findings were fixed:
the missing closeout was added and stale README wording that still referenced
hit points was removed. The minor finding was also fixed by adding executable
coverage for status surface hide/show behavior.

## Verification

- Baseline focused viewer suite before implementation:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q`
  -> `33 passed in 8.31s`.
- Test-first RED check after updating tests:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -m "not local_socket" -q`
  -> six expected failures against the old Hit Points/default/status behavior.
- Final focused viewer suite:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q`
  -> `36 passed in 1.30s`.
- Broad non-native, non-socket suite:
  `UV_CACHE_DIR=.uv-cache uv run pytest -q -m "not native_mediapipe and not local_socket" --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py`
  -> `418 passed, 12 deselected, 18 warnings in 4.38s`.
- Lint:
  `UV_CACHE_DIR=.uv-cache uv run ruff check .` -> `All checks passed!`.
- Format check:
  `UV_CACHE_DIR=.uv-cache uv run ruff format --check .` -> `71 files already formatted`.
- Type check:
  `UV_CACHE_DIR=.uv-cache uv run mypy` -> `Success: no issues found in 71 source files`.
- JavaScript syntax:
  `node --check src/chess_gaze/viewer_assets/scene_viewer.js` -> passed.

## Browser Smoke

The generated viewer was opened with the gstack browse tool from a temporary run
at `/private/tmp/chess-gaze-smoke.KjaMVo/run/viewer/index.html`. DOM inspection
confirmed:

- `Hit Points` was absent.
- `Hit Area` remained present.
- Angular Error loaded with `min="0.5"`, `value="0.5"`, and label `0.5 deg`.
- Opacity loaded with `value="0.04"` and label `4%`.
- The removed accumulated/frame sentence was absent.

Canvas and pixel verification could not be completed in that headless browser
session because Chromium failed to create a WebGL context with:
`THREE.WebGLRenderer: Error creating WebGL context`.

## Residual Risk

The full native/real-video gate was attempted separately and aborted in the
native MediaPipe face landmarker path in
`tests/chess_gaze/test_eye_observation_real_video.py` with `Fatal Python error:
Aborted`. That failure is outside the edited viewer asset surface. The broad
non-native suite and focused viewer suite passed after the implementation.
