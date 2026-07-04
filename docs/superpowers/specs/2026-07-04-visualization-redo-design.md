# Visualization Redo Design

Date: 2026-07-04

## Status

Approved by direct user request on 2026-07-04. The user also approved working
directly on `main` and requested meaningful commits along the way.

This spec supersedes the viewer-control portions of:

- `docs/superpowers/specs/2026-06-27-hit-area-viewer-design.md`
- `docs/superpowers/specs/2026-06-29-sphere-hit-projection-design.md`

Historical specs and closeouts may keep their historical facts, but current
viewer behavior is defined here.

Follow-up correction: the user clarified on 2026-07-04 that Hit Points must be
removed from viewer data as well. The active data-contract design is
`docs/superpowers/specs/2026-07-04-hit-area-only-viewer-data-design.md`.

## Goal

Simplify the 3D scene viewer's gaze-hit visualization so the gaze sphere shows
hit-area patches only, starts at the last frame, and avoids repeated status
sentences over the scene.

## Required Behavior

- Permanently remove the `Hit Points` visualization control and rendered hit
  point layer from the browser viewer.
- Keep the `Hit Area` layer, default checked.
- Set the angular-error slider minimum to `0.5`, maximum to `12`, step to `0.5`,
  and default to `0.5`.
- Set hit-area opacity default to `0.04` and show the default readout as `4%`.
- When scene data loads, initialize the frame slider and numeric frame input to
  the last available frame. Empty scene data remains at `0`.
- Remove the live frame sentence from both status-message locations. In
  particular, neither location should display strings like
  `Accumulated mode. Frame 1 of 16050: sphere hit is valid.`
- Keep genuine loading and error messages visible, including missing scene-data
  and module-import failures.
- The original visualization-only request did not change persisted
  scene/viewer schemas. The follow-up hit-area-only data contract supersedes
  that constraint for `viewer/scene-data.json`.

## Implementation Surface

Expected source changes:

- `src/chess_gaze/viewer_assets/index.html`
- `src/chess_gaze/viewer_assets/scene_viewer.js`
- `src/chess_gaze/viewer_assets/styles.css`
- `src/chess_gaze/scene_viewer.py`
- `tests/chess_gaze/test_scene_viewer.py`
- `README.md`

No Python schema module should change. `scene_viewer.js` is an intentionally deep
single packaged viewer asset; this task removes behavior and does not justify a
new frontend module or build pipeline.

## Design Notes

Hit points were initially removed from the visualization surface only. The
follow-up hit-area-only data spec supersedes that decision and removes the
duplicated `ViewerSceneData.valid_hit_points` payload too. Per-frame
`sphere_hit` records remain because Hit Area derives from them.

Accumulated hit-area prefix counts should continue to use the cached
`hitAreaPatchFrameIndices` array. The top-bar `Hits` metric can continue to
show the total valid sphere-hit count from scene data because it is summary
information, not a rendered layer.

The fallback status overlay remains for loading and errors only. Successful
scene loading should hide the overlay rather than replacing it with a per-frame
sentence. The header status text should likewise be empty after successful
scene loading.

## Testing

Use test-first development.

Focused tests:

- Generated HTML no longer includes `toggle-hit-points` or visible `Hit Points`.
- Generated JS no longer queries, renders, or toggles hit points.
- Angular-error controls expose min/default `0.5` and label `0.5 deg`.
- Opacity controls expose default `0.04` and label `4%`.
- `applySceneData()` calls `setFrameIndex(maxIndex)`, not `setFrameIndex(0)`.
- Runtime probes verify empty scene data still initializes frame index `0`.
- Runtime probes verify successful status updates do not write the removed
  frame sentence to `frame-status` or `.fallback-status`.
- Error paths still write error messages to both status surfaces.

Focused command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

`local_socket` tests require unsandboxed loopback socket permission.

Required gates:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
UV_CACHE_DIR=.uv-cache uv run pytest -q --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Browser smoke, if local serving/browser tooling is available:

- Load a generated viewer.
- Verify the canvas is nonblank.
- Verify frame slider and frame number initialize to the last frame.
- Verify `Hit Points` is absent.
- Verify `Hit Area`, angular-error, opacity, and sphere-radius controls remain.
- Verify no removed frame sentence appears in the header or overlay.

## Acceptance Criteria

1. The viewer has no `Hit Points` control.
2. The viewer draws no current or accumulated hit-point visualization.
3. `Hit Area` remains available and default checked.
4. Angular error defaults to `0.5 deg` and cannot go below `0.5`.
5. Hit-area opacity defaults to `4%`.
6. Scene load starts on the last frame for non-empty scene data.
7. Empty scene data remains safe at frame index `0`.
8. Removed live frame sentence is absent from both status-message locations.
9. Loading and error status messages are still shown.
10. No schema, model, checkpoint, dependency, or external URL changes are made.
11. Tests and closeout document the verified subset and any blocked checks.
