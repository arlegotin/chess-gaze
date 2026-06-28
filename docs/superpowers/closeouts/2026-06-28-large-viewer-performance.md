# Large Scene Viewer Performance Closeout

Date: 2026-06-28

## Summary

Fixed the large generated scene viewer without dropping data. The explicit file
URL entrypoint remains `viewer/index.html` and still embeds the full Carlsen
scene payload. The loopback server now serves a lightweight `served.html` that
loads `scene-data.json` externally, avoiding duplicated JSON parse work when the
viewer is opened through `chess-gaze view`.

The main runtime fix is in the Three.js scene asset:

- accumulated hit points are one `THREE.Points` object;
- accumulated hit areas are one indexed `BufferGeometry`;
- frame changes update draw ranges by prefix count;
- angular-error changes update the existing hit-area position buffer in place;
- frustum culling is disabled for the single accumulated hit-area mesh because
  the buffer positions mutate in place as angular error changes;
- rendering is scheduled on demand instead of continuously resizing and rendering.

## Root Cause

The old accumulated mode rebuilt rendering work proportional to all prior frames.
On the target run, slider changes near the end traversed 16,050 frames and
recreated thousands of sphere and hit-area mesh objects. That turned normal frame
navigation into multi-second main-thread work and made the UI appear frozen.

The durable surface was the generated browser app, not scene record generation.
The scene data was already complete and valid; the renderer was using the data in
an object-per-hit, rebuild-per-frame shape.

## Artifacts Analyzed

Target run:

- `artifacts/output/carlsen_1/runs/20260628T101348Z-e546cf6a`
- 16,050 viewer frames
- 15,459 valid hit points
- rebuilt `viewer/index.html`: 58,245,968 bytes
- rebuilt `viewer/served.html`: 6,264 bytes
- rebuilt `viewer/scene-data.json`: 87,967,347 bytes

Real verification run:

- input: `artifacts/input/nakamura_short.mp4`
- fresh run: `artifacts/output/nakamura_short/runs/20260628T143730Z-7a097c03`
- final status: complete
- decoded/frame/scene records: 180 / 180 / 180
- schema validation: true
- counts match: true
- viewer frames: 180
- valid hit points: 180
- rebuilt `viewer/index.html`: 697,043 bytes
- rebuilt `viewer/served.html`: 6,264 bytes
- rebuilt `viewer/scene-data.json`: 993,235 bytes

## Third-Party Docs Checked

Verified on 2026-06-28:

- Three.js `BufferGeometry` documentation/source for attribute buffers, indices,
  draw ranges, and disposal:
  <https://threejs.org/docs/#api/en/core/BufferGeometry>
- Three.js `Points` and `PointsMaterial` documentation for rendering many point
  markers with one geometry/material:
  <https://threejs.org/docs/#api/en/objects/Points>
  <https://threejs.org/docs/#api/en/materials/PointsMaterial>
- Three.js `InstancedMesh` documentation as an alternative for many repeated
  meshes. It would reduce draw calls, but it would still keep sphere meshes and
  was less direct than `Points` for hit markers:
  <https://threejs.org/docs/#api/en/objects/InstancedMesh>
- Three.js `MeshBasicMaterial` documentation. Hit areas do not need normals for
  lighting because this material is not light-reactive:
  <https://threejs.org/docs/#api/en/materials/MeshBasicMaterial>
- MDN `requestAnimationFrame()` documentation for one-shot repaint scheduling:
  <https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame>
- MDN `ResizeObserver` documentation for event-driven canvas resize handling:
  <https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver>
- MDN Web Workers documentation for the structured-clone/transferable tradeoff.
  A worker was not used because the hot path was fixed by changing geometry
  ownership and update granularity; moving 80+ MB payloads across worker
  boundaries would add complexity without addressing draw-object churn:
  <https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers>

## Before And After

Baseline on the original Carlsen file URL showed main-thread slider stalls:

| Frame value | Before sync ms | Before two-frame ms |
| --- | ---: | ---: |
| 0 | 3 | 468 |
| 100 | 945 | 1,913 |
| 1,000 | 7,437 | 8,827 |
| 4,000 | 2,584 | 4,146 |
| 8,000 | 5,597 | 13,475 |
| 12,000 | 16,124 | 25,783 |
| 16,049 | 8,162 | 16,905 |

After rebuild, exact file URL `viewer/index.html`:

| Action | After sync ms |
| --- | ---: |
| frame 0 | 2 |
| frame 100 | 1 |
| frame 1,000 | 1 |
| frame 4,000 | 0 |
| frame 8,000 | 0 |
| frame 12,000 | 0 |
| frame 16,049 | 0 |
| opacity change | 0 |
| angular error 4 deg | 8 |
| angular error 12 deg | 4 |
| hit-area off | 1 |
| hit-area on | 0 |

After rebuild, served Carlsen viewer:

| Action | After sync ms |
| --- | ---: |
| frame sweep 0..16,049 | 0-2 |
| layer toggles | 0-5 |
| opacity changes | 0-1 |
| angular error 4/12 deg | 11 / 4 |

After rebuild, fresh Nakamura viewer:

| Action | After sync ms |
| --- | ---: |
| frame sweep 0..179 | 0-2 |
| layer toggles | 0 |
| opacity change | 0 |
| angular error 4/12 deg | 1 / 0 |

The two-frame timing column is not used as final evidence after the fix because
the DevTools-controlled Chrome tab throttled `requestAnimationFrame` to about 1
Hz while idle. Direct synchronous input timings and page state checks remained
stable, and the same tab showed no console errors.

## Data Preservation

No scene records or valid-hit records are removed. The renderer keeps all
accumulated point and hit-area data in buffers and reveals prefixes with
`setDrawRange()`. The final Carlsen file URL reported all 16,050 frames and all
15,459 valid hits from embedded scene data, with accumulated status reaching
`15459 of 15459` at the final frame. The fresh Nakamura run reported all 180
frames and 180 hits.

## Gates

Passed:

- `node --check src/chess_gaze/viewer_assets/scene_viewer.js`
- `UV_CACHE_DIR=.uv-cache uv run ruff check .`
- `UV_CACHE_DIR=.uv-cache uv run ruff format --check .`
- `UV_CACHE_DIR=.uv-cache uv run mypy`
- `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q`
  - 30 passed
- `UV_CACHE_DIR=.uv-cache uv run pytest -q`
  - 372 passed
  - 18 existing torch `jit.script` deprecation warnings

Browser checks:

- Carlsen served viewer loaded `served.html`, `scene_viewer.js`, and
  `scene-data.json`; no console errors.
- Carlsen exact file URL loaded embedded scene data and blob module source; no
  console errors.
- Carlsen exact file URL WebGL pixel hashes differed for hit-area on at 8 deg,
  hit-area off, hit-area on at 4 deg, and hit-area on at 12 deg. Final state
  remained `16050 / 16050` and `15459 of 15459`.
- Nakamura served viewer loaded `served.html`, `scene_viewer.js`, and
  `scene-data.json`; no console errors.
- gstack headless browser could not create a WebGL context in this environment,
  so it was not used as visual evidence.
- Chrome DevTools screenshot capture timed out even with the hit area disabled,
  while JavaScript evaluation and WebGL `readPixels()` checks remained
  responsive. This is recorded as a tooling limitation rather than ignored.

## Follow-Up Risk

The generated viewer asset is intentionally a deep module and is documented in
`docs/development/architecture/source-layout.md`. If it crosses roughly 1,500
lines or gains a second independent viewer mode, split planning should happen
before adding more responsibilities.

Runtime visual correctness is protected by source-contract tests, scene-data
contract tests, browser state checks, and WebGL pixel-hash checks. The only
residual verification gap is that this local environment could not produce
ordinary screenshot artifacts for the WebGL canvas.
