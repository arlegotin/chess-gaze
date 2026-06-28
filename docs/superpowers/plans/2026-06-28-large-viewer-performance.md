# Large Viewer Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generated scene viewers with large per-frame datasets smooth and responsive while preserving every frame record and every valid hit.

**Architecture:** Keep the strict scene data contract intact. Change the viewer from per-frame object recreation to cached GPU-friendly collections: accumulated hit points become one `THREE.Points` object backed by `BufferGeometry`, and accumulated hit areas become one reusable indexed `BufferGeometry` whose draw range exposes a prefix of precomputed per-frame patch vertices. Keep direct `file://` compatibility for generated `index.html`, but stop embedding scene data when the viewer is served over localhost by adding a separate `standalone.html` for direct file opening.

**Tech Stack:** Python 3.12, pytest, generated static HTML/CSS/JS viewer assets, browser smoke tests, Three.js `0.185.0` from the existing ADR-0003 import map.

## Global Constraints

- Work in the current branch.
- Do not throw away, downsample, cluster, merge, smooth, or otherwise reduce any scene frame or hit data.
- Preserve one `ViewerSceneData.frames[]` entry per decoded frame.
- Preserve one valid hit point per valid monitor-hit frame.
- Do not change scene artifact schemas.
- Do not add a frontend build system or new runtime/browser dependency.
- Keep Three.js `0.185.0` loaded from pinned jsDelivr URLs per ADR-0003.
- Keep direct `file://` viewing available for generated artifacts.
- Prefer `chess-gaze view <run-dir>` for large runs so the default served viewer can load `scene-data.json` instead of parsing an embedded duplicate.
- Accumulated hit areas must still derive from `frames[]`, not `valid_hit_points[]`.
- `Hit Points` and `Hit Area` remain independent layer toggles.
- Opacity changes must not rebuild hit-area geometry.
- Angular-error changes may rebuild cached hit-area geometry once, then frame changes must only update visibility/draw ranges.
- Use test-first development.
- Use `artifacts/input/nakamura_short.mp4` for real verification.
- Make meaningful commits along the way.

---

## Investigation Summary

Measured target artifact:

- `viewer/index.html`: 58,237,571 bytes.
- `viewer/scene-data.json`: 87,967,347 bytes.
- `frames[]`: 16,050 entries.
- `valid_hit_points[]`: 15,459 entries.
- `records/scene_frames.jsonl`: 16,050 records and semantically identical to `viewer.scene-data.json.frames`.
- `valid_hit_points[]` is derivable from `frames[].main_monitor_hit`, but must remain in the JSON contract for compatibility.

Measured Chrome DevTools before-fix timings on the target viewer:

- A single accumulated slider move to frame 12,000 took `16,124 ms` synchronously and `25,783 ms` through two paints.
- With `Hit Area` off and `Hit Points` on, a move to frame 8,000 took `24,249 ms` synchronously and `31,114 ms` through two paints.
- With `Hit Points` off and `Hit Area` on, a move to frame 16,049 took `4,542 ms` synchronously and `24,548 ms` through two paints.

Root cause:

- `renderAccumulatedHits()` clears and recreates all accumulated hit-point spheres on every frame change.
- Each hit point calls `new THREE.SphereGeometry(radius, 24, 16)`, producing thousands of mesh/geometries for large runs.
- `renderAccumulatedHitAreas()` clears and recreates all accumulated hit-area geometries and meshes on every frame change.
- `updateStatusPanel()` filters all valid hit points on every frame change.
- The animation loop reads canvas layout and calls resize/render every `requestAnimationFrame`, even when nothing changed.
- The generated `index.html` embeds the full scene data and duplicates `scene-data.json`; served viewing should not pay this duplicate HTML parse path.

Third-party evidence checked on 2026-06-28:

- Three.js `0.185.0` npm metadata: MIT, repository `git+https://github.com/mrdoob/three.js.git`, integrity `sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEAuj25bNAj7k1QQdf+srZywVK6w==`.
- Three.js `InstancedMesh` source says instancing is for rendering many objects with the same geometry/material and helps reduce draw calls.
- Three.js `BufferGeometry` source documents `setDrawRange(start, count)`, `setAttribute()`, and `dispose()` for GPU resources.
- Three.js `Points` source uses `BufferGeometry` and honors geometry draw ranges in raycasting/render paths.
- MDN Web Workers docs say workers run scripts in background threads without interfering with the UI, but data passed between workers and the page is copied unless transferable objects are used.
- MDN `requestAnimationFrame()` docs say callbacks run before repaint, normally match display refresh, and are one-shot.
- MDN `ResizeObserver` docs define it as reporting changes to element content or border box dimensions.

Decision:

- Do not add a worker for this fix. The current dominant cost is Three.js object churn and draw-call/object count, not only JSON parsing. Worker transfer would copy the already large parsed object unless introducing a new binary data format, which would be a schema and artifact decision outside this bug fix.
- Use `THREE.Points` for accumulated hit points instead of sphere meshes. This preserves every hit as one rendered point and matches the original 3D viewer spec's point-cloud intent.
- Use one cached indexed `BufferGeometry` for accumulated hit-area triangle fans. Preserve every valid patch, but reveal only the prefix up to the selected frame through `setDrawRange()`.
- Keep current-frame head, eyes, ray, current hit sphere, and current hit-area patch as ordinary current-frame objects because their count is constant.

---

### Task 1: Python Viewer Output Contract

**Files:**
- Modify: `src/chess_gaze/scene_viewer.py`
- Modify: `src/chess_gaze/viewer_assets/index.html`
- Modify: `tests/chess_gaze/test_scene_viewer.py`

**Interfaces:**
- Consumes: `build_scene_viewer()`, `write_viewer_scene_data()`, existing template `viewer_assets/index.html`.
- Produces: generated `viewer/index.html`, `viewer/standalone.html`, `viewer/scene-data.json`.

- [ ] **Step 1: Write failing tests for served/standalone split**

In `tests/chess_gaze/test_scene_viewer.py`, add or update tests so a built viewer must satisfy:

```python
def test_build_scene_viewer_writes_server_and_standalone_indexes(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer

    assert (layout.viewer_dir / "index.html").is_file()
    assert (layout.viewer_dir / "standalone.html").is_file()
    assert (layout.viewer_dir / "scene-data.json").is_file()


def test_generated_index_fetches_scene_data_without_embedding_payload(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, viewer_data = built_viewer
    html = (layout.viewer_dir / "index.html").read_text(encoding="utf-8")

    assert 'type="module" src="./scene_viewer.js"' in html
    assert 'id="scene-data-json"' not in html
    assert 'id="scene-viewer-source"' not in html
    assert "window.__CHESS_GAZE_SCENE_DATA__" not in html
    assert viewer_data.run_id not in html


def test_generated_standalone_embeds_file_url_bootstrap_and_scene_data(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, viewer_data = built_viewer
    html = (layout.viewer_dir / "standalone.html").read_text(encoding="utf-8")

    assert 'type="module" src="./scene_viewer.js"' not in html
    assert 'id="scene-data-json"' in html
    assert 'id="scene-viewer-source"' in html
    assert "window.__CHESS_GAZE_SCENE_DATA__" in html
    assert viewer_data.run_id in html
```

Update existing import-map/external URL tests to check both `index.html` and `standalone.html` where appropriate.

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_build_scene_viewer_writes_server_and_standalone_indexes tests/chess_gaze/test_scene_viewer.py::test_generated_index_fetches_scene_data_without_embedding_payload tests/chess_gaze/test_scene_viewer.py::test_generated_standalone_embeds_file_url_bootstrap_and_scene_data -q
```

Expected before implementation: failures for missing `standalone.html` and embedded payload still present in `index.html`.

- [ ] **Step 3: Implement split output**

Change `build_scene_viewer()` so it:

1. copies packaged assets;
2. writes `scene-data.json`;
3. leaves `index.html` as the served/lightweight template with normal module script and no embedded scene data;
4. writes `standalone.html` as the file-url-compatible embedded copy.

Rename `_write_file_url_compatible_index()` to `_write_file_url_compatible_standalone()` and write to `standalone.html`. Keep the existing embedded bootstrap behavior in `standalone.html`.

Update `_file_url_bootstrap_script()` error text if needed, but do not change scene data semantics.

- [ ] **Step 4: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

Expected after implementation: all viewer tests pass.

- [ ] **Step 5: Commit**

Run:

```sh
git add src/chess_gaze/scene_viewer.py src/chess_gaze/viewer_assets/index.html tests/chess_gaze/test_scene_viewer.py
git commit -m "fix: split served and standalone viewer data loading"
```

### Task 2: Cached Accumulated Viewer Geometry

**Files:**
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Modify: `tests/chess_gaze/test_scene_viewer.py`

**Interfaces:**
- Consumes: existing `ViewerSceneData.frames[]`, `valid_hit_points[]`, hit-area controls, hit-point toggle, hit-area toggle.
- Produces: cached accumulated hit-point and hit-area render state with prefix visibility.

- [ ] **Step 1: Write failing tests for source contract**

In `tests/chess_gaze/test_scene_viewer.py`, add source-contract assertions that require:

```python
def test_generated_viewer_caches_accumulated_geometry_for_large_runs(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    js = (layout.viewer_dir / "scene_viewer.js").read_text(encoding="utf-8")

    assert "new THREE.Points(" in js
    assert "new THREE.PointsMaterial(" in js
    assert "buildAccumulatedHitPoints" in js
    assert "buildAccumulatedHitAreaMesh" in js
    assert "setDrawRange(0, visibleHitAreaTriangleIndexCount" in js
    assert "hitPointFrameIndices" in js
    assert "hitAreaPatchFrameIndices" in js
    assert "upperBoundFrameIndex" in js
    assert "for (const hit of state.sceneData.valid_hit_points)" not in js
    assert "state.sceneData.frames.slice(0, state.frameIndex + 1)" not in js
```

Add a second assertion to keep the point and area toggles independent:

```python
def test_generated_viewer_keeps_accumulated_layers_independent(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    js = (layout.viewer_dir / "scene_viewer.js").read_text(encoding="utf-8")

    visibility_body = js.split("function updateAccumulatedVisibility() {", 1)[1].split(
        "\nfunction ", 1
    )[0]
    assert "elements.toggles.hitPoints.checked" in visibility_body
    assert "elements.toggles.hitArea.checked" in visibility_body
    assert "hitPoints.checked && hitArea.checked" not in visibility_body
```

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_caches_accumulated_geometry_for_large_runs tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_keeps_accumulated_layers_independent -q
```

Expected before implementation: failures for missing cached geometry functions and old loop patterns.

- [ ] **Step 3: Add render cache state**

In `scene_viewer.js`, add state fields:

```js
renderCache: {
  hitPoints: null,
  hitPointFrameIndices: [],
  hitAreas: null,
  hitAreaPatchFrameIndices: [],
  hitAreaTriangleIndexCounts: [],
  hitAreaAngularErrorDegrees: null,
},
renderRequested: false,
animationFrameRequested: false,
canvasWidth: 0,
canvasHeight: 0,
```

Add shared reusable objects near helpers if needed:

```js
const scratchMatrix = new THREE.Matrix4();
```

- [ ] **Step 4: Replace accumulated hit-point spheres with `THREE.Points`**

Implement:

```js
function upperBoundFrameIndex(frameIndices, frameIndex) { ... }

function buildAccumulatedHitPoints() {
  const positions = [];
  const frameIndices = [];
  for (const hit of state.sceneData.valid_hit_points || []) {
    const point = finiteVector(hit.point_scene_m);
    if (point && Number.isInteger(hit.frame_index)) {
      positions.push(point.x, point.y, point.z);
      frameIndices.push(hit.frame_index);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setDrawRange(0, 0);
  const points = new THREE.Points(geometry, materials.accumulatedHitPoints);
  points.userData.layer = "hitPoints";
  groups.accumulated.add(points);
  state.renderCache.hitPoints = points;
  state.renderCache.hitPointFrameIndices = frameIndices;
}
```

Use `THREE.PointsMaterial` for accumulated hit points. Keep current-frame hit point as a sphere.

- [ ] **Step 5: Replace accumulated hit-area meshes with one cached geometry**

Implement:

```js
function hitAreaPatchVertices(frame, angularErrorDegreesValue) { ... }

function buildAccumulatedHitAreaMesh() {
  const positions = [];
  const indices = [];
  const patchFrameIndices = [];
  const triangleIndexCounts = [];
  for (const frame of state.sceneData.frames || []) {
    const patchVertices = hitAreaPatchVertices(frame, angularErrorDegrees());
    if (!patchVertices || !Number.isInteger(frame.frame_index)) {
      continue;
    }
    const vertexOffset = positions.length / 3;
    positions.push(...patchVertices);
    for (let index = 0; index < HIT_AREA_SEGMENTS; index += 1) {
      indices.push(
        vertexOffset,
        vertexOffset + index + 1,
        vertexOffset + ((index + 1) % HIT_AREA_SEGMENTS) + 1,
      );
    }
    patchFrameIndices.push(frame.frame_index);
    triangleIndexCounts.push(HIT_AREA_SEGMENTS * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setIndex(indices);
  geometry.setDrawRange(0, 0);
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, materials.hitArea);
  mesh.userData.layer = "hitArea";
  groups.accumulated.add(mesh);
  state.renderCache.hitAreas = mesh;
  state.renderCache.hitAreaPatchFrameIndices = patchFrameIndices;
  state.renderCache.hitAreaTriangleIndexCounts = triangleIndexCounts;
  state.renderCache.hitAreaAngularErrorDegrees = angularErrorDegrees();
}
```

Do not use `frames.slice()`. Do not create one mesh per accumulated patch.

- [ ] **Step 6: Update prefix visibility only on frame changes**

Implement:

```js
function visibleHitPointCount() { ... }
function visibleHitAreaTriangleIndexCount() { ... }
function updateAccumulatedVisibility() { ... }
function rebuildAccumulatedHitAreasForAngularError() { ... }
```

`setFrameIndex()` should:

1. update the current frame;
2. update cached accumulated visibility;
3. update status.

It must not rebuild all accumulated objects on normal slider movement.

- [ ] **Step 7: Update hit-area controls and toggles**

Angular-error input should rebuild the hit-area cache once:

```js
elements.hitAreaErrorDegrees.addEventListener("input", () => {
  updateHitAreaErrorLabel();
  renderCurrentFrame();
  rebuildAccumulatedHitAreasForAngularError();
  updateAccumulatedVisibility();
  requestRender();
});
```

Opacity input must only update material opacity and request render.

- [ ] **Step 8: Verify GREEN**

Run:

```sh
node --check src/chess_gaze/viewer_assets/scene_viewer.js
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

Expected after implementation: JavaScript parses and viewer tests pass.

- [ ] **Step 9: Commit**

Run:

```sh
git add src/chess_gaze/viewer_assets/scene_viewer.js tests/chess_gaze/test_scene_viewer.py
git commit -m "fix: cache accumulated viewer geometry"
```

### Task 3: On-Demand Rendering and Status Prefix Counts

**Files:**
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Modify: `tests/chess_gaze/test_scene_viewer.py`

**Interfaces:**
- Consumes: existing `setFrameIndex()`, playback controls, OrbitControls, resize behavior.
- Produces: rendering that occurs when state changes, while playing, or while controls are damping.

- [ ] **Step 1: Write failing tests for render scheduling**

Add source-contract assertions:

```python
def test_generated_viewer_renders_on_demand_and_uses_prefix_counts(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    js = (layout.viewer_dir / "scene_viewer.js").read_text(encoding="utf-8")

    assert "function requestRender()" in js
    assert "function renderFrame()" in js
    assert "new ResizeObserver(" in js
    assert 'controls.addEventListener("change", requestRender)' in js
    assert "window.requestAnimationFrame(renderFrame)" in js
    assert "state.sceneData?.valid_hit_points.filter" not in js
    assert "validHitsToFrame = visibleHitPointCount()" in js
    assert "resizeRenderer();" not in js.split("function renderFrame()", 1)[1].split(
        "\n}", 1
    )[0]
```

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_renders_on_demand_and_uses_prefix_counts -q
```

Expected before implementation: missing `requestRender()`, `ResizeObserver`, and prefix-count status behavior.

- [ ] **Step 3: Add on-demand render scheduler**

Replace the unconditional `animate()` loop with:

```js
function requestRender() {
  state.renderRequested = true;
  if (!state.animationFrameRequested) {
    state.animationFrameRequested = true;
    window.requestAnimationFrame(renderFrame);
  }
}

function renderFrame() {
  state.animationFrameRequested = false;
  const controlsNeedRender = controls.enableDamping && controls.update();
  if (state.renderRequested || state.playing || controlsNeedRender) {
    state.renderRequested = false;
    renderer.render(scene, camera);
  }
  if (state.playing || controlsNeedRender || state.renderRequested) {
    requestRender();
  }
}
```

Every state-changing control handler must call `requestRender()` after changing scene objects.

- [ ] **Step 4: Use `ResizeObserver` for canvas size**

Replace per-frame size reads with:

```js
function resizeRenderer() {
  const rect = elements.canvas.getBoundingClientRect();
  ...
  requestRender();
}

const resizeObserver = new ResizeObserver(resizeRenderer);
resizeObserver.observe(elements.canvas);
```

Keep `window.addEventListener("resize", resizeRenderer)` as a fallback if desired.

- [ ] **Step 5: Use prefix counts in status**

`updateStatusPanel()` should use `visibleHitPointCount()` instead of filtering all hits on every frame.

- [ ] **Step 6: Verify GREEN**

Run:

```sh
node --check src/chess_gaze/viewer_assets/scene_viewer.js
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

- [ ] **Step 7: Commit**

Run:

```sh
git add src/chess_gaze/viewer_assets/scene_viewer.js tests/chess_gaze/test_scene_viewer.py
git commit -m "fix: render scene viewer on demand"
```

### Task 4: Real Artifact Verification and Documentation Closeout

**Files:**
- Modify: `docs/superpowers/closeouts/2026-06-28-large-viewer-performance.md`
- Modify: `README.md` only if the correct user workflow changes.

**Interfaces:**
- Consumes: implemented viewer changes, target carlsen artifact, fresh `nakamura_short.mp4` run.
- Produces: measured verification evidence and closeout.

- [ ] **Step 1: Run focused tests and static checks**

Run:

```sh
node --check src/chess_gaze/viewer_assets/scene_viewer.js
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

- [ ] **Step 2: Generate fresh Nakamura short run**

Run:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK \
  -u PYTORCH_MPS_FAST_MATH \
  -u PYTORCH_MPS_PREFER_METAL \
  UV_CACHE_DIR=.uv-cache \
  uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 \
  --output-root artifacts/output \
  --models-root models \
  --no-resume
```

Record stdout run directory and viewer path.

- [ ] **Step 3: Verify fresh run artifacts**

Run the QA audit:

```sh
RUN_DIR=<fresh-run-dir> .venv/bin/python -B - <<'PY'
from pathlib import Path
from chess_gaze.qa_summary import QASummary
from chess_gaze.scene_records import ViewerSceneData
import os

run = Path(os.environ["RUN_DIR"])
qa = QASummary.model_validate_json((run / "qa_summary.json").read_text())
viewer = ViewerSceneData.model_validate_json((run / "viewer/scene-data.json").read_text())

print(qa.final_status)
print(qa.counts.model_dump())
print(qa.artifact_validation.schema_validation_passed)
print(qa.artifact_validation.counts_match)
print(viewer.frame_count, len(viewer.frames), len(viewer.valid_hit_points))
print((run / "viewer/index.html").stat().st_size)
print((run / "viewer/standalone.html").stat().st_size)
print((run / "viewer/scene-data.json").stat().st_size)
PY
```

Expected: complete status, schema/count validation true, 180 viewer frames for `nakamura_short.mp4`, lightweight `index.html`, embedded `standalone.html`.

- [ ] **Step 4: Browser-measure target large artifact after rebuilding viewer**

Regenerate or patch the target run viewer with the new build code without changing source scene data. Use a helper command or Python snippet that loads the run's scene artifacts and calls `build_scene_viewer()` if an existing API supports it.

Serve the rebuilt viewer:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view artifacts/output/carlsen_1/runs/20260628T101348Z-e546cf6a
```

Using Chrome DevTools, measure slider changes at values `100`, `1000`, `4000`, `8000`, `12000`, and `16049` with both Hit Points and Hit Area enabled. Record synchronous and two-frame timings. The target is smooth human interaction: normal slider moves should complete in tens of milliseconds, not seconds, and must remain below `250 ms` for the measured values on the local machine.

- [ ] **Step 5: Browser smoke fresh Nakamura short viewer**

Serve the fresh Nakamura short run and verify:

- no console errors;
- canvas nonblank;
- slider responds;
- `Hit Area` toggle changes rendered pixels;
- angular-error slider changes rendered pixels;
- `Hit Points` toggle remains independent from `Hit Area`.

- [ ] **Step 6: Full suite if practical**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

If blocked by local native/runtime constraints, record exact output and run the broadest meaningful subset.

- [ ] **Step 7: Write closeout**

Create `docs/superpowers/closeouts/2026-06-28-large-viewer-performance.md` with:

- root cause;
- artifacts analyzed;
- third-party docs evidence;
- implementation summary;
- before/after timings;
- Nakamura short real-run evidence;
- gates run;
- residual uncertainty.

- [ ] **Step 8: Commit closeout**

Run:

```sh
git add docs/superpowers/closeouts/2026-06-28-large-viewer-performance.md README.md
git commit -m "docs: close out large viewer performance fix"
```
