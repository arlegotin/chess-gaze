# Hit Area Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a viewer-side angular-error hit-area layer that preserves existing hit points, accumulates hit-area patches in accumulated mode, and verifies against `artifacts/input/nakamura_short.mp4`.

**Architecture:** Keep scene artifacts unchanged. Compute hit-area ellipses inside `scene_viewer.js` from existing `viewer.frames[]` fields and expose them through a separate checkbox plus live angular-error slider. Instant mode renders the current-frame patch; accumulated mode renders patches for all valid frames through the current slider frame.

**Tech Stack:** Python 3.12, pytest, generated static HTML/CSS/JS viewer assets, Three.js `0.185.0` loaded by the existing ADR-0003 import map.

## Global Constraints

- Work in the current branch.
- Do not change scene artifact schemas for this feature.
- Do not add new runtime or browser dependencies.
- Default angular error is `8` degrees.
- Angular-error slider range is `[0, 12]` degrees with redraw-on-input.
- Hit-area opacity slider range is `[0, 1]`, default `0.24`, with redraw-on-input.
- The hit point and hit area are separate visual layers.
- In accumulated mode, `Hit Area` renders patches for all valid frames through the current slider frame independently from `Hit Points`.
- `artifacts/input/nakamura_short.mp4` must be used for real verification.
- Treat the patch as a typical angular-error visualization, not measured per-frame confidence.
- Use test-first development.

---

### Task 1: Viewer Contract Tests

**Files:**
- Modify: `tests/chess_gaze/test_scene_viewer.py`
- Modify: `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`

**Interfaces:**
- Consumes: generated viewer asset files from `build_scene_viewer()`.
- Produces: failing tests for the HTML controls, JS math/wiring, CSS roles, and Nakamura short geometry fields.

- [ ] **Step 1: Add failing viewer selector/source tests**

Add assertions to `test_generated_html_includes_required_selectors()` for:

```python
'data-testid="toggle-hit-area"',
'data-testid="hit-area-error-degrees"',
'data-testid="hit-area-error-label"',
```

Add a new test:

```python
def test_generated_viewer_exposes_hit_area_controls_and_math(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    html = (layout.viewer_dir / "index.html").read_text(encoding="utf-8")
    js = (layout.viewer_dir / "scene_viewer.js").read_text(encoding="utf-8")
    css = (layout.viewer_dir / "styles.css").read_text(encoding="utf-8")

    assert "Hit Area" in html
    assert "Angular Error" in html
    assert 'min="0"' in html
    assert 'max="12"' in html
    assert 'step="0.5"' in html
    assert 'value="8"' in html
    assert "DEFAULT_HIT_AREA_ANGULAR_ERROR_DEGREES = 8" in js
    assert "HIT_AREA_MIN_ANGULAR_ERROR_DEGREES = 5" in js
    assert "HIT_AREA_MAX_ANGULAR_ERROR_DEGREES = 12" in js
    assert "rayT * Math.tan(alphaRadians)" in js
    assert "minorRadius / normalDirectionDot" in js
    assert "direction.clone().sub(" in js
    assert "renderCurrentHitArea" in js
    assert "--color-hit-area:" in css
    assert ".hit-area-error-row" in css
```

- [ ] **Step 2: Add failing Nakamura short geometry-field test**

In `test_model_free_nakamura_video_scene_artifact_contract()`, after `viewer_data`
is built, assert at least one valid frame has all fields the viewer uses:

```python
valid_frame = next(
    frame for frame in viewer_data.frames if frame.main_monitor_hit.valid
)
assert valid_frame.main_monitor_hit.t is not None
assert valid_frame.main_monitor_hit.point_scene_m is not None
assert valid_frame.main_monitor_hit.point_camera_m is not None
assert valid_frame.unigaze_ray.direction_scene is not None
assert valid_frame.unigaze_ray.direction_camera is not None
assert viewer_data.monitor_plane.normal_camera is not None
assert viewer_data.axis_basis.right_camera is not None
```

- [ ] **Step 3: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_html_includes_required_selectors tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math tests/chess_gaze/test_scene_artifacts_real_video_contract.py::test_model_free_nakamura_video_scene_artifact_contract -q
```

Expected before implementation: viewer tests fail on missing selectors and math strings. The real-video geometry field additions may already pass.

- [ ] **Step 4: Commit**

```sh
git add tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py
git commit -m "test: cover viewer hit area contract"
```

### Task 2: Viewer Implementation

**Files:**
- Modify: `src/chess_gaze/viewer_assets/index.html`
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Modify: `src/chess_gaze/viewer_assets/styles.css`

**Interfaces:**
- Consumes: current `ViewerSceneData` JSON, especially `frames[].unigaze_ray`, `frames[].main_monitor_hit`, `monitor_plane`, and `axis_basis`.
- Produces: `Hit Area` toggle, angular-error slider/readout, and current-frame translucent ellipse mesh.

- [ ] **Step 1: Add HTML controls**

Add `Hit Area` to the scene-layer fieldset and add a compact slider row:

```html
<label><input data-testid="toggle-hit-area" type="checkbox" checked> Hit Area</label>
<div class="hit-area-error-row">
  <label for="hit-area-error-degrees">Angular Error</label>
  <input
    id="hit-area-error-degrees"
    data-testid="hit-area-error-degrees"
    type="range"
    min="0"
    max="12"
    value="8"
    step="0.5"
  >
  <output data-testid="hit-area-error-label" for="hit-area-error-degrees">8 deg</output>
</div>
```

- [ ] **Step 2: Add JS constants and element bindings**

Add:

```js
const DEFAULT_HIT_AREA_ANGULAR_ERROR_DEGREES = 8;
const HIT_AREA_MIN_ANGULAR_ERROR_DEGREES = 5;
const HIT_AREA_MAX_ANGULAR_ERROR_DEGREES = 12;
const HIT_AREA_SEGMENTS = 72;
const HIT_AREA_PLANE_OFFSET_M = 0.001;
const HIT_AREA_VECTOR_EPSILON = 1e-8;
```

Bind:

```js
hitAreaErrorDegrees: document.querySelector('[data-testid="hit-area-error-degrees"]'),
hitAreaErrorLabel: document.querySelector('[data-testid="hit-area-error-label"]'),
hitArea: document.querySelector('[data-testid="toggle-hit-area"]'),
```

Add `hitAreaErrorDegrees` to disabled controls.

- [ ] **Step 3: Add hit-area material**

Add `hitArea` to `COLORS` and `materials`:

```js
hitArea: 0xc43d7a,
hitArea: new THREE.MeshBasicMaterial({
  color: COLORS.hitArea,
  transparent: true,
  opacity: DEFAULT_HIT_AREA_OPACITY,
  side: THREE.DoubleSide,
  depthWrite: false,
})
```

- [ ] **Step 4: Add vector helpers**

Implement helpers:

```js
function finiteNumber(value) { ... }
function normalizedVector(record) { ... }
function cameraDirectionToScene(cameraDirection) { ... }
function monitorNormalScene() { ... }
function monitorRightScene() { ... }
function angularErrorDegrees() { ... }
function updateHitAreaErrorLabel() { ... }
```

- [ ] **Step 5: Add geometry builder and renderer**

Implement:

```js
function hitAreaGeometry(frame, angularErrorDegreesValue) { ... }
function addHitArea(group, geometry) { ... }
function renderCurrentHitArea(frame) { ... }
```

`hitAreaGeometry()` must:

- require valid hit, valid point, finite `ray_t_m`, and usable direction;
- compute `minorRadius = rayT * Math.tan(alphaRadians)`;
- compute `majorRadius = minorRadius / normalDirectionDot`;
- orient the major axis along direction projected onto the plane;
- return a triangle-fan `THREE.BufferGeometry`.

Call `renderCurrentHitArea(frame)` before drawing the current hit sphere so the
point estimate remains visible.

- [ ] **Step 6: Add redraw wiring**

In `bindControls()`, add input handling:

```js
elements.hitAreaErrorDegrees.addEventListener("input", () => {
  updateHitAreaErrorLabel();
  renderCurrentFrame();
});
```

Keep the existing toggle loop so `toggle-hit-area` redraws with other layers.

- [ ] **Step 7: Add CSS**

Add a hit-area color variable and slider row styles:

```css
--color-hit-area: #c43d7a;

.hit-area-error-row { ... }
.hit-area-error-row label { ... }
.hit-area-error-row output { ... }
```

- [ ] **Step 8: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected after implementation: focused viewer and Nakamura short model-free tests pass.

- [ ] **Step 9: Commit**

```sh
git add src/chess_gaze/viewer_assets/index.html src/chess_gaze/viewer_assets/scene_viewer.js src/chess_gaze/viewer_assets/styles.css tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py
git commit -m "feat: add viewer hit area layer"
```

### Task 3: Docs And Real Verification

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/closeouts/2026-06-27-hit-area-viewer.md`

**Interfaces:**
- Consumes: implemented viewer and generated Nakamura short run.
- Produces: user-facing documentation and closeout evidence.

- [ ] **Step 1: Update README**

Add a short viewer note:

```markdown
The viewer also includes a `Hit Area` layer. It keeps the hit point as the point
estimate and overlays translucent angular-error patches on the monitor plane.
In `Accumulated` mode, hit-area patches accumulate like hit points but remain
controlled by the separate `Hit Area` toggle. The default typical angular error
is 8 degrees and can be adjusted from 0 to 12 degrees in the viewer. Hit-area
opacity defaults to 24% and is adjustable in the same control group. This is a
display assumption, not per-frame UniGaze confidence.
```

- [ ] **Step 2: Run focused gates**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

- [ ] **Step 3: Generate real Nakamura short run**

Prefer default model-backed analysis:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --output-root artifacts/output --models-root models
```

If native runtime fails in the sandbox, rerun unsandboxed. If model-backed
analysis is blocked by missing local assets or native MediaPipe, record the exact
error and generate a model-free run through the existing deterministic real-video
test instead.

- [ ] **Step 4: Browser smoke**

Serve the generated run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view <run-dir>
```

Use browser automation to verify:

- no console errors;
- canvas nonblank;
- `toggle-hit-area` changes rendered pixels;
- `hit-area-error-degrees` changes rendered pixels;
- turning hit points off leaves the hit-area layer independent;
- frame count and valid hit count match `viewer/scene-data.json`.

- [ ] **Step 5: Optional broad gate**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

If ignored local media or native runtime constraints block a full pass, record
the exact failures.

- [ ] **Step 6: Write closeout**

Create `docs/superpowers/closeouts/2026-06-27-hit-area-viewer.md` with:

- summary;
- proposal decisions;
- root cause or "no defect root cause";
- dependency evidence;
- test commands and results;
- Nakamura short run path and counts;
- browser smoke evidence;
- residual uncertainty.

- [ ] **Step 7: Commit**

```sh
git add README.md docs/superpowers/closeouts/2026-06-27-hit-area-viewer.md
git commit -m "docs: close out viewer hit area"
```

---

### Task 4: Accumulated Hit Area Follow-Up

**Files:**
- Modify: `tests/chess_gaze/test_scene_viewer.py`
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-27-hit-area-viewer-design.md`
- Modify: `docs/superpowers/plans/2026-06-27-hit-area-viewer.md`
- Modify: `docs/superpowers/closeouts/2026-06-27-hit-area-viewer.md`

**Interfaces:**
- Consumes: current `ViewerSceneData.frames[]` records, especially
  `frame_index`, `unigaze_ray`, `main_monitor_hit.ray_t_m`, and
  `main_monitor_hit.point_scene_m`.
- Produces: accumulated-mode hit-area patches controlled by `Hit Area`, while
  accumulated point spheres remain controlled by `Hit Points`.

- [ ] **Step 1: Write the failing source-contract test**

Add assertions to
`tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math`:

```python
assert "renderAccumulatedHitAreas" in js
assert "state.sceneData.frames.slice(0, state.frameIndex + 1)" in js
assert "addHitArea(groups.accumulated, geometry)" in js
assert "elements.toggles.hitArea.checked" in js
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math -q
```

Expected: FAIL because the current viewer has no `renderAccumulatedHitAreas`
implementation.

- [ ] **Step 3: Extend the accumulated renderer**

In `src/chess_gaze/viewer_assets/scene_viewer.js`, add:

```js
function renderAccumulatedHitAreas() {
  if (!elements.toggles.hitArea.checked || !state.sceneData) {
    return;
  }
  for (const frame of state.sceneData.frames.slice(0, state.frameIndex + 1)) {
    const geometry = hitAreaGeometry(frame, angularErrorDegrees());
    if (geometry) {
      addHitArea(groups.accumulated, geometry);
    }
  }
}
```

Then remove `!elements.toggles.hitPoints.checked` from the accumulated renderer
early return, wrap the point loop in `if (elements.toggles.hitPoints.checked)`,
and call `renderAccumulatedHitAreas()` after the point loop. Update the angular
error slider handler to call `renderAccumulatedHits()` after
`renderCurrentFrame()`.

- [ ] **Step 4: Run focused tests and syntax check**

Run:

```sh
node --check src/chess_gaze/viewer_assets/scene_viewer.js
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math -q
```

Expected: PASS.

- [ ] **Step 5: Update user-facing docs**

Change the README hit-area note to:

```markdown
The viewer also includes a `Hit Area` layer. It keeps the hit point as the point
estimate and overlays translucent angular-error patches on the monitor plane.
In `Accumulated` mode, hit-area patches accumulate like hit points but remain
controlled by the separate `Hit Area` toggle. The default typical angular error
is 8 degrees and can be adjusted from 0 to 12 degrees in the viewer. Hit-area
opacity defaults to 24% and is adjustable in the same control group. This is a
display assumption, not per-frame UniGaze confidence.
```

- [ ] **Step 6: Verify with Nakamura short**

Generate or regenerate a run from `artifacts/input/nakamura_short.mp4`, serve
the viewer, switch to accumulated mode, and verify through browser automation
that the canvas pixels change when `Hit Area` is toggled, when the angular-error
slider changes, and when `Hit Points` is toggled separately.

- [ ] **Step 7: Commit**

```sh
git add README.md docs/superpowers/specs/2026-06-27-hit-area-viewer-design.md docs/superpowers/plans/2026-06-27-hit-area-viewer.md docs/superpowers/closeouts/2026-06-27-hit-area-viewer.md tests/chess_gaze/test_scene_viewer.py src/chess_gaze/viewer_assets/scene_viewer.js
git commit -m "feat: accumulate viewer hit areas"
```

## Final Review

After all tasks, request a subagent code review over the full branch diff. Fix
Critical and Important findings before final response.
