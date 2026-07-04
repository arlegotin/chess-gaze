# Visualization Redo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redo the generated 3D scene viewer visualization so it removes hit points, keeps hit areas, uses lower defaults, opens on the last frame, and removes per-frame status sentences.

**Architecture:** Keep the existing static viewer package and scene-data schema. Change only the generated viewer UI/runtime and its direct documentation/tests; preserve cached accumulated hit-area geometry and the file-url/served viewer split.

**Tech Stack:** Python 3.12, pytest, generated static HTML/CSS/JS viewer assets, Three.js `0.185.0` from the existing ADR-0003 import map.

## Global Constraints

- Work directly on `main`, as approved by the user on 2026-07-04.
- Make meaningful commits along the way.
- Use test-first development.
- Do not change persisted scene/viewer schemas.
- Do not add a frontend build system or new runtime/browser dependency.
- Keep Three.js `0.185.0` loaded from pinned jsDelivr URLs per ADR-0003.
- Keep direct `file://` viewing and localhost served viewing available.
- Keep `Hit Area` as the only gaze-hit visualization layer.
- Hit-area angular-error slider min/default is `0.5`, max `12`, step `0.5`.
- Hit-area opacity default is `0.04`, with a `4%` readout.
- Successful scene loading initializes to the last frame; empty data stays at `0`.
- Loading and error statuses remain visible; successful per-frame status sentences are removed from both status surfaces.

---

### Task 1: Lock The Viewer Contract In Tests

**Files:**

- Modify: `tests/chess_gaze/test_scene_viewer.py`

**Interfaces:**

- Consumes: generated `index.html`, `scene_viewer.js`, `styles.css`.
- Produces: failing tests that describe the requested viewer behavior.

- [ ] Update generated-selector tests so `toggle-hit-points` is absent and `toggle-hit-area`, angular-error, opacity, and status surfaces remain covered.
- [ ] Update hit-area control assertions for `min="0.5"`, `value="0.5"`, label `0.5 deg`, and opacity `value="0.04"` / `4%`.
- [ ] Add JS source assertions that hit-point query/render/toggle paths are gone.
- [ ] Add source assertions that `applySceneData()` initializes with `setFrameIndex(maxIndex)` and does not call `setFrameIndex(0)`.
- [ ] Add JS extraction/probe coverage for empty scene data and status-surface behavior.
- [ ] Run the focused test selection and verify it fails for the expected old viewer behavior.

### Task 2: Implement Viewer Runtime And Template Changes

**Files:**

- Modify: `src/chess_gaze/viewer_assets/index.html`
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Modify: `src/chess_gaze/viewer_assets/styles.css`
- Modify: `src/chess_gaze/scene_viewer.py`

**Interfaces:**

- Consumes: tests from Task 1.
- Produces: generated viewer assets satisfying the new contract.

- [ ] Remove the `Hit Points` checkbox from `index.html`.
- [ ] Change angular-error and opacity defaults in HTML and JS constants.
- [ ] Remove current-hit material, accumulated hit-point cache fields, hit-point builders/updaters/visibility, and current-frame hit-point sphere rendering from JS.
- [ ] Keep hit-count summary derived from valid sphere hits.
- [ ] Change `applySceneData()` to initialize to the computed last index.
- [ ] Split status handling so successful scene load clears/hides status surfaces while loading/error messages still write to both.
- [ ] Update CSS so the fallback overlay can be hidden when ready and remove no-longer-needed hit-point styling only if unused.
- [ ] Update the file-url bootstrap error text path in `scene_viewer.py` only if selectors or status semantics require it.
- [ ] Run focused tests and iterate until green.

### Task 3: Update Docs, Review, And Verify

**Files:**

- Modify: `README.md`
- Create: `docs/superpowers/closeouts/2026-07-04-visualization-redo.md`

**Interfaces:**

- Consumes: implemented viewer behavior and test evidence.
- Produces: current user-facing docs and closeout evidence.

- [ ] Update README viewer text to remove hit-point visualization claims and document the new angular-error/opacity defaults.
- [ ] Run focused viewer tests with loopback permission.
- [ ] Run broad non-real-video pytest subset.
- [ ] Run ruff check, ruff format check, and mypy.
- [ ] Perform browser smoke if local serving/browser tools are available; otherwise record the exact blocker.
- [ ] Run subagent review on the final diff and address important findings.
- [ ] Write closeout with root cause, durable surface changed, tests, browser evidence/blockers, and residual risk.
- [ ] Commit the final implementation and documentation changes.

## Self-Review

- Spec coverage: all five user-requested changes are covered by Tasks 1 and 2; docs, closeout, and verification are covered by Task 3.
- Placeholder scan: no `TBD`/`TODO` placeholders are present.
- Type/name consistency: existing selectors are preserved except the intentionally removed `toggle-hit-points`; no schema types change.
