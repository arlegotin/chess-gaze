# Remote Three.js Viewer Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Stop baking Three.js source files into `src/chess_gaze/viewer_assets/vendor/` and generated run viewers; load pinned Three.js modules from remote URLs when the page renders.

**Architecture:** Keep the generated viewer as Python-packaged static HTML/CSS/JS with embedded scene data for direct `file://` opening. Replace copied local Three.js modules with a pinned import map that resolves `three` and `three/addons/` to jsDelivr URLs for `three@0.185.0`. Centralize dependency URLs and metadata so scene manifests, generated HTML, tests, and docs cannot drift.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, mypy, local static HTML/CSS/JavaScript, Three.js `0.185.0` loaded from pinned jsDelivr npm URLs.

## Global Constraints

- Do not add `package.json`, `node_modules`, Playwright, or a frontend build.
- Do not use `latest`, floating CDN aliases, protocol-relative URLs, or unapproved remote hosts.
- Allow only these remote viewer module URLs:
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js`
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.core.js`
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/`
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/controls/OrbitControls.js`
- Continue embedding `viewer/scene-data.json` into generated `viewer/index.html` so direct file opening does not need local JSON fetches.
- Preserve the existing loopback-only `chess-gaze view <run-dir>` static server security boundary.
- Update canonical docs and add an ADR because this changes durable dependency-loading behavior.

---

### Task 1: Remote Dependency Contract Tests

**Files:**
- Modify: `tests/chess_gaze/test_scene_viewer.py`
- Modify: `tests/test_package_metadata.py`
- Modify: `tests/chess_gaze/test_scene_artifacts.py`
- Modify: `tests/chess_gaze/test_scene_records.py`

**Interfaces:**
- Consumes: current `build_scene_viewer()`, packaged `viewer_assets`, and `SceneManifest.viewer_dependency`.
- Produces: failing tests that define the remote dependency contract before implementation.

- [x] **Step 1: Replace vendor-copy expectations**

In `tests/chess_gaze/test_scene_viewer.py`, replace the local vendor copy test with assertions that generated viewers include `index.html`, `scene-data.json`, `scene_viewer.js`, and `styles.css`, and do not contain `viewer/vendor/`.

- [x] **Step 2: Replace local-only asset assertions**

Update the generated HTML/JS/CSS test so the only external URLs in generated viewer assets are the pinned jsDelivr Three.js URLs from Global Constraints. Assert there is no `http://`, no protocol-relative `//`, no `latest`, no `telemetry`, and no `./vendor/` import.

- [x] **Step 3: Update file-url bootstrap assertions**

Update the generated index test so it still asserts embedded scene data and `scene-viewer-source`, but asserts `three-core-source`, `three-module-source`, and `orbit-controls-source` are absent.

- [x] **Step 4: Replace Node local-vendor import tests**

Remove Node importability checks for local vendor modules. Add deterministic unit assertions for the pinned import map and package dependency manifest instead.

- [x] **Step 5: Verify red**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/test_package_metadata.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py -q
```

Expected before implementation: failures show generated viewers still copy/embed local vendor files and package metadata still expects local vendored modules.

### Task 2: Remote Loading Implementation

**Files:**
- Create: `src/chess_gaze/viewer_dependencies.py`
- Modify: `src/chess_gaze/scene_viewer.py`
- Modify: `src/chess_gaze/scene_artifacts.py`
- Modify: `src/chess_gaze/scene_records.py`
- Modify: `src/chess_gaze/viewer_assets/index.html`
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Delete: `src/chess_gaze/viewer_assets/vendor/three.module.js`
- Delete: `src/chess_gaze/viewer_assets/vendor/three.core.js`
- Delete: `src/chess_gaze/viewer_assets/vendor/OrbitControls.js`
- Delete: `src/chess_gaze/viewer_assets/vendor/THREE_LICENSE.txt`
- Delete: `src/chess_gaze/viewer_assets/vendor/vendor_manifest.json`
- Create: `src/chess_gaze/viewer_assets/viewer_dependency_manifest.json`

**Interfaces:**
- Consumes: pinned remote module constants from `viewer_dependencies.py`.
- Produces: generated viewers that embed an import map and app source, load Three.js from remote pinned URLs, and remove legacy generated `viewer/vendor/` directories.

- [x] **Step 1: Add dependency constants**

Create `viewer_dependencies.py` with constants for package name, version, npm metadata, jsDelivr module URLs, and a helper that returns the import map JSON payload.

- [x] **Step 2: Change viewer JS imports**

Change `scene_viewer.js` imports to:

```js
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
```

- [x] **Step 3: Generate import-map bootstrap**

Update `_write_file_url_compatible_index()` to embed:

- `scene-data-json`;
- an import map for `three` and `three/addons/`;
- `scene-viewer-source`;
- a dynamic import bootstrap for the embedded viewer source.

Do not read or embed local Three.js source files.

- [x] **Step 4: Remove stale generated vendor assets**

Before copying packaged viewer assets, remove `viewer_dir / "vendor"` if it exists so rebuilding a viewer over an old run cannot leave stale local Three.js scripts behind.

- [x] **Step 5: Persist remote provenance**

Extend `SceneViewerDependencyRecord` with backward-compatible `cdn_provider` and `module_urls` fields. Populate them from the centralized dependency constants.

- [x] **Step 6: Verify green**

Run the focused suite from Task 1. Expected after implementation: all focused tests pass.

### Task 3: Docs, Verification, And Review

**Files:**
- Create: `docs/development/decisions/0003-load-three-viewer-modules-from-pinned-cdn.md`
- Modify: `docs/development/architecture/source-layout.md`
- Modify: `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`
- Modify: `docs/superpowers/plans/2026-06-26-3d-scene-artifact-viewer.md`
- Modify: `README.md`
- Create: `docs/superpowers/closeouts/2026-06-26-remote-three-viewer-assets.md`

**Interfaces:**
- Consumes: verified npm/jsDelivr metadata and implementation evidence.
- Produces: updated canonical docs, an ADR, fresh test evidence, browser smoke evidence, and subagent review.

- [x] **Step 1: Update docs**

Record the decision, source evidence, selected and rejected options, runtime limitations, and verification commands.

- [x] **Step 2: Run full local gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

- [x] **Step 3: Browser smoke**

Generate or reuse a viewer, serve it with `chess-gaze view`, then verify in Chrome:

- no console errors;
- network requests include the pinned jsDelivr Three.js module URLs;
- generated viewer does not request `/vendor/...`;
- canvas has non-background pixels;
- frame controls still update status text.

- [x] **Step 4: Subagent review**

Dispatch a reviewer subagent with the diff and require spec compliance plus code quality verdicts. Fix any Critical or Important findings and rerun focused tests.
