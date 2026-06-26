# Task 7 Report: Vendored Viewer Assets And Package Resources

## Scope

Implemented Task 7 only: packaged static viewer asset shells, vendored Three.js
0.185.0 files, package metadata coverage, and resource packaging verification.
Full viewer rendering behavior remains owned by Task 8.

## RED Evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_package_metadata.py -q
```

Result before assets existed:

```text
1 failed, 2 passed in 0.01s
FAILED tests/test_package_metadata.py::test_viewer_assets_are_packaged
AssertionError: assert False
missing src/chess_gaze/viewer_assets/index.html
```

## Vendoring Method

Downloaded only the approved npm tarball:

```sh
curl -L https://registry.npmjs.org/three/-/three-0.185.0.tgz -o /private/tmp/chess-gaze-three.HsOi4M/three-0.185.0.tgz
```

Network escalation was required for npm registry access. Verified tarball
integrity by computing SHA-512 base64:

```text
+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEAuj25bNAj7k1QQdf+srZywVK6w==
```

Copied only:

- `package/build/three.module.js` to `viewer_assets/vendor/three.module.js`
- `package/examples/jsm/controls/OrbitControls.js` to `viewer_assets/vendor/OrbitControls.js`
- `package/LICENSE` to `viewer_assets/vendor/THREE_LICENSE.txt`

No tarball, extraction directory, `package.json`, `node_modules`, CDN, or
frontend build tooling was added.

## Copied File SHA-256 Values

- `vendor/three.module.js`: `bbf5ed13fe4373f5bd38b14ea8e62e9f157327da5638edc6d3863e08b167c9c7`
- `vendor/OrbitControls.js`: `faabb4e8dfd9235ee4a9fd7c9a3d75f90f1689dbd4944bd6fd32117dacec5f93`
- `vendor/THREE_LICENSE.txt`: `8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc`

These are recorded in `vendor_manifest.json` and verified by
`tests/test_package_metadata.py::test_viewer_assets_are_packaged`.

## GREEN Evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_package_metadata.py -q
```

Result:

```text
3 passed in 0.01s
```

Focused Ruff:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check tests/test_package_metadata.py
```

Result:

```text
All checks passed!
```

Static remote-reference scan:

```sh
rg -n "https?://|cdn|telemetry" src/chess_gaze/viewer_assets/index.html src/chess_gaze/viewer_assets/scene_viewer.js src/chess_gaze/viewer_assets/styles.css
```

Result: no matches.

## Packaging Decision

`pyproject.toml` was not changed. A wheel built with the current Hatch config
included all packaged non-Python resources:

```text
chess_gaze/viewer_assets/index.html
chess_gaze/viewer_assets/scene_viewer.js
chess_gaze/viewer_assets/styles.css
chess_gaze/viewer_assets/vendor/OrbitControls.js
chess_gaze/viewer_assets/vendor/THREE_LICENSE.txt
chess_gaze/viewer_assets/vendor/three.module.js
chess_gaze/viewer_assets/vendor/vendor_manifest.json
```

The wheel build required network escalation for Hatchling build isolation.

## Changed Files

- `src/chess_gaze/viewer_assets/index.html`
- `src/chess_gaze/viewer_assets/scene_viewer.js`
- `src/chess_gaze/viewer_assets/styles.css`
- `src/chess_gaze/viewer_assets/vendor/three.module.js`
- `src/chess_gaze/viewer_assets/vendor/OrbitControls.js`
- `src/chess_gaze/viewer_assets/vendor/THREE_LICENSE.txt`
- `src/chess_gaze/viewer_assets/vendor/vendor_manifest.json`
- `tests/test_package_metadata.py`
- `.superpowers/sdd/task-7-report-3d-scene.md`

## Residual Risks

- Full Three.js scene rendering, playback behavior, and browser smoke checks are
  intentionally deferred to Task 8.
- The static shell loads viewer modules only after local scene data is available
  so the required missing-data fallback remains visible when `scene-data.json`
  is absent.

## Review Fixes Round 1

Reviewer findings:

- `vendor/three.module.js` imports `./three.core.js`, but `three.core.js` was
  not packaged.
- `vendor/OrbitControls.js` imported the bare specifier `three`, which cannot
  resolve without npm package resolution, import maps, or build tooling.

RED evidence after adding the regression test and before fixing vendor files:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_package_metadata.py -q
```

Result:

```text
2 failed, 2 passed in 0.05s
FAILED tests/test_package_metadata.py::test_viewer_assets_are_packaged
missing src/chess_gaze/viewer_assets/vendor/three.core.js
FAILED tests/test_package_metadata.py::test_viewer_vendor_modules_are_esm_importable
Cannot find module .../vendor/three.core.js
Cannot find package 'three' imported from .../vendor/OrbitControls.js
```

Fixes:

- Copied `package/build/three.core.js` from the same Three.js 0.185.0 npm
  tarball extraction into `src/chess_gaze/viewer_assets/vendor/three.core.js`.
- Patched only the `OrbitControls.js` import source from bare `three` to local
  `./three.module.js`.
- Updated `vendor_manifest.json` with `three.core.js` and explicit
  `local_patches` metadata for the OrbitControls import patch.

Current copied file SHA-256 values:

- `vendor/three.module.js`: `bbf5ed13fe4373f5bd38b14ea8e62e9f157327da5638edc6d3863e08b167c9c7`
- `vendor/three.core.js`: `78b2c4218834ca8670547ed2315bfc21a00ff4dc3403bbffc8c31493d31d14de`
- `vendor/OrbitControls.js`: `3f40137b0620b375637d6bce55dc830d86d79cb80b4d93aaf1b8ca6c5cb4741a`
- `vendor/THREE_LICENSE.txt`: `8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc`

Patch metadata:

```json
{
  "packaged_path": "vendor/OrbitControls.js",
  "description": "Replaced bare 'three' import with local './three.module.js' import."
}
```

Validation:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_package_metadata.py -q
```

```text
4 passed in 0.05s
```

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check tests/test_package_metadata.py
```

```text
All checks passed!
```

Optional direct Node check:

```sh
node --experimental-default-type=module --input-type=module -e "await import('./src/chess_gaze/viewer_assets/vendor/three.module.js'); await import('./src/chess_gaze/viewer_assets/vendor/OrbitControls.js');"
```

Result: exit code 0.
