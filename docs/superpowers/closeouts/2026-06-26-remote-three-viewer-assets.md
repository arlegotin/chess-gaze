# Remote Three.js Viewer Assets Closeout

Date: 2026-06-26

## Summary

Removed committed Three.js source files from `src/chess_gaze/viewer_assets/vendor/`
and changed generated scene viewers to load pinned Three.js `0.185.0` ESM
modules from jsDelivr when the page renders.

Behavior now shipped:

- packaged viewer assets contain only `index.html`, `scene_viewer.js`,
  `styles.css`, and `viewer_dependency_manifest.json`;
- generated `viewer/` directories no longer contain `vendor/`;
- rebuilding a viewer over an old run removes stale `viewer/vendor/` files;
- generated `index.html` embeds `scene-data.json`, an import map for Three.js,
  embedded app source, and a small bootstrap that imports the app from a Blob
  module URL;
- app JS imports `three` and
  `three/addons/controls/OrbitControls.js` through the import map;
- scene manifests persist CDN provider and pinned module URL provenance while
  remaining backward-compatible with older manifests that lack those fields.

Pinned remote module URLs:

- `https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js`
- `https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.core.js`
- `https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/`
- `https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/controls/OrbitControls.js`

Primary implementation files:

- `src/chess_gaze/viewer_dependencies.py`
- `src/chess_gaze/scene_viewer.py`
- `src/chess_gaze/scene_artifacts.py`
- `src/chess_gaze/scene_records.py`
- `src/chess_gaze/viewer_assets/`
- `tests/chess_gaze/test_scene_viewer.py`
- `tests/test_package_metadata.py`

Docs updated:

- `docs/development/decisions/0003-load-three-viewer-modules-from-pinned-cdn.md`
- `docs/development/architecture/source-layout.md`
- `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`
- `docs/superpowers/plans/2026-06-26-3d-scene-artifact-viewer.md`
- `README.md`

## Verification

Focused TDD red evidence:

- After updating tests before production code:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/test_package_metadata.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py -q`
  failed as expected with 8 remote-contract failures against the old local
  vendor implementation.

Focused green evidence:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/test_package_metadata.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py -q
```

Result: `55 passed in 1.43s`.

Static gates:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Results:

- Ruff check: `All checks passed!`
- Ruff format: `56 files already formatted`
- mypy: `Success: no issues found in 56 source files`

Full pytest:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
```

Result: `7 failed, 232 passed, 7 skipped, 18 warnings in 641.22s`.

All 7 failures are missing local mandatory real-video files:

- `artifacts/input/test_1.mp4`
- `artifacts/input/test_2.mp4`

The failing tests were:

- `tests/chess_gaze/test_pipeline_real_video_contract.py` two parametrized
  cases;
- `tests/chess_gaze/test_qa_summary_real_video_contract.py` two parametrized
  cases;
- `tests/chess_gaze/test_video_decode_real_video.py` two parametrized cases;
- `tests/chess_gaze/test_visualization_real_video.py` one case.

## Browser Smoke

Regenerated the existing `nakamura_1` run's scene/viewer layer from persisted
run artifacts:

```text
artifacts/output/nakamura_1/runs/20260626T042921Z-d0f9cfa2/viewer/index.html
```

Served it with:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view artifacts/output/nakamura_1/runs/20260626T042921Z-d0f9cfa2
```

Chrome DevTools evidence:

- no console errors, warnings, or issues;
- network requests:
  - `/` `200`;
  - `/styles.css` `200`;
  - Blob app module `200`;
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js` `200`;
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/controls/OrbitControls.js` `200`;
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.core.js` `200`;
- no `/vendor/` requests;
- status initialized to `Instant mode. Frame 1 of 1973: monitor hit is valid.`;
- hit count was `1958`;
- WebGL context existed;
- canvas was nonblank (`toDataURL` length `78546`);
- clicking next frame changed frame label from `1 / 1973` to `2 / 1973`.

Served `viewer_dependency_manifest.json` returned status `200` and module URL
metadata matching the pinned dependency contract, with no `copied_files` or
`local_patches` fields.

## Residual Risk

The viewer now requires network access to jsDelivr on first render unless the
browser already has the pinned modules cached. This is intentional per ADR-0003
and is the main behavior change from the original offline local-vendor viewer.

Project viewer code does not upload scene data, frames, crops, or model outputs.
However, remote Three.js modules execute in the same page as embedded scene data;
the no-upload privacy claim depends on trusting jsDelivr and the pinned npm
package content. A future offline/cache mode or browser-enforced integrity
control would be needed for stronger supply-chain isolation.
