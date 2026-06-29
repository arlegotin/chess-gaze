# Sphere Hit Projection Closeout

Date: 2026-06-29

## Summary

Replaced the old screen/monitor-plane gaze projection surface with a gaze sphere
centered on the robust scene center, which is the robust average eye-midpoint
position in scene space. The default persisted sphere radius is `0.700 m`, the
same plausible desktop display distance used by the previous projection model.

The generated records now persist `sphere_hit` records with ray distance,
scene-space point, sphere radius, angular coordinates, and hemisphere. The run
manifest, scene summary, viewer data, equivalence harness, and benchmark harness
use sphere-hit counts and sphere angular bounds. Active monitor-plane schema
records and invalid reasons were removed because old runs will be wiped.

The browser viewer now draws a translucent gaze sphere and projects the current
hit point, accumulated hit points, and angular hit-area patches from the saved
gaze rays onto the selected sphere radius. The viewer exposes a `Sphere Radius`
slider from `0.350 m` to `1.200 m` with a default of the persisted radius.

## Root Cause

The previous projection model treated gaze as landing on a hypothetical flat
monitor plane. That made hit points and hit areas depend on screen-plane
geometry even though the scene center and likely display distance were already
the more durable assumptions. The durable runtime surface was the scene
projection contract: artifact math, strict schemas, viewer-data generation,
equivalence comparison, and generated viewer rendering all needed to move
together.

## Artifacts Analyzed

Required real-video verification used:

- input: `artifacts/input/nakamura_short.mp4`
- fresh run: `artifacts/output/nakamura_short/runs/20260629T205811Z-adc9f4be`
- final status: complete
- decoded frames: 180
- scene frame records: 180
- valid sphere-hit frames: 180
- viewer frames: 180
- viewer valid hit points: 180
- persisted gaze-sphere radius: `0.7 m`
- old fields absent from fresh artifacts: `main_monitor_hit`, `monitor_hit`,
  `monitor_plane`, and `plane_uv_m`

Visual browser evidence:

- full page screenshot: `/private/tmp/chess-gaze-sphere-viewer-smoke.png`
- canvas-only screenshot: `/private/tmp/chess-gaze-sphere-canvas-smoke.png`
- canvas screenshot histogram: `(2100, 3082)` pixels, `2227` unique RGBA
  colors, `653146` non-background pixels
- visual inspection confirmed the sphere, hit-area patch, axes, head, eyes, and
  sphere hit marker render in the canvas.

## Third-Party Docs Checked

Verified on 2026-06-29:

- Three.js `SphereGeometry` documentation for using a unit sphere mesh scaled to
  the selected radius:
  <https://threejs.org/docs/#api/en/geometries/SphereGeometry>
- Three.js `BufferGeometry` and `BufferAttribute` documentation for cached
  accumulated hit-point and hit-area buffers:
  <https://threejs.org/docs/#api/en/core/BufferGeometry>
  <https://threejs.org/docs/#api/en/core/BufferAttribute>
- Three.js `Points` and `PointsMaterial` documentation for accumulated hit
  marker rendering:
  <https://threejs.org/docs/#api/en/objects/Points>
  <https://threejs.org/docs/#api/en/materials/PointsMaterial>
- Three.js `MeshBasicMaterial` documentation for translucent unlit sphere and
  hit-area surfaces:
  <https://threejs.org/docs/#api/en/materials/MeshBasicMaterial>

## Implementation Notes

- `scene_calibration.py` now persists `gaze_sphere_radius_m` instead of monitor
  distance/size assumptions.
- `sphere_projection.py` owns ray-sphere intersection, hit construction,
  invalid-hit construction, angle conversion, and run-level sphere geometry.
- `scene_records.py` now uses v2 frame/manifest/viewer schemas with
  `gaze_sphere`, `sphere_hit`, `valid_sphere_hit_frames`, and
  `sphere_hit_angle_bounds`.
- `scene_artifacts.py` writes sphere hits directly into
  `records/scene_frames.jsonl`, `scene/scene_manifest.json`,
  `scene/scene_summary.json`, and `viewer/scene-data.json`.
- `run_equivalence.py` compares sphere-hit angular deltas instead of monitor
  plane hit coordinates.
- `viewer_assets/scene_viewer.js` reprojects current and accumulated hits from
  gaze rays when the sphere radius slider changes, including hit-area patch
  vertices.
- Active schema leftovers for `SceneMonitorPlaneRecord`,
  `RAY_PARALLEL_TO_MONITOR`, `RAY_COPLANAR_WITH_MONITOR`,
  `MONITOR_PLANE_DEGENERATE`, and old generic ray-intersection invalid reasons
  were removed after a stale-reference scan.

## Gates

Passed:

- `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py tests/chess_gaze/test_scene_viewer.py tests/test_package_metadata.py -q`
  - 112 passed
- `UV_CACHE_DIR=.uv-cache uv run pytest -q`
  - 413 passed
  - 18 existing torch `jit.script` deprecation warnings
- `UV_CACHE_DIR=.uv-cache uv run ruff check .`
  - all checks passed
- `UV_CACHE_DIR=.uv-cache uv run ruff format --check .`
  - 71 files already formatted
- `UV_CACHE_DIR=.uv-cache uv run mypy`
  - success, 71 source files
- `UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --output-root artifacts/output --no-resume`
  - produced `artifacts/output/nakamura_short/runs/20260629T205811Z-adc9f4be`
- Fresh artifact contract script:
  - validated 180 v2 `SceneFrameRecord` records
  - validated v2 scene manifest and v2 viewer data
  - validated 180 valid sphere hits and 180 viewer hit points
  - validated no `main_monitor_hit`, `monitor_hit`, `monitor_plane`, or
    `plane_uv_m` in fresh scene/viewer artifacts
- QA summary assertion:
  - `artifact_validation.final_status == "complete"`
  - counts match
  - schema validation passed

Browser checks:

- Served viewer URL: `http://127.0.0.1:61582/`
- Expected network requests returned 200:
  - `/`
  - `styles.css`
  - `scene_viewer.js`
  - `scene-data.json`
  - pinned Three.js `0.185.0` module URLs
- Chrome DevTools console had no errors, warnings, or issues.
- DOM assertions passed:
  - frame status: `Accumulated mode. Frame 1 of 180: sphere hit is valid.`
  - hit status: `valid sphere hit`
  - accumulated status: `1 of 180`
  - sphere radius value/label: `0.7` / `0.70 m`
  - sphere radius min/max/step: `0.35` / `1.20` / `0.01`
  - gaze sphere and hit area toggles checked

## Residual Risk

The chosen default radius is still an assumption, not a measured display
distance. The viewer slider makes sensitivity visible by reprojecting hits
without rewriting artifacts, but it does not infer a true physical screen.

Playwright could not run because its managed Chromium binary was absent in this
environment. Browser verification used the available Chrome DevTools connector
instead. Direct WebGL `readPixels()` returned transparent zeros in that path,
consistent with a non-preserved drawing buffer, so final visual pixel evidence
comes from a DevTools canvas-element screenshot plus image histogram.
