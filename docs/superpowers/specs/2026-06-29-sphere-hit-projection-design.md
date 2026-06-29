# Sphere Hit Projection Design

Date: 2026-06-29

## Status

Approved design spec for replacing the current monitor-plane hit projection with
a head-centered gaze sphere projection.

This spec supersedes the monitor-plane hit portions of:

- `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`
- `docs/superpowers/specs/2026-06-27-hit-area-viewer-design.md`

Old run artifacts do not need migration or compatibility. Existing ignored runs
may be deleted before or after this change.

## Goal

Project gaze hit points and angular-error hit areas onto a sphere centered at the
robust scene center instead of onto the current hypothetical monitor plane.

The sphere represents "where this gaze ray lands at a plausible screen distance
from the user's stable head/eye center." It is still a pseudo-metric
visualization and artifact surface, not a calibrated physical monitor or
chessboard measurement.

## Findings

The current scene center is already the right sphere center. The code estimates
per-frame eye midpoints from assumed interpupillary distance, then computes a
MAD-screened geometric median as the robust scene center. In final
`scene_pseudo_m` records, that center is the origin `(0, 0, 0)`.

The user phrased the center as "average head position"; the implementation should
keep the existing robust eye-midpoint scene center rather than changing to an
arithmetic mean or to the per-frame head ellipsoid center. It is more stable,
already persisted with estimator diagnostics, and matches the existing coordinate
contract.

Visual inspection of `artifacts/input/nakamura_short.mp4` showed a streamer face
inset in the lower-left and a chessboard on the right. The visible chessboard is
not enough evidence for a calibrated physical screen surface. A head-centered
sphere is therefore a better direction-distribution surface than the current
frontoparallel plane for this viewer.

`artifacts/input/nakamura_short.mp4` was verified as:

- H.264 MP4, `1920x1080`
- `60 fps`
- `3.000s`
- `180` video frames
- SHA-256 `6364e160934c7a8de4318095172edeaf457f008f07a57f4266b2882225b5cb88`

No retained PNG/JPEG run images exist under `artifacts/output` by default. Three
representative frames were extracted to `/tmp` and visually inspected during
design exploration.

## Current Implementation Surfaces

The durable plane-specific surfaces to replace are:

- `SceneMonitorHitRecord`
- `SceneMonitorPlaneRecord` where used as a projection surface
- `SceneFrameRecord.main_monitor_hit`
- `SceneSummary.valid_monitor_hit_frames`
- `SceneSummary.invalid_monitor_hit_reasons`
- `SceneSummary.monitor_hit_bounds`
- `ViewerSceneData.valid_hit_points` values currently derived from
  `main_monitor_hit.point_scene_m`
- viewer labels and controls for monitor plane, monitor rectangle, extended
  plane, monitor hit, and plane hit areas
- run equivalence tolerance/report fields named around `monitor_uv_m`

Because old run compatibility is not required, the implementation should use
schema-versioned replacements instead of keeping misleading field names.

## Approaches Considered

### A. Viewer-only sphere rendering over old plane records

Rejected. It would be the smallest change, but persisted artifacts would still
say `main_monitor_hit`, `plane_uv_m`, and `valid_monitor_hit_frames` while the
viewer showed sphere hits. That would be brittle for future agents and tools.

### B. Parallel sphere fields beside monitor-plane fields

Rejected. This would support migration, but the user explicitly approved wiping
old runs. Keeping two hit concepts would add avoidable schema and viewer
complexity.

### C. Replace canonical hit projection with sphere projection

Selected. Build a strict sphere hit contract, update summaries/viewer data/docs,
and regenerate runs. This is the cleanest durable boundary.

## Sphere Contract

Add a run-level projection surface record, replacing the current monitor plane
as the gaze landing surface:

```json
{
  "center_scene_m": {"space": "scene_pseudo_m", "x": 0.0, "y": 0.0, "z": 0.0},
  "radius_m": 0.7,
  "radius_source": "DEFAULT_GAZE_SPHERE_RADIUS_M",
  "center_source": "robust_scene_center",
  "surface_frame": "gaze_sphere_pseudo_m"
}
```

Because old run compatibility is explicitly out of scope, persisted assumptions
should use the sphere-specific `DEFAULT_GAZE_SPHERE_RADIUS_M` name rather than
the previous monitor-distance name.

The manifest should still preserve the robust scene center in camera
coordinates, scene axes, camera model, assumptions, and robust estimator
diagnostics.

Coordinate frame additions:

- `gaze_sphere_pseudo_m`: coordinates on the gaze sphere in scene space, centered
  at the robust scene center.

## Per-frame Sphere Hit

Replace `main_monitor_hit` with `sphere_hit`.

For each valid UniGaze ray:

```text
C = sphere center in scene coordinates, normally (0, 0, 0)
O = ray origin in scene coordinates
D = normalized ray direction in scene coordinates
R = sphere radius
oc = O - C
a = dot(D, D)
b = 2 * dot(oc, D)
c = dot(oc, oc) - R^2
discriminant = b^2 - 4ac
t candidates = (-b +/- sqrt(discriminant)) / (2a)
chosen t = smallest finite t >= 0
P = O + t * D
```

Persist for valid hits:

- `point_scene_m`
- `ray_t_m`
- `radius_m`
- `theta_radians`, horizontal azimuth in scene coordinates using
  `atan2(point.x, -point.z)` so monitor-forward maps near zero
- `phi_radians`, elevation using `asin(point.y / radius)`
- `hemisphere`, one of `front`, `rear`, or `equator`; scene front is negative
  Z, so `front` means `point.z < 0`, `rear` means `point.z > 0`, and `equator`
  means near zero within a small epsilon
- `reason_invalid = null`

Persist invalid hits with explicit reasons:

- `UNIGAZE_INVALID`
- `EYE_MIDPOINT_INVALID`
- `SPHERE_RADIUS_INVALID`
- `RAY_SPHERE_DISCRIMINANT_NEGATIVE`
- `RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN`
- `RAY_SPHERE_INTERSECTION_NON_FINITE`
- `NON_FINITE_INPUT`

Selected policy: every finite valid ray may hit the sphere, including rear
hemisphere hits. The viewer is a direction surface; it must not silently discard
away-from-screen gaze unless a future user-approved calibration contract defines
a front-only surface.

## Summary Contract

Replace monitor-specific summary fields:

- `valid_monitor_hit_frames` -> `valid_sphere_hit_frames`
- `invalid_monitor_hit_reasons` -> `invalid_sphere_hit_reasons`
- `monitor_hit_bounds` -> `sphere_hit_angle_bounds`

Angle bounds:

```json
{
  "theta_min_radians": -0.42,
  "theta_max_radians": 0.38,
  "phi_min_radians": -0.21,
  "phi_max_radians": 0.19
}
```

The QA summary should continue counting scene frames and validating schemas, but
should reference sphere artifacts and sphere hit counts.

## Viewer Behavior

Replace the plane-centered viewer with a sphere-centered viewer.

Static scene:

- Transparent gaze sphere centered at `(0, 0, 0)`, default radius from the
  persisted sphere surface.
- Optional axes and grid remain useful for orientation.
- Remove or rename monitor plane, monitor rectangle, and extended plane controls.
  No UI label should claim "screen", "plane", or "monitor hit" for the new
  projection.

Controls:

- `Gaze Sphere` toggle, default checked.
- `Hit Points` toggle, default checked.
- `Hit Area` toggle, default checked.
- `Angular Error` slider, min `0`, max `12`, step `0.5`, default `8`.
- `Hit Area Opacity` slider, min `0`, max `1`, step `0.01`, default `0.24`.
- `Sphere Radius` slider, default persisted radius, min `0.35m`, max `1.20m`,
  step `0.01m`.

The radius slider is required unless implementation exposes a genuine blocker.
Runtime code should clamp to a finite positive value. If a user-selected radius
places some ray origins outside the sphere, those frames should truthfully draw
no sphere hit when the ray misses or intersects only behind the origin; the
status text should reflect the invalid reason.

Rendering:

- Current hit point renders at `sphere_hit.point_scene_m`, offset slightly along
  the sphere normal to avoid z-fighting.
- Accumulated hit points come from canonical sphere hits, not from old
  `valid_hit_points` derived from monitor hits.
- If the radius slider differs from the persisted radius, viewer hit positions
  and hit areas are recomputed from `frames[].unigaze_ray` at render time.
- UniGaze ray should draw from ray origin to the current sphere hit when valid;
  if no hit exists for the current radius, draw a warning segment.

Hit areas:

Use exact cone-to-sphere sampling instead of the old flat ellipse approximation.

For angular error `alpha` around ray direction `D`, choose orthonormal tangent
vectors `U` and `V` perpendicular to `D`. For each sampled angle `theta`:

```text
boundary_direction =
  normalize(D * cos(alpha) + (U * cos(theta) + V * sin(theta)) * sin(alpha))
boundary_point = ray_sphere_intersection(O, boundary_direction, C, R)
```

Build an indexed `THREE.BufferGeometry` triangle fan from the center hit and the
boundary points. If any boundary ray fails to intersect, omit that patch for the
active radius.

For large runs, accumulated hit points and hit areas must keep the existing
performance principle: one cached `THREE.Points` object for points, one indexed
mesh for patches, typed arrays updated in place for angular-error/radius changes,
and `setDrawRange()` for prefix visibility. Do not rebuild per-frame meshes on
every slider input.

## Library Evidence

Verified on 2026-06-29 from primary sources.

Three.js remains the existing viewer runtime dependency, pinned to `0.185.0` by
ADR-0003. Its package metadata declares module export
`./build/three.module.js`, addon export `./examples/jsm/*`, repository
`git+https://github.com/mrdoob/three.js.git`, and MIT license:
`https://cdn.jsdelivr.net/npm/three@0.185.0/package.json`.

Three.js `SphereGeometry` supports a generated sphere with fixed constructor
parameters after instantiation and clamps segment counts to minimums:
`https://cdn.jsdelivr.net/npm/three@0.185.0/src/geometries/SphereGeometry.js`.
Use `SphereGeometry(1, widthSegments, heightSegments)` and object scaling for a
live radius slider rather than reconstructing the sphere geometry on every
input.

Three.js `BufferGeometry` supports `setAttribute()`, `setIndex()`, and
`setDrawRange()`. Its source warns that `setFromPoints()` cannot grow an
existing position buffer; create correctly sized typed arrays for accumulated
geometry:
`https://cdn.jsdelivr.net/npm/three@0.185.0/src/core/BufferGeometry.js`.

Three.js `MeshBasicMaterial` is not affected by lights:
`https://cdn.jsdelivr.net/npm/three@0.185.0/src/materials/MeshBasicMaterial.js`.
Use transparent materials with `depthWrite=false` for the sphere and hit-area
overlays to reduce occlusion/z-fighting.

OpenCV camera coordinates remain `+X` right, `+Y` down, `+Z` forward per its
calibration/PnP documentation:
`https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html` and
`https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html`. Keep the repo's
existing scene axis conversion tests and do not simplify sign conventions
without regression evidence.

Pydantic schema changes must preserve strict validation and forbidden unknown
fields, matching the repository's existing `StrictSchemaModel` pattern and
Pydantic v2 strict-model guidance:
`https://pydantic.dev/docs/validation/latest/concepts/strict_mode/`,
`https://pydantic.dev/docs/validation/latest/concepts/validators/`, and
`https://pydantic.dev/docs/validation/latest/concepts/models/`.

## Source Integration

Expected files:

- `src/chess_gaze/scene_records.py`
- `src/chess_gaze/scene_geometry.py`
- `src/chess_gaze/scene_artifacts.py`
- `src/chess_gaze/scene_viewer.py`
- `src/chess_gaze/viewer_assets/index.html`
- `src/chess_gaze/viewer_assets/scene_viewer.js`
- `src/chess_gaze/viewer_assets/styles.css`
- `src/chess_gaze/qa_summary.py`
- `src/chess_gaze/run_equivalence.py`
- `README.md`
- relevant tests under `tests/chess_gaze/`
- a closeout under `docs/superpowers/closeouts/`

`scene_geometry.py`, `scene_records.py`, `scene_artifacts.py`, and
`viewer_assets/scene_viewer.js` are already over the source-layout review
threshold. The implementation must either keep edits cohesive and document the
deep-module rationale in closeout, or split a meaningful deep module if the
change pushes it toward an unclear boundary. Avoid pass-through modules.

## Testing

Use test-first development.

Required focused tests:

- sphere hit schema validates valid and invalid states;
- sphere radius rejects non-finite and non-positive values;
- ray-sphere intersection covers origin-inside, origin-outside hit, tangent,
  miss, both intersections behind origin, non-finite input, and zero direction;
- straight-ahead gaze maps near the front of the sphere with expected scene sign;
- rear-hemisphere valid rays are accepted by policy;
- scene artifacts write `sphere_hit` and no longer write `main_monitor_hit`;
- scene summary uses sphere hit count/reasons/bounds;
- viewer data derives hit points from sphere hits;
- viewer HTML exposes sphere controls and removes monitor-plane controls;
- viewer JS contains ray-sphere intersection, cone-to-sphere hit area sampling,
  radius slider handling, cached accumulated geometry, and independent hit
  point/hit area toggles;
- run equivalence compares sphere hit angles or point coordinates, not
  `monitor_uv_m`.

Required real-video tests:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_video_decode_real_video.py \
  tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Required focused gates:

```sh
node --check src/chess_gaze/viewer_assets/scene_viewer.js
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_scene_geometry.py \
  tests/chess_gaze/test_scene_records.py \
  tests/chess_gaze/test_scene_artifacts.py \
  tests/chess_gaze/test_scene_viewer.py \
  tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Run full pytest when practical:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

Required model-backed verification:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze \
  artifacts/input/nakamura_short.mp4 \
  --output-root artifacts/output \
  --models-root models
```

If native MediaPipe/Metal runtime blocks the model-backed command in the managed
sandbox, record the exact command and error, and run the broadest model-free and
non-native test subset.

Required browser verification:

- generated viewer loads through `file://` or direct file open path;
- generated viewer loads through `uv run chess-gaze view <run-dir>`;
- no console errors;
- canvas is nonblank;
- `Gaze Sphere` toggle changes pixels;
- sphere radius slider at min/default/max changes pixels;
- `Hit Area` toggle changes pixels;
- angular-error slider changes pixels;
- `Hit Points` off with `Hit Area` on still leaves patches visible in
  accumulated mode;
- final accumulated hit count matches the canonical valid sphere hit count.

## Acceptance Criteria

1. New completed runs project gaze hits onto the sphere, not onto a plane.
2. The sphere center is the robust scene center in scene coordinates.
3. The default sphere radius is the sphere-specific plausible screen-distance
   assumption, `DEFAULT_GAZE_SPHERE_RADIUS_M = 0.700m`.
4. The viewer exposes a live sphere radius slider over a plausible range unless
   an implementation blocker is recorded.
5. `records/scene_frames.jsonl` contains `sphere_hit` and no
   `main_monitor_hit`.
6. Scene summaries, QA summaries, run equivalence, README, and viewer labels use
   sphere terminology.
7. Hit areas are sampled angular cones intersected with the sphere.
8. Valid rays may produce rear-hemisphere sphere hits; this is recorded as an
   intentional direction-surface policy.
9. No old run migration or compatibility layer is required.
10. `artifacts/input/nakamura_short.mp4` is used for real verification.

## Residual Uncertainty

The sphere radius is still an assumption, not a measured display distance.
Changing the viewer slider changes visualization geometry, not model output.

The sphere does not solve camera-to-monitor calibration, mirror policy, or
chessboard localization. It gives a cleaner direction-space projection than a
flat plane, but it must not be described as a real screen measurement.
