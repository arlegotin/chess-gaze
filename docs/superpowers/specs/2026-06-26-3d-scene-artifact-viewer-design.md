# 3D Scene Artifact and Viewer Design

Date: 2026-06-26

## Status

Active design spec for extending `chess-gaze` from per-frame 2D gaze evidence to
3D scene artifacts and an interactive browser viewer.

This spec depends on the existing frame-level pipeline design:
`docs/superpowers/specs/2026-06-24-frame-gaze-analysis-pipeline-design.md`.

The implementation plan must be written separately under
`docs/superpowers/plans/` after this spec is reviewed.

## Goal

Extend `chess-gaze analyze` so every completed analysis run contains enough
machine-readable evidence to reconstruct and debug an approximate 3D scene:

- streamer head and eyes through time;
- a robust scene center based on eye-midpoint positions;
- a scene coordinate frame with explicit up, right, semantic forward, and
  transform-back axes;
- a main-monitor center placed from the dominant UniGaze direction, with a
  camera-stable plane orientation plus explicit human/setup assumptions where
  monocular data cannot determine scale or monitor pose;
- one per-frame scene record for every decoded frame;
- one monitor-plane hit point for every frame that has a finite UniGaze ray and a
  valid ray-plane intersection;
- a generated 3D viewer that renders the scene, the current frame, and either
  instant or accumulated gaze points.

The result is intentionally an evidence and visualization layer. It must not
claim calibrated real-world accuracy when the video does not contain enough data
to prove camera intrinsics, camera-to-monitor pose, or metric depth.

## Stakeholders

- Primary user: the repo owner analyzing local chess-stream clips, especially
  `artifacts/input/nakamura_1.mp4`.
- Future coding agents: implementers need precise schemas, constants, and
  failure modes so they can debug without guessing what "3D scene" means.
- Future tools: board/monitor mapping, gaze heatmaps, and manual QA need
  lossless per-frame scene records rather than only a rendered page.

## Verified Current State

Verified on 2026-06-26.

| Area | Current evidence |
| --- | --- |
| CLI | `chess-gaze analyze <video>` is built in `src/chess_gaze/cli.py:40` and calls `analyze_video()` at `src/chess_gaze/cli.py:68`. |
| Pipeline | `src/chess_gaze/pipeline.py:122` resolves config, validates input/model assets, creates a run, writes manifests, iterates decoded frames, writes `records/frames.jsonl`, and writes `qa_summary.json`. |
| Run layout | `src/chess_gaze/artifact_runs.py:14` defines raw frames, processed frames, crops, and `records/`. |
| Frame schema | `src/chess_gaze/frame_records.py:138` persists strict `FrameRecord` values: face, eyes, head pose, gaze angles, and errors. |
| Strict schemas | `src/chess_gaze/geometry.py:24` rejects unknown fields and non-finite floats at artifact boundaries. |
| Head pose gap | `src/chess_gaze/head_pose.py:69` computes rich in-memory pose evidence, but `src/chess_gaze/frame_records.py:64` persists only valid/yaw/pitch/roll/reason. |
| UniGaze gap | `src/chess_gaze/gaze_observation.py:108` computes a unit vector, but `src/chess_gaze/frame_observation.py:339` persists only yaw/pitch/reason. |
| QA summary | `src/chess_gaze/qa_summary.py:27` knows only current artifacts and count checks stop at `records/frames.jsonl`, raw frames, processed frames, and crops. |
| Frontend | No committed `package.json`, HTML viewer, JS/CSS app, dev server, Three.js, Playwright, or browser QA exists. Existing visualization is OpenCV JPEG overlays in `src/chess_gaze/visualization.py:36`. |
| Local media | `artifacts/input/nakamura_1.mp4` exists locally, is 1920x1080, has 1973 decoded frames, and has sha256 `eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45`. |
| Local models | README-listed model assets exist locally and their sha256 values match README: MediaPipe `64184e...bc9ff`; UniGaze `a336e...8c8f`. |

The current artifact contract already preserves every decoded frame. This feature
must extend that contract. It must not replace `records/frames.jsonl`, sample
frames, smooth existing per-frame model outputs, or use visualization JPEGs as
source evidence.

## Interpretation Of The User Request

The requested "average" values are robust central estimates, not arithmetic
means. The implementation must use estimators that resist tracking spikes,
wrong-face frames, transient gaze failures, and outlier poses.

The requested "each frame produces one point" is implemented as:

- every decoded frame produces exactly one `SceneFrameRecord` line;
- every frame with a valid finite UniGaze ray and a valid forward ray-plane
  intersection produces exactly one monitor hit point;
- frames without a truthful hit point keep their record and store an explicit
  invalid reason instead of fabricating, interpolating, hiding, merging, or
  copying another frame's point.

That exception is required for truthfulness. A missing face, missing eyes,
invalid UniGaze output, parallel ray, or behind-plane intersection cannot produce
an honest gaze point.

## Approaches Considered

### A. Derive 3D only from current `frames.jsonl`

Rejected as insufficient. Current records contain 2D face/eye points and
yaw/pitch angles, but they drop MediaPipe z, candidate lists, facial transform
details, head pose matrices/quaternions, PnP evidence, crop transforms, and the
already-computed UniGaze unit vector. A viewer could be made, but debugging would
be weak and future agents would not know whether failures came from model output,
projection, assumptions, or rendering.

### B. Add a scene artifact layer beside current records

Selected. Keep the existing frame contract intact and add:

- strict scene schemas;
- per-frame scene JSONL;
- scene manifest and summary JSON;
- viewer data JSON;
- generated browser viewer files.

This gives agents a stable, debuggable seam while keeping existing tests and QA
artifacts meaningful.

### C. Build a full frontend application

Rejected for this step. The repo is Python-only and has no frontend stack. The
needed viewer can be generated into each run directory from Python source
templates. A full app can be added later only if repeated viewer workflows prove
that a maintained frontend is worth the added build system.

## Output Layout

Each completed run must add these artifacts:

```text
artifacts/output/<video-stem>/runs/<run-id>/
  scene/
    scene_manifest.json
    scene_summary.json
  records/
    frames.jsonl
    errors.jsonl
    scene_frames.jsonl
  viewer/
    index.html
    scene-data.json
```

ADR-0003 supersedes the original local-vendor constraint for Three.js runtime
loading. The generated viewer must keep scene data, frames, crops, and model
artifacts local, but it may load pinned Three.js `0.185.0` ESM modules from
jsDelivr at page render time. Do not use floating CDN aliases, unpinned versions,
remote telemetry, or uploaded artifacts.

The CLI should continue printing the run directory. It may also print a second
line with the viewer path:

```text
viewer: artifacts/output/<video-stem>/runs/<run-id>/viewer/index.html
```

If the viewer needs a local HTTP server because browser `file://` rules block ES
module or JSON loading, the implementation must add:

```sh
uv run chess-gaze view <run-dir>
```

This command starts a localhost-only static server rooted at `viewer/`, prints
the URL, and never uploads artifacts.

## Coordinate Frames

All scene records must name the coordinate frame of every 3D value.

| Frame ID | Meaning |
| --- | --- |
| `image_px` | Source decoded image pixels, origin top-left, x right, y down. |
| `camera_opencv_pseudo_m` | OpenCV-style camera frame: +X image-right, +Y image-down, +Z camera-forward into the image. Units are pseudo-meters derived from assumptions unless calibration proves metric scale. |
| `scene_pseudo_m` | Scene frame centered at the robust eye-midpoint center. +X scene-right, +Y scene-up, +Z scene-back in the `right_up_back_columns_right_handed` transform basis. Semantic forward gaze toward the monitor is therefore negative scene Z. Units match `camera_opencv_pseudo_m`. |
| `monitor_plane_pseudo_m` | Main monitor plane coordinates. +U monitor-right, +V monitor-up, origin at inferred monitor center. |
| `three_view` | Browser display mapping from `scene_pseudo_m`: x stays x, y stays y-up, z is rendered with Three.js camera conventions. |

The mathematical frame is `camera_opencv_pseudo_m`. The viewer frame is only a
rendering transform. Persist the math values, not only Three.js-ready values.
Scene X must preserve camera/image horizontal ordering: a camera-right gaze
direction cannot become scene-left because the estimated dominant gaze direction
is oblique.

## Explicit Assumptions And Constants

These constants must be centralized in `calibration.py` or a new
`scene_calibration.py`, persisted in `scene_manifest.json`, and named in any
derived value that uses them.

Assume a middle-aged adult male desktop streamer only where the video cannot
determine a value.

| Constant | Value | Unit | Purpose | Uncertainty |
| --- | ---: | --- | --- | --- |
| `DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M` | 0.063 | m | Estimate face depth from 2D pupil distance. | medium |
| `DEFAULT_MONITOR_DISTANCE_FROM_EYES_M` | 0.700 | m | Place main-monitor plane center along robust UniGaze direction. | high |
| `DEFAULT_MONITOR_WIDTH_M` | 0.600 | m | Draw a typical 16:9 main monitor rectangle. | medium |
| `DEFAULT_MONITOR_HEIGHT_M` | 0.340 | m | Draw a typical 16:9 main monitor rectangle. | medium |
| `DEFAULT_EXTENDED_PLANE_SCALE` | 3.0 | multiplier | Draw monitor plane beyond physical monitor bounds. | low |
| `DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M` | 0.090 | m | Transparent head ellipsoid. | medium |
| `DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M` | 0.120 | m | Transparent head ellipsoid. | medium |
| `DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M` | 0.100 | m | Transparent head ellipsoid. | medium |
| `DEFAULT_EYE_SPHERE_RADIUS_M` | 0.012 | m | Viewer eye spheres. | medium |
| `DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M` | `[0.0, 0.035, 0.020]` | m in head-local axes | Place ellipsoid center below and slightly behind eye midpoint. | high |
| `RAY_PLANE_PARALLEL_EPSILON` | `1e-6` | unitless | Degenerate ray-plane intersection threshold. | low |
| `DEFAULT_SCENE_CENTER_CAMERA_M` | `[0.0, 0.0, 0.650]` | pseudo-m | Fallback scene center when eye-midpoint data is insufficient. | high |
| `SCENE_CENTER_MIN_AXIS_TOLERANCE_M` | 0.015 | pseudo-m | Prevent zero MAD from rejecting natural small head motion. | medium |
| `MIN_SCENE_CENTER_INLIER_FRAMES` | 5 | frames | Minimum valid frames before using data-derived center. | low |
| `MIN_MAIN_DIRECTION_INLIER_FRAMES` | 5 | frames | Minimum valid UniGaze rays before using data-derived monitor-center direction. | low |
| `DIRECTION_INLIER_ANGLE_RADIANS` | 0.35 | rad | Default angular inlier cutoff for dominant gaze direction. | medium |

If implementation changes any constant, the active spec or an ADR must explain
why and preserve the old value in the decision record.

## Scene Artifact Schemas

Use strict Pydantic models at write and read boundaries. Reject unknown fields,
non-finite floats, and inconsistent validity states.

### `scene/scene_manifest.json`

One JSON object per run:

```json
{
  "schema_version": "gaze-scene-manifest-v1",
  "run_id": "20260626T120000Z-a1b2c3d4",
  "source_video_path": "artifacts/input/nakamura_1.mp4",
  "source_video_sha256": "eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45",
  "source_artifacts": {
    "frame_records": "records/frames.jsonl",
    "scene_frame_records": "records/scene_frames.jsonl",
    "scene_summary": "scene/scene_summary.json",
    "viewer": "viewer/index.html"
  },
  "coordinate_frames": {
    "math_frame": "camera_opencv_pseudo_m",
    "scene_frame": "scene_pseudo_m",
    "monitor_frame": "monitor_plane_pseudo_m",
    "viewer_frame": "three_view"
  },
  "camera_model": {
    "policy": "estimated_pinhole_from_image_size",
    "fx_px": 1920.0,
    "fy_px": 1920.0,
    "cx_px": 960.0,
    "cy_px": 540.0,
    "metric_translation_allowed": false,
    "uncertainty": "high"
  },
  "assumptions": {
    "subject_profile": "middle_aged_adult_male_desktop_streamer",
    "interpupillary_distance_m": 0.063,
    "monitor_distance_from_eyes_m": 0.7,
    "monitor_width_m": 0.6,
    "monitor_height_m": 0.34,
    "head_ellipsoid_radii_m": [0.09, 0.12, 0.1],
    "eye_sphere_radius_m": 0.012
  },
  "robust_estimators": {
    "scene_center": {
      "method": "geometric_median_after_mad_screen",
      "candidate_frame_count": 1900,
      "inlier_frame_count": 1850,
      "fallback_used": false
    },
    "main_unigaze_direction": {
      "method": "angular_ransac_then_normalized_inlier_mean",
      "candidate_frame_count": 1800,
      "inlier_frame_count": 1550,
      "inlier_angle_radians": 0.35,
      "fallback_used": false
    },
    "scene_orientation": {
      "method": "camera_stable_right_up_back_axes",
      "candidate_frame_count": 0,
      "fallbacks": []
    }
  },
  "scene_center_camera_m": [0.0, 0.0, 0.65],
  "scene_axes_camera": {
    "right": [1.0, 0.0, 0.0],
    "up": [0.0, -1.0, 0.0],
    "back": [0.0, 0.0, -1.0],
    "forward": [0.0, 0.0, 1.0]
  },
  "main_monitor_plane": {
    "center_camera_m": [0.0, 0.0, 1.35],
    "normal_camera": [0.0, 0.0, -1.0],
    "right_camera": [1.0, 0.0, 0.0],
    "up_camera": [0.0, -1.0, 0.0],
    "physical_width_m": 0.6,
    "physical_height_m": 0.34,
    "extended_width_m": 1.8,
    "extended_height_m": 1.02,
    "distance_from_scene_center_m": 0.7,
    "distance_source": "DEFAULT_MONITOR_DISTANCE_FROM_EYES_M"
  },
  "viewer": {
    "library": "three",
    "version": "0.185.0",
    "source": "npm:three",
    "license": "MIT",
    "dist_integrity": "sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEAuj25bNAj7k1QQdf+srZywVK6w=="
  },
  "generated_at_utc": "2026-06-26T12:00:00Z"
}
```

### `records/scene_frames.jsonl`

One line per decoded frame. Count and frame indices must match
`video_manifest.frame_count_decoded` exactly.

Each line:

```json
{
  "schema_version": "gaze-scene-frame-v1",
  "frame_id": "f000000123",
  "frame_index": 123,
  "timestamp_seconds": 2.05,
  "source_frame_status": "OK",
  "valid_for_scene_center": true,
  "valid_for_main_monitor_direction": true,
  "camera": {
    "fx_px": 1920.0,
    "fy_px": 1920.0,
    "cx_px": 960.0,
    "cy_px": 540.0,
    "depth_source": "interpupillary_distance_assumption"
  },
  "left_eye": {
    "valid": true,
    "image_px": [760.0, 455.0],
    "camera_m": [-0.065, -0.010, 0.700],
    "reason_invalid": null
  },
  "right_eye": {
    "valid": true,
    "image_px": [895.0, 455.0],
    "camera_m": [-0.002, -0.010, 0.700],
    "reason_invalid": null
  },
  "eye_midpoint": {
    "valid": true,
    "origin_policy": "both_eyes_required",
    "camera_m": [-0.0335, -0.010, 0.700],
    "scene_m": [-0.010, 0.020, 0.030],
    "reason_invalid": null
  },
  "head": {
    "valid": true,
    "ellipsoid_center_camera_m": [-0.0335, 0.025, 0.720],
    "ellipsoid_radii_m": [0.09, 0.12, 0.1],
    "yaw_radians": 0.02,
    "pitch_radians": -0.04,
    "roll_radians": 0.01,
    "orientation_source": "head_pose_yaw_pitch_roll",
    "reason_invalid": null
  },
  "unigaze_ray": {
    "valid": true,
    "origin_camera_m": [-0.0335, -0.010, 0.700],
    "direction_camera": [0.01, -0.02, 0.99975],
    "direction_source": "appearance_gaze_unigaze_pitch_yaw",
    "pitch_radians": 0.02,
    "yaw_radians": 0.01,
    "reason_invalid": null
  },
  "main_monitor_hit": {
    "valid": true,
    "point_camera_m": [-0.0265, -0.024, 1.400],
    "point_scene_m": [0.006, 0.014, -0.700],
    "plane_uv_m": [0.006, 0.014],
    "within_physical_monitor": true,
    "within_extended_plane": true,
    "ray_t_m": 0.7002,
    "denominator": -0.99975,
    "signed_origin_distance_m": 0.7000,
    "reason_invalid": null
  },
  "diagnostics": {
    "warnings": [],
    "source_error_codes": []
  }
}
```

Validity invariants:

- If `left_eye.valid=true`, both `image_px` and `camera_m` are present.
- If `eye_midpoint.valid=true`, both eyes must be valid and
  `origin_policy` must be `both_eyes_required`. Single-eye origin fallback is
  out of scope for this version.
- If `unigaze_ray.valid=true`, origin and direction are finite and the direction
  has Euclidean norm within `[0.999, 1.001]`.
- If `main_monitor_hit.valid=true`, `unigaze_ray.valid=true`, all point fields
  are finite, `ray_t_m >= 0`, and the intersection is not parallel.
- Invalid nested objects must include a non-null `reason_invalid`.

### `scene/scene_summary.json`

One JSON object optimized for QA and debugging:

```json
{
  "schema_version": "gaze-scene-summary-v1",
  "run_id": "20260626T120000Z-a1b2c3d4",
  "decoded_frames": 1973,
  "scene_frame_records": 1973,
  "valid_eye_midpoint_frames": 1900,
  "valid_unigaze_ray_frames": 1800,
  "valid_monitor_hit_frames": 1785,
  "invalid_monitor_hit_reasons": {
    "UNIGAZE_INVALID": 120,
    "RAY_PARALLEL_TO_MONITOR": 3,
    "RAY_INTERSECTION_BEHIND_ORIGIN": 65
  },
  "monitor_hit_bounds": {
    "u_min_m": -0.42,
    "u_max_m": 0.38,
    "v_min_m": -0.21,
    "v_max_m": 0.19
  },
  "representative_scene_warning_frame_ids": [
    "f000000010",
    "f000001234"
  ],
  "artifact_validation": {
    "scene_frame_count_matches_decoded": true,
    "viewer_exists": true,
    "scene_manifest_valid": true,
    "scene_summary_valid": true
  }
}
```

`qa_summary.json` must reference scene artifacts and include scene record counts
in byte counts and validation. It does not need to duplicate every scene metric.

## Scene Inference Algorithms

### 1. Per-frame pseudo-metric eye positions

For frames with both eye pupil centers in `image_px`:

1. Use estimated pinhole intrinsics:
   - `fx = fy = max(frame_width, frame_height)`;
   - `cx = frame_width / 2`;
   - `cy = frame_height / 2`.
2. Compute 2D inter-pupil distance in pixels.
3. If distance is finite and greater than zero, estimate depth:
   `z = DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M * fx / pupil_distance_px`.
4. Back-project each pupil:
   - `x = (u - cx) * z / fx`;
   - `y = (v - cy) * z / fy`;
   - `z = z`.
5. Store both eyes, midpoint, source constants, and diagnostics.

This is pseudo-metric unless calibrated intrinsics and scale are later added.
Do not claim measured meters in UI copy or closeout.

### 2. Robust scene center

Candidate points are valid per-frame eye midpoints in
`camera_opencv_pseudo_m`.

Algorithm:

1. Drop non-finite candidates.
2. Compute component medians and per-component MAD values.
3. Keep candidates within
   `max(3.5 * MAD, SCENE_CENTER_MIN_AXIS_TOLERANCE_M)` on each axis.
4. Run Weiszfeld geometric median on the surviving 3D points.
5. Persist candidate count, inlier count, MAD values, iteration count, and
   convergence tolerance.
6. If fewer than `MIN_SCENE_CENTER_INLIER_FRAMES` survive, use the explicit
   fallback `DEFAULT_SCENE_CENTER_CAMERA_M = [0.0, 0.0, 0.65]` and mark
   `fallback_used=true`, `uncertainty="high"`.

The geometric median is selected because it is rotation-aware in Euclidean space
and is less sensitive to outliers than the arithmetic mean.

### 3. Per-frame UniGaze ray

Use `appearance_gaze` as the UniGaze source. Do not silently substitute
`recommended_gaze` for the main monitor ray. `recommended_gaze` may be stored as
a diagnostic overlay later, but this feature's monitor plane is based on
UniGaze.

For valid `appearance_gaze`:

1. Convert pitch/yaw to a camera direction by preserving
   `pitch_yaw_to_unit_vector()` yaw and forward-Z semantics, then negating the
   vector Y component when entering `camera_opencv_pseudo_m`. Frame-record gaze
   angles and 2D overlays use positive pitch as image-up; OpenCV camera space
   uses +Y image-down.
2. Normalize the vector and validate finite norm.
3. Set ray origin to the per-frame eye midpoint when both eyes are valid.
4. If eyes are invalid, set the ray invalid with reason
   `EYE_MIDPOINT_INVALID`; do not create a false origin.

Implementation must add regression tests that lock the yaw/pitch sign convention
against existing 2D overlay expectations.

### 4. Robust main-monitor center direction

Candidate directions are valid per-frame UniGaze unit vectors.

Algorithm:

1. Drop non-finite and non-unit directions.
2. Normalize all candidates.
3. Seed a deterministic angular RANSAC search using candidate directions sampled
   at fixed quantiles of frame order plus the coordinate-wise median direction.
4. For each seed, count inliers with angular distance <=
   `DIRECTION_INLIER_ANGLE_RADIANS`.
5. Choose the seed with maximum inliers; tie-break by lower median angular
   residual, then lower frame index of seed.
6. Compute the normalized mean of the inlier unit vectors.
7. If no candidate set has at least `MIN_MAIN_DIRECTION_INLIER_FRAMES`, use
   fallback direction `[0.0, 0.0, 1.0]` and mark high uncertainty.

The dominant ray direction estimates where the main monitor center lies relative
to the robust scene center. It is not a measured monitor surface normal and must
not rotate the scene horizontal/depth axes.

```text
monitor_center_camera =
  scene_center_camera + dominant_unigaze_direction_camera * monitor_distance_m
```

The direction estimate and its inlier diagnostics are persisted so the inferred
center remains auditable.

### 5. Scene orientation

Build axes in camera coordinates and keep them camera-stable:

1. `scene_right_camera = [1.0, 0.0, 0.0]`.
2. `scene_up_camera = [0.0, -1.0, 0.0]`.
3. `scene_back_camera = [0.0, 0.0, -1.0]`.
4. `scene_forward_camera = [0.0, 0.0, 1.0]`.
5. Validate finite axes, unit norms, pairwise dot products near zero, and
   determinant near `+1` for `[right, up, back]`.

Do not rotate scene axes from dominant gaze or eye-pair evidence. Eye labels and
eye-pair ordering remain validation evidence, but they are not a basis-estimation
input for the scene frame. This keeps camera/image right monotonic with scene
right and prevents forward depth from changing the horizontal sign.

Do not infer mirror policy unless explicit evidence exists. Persist
`mirror_policy="unknown"` when unknown.

### 6. Main-monitor plane

Place the monitor plane at:

```text
monitor_center_camera =
  scene_center_camera + dominant_unigaze_direction_camera * monitor_distance_m
```

Use `DEFAULT_MONITOR_DISTANCE_FROM_EYES_M` unless a future calibration artifact
provides measured monitor distance.

Plane basis:

- `normal_camera = scene_back_camera`;
- `up_camera = scene_up_camera`;
- `right_camera = scene_right_camera`.

This is a frontoparallel scene plane, not a calibrated physical monitor pose.
`monitor_plane_pseudo_m.u` is relative to the inferred monitor center. For
left/right ray semantics, use `unigaze_ray.direction_scene.x`; the sign of
`plane_uv_m[0]` alone can differ when the inferred center is off-axis.

Persist physical monitor dimensions and extended plane dimensions. The viewer
must draw both:

- a monitor rectangle for assumed physical screen size;
- a larger transparent plane for gaze points outside the assumed monitor bounds.

### 7. Ray-plane intersection

For each valid UniGaze ray:

```text
denom = dot(plane_normal, ray_direction)
signed_distance = dot(plane_normal, ray_origin - plane_center)
```

Invalid cases:

- `abs(denom) < RAY_PLANE_PARALLEL_EPSILON` and signed distance is not near zero:
  `RAY_PARALLEL_TO_MONITOR`;
- `abs(denom) < RAY_PLANE_PARALLEL_EPSILON` and signed distance is near zero:
  `RAY_COPLANAR_WITH_MONITOR`;
- `t = -signed_distance / denom` is non-finite:
  `RAY_INTERSECTION_NON_FINITE`;
- `t < 0`:
  `RAY_INTERSECTION_BEHIND_ORIGIN`.

Valid case:

```text
point = ray_origin + t * ray_direction
u = dot(point - plane_center, monitor_right)
v = dot(point - plane_center, monitor_up)
```

Bounds:

- `within_physical_monitor` means `abs(u) <= monitor_width / 2` and
  `abs(v) <= monitor_height / 2`, with a small epsilon.
- `within_extended_plane` means the same check against extended dimensions.

Do not clamp points to the monitor rectangle. Out-of-bounds gaze points are still
valid points and must be drawn on the extended plane.

## Viewer Requirements

The generated viewer must be a usable first screen, not a landing page.

Required controls:

- 3D viewport with mouse/touch orbit, pan, and zoom.
- Temporal slider with exact frame index labels.
- Mode switch with two labels:
  - `Instant`: shows only the current frame's head, eyes, UniGaze ray, and hit
    point.
  - `Accumulated`: shows the current frame plus all valid monitor hit points
    with `frame_index <= slider`.
- Play/pause button.
- Step previous/next buttons.
- Numeric frame input or scrub label that allows jumping to exact frame.
- Toggles for head, eyes, UniGaze ray, monitor plane, physical monitor rectangle,
  extended plane, axes, and hit points.

Required 3D objects:

- Transparent head ellipsoid for the current frame when head origin is valid.
- Two eye spheres at current-frame eye positions when valid.
- UniGaze line from current-frame eye midpoint to current-frame monitor hit when
  valid.
- If ray is valid but hit is invalid, draw the ray segment in warning color up
  to a fixed length and show reason in the frame status panel.
- Current hit point on the monitor plane when valid.
- Accumulated hit points on the monitor plane in `Accumulated` mode, one point
  per valid frame, no clustering, no heatmap substitution, no deduplication.
- Main monitor physical rectangle.
- Extended monitor plane.
- Scene axes with labels or tooltip-accessible legend.

Theme:

- Light background.
- Use restrained, meaningful colors:
  - head: translucent slate;
  - left eye: calm blue;
  - right eye: warm coral;
  - UniGaze ray: deep teal;
  - current hit: dark violet;
  - accumulated hits: muted amber with sufficient opacity;
  - monitor plane: soft neutral gray;
  - warning/invalid: muted red/orange.
- Avoid neon colors, dark-only palettes, gradient backgrounds, decorative orbs,
  and marketing-style cards.

Performance:

- For the expected `nakamura_1.mp4` size, render all valid points directly.
- Use one vertex per valid frame in a `BufferGeometry` or equivalent. This is
  allowed as a rendering optimization only if the data model still preserves one
  point per frame and point identity can be mapped back to `frame_id`.
- Do not downsample or merge points to improve performance without a separate
  user-approved spec.

Accessibility and debugging:

- The frame status panel must show frame ID, timestamp, source frame status,
  current validity state, and invalid reason if no hit point exists.
- The viewer must expose the run ID and source video stem.
- The viewer must not include explanatory marketing text or feature tutorials.
  UI labels and status values are allowed.

## Dependency And Library Decision Matrix

Verified on 2026-06-26 unless noted.

| Candidate | Fit | Source evidence | Package metadata | Decision |
| --- | --- | --- | --- | --- |
| Three.js | Strong fit for custom 3D geometry, orbit/pan/zoom, lines, points, planes, transparent ellipsoids, and browser rendering. | Official docs and source show `OrbitControls` supports orbiting, dollying, and panning around a target, and imports from bare specifier `three` in release `r185`. Source: `https://threejs.org/docs/`, `https://raw.githubusercontent.com/mrdoob/three.js/r185/examples/jsm/controls/OrbitControls.js`. | `npm view three@0.185.0`: version `0.185.0`, MIT, repo `git+https://github.com/mrdoob/three.js.git`, tarball `https://registry.npmjs.org/three/-/three-0.185.0.tgz`, integrity `sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEAuj25bNAj7k1QQdf+srZywVK6w==`. jsDelivr serves pinned `three@0.185.0` module URLs with CORS and immutable version headers. | Select for viewer. Pin version and load through the ADR-0003 import map from jsDelivr; do not vendor source files. |
| Babylon.js | Capable 3D engine but larger and more app/game-engine oriented than this viewer needs. | Official repo: `https://github.com/BabylonJS/Babylon.js`. | `npm view @babylonjs/core`: version `9.14.0`, Apache-2.0, repo `git+https://github.com/BabylonJS/Babylon.js.git`, integrity `sha512-gEXo5KF8wEu+k0bZbNLLfwW2LoIQm6d2ljdOc16p2kLapclVnHc4nZOk8q6l8N4scWD7GYM9WuX3au1namokiw==`. | Reject for first viewer. Keep as fallback only if viewer becomes an editor-like app. |
| Plotly 3D | Good for quick scientific plots, weak fit for animated head/eye/ray scene with direct object control and precise per-frame interaction. | Official repo: `https://github.com/plotly/plotly.js`. | `npm view plotly.js-dist-min`: version `3.6.0`, MIT, repo `git+https://github.com/plotly/plotly.js.git`, integrity `sha512-VR9jO2YdcEwbzVwtRyPE0eAieXFv1x5q6M9nnIgUS8FggahPrjiID6kzpnTYABwLX0gZkgEc0zxS6gQgVmgHzw==`. | Reject. It is a plotting library, not the right scene runtime. |
| Custom Canvas/WebGL | Could avoid JS dependency, but would force custom camera controls, 3D math, hit testing, and rendering edge cases. | No external source needed. | No package. | Reject. Reimplementing a 3D engine is not the quality path. |
| SciPy | Useful for `least_squares(loss="soft_l1")` and `median_abs_deviation`, but not required for this spec's selected NumPy geometric median and angular RANSAC. | Docs: `https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html`, `https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html`. PyPI JSON confirms Python 3.12 classifiers and BSD-style license text. | PyPI `scipy` current docs opened as v1.18.0. | Defer. Add only if implementation proves NumPy-only estimators are insufficient. |
| scikit-learn RANSAC | Adds broad ML dependency for a small geometry-specific consensus problem. | Maintainer docs would need review before use. | Not checked because not selected. | Reject for this spec. Implement deterministic geometry-specific RANSAC in NumPy. |
| Parquet/Arrow | Efficient for large analytics, less readable and less browser-friendly than JSONL. | Arrow/Parquet docs reviewed by research subagent. | Not selected. | Reject as primary scene artifact. JSON and JSONL remain canonical. |
| glTF/GLB | Good for static 3D assets, not for primary per-frame numeric truth or debugging diagnostics. | glTF 2.0 spec reviewed by research subagent. | Not selected. | Defer for optional export only. |

OpenCV remains part of the pipeline. Its PnP documentation states the camera
frame convention used by this spec: +X right, +Y down, +Z forward, and
`solvePnP` estimates rotation/translation from 3D object points and 2D image
projections. Source: `https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html`.

MediaPipe Face Landmarker remains a source of face/landmark/transform evidence.
Official docs state that the result can contain face landmarks, blendshapes, and
an optional facial transformation matrix. Source:
`https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python`.

NumPy remains the core numerical dependency. Official docs cover median and
percentile APIs used for robust summaries:
`https://numpy.org/doc/stable/reference/generated/numpy.median.html`,
`https://numpy.org/doc/stable/reference/generated/numpy.percentile.html`.

## Source Integration Points

Expected implementation modules:

| File | Change |
| --- | --- |
| `src/chess_gaze/scene_records.py` | New strict Pydantic schemas for scene manifest, scene frame records, scene summary, vector/plane records, invalid reason enums. |
| `src/chess_gaze/scene_geometry.py` | New 3D vector math, robust estimators, axis construction, ray-plane intersection, camera back-projection. |
| `src/chess_gaze/scene_artifacts.py` | New writer/reader that builds scene artifacts from frame records and video metadata. |
| `src/chess_gaze/scene_viewer.py` | New viewer generator and optional local static-server helper. |
| `src/chess_gaze/artifact_runs.py` | Add `scene_dir` and `viewer_dir` to `RunLayout`; create directories. |
| `src/chess_gaze/pipeline.py` | After all frame records are written and before final QA summary validation, build scene artifacts and viewer. Return paths in `AnalyzeResult`. |
| `src/chess_gaze/qa_summary.py` | Include scene artifacts in source artifact map, byte counts, and validation. |
| `src/chess_gaze/calibration.py` or `src/chess_gaze/scene_calibration.py` | Centralize assumptions and constants. |
| `src/chess_gaze/cli.py` | Optionally add `chess-gaze view <run-dir>` if local HTTP serving is required. |
| `pyproject.toml` | Include packaged viewer assets if vendored through the Python package; keep JS tooling minimal. |
| `README.md` | Update artifact list and viewer usage after implementation. |

If implementation changes `FrameRecord` to preserve richer head/gaze evidence,
it must be schema-versioned and all tests/fake records must be updated together.
If it can produce the selected scene artifacts from current `FrameRecord`
without losing required diagnostics, prefer the separate `SceneFrameRecord`
layer and leave `FrameRecord` stable.

## QA And Debuggability Requirements

The implementation must make failures inspectable from disk without opening the
viewer.

Required diagnostics:

- scene-center candidate count, inlier count, MAD thresholds, fallback state;
- main-direction candidate count, inlier count, angle threshold, angular residual
  percentiles, fallback state;
- scene-axis source and every fallback used;
- per-frame invalid reason for each missing eye, midpoint, UniGaze ray, and hit;
- per-frame ray-plane denominator, signed distance, and ray `t` when available;
- summary invalid reason counts and representative frame IDs;
- viewer dependency version, license, tarball, and integrity metadata;
- source video hash and source frame record hash or byte count.

No scene artifact may contain NaN, Infinity, or silently coerced types.

## Testing Plan

| Layer | What | Expected count |
| --- | --- | ---: |
| Unit | 3D vector schema rejects non-finite values and invalid unit vectors. | +6 |
| Unit | Back-project eye points from known image coordinates and IPD constant. | +4 |
| Unit | Geometric median resists injected outlier eye midpoints. | +4 |
| Unit | Angular RANSAC selects the dominant UniGaze direction under outliers. | +4 |
| Unit | Scene orientation handles normal cases and degenerate cross products. | +5 |
| Unit | Ray-plane intersection covers valid, parallel, coplanar, behind-origin, and non-finite cases. | +8 |
| Unit | Scene frame schema enforces validity invariants. | +8 |
| Integration | Tiny synthetic video with fake observer writes `scene_manifest.json`, `scene_summary.json`, `records/scene_frames.jsonl`, and viewer files. | +2 |
| Integration | Scene frame count equals decoded frame count and frame indices are contiguous. | +2 |
| Integration | QA summary validates scene artifacts and includes scene byte counts. | +2 |
| Integration | Viewer data contains exactly one point per valid monitor hit and no downsampled/merged point list. | +2 |
| Browser smoke | Open generated viewer through local server, assert canvas is nonblank and slider/mode switch update rendered state. | +2 |
| Real-video smoke | Run model-free deterministic pipeline on `artifacts/input/nakamura_1.mp4` after scene writer exists. | +1 |
| Real-video smoke | Run default model pipeline on `artifacts/input/nakamura_1.mp4` after scene artifacts and viewer are integrated. | +1 |

Real-model runs may need unsandboxed execution on this machine because README
documents MediaPipe native macOS GL/Metal initialization failures inside the
managed sandbox. If blocked, record the exact command, error, and environment in
the closeout.

## Required Real-Video Development Checkpoints

Do not postpone `nakamura_1.mp4` until the end.

1. After scene schemas and writer work with fake observers, run a model-free
   deterministic analysis over `artifacts/input/nakamura_1.mp4` and verify:
   - decoded frames: 1973;
   - scene frame records: 1973;
   - viewer files exist;
   - scene summary count validation passes.
2. After UniGaze ray conversion is wired, run enough real model frames to inspect
   valid/invalid reason counts. If the implementation has no frame-limit
   command, run the full video.
3. After monitor-plane inference is wired, run `nakamura_1.mp4` and inspect
   `scene_summary.json` bounds and invalid reason counts before implementing the
   viewer.
4. After viewer generation is wired, open the generated viewer and verify:
   - head/eyes/ray/hit update when the slider moves;
   - `Instant` shows only current hit;
   - `Accumulated` shows all valid hit points up to the slider;
   - no point count mismatch against `scene_summary.valid_monitor_hit_frames`.
5. Before closeout, run a full default analysis:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

Record the run directory, counts, scene summary, and viewer smoke result.

## Acceptance Criteria

1. `uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models` writes a completed run or reports an exact environment/model blocker.
2. Every completed run contains `scene/scene_manifest.json`,
   `scene/scene_summary.json`, `records/scene_frames.jsonl`, and
   `viewer/index.html`.
3. `records/scene_frames.jsonl` has exactly one valid JSON line per decoded
   frame, with contiguous `frame_index` values from zero.
4. Every finite valid UniGaze ray with a valid forward monitor-plane
   intersection produces exactly one persisted `main_monitor_hit` point.
5. Invalid frames retain scene records with explicit invalid reasons. No point is
   fabricated, interpolated, copied, clamped, clustered, hidden, merged, or
   deduplicated.
6. Scene center is a geometric median over valid eye midpoints after outlier
   screening, or an explicit persisted fallback if too few valid frames exist.
7. Main-monitor center direction is inferred from robust dominant UniGaze
   direction, or an explicit persisted fallback if too few valid rays exist.
8. Main-monitor normal is camera-stable `scene_back_camera`, and monitor right
   and up match the scene right/up axes.
9. Monitor distance, monitor size, IPD, and head dimensions are persisted as
   explicit assumptions with units and uncertainty.
10. Scene axes are finite, unit-length within tolerance, mutually orthogonal
   within tolerance, and have determinant near `+1`, or the manifest records the
   fallback axes used.
11. Ray-plane intersection diagnostics include denominator, signed distance,
    `t`, bounds booleans, and invalid reason.
12. `qa_summary.json` validates scene artifacts and includes their byte counts.
13. The generated viewer renders a light-themed 3D scene with orbit/pan/zoom,
    head ellipsoid, eye spheres, UniGaze ray, monitor plane, current hit point,
    and accumulated points mode.
13. The temporal slider supports exact frame scrubbing and mode switch between
    `Instant` and `Accumulated`.
14. Browser smoke proves the canvas is nonblank and interaction changes rendered
    state.
15. Local gates pass:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

If full gates fail because local media files other than `nakamura_1.mp4` are
absent, report the exact missing-file failures and run the broadest meaningful
subset that excludes absent-media tests.

## Out Of Scope

- Chessboard detection, board square mapping, or move parsing.
- Claiming calibrated metric accuracy without calibration evidence.
- Inferring the true physical monitor model from video.
- User-editable calibration UI.
- Heatmaps, point clustering, point aggregation, smoothing, or temporal
  interpolation.
- Replacing existing processed-frame OpenCV visualizations.
- Uploading video, frames, records, or viewer data.
- A general frontend app or hosted web service.

## Rollback Plan

The feature writes ignored run artifacts and committed source/docs. A bad release
can be rolled back by reverting the source/docs commit. Existing run directories
may keep extra `scene/`, `records/scene_frames.jsonl`, and `viewer/` artifacts;
they are ignored local outputs and can be deleted manually without affecting
source state.

The implementation must not migrate or rewrite old run directories in place.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Monocular video cannot prove real metric depth. | Label coordinates `pseudo_m`, persist assumptions, and avoid calibrated claims. |
| UniGaze direction sign may be misinterpreted. | Lock conversion tests against existing overlay convention and inspect `nakamura_1.mp4` before viewer work builds on it. |
| Wrong-face or missed-eye frames distort robust estimates. | Use finite gates, MAD screening, geometric median, angular RANSAC, and persist inlier counts. |
| Browser `file://` blocks local module/JSON loading. | Embed scene data and app source in `index.html`; use an import map for pinned remote Three.js modules. |
| Viewer silently drops points for performance. | Test exact point counts and forbid downsampling/merging. |
| Scene artifacts drift from `frames.jsonl`. | Build scene artifacts from validated frame records and require count/index equality in QA summary. |
| Additional frontend tooling bloats the Python repo. | Generate viewer from Python templates and pinned remote dependency metadata; avoid a full frontend stack in this step. |

## Closeout Requirements For Implementation

The closeout must record:

- commands run and exact pass/fail output summaries;
- `nakamura_1.mp4` run directory and decoded/scene/valid-hit counts;
- real-model blockers, if any;
- viewer smoke evidence, including browser/server path and what was verified;
- root cause for any defect found during development;
- every changed assumption constant;
- dependency/version/integrity evidence for viewer assets;
- residual uncertainty, especially metric calibration and direction-sign risk.
