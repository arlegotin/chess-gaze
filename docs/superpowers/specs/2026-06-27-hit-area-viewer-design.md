# Hit Area Viewer Design

Date: 2026-06-27

## Status

Active design spec for adding a viewer-side gaze hit-area display to the
existing 3D scene viewer.

This spec extends:

- `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`
- `docs/superpowers/specs/2026-06-24-frame-gaze-analysis-pipeline-design.md`

## Goal

Keep the existing monitor hit point as the per-frame point estimate, and add a
separate translucent "Hit Area" display showing a typical angular-error cone
projected onto the inferred monitor plane.

The hit area is a visual uncertainty aid, not a replacement for the hit point,
not a per-frame confidence interval, and not a measured calibration result.

## Proposal Analysis

The proposal's core geometry is sound for this viewer's current needs:

```text
alpha = angular error radius in radians
t = main_monitor_hit.ray_t_m
minor_radius = t * tan(alpha)
major_radius = minor_radius / abs(dot(monitor_normal, gaze_direction))
```

This is a first-order projection approximation for a circular angular cone
around the gaze ray onto the monitor plane. It is most faithful for small angles;
the selected 5 to 12 degree UI range keeps the approximation in that intended
regime.

The proposal's most important correctness constraint is also sound: the point
and patch mean different things. The existing point is the model-derived ray
intersection. The patch is a derived typical angular-error visualization. The
viewer must render them independently and must never hide, merge, cluster, or
replace valid hit points.

The proposal's 5.39 degree dataset average is useful evidence, but it is not an
honest visual default for this repo's webcam/chess/streamer data. Use 8 degrees
as the default typical angular error because:

- UniGaze does not emit per-frame confidence or standard deviation;
- the monitor, camera, eye depth, and subject geometry are pseudo-metric and
  assumption-derived;
- the source domain is unconstrained streamer webcam footage, not the paper's
  evaluation datasets.

The 8 degree value is therefore an explicit viewer assumption. The UI must let
users redraw at values from 5 to 12 degrees without regenerating artifacts.

## Evidence

Verified on 2026-06-27.

| Source | Evidence | Impact |
| --- | --- | --- |
| UniGaze paper `https://arxiv.org/html/2502.02307v2` | Section 3.2 says the gaze head predicts a 2D polar-angle vector and trains with L1 loss. Table 6 reports UniGaze-H joint-dataset errors `4.46`, `5.08`, `3.20`, `5.16`, `9.07` degrees across five test sets. | Dataset averages can inform a typical display radius, but they are not per-frame uncertainty. |
| Installed `unigaze==0.1.3` package | `.venv/lib/python3.12/site-packages/unigaze/models/mae_gaze.py` defines `self.gaze_fc = nn.Linear(embed_dim, 2)` and returns `output_dict["pred_gaze"] = pred_gaze`. | The current model contract has only two output values. Do not invent confidence, variance, or std. |
| Repo wrapper | `src/chess_gaze/gaze_observation.py` requires `pred_gaze` shape `(batch, 2)` and stores `confidence=None`, `confidence_source="not_provided_by_unigaze"`. | Viewer labels must avoid confidence language. |
| Confidence-aware gaze paper `https://arxiv.org/abs/2303.10062` | The abstract distinguishes models that predict uncertainty together with gaze angles from ordinary gaze angle estimators. | Reliable per-frame uncertainty requires a model trained to output it; this repo does not have that. |
| Three.js docs/source | Current viewer already uses pinned `three@0.185.0` via ADR-0003. Official docs list `BufferGeometry`, `MeshBasicMaterial`, `Vector3`, and `OrbitControls`; r185 `OrbitControls.js` documents orbit, dolly, and pan behavior. | Use existing Three.js APIs and avoid a new browser dependency. |

No new model, checkpoint, inference library, or browser package is selected by
this task. ADR-0003 remains the binding dependency decision for Three.js.

## Approaches Considered

### A. Add a Python schema field for hit-area geometry

Rejected for this task. The viewer payload already includes all required fields
in `frames[]`: `unigaze_ray.direction_scene`, `unigaze_ray.direction_camera`,
`main_monitor_hit.point_scene_m`, `main_monitor_hit.ray_t_m`, and run-level
monitor geometry. Persisting derived patch geometry would duplicate viewer
math and require a schema change without improving artifact truth.

### B. Viewer-side current-frame patch

Selected. Compute the hit-area patch from existing scene data at render time.
This keeps the artifact contract stable, supports immediate redraw when the
angular-error slider changes, and avoids implying per-frame measured
uncertainty.

### C. Accumulate translucent patches for all frames

Selected in the 2026-06-28 follow-up. Accumulated mode should make hit-area
patches accumulated like hit points because users compare gaze history in that
mode. The display must still avoid implying calibrated probability: every patch
uses the same user-selected angular-error radius and remains a visualization of
the assumed cone footprint for that frame.

## Viewer Behavior

Add controls under `Scene Layers`:

- `Hit Area` checkbox, default checked.
- `Angular Error` range slider, min `5`, max `12`, step `0.5`, default `8`.
- A compact numeric readout such as `8 deg`.

Rendering rules:

- In `Instant` mode, draw a translucent flat patch only when `Hit Area` is
  enabled and the current frame has a valid `main_monitor_hit` and usable ray
  direction.
- Draw the hit-area patch independently from `Hit Points`. Turning hit points
  off must not force hit area off.
- Continue drawing the point estimate whenever `Hit Points` is enabled.
- In `Accumulated` mode, accumulated points still depend on `Hit Points`, and
  accumulated hit-area patches depend on `Hit Area`. Turning either layer off
  must not force the other layer off.
- Accumulated hit-area patches must be derived from `frames[]` with
  `frame_index <= current slider index`, not from `valid_hit_points[]`, because
  the summary points do not carry ray direction or `ray_t_m`.
- If the ray is invalid, the hit is invalid, `ray_t_m` is missing, or the
  direction/normal math is non-finite, draw no patch and keep the existing
  invalid reason status behavior.

Geometry rules:

1. Convert monitor normal to scene space when possible using `axis_basis`.
   Fallback to scene `+Z` only if the payload lacks a usable basis.
2. Use `unigaze_ray.direction_scene` when present. Fallback to converting
   `direction_camera` through `axis_basis`.
3. Normalize both vectors.
4. Compute:

```text
alpha_radians = angular_error_degrees * pi / 180
minor_radius_m = ray_t_m * tan(alpha_radians)
major_radius_m = minor_radius_m / abs(dot(normal_scene, direction_scene))
```

5. Orient the major axis along the gaze direction projected onto the monitor
   plane:

```text
major_axis = normalize(direction_scene - normal_scene * dot(direction_scene, normal_scene))
```

If the projected direction is too small, use the monitor right axis converted to
scene space.

6. Compute the minor axis as the normalized cross product of the monitor normal
   and the major axis.
7. Build a `THREE.BufferGeometry` triangle fan on the monitor plane. Use enough
   segments for a smooth patch without creating unnecessary vertices.
8. Use a transparent `THREE.MeshBasicMaterial` with `depthWrite=false` so the
   patch reads as a surface overlay and does not obscure the point estimate.

The approximation intentionally does not clamp to the physical monitor or the
extended plane. A large patch for an oblique ray is truthful about the selected
angular radius and viewing geometry.

## UI And Styling

The existing control panel stays dense and work-focused. Do not introduce a
landing page, cards inside cards, explanatory tutorial copy, gradients, or
decorative backgrounds.

Use one new semantic color:

- hit area: translucent magenta/rose distinct from current-hit violet and
  accumulated-hit amber.

The control layout must remain responsive at desktop and mobile widths. The
slider row must not cause text overlap.

## Source Integration

Expected files:

- `src/chess_gaze/viewer_assets/index.html`
- `src/chess_gaze/viewer_assets/scene_viewer.js`
- `src/chess_gaze/viewer_assets/styles.css`
- `tests/chess_gaze/test_scene_viewer.py`
- `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`
- `README.md`

Do not add a new Python source module. Do not expand `scene_records.py` or
`scene_geometry.py`; both are already deep modules and this feature can avoid
touching them.

## Testing

Use test-first development.

Required focused tests:

- Generated viewer HTML includes `toggle-hit-area`, `hit-area-error-degrees`,
  and `hit-area-error-label`.
- Generated viewer JavaScript includes the default/range constants, hit-area
  toggle wiring, current-frame and accumulated patch rendering, and the
  proposal's radius formula.
- Generated CSS includes a semantic hit-area color and slider row styling.
- The Nakamura short real-video model-free contract confirms `frames[]` contain
  the fields the viewer needs: valid hit point, `ray_t_m`, gaze directions, and
  monitor plane basis.

Required real verification:

- Run a model-free deterministic pipeline test over
  `artifacts/input/nakamura_short.mp4`.
- Run a default model-backed analysis over `artifacts/input/nakamura_short.mp4`
  if local models and native runtime allow it; otherwise record the exact
  blocker.
- Serve or open the generated viewer and verify through browser automation:
  canvas nonblank, no console errors, `Hit Area` toggle changes rendered pixels
  in accumulated mode, angular-error slider changes rendered pixels, and hit
  points remain controlled separately.

Required gates:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Run the full pytest suite if practical. If local ignored media files or native
MediaPipe limitations block a clean full pass, record the exact failure and run
the broadest meaningful subset.

## Acceptance Criteria

1. Existing hit points are unchanged and still render as point estimates.
2. `Hit Area` is a separate layer toggle.
3. Angular error defaults to `8` degrees and redraws live over `[5, 12]`.
4. Patch radii use `ray_t_m * tan(alpha)` and the oblique ellipse divisor
   `abs(dot(monitor_normal, gaze_direction))`.
5. Patch orientation follows the gaze projection on the monitor plane.
6. Invalid or incomplete frames draw no patch and retain existing status text.
7. No schema change is introduced.
8. No new external dependency is introduced.
9. In `Accumulated` mode, `Hit Area` renders all valid per-frame patches through
   the current slider frame independently from `Hit Points`.
10. `artifacts/input/nakamura_short.mp4` is used for real verification.
11. Closeout records commands, browser evidence, and residual uncertainty.

## Residual Uncertainty

The patch visualizes a typical angular-error radius, not a statistically
calibrated probability contour. It does not include uncertainty from eye depth,
monitor distance, face tracking, head pose, unknown mirror policy, or model
domain shift except through the user's chosen angular radius.

The ellipse formula is a first-order projection approximation. It is appropriate
for a small visual cone and a viewer-side explanatory patch. It is not a full
conic-section solver.
