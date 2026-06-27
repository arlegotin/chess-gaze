# Hit Area Viewer Closeout

Date: 2026-06-27, updated 2026-06-28

## Summary

Added a viewer-side `Hit Area` layer for the 3D scene viewer.

The existing monitor hit point remains the point estimate. The new layer draws
separate translucent angular-error patches on the monitor plane. Instant mode
draws the current-frame patch; accumulated mode now accumulates hit-area patches
like hit points while keeping `Hit Area` independent from `Hit Points`. The
viewer defaults to an 8 degree typical angular error and allows live adjustment
from 5 to 12 degrees.

No scene artifact schema changed, and no new dependency was added.

## Proposal Decisions

- Selected viewer-side derived rendering instead of persisting hit-area geometry.
- Updated accumulated mode in the 2026-06-28 follow-up so hit-area patches
  accumulate like hit points while remaining controlled by the separate
  `Hit Area` toggle.
- Used the requested small-angle approximation:
  `minor = ray_t_m * tan(alpha)` and
  `major = minor / abs(dot(monitor_normal, gaze_direction))`.
- Oriented the ellipse along the gaze direction projected onto the monitor plane.
- Treated the patch as a typical angular-error visualization, not measured
  per-frame confidence.

## Root Cause

This was feature work rather than a defect repair. The durable boundary was the
viewer layer: existing `viewer.frames[]` data already included `ray_t_m`, gaze
direction, hit point, monitor plane, and axis basis, so changing scene schemas
would have duplicated derived display data without improving truthfulness.

## Dependency Evidence

Verified on 2026-06-27:

- Installed `unigaze==0.1.3` source:
  `.venv/lib/python3.12/site-packages/unigaze/models/mae_gaze.py` defines
  `self.gaze_fc = nn.Linear(embed_dim, 2)` and returns only `pred_gaze`.
- Repo wrapper `src/chess_gaze/gaze_observation.py` requires `pred_gaze` shape
  `(batch, 2)` and stores `confidence=None` with
  `confidence_source="not_provided_by_unigaze"`.
- UniGaze paper `https://arxiv.org/html/2502.02307v2` reports dataset angular
  errors but not per-frame variance/confidence for this repo's data.
- A confidence-aware gaze paper `https://arxiv.org/abs/2303.10062` was checked
  as contrast evidence: reliable per-frame uncertainty requires a model that
  predicts it. This repo does not have that output.
- Three.js remains the existing ADR-0003 dependency: pinned `three@0.185.0`
  from jsDelivr. The implementation uses existing Three.js primitives and adds
  no package or CDN URL.

## Implementation

Changed files:

- `src/chess_gaze/viewer_assets/index.html`
- `src/chess_gaze/viewer_assets/scene_viewer.js`
- `src/chess_gaze/viewer_assets/styles.css`
- `tests/chess_gaze/test_scene_viewer.py`
- `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`
- `README.md`
- `docs/superpowers/specs/2026-06-27-hit-area-viewer-design.md`
- `docs/superpowers/plans/2026-06-27-hit-area-viewer.md`

The new viewer controls are:

- `toggle-hit-area`
- `hit-area-error-degrees`
- `hit-area-error-label`

`scene_viewer.js` is 739 lines after the accumulated follow-up, below the 800-line
source-layout review trigger.

## TDD And Review Evidence

RED command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_html_includes_required_selectors tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math tests/chess_gaze/test_scene_artifacts_real_video_contract.py::test_model_free_nakamura_video_scene_artifact_contract -q
```

Result: `2 failed, 1 passed in 41.85s`. Failures were the intended missing
`toggle-hit-area` selector and missing `Hit Area` text.

Focused viewer source checks after implementation:

```sh
node --check src/chess_gaze/viewer_assets/scene_viewer.js
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_html_includes_required_selectors tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math -q
```

Results:

- `node --check`: exit `0`
- `2 passed in 1.34s`

Task review:

- Subagent review of `d61d6d8..d1a6168`: spec compliant, task quality approved.
- Findings: no Critical, no Important. Minor residual risk: source-string tests
  cannot prove all geometry behavior without browser smoke.

Final branch review before completion found one Important issue:

- `hitAreaGeometry()` could render a patch from an inconsistent future payload
  where `main_monitor_hit.valid=true` but `unigaze_ray.valid=false` and stale
  direction fields were still present.

Fix applied after final review:

- added an explicit `frame?.unigaze_ray?.valid` guard before hit-area geometry;
- changed degenerate projected-direction handling so monitor-right is the first
  fallback axis before an arbitrary safe axis;
- added source-level regression assertions for the invalid-ray guard and
  projected-direction fallback.

Focused regression after the fix:

```sh
node --check src/chess_gaze/viewer_assets/scene_viewer.js
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math -q
```

Results:

- `node --check`: exit `0`
- `1 passed in 1.23s`

Accumulated follow-up RED command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math -q
```

Result: `1 failed in 1.33s`. The failure was the intended missing
`renderAccumulatedHitAreas` implementation.

Accumulated follow-up focused checks after implementation:

```sh
node --check src/chess_gaze/viewer_assets/scene_viewer.js
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_generated_viewer_exposes_hit_area_controls_and_math -q
```

Results:

- `node --check`: exit `0`
- `1 passed in 1.08s`

## Verification Commands

Focused suite in sandbox:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Result: `23 passed, 2 failed` because sandboxed socket bind raised
`PermissionError: [Errno 1] Operation not permitted` in pre-existing loopback
static-server tests.

Focused suite with loopback permission:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Result before final-review fix: `25 passed in 41.44s`.

Result after final-review fix: `25 passed in 43.74s`.

Accumulated follow-up viewer suite with loopback permission:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

Result: `24 passed in 1.48s`.

Accumulated follow-up real-video contract:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Result: `1 passed in 39.44s`.

Static gates:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Results:

- `All checks passed!`
- `65 files already formatted`
- `Success: no issues found in 65 source files`

These static gates were rerun after the final-review fix with the same passing
results.

They were rerun after the accumulated follow-up with the same passing results:

- `All checks passed!`
- `65 files already formatted`
- `Success: no issues found in 65 source files`

Full suite with loopback permission:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

Result: `348 passed, 18 warnings in 147.61s`. The warnings were existing
PyTorch `torch.jit.script` deprecation warnings from gaze-observation tests.

Accumulated follow-up full-suite result:
`348 passed, 18 warnings in 144.56s`. The warnings were the same existing
PyTorch `torch.jit.script` deprecation warnings.

## Nakamura Short Real Run

Default model-backed verification command:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --output-root artifacts/output --models-root models
```

Result:

```text
artifacts/output/nakamura_short/runs/20260627T221641Z-b028e67a
viewer: artifacts/output/nakamura_short/runs/20260627T221641Z-b028e67a/viewer/index.html
```

Runtime notes:

- MediaPipe initialized successfully with native GL/Metal when run unsandboxed.
- The command printed duplicate AVFoundation class warnings from `cv2` and `av`
  bundled FFmpeg libraries; the run completed successfully.

Artifact audit:

- `qa_summary.final_status`: `complete`
- decoded frames: `180`
- scene frame records: `180`
- viewer frame count: `180`
- valid monitor hit frames: `180`
- valid hit points: `180`
- invalid monitor hit reasons: `{}`
- sample frame `0`:
  - `ray_t_m`: `0.7138479244522792`
  - `direction_scene`: `x=0.1593538519029787`,
    `y=-0.232774178266643`, `z=-0.9593865393135202`
  - `hit.point_scene_m`: `x=0.17687001767775795`,
    `y=-0.16475496415599195`, `z=-0.5692533823694435`

Accumulated follow-up regenerated run:

```text
artifacts/output/nakamura_short/runs/20260627T224815Z-d873457e
viewer: artifacts/output/nakamura_short/runs/20260627T224815Z-d873457e/viewer/index.html
```

Artifact audit:

- `qa_summary.final_status`: `complete`
- viewer frame count: `180`
- valid monitor hit frames: `180`
- valid hit points: `180`
- first valid `ray_t_m`: `0.7138479244522792`
- last frame index: `179`
- generated viewer asset includes `renderAccumulatedHitAreas`,
  `state.sceneData.frames.slice(0, state.frameIndex + 1)`, and
  `addHitArea(groups.accumulated, geometry)`.

## Browser Smoke

Served the fresh run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view artifacts/output/nakamura_short/runs/20260627T221641Z-b028e67a
```

URL: `http://127.0.0.1:63988/`

Chrome DevTools verification:

- page title: `Chess Gaze Scene Viewer`
- status: `Instant mode. Frame 1 of 180: monitor hit is valid.`
- frame label: `1 / 180`
- hit count: `180`
- `Hit Area` checkbox present and checked
- angular-error slider present with value `8`, then changed to `12`
- no console messages
- network requests limited to:
  - local document `/`
  - local `styles.css`
  - blob app module
  - pinned Three.js `0.185.0` module URLs:
    `three.module.js`, `OrbitControls.js`, `three.core.js`

Canvas pixel evidence:

| State | PNG data URL length | Sample hash |
| --- | ---: | ---: |
| initial, 8 deg, hit area on, hit points on | `233982` | `3157419193` |
| hit area off | `226802` | `930078944` |
| hit area on, 12 deg | `237790` | `3967822218` |
| hit area on, 12 deg, hit points off | `230054` | `3525652923` |

These checks prove:

- the canvas was nonblank;
- the `Hit Area` toggle changed rendered pixels;
- the angular-error slider changed rendered pixels;
- turning `Hit Points` off changed pixels separately while leaving `Hit Area`
  enabled.

Screenshot captured at:

`/private/tmp/chess-gaze-hit-area-viewer.png`

Accumulated follow-up browser smoke used the regenerated run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view artifacts/output/nakamura_short/runs/20260627T224815Z-d873457e
```

URL: `http://127.0.0.1:64866/`

Chrome DevTools verification:

- status: `Accumulated mode. Frame 180 of 180: monitor hit is valid.`
- accumulated count: `180 of 180`
- no console messages
- network requests limited to local viewer files plus pinned Three.js `0.185.0`
  module URLs

Accumulated canvas pixel evidence:

| State | PNG data URL length | Sample hash |
| --- | ---: | ---: |
| accumulated, frame 180, 8 deg, hit area on, hit points on | `741122` | `536949490` |
| accumulated, hit area off | `493274` | `488543973` |
| accumulated, hit area on, 12 deg | `753306` | `768258110` |
| accumulated, hit area on, 12 deg, hit points off | `504494` | `2717483482` |
| accumulated, hit area off, hit points off | `230882` | `1325951166` |

These checks prove:

- accumulated-mode `Hit Area` changed rendered pixels;
- the angular-error slider changed accumulated patch pixels;
- turning `Hit Points` off changed pixels separately;
- hit-area patches remained visible when `Hit Points` was off.

Screenshot captured at:

`/private/tmp/chess-gaze-accumulated-hit-area.png`

## Residual Uncertainty

The hit area is not a calibrated probability contour. It does not include
uncertainty from pseudo-metric eye depth, inferred monitor distance, face
tracking, head pose, unknown mirror policy, or streamer-domain shift except
through the user-selected angular radius.

The projection is a small-angle ellipse approximation rather than a full conic
section solve. The UI range is intentionally limited to 5 to 12 degrees to keep
that approximation in its intended regime.
