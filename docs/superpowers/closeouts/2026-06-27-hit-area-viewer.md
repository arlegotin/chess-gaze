# Hit Area Viewer Closeout

Date: 2026-06-27

## Summary

Added a viewer-side `Hit Area` layer for the 3D scene viewer.

The existing monitor hit point remains the point estimate. The new layer draws a
separate translucent current-frame angular-error patch on the monitor plane. The
viewer defaults to an 8 degree typical angular error and allows live adjustment
from 5 to 12 degrees.

No scene artifact schema changed, and no new dependency was added.

## Proposal Decisions

- Selected viewer-side derived rendering instead of persisting hit-area geometry.
- Kept accumulated mode as accumulated point estimates only; the hit-area patch
  remains current-frame.
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

`scene_viewer.js` is 718 lines after this change, below the 800-line
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

Result: `25 passed in 41.44s`.

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

Full suite with loopback permission:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

Result: `348 passed, 18 warnings in 147.61s`. The warnings were existing
PyTorch `torch.jit.script` deprecation warnings from gaze-observation tests.

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

## Residual Uncertainty

The hit area is not a calibrated probability contour. It does not include
uncertainty from pseudo-metric eye depth, inferred monitor distance, face
tracking, head pose, unknown mirror policy, or streamer-domain shift except
through the user-selected angular radius.

The projection is a small-angle ellipse approximation rather than a full conic
section solve. The UI range is intentionally limited to 5 to 12 degrees to keep
that approximation in its intended regime.
