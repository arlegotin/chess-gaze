# Anatomical Scene Coordinate Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the 3D scene so persisted scene axes, gaze rays, monitor hits, and head placement match the streamer's anatomical left/right and front/back for frontal desktop webcam videos.

**Architecture:** Keep raw frame-record yaw/pitch semantics unchanged: positive yaw still means image-right in processed overlays and positive pitch still means image-up. Convert those angles at the scene boundary into a physical frontal-webcam camera ray, then project into a right-handed human-centered scene basis: +X streamer right, +Y up, +Z back, with monitor-directed gaze in negative scene Z. The viewer should render persisted scene data directly but label and default-view that data according to the human-centered contract.

**Tech Stack:** Python 3.12, Pydantic scene records, pytest, uv, generated Three.js viewer assets, real `artifacts/input/nakamura_1.mp4`.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Work in the current branch `fixes-3`.
- Treat this as root-cause coordinate repair, not a viewer-only cosmetic patch.
- Use test-first repair: write failing regression tests before production changes.
- Do not change frame-record UniGaze overlay semantics: positive yaw remains image-right; positive pitch remains image-up.
- Do not change OpenCV back-projection semantics: image pixels use +X right, +Y down, +Z into the image.
- Persist a right-handed `right_up_back_columns_right_handed` scene basis.
- Use `artifacts/input/nakamura_1.mp4` and `artifacts/output/nakamura_1/runs/20260626T123553Z-a0f00fd3` for real verification.
- Make meaningful commits along the way.

---

### Task 1: Lock Anatomical Scene Semantics

**Files:**
- Modify: `tests/chess_gaze/test_scene_geometry.py`
- Modify: `tests/chess_gaze/test_scene_artifacts.py`
- Modify: `tests/chess_gaze/test_scene_records.py`
- Modify: `src/chess_gaze/scene_geometry.py`
- Modify: `src/chess_gaze/scene_artifacts.py`
- Modify: `src/chess_gaze/scene_records.py`

**Interfaces:**
- Consumes: `GazeAngles` where positive yaw means image-right and positive pitch means image-up.
- Produces: `SceneAxisBasisRecord` with +X streamer right, +Y scene up, +Z scene back; `SceneUniGazeRayRecord.direction_scene` where image-right/his-left is negative scene X and monitor-directed gaze is negative scene Z.

- [x] **Step 1: Add failing tests**

Add tests proving:

- `build_scene_axis_basis()` returns `right_camera=(-1,0,0)`, `up_camera=(0,-1,0)`, `back_camera=(0,0,1)`, `forward_camera=(0,0,-1)`.
- positive frame yaw maps to camera +X but scene negative X, while positive pitch maps to scene positive Y.
- neutral scene gaze has negative scene Z, not positive/backward Z.
- monitor plane and hit points are in front of the face at negative scene Z.
- the head ellipsoid center is below and behind the eye midpoint, so eyes are on the front side of the head.
- `source_artifacts.viewer` points to `viewer/index.html`.

- [x] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_scene_geometry.py \
  tests/chess_gaze/test_scene_artifacts.py::test_scene_frame_direction_maps_positive_pitch_to_scene_up \
  tests/chess_gaze/test_scene_records.py::test_scene_manifest_serializes_structured_spec_fields -q
```

Expected: new anatomical tests fail under the current image-side scene basis.

- [x] **Step 3: Implement the durable boundary fix**

Change the scene conversion boundary:

- `_frame_gaze_angles_to_camera_direction()` returns `(x, -image_up_y, -z)` so UniGaze yaw/pitch become a physical eye-to-monitor ray for a frontal webcam setup.
- `build_scene_axis_basis()` returns `right_camera=(-1,0,0)`, `up_camera=(0,-1,0)`, `back_camera=(0,0,1)`, `forward_camera=(0,0,-1)`.
- `build_monitor_plane()` keeps monitor normal equal to `axes.back_camera` and uses the corrected physical gaze direction for center placement.
- `_head_record()` continues applying the camera-space offset `(0,+Y,+Z)`, which now maps to scene `(0,-Y,+Z)`: below and behind the eye midpoint.
- `_build_manifest()` writes `viewer="viewer/index.html"` and an orientation method name that explicitly says anatomical frontal-webcam axes.

- [x] **Step 4: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_viewer.py -q
```

Expected: focused scene and viewer tests pass.

- [x] **Step 5: Commit**

Commit the test and implementation repair with a message describing the anatomical scene-coordinate fix.

### Task 2: Repair Viewer Orientation And Labels

**Files:**
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Modify: `src/chess_gaze/viewer_assets/index.html`
- Modify: `tests/chess_gaze/test_scene_viewer.py`

**Interfaces:**
- Consumes: persisted `scene_pseudo_m` directly.
- Produces: a default front-of-face viewer camera and axis labels that do not imply blue +Z is gaze direction.

- [x] **Step 1: Add tests for generated viewer assets**

Assert the generated viewer labels +X as streamer right and +Z as streamer back, and the default camera starts on the front/monitor side of the scene.

- [x] **Step 2: Implement viewer changes**

Set the initial Three camera on negative scene Z looking toward the head and monitor, update the key light accordingly, and change the axis legend from generic depth to streamer back/monitor-front language.

- [x] **Step 3: Verify and commit**

Run `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q`, then commit.

### Task 3: Repair Docs And Closeout

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`
- Modify: `docs/superpowers/plans/2026-06-26-scene-horizontal-coordinate-repair.md`
- Modify or create: `docs/superpowers/closeouts/2026-06-26-anatomical-scene-coordinate-repair.md`

**Interfaces:**
- Consumes: root-cause evidence from artifact, code, visual, and third-party documentation analysis.
- Produces: non-contradictory guidance for future agents and users.

- [x] **Step 1: Replace stale image-side scene guidance**

Update docs so `scene_pseudo_m` says +X is streamer/anatomical right for frontal webcam assumptions, +Y is up, +Z is streamer back, and monitor-directed gaze is negative scene Z.

- [x] **Step 2: Record root cause and verification**

Write a closeout with the failed prior invariant, exact frame evidence, third-party docs checked, tests run, real Nakamura run ID, and residual uncertainty about mirrored webcam feeds.

- [x] **Step 3: Commit docs**

Commit documentation updates separately from code.

### Task 4: Real Nakamura Verification

**Files:**
- Generate ignored artifacts under `artifacts/output/nakamura_1/runs/`
- Inspect: `artifacts/input/nakamura_1.mp4`

**Interfaces:**
- Consumes: real input video and local model assets.
- Produces: a fresh verified run with anatomical scene-coordinate evidence.

- [x] **Step 1: Rebuild scene artifacts from the existing real run**

Use `build_scene_artifacts()` and `build_scene_viewer()` on the reported run to quickly verify corrected scene data against identical source `frames.jsonl`.

- [x] **Step 2: Run full real-video analysis**

Run:

```sh
MPLCONFIGDIR=/Volumes/git/legotin/chess-gaze/.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

- [x] **Step 3: Numeric audit**

Verify representative frames:

- frame 90 and 154: image-right/his-left gives `direction_scene.x < 0`.
- frame 1568 and 1651: image-left/his-right gives `direction_scene.x > 0`.
- all valid rays point toward negative scene Z.
- eye midpoint is in front of head center: `eye_midpoint.scene_m.z < head.scene_m.z`.
- monitor center and hit points are in front of the face: negative scene Z.

- [x] **Step 4: Browser visual verification**

Serve the fresh viewer, capture screenshots for frames 90, 154, 1568, and 1651, and verify the default view, ray, monitor, head, and eyes agree with the processed frames.

- [x] **Step 5: Final gates**

Run focused tests, broad available tests, Ruff, and mypy or report exact remaining failures.
