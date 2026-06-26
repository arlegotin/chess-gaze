# Scene Horizontal Coordinate Repair Implementation Plan

> **Superseded 2026-06-26:** This plan fixed image-left/image-right ordering
> but not streamer anatomical left/right or head front/back. Do not use it as
> current coordinate guidance. Use
> `docs/superpowers/plans/2026-06-26-anatomical-scene-coordinate-repair.md`
> and the active 3D scene spec instead.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the systematic left/right confusion in 3D scene and monitor coordinates by making monitor horizontal coordinates preserve camera image-left/image-right ordering for a fixed eye origin.

**Architecture:** Keep frame-record and camera-space gaze semantics unchanged: positive frame yaw means image-right, positive pitch means image-up, and `camera_opencv_pseudo_m` uses +X image-right, +Y image-down, +Z forward. The durable boundary is the scene/monitor axis construction: dominant UniGaze direction estimates where the monitor center lies, but monocular gaze does not prove the monitor surface normal or justify rotating scene-right/depth. Scene axes and the monitor plane normal must remain camera-stable so a camera-right gaze direction cannot project to scene-left.

**Tech Stack:** Python 3.12, Pydantic records, pytest, uv, generated Three.js viewer assets, real `artifacts/input/nakamura_1.mp4`.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Work in the current branch `fixes-3`.
- Treat this as root-cause coordinate repair, not a viewer-only cosmetic patch.
- Use test-first repair: write failing regression tests before production changes.
- Do not change frame-record UniGaze semantics: positive yaw remains image-right; positive pitch remains image-up.
- Do not change OpenCV camera semantics: +X image-right, +Y image-down, +Z camera-forward.
- Do not substitute `recommended_gaze` for scene rays.
- Keep `scene_pseudo_m` right-handed with `right_up_back_columns_right_handed` and determinant near `+1`.
- Treat robust main UniGaze direction as monitor-center placement evidence, not as scene-axis or monitor-normal evidence.
- Use `artifacts/input/nakamura_1.mp4` and the reported run `artifacts/output/nakamura_1/runs/20260626T104848Z-21353a29` for real verification.
- Make meaningful commits along the way.

---

### Task 1: Lock Camera-Stable Horizontal Scene Semantics

**Files:**
- Modify: `tests/chess_gaze/test_scene_geometry.py`
- Modify: `src/chess_gaze/scene_geometry.py`

**Interfaces:**
- Consumes: `build_scene_axis_basis(...)`, `build_monitor_plane(...)`, `camera_point_to_scene(...)`, and `intersect_ray_with_monitor(...)`.
- Produces: scene X and monitor U values that preserve camera-space image-left/image-right signs for gaze directions and hit points.

- [x] **Step 1: Write failing regression tests**

Add tests in `tests/chess_gaze/test_scene_geometry.py` proving camera-right remains scene-right even with the oblique robust main direction from the reported Nakamura run:

```python
def test_scene_axis_basis_does_not_rotate_camera_right_into_scene_left() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    main_direction = _dominant_direction_estimate(
        _normalized_camera_unit_vector(0.5935890162117526, 0.19115407280621485, 0.7817366566065327)
    )
    axes = scene_geometry.build_scene_axis_basis(
        main_direction,
        [_camera_unit_vector(1.0, 0.0, 0.0)],
        assumptions,
    )

    scene_direction = scene_geometry.camera_point_to_scene(
        _camera_point(0.3711975714751555, 0.4338013084207659, 0.820992562538406),
        _camera_point(0.0, 0.0, 0.0),
        axes,
    )

    assert axes.right_camera.x > 0.99
    assert axes.up_camera.y < -0.99
    assert axes.back_camera.z < -0.99
    assert scene_direction.x > 0.0
    assert scene_direction.y < 0.0
    assert scene_direction.z < 0.0
```

Add a second regression proving the monitor center may be placed obliquely from dominant gaze, but the plane normal remains camera-stable:

```python
def test_build_monitor_plane_keeps_camera_stable_normal_for_oblique_center() -> None:
    scene_geometry = _scene_geometry()
    assumptions = default_scene_assumptions()
    scene_center = _scene_center_estimate()
    main_direction = _dominant_direction_estimate(
        _normalized_camera_unit_vector(0.5935890162117526, 0.19115407280621485, 0.7817366566065327)
    )
    axes = scene_geometry.build_scene_axis_basis(
        main_direction,
        [_camera_unit_vector(1.0, 0.0, 0.0)],
        assumptions,
    )
    monitor = scene_geometry.build_monitor_plane(
        scene_center,
        main_direction,
        axes,
        assumptions,
    )

    assert monitor.center_camera_m.x > scene_center.point_camera_m.x
    assert monitor.normal_camera.x == pytest.approx(0.0)
    assert monitor.normal_camera.y == pytest.approx(0.0)
    assert monitor.normal_camera.z == pytest.approx(-1.0)
```

- [x] **Step 2: Run tests to verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py::test_scene_axis_basis_does_not_rotate_camera_right_into_scene_left tests/chess_gaze/test_scene_geometry.py::test_build_monitor_plane_keeps_camera_stable_normal_for_oblique_center -q
```

Expected: both tests fail with the old oblique `right_camera`/`back_camera` construction because the scene basis and monitor normal use dominant gaze as the depth axis.

- [x] **Step 3: Implement the durable axis repair**

In `src/chess_gaze/scene_geometry.py`, make scene axes camera-stable:

- set `right_camera = (1.0, 0.0, 0.0)`;
- set `up_camera = (0.0, -1.0, 0.0)`;
- set `back_camera = (0.0, 0.0, -1.0)`;
- set `forward_camera = (0.0, 0.0, 1.0)` for the camera-stable scene frame;
- keep robust main UniGaze direction in `RobustDirectionEstimate` and use it only to place `monitor.center_camera_m`;
- set `monitor.normal_camera = axes.back_camera`, making the inferred monitor plane frontoparallel in the scene frame.

- [x] **Step 4: Run focused tests to verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Expected: all scene geometry tests pass.

- [x] **Step 5: Commit**

Run:

```sh
git add tests/chess_gaze/test_scene_geometry.py src/chess_gaze/scene_geometry.py docs/superpowers/plans/2026-06-26-scene-horizontal-coordinate-repair.md
git commit -m "fix: preserve scene monitor horizontal ordering"
```

### Task 2: Verify Existing Artifacts And Regenerate Nakamura Run

**Files:**
- Inspect: `artifacts/output/nakamura_1/runs/20260626T104848Z-21353a29/`
- Generate: fresh run under `artifacts/output/nakamura_1/runs/`

**Interfaces:**
- Consumes: `artifacts/input/nakamura_1.mp4`.
- Produces: fresh real-video evidence that frame 90's positive yaw maps to the right of a straight-ahead reference and that no pitch/up-down repair regressed.

- [x] **Step 1: Audit reported run numerically**

Run a Python audit that reads the reported run and records:

- frame 90 `appearance_gaze.yaw_radians`, `direction_camera.x`, `direction_scene.x`, and `plane_uv_m`;
- scene basis determinant and right/up/back vectors;
- count of valid-hit frames where camera horizontal ordering disagrees with monitor U ordering against a straight-ahead reference from the same origin.

Evidence from `artifacts/output/nakamura_1/runs/20260626T104848Z-21353a29`:

- frame 90: `direction_camera.x = +0.3711975714751555` but
  `direction_scene.x = -0.2008572172489131`;
- old axes: `right_camera = (0.7964226551329736, 0.0,
  -0.6047404024793983)`, `back_camera = (-0.5935890162117526,
  -0.19115407280621485, -0.7817366566065327)`;
- old monitor normal equaled old `back_camera`;
- straight-ahead-relative monitor-U ordering mismatches: 88 of 1958 compared
  valid rays.

- [x] **Step 2: Re-run Nakamura analysis**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

If sandboxed MediaPipe fails, rerun unsandboxed and record the exact sandbox error.

Real verification run:

```sh
MPLCONFIGDIR=/Volumes/git/legotin/chess-gaze/.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

Generated `artifacts/output/nakamura_1/runs/20260626T123553Z-a0f00fd3`
and completed cleanly after emitting known PyAV/OpenCV duplicate native class
warnings and MediaPipe telemetry upload warnings.

- [x] **Step 3: Audit fresh run**

Run the same Python audit on the fresh run. Expected:

- `scene_axes_camera.determinant_right_up_back` remains near `+1`;
- frame 90 keeps positive camera X for positive yaw;
- positive camera yaw maps to larger monitor U than straight-ahead for the same origin;
- frame 1651 still maps positive pitch to camera-up and scene-up.

Evidence from `artifacts/output/nakamura_1/runs/20260626T123553Z-a0f00fd3`:

- frame counts: 1973 `records/frames.jsonl`, 1973
  `records/scene_frames.jsonl`, 1973 viewer frames;
- axes: `right_camera = (1,0,0)`, `up_camera = (0,-1,0)`,
  `back_camera = (0,0,-1)`, `forward_camera = (0,0,1)`,
  `determinant_right_up_back = 1.0`;
- monitor normal: `(0,0,-1)`;
- frame 90: `direction_camera.x = +0.3711975714751555`,
  `direction_scene.x = +0.3711975714751555`, straight-ahead-relative
  `delta_u = +0.269131707767523`;
- frame 154: `direction_camera.x = +0.8417334780683987`,
  `direction_scene.x = +0.8417334780683987`;
- frame 1568: `direction_camera.x = -0.8151148789774965`,
  `direction_scene.x = -0.8151148789774965`;
- frame 1651: `direction_camera.y = -0.7382197511455898`,
  `direction_scene.y = +0.7382197511455898`;
- aggregate ray checks: 1973 valid rays, 0 X-sign mismatches, 0 Y-sign
  mismatches after the intentional camera-down to scene-up flip, 0 positive
  scene-Z directions, 0 viewer direction payload mismatches;
- straight-ahead-relative monitor-U ordering mismatches: 0 of 1973 compared
  valid rays.

- [x] **Step 4: Browser smoke**

Serve the fresh viewer with:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view <fresh-run-dir> --host 127.0.0.1 --port 0
```

Open it in Chrome, inspect frames 90, 154, 1568, and 1651, and save screenshots under `/private/tmp/`.

Evidence:

- served `http://127.0.0.1:57632/` for the fresh run;
- Chrome loaded 1973 frames and 1973 hit points;
- frame-control inspection confirmed frames 90, 154, 1568, and 1651 expose
  the same `unigaze_ray.direction_scene` vectors as `records/scene_frames.jsonl`;
- screenshots saved:
  `/private/tmp/chess-gaze-fresh-viewer-frame90.png`,
  `/private/tmp/chess-gaze-fresh-viewer-frame154.png`,
  `/private/tmp/chess-gaze-fresh-viewer-frame1568.png`,
  `/private/tmp/chess-gaze-fresh-viewer-frame1651.png`;
- browser console: no messages;
- network: viewer document, CSS, `scene-data.json`, and pinned Three.js modules
  returned HTTP 200.

- [ ] **Step 5: Commit closeout updates if needed**

Only generated run artifacts are ignored; commit tracked documentation and tests only.

### Task 3: Repair Documentation And Closeout

**Files:**
- Modify: `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`
- Modify: `docs/superpowers/plans/2026-06-26-3d-scene-artifact-viewer.md`
- Create: `docs/superpowers/closeouts/2026-06-26-scene-horizontal-coordinate-repair.md`

**Interfaces:**
- Consumes: root-cause evidence from Tasks 1 and 2.
- Produces: non-contradictory coordinate guidance and closeout evidence for future agents.

- [x] **Step 1: Update coordinate guidance**

Clarify that `scene_pseudo_m` horizontal ordering must preserve image-left/image-right monotonicity and that robust UniGaze direction remains a semantic forward/monitor placement vector, not permission to rotate horizontal coordinates until depth reverses left/right.

- [x] **Step 2: Write closeout**

Record:

- root cause;
- why raw UniGaze yaw/pitch and eye labels were ruled out;
- third-party coordinate docs consulted;
- durable surface changed;
- tests added;
- real Nakamura verification;
- residual uncertainty.

- [x] **Step 3: Run focused doc/source checks**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Evidence:

- `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_viewer.py -q`
  passed with `93 passed in 1.91s`;
- `UV_CACHE_DIR=.uv-cache uv run ruff check .` passed;
- `UV_CACHE_DIR=.uv-cache uv run ruff format --check .` passed with
  `56 files already formatted`;
- `UV_CACHE_DIR=.uv-cache uv run mypy` passed with
  `Success: no issues found in 56 source files`.

- [ ] **Step 4: Commit**

Run:

```sh
git add docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md docs/superpowers/plans/2026-06-26-3d-scene-artifact-viewer.md docs/superpowers/plans/2026-06-26-scene-horizontal-coordinate-repair.md docs/superpowers/closeouts/2026-06-26-scene-horizontal-coordinate-repair.md
git commit -m "docs: record scene horizontal coordinate repair"
```

### Task 4: Broad Verification And Review

**Files:**
- No source files unless review finds a defect.

**Interfaces:**
- Consumes: all repaired source/docs.
- Produces: final confidence and review evidence.

- [x] **Step 1: Run broad gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

If full pytest fails due absent legacy media fixtures, record exact failing tests and run the broad available subset that excludes only absent-media tests.

Evidence:

- `UV_CACHE_DIR=.uv-cache uv run pytest` completed in 636.46s with
  `237 passed, 7 skipped, 7 failed, 18 warnings`; every failure asserted missing
  legacy media `artifacts/input/test_1.mp4` or `artifacts/input/test_2.mp4`
  in:
  - `tests/chess_gaze/test_pipeline_real_video_contract.py` two parametrized cases;
  - `tests/chess_gaze/test_qa_summary_real_video_contract.py` two parametrized cases;
  - `tests/chess_gaze/test_video_decode_real_video.py` two parametrized cases;
  - `tests/chess_gaze/test_visualization_real_video.py`.
- Broad available subset excluding only those four absent-media failure files
  passed in 712.46s with `237 passed, 7 skipped, 18 warnings`:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
```

- `UV_CACHE_DIR=.uv-cache uv run ruff check .` passed;
- `UV_CACHE_DIR=.uv-cache uv run ruff format --check .` passed with
  `56 files already formatted`;
- `UV_CACHE_DIR=.uv-cache uv run mypy` passed with
  `Success: no issues found in 56 source files`.

- [x] **Step 2: Request final code review**

Dispatch a fresh review subagent over the branch diff, asking specifically for coordinate sign, monitor U ordering, scene axis determinant, viewer rendering, third-party contract, and documentation contradictions.

- [x] **Step 3: Fix Critical or Important findings**

Use focused tests first for any required code fix, then rerun relevant gates and update this plan or closeout if the root-cause understanding changes.

Review evidence:

- Review subagent `019f0421-eae8-7fa1-ad72-044c848756ec` found one
  Important issue: runtime scene vectors were fixed, but
  `scene_manifest.robust_estimators.scene_orientation` still advertised the old
  `eye_pair_right_and_head_up_with_camera_axis_fallbacks` estimator and
  candidate count.
- Fixed by changing `SceneOrientationEstimatorRecord.method`,
  `_build_manifest()`, and manifest tests to persist
  `method = "camera_stable_right_up_back_axes"` and
  `candidate_frame_count = 0`.
- Minor review note fixed by replacing the stale historical-plan instruction
  about degenerate right-vs-forward/up-vs-normal projection fallbacks.

Post-review verification:

- RED before fix:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_records.py::test_scene_manifest_serializes_structured_spec_fields tests/chess_gaze/test_scene_artifacts.py::test_build_scene_artifacts_writes_strict_manifest_summary_and_frames -q`
  failed because the schema rejected `camera_stable_right_up_back_axes` and the
  writer emitted the old method.
- GREEN after fix: same command passed with `2 passed in 1.97s`.
- Focused scene suite passed with `93 passed in 2.93s`.
- Rebuilt scene artifacts from the real
  `artifacts/output/nakamura_1/runs/20260626T123553Z-a0f00fd3` run produced
  orientation metadata
  `{"method": "camera_stable_right_up_back_axes", "candidate_frame_count": 0,
  "fallbacks": []}` and retained 1973 valid rays with 0 X-sign, 0 Y-sign, and
  0 positive scene-Z mismatches.
- Broad available subset passed after the final review fix in 587.41s with
  `237 passed, 7 skipped, 18 warnings`.
- Final `ruff check .`, `ruff format --check .`, and `mypy` passed.
