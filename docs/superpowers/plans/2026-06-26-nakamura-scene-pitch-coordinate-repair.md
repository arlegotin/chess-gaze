# Nakamura Scene Pitch Coordinate Repair Implementation Plan

> **Superseded 2026-06-26:** This plan repaired pitch/up-down semantics only.
> Do not use it as current left/right or front/back scene-coordinate guidance.
> Use `docs/superpowers/plans/2026-06-26-anatomical-scene-coordinate-repair.md`
> and the active 3D scene spec instead.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: completed on 2026-06-26. See
`docs/superpowers/closeouts/2026-06-26-nakamura-scene-pitch-coordinate-repair.md`
for root cause, verification evidence, and residual limitations.

**Goal:** Repair the systematic gaze pitch sign mismatch that makes upward-looking frames render as downward-looking scene rays.

**Architecture:** Frame gaze angles remain image-overlay semantic values: positive yaw means image-right and positive pitch means image-up. Scene camera vectors are OpenCV-style: +X image-right, +Y image-down, +Z camera-forward. The durable boundary is the conversion from frame-record gaze angles into scene camera vectors.

**Tech Stack:** Python 3.12, Pydantic records, pytest, uv, existing scene artifact and viewer modules.

## Global Constraints

- Work in the current branch.
- Use `artifacts/input/nakamura_1.mp4` for real verification.
- Treat fixes as root-cause engineering work, not symptom suppression.
- Use test-first repair: write a failing regression before production changes.
- Keep coordinate semantics explicit: `image_px` has y down; `camera_opencv_pseudo_m` has +Y down; scene axes use `right_up_back_columns_right_handed`.
- Do not silently substitute `recommended_gaze` for scene UniGaze rays.
- Make meaningful commits along the way.

---

### Task 1: Lock Pitch-To-Camera Sign Semantics

**Files:**
- Modify: `tests/chess_gaze/test_scene_geometry.py`
- Modify: `src/chess_gaze/scene_geometry.py`

**Interfaces:**
- Consumes: `FrameRecord.appearance_gaze` yaw/pitch values in image-overlay convention.
- Produces: `unigaze_ray_from_frame(...).direction_camera` in `camera_opencv_pseudo_m`.

- [x] **Step 1: Write the failing tests**

Add focused tests proving that positive pitch maps to negative camera Y and that the reported Nakamura frame values produce an upward camera ray:

```python
def test_unigaze_ray_from_frame_maps_positive_pitch_to_camera_up() -> None:
    scene_geometry = _scene_geometry()
    midpoint = _midpoint_record(
        valid=True,
        camera_point=_camera_point(0.0, 0.0, 0.7),
        scene_point=_scene_point(0.0, 0.0, 0.0),
        reason_invalid=None,
    )

    ray = scene_geometry.unigaze_ray_from_frame(
        _frame_record_with_gazes(
            appearance_gaze=_gaze_angles(
                valid=True,
                pitch_radians=0.2,
                yaw_radians=0.0,
                reason_invalid=None,
            ),
            recommended_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
        ),
        midpoint,
    )

    assert ray.valid is True
    assert ray.direction_camera is not None
    assert ray.direction_camera.x == pytest.approx(0.0)
    assert ray.direction_camera.y < 0.0
    assert ray.direction_camera.z > 0.0
```

Add a second test using the frame 1651 appearance gaze values:

```python
def test_unigaze_ray_from_frame_preserves_nakamura_upward_gaze_direction() -> None:
    scene_geometry = _scene_geometry()
    midpoint = _midpoint_record(
        valid=True,
        camera_point=_camera_point(-0.46211833054305185, 0.08590315491459204, 1.6175790317537708),
        scene_point=_scene_point(-0.20316321340576493, 0.08391311089468123, -0.0871501757382564),
        reason_invalid=None,
    )

    ray = scene_geometry.unigaze_ray_from_frame(
        _frame_record_with_gazes(
            appearance_gaze=_gaze_angles(
                valid=True,
                pitch_radians=0.8304274082183838,
                yaw_radians=-0.42343080043792725,
                reason_invalid=None,
            ),
            recommended_gaze=_invalid_angles(ErrorCode.GAZE_ESTIMATORS_DISAGREE),
            frame_index=1651,
        ),
        midpoint,
    )

    assert ray.valid is True
    assert ray.direction_camera is not None
    assert ray.direction_camera.x < 0.0
    assert ray.direction_camera.y < 0.0
    assert ray.direction_camera.z > 0.0
```

- [x] **Step 2: Run tests to verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py::test_unigaze_ray_from_frame_maps_positive_pitch_to_camera_up tests/chess_gaze/test_scene_geometry.py::test_unigaze_ray_from_frame_preserves_nakamura_upward_gaze_direction -q
```

Expected: both tests fail because current scene conversion stores positive pitch as positive camera Y.

- [x] **Step 3: Implement the minimal durable conversion**

Add a scene-local conversion helper in `src/chess_gaze/scene_geometry.py`:

```python
def _frame_gaze_angles_to_camera_direction(
    *, pitch_radians: float, yaw_radians: float
) -> tuple[float, float, float]:
    x, image_up_y, z = pitch_yaw_to_unit_vector(
        pitch_radians=pitch_radians,
        yaw_radians=yaw_radians,
    )
    return (x, -image_up_y, z)
```

Use it only in `unigaze_ray_from_frame()` when producing `direction_camera`.

- [x] **Step 4: Run focused tests to verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py::test_unigaze_ray_from_frame_maps_positive_pitch_to_camera_up tests/chess_gaze/test_scene_geometry.py::test_unigaze_ray_from_frame_preserves_nakamura_upward_gaze_direction tests/chess_gaze/test_scene_geometry.py::test_unigaze_ray_from_frame_uses_appearance_gaze_not_recommended_gaze tests/chess_gaze/test_scene_geometry.py::test_unigaze_ray_from_frame_matches_pitch_yaw_sign_convention -q
```

Expected: all selected tests pass after updating any stale expected values to the OpenCV camera convention.

- [x] **Step 5: Commit**

Run:

```sh
git add tests/chess_gaze/test_scene_geometry.py src/chess_gaze/scene_geometry.py
git commit -m "fix: map scene gaze pitch into opencv camera space"
```

### Task 2: Repair Coordinate Documentation And Close The Contradiction

**Files:**
- Modify: `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`
- Modify: `docs/superpowers/plans/2026-06-26-3d-scene-artifact-viewer.md`
- Create: `docs/superpowers/closeouts/2026-06-26-nakamura-scene-pitch-coordinate-repair.md`

**Interfaces:**
- Consumes: AGENTS authority order and current implemented `right_up_back_columns_right_handed` scene basis.
- Produces: Non-contradictory coordinate guidance for future coding agents.

- [x] **Step 1: Update canonical spec text**

Change scene frame wording from `+Z scene-forward` to the implemented and schema-validated `+Z scene-back` / `right_up_back_columns_right_handed` convention. Clarify that frame-record positive pitch is image-up, so conversion to `camera_opencv_pseudo_m` negates the vector Y component.

- [x] **Step 2: Update historical plan notes**

Replace the misleading instruction to reuse `pitch_yaw_to_unit_vector()` directly with the corrected scene conversion boundary. Preserve the historical plan as history, but add a note that this repair supersedes that line.

- [x] **Step 3: Write closeout**

Record:

- root cause;
- durable surface changed;
- regression tests added;
- real Nakamura artifact verification;
- remaining uncertainty, if any.

- [x] **Step 4: Run doc/source focused checks**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py -q
```

- [x] **Step 5: Commit**

Run:

```sh
git add docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md docs/superpowers/plans/2026-06-26-3d-scene-artifact-viewer.md docs/superpowers/closeouts/2026-06-26-nakamura-scene-pitch-coordinate-repair.md
git commit -m "docs: record scene coordinate sign repair"
```

### Task 3: Verify On Real Nakamura Video

**Files:**
- Inspect/write generated artifacts under `artifacts/output/nakamura_1/runs/`.

**Interfaces:**
- Consumes: `artifacts/input/nakamura_1.mp4`, local models under `models`.
- Produces: Fresh scene artifacts showing frame 1651 upward gaze as negative camera Y and positive scene-up direction when visually upward.

- [x] **Step 1: Re-run Nakamura analysis**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

Expected: command succeeds and prints or creates a fresh run under `artifacts/output/nakamura_1/runs/`.

- [x] **Step 2: Inspect frame 1651 records**

Use `jq` or Python to read the fresh run's `records/frames.jsonl` and `records/scene_frames.jsonl` for `frame_index == 1651`.

Expected:

- `appearance_gaze.pitch_radians` remains positive for the visibly upward source frame.
- `unigaze_ray.direction_camera.y` is negative.
- `unigaze_ray.direction_scene.y` is positive when the scene axis basis up vector is aligned with camera up.

- [x] **Step 3: Visually inspect generated frame**

Open:

```sh
artifacts/output/nakamura_1/runs/<fresh-run-id>/processed_frames/f000001651.jpg
```

Expected: source overlay still shows upward gaze, confirming frame-record semantics did not regress.

- [x] **Step 4: Run real-video contract tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts_real_video_contract.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py -q
```

- [x] **Step 5: Commit if tracked verification artifacts or closeout updates changed**

Run:

```sh
git status --short
git add docs/superpowers/closeouts/2026-06-26-nakamura-scene-pitch-coordinate-repair.md
git commit -m "docs: close out nakamura scene pitch verification"
```

### Task 4: Broad Verification And Review

**Files:**
- No source files unless review finds a defect.

**Interfaces:**
- Consumes: all repaired source/docs.
- Produces: final confidence that coordinate interpretation is consistent.

- [x] **Step 1: Run broad local gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

- [x] **Step 2: Request final code review**

Dispatch a fresh review subagent with the branch diff, asking specifically for coordinate sign, scene axis, test, and doc contradictions.

- [x] **Step 3: Fix any Critical or Important findings**

Use focused tests first for any required code fix, then rerun the relevant broad gate.
