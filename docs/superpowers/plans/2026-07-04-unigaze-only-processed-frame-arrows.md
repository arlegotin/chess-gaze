# UniGaze-Only Processed-Frame Arrows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove pupil-derived gaze arrows/calculation/status logic from the default observer and processed-frame renderer so UniGaze is the only gaze vector shown.

**Architecture:** Preserve the current `FrameRecord` schema for artifact compatibility while changing default observer semantics: `appearance_gaze` remains UniGaze, `recommended_gaze` mirrors UniGaze, and `geometric_gaze` is an invalid legacy field. Keep processed-frame rendering in `visualization.py`, removing extra gaze vectors and increasing UniGaze/head-axis visibility with local drawing helpers.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, OpenCV, Pillow, pytest, ruff, mypy, uv.

## Global Constraints

- Work on the current branch; do not create a new worktree.
- Use Superpowers workflow discipline and subagents.
- Use TDD: write focused failing tests before production code.
- Preserve `FrameRecord` schema fields for compatibility; do not add a schema migration.
- Do not select or change external models, checkpoints, inference libraries, or core dependencies.
- Do not introduce speculative abstractions or pass-through modules.
- Use `uv` for project commands.
- Make meaningful commits along the way.

---

## File Structure

- `src/chess_gaze/gaze_observation.py`: keep UniGaze model/crop/vector helpers; remove pupil-derived geometric/recommended synthesis APIs.
- `src/chess_gaze/frame_observation.py`: own default observer semantics, frame status, and schema-compatible gaze field population.
- `src/chess_gaze/visualization.py`: own processed-frame overlay drawing.
- `tests/chess_gaze/test_gaze_observation.py`: verify removed public helpers are no longer exported.
- `tests/chess_gaze/test_frame_observation.py`: verify UniGaze-only observer semantics and status behavior.
- `tests/chess_gaze/test_visualization.py`: verify processed-frame overlay behavior.
- `tests/chess_gaze/test_visualization_real_video.py`: keep fixture records aligned with new semantics.
- `docs/superpowers/closeouts/2026-07-04-unigaze-only-processed-frame-arrows.md`: final evidence and residual risk.

### Task 1: Tests For UniGaze-Only Contracts

**Files:**
- Modify: `tests/chess_gaze/test_gaze_observation.py`
- Modify: `tests/chess_gaze/test_frame_observation.py`
- Modify: `tests/chess_gaze/test_visualization.py`
- Modify: `tests/chess_gaze/test_visualization_real_video.py`

**Interfaces:**
- Consumes: current production behavior.
- Produces: failing tests that define the new observer and renderer contract.

- [ ] **Step 1: Update gaze-observation imports and add removed-helper test**

Remove `GazeThresholds`, `compute_per_eye_geometric_gaze`, and
`synthesize_recommended_gaze` from the import list in
`tests/chess_gaze/test_gaze_observation.py`. Add:

```python
def test_gaze_observation_no_longer_exports_pupil_geometric_helpers() -> None:
    import chess_gaze.gaze_observation as gaze_observation

    assert not hasattr(gaze_observation, "compute_per_eye_geometric_gaze")
    assert not hasattr(gaze_observation, "synthesize_recommended_gaze")
    assert not hasattr(gaze_observation, "GazeThresholds")
```

Delete the old per-eye geometric and recommended-gaze synthesis tests from
`test_per_eye_geometric_gaze_uses_independent_eye_offsets` through
`test_recommended_gaze_averages_agreeing_estimators`.

- [ ] **Step 2: Update frame-observer expectations**

In `test_model_backed_frame_observer_maps_model_outputs_to_frame_record`, change
the gaze assertions to:

```python
assert record.geometric_gaze.valid is False
assert record.geometric_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
assert record.appearance_gaze.valid is True
assert record.recommended_gaze.valid is True
assert record.recommended_gaze.yaw_radians == record.appearance_gaze.yaw_radians
assert record.recommended_gaze.pitch_radians == record.appearance_gaze.pitch_radians
```

In `test_model_backed_frame_observer_preserves_missing_right_eye_reason`, change
only geometric-gaze expectations to:

```python
assert record.geometric_gaze.valid is False
assert record.geometric_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
```

Replace `test_model_backed_frame_observer_marks_gaze_disagreement_as_warning`
with:

```python
def test_model_backed_frame_observer_uses_unigaze_without_disagreement_status(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_face_observation(candidate)),
        gaze_model=_DisagreeingGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    record = observer(_observer_frame())

    assert record.face.present is True
    assert record.left_eye.present is True
    assert record.right_eye.present is True
    assert record.head_pose.valid is True
    assert record.appearance_gaze.valid is True
    assert record.recommended_gaze == record.appearance_gaze
    assert record.status is FrameStatus.OK
    assert record.errors == []
```

Replace `test_model_backed_observer_marks_multiple_candidates_and_gaze_disagreement_warning`
with a test asserting only `MULTIPLE_FACE_CANDIDATES` remains in errors and
status is `WARNING`.

In `test_model_backed_frame_observer_batch_maps_model_rows_to_frames`, change
the second-record assertions to:

```python
assert records[0].recommended_gaze == records[0].appearance_gaze
assert records[1].recommended_gaze == records[1].appearance_gaze
assert records[1].recommended_gaze.valid is True
```

- [ ] **Step 3: Add visualization pixel contract tests**

In `tests/chess_gaze/test_visualization.py`, import the private constants for
focused pixel assertions:

```python
from chess_gaze.visualization import (
    _APPEARANCE_GAZE_COLOR,
    _GEOMETRIC_GAZE_COLOR,
    _RECOMMENDED_GAZE_COLOR,
    render_processed_frame,
)
```

Add helpers:

```python
def _dominant_color_count_near(
    image: np.ndarray, *, x: int, y: int, color: tuple[int, int, int], radius: int = 7
) -> int:
    y_min = max(0, y - radius)
    y_max = min(image.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(image.shape[1], x + radius + 1)
    patch = image[y_min:y_max, x_min:x_max].astype(np.int16)
    target = np.array(color, dtype=np.int16)
    distance = np.max(np.abs(patch - target), axis=2)
    return int(np.count_nonzero(distance <= 45))
```

Add:

```python
def test_processed_frame_renders_only_unigaze_gaze_vector(tmp_path: Path) -> None:
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    record = _observed_record()
    output_path = tmp_path / "unigaze-only.jpg"

    render_processed_frame(frame, record, output_path, quality=100)

    rendered = _rgb_jpeg(output_path)
    assert _dominant_color_count_near(
        rendered, x=108, y=85, color=_APPEARANCE_GAZE_COLOR, radius=10
    ) > 0
    assert _dominant_color_count_near(
        rendered, x=84, y=69, color=_GEOMETRIC_GAZE_COLOR, radius=8
    ) == 0
    assert _dominant_color_count_near(
        rendered, x=140, y=70, color=_GEOMETRIC_GAZE_COLOR, radius=8
    ) == 0
    assert _dominant_color_count_near(
        rendered, x=112, y=84, color=_RECOMMENDED_GAZE_COLOR, radius=8
    ) == 0
```

Add:

```python
def test_processed_frame_does_not_draw_unigaze_label_text(tmp_path: Path) -> None:
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    record = _observed_record()
    output_path = tmp_path / "unlabeled-unigaze.jpg"

    render_processed_frame(frame, record, output_path, quality=100)

    rendered = _rgb_jpeg(output_path)
    label_region = rendered[76:96, 110:170]
    assert int(np.count_nonzero(label_region)) < 80
```

- [ ] **Step 4: Align real-video visualization fixture**

In `tests/chess_gaze/test_visualization_real_video.py`, make fixture
`geometric_gaze` invalid and make `recommended_gaze` match `appearance_gaze`:

```python
payload["geometric_gaze"] = {
    "valid": False,
    "yaw_radians": None,
    "pitch_radians": None,
    "reason_invalid": "GAZE_MODEL_FAILED",
}
payload["appearance_gaze"] = {
    "valid": True,
    "yaw_radians": 0.04,
    "pitch_radians": -0.02,
    "reason_invalid": None,
}
payload["recommended_gaze"] = payload["appearance_gaze"]
```

- [ ] **Step 5: Run tests to verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q
```

Expected: FAIL because old helpers still exist, frame observer still calculates
geometric/recommended disagreement, and visualization still draws geometric and
recommended arrows plus the UniGaze label.

- [ ] **Step 6: Commit failing tests**

```sh
git add tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py tests/chess_gaze/test_visualization_real_video.py
git commit -m "test: define unigaze-only gaze arrow contract"
```

### Task 2: UniGaze-Only Observer And Renderer Implementation

**Files:**
- Modify: `src/chess_gaze/gaze_observation.py`
- Modify: `src/chess_gaze/frame_observation.py`
- Modify: `src/chess_gaze/visualization.py`

**Interfaces:**
- Consumes: failing tests from Task 1.
- Produces: UniGaze-only default observer semantics and processed-frame overlay.

- [ ] **Step 1: Remove pupil-derived gaze helpers**

In `src/chess_gaze/gaze_observation.py`, remove
`DEFAULT_GEOMETRIC_IRIS_SCALE_RADIANS`, `GazeThresholds`, `RecommendedGaze`,
`compute_per_eye_geometric_gaze`, `synthesize_recommended_gaze`,
`_valid_angle_sources`, `_mean_method`, `_max_pairwise_delta`,
`_invalid_recommended_gaze`, `_first_invalid_reason`, and `_eye_offset_xy`.
Keep `pitch_yaw_to_unit_vector` and `_require_finite`.

- [ ] **Step 2: Simplify frame evidence**

In `src/chess_gaze/frame_observation.py`, remove imports of the deleted gaze
helpers. Remove `gaze_thresholds`, `left_geometric`, `right_geometric`, and
`geometric_gaze` from `_FrameEvidence`. Remove the calls to
`compute_per_eye_geometric_gaze()` and `_combine_eye_gazes()`.

- [ ] **Step 3: Populate schema-compatible gaze fields from UniGaze**

In `_record_from_evidence()`, replace recommended synthesis with:

```python
appearance_gaze_record = _face_model_gaze_record(appearance_gaze)
return FrameRecord(
    frame_id=evidence.frame.frame_id,
    frame_index=evidence.frame.frame_index,
    status=_frame_status(
        errors=evidence.errors,
        face=evidence.face_record,
        left_eye=evidence.left_eye,
        right_eye=evidence.right_eye,
        head_pose=evidence.head_pose_record,
        appearance_gaze=appearance_gaze,
    ),
    timestamp_seconds=evidence.frame.timestamp_seconds,
    face=evidence.face_record,
    left_eye=evidence.left_eye,
    right_eye=evidence.right_eye,
    head_pose=evidence.head_pose_record,
    geometric_gaze=_invalid_gaze(ErrorCode.GAZE_MODEL_FAILED),
    appearance_gaze=appearance_gaze_record,
    recommended_gaze=appearance_gaze_record,
    errors=evidence.errors,
)
```

Remove `_combine_eye_gazes()` and `_warning_only_recommended_gaze_disagreement()`.
Change `_frame_status()` to accept no `recommended_gaze` parameter and to return:

```python
if not (
    face.present
    and left_eye.present
    and right_eye.present
    and head_pose.valid
    and appearance_gaze.valid
):
    return FrameStatus.ERROR
if errors:
    if _only_warning_errors(errors):
        return FrameStatus.WARNING
    return FrameStatus.ERROR
return FrameStatus.OK
```

Keep `FRAME_WARNING_ERROR_CODES` with only `MULTIPLE_FACE_CANDIDATES`.

- [ ] **Step 4: Remove non-UniGaze arrows and strengthen visible overlays**

In `src/chess_gaze/visualization.py`, stop calling `_draw_eye_gaze_vectors()`
and remove that function. Replace `_draw_face_gaze_vectors()` with
`_draw_unigaze_vector()` that only draws `record.appearance_gaze` from the face
center and passes `label=None`.

Update `_draw_gaze_vector()` to support an outline, thickness, and length scale:

```python
def _draw_gaze_vector(
    image: np.ndarray,
    origin: Point2D,
    gaze: GazeAngles,
    color: Color,
    *,
    label: str | None,
    thickness: int = 2,
    length_scale: float = 0.25,
    outline: bool = False,
) -> None:
    ...
    length = max(30.0, min(image.shape[:2]) * length_scale)
    ...
    if outline:
        cv2.arrowedLine(
            image,
            start,
            end,
            _TEXT_SHADOW_COLOR,
            thickness + 3,
            line_type=cv2.LINE_AA,
            tipLength=0.28,
        )
    cv2.arrowedLine(
        image,
        start,
        end,
        color,
        thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.28,
    )
```

Call it from `_draw_unigaze_vector()` with `thickness=4`,
`length_scale=0.36`, and `outline=True`.

Add a head-axis helper:

```python
def _draw_axis_arrow(image: np.ndarray, start: Pixel, end: Pixel, color: Color) -> None:
    cv2.arrowedLine(image, start, end, _TEXT_SHADOW_COLOR, 5, line_type=cv2.LINE_AA)
    cv2.arrowedLine(image, start, end, color, 3, line_type=cv2.LINE_AA)
```

Use `_draw_axis_arrow()` for x, y, and z head axes.

- [ ] **Step 5: Run focused GREEN verification**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit implementation**

```sh
git add src/chess_gaze/gaze_observation.py src/chess_gaze/frame_observation.py src/chess_gaze/visualization.py tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py tests/chess_gaze/test_visualization_real_video.py
git commit -m "feat: use unigaze-only processed-frame arrows"
```

### Task 3: Verification, Review, And Closeout

**Files:**
- Create: `docs/superpowers/closeouts/2026-07-04-unigaze-only-processed-frame-arrows.md`

**Interfaces:**
- Consumes: completed implementation from Task 2.
- Produces: fresh gate evidence, subagent review, and closeout.

- [ ] **Step 1: Run focused tests**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q
```

- [ ] **Step 2: Run broad local tests excluding documented heavy/real-video gates**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
```

- [ ] **Step 3: Run lint, format, and typing**

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

- [ ] **Step 4: Run real-video visualization smoke if artifact exists**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_visualization_real_video.py -q
```

If `artifacts/input/nakamura_short.mp4` is missing, record the exact failure in
the closeout instead of claiming the smoke passed.

- [ ] **Step 5: Request subagent code review**

Dispatch a reviewer with the branch diff and this plan. Fix Critical and
Important findings before final closeout.

- [ ] **Step 6: Write closeout**

Create `docs/superpowers/closeouts/2026-07-04-unigaze-only-processed-frame-arrows.md`
with:

```markdown
# UniGaze-Only Processed-Frame Arrows Closeout

Date: 2026-07-04

## Summary

[Summarize implemented behavior.]

## Root Cause / Durable Surface

[Explain that default processing had kept a legacy pupil-derived geometric
comparison path that no longer matched the useful vector surface.]

## Verification

[Paste exact commands and pass/fail outcomes.]

## Residual Risk

[List any blocked real-data or environment-specific checks.]
```

- [ ] **Step 7: Commit closeout and any review fixes**

```sh
git add docs/superpowers/closeouts/2026-07-04-unigaze-only-processed-frame-arrows.md
git commit -m "docs: close out unigaze-only processed-frame arrows"
```

## Self-Review

- Spec coverage: all acceptance criteria map to Tasks 1-3.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation step remains.
- Type consistency: `GazeAngles`, `FaceModelGaze`, `FrameRecord`, and
  `FrameStatus` names match current source.
