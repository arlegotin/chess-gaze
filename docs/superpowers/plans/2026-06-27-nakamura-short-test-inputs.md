# Nakamura Short Test Inputs Implementation Plan

> Fixture note, 2026-06-29: `artifacts/input/nakamura_short.mp4` was replaced
> after this plan was completed. Current fixture expectations and digest are in
> [2026-06-29-nakamura-short-video-refresh.md](../closeouts/2026-06-29-nakamura-short-video-refresh.md).
> Historical values below describe the prior clip.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every test that consumes a real video from `artifacts/input/` with `artifacts/input/nakamura_short.mp4`, preserving useful real-video coverage while removing dependencies on removed or slower clips.

**Architecture:** This is a tests-only repair. Model-free tests will assert the short clip's actual decode contract, model-backed tests will reuse one shared short-video sample set, and schema/benchmark fixture literals will point at the short clip so test data no longer encodes legacy real inputs.

**Tech Stack:** Python 3.12, pytest, uv, PyAV, MediaPipe Face Landmarker, UniGaze local checkpoint, Pydantic schemas.

## Global Constraints

- Work in the current branch `optimize-inference`; do not create or switch to a git worktree because the user explicitly required current-branch work.
- Do not change production code for this task.
- Use `artifacts/input/nakamura_short.mp4` for every test that consumes or represents a real video from `artifacts/input/`.
- Local verified `nakamura_short.mp4` sha256 is `4f4f4f0035dacd0e469e50ed1c8f78d4de93964474f3a5656117cc4d13383d6e`.
- Local verified `nakamura_short.mp4` decoded frame count is `180`, width is `1920`, and height is `1080`.
- Use sampled short-video frame indices `(0, 30, 60, 90, 120, 150, 179)` for real model-backed sample tests.
- Preserve synthetic manifest-only fixture strings such as `artifacts/input/synthetic_scene_source.mp4`; they are not real videos consumed from `artifacts/input/`.
- Use `uv` for all Python verification commands.
- Commit meaningful milestones: plan, test updates, closeout/final verification.

---

### Task 1: Model-Free Real-Video Contract Tests

**Files:**
- Modify: `tests/chess_gaze/test_video_decode_real_video.py`
- Modify: `tests/chess_gaze/test_pipeline_real_video_contract.py`
- Modify: `tests/chess_gaze/test_qa_summary_real_video_contract.py`
- Modify: `tests/chess_gaze/test_visualization_real_video.py`
- Modify: `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`

**Interfaces:**
- Consumes: `artifacts/input/nakamura_short.mp4`, `inspect_video()`, `iter_decoded_frames()`, `analyze_video()`, `render_processed_frame()`, `build_scene_artifacts()`.
- Produces: model-free tests that assert a single 180-frame short-video contract.

- [ ] **Step 1: Verify the existing RED failure**

Run:

```bash
uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py -q
```

Expected: FAIL because `artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4` are missing.

- [ ] **Step 2: Replace old parametrized real videos with the short clip**

In `test_video_decode_real_video.py`, replace the parametrized video list with:

```python
NAKAMURA_SHORT_VIDEO = Path("artifacts/input/nakamura_short.mp4")
NAKAMURA_SHORT_FRAME_COUNT = 180


@pytest.mark.parametrize(
    ("path", "expected_count"),
    [(NAKAMURA_SHORT_VIDEO, NAKAMURA_SHORT_FRAME_COUNT)],
)
```

Apply the same `NAKAMURA_SHORT_VIDEO` and `NAKAMURA_SHORT_FRAME_COUNT` constants to `test_pipeline_real_video_contract.py` and `test_qa_summary_real_video_contract.py`, preserving the existing test bodies and changing only the parameter values and expected count.

- [ ] **Step 3: Point visualization coverage at the short clip**

In `test_visualization_real_video.py`, replace:

```python
MANDATORY_VIDEO_PATHS = (
    Path("artifacts/input/test_1.mp4"),
    Path("artifacts/input/test_2.mp4"),
)
```

with:

```python
MANDATORY_VIDEO_PATHS = (Path("artifacts/input/nakamura_short.mp4"),)
```

- [ ] **Step 4: Rebaseline scene artifact real-video contract to 180 frames**

In `test_scene_artifacts_real_video_contract.py`, replace:

```python
video_path = Path("artifacts/input/nakamura_1.mp4")
```

with:

```python
video_path = Path("artifacts/input/nakamura_short.mp4")
expected_frame_count = 180
```

Then replace every hard-coded `1973` expectation in that test body with `expected_frame_count`, and replace the last-frame index assertion with:

```python
assert viewer_data.valid_hit_points[-1].frame_index == expected_frame_count - 1
```

- [ ] **Step 5: Run focused GREEN verification**

Run:

```bash
uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected: PASS with all tests in these five files using `nakamura_short.mp4`.

- [ ] **Step 6: Commit**

Run:

```bash
git add tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py
git commit -m "test: use short Nakamura video for model-free contracts"
```

---

### Task 2: Model-Backed Real-Video Sample Tests

**Files:**
- Modify: `tests/chess_gaze/test_face_observation_real_video.py`
- Modify: `tests/chess_gaze/test_eye_observation_real_video.py`
- Modify: `tests/chess_gaze/test_head_pose_real_video.py`
- Modify: `tests/chess_gaze/test_gaze_observation_real_video.py`

**Interfaces:**
- Consumes: `MediaPipeFaceObserver`, `observe_eyes()`, `estimate_head_pose()`, `UniGazeModel`, `ModelBackedFrameObserver`, `iter_decoded_frames()`.
- Produces: model-backed tests that sample `nakamura_short.mp4` frames `(0, 30, 60, 90, 120, 150, 179)`.

- [ ] **Step 1: Set one short-video sample map in each file**

In all four files, replace any `SAMPLED_FRAME_INDICES` map that names `test_1.mp4` and `test_2.mp4` with:

```python
NAKAMURA_SHORT_VIDEO = Path("artifacts/input/nakamura_short.mp4")
NAKAMURA_SHORT_FRAME_INDICES = (0, 30, 60, 90, 120, 150, 179)
SAMPLED_FRAME_INDICES = {
    NAKAMURA_SHORT_VIDEO: NAKAMURA_SHORT_FRAME_INDICES,
}
```

- [ ] **Step 2: Replace removed-video repair samples in face observation tests**

In `test_face_observation_real_video.py`, delete the `TEST_0_RECOVERED_FRAME_INDICES`, `MIX_2_REPORTED_VISIBLE_FACE_REGIONS`, and `NAKAMURA_OVEREXPANDED_FULL_FRAME_REGIONS` constants. Add:

```python
NAKAMURA_SHORT_EXPECTED_FACE_BOXES = {
    "f000000000": (329.6, 669.6, 518.1, 873.4),
    "f000000030": (333.0, 710.5, 526.4, 903.9),
    "f000000060": (328.3, 676.9, 529.2, 888.5),
    "f000000090": (353.8, 650.5, 550.0, 903.1),
    "f000000120": (371.5, 686.1, 578.9, 938.1),
    "f000000150": (369.5, 688.5, 581.1, 941.5),
    "f000000179": (354.2, 668.4, 576.7, 947.8),
}
```

Replace the three removed-video-specific tests with two short-video tests:

```python
def test_mediapipe_face_observer_recovers_nakamura_short_visible_faces() -> None:
    ...
    sampled_frames = _sample_frames(video_path, NAKAMURA_SHORT_FRAME_INDICES)
    ...
    assert recovered_boxes == NAKAMURA_SHORT_EXPECTED_FACE_BOXES
```

and:

```python
def test_mediapipe_observer_keeps_nakamura_short_faces_bounded() -> None:
    ...
    assert width <= 230.0
    assert height <= 285.0
    assert recovered_boxes == NAKAMURA_SHORT_EXPECTED_FACE_BOXES
```

Both tests must keep the existing model-registry checksum checks and `observer.close()` cleanup pattern.

- [ ] **Step 3: Replace removed `test_0.mp4` samples in head-pose tests**

In `test_head_pose_real_video.py`, replace the `TEST_0_*` constants with:

```python
NAKAMURA_SHORT_TRANSFORM_POSE_FRAME_INDICES = (0, 30, 60, 90, 120, 150, 179)
NAKAMURA_SHORT_DOWN_LOOKING_FRAME_INDICES = frozenset(
    NAKAMURA_SHORT_TRANSFORM_POSE_FRAME_INDICES
)
```

Rename `test_head_pose_uses_mediapipe_transform_on_test0_pnp_failure_frames` to:

```python
def test_head_pose_uses_mediapipe_transform_on_nakamura_short_frames() -> None:
```

Set:

```python
video_path = REPO_ROOT / NAKAMURA_SHORT_VIDEO
sampled_frames = _sample_frames(
    video_path,
    NAKAMURA_SHORT_TRANSFORM_POSE_FRAME_INDICES,
)
```

Update the final `valid_frame_ids` assertion to:

```python
assert valid_frame_ids == [
    "f000000000",
    "f000000030",
    "f000000060",
    "f000000090",
    "f000000120",
    "f000000150",
    "f000000179",
]
```

and check `frame.frame_index in NAKAMURA_SHORT_DOWN_LOOKING_FRAME_INDICES`.

- [ ] **Step 4: Replace removed `test_0.mp4` samples in gaze tests**

In `test_gaze_observation_real_video.py`, replace:

```python
TEST_0_RECOMMENDED_FRAME_INDICES = (90, 155, 217)
```

with:

```python
NAKAMURA_SHORT_RECOMMENDED_FRAME_INDICES = (0, 90, 179)
```

Rename `test_default_model_observer_recommends_gaze_on_repaired_test0_frames` to:

```python
def test_default_model_observer_recommends_gaze_on_nakamura_short_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
```

Set `video_path = REPO_ROOT / NAKAMURA_SHORT_VIDEO`, use `tmp_path / "nakamura-short-observer"`, sample `NAKAMURA_SHORT_RECOMMENDED_FRAME_INDICES`, and assert:

```python
assert [record.frame_index for record in records] == list(
    NAKAMURA_SHORT_RECOMMENDED_FRAME_INDICES
)
```

- [ ] **Step 5: Run focused GREEN verification**

Run:

```bash
uv run pytest tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_eye_observation_real_video.py tests/chess_gaze/test_head_pose_real_video.py tests/chess_gaze/test_gaze_observation_real_video.py -q
```

Expected: PASS when local MediaPipe and UniGaze assets are present.

- [ ] **Step 6: Commit**

Run:

```bash
git add tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_eye_observation_real_video.py tests/chess_gaze/test_head_pose_real_video.py tests/chess_gaze/test_gaze_observation_real_video.py
git commit -m "test: rebaseline model-backed samples on short Nakamura clip"
```

---

### Task 3: Metadata-Only Test Fixtures

**Files:**
- Modify: `tests/chess_gaze/test_frame_records.py`
- Modify: `tests/chess_gaze/test_scene_records.py`
- Modify: `tests/chess_gaze/test_unigaze_batch_benchmark.py`

**Interfaces:**
- Consumes: schema constructors and benchmark report helper functions.
- Produces: fixture payloads that represent `nakamura_short.mp4` instead of `nakamura_1.mp4`.

- [ ] **Step 1: Update frame manifest fixture literals**

In `test_frame_records.py`, replace every test fixture occurrence of:

```python
"artifacts/input/nakamura_1.mp4"
```

with:

```python
"artifacts/input/nakamura_short.mp4"
```

and replace the related decoded frame fixture value:

```python
frame_count_decoded=1973
```

or:

```python
"frame_count_decoded": 1973
```

with `180`.

- [ ] **Step 2: Update scene record fixture literals**

In `test_scene_records.py`, replace:

```python
"source_video_path": "artifacts/input/nakamura_1.mp4"
```

with:

```python
"source_video_path": "artifacts/input/nakamura_short.mp4"
```

and replace every fixture `source_video_stem` value of `"nakamura_1"` with `"nakamura_short"`.

- [ ] **Step 3: Update benchmark report helper literals**

In `test_unigaze_batch_benchmark.py`, replace the `_report()` helper values:

```python
source_video="artifacts/input/nakamura_1.mp4",
decoded_frame_count=1973,
baseline_run_dir="artifacts/output/nakamura_1/runs/baseline",
```

with:

```python
source_video="artifacts/input/nakamura_short.mp4",
decoded_frame_count=180,
baseline_run_dir="artifacts/output/nakamura_short/runs/baseline",
```

Replace the `_candidate()` default:

```python
qa_decoded_frames: int | None = 1973,
```

with:

```python
qa_decoded_frames: int | None = 180,
```

- [ ] **Step 4: Verify no legacy real-video test literals remain**

Run:

```bash
rg -n "test_0\\.mp4|test_1\\.mp4|test_2\\.mp4|mix_2\\.mp4|nakamura_1\\.mp4|bortnyk_1\\.mp4|carlsen_1\\.mp4|gotham_1\\.mp4|kramnik_1\\.mp4|nepo_1\\.mp4" tests
```

Expected: no output.

- [ ] **Step 5: Run focused GREEN verification**

Run:

```bash
uv run pytest tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_unigaze_batch_benchmark.py
git commit -m "test: point metadata fixtures at short Nakamura clip"
```

---

### Task 4: Final Verification and Closeout

**Files:**
- Create: `docs/superpowers/closeouts/2026-06-27-nakamura-short-test-inputs.md`

**Interfaces:**
- Consumes: completed test updates from Tasks 1-3.
- Produces: closeout documenting root cause, durable surface changed, regression evidence, exact verification results, and residual risks.

- [ ] **Step 1: Run all changed test files together**

Run:

```bash
uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_eye_observation_real_video.py tests/chess_gaze/test_head_pose_real_video.py tests/chess_gaze/test_gaze_observation_real_video.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broad local gates**

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: PASS. If a broad gate fails for unrelated pre-existing reasons, record the exact failure and run the broadest useful subset that isolates this change.

- [ ] **Step 3: Write the closeout**

Create `docs/superpowers/closeouts/2026-06-27-nakamura-short-test-inputs.md` with:

```markdown
# Nakamura Short Test Inputs Closeout

## Summary

- Replaced tests that consumed removed or slower `artifacts/input/*.mp4` real videos with `artifacts/input/nakamura_short.mp4`.
- Preserved model-free decode, pipeline, QA summary, visualization, scene artifact, MediaPipe, eye, head-pose, UniGaze, schema, and benchmark fixture coverage.

## Root Cause

Tests encoded mandatory dependencies on legacy local videos that are no longer present or are slower than the short canonical clip.

## Durable Surface Changed

The test suite now has one canonical real-video fixture for local real-video coverage: `artifacts/input/nakamura_short.mp4`.

## Verification

- Record the exact focused and broad command outputs from Steps 1 and 2.

## Residual Risk

- Record any skipped model-backed test reasons or broad-gate failures exactly; otherwise state that no residual test-input risk is known.
```

- [ ] **Step 4: Final search audit**

Run:

```bash
rg -n "test_0\\.mp4|test_1\\.mp4|test_2\\.mp4|mix_2\\.mp4|nakamura_1\\.mp4|bortnyk_1\\.mp4|carlsen_1\\.mp4|gotham_1\\.mp4|kramnik_1\\.mp4|nepo_1\\.mp4" tests docs/superpowers/closeouts/2026-06-27-nakamura-short-test-inputs.md
```

Expected: no output from `tests`; closeout mentions legacy names only in explanatory prose if needed.

- [ ] **Step 5: Commit closeout**

Run:

```bash
git add docs/superpowers/closeouts/2026-06-27-nakamura-short-test-inputs.md
git commit -m "docs: close out short Nakamura test input repair"
```

- [ ] **Step 6: Final review**

Dispatch a final code-review subagent with the branch diff package. Fix any Critical or Important findings, rerun covering tests, and commit the fix before final response.

---

## Self-Review

Spec coverage: this plan covers removed-video failures, all test real-video references under `artifacts/input/`, preservation of valuable model-free/model-backed coverage, focused and broad verification, current-branch work, meaningful commits, and closeout documentation.

Placeholder scan: none remain.

Type consistency: all tasks use `Path`, pytest, existing helper names, and `nakamura_short.mp4` frame count `180`.
