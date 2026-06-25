# Mix 1 Overlay Quality Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair overlay placement and status semantics for the freshly generated `mix_1` run without treating valid gaze-estimator divergence as a frame error.

**Architecture:** Keep the durable fix at the frame-observation boundary. Face-region arbitration should choose the best reliable MediaPipe face observation before eyes, head pose, gaze, and visualization consume it; frame status should distinguish hard observation failures from warning-only gaze disagreement.

**Tech Stack:** Python 3.12, MediaPipe Face Landmarker, OpenCV drawing, Pydantic frame records, pytest, uv.

## Global Constraints

- Work in the current branch.
- Use TDD for behavior changes.
- Preserve raw and processed full-frame artifact contracts.
- Missing white recommended-gaze arrow is not an error when `recommended_gaze.valid` is false due only to `GAZE_ESTIMATORS_DISAGREE`.
- Do not select a model or dependency; use existing local MediaPipe and UniGaze assets.
- Use current source-layout rules and avoid speculative abstractions.

---

### Task 1: Face-Region Arbitration

**Files:**
- Modify: `tests/chess_gaze/test_face_observation.py`
- Modify: `src/chess_gaze/face_observation.py`

**Interfaces:**
- Consumes: `MediaPipeFaceObserver.observe(rgb_frame, frame_id=...) -> FaceObservation`
- Produces: a selected `FaceSelection` whose `FaceCandidate` coordinates are in full source-image pixels.

- [ ] **Step 1: Write failing tests**

Add tests that simulate:

- full-frame multiple candidates where the focused right-half candidate overlaps the face and should be selected;
- full-frame low/partial bottom-face detection where the left-half candidate starts materially higher and should be selected.

Run:

```sh
uv run pytest tests/chess_gaze/test_face_observation.py -q
```

Expected before implementation: the new tests fail because `MediaPipeFaceObserver.observe()` returns the first successful full-frame selection.

- [ ] **Step 2: Implement minimal arbitration**

Modify `MediaPipeFaceObserver.observe()` to inspect fallback half-frame detections when the full-frame detection is ambiguous or plausibly partial, then prefer a focused half-frame selection only when it overlaps the full-frame detection and is not clipped by the vertical seam.

- [ ] **Step 3: Verify focused tests**

Run:

```sh
uv run pytest tests/chess_gaze/test_face_observation.py -q
```

Expected after implementation: all face-observation tests pass.

### Task 2: Warning-Only Gaze Disagreement

**Files:**
- Modify: `tests/chess_gaze/test_frame_observation.py`
- Modify: `tests/chess_gaze/test_qa_summary.py`
- Modify: `src/chess_gaze/errors.py`
- Modify: `src/chess_gaze/frame_observation.py`
- Modify: `src/chess_gaze/qa_summary.py`
- Modify: `src/chess_gaze/visualization.py`

**Interfaces:**
- Consumes: `ErrorCode.GAZE_ESTIMATORS_DISAGREE`
- Produces: `FrameStatus.WARNING` for warning-only records instead of `FrameStatus.ERROR`.

- [ ] **Step 1: Write failing tests**

Add tests that require a model-backed frame with valid face, eyes, head pose, and appearance gaze but invalid recommended gaze due only to estimator disagreement to be `WARNING`, not `ERROR`. Add a QA-summary test that warning-only records are counted as warnings but not representative failures.

Run:

```sh
uv run pytest tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_qa_summary.py -q
```

Expected before implementation: the new tests fail because all record errors force `ERROR`, and representative failures include warning-only records.

- [ ] **Step 2: Implement warning semantics**

Add `FrameStatus.WARNING`, teach frame-status calculation to return it for warning-only disagreement, color warning status separately in visualization, and keep representative failures focused on hard errors.

- [ ] **Step 3: Verify focused tests**

Run:

```sh
uv run pytest tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_qa_summary.py -q
```

Expected after implementation: all focused tests pass.

### Task 3: Artifact Regeneration and Verification

**Files:**
- Create: `docs/superpowers/closeouts/2026-06-25-mix-1-overlay-quality-repair.md`

**Interfaces:**
- Consumes: `artifacts/input/mix_1.mp4`
- Produces: a fresh run under `artifacts/output/mix_1/runs/`

- [ ] **Step 1: Regenerate `mix_1`**

Run:

```sh
uv run chess-gaze analyze artifacts/input/mix_1.mp4
```

Expected: complete run with 240 raw frames, 240 processed frames, and 240 frame records.

- [ ] **Step 2: Visual and numeric verification**

Inspect the new processed frames corresponding to old problem ranges. Verify that head boxes and eye overlays are anchored to the visible head/eyes materially better than the old run, and that missing white arrows on warning-only divergence frames are not treated as errors.

- [ ] **Step 3: Run broad gates**

Run:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: all gates pass, or exact failures are recorded in the closeout.

- [ ] **Step 4: Write closeout and commit**

Write root cause, durable surface changed, regression tests, visual evidence, and remaining limitations in the closeout. Commit coherent slices with tests and verification evidence.
