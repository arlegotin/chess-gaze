# Mix 2 Face Observation Quality Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the `mix_2` face-observation failures reported in `artifacts/output/mix_2/runs/20260625T203626Z-f401b16f` without masking real detector uncertainty in visualization.

**Architecture:** Keep the durable fix at `MediaPipeFaceObserver`. Expand deterministic image-mode detection regions enough to cover small picture-in-picture webcam panes, then arbitrate candidates by face-likeness and layout evidence before downstream eyes, head pose, gaze, and rendering consume one selected full-image coordinate set.

**Tech Stack:** Python 3.12, MediaPipe Face Landmarker IMAGE mode, OpenCV drawing, Pydantic frame records, pytest, uv.

## Global Constraints

- Work in the current branch.
- Use Superpowers and subagents.
- Use TDD for behavior changes.
- Preserve raw and processed full-frame artifact contracts.
- Do not add a new model, checkpoint, inference library, or core dependency.
- Keep visualization a renderer of frame records; do not hide wrong observations with render-time smoothing.
- Use current source-layout rules and avoid speculative abstractions.
- Verify the reported frames one by one: `f000000237`, `f000000265`-`f000000266`, `f000000268`, `f000000422`-`f000000423`, `f000000510`, `f000000524`, and `f000000532`.

---

### Task 1: Deterministic Region Coverage And Candidate Arbitration

**Files:**
- Modify: `tests/chess_gaze/test_face_observation.py`
- Modify: `src/chess_gaze/face_observation.py`

**Interfaces:**
- Consumes: `MediaPipeFaceObserver.observe(rgb_frame, frame_id=...) -> FaceObservation`
- Produces: a selected `FaceSelection` whose `FaceCandidate` coordinates are in full source-image pixels.

- [ ] **Step 1: Write failing unit tests**

Add focused synthetic tests that simulate the `mix_2` failure modes:

- full frame misses, left half has multiple candidates, and a tighter top-left region selects the real face;
- full frame has a tall false positive, while a focused left/top region selects the real face;
- full and half regions miss a small right-side webcam face, while a top-right region recovers it;
- full and half regions miss a small right-side webcam face, while a right-upper-middle region recovers it.

Run:

```sh
uv run pytest tests/chess_gaze/test_face_observation.py -q
```

Expected before implementation: the new tests fail because the observer either selects the area-largest wrong half-frame candidate, returns a bad full-frame candidate early, or never tries smaller upper/right regions.

- [ ] **Step 2: Implement minimal production fix**

Modify `MediaPipeFaceObserver.observe()` so it evaluates all deterministic regions for a frame before final selection. Region list must remain deterministic and finite. Add regions that match the observed split-screen layout classes:

- full frame;
- left half and right half;
- left top and right top;
- left upper band and right upper band;
- right upper-middle band.

Select from region results with a bounded scoring rule that:

- preserves existing full-frame selection for unambiguous, plausible observations;
- permits a non-full focused region to replace a full-frame candidate when the focused candidate has a clearly more face-like aspect ratio and the full candidate is implausibly large for the observed small webcam pane;
- permits a tighter non-full focused region to replace a multiple-candidate half-frame selection when it overlaps the better local face region;
- uses focused-region results when full and half-frame regions miss visible small faces;
- rejects seam-clipped candidates.

- [ ] **Step 3: Verify focused unit tests**

Run:

```sh
uv run pytest tests/chess_gaze/test_face_observation.py -q
```

Expected after implementation: all face-observation tests pass.

- [ ] **Step 4: Commit**

```sh
git add tests/chess_gaze/test_face_observation.py src/chess_gaze/face_observation.py docs/superpowers/plans/2026-06-25-mix-2-face-observation-quality-repair.md
git commit -m "fix: improve face observation region arbitration"
```

### Task 2: Mix 2 Real-Video Regression And Artifact Verification

**Files:**
- Modify: `tests/chess_gaze/test_face_observation_real_video.py`
- Create: `docs/superpowers/closeouts/2026-06-25-mix-2-face-observation-quality-repair.md`

**Interfaces:**
- Consumes: `artifacts/input/mix_2.mp4`
- Produces: a fresh run under `artifacts/output/mix_2/runs/`

- [ ] **Step 1: Write failing real-video regression**

Add a bounded `mix_2.mp4` real-video test for the reported frames. For each frame, assert `MediaPipeFaceObserver.observe()` returns a present face and that the selected bounding box center falls inside the manually observed face region for that scene:

- `f000000237`: left webcam face region around `x=300..450`, `y=130..310`;
- `f000000265`, `f000000266`, `f000000268`: left webcam face region around `x=270..420`, `y=130..310`;
- `f000000422`, `f000000423`: right-top webcam face region around `x=930..1100`, `y=200..360`;
- `f000000510`, `f000000524`: right-top webcam face region around `x=880..1040`, `y=170..330`;
- `f000000532`: right-top webcam face region around `x=790..970`, `y=160..330`.

Run:

```sh
uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_mix2_reported_visible_faces -q
```

Expected before implementation: the new test fails for the reported misses and wrong-location frames.

- [ ] **Step 2: Regenerate `mix_2`**

Run:

```sh
uv run chess-gaze analyze artifacts/input/mix_2.mp4
```

Expected: complete run with `540` raw frames, `540` processed frames, and `540` frame records.

- [ ] **Step 3: Visual and numeric verification**

Inspect the fresh processed frames against the old run for each reported issue:

- `f000000237`;
- `f000000265`-`f000000266`;
- `f000000268`;
- `f000000422`-`f000000423`;
- `f000000510`;
- `f000000524`;
- `f000000532`.

Confirm that face boxes and landmarks are anchored to the visible face regions and that frame-record boxes are numerically within the expected region for each scene.

- [ ] **Step 4: Run broad gates**

Run:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: all gates pass, or exact failures are recorded in the closeout.

- [ ] **Step 5: Request final review, write closeout, and commit**

Request subagent code review on the branch diff. Write root cause, durable surface changed, regression tests, visual evidence, current primary-source research, and remaining limitations in the closeout.

Commit:

```sh
git add tests/chess_gaze/test_face_observation_real_video.py docs/superpowers/closeouts/2026-06-25-mix-2-face-observation-quality-repair.md
git commit -m "test: cover mix 2 visible face recoveries"
```
