# Carlsen Face Arbitration Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair visible-face misses and wrong plaque/background face selections in `carlsen_1` without frame-specific rules.

**Architecture:** Keep MediaPipe Face Landmarker in deterministic `IMAGE` mode. The durable boundary is `MediaPipeFaceObserver` candidate arbitration: it must convert every region candidate into full-frame coordinates, reject seam-clipped focused candidates, and choose the candidate supported by deterministic cross-region evidence before frame records, eye crops, gaze, scene artifacts, and overlays consume it.

**Tech Stack:** Python 3.12, uv, pytest, MediaPipe Face Landmarker, NumPy, PyAV, Pillow contact-sheet inspection.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Work in the current branch, per user request.
- Use installed Superpowers skills for debugging, TDD, subagents, and verification.
- Use subagents for independent code/artifact/doc review.
- Use `artifacts/input/nakamura_short.mp4` for real verification.
- Use the existing `artifacts/input/carlsen_1.mp4` and reported run artifacts for diagnosis and focused verification.
- Do not add temporal smoothing, MediaPipe `VIDEO` mode, tracking, interpolation, or cross-frame averaging.
- Do not hardcode the reported frame numbers, Carlsen-specific coordinates, or plaque-specific exclusions in production code.
- Do not add a new model, checkpoint, inference library, or core dependency.
- Preserve MediaPipe crop-to-full-frame coordinate conversion.
- Make meaningful commits along the way.

---

### Task 1: Lock Candidate-Level Fallback Regression

**Files:**
- Modify: `tests/chess_gaze/test_face_observation_region_arbitration.py`

**Interfaces:**
- Consumes: `MediaPipeFaceObserver.observe(rgb_frame, frame_id=...)`.
- Produces: selected `FaceCandidate` with full-image pixel coordinates.

- [ ] **Step 1: Write failing unit regression**

Add a fake MediaPipe sequence where full frame misses, a broad left-half region detects the real face, and a focused upper-band region returns both the real face and a larger false positive. The expected selected candidate is the cross-region-supported real face, not the larger single-region false positive.

- [ ] **Step 2: Run RED**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate -q
```

Expected before implementation: FAIL because current fallback selection returns the upper-band selection whose primary candidate is the larger false positive.

- [ ] **Step 3: Commit failing regression**

```sh
git add tests/chess_gaze/test_face_observation_region_arbitration.py docs/superpowers/plans/2026-06-29-carlsen-face-arbitration-repair.md
git commit -m "test: reproduce carlsen fallback face jumps"
```

### Task 2: Repair Fallback Candidate Arbitration

**Files:**
- Modify: `src/chess_gaze/face_observation.py`
- Modify: `tests/chess_gaze/test_face_observation_region_arbitration.py`

**Interfaces:**
- Consumes: all non-full-frame `FaceSelection.candidates`.
- Produces: a fallback `FaceSelection` whose primary candidate can be any valid non-seam candidate, not only a region's area-selected primary.

- [ ] **Step 1: Implement minimal fix**

Update `_select_fallback_face()` so it scores every valid, non-seam focused-region candidate. Add an overlap-consensus multiplier using matching candidates from other deterministic regions. Preserve existing single-region fallback behavior when no overlap evidence exists.

- [ ] **Step 2: Run GREEN**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate -q
```

Expected after implementation: PASS.

- [ ] **Step 3: Run focused arbitration suite**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_region_arbitration.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit arbitration fix**

```sh
git add src/chess_gaze/face_observation.py tests/chess_gaze/test_face_observation_region_arbitration.py
git commit -m "fix: prefer consensus face candidates"
```

### Task 3: Lock Recoverable Miss Regression

**Files:**
- Modify: `tests/chess_gaze/test_face_observation.py`
- Modify: `src/chess_gaze/face_observation.py`

**Interfaces:**
- Consumes: deterministic detection region list from `_detection_regions(...)`.
- Produces: finite focused regions that include the upper-left player-camera pane without assuming a specific video or frame ID.

- [ ] **Step 1: Write failing unit regression**

Add a fake MediaPipe sequence where full frame, left half, left top, and left upper band miss, but a narrower upper-left inner region detects a visible face. The expected selected candidate is that recovered face in full-frame coordinates.

- [ ] **Step 2: Run RED**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py::test_mediapipe_observer_recovers_visible_face_from_left_upper_inner_region -q
```

Expected before implementation: FAIL because the region list has no left upper-inner crop.

- [ ] **Step 3: Add deterministic region**

Add a left upper-inner focused crop to `_detection_regions()`, using frame fractions rather than video-specific coordinates. Keep the existing full-frame, half-frame, top, upper-band, and right-upper-middle regions.

- [ ] **Step 4: Run GREEN**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py::test_mediapipe_observer_recovers_visible_face_from_left_upper_inner_region -q
```

Expected after implementation: PASS.

- [ ] **Step 5: Commit region fix**

```sh
git add src/chess_gaze/face_observation.py tests/chess_gaze/test_face_observation.py
git commit -m "fix: add left upper inner face recovery region"
```

### Task 3B: Lock Inner-Region Consensus Regression

**Context:** Real Carlsen probing after Task 3 showed that `left_upper_inner`
recovers the visible person face on previously missing frames, but some early
frames still select the background/plaque because two broader upper-left crops
agree on the distractor while only one inner crop sees the person face. The
durable fix must give the inner player-camera crop its own deterministic
cross-region evidence without weakening the consensus precedence rule globally.

**Files:**
- Modify: `src/chess_gaze/face_observation.py`
- Modify: `tests/chess_gaze/test_face_observation.py`
- Modify: `tests/chess_gaze/test_face_observation_region_arbitration.py`
- Modify: `.superpowers/sdd/task-3-report.md`

**Interfaces:**
- Consumes: deterministic detection region list from `_detection_regions(...)`.
- Produces: finite paired upper-left inner crops that allow the visible person
  face to form overlap consensus inside the player-camera pane without
  frame-specific coordinates.

- [x] **Step 1: Write failing unit regression**

Add a fake MediaPipe sequence where full frame misses, broad `left_top` and
`left_upper_band` agree on a larger false positive, `left_upper_inner` detects
the real face, and a second nearby upper-left inner crop also detects the same
real face. The expected selected candidate is the inner consensus face, not the
broad-crop false positive.

- [x] **Step 2: Run RED**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_inner_consensus_over_broad_upper_false_positive -q
```

Expected before implementation: FAIL because the second inner consensus crop
does not exist, so the broad upper-left false-positive cluster still wins.

- [x] **Step 3: Add paired deterministic inner crop**

Add a second upper-left inner crop using frame fractions rather than video-
specific coordinates. Keep all existing regions and preserve crop-to-full-frame
coordinate conversion.

- [x] **Step 4: Run focused tests and real probe**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_region_arbitration.py -q
```

Then rerun the bounded Carlsen probe on the reported frames and record whether
all selected boxes are on the visible person face.

- [x] **Step 5: Commit inner consensus fix**

```sh
git add src/chess_gaze/face_observation.py tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_region_arbitration.py .superpowers/sdd/task-3-report.md docs/superpowers/plans/2026-06-29-carlsen-face-arbitration-repair.md
git commit -m "fix: add paired inner face consensus region"
```

### Task 4: Real-Video Verification And Closeout

**Files:**
- Modify: `tests/chess_gaze/test_face_observation_real_video.py`
- Create: `docs/superpowers/closeouts/2026-06-29-carlsen-face-arbitration-repair.md`

**Interfaces:**
- Consumes: `artifacts/input/carlsen_1.mp4`, `artifacts/input/nakamura_short.mp4`, local model assets under `models/`.
- Produces: fresh verification evidence for reported frames and Nakamura short.

- [ ] **Step 1: Add bounded real-video regression**

Add a real-video test that samples the reported Carlsen frames and adjacent controls, asserts visible-face frames select the person face region rather than the plaque/background region, and asserts the previously missing visible frames recover a face.

- [ ] **Step 2: Run RED/GREEN evidence as applicable**

Run the new test before and after the production fix if it was not added before implementation. If the test is added after the minimal unit fix because the local MediaPipe sandbox was blocked, record that exception in the closeout.

- [ ] **Step 3: Run real Nakamura short verification**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_recovers_nakamura_short_visible_faces tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_nakamura_short_faces_bounded -q
```

Expected: both pass with current `NAKAMURA_SHORT_EXPECTED_FACE_BOXES` or a justified fixture update if the source video checksum changed.

- [ ] **Step 4: Regenerate and inspect artifacts**

Run `chess-gaze analyze artifacts/input/nakamura_short.mp4` and a bounded Carlsen verification path. If full `carlsen_1.mp4` regeneration is too expensive for this turn, run direct observer probes on all reported frames and create contact sheets from generated/available processed frames. Record the exact command and reason.

- [ ] **Step 5: Write closeout and commit**

Record root cause, durable surface changed, regression coverage, real-video evidence, third-party docs checked, commands run, and remaining limitations.

```sh
git add tests/chess_gaze/test_face_observation_real_video.py docs/superpowers/closeouts/2026-06-29-carlsen-face-arbitration-repair.md
git commit -m "docs: close out carlsen face arbitration repair"
```

### Task 5: Final Review And Gates

**Files:**
- Inspect branch diff and all modified files.

**Interfaces:**
- Consumes: committed task diffs.
- Produces: review evidence and final gate output.

- [ ] **Step 1: Request code review**

Dispatch a reviewer subagent over the branch diff and fix Critical/Important findings.

- [ ] **Step 2: Run broad gates**

Run:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run ruff check .
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run ruff format --check .
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run mypy
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_region_arbitration.py tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_qa_summary.py -q
```

Expected: all pass, or exact failures are recorded in the closeout.

- [ ] **Step 3: Commit final fixes if needed**

Commit any review or gate fixes with a narrow subject line.
