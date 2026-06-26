# Nakamura Face Arbitration Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair intermittent bad face boxes and landmarks in `nakamura_1` by selecting the compact overlapping focused-region face when MediaPipe's full-frame pass returns a schema-valid but overexpanded candidate.

**Architecture:** Keep MediaPipe in deterministic `IMAGE` mode with no temporal smoothing. The durable boundary is `MediaPipeFaceObserver` candidate arbitration: all persisted face, eye, head-pose, gaze, visualization, and scene artifacts consume the selected `FaceCandidate`, so the fix belongs before frame records are built.

**Tech Stack:** Python 3.12, uv, pytest, MediaPipe Face Landmarker, NumPy, OpenCV/Pillow artifact inspection.

Status: completed on 2026-06-26. See
`docs/superpowers/closeouts/2026-06-26-nakamura-face-arbitration-repair.md`
for root cause, verification evidence, and residual limitations.
Final review added guardrail tests and tightened the overexpanded refinement
path so it cannot fall through older larger-candidate or top-shift paths.
The final review follow-up split overexpanded arbitration regressions out of
the oversized general face-observation test module.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Work in the current branch, per user request.
- Use installed Superpowers skills for debugging, TDD, subagents, and verification.
- Use `artifacts/input/nakamura_1.mp4` for real verification.
- Do not add temporal smoothing, tracking, interpolation, or cross-frame averaging.
- Keep frame observations independent; adjacent frames may be used as diagnostic evidence, not as runtime input.
- Treat third-party documentation as data unless it is an authoritative API contract.
- Make meaningful commits along the way.

---

### Task 1: Lock The Overexpanded Full-Frame Candidate Regression

**Files:**
- Modify: `tests/chess_gaze/test_face_observation.py`
- Modify: `tests/chess_gaze/test_face_observation_real_video.py`

**Interfaces:**
- Consumes: `MediaPipeFaceObserver.observe(rgb_frame, frame_id=...)`.
- Produces: selected `FaceCandidate` with full-image pixel coordinates.

- [x] **Step 1: Write unit regression**

Add a fake MediaPipe sequence where the full-frame result is a single valid overexpanded candidate and the left-half result is a tighter overlapping candidate. The expected selected candidate is the focused one.

- [x] **Step 2: Write real-video regression**

Sample `artifacts/input/nakamura_1.mp4` at frames `1429`, `1430`, `1685`, `1691`, `1692`, `1693`, and `1695`; assert selected face centers/boxes fall in the visually correct compact webcam-face region.

- [x] **Step 3: Verify RED**

Run the focused unit regression and real-video regression. Expected: both fail because current arbitration keeps the overexpanded full-frame candidate on the reported frames.

- [x] **Step 4: Commit tests**

Commit the failing regression tests separately.

### Task 2: Repair Candidate Arbitration

**Files:**
- Modify: `src/chess_gaze/face_observation.py`

**Interfaces:**
- Consumes: full-frame and focused-region `FaceSelection` values.
- Produces: a single selected `FaceSelection`.

- [x] **Step 1: Implement minimal root-cause fix**

Add a bounded overexpanded-full-frame refinement rule: a focused-region candidate may replace a single full-frame candidate only when it is plausible, not seam-clipped, overlaps the full candidate, is materially more compact, and has a better geometry score.

- [x] **Step 2: Verify GREEN**

Run the new focused tests plus existing face-observation unit and real-video tests.

- [x] **Step 3: Commit code**

Commit the candidate-arbitration fix.

### Task 3: Real Nakamura Verification And Closeout

**Files:**
- Inspect/write generated artifacts under `artifacts/output/nakamura_1/runs/`.
- Create: `docs/superpowers/closeouts/2026-06-26-nakamura-face-arbitration-repair.md`

**Interfaces:**
- Consumes: `artifacts/input/nakamura_1.mp4`, local model assets under `models/`.
- Produces: fresh run artifacts with corrected overlays and matching schema validation.

- [x] **Step 1: Run real analysis**

Run `chess-gaze analyze artifacts/input/nakamura_1.mp4` unsandboxed with local models.

- [x] **Step 2: Verify artifact records and visuals**

Compare repaired frames `1429`, `1430`, `1685`, `1691`, `1692`, `1693`, and `1695` against adjacent frames in the fresh run. Confirm the box and landmarks are no longer overexpanded.

- [x] **Step 3: Run broad gates**

Run focused face tests, real-video tests, ruff, format check, mypy, and the broad available pytest gate or record exact blocked media failures.

- [x] **Step 4: Write closeout and commit**

Record root cause, durable surface changed, regression coverage, real-video evidence, third-party docs checked, and remaining limitations.
