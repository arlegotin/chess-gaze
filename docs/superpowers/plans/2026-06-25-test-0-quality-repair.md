# Test 0 Quality Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve default real-model analysis quality on `artifacts/input/test_0.mp4` by fixing the face-detection, head-pose, and gaze-arbitration seams that produced missing visible faces/eyes and wrong-direction arrows.

**Architecture:** Keep the default frame-independent pipeline, but make each frame more robust: MediaPipe full-frame detection remains primary, deterministic split-frame retries recover scale/context misses, MediaPipe facial transformation matrices provide head-pose angles when available while PnP remains recorded evidence, and recommended gaze is emitted only when estimators agree within a defensible angular threshold or when appearance gaze is the only valid estimator. Regression tests cover pure seams plus bounded `test_0.mp4` samples.

**Tech Stack:** Python 3.12, uv-managed package, pytest, MediaPipe Face Landmarker, OpenCV headless, PyAV, NumPy, PyTorch/UniGaze.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Use installed Superpowers skills for development flow.
- Work only on the current branch.
- Use `artifacts/input/test_0.mp4` only for real-video tests and smoke checks in this repair.
- Do not run `test_1.mp4` or `test_2.mp4` verification for this task.
- Do not introduce temporal smoothing, tracking, interpolation, or cross-frame averaging.
- Do not download models or use network access during analysis.
- Keep local videos, model binaries, and generated artifacts ignored.
- Preserve every decoded source frame; no sampling in production analysis.
- Keep MediaPipe Face Landmarker in `IMAGE` mode.
- Keep UniGaze `pred_gaze[:, 0]` as pitch radians and `pred_gaze[:, 1]` as yaw radians.
- Keep `target_image_px`, `target_board_norm`, and `target_square` null in this implementation.
- Standard gates after implementation are focused pytest, `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy`, with exact failures reported if sandbox/network blocks occur.

---

### Task 1: Face Detection Split-Frame Recovery

**Files:**
- Modify: `src/chess_gaze/face_observation.py`
- Test: `tests/chess_gaze/test_face_observation.py`
- Test: `tests/chess_gaze/test_face_observation_real_video.py`

**Interfaces:**
- Produces: `DETECTION_REGION_FULL_FRAME`, `DETECTION_REGION_LEFT_HALF`, `DETECTION_REGION_RIGHT_HALF`
- Preserves: `MediaPipeFaceObserver.observe(rgb_frame, frame_id=...) -> FaceObservation`

- [ ] Write a failing unit test proving `MediaPipeFaceObserver.observe()` retries deterministic left/right half-frame regions when full-frame detection returns no candidates.
- [ ] The test must verify recovered candidate coordinates are translated back to source `image_px` and `image_norm` spaces.
- [ ] Run `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py -q` and verify RED.
- [ ] Implement full-frame primary detection plus left-half and right-half fallback detection inside `MediaPipeFaceObserver`.
- [ ] Preserve source image dimensions in recovered `FaceCandidate` records.
- [ ] Add a bounded real `test_0.mp4` regression asserting frames `f000000080`, `f000000217`, `f000000247`, and `f000000258` produce selected faces.
- [ ] Run the focused face tests and verify GREEN.
- [ ] Commit with message `fix: recover test0 split-frame faces`.

### Task 2: Head Pose From MediaPipe Transform

**Files:**
- Modify: `src/chess_gaze/head_pose.py`
- Test: `tests/chess_gaze/test_head_pose.py`
- Test: `tests/chess_gaze/test_head_pose_real_video.py`

**Interfaces:**
- Preserves: `estimate_head_pose(face, calibration, image_size) -> HeadPoseObservation`
- Produces: valid yaw/pitch/roll from finite MediaPipe facial transformation matrix when available.

- [ ] Write a failing unit test where `solvePnP` reprojection exceeds the current threshold but a finite MediaPipe facial transformation matrix is present; expected result is valid head pose with transform-derived yaw/pitch/roll and preserved PnP reprojection evidence.
- [ ] Write a failing unit test that finite transform angles stay near the matrix orientation and do not wrap to near `+/-pi`.
- [ ] Run `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_head_pose.py -q` and verify RED.
- [ ] Implement transform-derived head-pose validity as the primary path when MediaPipe provides a finite 4x4 facial transformation matrix.
- [ ] Continue running PnP to capture point count, reprojection error, and threshold evidence; if transform is absent, keep the existing PnP validity gate.
- [ ] Add a bounded real `test_0.mp4` regression asserting frames `f000000000`, `f000000090`, `f000000144`, `f000000155`, and `f000000289` yield valid finite head-pose angles with absolute pitch below `1.0` radian.
- [ ] Run focused head-pose tests and verify GREEN.
- [ ] Commit with message `fix: use mediapipe transform for head pose`.

### Task 3: Gaze Arbitration And Error Reasons

**Files:**
- Modify: `src/chess_gaze/gaze_observation.py`
- Modify: `src/chess_gaze/frame_observation.py`
- Test: `tests/chess_gaze/test_gaze_observation.py`
- Test: `tests/chess_gaze/test_frame_observation.py`

**Interfaces:**
- Preserves: `synthesize_recommended_gaze(left, right, face, thresholds=...) -> RecommendedGaze`
- Produces: accurate invalid reasons and stricter estimator agreement.

- [ ] Write a failing unit test where UniGaze is valid but both geometric eyes are invalid from `HEAD_POSE_INVALID`; expected recommended gaze is valid from appearance-only, not `GAZE_MODEL_FAILED`.
- [ ] Write a failing unit test where all estimators are valid but one differs by more than the threshold; expected `GAZE_ESTIMATORS_DISAGREE`.
- [ ] Run focused gaze tests and verify RED.
- [ ] Lower the default recommended-gaze pairwise threshold from `math.pi` to a documented non-permissive value.
- [ ] Update synthesis to use appearance-only when geometric evidence is unavailable for a non-UniGaze reason, and to return the first missing non-model reason when no valid estimator is available.
- [ ] Run focused gaze/frame tests and verify GREEN.
- [ ] Commit with message `fix: tighten gaze synthesis`.

### Task 4: Test 0 End-To-End Verification

**Files:**
- Test: `tests/chess_gaze/test_pipeline_real_video_contract.py`
- Create: `docs/superpowers/closeouts/2026-06-25-test-0-quality-repair.md`

**Interfaces:**
- Preserves public CLI command `chess-gaze analyze`.

- [ ] Add or update bounded `test_0.mp4` real tests so they exercise the default model-backed observer on representative failing frames without processing other videos.
- [ ] Run the default CLI on full `artifacts/input/test_0.mp4` only, writing to a temporary output root.
- [ ] Rebuild/inspect the QA summary and compare against the failed run: face-present rate must improve from 0.7767 to at least 0.95; head-pose-valid rate must improve from 0.2667 to at least 0.90; recommended-gaze-valid rate must improve materially and must not accept large estimator disagreements.
- [ ] Generate contact sheets from the new processed frames and inspect representative old-failure frames visually.
- [ ] Run full local gates.
- [ ] Request subagent code review and fix Critical/Important findings.
- [ ] Write closeout with root cause, changed durable surfaces, regression evidence, visual QA notes, test outputs, and residual limitations.
- [ ] Commit with message `test: verify test0 real-model quality`.
