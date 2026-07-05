# Gaze Precision Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve default gaze precision by correcting UniGaze preprocessing, then add benchmarkable target-plane and affine-calibration surfaces without making unsupported accuracy claims.

**Architecture:** Keep frame records stable. Make UniGaze preprocessing an explicit calibration contract with a legacy A/B profile and a reference-like default. Add target-plane and affine-calibration math as tested precision surfaces that require explicit calibration data before they can affect point-of-gaze claims.

**Tech Stack:** Python 3.12, uv-managed package, pytest, Pydantic, NumPy, OpenCV, MediaPipe, PyTorch/UniGaze, Three.js viewer artifacts.

## Global Constraints

- `AGENTS.md` is the governing repo contract.
- Superpowers development flow, TDD, systematic debugging, subagents, and verification-before-completion apply.
- Do not change UniGaze checkpoint or core dependency versions in this plan.
- Do not invent UniGaze confidence.
- Preserve default MPS runtime semantics: device `mps`, batch size `7`.
- Use `.venv/bin/python` for real MPS benchmark commands in this environment.
- Keep true point-of-gaze accuracy claims limited to datasets with labels.

---

### Task 1: Preprocessing Contract

**Files:**
- Modify: `src/chess_gaze/frame_records.py`
- Modify: `src/chess_gaze/calibration.py`
- Modify: `src/chess_gaze/configuration.py`
- Modify: `src/chess_gaze/cli.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/gaze_observation.py`
- Modify: `src/chess_gaze/frame_observation.py`
- Test: `tests/chess_gaze/test_calibration.py`
- Test: `tests/chess_gaze/test_configuration.py`
- Test: `tests/chess_gaze/test_cli.py`
- Test: `tests/chess_gaze/test_gaze_observation.py`
- Test: `tests/chess_gaze/test_frame_observation.py`

**Interfaces:**
- Produces `unigaze_preprocessing_profile` values:
  `legacy_bbox_rgb01` and `reference_face2x_imagenet`.
- Produces `normalize_face_crop(..., profile: str, crop_scale: float, image_mean: tuple[float, float, float] | None, image_std: tuple[float, float, float] | None)`.
- Preserves `FrameRecord` gaze fields.

- [ ] Write failing tests for default config/calibration using `reference_face2x_imagenet`.
- [ ] Write failing tests for explicit legacy profile preserving current tensor values.
- [ ] Write failing tests for default profile expanding the crop by `2.0` and applying ImageNet normalization.
- [ ] Write failing CLI/pipeline tests proving `--unigaze-preprocessing-profile` reaches `AnalyzeRequest`.
- [ ] Implement config, calibration, normalizer, and observer plumbing.
- [ ] Run focused tests and commit.

### Task 2: Target-Plane Geometry

**Files:**
- Modify: `src/chess_gaze/scene_records.py`
- Modify: `src/chess_gaze/scene_geometry.py`
- Modify: `src/chess_gaze/scene_artifacts.py`
- Test: `tests/chess_gaze/test_scene_records.py`
- Test: `tests/chess_gaze/test_scene_geometry.py`
- Test: `tests/chess_gaze/test_scene_artifacts.py`

**Interfaces:**
- Produces `ConfiguredTargetPlane` geometry helper.
- Produces `intersect_ray_with_target_plane(origin_camera_m, direction_camera, plane)`.
- Adds optional scene manifest/viewer-data target-plane records only when configured.

- [ ] Write failing tests for ray-plane intersection, parallel rays, behind-origin hits, and normalized plane coordinates.
- [ ] Write failing tests proving no target-plane hit is emitted when no target plane is configured.
- [ ] Implement target-plane records and geometry.
- [ ] Thread optional configured plane through scene artifact generation without changing default sphere behavior.
- [ ] Run focused scene tests and commit.

### Task 3: Affine Gaze Calibrator

**Files:**
- Create: `src/chess_gaze/gaze_calibration.py`
- Test: `tests/chess_gaze/test_gaze_calibration.py`

**Interfaces:**
- Produces `GazeCalibrationSample`.
- Produces `fit_affine_gaze_calibrator(samples, ridge_lambda=...)`.
- Produces `evaluate_affine_gaze_calibrator(model, samples)`.

- [ ] Write failing tests for exact affine recovery on synthetic samples.
- [ ] Write failing tests proving held-out error is reported separately from training error.
- [ ] Write failing tests for insufficient samples and non-finite inputs.
- [ ] Implement ridge affine fitting with NumPy.
- [ ] Run focused tests and commit.

### Task 4: Precision Benchmark Report

**Files:**
- Create: `src/chess_gaze/gaze_precision_benchmark.py`
- Modify: `src/chess_gaze/cli.py`
- Test: `tests/chess_gaze/test_gaze_precision_benchmark.py`
- Test: `tests/chess_gaze/test_cli.py`

**Interfaces:**
- Produces `PrecisionRunMetrics`.
- Produces `compare_precision_runs(baseline_run_dir, candidate_run_dir)`.
- Adds CLI `chess-gaze benchmark-precision <baseline-run-dir> <candidate-run-dir>`.

- [ ] Write failing tests for metric extraction from tiny synthetic run artifacts.
- [ ] Write failing tests for CLI request/printing contract.
- [ ] Implement JSON report with valid rates, sphere-hit counts, yaw/pitch summaries, frame-to-frame angular stability, and explicit `ground_truth_accuracy_available=false` unless labels are supplied.
- [ ] Run focused tests and commit.

### Task 5: Real-Video Benchmark And Docs

**Files:**
- Modify: `docs/development/architecture/source-layout.md`
- Create: `docs/development/decisions/0007-gaze-precision-preprocessing-and-calibration.md`
- Create: `docs/superpowers/closeouts/2026-07-05-gaze-precision-improvement.md`

**Interfaces:**
- Produces benchmark numbers for:
  `legacy_bbox_rgb01` vs `reference_face2x_imagenet` on `nakamura_short.mp4`.

- [ ] Run legacy real-video benchmark with MPS:
  `.venv/bin/python -c "from chess_gaze.cli import main; raise SystemExit(main(['analyze','artifacts/input/nakamura_short.mp4','--no-resume','--progress','off','--qa-summary','--unigaze-preprocessing-profile','legacy_bbox_rgb01']))"`.
- [ ] Run default real-video benchmark with MPS:
  `.venv/bin/python -c "from chess_gaze.cli import main; raise SystemExit(main(['analyze','artifacts/input/nakamura_short.mp4','--no-resume','--progress','off','--qa-summary']))"`.
- [ ] Run `chess-gaze benchmark-precision` on the two runs and save/report the JSON.
- [ ] Run focused and broad verification gates.
- [ ] Request final code review, fix Critical/Important findings, and commit closeout.
