# Gaze Precision Hypothesis Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate H0-H10 in order using only `*_short.mp4` real-video inputs, retaining only independently verified correctness fixes or honest observable wins.

**Architecture:** Extend the existing run artifacts and gaze comparator so every paired experiment proves source, timestamp, model, and non-variable identity. Use the current UniGaze, calibration, face-observation, and scene modules for retained behavior; use one ignored temporary probe for H3-H10 and delete it before closeout.

**Tech Stack:** Python 3.12, PyAV, MediaPipe 0.10.35, NumPy 2.5, OpenCV 4.13, PyTorch 2.12.1/MPS, UniGaze 0.1.3, Pydantic 2.13, pytest, Ruff, mypy, Git.

## Global Constraints

- `AGENTS.md` and the approved design at `docs/superpowers/specs/2026-07-12-gaze-precision-hypothesis-evaluation-design.md` govern this plan.
- Work on the current `improvements-1` branch. Do not create a worktree or switch branches.
- Empirical inputs are discovered only with `artifacts/input/*_short.mp4`; the approved SHA-256 values are mandatory.
- Tests may use arrays or generated fixtures whose filenames end in `_short.mp4`.
- Change one declared experimental variable at a time and run baseline/candidate from the same source tree.
- A stability, consistency, coverage, or runtime proxy never establishes gaze accuracy.
- Preserve raw gaze. Do not invent UniGaze confidence or attention labels.
- Use current NumPy/OpenCV/MediaPipe before adding code and add no dependency for these experiments.
- Use `.venv/bin/python` or `.venv/bin/chess-gaze` outside the sandbox for MPS inference; use `UV_CACHE_DIR=.uv-cache uv run` for ordinary checks.
- Every production change follows observed RED, minimal GREEN, focused verification, and a meaningful commit.
- Stop a hypothesis after three failed repair attempts and reassess its architecture.
- Failed, blocked, and inconclusive mechanisms leave no production helper, toggle, schema field, dependency, or abstraction.
- Generated runs and the temporary probe stay ignored. Only summarized evidence enters the closeout.

---

## File Ownership Map

- `src/chess_gaze/frame_records.py`: persisted video/model provenance and strict artifact schemas.
- `src/chess_gaze/video_decode.py`: decoded frame identity, PTS usability, and orientation metadata.
- `src/chess_gaze/unigaze_runtime.py`: verified UniGaze asset/runtime provenance.
- `src/chess_gaze/gaze_precision_benchmark.py`: run comparability, proxy metrics, and report CLI.
- `src/chess_gaze/scene_records.py` and `scene_artifacts.py`: finite target-plane summary semantics.
- `src/chess_gaze/unigaze_preprocessing.py`, `gaze_observation.py`, and `frame_observation.py`: H1 geometric input/output contract if its prerequisite gate passes.
- `src/chess_gaze/gaze_calibration.py`: H2 numerical correctness only.
- `src/chess_gaze/face_observation.py` and conditional `face_selection.py`: H3 identity behavior if a variant passes.
- `src/chess_gaze/pipeline.py` and `analysis_resume.py`: temporal-state resume boundary if H3 passes.
- `artifacts/experiments/2026-07-12-gaze-precision/probe.py`: ignored H3-H10 probe; delete before final commit.
- `docs/superpowers/closeouts/2026-07-12-gaze-precision-hypothesis-evaluation.md`: final H0-H10 evidence ledger.

### Task 1: Persist Trustworthy Timestamp And Model Provenance

**Files:**
- Modify: `src/chess_gaze/frame_records.py`
- Modify: `src/chess_gaze/video_decode.py`
- Modify: `src/chess_gaze/unigaze_runtime.py`
- Test: `tests/chess_gaze/test_frame_records.py`
- Test: `tests/chess_gaze/test_video_decode.py`
- Test: `tests/chess_gaze/test_unigaze_runtime.py`
- Test: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Produces `VideoManifest.pts_sequence_sha256: str | None`.
- Produces `VideoManifest.pts_sequence_usable: bool`.
- Produces `InferenceRuntimeRecord.unigaze_model_checksum_sha256: str | None`.
- Produces `_decoded_pts_identity(frames: Iterable[av.VideoFrame]) -> tuple[int, str, bool]`.

- [ ] **Step 1: Add failing timestamp provenance tests**

Add tests that inspect a generated `tiny_short.mp4` twice and assert the same 64-character hash and `pts_sequence_usable is True`. Directly test `_decoded_pts_identity` with fake frames containing a missing PTS, duplicate PTS, decreasing PTS, and non-positive time base; each must return `usable is False`.

```python
inspection = inspect_video(path)
second = inspect_video(path)
assert inspection.video_manifest.pts_sequence_usable is True
assert len(inspection.video_manifest.pts_sequence_sha256 or "") == 64
assert second.video_manifest.pts_sequence_sha256 == (
    inspection.video_manifest.pts_sequence_sha256
)
```

- [ ] **Step 2: Add failing model checksum and legacy-read tests**

Assert `prepare_unigaze_runtime()` copies `_asset(tmp_path).checksum_sha256` into its inference record. Assert old `VideoManifest` and `InferenceRuntimeRecord` payloads without the new fields still parse as `None`/`False`, and external/legacy observer records reject a non-null model checksum.

```python
assert prepared.inference.unigaze_model_checksum_sha256 == "abc123"
assert legacy_video.pts_sequence_sha256 is None
assert legacy_video.pts_sequence_usable is False
```

- [ ] **Step 3: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_video_decode.py \
  tests/chess_gaze/test_unigaze_runtime.py \
  tests/chess_gaze/test_frame_records.py -q
```

Expected: failures name the three missing schema attributes and `_decoded_pts_identity`.

- [ ] **Step 4: Implement streamed PTS identity**

Add compatible defaults to `VideoManifest`, then hash one canonical row per decoded frame without retaining frames:

```python
def _decoded_pts_identity(
    frames: Iterable[av.VideoFrame],
) -> tuple[int, str, bool]:
    digest = hashlib.sha256()
    count = 0
    usable = True
    previous_seconds: float | None = None
    for frame in frames:
        count += 1
        time_base = frame.time_base
        time_base_text = (
            "null"
            if time_base is None
            else f"{time_base.numerator}/{time_base.denominator}"
        )
        digest.update(f"{frame.pts}\t{time_base_text}\n".encode())
        seconds = _frame_pts_seconds(frame)
        if (
            seconds is None
            or not math.isfinite(seconds)
            or previous_seconds is not None
            and seconds <= previous_seconds
        ):
            usable = False
        previous_seconds = seconds
    return count, digest.hexdigest(), usable and count > 0
```

Use its count/hash/flag in `inspect_video()`. Add `import hashlib`, `import math`, and `Iterable`; do not add another manifest.

- [ ] **Step 5: Persist the verified UniGaze checksum**

Add `unigaze_model_checksum_sha256: str | None = None` to `InferenceRuntimeRecord`. Set `asset.checksum_sha256` in `prepare_unigaze_runtime()`, set `None` in external/legacy records, and extend the existing validator so external/legacy records cannot claim a checksum. Permit `None` on old default-model artifacts; Task 3 rejects such runs for new comparisons.

- [ ] **Step 6: Run GREEN and schema regressions**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_video_decode.py \
  tests/chess_gaze/test_frame_records.py \
  tests/chess_gaze/test_unigaze_runtime.py \
  tests/chess_gaze/test_pipeline_contract.py -q
```

Expected: all tests pass.

### Task 2: Separate Finite-Plane Hits From Infinite Intersections

**Files:**
- Modify: `src/chess_gaze/scene_records.py`
- Modify: `src/chess_gaze/scene_artifacts.py`
- Test: `tests/chess_gaze/test_scene_artifacts.py`
- Test: `tests/chess_gaze/test_scene_records.py`

**Interfaces:**
- Produces `SceneSummary.in_bounds_target_plane_hit_frames: int | None`.
- Preserves `valid_target_plane_hit_frames` as all valid mathematical intersections.

- [ ] **Step 1: Write the failing aggregation regression**

Create two valid target-plane hits, one with `inside_bounds=True` and one with `inside_bounds=False`, and assert:

```python
assert summary.valid_target_plane_hit_frames == 2
assert summary.in_bounds_target_plane_hit_frames == 1
```

- [ ] **Step 2: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_scene_artifacts.py::test_build_scene_artifacts_counts_in_bounds_target_plane_hits_separately -q
```

Expected: `SceneSummary` has no `in_bounds_target_plane_hit_frames` field.

- [ ] **Step 3: Add the compatible summary field and count**

Use `None` for legacy summaries and an integer for every fresh summary:

```python
in_bounds_target_plane_hit_frames=sum(
    1
    for frame in scene_frames
    if frame.target_plane_hit is not None
    and frame.target_plane_hit.valid
    and frame.target_plane_hit.inside_bounds is True
)
```

- [ ] **Step 4: Run GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_scene_artifacts.py \
  tests/chess_gaze/test_scene_records.py \
  tests/chess_gaze/test_scene_viewer.py \
  tests/chess_gaze/test_qa_summary.py -q
```

Expected: all tests pass.

### Task 3: Make Gaze Comparisons Provenance-Safe And Time-Normalized

**Files:**
- Modify: `src/chess_gaze/gaze_precision_benchmark.py`
- Modify: `tests/chess_gaze/test_gaze_precision_benchmark.py`

**Interfaces:**
- Produces `GazePrecisionExperimentalVariable = Literal["unigaze_preprocessing"]` initially.
- Produces `compare_gaze_precision_runs(..., experimental_variable: GazePrecisionExperimentalVariable, generated_at_utc: datetime | None = None)`.
- Produces v2 report fields for source/model/PTS identity, in-bounds hits, and ray speed.

- [ ] **Step 1: Expand the synthetic run fixture with complete manifests**

Make `_write_run()` write `run_manifest.json`, `video_manifest.json`, calibration, frames, and scene summary. Give default-model runs a non-null checksum and usable PTS hash. Retain knobs for source SHA, dimensions, timestamps, checksum, preprocessing profile, and one unrelated calibration field.

- [ ] **Step 2: Write failing identity and declared-variable tests**

Cover different source SHA, dimensions/count, PTS hash, frame/timestamp sequence, model checksum, runtime setting, unrelated calibration setting, unknown variable, and no actual declared difference. Each must raise `ValueError` naming the mismatched sorted field path. A preprocessing-only difference must pass.

```python
with pytest.raises(ValueError, match="inference.unigaze_model_checksum_sha256"):
    compare_gaze_precision_runs(
        baseline,
        candidate,
        experimental_variable="unigaze_preprocessing",
    )
```

- [ ] **Step 3: Write failing speed and finite-hit tests**

Assert a `0.1` radian step over `0.5` seconds equals
`math.degrees(0.1) / 0.5`. Assert unusable PTS yields `None` for all speed
metrics. Assert intersection and in-bounds counts remain distinct.

- [ ] **Step 4: Write the failing module-CLI test**

Invoke `main()` without `--experimental-variable` and expect argparse usage failure. Invoke it with `--experimental-variable unigaze_preprocessing` and assert the JSON report carries that value.

- [ ] **Step 5: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_gaze_precision_benchmark.py -q
```

Expected: failures show the missing required argument, absent provenance checks, and absent speed/in-bounds fields.

- [ ] **Step 6: Implement the exact allowlist and comparison boundary**

```python
EXPERIMENTAL_VARIABLE_FIELDS = {
    "unigaze_preprocessing": frozenset(
        {
            "calibration.unigaze_preprocessing_profile",
            "calibration.unigaze_face_crop_scale",
            "calibration.unigaze_image_mean_rgb",
            "calibration.unigaze_image_std_rgb",
        }
    )
}
```

Load the run manifest with `read_run_manifest_artifact_json()`. Require each
embedded video manifest to equal its standalone manifest, exact source/dimension/
count/PTS/model identity, contiguous identical frame IDs/indices, exact timestamp
sequences, and equality of retention, QA, runtime, and calibration fields after
removing only the declared allowlist. Ignore only volatile run ID, creation time,
and run path. Reject a missing model checksum or a declared group with no actual
difference.

- [ ] **Step 7: Implement speed metrics and v2 report fields**

```python
speed = math.degrees(_angular_distance(previous, current)) / delta_seconds
```

Reset across invalid gaze. Return `None` when `pts_sequence_usable` is false;
raise if a run claiming usable PTS contains a non-positive/non-finite timestamp
delta. Preserve radians/frame metrics as descriptive fields.

- [ ] **Step 8: Run GREEN and focused H0 gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_frame_records.py \
  tests/chess_gaze/test_video_decode.py \
  tests/chess_gaze/test_unigaze_runtime.py \
  tests/chess_gaze/test_pipeline_contract.py \
  tests/chess_gaze/test_analysis_resume.py \
  tests/chess_gaze/test_scene_records.py \
  tests/chess_gaze/test_scene_artifacts.py \
  tests/chess_gaze/test_scene_viewer.py \
  tests/chess_gaze/test_qa_summary.py \
  tests/chess_gaze/test_gaze_precision_benchmark.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit H0**

```sh
git add src/chess_gaze/frame_records.py src/chess_gaze/video_decode.py \
  src/chess_gaze/unigaze_runtime.py src/chess_gaze/scene_records.py \
  src/chess_gaze/scene_artifacts.py \
  src/chess_gaze/gaze_precision_benchmark.py \
  tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_video_decode.py \
  tests/chess_gaze/test_unigaze_runtime.py \
  tests/chess_gaze/test_pipeline_contract.py \
  tests/chess_gaze/test_scene_records.py \
  tests/chess_gaze/test_scene_artifacts.py \
  tests/chess_gaze/test_gaze_precision_benchmark.py
git commit -m "fix: make gaze comparisons provenance-safe"
```

### Task 4: Repair Short-Input Drift And Record H0 Baselines

**Files:**
- Modify: `tests/chess_gaze/test_video_decode_real_video.py`
- Modify: `tests/chess_gaze/test_pipeline_real_video_contract.py`
- Modify: `tests/chess_gaze/test_qa_summary_real_video_contract.py`
- Produce ignored runs under: `artifacts/output/gaze-hypotheses/h0/`

**Interfaces:**
- Produces fresh H0 reference and legacy run directories for all three approved inputs.
- Produces valid `unigaze_preprocessing` comparison reports for Task 5.

- [ ] **Step 1: Reproduce the stale 180-frame expectation**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_video_decode_real_video.py \
  tests/chess_gaze/test_pipeline_real_video_contract.py \
  tests/chess_gaze/test_qa_summary_real_video_contract.py -q
```

Expected: the current 1,200-frame `nakamura_short.mp4` fails against the three
hard-coded 180-frame expectations.

- [ ] **Step 2: Update only the evidenced constants**

Set `NAKAMURA_SHORT_FRAME_COUNT = 1200` in the three tests. Do not loosen the
count, replace it with self-derived output, or touch tests that use non-short
videos.

- [ ] **Step 3: Run GREEN**

Repeat Step 1. Expected: all three tests pass.

- [ ] **Step 4: Verify the allowed corpus before inference**

Run:

```sh
find artifacts/input -maxdepth 1 -name '*_short.mp4' -print | sort
.venv/bin/python -c "from pathlib import Path; from chess_gaze.video_decode import inspect_video; expected={'carlsen_short.mp4':'48505b38898a843c5b03d9cfa717efda2a915f0c5399c81369be20d316f6fc01','nakamura_short.mp4':'6524928897505e614a0eae419a1b7bd0e2a8dff25ffed22db2706d02bbf909bc','nepo_short.mp4':'aa24fb658a3a3723d8b953d01c5ddf174d60978b6a5a2312c5c79f4b23c36b8c'}; paths=sorted(Path('artifacts/input').glob('*_short.mp4')); assert {p.name for p in paths}==set(expected); [(_ for _ in ()).throw(AssertionError(p)) if inspect_video(p).source_sha256 != expected[p.name] else None for p in paths]; print('approved short corpus verified')"
```

Expected: exactly three paths and `approved short corpus verified`.

- [ ] **Step 5: Run paired H0 preprocessing controls on MPS**

Run each command outside the sandbox and capture its printed run directory:

```sh
.venv/bin/chess-gaze analyze artifacts/input/carlsen_short.mp4 --models-root models --output-root artifacts/output/gaze-hypotheses/h0/reference --unigaze-device mps --unigaze-batch-size 7 --unigaze-preprocessing-profile reference_face2x_imagenet --no-resume --progress off --qa-summary
.venv/bin/chess-gaze analyze artifacts/input/nakamura_short.mp4 --models-root models --output-root artifacts/output/gaze-hypotheses/h0/reference --unigaze-device mps --unigaze-batch-size 7 --unigaze-preprocessing-profile reference_face2x_imagenet --no-resume --progress off --qa-summary
.venv/bin/chess-gaze analyze artifacts/input/nepo_short.mp4 --models-root models --output-root artifacts/output/gaze-hypotheses/h0/reference --unigaze-device mps --unigaze-batch-size 7 --unigaze-preprocessing-profile reference_face2x_imagenet --no-resume --progress off --qa-summary
.venv/bin/chess-gaze analyze artifacts/input/carlsen_short.mp4 --models-root models --output-root artifacts/output/gaze-hypotheses/h0/legacy --unigaze-device mps --unigaze-batch-size 7 --unigaze-preprocessing-profile legacy_bbox_rgb01 --no-resume --progress off --qa-summary
.venv/bin/chess-gaze analyze artifacts/input/nakamura_short.mp4 --models-root models --output-root artifacts/output/gaze-hypotheses/h0/legacy --unigaze-device mps --unigaze-batch-size 7 --unigaze-preprocessing-profile legacy_bbox_rgb01 --no-resume --progress off --qa-summary
.venv/bin/chess-gaze analyze artifacts/input/nepo_short.mp4 --models-root models --output-root artifacts/output/gaze-hypotheses/h0/legacy --unigaze-device mps --unigaze-batch-size 7 --unigaze-preprocessing-profile legacy_bbox_rgb01 --no-resume --progress off --qa-summary
```

- [ ] **Step 6: Compare each same-source pair and inspect provenance**

For each printed reference/legacy pair, run:

```sh
.venv/bin/python -m chess_gaze.gaze_precision_benchmark REFERENCE_RUN LEGACY_RUN \
  --experimental-variable unigaze_preprocessing \
  --output artifacts/output/gaze-hypotheses/h0/CLIP-comparison.json
```

Replace `REFERENCE_RUN`, `LEGACY_RUN`, and `CLIP` with the exact printed paths
and one of `carlsen_short`, `nakamura_short`, or `nepo_short`. Expected: three
reports, non-null PTS/model hashes, `pts_sequence_usable=true`, and no comparator
error. Deliberately cross-pair Carlsen and Nakamura once; expected: `ValueError`
names source identity.

- [ ] **Step 7: Commit the real-input drift repair**

```sh
git add tests/chess_gaze/test_video_decode_real_video.py \
  tests/chess_gaze/test_pipeline_real_video_contract.py \
  tests/chess_gaze/test_qa_summary_real_video_contract.py
git commit -m "test: align short-video evidence"
```

### Task 5: Run The H1 Asset, License, And Landmark-Mapping Gate

**Files:**
- Create ignored evidence under: `artifacts/experiments/2026-07-12-gaze-precision/h1/`
- Conditionally modify later: `src/chess_gaze/model_registry.json`
- Conditionally create later: `docs/development/decisions/0007-restore-unigaze-geometric-normalization.md`

**Interfaces:**
- Produces a binary H1 prerequisite decision: `pass` or `blocked`.
- Verifies pinned upstream revision `9c240fbe33f3d6146970a77b7c8fa06a7e60019e` before trusting it.
- Verifies face-model SHA-256 `0c943d1d48627d97038b64f9a73816b9ab80a002ce81a8f04d532da2f4c337d7` before use.

- [x] **Step 1: Verify and fetch the pinned primary-source oracle outside the sandbox**

```sh
git ls-remote https://github.com/ut-vision/UniGaze.git refs/heads/main
mkdir -p artifacts/experiments/2026-07-12-gaze-precision/h1
curl -L https://raw.githubusercontent.com/ut-vision/UniGaze/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/unigaze/gazelib/gaze/normalize.py -o artifacts/experiments/2026-07-12-gaze-precision/h1/normalize.py
curl -L https://raw.githubusercontent.com/ut-vision/UniGaze/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/unigaze/predict_gaze_video.py -o artifacts/experiments/2026-07-12-gaze-precision/h1/predict_gaze_video.py
curl -L https://raw.githubusercontent.com/ut-vision/UniGaze/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/unigaze/data/face_model.txt -o artifacts/experiments/2026-07-12-gaze-precision/h1/face_model.txt
shasum -a 256 artifacts/experiments/2026-07-12-gaze-precision/h1/face_model.txt
```

Expected: `ls-remote` reports the pinned revision or a newer revision that is
recorded without silently changing the oracle; the face-model digest exactly
matches the approved value. A mismatch blocks H1.

- [x] **Step 2: Resolve code and asset licensing**

Read the pinned normalizer header, root `LICENSE.txt`, model card, and face-model
file. Record separately: normalizer code license, face-model data license,
required citation, modification notice, noncommercial restriction, and whether
the repository may redistribute the six required points. If the face-model
license is absent or contradictory, mark H1 `blocked` and skip Tasks 6-7.

- [x] **Step 3: Verify the six landmark semantics on approved frames**

The upstream order is dlib `[36,39,42,45,31,35]` and face-model rows
`[20,23,26,29,15,19]`. Test the candidate MediaPipe order
`[33,133,362,263,98,327]` by drawing labeled points on frames 0, midpoint, and
last from every approved clip. Confirm four eye corners and two nose sides in
the same image-left-to-right order. If any point/order is ambiguous, mark H1
`blocked`; do not substitute nose tip or mouth points.

- [x] **Step 4: Decide H1 before production edits**

Proceed to Task 6 only when revision, digest, license, and all six landmark
semantics pass. Otherwise write the exact blocker and evidence paths into the
future closeout notes, delete any repo edits made for H1, and continue directly
to Task 8. Keep the ignored primary-source evidence until closeout is written.

### Task 6: Implement H1 Geometric Normalization Test-First (Conditional)

**Files:**
- Modify: `src/chess_gaze/unigaze_preprocessing.py`
- Modify: `src/chess_gaze/gaze_observation.py`
- Modify: `src/chess_gaze/frame_observation.py`
- Modify: `src/chess_gaze/frame_records.py`
- Modify: `src/chess_gaze/calibration.py`
- Modify: `src/chess_gaze/configuration.py`
- Modify: `src/chess_gaze/cli.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/model_registry.json`
- Create: `tests/chess_gaze/test_unigaze_preprocessing.py`
- Create: `tests/chess_gaze/test_unigaze_preprocessing_real_video.py`
- Modify: `tests/chess_gaze/test_gaze_observation.py`
- Modify: `tests/chess_gaze/test_frame_observation.py`
- Modify: `tests/chess_gaze/test_model_assets.py`
- Modify: `tests/chess_gaze/test_configuration.py`
- Modify: `tests/chess_gaze/test_calibration.py`

**Interfaces:**
- Produces profile `official_geometric_v1` while retaining `reference_face2x_imagenet` for paired rollback.
- Produces `NormalizedFaceCrop.camera_from_normalized_rotation` and explicit homography direction.
- Produces `camera_gaze_from_unigaze_prediction(...) -> FaceModelGaze`.
- No frame may silently fall back from requested geometric normalization to bbox resize.

- [x] **Step 1: Write RED equation-oracle tests**

Using fixed numeric camera, rotation/translation, face centre, and a small RGB
array, save expected `R`, `W`, and warped pixels generated by the pinned oracle.
Assert local values match with explicit tolerances and degenerate centre/basis
raises `ValueError`.

```python
np.testing.assert_allclose(actual.normalized_from_camera_rotation, expected_r, atol=1e-10)
np.testing.assert_allclose(actual.normalized_image_from_image_homography, expected_w, atol=1e-10)
np.testing.assert_allclose(actual.warped_rgb, expected_image, atol=1)
```

- [x] **Step 2: Run equation RED**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_unigaze_preprocessing.py -q
```

Expected: missing geometric API/profile.

- [x] **Step 3: Implement the independent equation boundary**

Use the verified crop-local six landmarks and face-model points with
`cv2.solvePnP`. Build dummy `K` with focal `crop_width * 4`, normalized `K` with
focal `960`, distance `600`, and size `224`. Compute face centre as the mean of
the transformed two-eye centre and two-nose centre, then:

```python
distance = np.linalg.norm(center)
scale = np.diag((1.0, 1.0, 600.0 / distance))
forward = center.reshape(3) / distance
down = np.cross(forward, head_rotation[:, 0])
down /= np.linalg.norm(down)
right = np.cross(down, forward)
right /= np.linalg.norm(right)
normalized_from_camera = np.column_stack((right, down, forward)).T
homography = normalized_k @ scale @ normalized_from_camera @ np.linalg.inv(camera_k)
warped = cv2.warpPerspective(crop, homography, (224, 224))
```

Independently cite the pinned equations; do not paste upstream implementation
text. Validate finite shapes, non-zero norms, `solvePnP` success, and invertible
matrices.

- [x] **Step 4: Write RED output-sign and inverse-rotation tests**

Cover identity model vector `(0,0,1)`, positive physical camera-ray x ->
positive repository yaw, positive physical image-up -> positive pitch, a known
yaw rotation, the pinned oracle result, and horizontal-flip equivariance. The
tests must distinguish UniGaze's camera-space model vector from the physical
ray: the pinned training preparation labels `-gaze_norm`, and the pinned video
inference projects `-R^-1 * model_vector`. Legacy profile tests must keep the
existing behavior, which is consistent with this physical-ray convention.

- [x] **Step 5: Implement row-aligned camera-space conversion**

```python
unigaze_model_vector_camera = camera_from_normalized_rotation @ np.asarray(
    pitch_yaw_to_unit_vector(
        pitch_radians=pitch_radians,
        yaw_radians=yaw_radians,
    )
)
unigaze_model_vector_camera /= np.linalg.norm(unigaze_model_vector_camera)
camera_pitch = math.asin(float(unigaze_model_vector_camera[1]))
camera_yaw = math.atan2(
    -float(unigaze_model_vector_camera[0]),
    float(unigaze_model_vector_camera[2]),
)
```

The physical camera ray is `-unigaze_model_vector_camera`. The repository's
stored angle vector is not itself that OpenCV ray: the existing scene boundary
maps stored angles to `(x, -y, -z)`, so the yaw-only x negation above is the
required composition, not an optional display convention.

Extend `predict_batch()` with a strict row-aligned sequence of inverse rotations;
`None` selects legacy conversion. Thread each crop's own rotation through
`ModelBackedFrameObserver`; a geometric failure invalidates only that frame.

- [x] **Step 6: Persist profile, asset, and reproducibility evidence**

Register the verified face-model asset with URL, checksum, license, approval,
input contract, and output contract. Persist the selected profile plus per-frame
homography and `camera_from_normalized_rotation`, or prove in a test that the
persisted landmarks/profile/asset hash deterministically reproduce both values.
Strict fixtures must reject non-finite or wrong-shape matrices.

- [x] **Step 7: Run focused GREEN**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_unigaze_preprocessing.py \
  tests/chess_gaze/test_gaze_observation.py \
  tests/chess_gaze/test_frame_observation.py \
  tests/chess_gaze/test_model_assets.py \
  tests/chess_gaze/test_configuration.py \
  tests/chess_gaze/test_calibration.py -q
```

Expected: all tests pass.

- [x] **Step 8: Run the guarded native fixed-frame oracle**

The new native test reads only the three approved `*_short.mp4` files, asserts
their exact hashes, and checks at least one fixed frame per clip against the
pinned oracle values.

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_unigaze_preprocessing_real_video.py \
  -m native_mediapipe -q
```

Expected outside sandbox: pass. A mismatch returns to Task 5; do not tune against
proxy jitter.

### Task 7: Evaluate And Either Keep Or Remove H1 (Conditional)

**Resumed H1 status, 2026-07-13:** Tasks 5-6 and Task 7 steps 1-4 passed after
the repository owner approved `face_model.txt` under `MG-NC-RAI-2.0`. The six
exact MPS run paths, three comparator reports, QA hashes, zero-loss coverage
gate, corrected vector-sign derivation, and non-accuracy boundary are recorded
in
[ADR-0007](../../development/decisions/0007-restore-unigaze-geometric-normalization.md)
and the closeout. The default flip and historical crop-replay benchmark pin are
implemented. Task 7 step 5 passed with the fresh broad, socket, static, and
11-node approved-input native gates; retained implementation commit `af8ed2e`
landed on the current branch.

**Files:**
- Produce ignored runs under: `artifacts/output/gaze-hypotheses/h1/`
- Conditionally create: `docs/development/decisions/0007-restore-unigaze-geometric-normalization.md`
- Conditionally update: `docs/development/architecture/source-layout.md`

**Interfaces:**
- Makes `official_geometric_v1` the default only if the independent correctness gate passes.
- Leaves no H1 production code if provenance/oracle/coverage fails.

- [x] **Step 1: Run reference and official profiles over every approved clip**

Use the six explicit commands from Task 4 with output roots
`artifacts/output/gaze-hypotheses/h1/reference` and
`artifacts/output/gaze-hypotheses/h1/official`, changing only the profile to
`reference_face2x_imagenet` or `official_geometric_v1`.

- [x] **Step 2: Build three provenance-safe reports**

For each same-clip pair:

```sh
.venv/bin/python -m chess_gaze.gaze_precision_benchmark REFERENCE_RUN OFFICIAL_RUN \
  --experimental-variable unigaze_preprocessing \
  --output artifacts/output/gaze-hypotheses/h1/CLIP-comparison.json
```

- [x] **Step 3: Apply the H1 retention gate**

Keep H1 only when the pinned oracle, asset/license/mapping, synthetic signs,
native fixed frames, and all three QA summaries pass, aggregate gaze coverage
loss is at most two percentage points, and no clip loses more than five points.
Report speed/jitter descriptively. If any correctness prerequisite fails, remove
the profile, asset entry, plumbing, schema fields, tests, and ADR; rerun the H0
focused gate to prove the tree returned to a clean supported state.

- [x] **Step 4: If kept, make the corrected profile default and write ADR-0007**

The ADR records selected/rejected alternatives, upstream revision/URLs,
face-model local path/checksum/license, Apple-MPS assumptions, input/output and
matrix direction contracts, lack of confidence, the exact non-accuracy claim,
and rollback profile.

- [x] **Step 5: Verify and commit only a kept H1**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze -q -m 'not native_mediapipe and not local_socket'
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
git add src/chess_gaze/unigaze_preprocessing.py \
  src/chess_gaze/gaze_observation.py \
  src/chess_gaze/frame_observation.py src/chess_gaze/frame_records.py \
  src/chess_gaze/calibration.py \
  src/chess_gaze/cli.py src/chess_gaze/pipeline.py \
  src/chess_gaze/model_assets.py src/chess_gaze/model_registry.json \
  src/chess_gaze/gaze_precision_benchmark.py \
  src/chess_gaze/scene_artifacts.py \
  src/chess_gaze/unigaze_batch_benchmark.py \
  tests/chess_gaze/test_unigaze_preprocessing.py \
  tests/chess_gaze/test_unigaze_preprocessing_real_video.py \
  tests/chess_gaze/test_gaze_observation.py \
  tests/chess_gaze/test_gaze_observation_real_video.py \
  tests/chess_gaze/test_gaze_precision_benchmark.py \
  tests/chess_gaze/test_frame_observation.py \
  tests/chess_gaze/test_model_assets.py \
  tests/chess_gaze/test_configuration.py tests/chess_gaze/test_calibration.py \
  tests/chess_gaze/test_cli.py tests/chess_gaze/test_pipeline_contract.py \
  tests/chess_gaze/test_scene_artifacts.py \
  tests/chess_gaze/test_unigaze_batch_benchmark.py \
  README.md docs/gaze-precision-hypotheses.md \
  docs/development/decisions/0007-restore-unigaze-geometric-normalization.md \
  docs/development/architecture/source-layout.md \
  docs/superpowers/specs/2026-07-12-gaze-precision-hypothesis-evaluation-design.md \
  docs/superpowers/plans/2026-07-12-gaze-precision-hypothesis-evaluation.md \
  docs/superpowers/closeouts/2026-07-12-gaze-precision-hypothesis-evaluation.md
git commit -m "fix: restore UniGaze geometric normalization"
```

If H1 is blocked/rejected, make no H1 commit.

### Task 8: Repair H2 Calibration Numerics Without Runtime Integration

**Files:**
- Modify: `src/chess_gaze/gaze_calibration.py`
- Modify: `tests/chess_gaze/test_gaze_calibration.py`

**Interfaces:**
- Zero ridge uses `np.linalg.lstsq(..., rcond=None)`.
- Positive ridge does not penalize coefficient column zero (the intercept).
- No offset-only model, runtime calibrated field, or video fitting is added.

- [ ] **Step 1: Write the rank-deficient RED regression**

Use at least five samples with duplicate/dependent feature rows and exact finite
targets. Assert fitting and held-out prediction remain finite.

- [ ] **Step 2: Run rank-deficient RED**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_gaze_calibration.py::test_fit_affine_gaze_calibrator_handles_rank_deficient_zero_ridge -q
```

Expected: `numpy.linalg.LinAlgError: Singular matrix` from the current normal
equation solve.

- [ ] **Step 3: Implement zero-ridge least squares**

```python
if ridge_lambda == 0.0:
    coefficients_t = np.linalg.lstsq(
        design_matrix,
        targets,
        rcond=None,
    )[0]
else:
    # implemented by the next RED/GREEN cycle
```

- [ ] **Step 4: Write and run the intercept RED regression**

Use samples whose four non-intercept features are zero and whose target is
constant `(2.0, -3.0)`. Fit with a large ridge and assert exact intercept and
zero slopes.

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_gaze_calibration.py::test_ridge_does_not_penalize_affine_intercept -q
```

Expected: current ridge shrinks the intercept.

- [ ] **Step 5: Implement slope-only ridge**

```python
regularizer = np.eye(_AFFINE_FEATURE_COUNT, dtype=np.float64) * ridge_lambda
regularizer[0, 0] = 0.0
coefficients_t = np.linalg.solve(
    design_matrix.T @ design_matrix + regularizer,
    design_matrix.T @ targets,
)
```

- [ ] **Step 6: Run GREEN and the existing held-out proof**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_gaze_calibration.py -q
```

Expected: all calibration tests pass, including separate held-out evaluation.

- [ ] **Step 7: Commit the independent H2 correctness repair**

```sh
git add src/chess_gaze/gaze_calibration.py \
  tests/chess_gaze/test_gaze_calibration.py
git commit -m "fix: make gaze calibration numerically robust"
```

### Task 9: Establish H3 Face-Selection Ground Truth At Worst Events

**Files:**
- Create ignored: `artifacts/experiments/2026-07-12-gaze-precision/probe.py`
- Produce ignored: `artifacts/experiments/2026-07-12-gaze-precision/h3-review.json`
- Produce ignored contact sheets under: `artifacts/experiments/2026-07-12-gaze-precision/h3-contact-sheets/`

**Interfaces:**
- Consumes the retained H1 run when H1 is kept; otherwise consumes the H0 reference run.
- Produces manual labels `streamer`, `other_person`, `false_crop`, `missing_visible_face`, or `ambiguous`.
- Only non-ambiguous incorrect labels are H3 correctness evidence.

- [ ] **Step 1: Create the hash-guarded ignored probe**

The probe must abort unless the glob resolves exactly the three approved names
and hashes. For consecutive present faces compute:

```python
def bbox_event(previous: BBox, current: BBox, diagonal: float) -> tuple[float, float]:
    previous_center = (
        (previous.x_min + previous.x_max) / 2.0,
        (previous.y_min + previous.y_max) / 2.0,
    )
    current_center = (
        (current.x_min + current.x_max) / 2.0,
        (current.y_min + current.y_max) / 2.0,
    )
    center_step = math.dist(previous_center, current_center) / diagonal
    previous_area = (previous.x_max - previous.x_min) * (
        previous.y_max - previous.y_min
    )
    current_area = (current.x_max - current.x_min) * (
        current.y_max - current.y_min
    )
    return center_step, abs(math.log(current_area / previous_area))
```

Also compute appearance-gaze angular steps with the existing
`pitch_yaw_to_unit_vector()`. Rank the union of the top 15 centre, scale, and
gaze events per clip; deduplicate events within two frames.

- [ ] **Step 2: Add contact-sheet rendering to the temporary probe**

For every event decode frames `i-2` through `i+2`, draw the persisted selected
bbox and frame ID with Pillow, and write one sheet. This rendering is evidence
only and is deleted before final commit.

- [ ] **Step 3: Run the review probe**

```sh
.venv/bin/python artifacts/experiments/2026-07-12-gaze-precision/probe.py \
  h3-review \
  --inputs 'artifacts/input/*_short.mp4' \
  --runs-root artifacts/output/gaze-hypotheses \
  --output artifacts/experiments/2026-07-12-gaze-precision/h3-review.json
```

Expected: ranked frame IDs and contact sheets for all three clips.

- [ ] **Step 4: Label every deduplicated event before changing selection**

Add one allowed manual label and a one-sentence visual rationale per reviewed
event. `ambiguous` cannot support retention. If there is no confirmed
`other_person`, `false_crop`, or `missing_visible_face`, mark H3 `rejected`, skip
Tasks 10-11, and continue to H4.

### Task 10: Test H3 Variants Independently In The Ignored Probe

**Files:**
- Modify ignored: `artifacts/experiments/2026-07-12-gaze-precision/probe.py`
- Produce ignored JSON under: `artifacts/experiments/2026-07-12-gaze-precision/h3-variants/`

**Interfaces:**
- Tests `current_candidate_continuity`, then `prior_roi`, then `mediapipe_video`.
- A selected bbox always comes from the current frame.
- Produces the smallest passing variant name or `rejected`.

- [ ] **Step 1: Add current-candidate continuity**

After ordinary IMAGE-mode region observations, rank current candidates by:

```python
(
    _bbox_iou(previous_bbox, candidate.bounding_box_image_px),
    -normalized_centroid_distance(previous_bbox, candidate.bounding_box_image_px),
    candidate.selection_score or 0.0,
)
```

Prefer an overlapping current candidate only when IoU is positive; otherwise
use ordinary arbitration/reacquisition. Reset the previous bbox on one miss.
Do not force a ten-region scan only for tracking.

- [ ] **Step 2: Add the prior-ROI variant separately**

Acquire normally. On the next frame, first detect in a clipped `1.5x` previous
bbox and remap with `_candidates_from_mediapipe_result()`. Accept only a current
ROI candidate whose IoU with the prior is positive; otherwise run ordinary
full arbitration. Count detector calls as runtime evidence, not accuracy.

- [ ] **Step 3: Add the MediaPipe VIDEO variant separately**

Use `RunningMode.VIDEO`, `num_faces=1`, full-frame input, and:

```python
timestamp_ms = round(frame.pts_seconds * 1000)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

Abort on absent, duplicate, or decreasing PTS. Do not combine VIDEO with the
prior ROI. Its opaque state makes resume equivalence a mandatory later gate.

- [ ] **Step 4: Run each face-only variant on all approved clips**

For each variant, review every baseline confirmed failure plus the candidate's
own top events from Task 9. Record face coverage, confirmed fixes/new failures,
bbox centre/scale tails, detector calls, and runtime.

- [ ] **Step 5: Select the smallest passing variant**

A variant passes only if it fixes every confirmed baseline selection failure,
creates zero new confirmed swaps/false crops, loses at most two percentage
points aggregate face coverage and five on any clip. Do not use gaze jitter as
the retention gate. If none passes, delete H3 experimental code and skip Task 11.

### Task 11: Implement And Verify The Smallest Passing H3 Variant (Conditional)

**Files:**
- Conditionally create: `src/chess_gaze/face_selection.py`
- Conditionally create: `tests/chess_gaze/test_face_selection.py`
- Modify: `src/chess_gaze/face_observation.py`
- Modify: `src/chess_gaze/calibration.py`
- Modify: `src/chess_gaze/configuration.py`
- Modify: `src/chess_gaze/frame_records.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/analysis_resume.py`
- Modify: `src/chess_gaze/gaze_precision_benchmark.py`
- Modify: relevant face, configuration, resume, pipeline, and benchmark tests.

**Interfaces:**
- Produces `FaceSelectionPolicy = Literal["independent_v1", "previous_bbox_continuity_v1"]` when continuity wins; use the winning name if another variant passes.
- Produces `select_region_face(..., previous_bbox: BBox | None = None) -> FaceSelection`.
- Produces `initial_selected_bbox(committed_records: Sequence[FrameRecord]) -> BBox | None`.
- Extends the comparator with declared group `face_selection` only when H3 is retained.

- [ ] **Step 1: Write selection RED tests after discarding exploratory implementation**

Cover: a distant area winner versus a smaller overlapping current candidate;
zero-overlap fallback to ordinary arbitration; empty current candidates never
return a stale bbox; one miss resets state; reordered candidates retain identity.

```python
selection = select_region_face(
    region_selections,
    full_frame_selection,
    previous_bbox=previous_bbox,
)
assert selection.primary_candidate_id == "overlapping_current"
```

- [ ] **Step 2: Run selection RED**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_face_selection.py -q
```

Expected: missing module/interface.

- [ ] **Step 3: Extract the approved selection boundary and implement minimal state**

Move existing face candidate/selection records, detection-region records,
arbitration, bbox helpers, and region policy into `face_selection.py` because
`face_observation.py` is already 1,203 lines and the architecture guide names
this split. `face_observation.py` continues to own MediaPipe lifecycle/result
mapping and directly uses the moved behavior; do not add a pass-through package.

For continuity, initialize `_previous_selected_bbox`, update it only from a
selected current candidate, and set it to `None` on a miss.

- [ ] **Step 4: Write resume-seed RED tests**

Assert a committed prefix ending in a present face returns its exact bbox; a
prefix ending absent returns `None`. Test uninterrupted and resumed stateful
selection across every cut of `[A, A, miss, B]`.

- [ ] **Step 5: Implement persisted-state seeding**

```python
def initial_selected_bbox(
    committed_records: Sequence[FrameRecord],
) -> BBox | None:
    if not committed_records:
        return None
    last_face = committed_records[-1].face
    return last_face.bounding_box if last_face.present else None
```

Pass the seed into `_default_observer_bundle_factory()` after
`prepare_resume_run()`. New runs pass `None`. If VIDEO won, replay skipped frames
through the detector instead; serialized fake state is not acceptable.

- [ ] **Step 6: Persist only the winning policy and comparator group**

Thread `face_selection_policy` through `AnalysisConfig`, `CalibrationRecord`,
request resolution, default calibration, and observer setup. Use config JSON for
paired runs; do not add a CLI flag. Add exact allowed path
`calibration.face_selection_policy` under comparator group `face_selection`.

- [ ] **Step 7: Run focused GREEN**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/chess_gaze/test_face_selection.py \
  tests/chess_gaze/test_face_observation.py \
  tests/chess_gaze/test_face_observation_region_arbitration.py \
  tests/chess_gaze/test_analysis_resume.py \
  tests/chess_gaze/test_pipeline_contract.py \
  tests/chess_gaze/test_gaze_precision_benchmark.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Run independent/current-policy full MPS pairs on all clips**

Create two temporary config files differing only in `face_selection_policy` and
run the same six-command pattern as Task 4 under
`artifacts/output/gaze-hypotheses/h3/independent` and `.../candidate`. Compare
each pair with `--experimental-variable face_selection`, rerun Task 9 review on
candidate outputs, and reapply the manual correctness/coverage gate.

- [ ] **Step 9: Prove real resume equivalence**

On `carlsen_short.mp4`, run one uninterrupted candidate analysis. Run another
candidate analysis whose progress callback raises after frame 301, then resume
the same run. Require `records/frames.jsonl` byte equality and
`run_equivalence.compare_runs()` success for derived artifacts. Any mismatch
rejects H3 and removes all conditional production changes.

- [ ] **Step 10: Commit only a fully passing H3**

```sh
git add src/chess_gaze/face_selection.py src/chess_gaze/face_observation.py \
  src/chess_gaze/calibration.py src/chess_gaze/configuration.py \
  src/chess_gaze/frame_records.py src/chess_gaze/pipeline.py \
  src/chess_gaze/analysis_resume.py \
  src/chess_gaze/gaze_precision_benchmark.py \
  tests/chess_gaze/test_face_selection.py \
  tests/chess_gaze/test_face_observation.py \
  tests/chess_gaze/test_face_observation_region_arbitration.py \
  tests/chess_gaze/test_calibration.py \
  tests/chess_gaze/test_configuration.py \
  tests/chess_gaze/test_frame_records.py \
  tests/chess_gaze/test_analysis_resume.py \
  tests/chess_gaze/test_pipeline_contract.py \
  tests/chess_gaze/test_gaze_precision_benchmark.py \
  docs/development/architecture/source-layout.md
git commit -m "fix: preserve face identity across frames"
```

If H3 rejects, delete every conditional field/module/test and make no H3 commit.

### Task 12: Probe H4 Decode Orientation And Horizontal Equivariance

**Files:**
- Modify ignored: `artifacts/experiments/2026-07-12-gaze-precision/probe.py`
- Conditionally modify: `src/chess_gaze/video_decode.py`
- Conditionally modify: `tests/chess_gaze/test_video_decode.py`

**Interfaces:**
- Produces metadata and flip-equivariance evidence.
- Retains code only for an independently failing decode/sign contract.

- [ ] **Step 1: Audit all approved source orientation metadata**

Use `inspect_video()` for every approved globbed input and record rotation,
decoded shape, first/last PTS, and timestamp usability. Expected current evidence
is `rotation_degrees=None` for all three; do not reinterpret `None` as unmirrored.

- [ ] **Step 2: Generate asymmetric rotation fixtures**

Create `rotation_0_short.mp4`, `rotation_90_short.mp4`,
`rotation_180_short.mp4`, and `rotation_270_short.mp4` with a 3x2 asymmetric
colour/quadrant marker and corresponding rotation metadata. Assert decoded
upright pixel arrangement and dimensions for each. Run the new test before any
fix and capture whether current decode fails.

- [ ] **Step 3: Run fixed-frame flip inference**

For frames 0, midpoint, and last of each approved clip, infer original and
horizontally flipped images. Map flipped output back exactly once by negating
camera x/yaw, then report angular disagreement and face coverage. Logos/text may
support manual mirror interpretation but never decide it automatically.

- [ ] **Step 4: Apply the H4 gate**

With no labeled left/centre/right targets and no rotated real input, default H4
outcome is `inconclusive`. If and only if Step 2 first demonstrates an actual
decode contract failure, write a focused RED regression, implement the smallest
orientation transform in `video_decode.py`, prove allowed unrotated frame bytes
unchanged, run the video-decode suite, and commit:

```sh
git add src/chess_gaze/video_decode.py tests/chess_gaze/test_video_decode.py
git commit -m "fix: apply declared video orientation"
```

Do not add `facecam_mirrored` merely to persist `unknown`.

### Task 13: Probe H5 Transparent Quality Evidence Without Shipping A Score

**Files:**
- Modify ignored: `artifacts/experiments/2026-07-12-gaze-precision/probe.py`
- Produce ignored: `artifacts/experiments/2026-07-12-gaze-precision/h5-quality.json`

**Interfaces:**
- Consumes retained baseline frame records and H3 manual labels.
- Produces per-signal proxy tables at 100%, 95%, and 90% retained coverage.
- Produces no production field, threshold, score, confidence, or rejection gate.

- [ ] **Step 1: Compute only available cheap signals**

Per frame compute face/eye width and height, both-iris availability, H7 iris
disagreement, head-pose magnitude, bbox centre/scale step, and appearance-gaze
step. Define a per-clip gaze-step outlier as `>= p95`. Do not add crop blur or
PnP error because current records do not contain them.

- [ ] **Step 2: Report each signal independently**

For each signal, report median inlier/outlier values, rank correlation using
NumPy ranks, outlier rate at top 100/95/90% quality, per-target-equivalent frame
coverage by clip, and values at confirmed H3 failures. Do not combine signals.

```python
ranks = np.argsort(np.argsort(np.asarray(values, dtype=np.float64)))
rank_correlation = float(np.corrcoef(ranks, outlier_labels)[0, 1])
```

- [ ] **Step 3: Record H5 as inconclusive and remove probe-only logic later**

Even monotonic proxy separation cannot establish accuracy. Retain no
`quality_score`, probability, weights, fields, thresholds, or tests.

### Task 14: Probe H6 Three-Vector Median Offline

**Files:**
- Modify ignored: `artifacts/experiments/2026-07-12-gaze-precision/probe.py`
- Produce ignored: `artifacts/experiments/2026-07-12-gaze-precision/h6-filter.json`

**Interfaces:**
- Produces raw-versus-filtered proxy metrics for all approved clips.
- Does not modify `FrameRecord.recommended_gaze` or scene projection.

- [ ] **Step 1: Add a self-checked centred median function**

```python
def median_vector(left: np.ndarray, center: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.median(np.stack((left, center, right)), axis=0)
    norm = np.linalg.norm(result)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("median gaze vector must have finite non-zero norm")
    return result / norm
```

Add assertions for three identical vectors, one angular outlier, and zero-vector
rejection before running real records.

- [ ] **Step 2: Apply exact gap/validity rules**

For frame `i`, require valid `i-1/i/i+1`, positive gaps, and both gaps no larger
than twice the clip median positive delta. Otherwise leave raw/invalid unchanged.

- [ ] **Step 3: Report precision and response proxies**

Report coverage and median/p95/p99 angular steps. Define raw events as per-clip
steps `>= p95`, cluster within two frames, match the maximum filtered step within
plus/minus two frames, and report signed frame/seconds offset and amplitude ratio.
Call this a raw-change proxy, never fixation precision or target-switch lag.

- [ ] **Step 4: Record H6 as inconclusive**

Delete filter code before final commit even if smoother because target accuracy,
fixations, and true switch times are absent.

### Task 15: Probe H7 Existing Iris Offsets

**Files:**
- Modify ignored: `artifacts/experiments/2026-07-12-gaze-precision/probe.py`
- Produce ignored: `artifacts/experiments/2026-07-12-gaze-precision/h7-iris.json`

**Interfaces:**
- Recomputes features from existing pupil centres and eye boxes.
- Produces no calibrator feature or schema change.

- [ ] **Step 1: Implement the two offsets and disagreement**

```python
horizontal = (pupil.x - (bbox.x_min + bbox.x_max) / 2.0) / (
    bbox.x_max - bbox.x_min
)
vertical = (pupil.y - (bbox.y_min + bbox.y_max) / 2.0) / (
    bbox.y_max - bbox.y_min
)
mean_horizontal = (left_horizontal + right_horizontal) / 2.0
mean_vertical = (left_vertical + right_vertical) / 2.0
disagreement = abs(left_horizontal - right_horizontal)
```

Reject non-present eyes, zero-size boxes, and non-finite results.

- [ ] **Step 2: Report availability and temporal noise**

Per clip report finite/both-eye availability, median/p05/p95, median/p95
consecutive absolute step, association with p95 gaze steps and H3 failures, and
observed eye pixel heights. Break temporal pairs at invalid/non-positive/>2x
median timestamp gaps.

- [ ] **Step 3: Record H7 as inconclusive**

No targets means no residual calibration comparison. Retain no feature or
regressor change.

### Task 16: Probe H8 Chess-Content Feasibility Without Inferring Attention

**Files:**
- Modify ignored: `artifacts/experiments/2026-07-12-gaze-precision/probe.py`
- Produce ignored: `artifacts/experiments/2026-07-12-gaze-precision/h8-board-events.json`

**Interfaces:**
- Produces board-layout and move-change feasibility evidence only.
- Produces no board schema, screen mapping, focus label, or snapping behavior.

- [ ] **Step 1: Verify board edges on first/middle/last frames**

Start from these inspected approximate rectangles and adjust only to visible
square edges before calculating:

```python
rectangles = {
    "carlsen_short.mp4": (979, 86, 1887, 997),
    "nakamura_short.mp4": (953, 77, 1878, 1002),
    "nepo_short.mp4": (0, 0, 721, 720),
}
```

Record final coordinates and whether the board stays aligned throughout each
clip.

- [ ] **Step 2: Detect and review board-local changes**

Divide each rectangle into 8x8, trim ten percent inside every square, and compute
grayscale mean absolute difference between consecutive frames. Rank by maximum
and top-two square deltas, cluster within three frames, and inspect the top five
events per clip as `piece_move`, `board_animation`, or `unrelated_noise`.

- [ ] **Step 3: Apply the feasibility gate and block integration**

Feasibility passes only if the board remains aligned and at least one board-local
change is visually verified per clip. Regardless, H8 focus integration is
`blocked`: no gaze-to-screen calibration or attention truth exists. Retain no
rectangle, detector, classifier, or square-snapping code.

### Task 17: Refresh The H9 Model Preflight Without Downloading A Model

**Files:**
- No production changes.
- Summarize evidence later in: `docs/superpowers/closeouts/2026-07-12-gaze-precision-hypothesis-evaluation.md`

**Interfaces:**
- Produces a current matrix for UniGaze-H14, ST-Gaze, L2CS-Net, and 3DGazeNet.
- Does not select or download an alternate checkpoint.

- [ ] **Step 1: Refresh only official primary sources**

Open the exact official repository/project/checkpoint URLs recorded in the
approved design. Record verification date, code and weight licenses separately,
current availability, required preprocessing, input/output/confidence contract,
temporal state, package/runtime requirements, published dataset/metric
conditions, and Apple-MPS evidence.

- [ ] **Step 2: Verify local assets and resolved versions**

```sh
shasum -a 256 models/mediapipe/face_landmarker.task
shasum -a 256 models/unigaze/unigaze_h14_joint.safetensors
UV_CACHE_DIR=.uv-cache uv run python -c "from importlib.metadata import version; print({name:version(name) for name in ('av','opencv-python-headless','mediapipe','numpy','torch','unigaze')})"
```

Expected model digests are the approved MediaPipe and UniGaze hashes. Record
alternate local asset as `not downloaded`, not an empty checksum.

- [ ] **Step 3: Record H9 as blocked**

H2 cannot provide held-out target accuracy from the allowed clips, so no valid
model bakeoff exists. Do not modify `pyproject.toml`, `uv.lock`, registry, model
adapters, or checkpoints. No ADR is needed because selection does not change.

### Task 18: Probe H10 Crop Sensitivity And A Sample Ensemble

**Files:**
- Modify ignored: `artifacts/experiments/2026-07-12-gaze-precision/probe.py`
- Produce ignored: `artifacts/experiments/2026-07-12-gaze-precision/h10-crops.json`

**Interfaces:**
- Produces fixed-frame angular sensitivity and sampled compute cost.
- Produces no production crop smoother or ensemble.

- [ ] **Step 1: Choose frames before measuring perturbations**

Per clip select three stable frames with the smallest combined bbox/gaze step
and three worst reviewed discontinuity frames from Task 9. Record frame IDs.

- [ ] **Step 2: Run five predefined bbox perturbations**

Use translation/scale variants:

```python
variants = (
    (-0.02, 0.0, 1.00),
    (0.00, 0.0, 1.00),
    (0.02, 0.0, 1.00),
    (0.00, 0.0, 0.95),
    (0.00, 0.0, 1.05),
)
```

Translate by bbox width fractions, scale about centre, clip to frame, reuse the
retained normalizer/model, and report pairwise angular spread/max centre
deviation. Define material before viewing results as median spread >=1 degree or
any frame >=3 degrees.

- [ ] **Step 3: Only when material, run the three-translation sample ensemble**

Normalize and average vectors for `-2%`, `0`, and `+2%`, renormalize, and report
sample angular-step effect plus measured inference-time ratio. Do not run a full
three-crop corpus ensemble.

- [ ] **Step 4: Record H10 as inconclusive**

No target truth means consistency/smoothness cannot establish improvement.
Delete all ensemble/crop-probe logic before final commit.

### Task 19: Clean Up, Verify, Review, And Close Out H0-H10

**Files:**
- Create: `docs/superpowers/closeouts/2026-07-12-gaze-precision-hypothesis-evaluation.md`
- Conditionally update: `README.md`
- Conditionally update: `docs/development/architecture/source-layout.md`
- Conditionally create/update: ADR files required by retained architecture changes.
- Delete generated task-only probe directory before commit.

**Interfaces:**
- Produces one evidence table with outcome `kept`, `rejected`, `inconclusive`, or `blocked` for every H0-H10.
- Produces fresh full-gate evidence and an independently reviewed final diff.

- [ ] **Step 1: Write the complete closeout from captured evidence**

For every hypothesis record: question tested, valid observable gate, outcome,
per-clip source/run/report paths, metric table, retained files/commit, and
residual risk. H3 lists every reviewed frame/label and resume result. H4 lists
metadata/fixture/flip results. H5-H7/H10 explicitly label proxy-only results.
H8/H9 name their prerequisite blockers. Include exact model/input hashes and
primary-source verification dates.

- [ ] **Step 2: Record root causes and durable repairs**

For each retained code change document the reproduced failure, underlying
cause, durable owning module, failing regression, focused GREEN evidence, and
any behavior deliberately not retained. Update README only if a default runtime
profile/policy changed; update source layout only for an actual retained module
or changed ownership.

- [ ] **Step 3: Remove every temporary experiment artifact and audit leftovers**

```sh
rm -rf artifacts/experiments/2026-07-12-gaze-precision
git status --short
rg -n 'quality_score|facecam_mirrored|three.frame|iris.*feature|crop.*ensemble|ST.Gaze|L2CS' src tests
```

Expected: no temporary file is staged or tracked; every source/test match is an
intentional retained contract already explained in the closeout. Remove any
failed/inconclusive toggle, field, helper, or test before continuing.

- [ ] **Step 4: Run the broad non-native/local-socket gate**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q \
  -m 'not native_mediapipe and not local_socket'
```

Expected: pass. This includes all model-free real-video contracts, whose inputs
are `*_short.mp4`.

- [ ] **Step 5: Run only approved-input native tests outside the sandbox**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q \
  tests/chess_gaze/test_eye_observation_real_video.py::test_eye_observation_matches_real_video_evidence \
  tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_matches_real_video_evidence \
  tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_nakamura_short_visible_faces \
  tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_nakamura_short_faces_bounded \
  tests/chess_gaze/test_gaze_observation_real_video.py \
  tests/chess_gaze/test_head_pose_real_video.py \
  tests/chess_gaze/test_pipeline_real_video_contract.py::test_nakamura_short_default_model_pipeline_does_not_create_crop_directory \
  tests/chess_gaze/test_pipeline_real_video_contract.py::test_nakamura_short_save_crops_retains_crop_images_only
```

Also include `tests/chess_gaze/test_unigaze_preprocessing_real_video.py` when H1
was retained. Do not run the Carlsen `carlsen_1.mp4` or Nepo `nepo_2.mp4` node
IDs; list these exact exclusions in the closeout.

- [ ] **Step 6: Run local-socket, lint, format, and type gates**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q -m local_socket
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
git diff --check
```

Expected: zero failures/errors and no whitespace errors.

- [ ] **Step 7: Request independent code review**

Invoke `superpowers:requesting-code-review`. Ask a fresh reviewer to compare the
approved spec and plan against the complete branch diff, prioritizing incorrect
claims, provenance gaps, coordinate/sign mistakes, resume divergence, schema
compatibility, hidden use of non-short inputs, and failed-experiment leftovers.
Verify every finding locally; fix Critical/Important findings test-first.

- [ ] **Step 8: Re-run every affected focused gate and the full Steps 4-6**

Fresh output after review fixes is mandatory. Record exact pass counts and
commands in the closeout; do not reuse pre-review results.

- [ ] **Step 9: Commit the closeout and canonical documentation**

Stage only files shown by `git status --short` as intentional task results:

```sh
git add docs/superpowers/closeouts/2026-07-12-gaze-precision-hypothesis-evaluation.md
git diff --cached --check
git commit -m "docs: close out gaze precision hypotheses"
```

Before the commit, add `README.md` and
`docs/development/architecture/source-layout.md` individually only when this
task changed them. Add the exact ADR-0007 path only when H1 was retained. Never
stage the whole decisions directory. Verify final
`git status --short --branch` and `git log --oneline` show a clean current branch
and all meaningful checkpoints.
