# Test 0 Quality Repair Closeout

Date: 2026-06-25

## Request Summary

Investigate and repair poor default real-model performance on
`artifacts/input/test_0.mp4`: visible faces and eyes were missed, most frames
only drew the cyan UniGaze arrow, and several arrows pointed in visibly wrong
directions. Work was limited to the current branch and to `test_0.mp4` for
real-video verification.

## Root Causes

1. MediaPipe full-frame detection missed visible faces in split-screen layouts.
   The same Face Landmarker recovered those faces when run on deterministic
   left/right half-frame regions.
2. Head pose was gated by a synthetic canonical-face PnP reprojection threshold.
   On `test_0.mp4`, PnP often exceeded the threshold or produced wrapped angles,
   while MediaPipe's facial transformation matrix provided finite stable pose
   evidence.
3. MediaPipe transform pitch and UniGaze yaw used different sign conventions
   from the frame-record/overlay convention. This made otherwise usable arrows
   point in the wrong visual direction.
4. Recommended-gaze synthesis used an effectively permissive default threshold
   and collapsed some valid appearance-only cases into `GAZE_MODEL_FAILED`.
   This either accepted bad blends or hid the true failure reason.

## Durable Surfaces Changed

- `MediaPipeFaceObserver.observe()` now tries full frame first, then contiguous
  left-half and right-half regions, translating recovered landmarks and boxes
  back into source image coordinates.
- `estimate_head_pose()` now preserves PnP evidence but uses a finite MediaPipe
  facial transformation matrix as the valid pose source when available.
- MediaPipe transform pitch is converted into image-up-positive frame-record
  convention before downstream geometric gaze uses it.
- `UniGazeModel.predict()` converts UniGaze yaw into the renderer's
  image-right-positive convention.
- Recommended-gaze synthesis now uses a non-permissive default agreement
  threshold, permits appearance-only recommendations when it is the sole valid
  estimator, and refuses large estimator disagreements.

## Artifact Verification

Failed baseline run:
`artifacts/output/test_0/runs/20260625T182409Z-a810e406`

Fresh repaired run:
`/private/tmp/chess-gaze-test0-quality3/test_0/runs/20260625T192309Z-c4d80ee0`

Command:

```text
MPLCONFIGDIR=/private/tmp/chess-gaze-mpl .venv/bin/chess-gaze analyze artifacts/input/test_0.mp4 --output-root /private/tmp/chess-gaze-test0-quality3 --models-root models
```

The sandboxed CLI path still aborts in the managed macOS environment during
MediaPipe GL/Metal graph initialization, so the full real CLI verification was
run unsandboxed with local model assets. Non-fatal AVFoundation duplicate-class
warnings and a MediaPipe Clearcut upload warning were observed.

QA summary comparison:

```text
decoded/raw/processed/frame_records: 300 -> 300
crop_files: 466 -> 600
schema_validation_passed: true -> true
counts_match: true -> true
face_present_rate: 0.7767 -> 1.0000
both_eyes_present_rate: 0.7767 -> 1.0000
head_pose_valid_rate: 0.2667 -> 1.0000
face_gaze_valid_rate: 0.7767 -> 1.0000
recommended_gaze_valid_rate: 0.1333 -> 0.5433
errors_by_code: FACE_NOT_FOUND 134, GAZE_MODEL_FAILED 153,
  HEAD_POSE_INVALID 306, GAZE_ESTIMATORS_DISAGREE 40
  -> GAZE_ESTIMATORS_DISAGREE 137
```

Visual QA sampled old-failure frames:

- `f000000080`: face and both eyes are now detected; arrows no longer use the
  old wrapped upward pose; recommendation remains invalid because estimators
  still disagree.
- `f000000090`: now `status=OK` with a recommended arrow instead of only a cyan
  UniGaze arrow.
- `f000000155` and `f000000217`: side-view board frames are now `status=OK`;
  geometric, UniGaze, and recommended arrows point toward the board side.
- `f000000247`: face, eyes, and head pose are valid, but recommendation remains
  invalid because the estimators disagree. This is the intended fail-closed
  behavior.

## Regression Coverage

- Unit regression for split-frame face fallback and coordinate translation.
- Real `test_0.mp4` face regression for frames 80, 217, 247, and 258.
- Unit regression for transform-backed head pose when PnP reprojection is high.
- Real `test_0.mp4` head-pose regression for frames 0, 90, 144, 155, 217, and
  289.
- Unit regressions for appearance-only recommended gaze and large estimator
  disagreement.
- Unit regressions rejecting one-eye-only geometric recommendations and
  preferring head/eye causes over UniGaze failure when no estimator can produce
  a recommendation.
- Real `test_0.mp4` head-pose regression now checks pitch sign on down-looking
  sampled frames, not only finite angle magnitude.
- Real `test_0.mp4` model-backed observer regression for frames 90, 155, and
  217 producing `status=OK` with valid recommended gaze.

## Review Response

A read-only subagent review found one stale blocking issue from the intermediate
run where all recommendations were still invalid. The final current-tree run
above resolves that with `recommended_gaze_valid_rate=0.5433`. The review also
identified two real arbitration weaknesses: a single valid geometric eye could
be accepted as a recommendation, and UniGaze failure could hide a more
actionable head/eye cause when no source was valid. Both were fixed and covered
by unit regressions. The pitch-sign concern was addressed with a real `test_0`
sign assertion on sampled down-looking frames.

## Final Gates

```text
.venv/bin/python -m pytest tests/chess_gaze -q -k 'not real_video'
109 passed, 14 deselected, 18 warnings in 2.09s
```

```text
.venv/bin/python -m pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_test0_visible_split_frame_faces tests/chess_gaze/test_head_pose_real_video.py::test_head_pose_uses_mediapipe_transform_on_test0_pnp_failure_frames tests/chess_gaze/test_gaze_observation_real_video.py::test_default_model_observer_recommends_gaze_on_repaired_test0_frames -q
3 passed, 18 warnings in 13.98s
```

```text
.venv/bin/ruff check .
All checks passed!
```

```text
.venv/bin/ruff format --check .
44 files already formatted
```

```text
.venv/bin/mypy src tests
Success: no issues found in 44 source files
```

## Remaining Limitations

- The full official UniGaze video pipeline performs gaze-specific face
  normalization before inference. This repair fixes a documented yaw convention
  mismatch and prevents false recommendations, but it does not implement that
  full normalization pipeline. Remaining `GAZE_ESTIMATORS_DISAGREE` frames are
  therefore reported explicitly instead of forced into a recommendation.
- Board target mapping is still absent, so `target_image_px`,
  `target_board_norm`, and `target_square` remain null.
- Real `test_1.mp4` and `test_2.mp4` gates were intentionally not run for this
  repair because the task constrained real-video testing to `test_0.mp4`.
