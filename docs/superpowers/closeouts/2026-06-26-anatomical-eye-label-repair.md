# Anatomical Eye Label Repair Closeout

Date: 2026-06-26
Branch: fixes-3
Commits:

- `50fee0f fix: use anatomical face landmark labels`
- `94ea258 fix: require explicit missing-eye side for gaze`
- This closeout commit records the final reviewer follow-ups.

## Summary

The left/right eye bug was a source-label bug, not a viewer-only color bug.
`left_eye` and `right_eye` had been populated from image-side MediaPipe landmark
groups. For a frontal webcam, image-left is the streamer's anatomical right, so
the persisted records, processed-frame labels, 3D eye colors, PnP anchors, and
derived scene positions were all vulnerable to left/right inversion.

The repair makes `left_eye` mean the streamer's anatomical left eye everywhere
the current pipeline controls:

- MediaPipe landmark IDs are centralized in
  `src/chess_gaze/face_landmark_indices.py` using anatomical names.
- Eye observation now uses MediaPipe anatomical left eye `263/362` with iris
  `473-477`, and anatomical right eye `33/133` with iris `468-472`.
- PnP calibration and canonical face model points now use anatomical eye and
  mouth labels together, preserving a coherent solvePnP correspondence set.
- Geometric gaze no longer guesses missing-eye side from an optional attribute;
  callers must provide the explicit anatomical missing-eye reason.
- Scene, viewer, visualization, calibration, head-pose, and gaze tests now use
  fixtures where the streamer's left eye appears image-right.
- Real-video contract helper fixtures now use the same anatomical convention,
  and PnP has an independent frontal landmark test that does not derive image
  points from the production canonical face model.

Viewer and processed-frame colors after the repair:

- Blue / `#2f80c2` in the 3D viewer, brighter blue in processed frames:
  `left_eye`, the streamer's anatomical left eye.
- Orange-red / `#d46a5b` in the 3D viewer, orange-red in processed frames:
  `right_eye`, the streamer's anatomical right eye.

## Root Cause

The original code treated "left" and "right" as screen/image side. MediaPipe's
installed source labels the `263/362` eye group as `FACE_LANDMARKS_LEFT_EYE`
and the `33/133` eye group as `FACE_LANDMARKS_RIGHT_EYE`. Our code had those
groups reversed in `eye_observation.py`, and `calibration.py`/`head_pose.py`
used matching image-side PnP names. That internal consistency masked the bug in
tests, because several fixtures used the same production constants or placed
`left_eye` at lower image X.

The second related bug was in `gaze_observation.py`: when an eye was missing,
`_eye_invalid_reason()` looked for a `side` attribute and otherwise returned
`LEFT_EYE_NOT_FOUND`. Real `EyeObservation` records do not carry `side`, so
right-eye geometric failures could be misattributed as left-eye failures.

## Real Nakamura Verification

Real input used: `artifacts/input/nakamura_1.mp4`.

Fresh repaired run:
`artifacts/output/nakamura_1/runs/20260626T160949Z-bbc6c27e`.

The initial sandboxed run failed in MediaPipe native Metal/GL initialization
with `Check failed: service_ Service is unavailable`; rerunning outside the
sandbox completed model inference and artifact writing. I accidentally
interrupted the CLI while it was computing QA exposure samples, then completed
`qa_summary.json` using the pipeline QA finalizer against the same run layout.
Final artifact validation reports `final_status=complete` and
`schema_validation_passed=true`.

Final run counts:

- decoded frames: 1,973
- frame records: 1,973
- scene frame records: 1,973
- raw frames: 1,973
- processed frames: 1,973
- eye crops: 3,946
- valid monitor hits: 1,971
- invalid monitor hits: 2, both `RAY_INTERSECTION_BEHIND_ORIGIN`

Numerical side/coordinate audit over all 1,973 scene frames:

- `left_eye.image_px.x > right_eye.image_px.x`: 1,973 / 1,973
- `left_eye.scene_m.x < right_eye.scene_m.x`: 1,973 / 1,973, because scene
  `+X` is streamer right
- both eyes are in front of the head ellipsoid center: 1,973 / 1,973
- UniGaze ray has forward scene direction (`z < 0`): 1,973 / 1,973

First frame evidence:

- `left_eye.image_px.x = 533.2799377441406`
- `right_eye.image_px.x = 445.0989761352539`
- `left_eye.scene_m.x = -0.03750933990120747`
- `right_eye.scene_m.x = 0.025484564652183517`
- `unigaze_ray.direction_scene.z = -0.9211029928337371`

Visual checks:

- Processed-frame contact sheet from six Nakamura frames shows orange-red `R`
  on image-left and blue `L` on image-right, matching streamer anatomy.
- Chrome DevTools loaded the generated viewer from the run file URL with no
  console messages. The page reported frame `1 / 1973`, hits `1971`, valid
  WebGL context, and first-frame scene data matching the numerical audit.
- Viewer screenshot `/private/tmp/nakamura_viewer_160949.png` was nonblank:
  `2880x1882`, 3,506 distinct colors, 480,650 non-background-like pixels.

## Verification

Red tests observed before fixes:

- Anatomical landmark tests failed while `left_eye` still used MediaPipe `33`
  and `right_eye` used `263`.
- New missing-eye-side test failed because `compute_per_eye_geometric_gaze()`
  did not accept an explicit `missing_reason`.
- New visualization fixture/color tests failed because test `left_eye` was
  still image-left.

Passing focused tests:

- `uv run pytest tests/chess_gaze/test_eye_observation.py tests/chess_gaze/test_calibration.py::test_default_calibration_persists_named_constants tests/chess_gaze/test_head_pose.py::test_default_pnp_correspondences_use_streamer_anatomical_sides tests/chess_gaze/test_head_pose.py::test_pnp_uses_named_landmark_indices_from_calibration tests/chess_gaze/test_head_pose.py::test_valid_rotations_are_stored_as_matrix_quaternion_and_angles tests/chess_gaze/test_scene_artifacts.py::test_scene_frame_preserves_anatomical_eye_sides_for_frontal_webcam tests/chess_gaze/test_scene_geometry.py::test_back_project_eye_points_projects_eyes_and_midpoint_in_camera_space tests/chess_gaze/test_scene_geometry.py::test_back_project_eye_points_uses_euclidean_pupil_distance -q`
  -> 13 passed.
- `uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q`
  -> 28 passed.
- `uv run ruff format --check src/chess_gaze/frame_observation.py src/chess_gaze/gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_gaze_observation_real_video.py tests/chess_gaze/test_visualization.py`
  -> 6 files already formatted.
- `uv run ruff check src/chess_gaze/frame_observation.py src/chess_gaze/gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_gaze_observation_real_video.py tests/chess_gaze/test_visualization.py`
  -> all checks passed.
- `uv run pytest tests/chess_gaze/test_eye_observation.py tests/chess_gaze/test_calibration.py tests/chess_gaze/test_head_pose.py tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q`
  -> 118 passed.
- `uv run pytest tests/chess_gaze/test_head_pose.py::test_pnp_evidence_uses_independent_frontal_anatomical_landmarks tests/chess_gaze/test_head_pose.py::test_default_pnp_correspondences_use_streamer_anatomical_sides tests/chess_gaze/test_visualization.py::test_eye_overlay_colors_follow_streamer_anatomical_sides tests/chess_gaze/test_frame_observation.py::test_model_backed_frame_observer_preserves_missing_right_eye_reason -q`
  -> 4 passed.
- Direct helper contract script over
  `test_pipeline_real_video_contract.py`,
  `test_qa_summary_real_video_contract.py`,
  `test_scene_artifacts_real_video_contract.py`, and
  `test_visualization_real_video.py`
  -> all four helpers produce `left_eye.pupil_center.x >
  right_eye.pupil_center.x` on a dummy frontal frame.
- `uv run ruff format --check src tests`
  -> 57 files already formatted.
- `uv run ruff check .`
  -> all checks passed.

Code review:

- A read-only reviewer checked `7007e80..94ea258` and found no critical issues.
  Important follow-ups were addressed before closeout: commit this closeout,
  update stale real-video helper fixtures, add an independent PnP fixture, and
  update docs for the new helper signature and source layout map.

Broader pre-closeout test run:

- `uv run pytest tests/chess_gaze/test_eye_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_calibration.py tests/chess_gaze/test_head_pose.py tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_frame_records.py -q`
  -> 126 passed, 1 failed because `artifacts/input/test_1.mp4` and
  `artifacts/input/test_2.mp4` are missing in this checkout.
- A later broader run, started before the reviewer follow-up edits, reached
  138 passed and the same missing-input failure in
  `test_visualization_real_video.py`, then was interrupted because its result
  was stale after additional fixture/doc changes.
- A monolithic post-review real-video focused run was interrupted after it spent
  an unusually long time in full-video artifact cleanup. The same reviewer
  follow-ups were then verified with the targeted pytest and direct helper
  contract checks listed above.

## Residual Risk

This repair fixes the code-controlled side semantics and verifies them on the
real Nakamura input. It does not prove that UniGaze's learned gaze estimates are
semantically correct in every pose; it verifies that the records, overlays, and
scene viewer now preserve the streamer's anatomical left/right convention and
that the ray/scene axes are internally coherent for the tested run.

The full real `chess-gaze analyze` command for the fresh run did not end with
exit code 0 because I interrupted it during the slow QA exposure step after
core artifacts were written. The same run's QA summary was completed afterward
by calling the pipeline QA finalizer directly, and artifact validation then
reported complete.
