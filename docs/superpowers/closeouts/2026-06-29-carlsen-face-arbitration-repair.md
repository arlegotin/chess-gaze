# Carlsen Face Arbitration Repair Closeout

Date: 2026-06-29

## Summary

This closeout verifies the Carlsen face-arbitration repair already committed in:

- `5ef33ad` `fix: prefer consensus face candidates`
- `d7121b6` `fix: require consensus fallback precedence`
- `d6a6753` `fix: add left upper inner face recovery region`
- `23afa61` `fix: add paired inner face consensus region`

Task 4 added a bounded real-video regression in
`tests/chess_gaze/test_face_observation_real_video.py`, reran real MediaPipe
verification on the reported Carlsen windows plus adjacent controls, reran the
required Nakamura short checks, and recorded end-to-end evidence from a fresh
`chess-gaze analyze` run over `artifacts/input/nakamura_short.mp4`.

## Root Cause

The Carlsen failures were not a renderer, decode-order, or artifact-integrity
problem. The durable defect was inside `MediaPipeFaceObserver` candidate
arbitration:

- fallback scoring originally considered only each region selection's primary
  candidate, which let valid focused-region alternatives lose to worse
  full-frame/plaque-side choices;
- fallback precedence did not stop once a consensus-quality candidate existed,
  which let fallback logic override a better face candidate;
- the deterministic region set did not include focused crops that consistently
  exposed the visible upper-left player-camera face when the full-frame pass was
  weak or distracted by the nearby plaque/background.

The four committed fixes repaired that durable surface by scoring every valid
focused candidate, preserving consensus as the fallback boundary, and adding the
`left_upper_inner` and `left_upper_inner_nearby` crops.

## Durable Surface Changed

The repaired surface remains `src/chess_gaze/face_observation.py`,
specifically `MediaPipeFaceObserver` candidate generation and arbitration. Task
4 did not modify that module; it added real-video regression coverage and fresh
verification evidence around the already-committed repair.

## Regression Coverage

Added real-video regression:

- `test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region`

The test samples the reported Carlsen windows and adjacent controls:

- reported visible frames:
  `2036, 2037, 2042, 2050, 2062, 5694, 5695, 5697, 9029, 9030, 9031, 15079, 15080, 15081, 15082, 15083`
- adjacent controls:
  `2035, 2063, 5693, 5698, 9028, 9032, 15078, 15084`

For every sampled frame the regression asserts:

- the selected face is present;
- the primary selected center stays inside the visible person region:
  `450 <= x <= 660` and `240 <= y <= 430`.

Those bounds admit normal head motion across the sampled windows while rejecting
the old plaque/background selections, whose centers were around `x=750..775`,
`y=200..230`.

Existing Nakamura short real-video regressions remained unchanged and were
rerun as part of this closeout.

## Real-Video Evidence

### Fresh RED on pre-`23afa61` code

The new Carlsen regression was executed against a temporary worktree at
`d6a6753` with the new test copied in. It failed as expected:

```text
FAILED tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region
AssertionError: f000002042 selected center x=774.2 outside visible player-face bounds
```

That center lands in the historical plaque/background failure zone and confirms
the regression catches the pre-fix behavior.

### Fresh GREEN on current branch

Direct observer probing on the current branch over the reported windows and
adjacent controls returned present face selections with centers inside the
visible person region on all sampled frames:

```text
2035  center (555.2, 315.0)
2036  center (563.5, 324.2)
2037  center (568.4, 334.4)
2042  center (578.3, 347.6)
2050  center (575.1, 347.7)
2062  center (558.3, 338.6)
2063  center (556.9, 338.4)
5693  center (574.1, 381.2)
5694  center (574.2, 380.2)
5695  center (574.1, 379.9)
5697  center (574.3, 377.7)
5698  center (573.3, 376.5)
9028  center (565.9, 373.4)
9029  center (564.0, 374.3)
9030  center (562.8, 375.2)
9031  center (561.6, 375.2)
9032  center (560.7, 375.0)
15078 center (492.0, 321.0)
15079 center (492.0, 321.0)
15080 center (492.3, 320.8)
15081 center (492.6, 321.1)
15082 center (492.5, 321.2)
15083 center (492.3, 321.3)
15084 center (492.5, 321.1)
```

No sampled frame was missing a visible face, and none selected the historical
plaque/background region.

### Fresh Nakamura short analyze run

Command:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --output-root /private/tmp/chess-gaze-task4-output --models-root models
```

Observed result:

```text
/private/tmp/chess-gaze-task4-output/nakamura_short/runs/20260629T140320Z-eefca250
viewer: /private/tmp/chess-gaze-task4-output/nakamura_short/runs/20260629T140320Z-eefca250/viewer/index.html
```

Run validation:

```text
qa_summary.final_status: complete
qa_summary.counts.decoded_frames: 180
qa_summary.counts.frame_records: 180
qa_summary.counts.scene_frame_records: 180
qa_summary.rates.face_present_rate: 1.0
qa_summary.rates.head_pose_valid_rate: 1.0
scene_summary.valid_eye_midpoint_frames: 180
scene_summary.valid_unigaze_ray_frames: 180
scene_summary.valid_monitor_hit_frames: 180
scene_summary.artifact_validation.viewer_exists: true
```

## Third-Party Docs Checked

Verified on 2026-06-29:

- MediaPipe Face Landmarker Python guide:
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
  Confirmed `IMAGE`, `VIDEO`, and `LIVE_STREAM` running modes, and that
  `VIDEO`/`LIVE_STREAM` add tracking while `IMAGE` runs per input image.
- OpenCV drawing functions:
  https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
  Rechecked that renderer APIs draw the provided geometry directly, so renderer
  changes would not address a wrong selected face box.
- PyAV container API:
  https://pyav.basswood.io/docs/stable/api/container.html
  Rechecked that decoding remains a container/frame iteration boundary, not a
  candidate-arbitration boundary.

## Commands Run

Historical RED worktree setup:

```sh
git worktree add /private/tmp/chess-gaze-carlsen-red d6a6753
cp /Volumes/git/legotin/chess-gaze/tests/chess_gaze/test_face_observation_real_video.py /private/tmp/chess-gaze-carlsen-red/tests/chess_gaze/test_face_observation_real_video.py
mkdir -p /private/tmp/chess-gaze-carlsen-red/artifacts/input
mkdir -p /private/tmp/chess-gaze-carlsen-red/models/mediapipe
cp /Volumes/git/legotin/chess-gaze/artifacts/input/carlsen_1.mp4 /private/tmp/chess-gaze-carlsen-red/artifacts/input/carlsen_1.mp4
cp /Volumes/git/legotin/chess-gaze/models/mediapipe/face_landmarker.task /private/tmp/chess-gaze-carlsen-red/models/mediapipe/face_landmarker.task
```

Historical RED:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region -q
```

Outcome: `1 failed` on `d6a6753` with `f000002042 selected center x=774.2`.

Fresh GREEN:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region -q
```

Outcome: `1 passed in 24.27s`.

User-provided Nakamura selector command:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_recovers_nakamura_short_visible_faces tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_nakamura_short_faces_bounded -q
```

Outcome: `ERROR: not found` because the first selector omits `_face_` from the
actual test name.

Corrected targeted real-video verification:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_nakamura_short_visible_faces tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_nakamura_short_faces_bounded tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region -q
```

Outcome: `3 passed in 24.46s`.

Fresh end-to-end analyze:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --output-root /private/tmp/chess-gaze-task4-output --models-root models
```

Outcome: exit `0`; run `20260629T140320Z-eefca250` completed.

## Remaining Limitations

- Real MediaPipe verification requires host runtime access. The same Carlsen
  probe crashed inside the sandbox with:
  `graph_service.h:139 Check failed: service_ Service is unavailable.`
- I did not rerun a full `carlsen_1.mp4` analyze job in this task. The bounded
  direct observer probes covered all reported frames plus adjacent controls,
  which was the intended cheaper verification path.
- The fresh `nakamura_short` analyze run used the repo's default frame-image
  retention policy (`save_frame_images: false` in `run_manifest.json`), so the
  generated `raw_frames/` and `processed_frames/` directories are intentionally
  empty and no contact sheets were produced from that run.

## Source-Layout Review Note

`src/chess_gaze/face_observation.py` is currently `1207` lines, which is above
the repository's 800-line source-layout review trigger and below the 1,500-line
split-plan trigger.

This closeout did not change that file. The current size is still defensible
because one module owns a cohesive boundary: MediaPipe result normalization,
crop-region probing, candidate arbitration, and final face-observation assembly.
Before the next behavior expansion in this module, the project should plan a
split that preserves that cohesive surface while extracting internal arbitration
helpers or crop-region policy into named submodules with explicit tests.
