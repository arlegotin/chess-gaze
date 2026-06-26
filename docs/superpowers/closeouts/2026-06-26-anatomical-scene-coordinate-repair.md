# Anatomical Scene Coordinate Repair Closeout

Date: 2026-06-26

## Summary

Repaired the 3D scene coordinate contract so the persisted scene and generated
viewer are anatomical for frontal desktop webcam videos:

- `scene_pseudo_m +X` is the streamer's anatomical right, not image-right.
- `scene_pseudo_m +Y` is up.
- `scene_pseudo_m +Z` is the streamer's back.
- monitor-directed gaze is negative scene Z.
- image-right gaze from a face looking toward the webcam is the streamer's left
  and maps to negative scene X.

The fresh verified run is:

`artifacts/output/nakamura_1/runs/20260626T143547Z-da959b18`

Follow-up on 2026-06-26 found this repair fixed scene axes and ray directions
but did not repair source eye and PnP landmark labels that were still
image-side. See
`docs/superpowers/closeouts/2026-06-26-anatomical-eye-label-repair.md`.

## Root Cause

The previous fix used the wrong invariant. It preserved image-horizontal order
in the 3D scene, but the viewer is read by humans as a front-facing model of the
streamer. For a frontal webcam, image-right is the streamer's anatomical left.
That made frames where Nakamura looked to his left render as if he looked to his
right.

The same mistaken scene convention also inverted front/back semantics:

- previous scene `+Z` behaved like camera-back while the head offset still used
  camera `+Z` as behind the eyes;
- this placed the rendered eyes closer to the back side of the head;
- the generic blue `AxesHelper` Z label looked like a gaze-forward cue, while
  gaze-to-monitor should actually be negative scene Z under a human-centered
  right/up/back basis.

The durable boundary is the scene conversion layer, not the Three.js renderer.
Frame-record UniGaze overlay semantics remain unchanged: positive yaw is
image-right and positive pitch is image-up. Scene artifacts now convert that
canonical UniGaze vector into a physical frontal-webcam ray by negating camera
Y and Z at the scene boundary, then project it into a right-handed anatomical
basis:

```text
right_camera   = (-1,  0, 0)
up_camera      = ( 0, -1, 0)
back_camera    = ( 0,  0, 1)
forward_camera = ( 0,  0,-1)
```

## Recent Commit Audit

- `f103fb5 fix: preserve scene horizontal coordinates` fixed an image-ordering
  symptom but encoded the wrong invariant for a human-facing viewer.
- `f2f5924 test: type scene geometry helpers` did not change runtime behavior.
- `5c48768 docs: record scene horizontal coordinate repair` documented the
  incomplete image-side invariant and is now superseded.
- `55c9aa3 fix: align scene orientation manifest metadata` made metadata match
  the then-current implementation but preserved the wrong anatomical contract.
- `c4ef2e0 docs: plan anatomical scene coordinate repair` introduced the
  correct repair plan.
- `5370ba8 fix: align scene coordinates with streamer anatomy` implemented the
  durable scene-boundary and viewer-label fix.

No evidence was found that the recent commits damaged unrelated artifact
contracts. The damage was narrower but critical: they made the scene internally
consistent around image-space left/right instead of streamer anatomical
left/right.

## Third-Party Contract Checks

- OpenCV official calibration/pose documentation was checked for the camera
  convention used by this project: camera X follows image-right, Y follows
  image-down, and Z points forward into the image.
  Source: `https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html`
- Three.js official docs were checked for viewer behavior and axis rendering:
  `Vector3`, `PlaneGeometry`, `PerspectiveCamera`, `OrbitControls`, and
  `AxesHelper`.
  Sources:
  `https://threejs.org/docs/#api/en/math/Vector3`,
  `https://threejs.org/docs/#api/en/geometries/PlaneGeometry`,
  `https://threejs.org/docs/#api/en/cameras/PerspectiveCamera`,
  `https://threejs.org/docs/#examples/en/controls/OrbitControls`,
  `https://threejs.org/docs/#api/en/helpers/AxesHelper`
- Local installed UniGaze package source was checked. It outputs a 2-value gaze
  prediction; this project already normalizes frame-record yaw so positive yaw
  is image-right. The fix does not change that 2D overlay contract.
- Local MediaPipe usage was checked, but this closeout did not yet repair the
  image-side `left_eye`/`right_eye` source labels. That follow-up was handled in
  the anatomical eye-label repair closeout.

## Real-Artifact Evidence

Reported bad run:

`artifacts/output/nakamura_1/runs/20260626T123553Z-a0f00fd3`

Rebuilding only scene/viewer artifacts over a temp copy of that run produced:

- 1973 scene frames and 1973 viewer frames;
- orientation method `anatomical_frontal_webcam_right_up_back_axes`;
- basis determinant `1.0`;
- 0 image-right/his-left sign mismatches;
- 0 non-negative scene-Z gaze rays;
- 0 eye-behind-head cases;
- 0 viewer scene-data direction mismatches;
- 1971 valid monitor hits and 2 invalid behind-origin intersections.

Fresh full real-video run:

`artifacts/output/nakamura_1/runs/20260626T143547Z-da959b18`

Numeric audit across all 1973 frames:

- orientation method: `anatomical_frontal_webcam_right_up_back_axes`;
- scene axes in camera: right `(-1,0,0)`, up `(0,-1,0)`, back `(0,0,1)`,
  forward `(0,0,-1)`;
- determinant: `1.0`;
- scene/viewer frame counts: `1973 / 1973`;
- valid monitor hits: `1971`;
- invalid monitor hits: frames `1692` and `1693`,
  `RAY_INTERSECTION_BEHIND_ORIGIN`;
- `x_sign_mismatch_against_unigaze_yaw`: `0`;
- `z_nonnegative`: `0`;
- `nonunit_scene_direction`: `0`;
- `eye_not_front_of_head`: `0`;
- `viewer_direction_mismatch`: `0`.

Representative frames:

| Frame | UniGaze yaw | Scene direction | Interpretation |
| --- | ---: | --- | --- |
| 90 | `+0.424626` | `x=-0.371198, y=-0.433801, z=-0.820993` | image-right / streamer's left |
| 154 | `+1.186927` | `x=-0.841733, y=-0.419401, z=-0.339981` | image-right / streamer's left |
| 1568 | `-1.131385` | `x=+0.815115, y=+0.434489, z=-0.383154` | image-left / streamer's right |
| 1651 | `-0.423431` | `x=+0.277171, y=+0.738220, z=-0.614986` | image-left / streamer's right |

For those four frames, every eye midpoint is in front of the head center:
`eye_midpoint.scene_m.z < head.scene_m.z`.

## Browser Verification

Fresh viewer served from:

`http://127.0.0.1:58270/`

Chrome DevTools verification:

- rendered title: `Chess Gaze Scene Viewer`;
- frame count: `1973`;
- valid hit count: `1971`;
- axis labels: `X streamer right`, `Y scene up`, `Z streamer back`;
- WebGL2 canvas present with size `2100 x 2444`;
- no console messages;
- network requests limited to local viewer files and pinned Three.js
  `0.185.0` module URLs.

Screenshots captured:

- `/private/tmp/chess-gaze-fresh-viewer-frame0.png`
- `/private/tmp/chess-gaze-fresh-viewer-frame90.png`
- `/private/tmp/chess-gaze-fresh-viewer-frame154.png`
- `/private/tmp/chess-gaze-fresh-viewer-frame1568.png`
- `/private/tmp/chess-gaze-fresh-viewer-frame1651.png`

Viewport screenshot pixel checks were nonblank for all captured frames.
Visual inspection matched the corrected signs: frames 90 and 154 render toward
the streamer's left, while 1568 and 1651 render toward the streamer's right from
the front-facing viewer camera.

## Verification Commands

Red evidence before production changes:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_viewer.py -q
```

Failed 23 tests at the intended scene-coordinate and viewer-label seams.

Green evidence after production changes, rerun after the closeout was written:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_viewer.py -q
```

Result: `96 passed in 2.55s` when run with loopback permission for the local
viewer-server tests.

Additional local gates:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_qa_summary.py -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Results:

- `8 passed in 2.52s`;
- `12 passed in 4.24s`;
- `All checks passed!`;
- `56 files already formatted`;
- `Success: no issues found in 56 source files`.

Full-suite limitation:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

This did not produce a clean full-suite pass in the local checkout. Four tests
failed before coordinate-related tests because mandatory fixture videos are not
present locally:

- `artifacts/input/test_1.mp4`
- `artifacts/input/test_2.mp4`

The exact failing files were:

- `tests/chess_gaze/test_pipeline_real_video_contract.py`
- `tests/chess_gaze/test_qa_summary_real_video_contract.py`

A rerun excluding those two unavailable-video contract files reached
`130 passed, 7 skipped` with no failures before being interrupted in
image-heavy real-video QA work. This limitation is not evidence of a
coordinate-regression failure; the required `nakamura_1.mp4` model-backed run
below is the real-video verification for this coordinate repair.

Real run:

```sh
MPLCONFIGDIR=/Volumes/git/legotin/chess-gaze/.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

Result: exited `0` and printed:

```text
artifacts/output/nakamura_1/runs/20260626T143547Z-da959b18
viewer: artifacts/output/nakamura_1/runs/20260626T143547Z-da959b18/viewer/index.html
```

## Remaining Limitations

- Mirror policy is still unknown. The current contract is correct for the
  frontal desktop webcam assumption used by `nakamura_1.mp4`; mirrored webcam
  feeds need an explicit persisted mirror flag before anatomical left/right can
  be guaranteed.
- Absolute distances remain pseudo-meters derived from IPD and monitor-size
  assumptions, not calibrated room geometry.
- The two invalid monitor hits in the fresh run are expected ray-plane geometry
  outcomes, not coordinate sign failures.
