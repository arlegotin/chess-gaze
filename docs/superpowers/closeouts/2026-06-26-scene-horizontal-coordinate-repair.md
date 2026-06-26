# Scene Horizontal Coordinate Repair Closeout

Date: 2026-06-26

## Summary

Fixed a critical 3D scene coordinate bug where camera/image-right gaze could be
rendered as scene-left. The bug was not a one-frame artifact and was not caused
by swapped eye labels or raw UniGaze yaw sign. It came from using the robust
dominant UniGaze direction as both the scene depth axis and the monitor surface
normal. That made scene right an oblique vector with a large depth component, so
forward depth could overpower horizontal direction and flip scene X.

The durable fix is to keep the scene basis camera-stable:

- `right_camera = (1, 0, 0)`;
- `up_camera = (0, -1, 0)`;
- `back_camera = (0, 0, -1)`;
- `forward_camera = (0, 0, 1)`;
- monitor normal equals `back_camera`;
- robust dominant UniGaze direction is used only to place the inferred monitor
  center.

## Root Cause Evidence

Reported run:
`artifacts/output/nakamura_1/runs/20260626T104848Z-21353a29`

- Frame 90 processed overlay visibly points image-right/down.
- Frame 90 raw scene data had `direction_camera.x = +0.3711975714751555` but
  `direction_scene.x = -0.2008572172489131`.
- Old axes had `right_camera = (0.7964226551329736, 0.0,
  -0.6047404024793983)` and `back_camera = (-0.5935890162117526,
  -0.19115407280621485, -0.7817366566065327)`.
- Old monitor normal equaled old `back_camera`.
- Straight-ahead-relative monitor-U ordering mismatched horizontal camera
  direction in 88 of 1958 compared valid rays.

Subagent findings agreed on the failure boundary:

- artifact/visual analysis found no left/right eye swap and identified the
  mismatch at scene basis and monitor projection;
- code/docs analysis traced frame overlays to raw yaw/pitch but scene rays to
  the oblique axis basis;
- dependency analysis confirmed the local contracts: image X grows right,
  OpenCV camera X grows image-right, Three.js renders supplied vectors, and the
  viewer does not independently flip coordinates.

## Third-Party Contracts Checked

Primary documentation checked on 2026-06-26:

- OpenCV camera/projection and PnP docs:
  `https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html`,
  `https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html`;
- MediaPipe Face Landmarker and Face Mesh docs:
  `https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python`,
  `https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md`;
- PyAV `VideoFrame` docs:
  `https://pyav.org/docs/stable/api/video.html`;
- Three.js `Vector3`, `PlaneGeometry`, `PerspectiveCamera`, and `OrbitControls`
  docs:
  `https://threejs.org/docs/#api/en/math/Vector3`,
  `https://threejs.org/docs/#api/en/geometries/PlaneGeometry`,
  `https://threejs.org/docs/#api/en/cameras/PerspectiveCamera`,
  `https://threejs.org/docs/#examples/en/controls/OrbitControls`;
- UniGaze repository, package metadata, and model manifest:
  `https://github.com/ut-vision/UniGaze`,
  `https://pypi.org/project/unigaze/`,
  `https://huggingface.co/UniGaze/UniGaze-models`.

These sources support the local coordinate assumptions used by the fix. They do
not provide evidence that monocular gaze can infer the physical monitor surface
normal, so the old axis/normal inference was unjustified.

## Code Changed

Commit `f103fb5` (`fix: preserve scene horizontal coordinates`) changed:

- `src/chess_gaze/scene_geometry.py`
  - `build_scene_axis_basis()` now returns fixed camera-stable
    right/up/back/forward axes with determinant `+1`;
  - `build_monitor_plane()` still places the monitor center along robust
    dominant UniGaze direction but uses the camera-stable scene normal and
    right/up basis;
  - obsolete axis-projection fallback helpers were removed.
- `tests/chess_gaze/test_scene_geometry.py`
  - added a regression using the reported Nakamura frame-90 direction and old
    oblique dominant direction;
  - added a regression that monitor center may be oblique while monitor normal
    stays camera-stable;
  - updated monitor intersection expectations for the frontoparallel scene
    plane.

Docs updated in this closeout pass:

- `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`;
- `docs/superpowers/plans/2026-06-26-3d-scene-artifact-viewer.md`;
- `docs/superpowers/plans/2026-06-26-scene-horizontal-coordinate-repair.md`.

Review follow-up:

- A final review found that runtime scene vectors were fixed, but
  `scene_manifest.robust_estimators.scene_orientation` still advertised the old
  eye-pair/head-up estimator. That would mislead artifact consumers.
- The manifest schema and writer now persist
  `method = "camera_stable_right_up_back_axes"` and
  `candidate_frame_count = 0`.
- The historical implementation plan no longer instructs future agents to add
  right-vs-forward/up-vs-normal projection fallback behavior.

## Real Video Verification

Fresh run from `artifacts/input/nakamura_1.mp4`:
`artifacts/output/nakamura_1/runs/20260626T123553Z-a0f00fd3`

Command:

```sh
MPLCONFIGDIR=/Volumes/git/legotin/chess-gaze/.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

Result:

- analyzer completed cleanly and printed the fresh run/viewer paths;
- generated 1973 `frames.jsonl` records, 1973 `scene_frames.jsonl` records, and
  1973 viewer frames;
- axes are fixed camera-stable axes with `determinant_right_up_back = 1.0`;
- monitor normal is `(0, 0, -1)`;
- 1973 valid rays, 1973 valid monitor hits;
- 0 scene X sign mismatches against camera X;
- 0 scene Y sign mismatches after the intended camera-down to scene-up flip;
- 0 positive scene-Z directions;
- 0 viewer direction payload mismatches;
- 0 straight-ahead-relative monitor-U ordering mismatches out of 1973 compared
  valid rays.

Representative frames:

- frame 90: camera X `+0.3711975714751555`, scene X
  `+0.3711975714751555`, straight-ahead-relative `delta_u =
  +0.269131707767523`;
- frame 154: camera X `+0.8417334780683987`, scene X
  `+0.8417334780683987`;
- frame 1568: camera X `-0.8151148789774965`, scene X
  `-0.8151148789774965`;
- frame 1651: camera Y `-0.7382197511455898`, scene Y
  `+0.7382197511455898`.

Known runtime warnings during real runs:

- macOS duplicate `AVFFrameReceiver`/`AVFAudioReceiver` classes from both
  OpenCV and PyAV FFmpeg dylibs;
- MediaPipe Clearcut telemetry upload warnings.

Neither warning changed the generated coordinate artifacts.

## Browser Verification

Fresh viewer served from:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view artifacts/output/nakamura_1/runs/20260626T123553Z-a0f00fd3 --host 127.0.0.1 --port 57632
```

Chrome verification:

- viewer loaded 1973 frames and 1973 hit points;
- frames 90, 154, 1568, and 1651 exposed the same
  `unigaze_ray.direction_scene` values as `records/scene_frames.jsonl`;
- screenshots saved:
  - `/private/tmp/chess-gaze-fresh-viewer-frame90.png`;
  - `/private/tmp/chess-gaze-fresh-viewer-frame154.png`;
  - `/private/tmp/chess-gaze-fresh-viewer-frame1568.png`;
  - `/private/tmp/chess-gaze-fresh-viewer-frame1651.png`;
- visual inspection matched processed-frame overlay direction for frame 90
  image-right/down and frame 1568 image-left/up;
- browser console had no messages;
- viewer document, CSS, `scene-data.json`, and pinned Three.js modules returned
  HTTP 200.

## Test Evidence

Red test evidence before the code fix:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py::test_scene_axis_basis_does_not_rotate_camera_right_into_scene_left tests/chess_gaze/test_scene_geometry.py::test_build_monitor_plane_keeps_camera_stable_normal_for_oblique_center -q
```

Result: expected failures under the old oblique basis/monitor normal behavior.

Green focused evidence after the code fix:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Result: `41 passed`.

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_viewer.py -q
```

Result: `52 passed` when rerun with localhost-binding permissions for viewer
tests.

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_geometry.py tests/chess_gaze/test_scene_geometry.py
```

Result: passed.

Broad-gate evidence is recorded in the repair plan.

Post-review evidence:

- Targeted manifest metadata tests failed before the review fix and passed after
  it with `2 passed in 1.97s`.
- Focused scene suite passed after the review fix with `93 passed in 2.93s`.
- Broad available subset passed after the review fix with
  `237 passed, 7 skipped, 18 warnings` in 587.41s.
- Final `ruff check .`, `ruff format --check .`, and `mypy` passed.

## Residual Risk

- The scene remains pseudo-metric and frontoparallel, not calibrated physical
  monitor pose. This is intentional because the input video does not prove
  camera-to-monitor orientation.
- `plane_uv_m[0]` is relative to inferred monitor center. A negative U value can
  still occur for a camera-right ray if the inferred center is farther right
  than the hit. Use `unigaze_ray.direction_scene.x`, or compare U against a
  straight-ahead ray from the same origin, for left/right semantics.
- The duplicate OpenCV/PyAV FFmpeg dylib warning is outside this coordinate fix
  and should be addressed separately if native video failures appear.
