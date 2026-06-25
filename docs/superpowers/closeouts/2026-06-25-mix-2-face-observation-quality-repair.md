# Mix 2 Face Observation Quality Repair Closeout

Date: 2026-06-25

## Summary

Repaired the reported face-observation failures in
`artifacts/output/mix_2/runs/20260625T203626Z-f401b16f`.

The durable fix is split across two runtime boundaries:

- `MediaPipeFaceObserver` now keeps clear full-frame detections fast, but when a
  full-frame observation is missing or geometrically suspect it evaluates all
  deterministic recovery regions before final selection.
- `ModelBackedFrameObserver` now classifies frames with only warning-level
  diagnostics as `WARNING`, not `ERROR`, when face, eyes, head pose, appearance
  gaze, and the applicable recommended-gaze contract are otherwise valid.

Visualization was not changed. It was rendering persisted frame-record
coordinates correctly; the bad overlays came from selected face candidates
before rendering.

## Root Cause

The original failures were all upstream of rendering:

1. `f000000237`: full-frame detection missed, left-half detection returned two
   candidates, and area-only selection chose the larger false candidate on the
   board/background instead of the visible face.
2. `f000000265`, `f000000266`, `f000000268`: full-frame detection returned a
   single large false positive over the board/background and the observer
   accepted it before considering focused regions.
3. `f000000422`-`f000000423`, `f000000510`-`f000000524`, and `f000000532`:
   full, left-half, and right-half regions were too broad for the small
   right-side webcam faces; smaller upper/right regions recovered them.
4. The review pass exposed a status-contract issue: frames `f000000483` and
   `f000000484` had visually correct selected faces but were representative
   failures because `MULTIPLE_FACE_CANDIDATES` plus
   `GAZE_ESTIMATORS_DISAGREE` was promoted to frame `ERROR`.

Artifact integrity was not the cause. The old run had `540` decoded frames,
`540` raw frames, `540` processed frames, `540` frame records, matching counts,
and schema validation passed.

## Durable Changes

- Expanded deterministic IMAGE-mode detection regions to include full frame,
  left/right halves, left/right top regions, left/right upper bands, and a
  right-upper-middle band for compact webcam panes.
- Removed the premature half-frame/basic-selection return so non-clear frames
  are arbitrated after all deterministic region evidence is available.
- Required cross-region IoU consensus before replacing a large full-frame
  candidate with a smaller focused-region candidate.
- Generalized seam clipping checks to internal x and y region boundaries, not
  only left/right half boundaries.
- Kept focused-region replacement bounded by plausible face area, area ratio,
  geometry score, and existing candidate scoring.
- Added a frame-status warning-only diagnostic set for
  `GAZE_ESTIMATORS_DISAGREE` and `MULTIPLE_FACE_CANDIDATES`.

## Evidence

Original run:

- `artifacts/output/mix_2/runs/20260625T203626Z-f401b16f`
- `FACE_NOT_FOUND`: `36` record errors.
- `MULTIPLE_FACE_CANDIDATES`: `1` record error.
- `face_present_rate`: `0.9666666666666667`.
- Representative failures: `f000000237`, `f000000422`,
  `f000000423`, `f000000510`-`f000000524`, `f000000532`.

Final repaired run:

- `artifacts/output/mix_2/runs/20260625T213141Z-5d044b42`
- `540` decoded frames, `540` raw frames, `540` processed frames, `540` frame
  records, and `1080` crop files.
- Schema validation passed, counts matched, final status `complete`.
- `FACE_NOT_FOUND`: `0`.
- `errors_by_code`: `GAZE_ESTIMATORS_DISAGREE: 395`,
  `MULTIPLE_FACE_CANDIDATES: 2`.
- `errors_by_severity`: `warning: 397`.
- `face_present_rate`, `both_eyes_present_rate`, `head_pose_valid_rate`, and
  `face_gaze_valid_rate`: `1.0`.
- Representative failures: none.

Reported-frame before/after:

```text
f000000237 old box (453.0, 114.5, 588.9, 265.8) -> new box (320.7, 155.7, 432.0, 281.4)
f000000265 old box (381.8, 75.0, 597.4, 349.6) -> new box (288.6, 159.4, 399.2, 287.8)
f000000266 old box (376.8, 67.4, 590.2, 350.7) -> new box (288.7, 159.5, 399.3, 287.8)
f000000268 old box (381.3, 72.5, 597.4, 352.4) -> new box (288.4, 159.4, 399.5, 288.0)
f000000422 old FACE_NOT_FOUND -> new box (974.6, 228.7, 1069.2, 337.3)
f000000423 old FACE_NOT_FOUND -> new box (975.0, 228.6, 1069.4, 337.4)
f000000510 old FACE_NOT_FOUND -> new box (920.8, 200.6, 1009.0, 301.6)
f000000524 old FACE_NOT_FOUND -> new box (920.5, 201.8, 1008.3, 301.5)
f000000532 old FACE_NOT_FOUND -> new box (831.3, 183.6, 937.4, 310.3)
```

Visual contact sheets generated and inspected:

- `/private/tmp/chess-gaze-contact-sheets-final/issue_237_before_after.jpg`
- `/private/tmp/chess-gaze-contact-sheets-final/issue_265_268_before_after.jpg`
- `/private/tmp/chess-gaze-contact-sheets-final/issue_422_423_before_after.jpg`
- `/private/tmp/chess-gaze-contact-sheets-final/issue_422_423_face_zoom.jpg`
- `/private/tmp/chess-gaze-contact-sheets-final/issue_510_524_before_after.jpg`
- `/private/tmp/chess-gaze-contact-sheets-final/issue_510_524_face_zoom.jpg`
- `/private/tmp/chess-gaze-contact-sheets-final/issue_532_before_after.jpg`
- `/private/tmp/chess-gaze-contact-sheets-final/issue_532_face_zoom.jpg`
- `/private/tmp/chess-gaze-contact-sheets-final/warning_483_484_before_after.jpg`

Visual inspection confirmed:

- `f000000237` now stays on the left webcam face and is continuous with
  adjacent frames.
- `f000000265`-`f000000266` and `f000000268` no longer jump to the board.
- `f000000422`-`f000000423` now have overlays on the right-side webcam face.
- The `f000000510`-`f000000524` gap now has face overlays, including the
  user-reported `f000000510` and `f000000524`.
- `f000000532` now has an overlay consistent with `f000000531` and
  `f000000533`.
- `f000000483`-`f000000484` remain visually stable and are now warning-only,
  not representative failures.

Primary-source checks used during investigation:

- MediaPipe Face Landmarker Python docs, verified 2026-06-25: IMAGE mode uses
  `detect()`, VIDEO mode uses `detect_for_video()`, and VIDEO/LIVE_STREAM modes
  use tracking behavior for latency. Current repo calibration intentionally
  uses IMAGE mode, so the fix stays deterministic per frame.
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
- MediaPipe Face Detector Python docs, verified 2026-06-25: face detection has
  confidence and NMS options, but this repair did not select a new detector or
  dependency.
  https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python
- OpenCV drawing docs, verified 2026-06-25: rectangle, text, and drawing
  functions render provided geometry directly; this supported keeping the fix
  upstream of visualization.
  https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html

## Regression Coverage

Added unit regressions for:

- full-frame miss plus wrong area-largest half-frame candidate, rescued by a
  tighter top-left region;
- tall large full-frame false positive, replaced by a focused left/top face;
- small right-top webcam face recovered after full and half-frame misses;
- small right-upper-middle webcam face recovered after full and half-frame
  misses;
- focused regions still scanned before accepting an ambiguous half-frame face;
- large full-frame faces not replaced without focused-region consensus;
- seam-clipped focused-region candidates rejected on internal x/y boundaries;
- multiple face candidates with otherwise valid outputs marked `WARNING`;
- multiple face candidates plus recommended-gaze disagreement marked `WARNING`.

Added real-video regression:

- `tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_mix2_reported_visible_faces`
  samples `mix_2.mp4` frames `237`, `265`, `266`, `268`, `422`, `423`, `510`,
  `524`, and `532`, asserts a selected face is present, checks center location
  against the manually observed visible-face region, and records the expected
  repaired boxes.

## Verification

Fresh checks:

```sh
uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_mix2_reported_visible_faces tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_qa_summary.py -q
# 36 passed

uv run chess-gaze analyze artifacts/input/mix_2.mp4
# artifacts/output/mix_2/runs/20260625T213141Z-5d044b42

uv run pytest
# 129 passed, 7 failed, 7 skipped, 18 warnings
# All 7 failures are missing-file assertions for artifacts/input/test_1.mp4
# or artifacts/input/test_2.mp4, which are absent from this checkout.

uv run pytest --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
# 129 passed, 7 skipped, 18 warnings

uv run ruff check .
# All checks passed

uv run ruff format --check .
# 44 files already formatted

uv run mypy
# Success: no issues found in 44 source files
```

MediaPipe emitted non-fatal runtime warnings during the full `mix_2`
regeneration:

- duplicate AVFoundation receiver classes from bundled `cv2` and `av`
  libraries;
- offline Clearcut upload warnings.

The CLI completed successfully despite those warnings.

## Source-Layout Review

`src/chess_gaze/face_observation.py` is now `1047` lines, so the
`docs/development/architecture/source-layout.md` review threshold applies.

The file still owns one coherent deep-module responsibility: MediaPipe face
observation and candidate arbitration, including the private region and
geometry helpers required to return a single full-image `FaceSelection`. I did
not split it because extracting the helpers now would create pass-through files
without a stable independent seam. If candidate provenance becomes persisted,
or if another observer reuses the scoring policy, split the region/candidate
arbitration helpers behind an explicit domain interface.

## Remaining Limitations

This repair improves deterministic region coverage and arbitration for
picture-in-picture chess-stream layouts. It does not add temporal smoothing or
tracking, because the project currently records independent IMAGE-mode
observations and smoothing would hide bad candidate selection rather than fix
the observation boundary.

Frame records still persist only the selected face, not all candidate
provenance. That limits after-the-fact artifact debugging; a future schema
change could preserve candidate regions and scores as diagnostics, but it was
not required to fix the reported behavior.
