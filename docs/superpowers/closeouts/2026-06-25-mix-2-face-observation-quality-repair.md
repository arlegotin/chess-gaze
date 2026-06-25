# Mix 2 Face Observation Quality Repair Closeout

Date: 2026-06-25

## Summary

Repaired the reported face-observation failures in
`artifacts/output/mix_2/runs/20260625T203626Z-f401b16f`.

The durable fix is in `MediaPipeFaceObserver`: it now preserves the existing
fast path for clear full-frame detections, but expands deterministic recovery
regions when full-frame and half-frame evidence is ambiguous or missing. The
observer can now recover small picture-in-picture webcam faces from upper
subregions, and it can replace implausibly large full-frame false positives
with a smaller focused-region face candidate.

Visualization was not changed. It was drawing the persisted frame-record
coordinates correctly; the bad overlays were caused by selected face candidates
before rendering.

## Root Cause

There were three related face-observation failures:

1. `f000000237`: full-frame detection missed, left-half detection returned two
   candidates, and area-only selection chose the larger false candidate on the
   board/background instead of the visible face. A tighter top-left region
   isolates the real face.
2. `f000000265`, `f000000266`, `f000000268`: full-frame detection returned a
   single large false positive over the board/background. Because the candidate
   was tall and high in the image, the old refinement trigger accepted it early
   and never considered the correct left-half face candidate.
3. `f000000422`-`f000000423`, `f000000510`-`f000000524`, and `f000000532`:
   full, left-half, and right-half regions were still too broad for the small
   right-side webcam faces. Smaller top/right regions recovered those visible
   faces.

Artifact integrity was not the cause. The old run had `540` decoded frames,
`540` raw frames, `540` processed frames, `540` frame records, matching counts,
and schema validation passed.

## Evidence

Original run:

- `artifacts/output/mix_2/runs/20260625T203626Z-f401b16f`
- `FACE_NOT_FOUND`: `36` record errors.
- `MULTIPLE_FACE_CANDIDATES`: `1` record error.
- `face_present_rate`: `0.9666666666666667`.
- Representative failures: `f000000237`, `f000000422`,
  `f000000423`, `f000000510`-`f000000524`, `f000000532`.

Fresh repaired run:

- `artifacts/output/mix_2/runs/20260625T210528Z-2139b3b0`
- `540` decoded frames, `540` raw frames, `540` processed frames, and `540`
  frame records.
- Schema validation passed, counts matched, final status `complete`.
- `FACE_NOT_FOUND`: `0`.
- `MULTIPLE_FACE_CANDIDATES`: `0`.
- `face_present_rate`, `both_eyes_present_rate`, `head_pose_valid_rate`, and
  `face_gaze_valid_rate`: `1.0`.
- Representative failures: none.

Reported-frame before/after:

```text
f000000237 old box (453.0, 114.5, 588.9, 265.8) -> new box (320.7, 155.7, 432.0, 281.4)
f000000265 old box (381.8, 75.0, 597.4, 349.6) -> new box (289.0, 156.1, 398.8, 288.6)
f000000266 old box (376.8, 67.4, 590.2, 350.7) -> new box (288.9, 156.2, 398.7, 288.7)
f000000268 old box (381.3, 72.5, 597.4, 352.4) -> new box (288.7, 156.9, 399.1, 289.0)
f000000422 old FACE_NOT_FOUND -> new box (975.2, 229.7, 1069.6, 339.7)
f000000423 old FACE_NOT_FOUND -> new box (973.8, 230.0, 1069.5, 339.6)
f000000510 old FACE_NOT_FOUND -> new box (920.8, 200.6, 1009.0, 301.6)
f000000524 old FACE_NOT_FOUND -> new box (920.5, 201.8, 1008.3, 301.5)
f000000532 old FACE_NOT_FOUND -> new box (831.3, 183.6, 937.4, 310.3)
```

Visual inspection of contact sheets confirmed:

- `f000000237` now stays on the left webcam face and is continuous with
  adjacent frames.
- `f000000265`-`f000000266` and `f000000268` no longer jump to the board.
- `f000000422`-`f000000423` now have overlays on the right-side webcam face.
- The full `f000000510`-`f000000524` gap now has face overlays, including the
  user-reported `f000000510` and `f000000524`.
- `f000000532` now has an overlay consistent with `f000000531` and
  `f000000533`.

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
- tall large full-frame false positive, replaced by the focused left-half face;
- small right-top webcam face recovered after full and half-frame misses;
- small right-upper-middle webcam face recovered after full and half-frame
  misses.

Added real-video regression:

- `tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_mix2_reported_visible_faces`
  samples `mix_2.mp4` frames `237`, `265`, `266`, `268`, `422`, `423`, `510`,
  `524`, and `532`, asserts a selected face is present, checks center location
  against the manually observed visible-face region, and records the expected
  repaired boxes.

## Verification

Fresh passing checks:

```sh
uv run pytest tests/chess_gaze/test_face_observation.py -q
# 18 passed

uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_mix2_reported_visible_faces -q
# 1 passed

uv run chess-gaze analyze artifacts/input/mix_2.mp4
# artifacts/output/mix_2/runs/20260625T210528Z-2139b3b0

uv run pytest --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
# 121 passed, 7 skipped, 18 warnings

uv run ruff check .
# All checks passed

uv run ruff format --check .
# 44 files already formatted

uv run mypy
# Success: no issues found in 44 source files
```

Full `uv run pytest` was also run. It reported `121 passed, 7 failed, 7
skipped, 18 warnings`; all 7 failures were missing-file assertions for
`artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4`, which are absent
from this checkout.

MediaPipe emitted non-fatal runtime warnings during the full `mix_2`
regeneration:

- duplicate AVFoundation receiver classes from bundled `cv2` and `av`
  libraries;
- offline Clearcut upload warnings.

The CLI completed successfully despite those warnings.

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
