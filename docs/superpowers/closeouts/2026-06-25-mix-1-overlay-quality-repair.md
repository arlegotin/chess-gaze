# Mix 1 Overlay Quality Repair Closeout

Date: 2026-06-25

## Summary

Repaired the overlay-placement failure observed in
`artifacts/output/mix_1/runs/20260625T192919Z-94dffb12`.

The durable fix is in face-region arbitration. `MediaPipeFaceObserver` no
longer lets a questionable full-frame detection suppress a better focused
half-frame detection. It evaluates all eligible half-frame refinements, rejects
seam-clipped candidates, requires overlap with the full-frame detection set, and
selects by overlap-adjusted area instead of fixed left-before-right order.

The white recommended-gaze arrow is now treated according to its actual
semantics: if `recommended_gaze.valid` is false because the eye-derived gaze and
UniGaze disagree, the frame is `WARNING`, not `ERROR`. The renderer colors that
state separately and QA representative failures exclude warning-only records.

## Root Cause

The old observer stopped at the first successful full-frame face-landmarker
selection. In the bad ranges, MediaPipe produced a plausible but poorer
full-frame face mesh:

- `f000000040`-`f000000044`: full-frame detection returned multiple candidates
  and selected the larger lower-face/hand-biased candidate. The focused
  right-half detection produced the compact head box but was never considered.
- `f000000180`-`f000000189`: after the scene cut, full-frame detection sometimes
  returned a low partial face on cheek/hand. The focused left-half detection
  produced the more stable upper face box but was suppressed.

Visualization was drawing the stored record coordinates correctly. The wrong
placement came from candidate arbitration before eye, head-pose, and gaze
records were built.

## Evidence

Original run:

- `artifacts/output/mix_1/runs/20260625T192919Z-94dffb12`
- `f000000040` old face box: `(985.1, 226.8, 1152.8, 454.3)`
- `f000000044` old face box: `(987.6, 230.6, 1167.7, 449.2)`
- `f000000180` old face box: `(270.5, 499.7, 392.0, 589.2)`
- `f000000189` old face box: `(258.8, 485.2, 382.5, 603.1)`
- QA summary counted `GAZE_ESTIMATORS_DISAGREE` and
  `MULTIPLE_FACE_CANDIDATES` as warnings, but frame status still rendered red
  `ERROR` for warning-only gaze divergence.

Regenerated run:

- `artifacts/output/mix_1/runs/20260625T201751Z-a350a22a`
- `240` decoded frames, `240` frame records, `240` raw frames, `240` processed
  frames; schema validation passed and final status is `complete`.
- `f000000040` new face box: `(984.7, 207.8, 1148.1, 367.6)`
- `f000000044` new face box: `(984.5, 205.2, 1144.8, 364.1)`
- `f000000180` new face box: `(246.7, 444.7, 354.8, 567.3)`
- `f000000189` new face box: `(247.3, 445.2, 353.7, 566.1)`
- `f000000040`-`f000000044` now render compact boxes over the visible head
  region instead of the lower cheek/hand region.
- `f000000180`-`f000000189` now render the stable left-half face box over the
  upper visible head region. Hand occlusion remains visible and landmarks are
  still uncertain, but the gross coordinate-placement error is removed.
- Warning-only divergence frames render as `WARNING`; missing white `rec` arrow
  is expected when `recommended_gaze.valid=false`.

Primary-source checks used during investigation:

- MediaPipe Face Landmarker docs: IMAGE mode is for independent images, VIDEO
  and LIVE_STREAM modes use temporal tracking behavior, and the task returns
  face landmarks, blendshapes, and facial transformation matrices.
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- MediaPipe Python guide: `detect()` blocks on image-mode inference and returns
  `FaceLandmarkerResult` containing normalized landmarks.
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
- OpenCV drawing docs: drawing functions clip shapes outside image boundaries
  and draw directly on image matrices; this supported keeping the fix upstream
  of visualization.
  https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html

## Regression Coverage

Added regression tests for:

- Ambiguous full-frame multiple candidates where the focused right-half
  candidate must be preferred.
- Low/partial full-frame detection where the focused left-half candidate must be
  preferred.
- Both halves returning valid refinements, proving arbitration is scored rather
  than left-before-right order-dependent.
- Warning-only `GAZE_ESTIMATORS_DISAGREE` producing `FrameStatus.WARNING`.
- QA summary excluding warning-only records from representative failures.

## Verification

Fresh passing checks:

```sh
uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_qa_summary.py -q
# 23 passed

uv run pytest --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
# 116 passed, 7 skipped, 18 warnings

uv run ruff check .
# All checks passed

uv run ruff format --check .
# 44 files already formatted

uv run mypy
# Success: no issues found in 44 source files

uv run chess-gaze analyze artifacts/input/mix_1.mp4
# artifacts/output/mix_1/runs/20260625T201751Z-a350a22a
```

Full `uv run pytest` was attempted and failed only because local mandatory
verification files `artifacts/input/test_1.mp4` and
`artifacts/input/test_2.mp4` are absent in this checkout. The attempted full run
reported `116 passed, 7 failed, 7 skipped, 18 warnings`; all 7 failures were
missing-file assertions for those two videos.

## Remaining Limitations

The repair does not solve all landmark uncertainty under severe hand occlusion.
Frames `f000000180`-`f000000189` still contain a hand covering much of the face,
so eye/iris landmarks can remain imperfect. The change removes the gross wrong
face-region selection and correctly records the remaining gaze divergence as a
warning state rather than an error.

MediaPipe emitted non-fatal runtime warnings during regeneration:

- duplicate AVFoundation receiver classes from bundled `cv2` and `av`
  libraries;
- offline Clearcut upload warning.

Neither warning stopped analysis or artifact validation.
