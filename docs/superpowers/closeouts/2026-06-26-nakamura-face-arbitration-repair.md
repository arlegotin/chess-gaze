# Nakamura Face Arbitration Repair Closeout

Date: 2026-06-26

## Summary

Repaired intermittent bad face boxes and landmarks in
`artifacts/output/nakamura_1/runs/20260626T160949Z-bbc6c27e`.

The durable fix is in `MediaPipeFaceObserver` region arbitration. When the
full-frame MediaPipe pass returns a single valid but overexpanded face candidate,
the observer now evaluates focused deterministic regions and may replace that
candidate with a compact overlapping focused-region candidate. The replacement
is bounded by frame size, full-frame area, focused-candidate plausibility,
overlap, material compactness, and geometry-score improvement.

Implementation commits:

- `352696f docs: plan nakamura face arbitration repair`
- `1234348 test: reproduce nakamura face arbitration jumps`
- `ca59e47 fix: prefer compact overlapping face refinements`
- `452a64e fix: bound overexpanded face refinements`

Final review found that the first implementation let overexpanded full-frame
candidates enter older larger-candidate and top-shift refinement paths. The
final code preserves the pre-existing large-full-frame consensus path first,
then uses the new bounded overexpanded helper exclusively for remaining single
overexpanded full-frame candidates.

## Root Cause

The old observer often returned early after a single full-frame candidate. In
the bad Nakamura frames, MediaPipe's full-frame `IMAGE` detection produced
finite, schema-valid landmarks that covered the real face plus
headphones/background. Because the selected candidate passed schema validation,
all downstream layers faithfully consumed it:

- `ModelBackedFrameObserver` copied the selected face bbox/landmarks into
  `FrameRecord`.
- eye, head-pose, gaze, scene, and viewer artifacts derived from that selected
  face.
- `render_processed_frame()` drew the persisted coordinates directly.
- QA validated counts/schema, not visual semantic correctness.

Visualization, scene conversion, frame ordering, and JSONL integrity were not
the cause.

## Evidence

Bad run:

```text
artifacts/output/nakamura_1/runs/20260626T160949Z-bbc6c27e
```

User-reported bad frames plus one additional same-signature frame found during
whole-run artifact analysis:

```text
f000001429, f000001430, f000001685,
f000001691, f000001692, f000001693, f000001695
```

The shared artifact signature was an overexpanded face box, impossible vertical
eye separation, and large negative roll while adjacent raw frames remained
visually continuous. The anomaly detector
`abs(left_pupil_y - right_pupil_y) >= 60`, `face_height >= 270`, and
`head_roll <= -0.75` found exactly those seven frames in the bad run.

Direct MediaPipe-region probing showed the failing boundary. Example:

```text
f000001429 full_frame: (297.4, 542.8, 539.7, 836.8)
f000001429 left_half:  (308.0, 613.4, 480.0, 810.1)

f000001685 full_frame: (335.7, 503.6, 632.6, 843.1)
f000001685 left_half:  (324.5, 616.7, 514.5, 830.2)

f000001692 full_frame: (322.6, 495.8, 704.0, 864.4)
f000001692 left_half:  (327.3, 633.0, 517.1, 851.0)
```

## Third-Party Contracts Checked

Verified on 2026-06-26:

- MediaPipe Face Landmarker Python docs:
  `IMAGE` mode uses per-image `detect()`, while `VIDEO` uses
  `detect_for_video()` and tracking/timestamp behavior. The fix therefore keeps
  `IMAGE` mode and does not hide the bug behind temporal tracking.
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
- Installed MediaPipe `face_landmarker.py` result shape exposes
  `face_landmarks`, `face_blendshapes`, and `facial_transformation_matrixes`,
  not per-candidate confidence; the repo correctly stores nullable candidate
  scores.
- Installed MediaPipe normalized landmark containers document normalized image
  coordinates; the repo's crop-to-full-frame conversion remains the correct
  coordinate boundary.
- OpenCV drawing docs confirm drawing functions mutate the provided image and
  draw supplied geometry directly; renderer fixes would be the wrong boundary.
  https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
- PyAV decode supplies emitted decoded frames; the repo's frame identity remains
  decoder-emission order plus optional PTS evidence.
  https://pyav.basswood-io.com/docs/stable/api/container.html

## Regression Coverage

Added:

- unit regression for a single overexpanded full-frame candidate plus a compact
  overlapping left-half candidate;
- unit guardrails proving overexpanded-triggered refinement rejects larger
  focused candidates, top-shift candidates without compact geometry gain, and
  large full-frame overlaps without compact gain;
- real-video regression for `artifacts/input/nakamura_1.mp4` frames `1429`,
  `1430`, `1685`, `1691`, `1692`, `1693`, and `1695`.

Existing face-observation regressions still cover previous mix clip repairs:
full-frame misses, broad false positives, small webcam panes, focused-region
scanning, seam clipping, and large full faces without consensus.

## Real Nakamura Verification

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

Fresh run:

```text
artifacts/output/nakamura_1/runs/20260626T193202Z-ea469828
viewer: artifacts/output/nakamura_1/runs/20260626T193202Z-ea469828/viewer/index.html
```

Fresh repaired frame records:

```text
1429 bb (308.0, 613.4, 480.0, 810.1) w,h (172.0, 196.7) eye_dy 10.1 roll 0.109
1430 bb (308.3, 612.7, 481.0, 811.4) w,h (172.7, 198.7) eye_dy 9.9 roll 0.110
1685 bb (324.5, 616.7, 514.5, 830.2) w,h (190.0, 213.5) eye_dy 11.5 roll -0.177
1691 bb (327.1, 629.8, 516.6, 848.9) w,h (189.5, 219.0) eye_dy 8.9 roll -0.161
1692 bb (327.3, 633.0, 517.1, 851.0) w,h (189.8, 218.0) eye_dy 8.6 roll -0.162
1693 bb (327.9, 634.5, 517.2, 854.2) w,h (189.3, 219.7) eye_dy 8.4 roll -0.153
1695 bb (329.4, 640.4, 518.1, 858.0) w,h (188.8, 217.7) eye_dy 7.8 roll -0.154
```

The same-signature detector returned no frames in the fresh run.

Artifact validation:

```text
qa_summary final_status: complete
decoded_frames: 1973
frame_records: 1973
raw_frames: 1973
processed_frames: 1973
scene_frame_records: 1973
schema_validation_passed: true
counts_match: true
```

Scene validation:

```text
scene_frame_records: 1973
valid_eye_midpoint_frames: 1973
valid_unigaze_ray_frames: 1973
valid_monitor_hit_frames: 1973
scene_frame_count_matches_decoded: true
scene_manifest_valid: true
scene_summary_valid: true
viewer_exists: true
```

Visual contact sheets inspected:

```text
/private/tmp/chess-gaze-nakamura-debug/final/final_1427_1432_processed.jpg
/private/tmp/chess-gaze-nakamura-debug/final/final_1682_1687_processed.jpg
/private/tmp/chess-gaze-nakamura-debug/final/final_1690_1697_processed.jpg
```

Visual inspection confirmed the reported frames now have compact face boxes and
landmarks continuous with adjacent frames.

## Verification

RED evidence:

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py::test_mediapipe_observer_prefers_compact_left_half_over_overexpanded_full_frame -q
FAILED because only the full-frame region was evaluated.
```

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_rejects_nakamura_overexpanded_faces -q
FAILED on f000001429 selected face width 242.3 is overexpanded.
```

Focused post-fix gates:

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py -q
28 passed
```

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py -q
2 passed, 2 skipped
```

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_qa_summary.py -q
44 passed, 2 skipped
```

Static gates:

```text
UV_CACHE_DIR=.uv-cache uv run ruff check .
All checks passed!
```

```text
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
57 files already formatted
```

```text
UV_CACHE_DIR=.uv-cache uv run mypy
Success: no issues found in 57 source files
```

Full pytest was run and failed only because local legacy mandatory inputs are
absent from this checkout:

```text
UV_CACHE_DIR=.uv-cache uv run pytest -q
7 failed, 253 passed, 7 skipped, 18 warnings in 449.49s
```

All 7 failures were missing-file assertions for:

```text
artifacts/input/test_1.mp4
artifacts/input/test_2.mp4
```

Broad available subset excluding those absent-media contract files:

```text
UV_CACHE_DIR=.uv-cache uv run pytest \
  --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py \
  --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py \
  --ignore=tests/chess_gaze/test_video_decode_real_video.py \
  --ignore=tests/chess_gaze/test_visualization_real_video.py -q
253 passed, 7 skipped, 18 warnings in 449.76s
```

## Source-Layout Review

`src/chess_gaze/face_observation.py` is above the 800-line review threshold.
The file still owns one coherent deep-module responsibility: MediaPipe face
observation and candidate arbitration, including region probing, crop-to-image
coordinate conversion, candidate scoring, and final selection. I did not split
it during this repair because the new logic modifies private arbitration
helpers that change together and do not yet have a stable reusable interface.

If candidate provenance becomes persisted, or if another observer needs the
same arbitration policy, split region selection and scoring behind an explicit
domain interface rather than adding more private helper growth to this file.

## Remaining Limitations

- Frame records still persist only the selected face, not all region candidates;
  direct candidate provenance required a diagnostic probe during this repair.
- The repair does not introduce temporal smoothing or MediaPipe `VIDEO` mode.
  That is intentional: the project contract requires independent per-frame
  evidence.
- MediaPipe continues to emit non-fatal Clearcut telemetry warnings and
  cv2/PyAV duplicate AVFoundation class warnings in this macOS environment.
