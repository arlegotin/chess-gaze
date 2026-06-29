# Task 3 Report: Lock Recoverable Miss Regression

## What I implemented

- Added `test_mediapipe_observer_recovers_visible_face_from_left_upper_inner_region` to lock the regression where full frame, left half, left top, and left upper band all miss but a narrower upper-left crop finds the visible face.
- Added a deterministic `left_upper_inner` detection region in `src/chess_gaze/face_observation.py`.
- Defined the new region with frame fractions, not video-specific coordinates:
  - right edge: `3/8` of frame width
  - bottom edge: `4/9` of frame height
- Inserted the new region after `left_upper_band` and before the existing right-side upper regions, preserving full-frame coordinate remapping.
- Updated focused-region fake MediaPipe sequences and exact `detect_shapes` assertions to reflect the ninth detection call:
  - `1920x1080 -> (480, 720, 3)`
  - `1280x720 -> (320, 480, 3)`
  - `200x100 -> (44, 75, 3)`

## TDD RED/GREEN evidence

1. Added the regression test first in `tests/chess_gaze/test_face_observation.py`.

2. RED:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py::test_mediapipe_observer_recovers_visible_face_from_left_upper_inner_region -q
```

Output snippet:

```text
F
At index 6 diff: (486, 960, 3) != (480, 720, 3)
Right contains one more item: (525, 960, 3)
```

This showed the observer was still calling the old right-upper-band crop in that slot and had no left upper-inner region.

3. GREEN:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py::test_mediapipe_observer_recovers_visible_face_from_left_upper_inner_region -q
```

Output snippet:

```text
.
1 passed in 0.13s
```

## Focused suite result

Command:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_region_arbitration.py -q
```

Output snippet:

```text
...............................                                          [100%]
31 passed in 0.24s
```

## Files changed

- `src/chess_gaze/face_observation.py`
- `tests/chess_gaze/test_face_observation.py`
- `tests/chess_gaze/test_face_observation_region_arbitration.py`
- `.superpowers/sdd/task-3-report.md`

## Self-review notes

- Kept the change deterministic and per-frame only. No smoothing, tracking, interpolation, or temporal logic was introduced.
- Preserved the existing crop-to-full-frame coordinate conversion path by adding only a new `_DetectionRegion`.
- Kept existing regions and current consensus fallback behavior intact.
- Trimmed fake MediaPipe result sequences back to exact nine-call fixtures after introducing the new region.

## Concerns

- None.

## Task 3B: Lock Inner-Region Consensus Regression

### What I implemented

- Added `test_mediapipe_observer_prefers_inner_consensus_over_broad_upper_false_positive` to lock the case where `left_top` and `left_upper_band` agree on a plaque/background false positive while the true face is only visible in the upper-left player-camera pane.
- Added a second deterministic upper-left inner crop, `left_upper_inner_nearby`, in `src/chess_gaze/face_observation.py`.
- Defined the paired crop with frame fractions rather than video-specific coordinates:
  - right edge: `3/8` of frame width
  - bottom edge: `5/12` of frame height
- Kept all existing regions, preserved full-frame remapping, and left the existing fallback consensus precedence logic unchanged.
- Updated fake MediaPipe sequences and exact `detect_shapes` expectations to reflect the new tenth detection call:
  - `1920x1080 -> (450, 720, 3)`
  - `1280x720 -> (300, 480, 3)`
  - `200x100 -> (42, 75, 3)`

### TDD RED/GREEN evidence

1. Added the regression test first in
   `tests/chess_gaze/test_face_observation_region_arbitration.py`.

2. RED:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_inner_consensus_over_broad_upper_false_positive -q
```

Output snippet:

```text
E       AssertionError: assert 'broad_upper_false_positive' == 'paired_inner_consensus_real_face'
```

This confirmed the pre-fix observer still preferred the broad upper-left
consensus cluster over the true inner face.

3. GREEN:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_inner_consensus_over_broad_upper_false_positive -q
```

Output snippet:

```text
.
1 passed in 0.15s
```

### Focused suite result

Command:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_region_arbitration.py -q
```

Output snippet:

```text
................................                                         [100%]
32 passed in 0.20s
```

### Bounded Carlsen probe

Command run with real MediaPipe:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run python -c 'from pathlib import Path
from chess_gaze.calibration import default_calibration
from chess_gaze.face_observation import MediaPipeFaceObserver
from chess_gaze.video_decode import iter_decoded_frames
repo = Path(".").resolve()
video = repo / "artifacts/input/carlsen_1.mp4"
model = repo / "models/mediapipe/face_landmarker.task"
wanted = {2042, 2045, 2048, 2052}
observer = MediaPipeFaceObserver(model_asset_path=model, calibration=default_calibration())
try:
    for frame in iter_decoded_frames(video):
        if frame.frame_index not in wanted:
            continue
        observation = observer.observe(frame.rgb, frame_id=frame.frame_id)
        if observation.selection.present and observation.selection.primary_candidate_id is not None:
            candidate = next(item for item in observation.selection.candidates if item.candidate_id == observation.selection.primary_candidate_id)
            bbox = candidate.bounding_box_image_px
            print(frame.frame_id, "present", round(bbox.x_min, 1), round(bbox.y_min, 1), round(bbox.x_max, 1), round(bbox.y_max, 1), "center", round((bbox.x_min + bbox.x_max) / 2, 1), round((bbox.y_min + bbox.y_max) / 2, 1))
        else:
            print(frame.frame_id, "absent")
        wanted.remove(frame.frame_index)
        if not wanted:
            break
finally:
    observer.close()'
```

Observed selections:

- `f000002042`: `(505.2, 276.4, 651.4, 418.8)`, center `(578.3, 347.6)`
- `f000002045`: `(507.1, 279.3, 650.7, 422.2)`, center `(578.9, 350.8)`
- `f000002048`: `(503.4, 275.6, 652.4, 420.3)`, center `(577.9, 347.9)`
- `f000002052`: `(500.7, 276.9, 641.2, 415.9)`, center `(570.9, 346.4)`

Outcome: all four bounded Carlsen frames selected the visible person face in the
upper-left player-camera pane.

### Files changed

- `src/chess_gaze/face_observation.py`
- `tests/chess_gaze/test_face_observation.py`
- `tests/chess_gaze/test_face_observation_region_arbitration.py`
- `.superpowers/sdd/task-3-report.md`
- `docs/superpowers/plans/2026-06-29-carlsen-face-arbitration-repair.md`

### Concerns

- None.
