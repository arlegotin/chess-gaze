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
