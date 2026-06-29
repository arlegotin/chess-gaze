# Task 4 Report: Real-Video Verification And Closeout

## What I Implemented

- Added `test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region`
  in `tests/chess_gaze/test_face_observation_real_video.py`.
- Sampled the reported Carlsen windows plus adjacent controls and asserted:
  - every sampled visible-face frame recovers a face;
  - every selected center stays inside the visible person region
    (`450 <= x <= 660`, `240 <= y <= 430`), which rejects the historical
    plaque/background jump.
- Wrote the final closeout at
  `docs/superpowers/closeouts/2026-06-29-carlsen-face-arbitration-repair.md`.

## Commands Run And Exact Outcomes

1. Historical RED setup and run on `d6a6753`:

   - `git worktree add /private/tmp/chess-gaze-carlsen-red d6a6753`
   - copied the updated real-video test plus required Carlsen/model assets into
     the temporary worktree
   - `MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region -q`
   - outcome: `1 failed` with
     `AssertionError: f000002042 selected center x=774.2 outside visible player-face bounds`

2. Current-branch Carlsen regression:

   - `MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region -q`
   - outcome: `1 passed in 24.27s`

3. User-provided Nakamura short command:

   - `MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_recovers_nakamura_short_visible_faces tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_nakamura_short_faces_bounded -q`
   - outcome: `ERROR: not found` because the first selector omitted `_face_`
     from the actual test name

4. Corrected targeted real-video pytest command:

   - `MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_face_observer_recovers_nakamura_short_visible_faces tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_nakamura_short_faces_bounded tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region -q`
   - outcome: `3 passed in 24.46s`

5. Fresh end-to-end analyze:

   - `MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --output-root /private/tmp/chess-gaze-task4-output --models-root models`
   - outcome: exit `0`
   - run path:
     `/private/tmp/chess-gaze-task4-output/nakamura_short/runs/20260629T140320Z-eefca250`

## Real-Video Evidence

- Pre-fix `d6a6753` RED hit the historical wrong-selection signature:
  `f000002042` selected center `x=774.2`.
- Current-branch Carlsen probe over all sampled reported frames and adjacent
  controls kept centers inside the visible person region; representative centers
  include:
  - `f000002042`: `(578.3, 347.6)`
  - `f000005694`: `(574.2, 380.2)`
  - `f000009029`: `(564.0, 374.3)`
  - `f000015079`: `(492.0, 321.0)`
- Fresh `nakamura_short` analyze run completed with:
  - `qa_summary.final_status = complete`
  - `decoded_frames = 180`
  - `frame_records = 180`
  - `scene_frame_records = 180`
  - `face_present_rate = 1.0`
  - `scene_summary.valid_monitor_hit_frames = 180`

## Files Changed

- `tests/chess_gaze/test_face_observation_real_video.py`
- `docs/superpowers/closeouts/2026-06-29-carlsen-face-arbitration-repair.md`
- `.superpowers/sdd/task-4-report.md`

## Concerns / Limitations

- Real MediaPipe verification required host runtime access; the sandboxed probe
  crashed with `graph_service.h:139 Check failed: service_ Service is unavailable.`
- I did not rerun a full `carlsen_1.mp4` analyze job; I used the bounded direct
  probe path over all reported frames plus adjacent controls.
- The fresh `nakamura_short` analyze run kept `save_frame_images: false`, so
  its `raw_frames/` and `processed_frames/` directories are intentionally empty.
