# UniGaze-Only Processed-Frame Arrows Design

Date: 2026-07-04

## Status

Approved by direct user request on 2026-07-04. The user explicitly requested
Superpowers, subagents, thorough planning, implementation, testing,
verification, and meaningful commits on the current branch.

This spec supersedes the processed-frame gaze-vector portions of
`docs/superpowers/specs/2026-06-24-frame-gaze-analysis-pipeline-design.md`.
Historical documents may still describe earlier geometric or recommended gaze
behavior, but this file defines the current processed-frame arrow contract and
default observer semantics.

## Goal

Make UniGaze the only gaze vector used for default frame observation status and
processed-frame rendering, removing the old pupil-derived geometric arrows,
their calculation path, and the recommended-gaze disagreement status logic.

## Required Behavior

- Default model-backed frame observation must no longer calculate per-eye
  geometric gaze from pupil/iris offsets.
- Default model-backed frame observation must no longer compare UniGaze against
  pupil-derived geometric gaze.
- `GAZE_ESTIMATORS_DISAGREE` must no longer be emitted by the default observer
  merely because UniGaze differs from pupil geometry.
- `recommended_gaze` remains in `FrameRecord` for schema compatibility, but in
  default model-backed records it must mirror the UniGaze `appearance_gaze`
  validity and angles.
- `geometric_gaze` remains in `FrameRecord` for schema compatibility, but in
  default model-backed records it must be an invalid legacy field instead of a
  calculated pupil-derived vector.
- Frame status must depend on face, eye, head-pose, UniGaze validity, and
  non-gaze warnings/errors. Invalid UniGaze remains an error.
- Multiple face candidates remain a warning when all required observation
  surfaces are otherwise valid.
- Processed frames must render no geometric pupil arrows.
- Processed frames must render no `recommended_gaze` arrow.
- Processed frames must render exactly the UniGaze/`appearance_gaze` arrow when
  valid and a face center exists.
- The UniGaze arrow must have no text label.
- UniGaze processed-frame visibility must increase relative to the old overlay:
  a thicker high-contrast arrow, a darker outline, and a larger arrow length.
- Head-pose axes must remain, and their visibility must increase so they are
  easier to distinguish from face/eye overlays.
- Existing eye boxes, iris landmarks, pupil-center dots, face box, landmarks,
  and genuine frame status/error text remain.

## Implementation Surface

Expected source changes:

- `src/chess_gaze/gaze_observation.py`
- `src/chess_gaze/frame_observation.py`
- `src/chess_gaze/visualization.py`
- `tests/chess_gaze/test_gaze_observation.py`
- `tests/chess_gaze/test_frame_observation.py`
- `tests/chess_gaze/test_visualization.py`
- `tests/chess_gaze/test_visualization_real_video.py`

No model, checkpoint, inference dependency, CLI flag, or scene/viewer schema
change is part of this task.

## Data Compatibility

`FrameRecord` continues to require `geometric_gaze`, `appearance_gaze`, and
`recommended_gaze` so existing artifacts and scene/QA readers still validate.
The default observer changes the values it writes:

- `appearance_gaze`: UniGaze output, unchanged.
- `recommended_gaze`: schema-compatible alias of `appearance_gaze`.
- `geometric_gaze`: invalid placeholder with no yaw/pitch because the
  pupil-derived vector is no longer calculated.

Historical external observers may still populate old fields. Readers should
continue validating those artifacts unless a future schema migration removes the
fields.

## Testing

Use test-first development.

Focused tests:

- `gaze_observation` no longer exposes the pupil-derived geometric calculator or
  recommended-gaze synthesizer.
- Model-backed observation maps UniGaze to both `appearance_gaze` and
  `recommended_gaze`, leaves `geometric_gaze` invalid, and does not emit gaze
  disagreement warnings for large UniGaze angles.
- Invalid UniGaze still yields an error status and `GAZE_MODEL_FAILED`.
- Multiple face candidate warnings still produce warning status when UniGaze is
  valid.
- Processed-frame rendering changes the face-center UniGaze target region but
  does not change pupil-arrow target regions.
- Generated processed-frame output no longer includes the UniGaze label text.
- Head-pose axes and UniGaze arrow visibility are covered by pixel-region
  assertions.

Focused command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q
```

Required gates:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q
UV_CACHE_DIR=.uv-cache uv run pytest -q --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Real-data smoke when local artifacts are available:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_visualization_real_video.py -q
```

## Acceptance Criteria

1. The default observer no longer imports or calls pupil-derived geometric gaze
   calculation.
2. The default observer no longer emits default gaze-disagreement warnings.
3. `recommended_gaze` mirrors UniGaze for default observer records.
4. `geometric_gaze` is schema-compatible but not a calculated pupil vector.
5. Processed frames show only the UniGaze gaze arrow.
6. The UniGaze arrow has no label and is visibly stronger than before.
7. Head-pose axes remain and are easier to see.
8. Existing artifact schemas keep validating.
9. Focused tests, broad local gates, and closeout evidence document the verified
   subset and any blocked checks.
