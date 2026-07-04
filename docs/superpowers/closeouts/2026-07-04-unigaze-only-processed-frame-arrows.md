# UniGaze-Only Processed-Frame Arrows Closeout

Date: 2026-07-04

## Summary

Processed-frame gaze rendering now uses UniGaze as the only gaze vector. The
default observer no longer calculates pupil-derived geometric gaze, no longer
synthesizes a disagreement-prone recommended gaze, and no longer emits default
`GAZE_ESTIMATORS_DISAGREE` warning status from pupil/UniGaze comparison.

`FrameRecord` schema compatibility is preserved:

- `appearance_gaze` remains the UniGaze output.
- `recommended_gaze` mirrors `appearance_gaze` in default observer records.
- `geometric_gaze` remains present but is an invalid legacy field with
  `GAZE_MODEL_FAILED`.

Processed JPEG overlays now draw only a face-centered, unlabeled, thicker
outlined UniGaze arrow. Existing face boxes, face landmarks, eye boxes, iris
landmarks, pupil-center markers, eye labels, head-pose axes, and status/error
text remain. Head-pose axes now draw with a dark outline for stronger contrast.

## Root Cause / Durable Surface

The old default observer still treated pupil-derived iris offsets as a peer gaze
estimator and used disagreement with UniGaze to drive `recommended_gaze` and
warning status. That no longer matched the useful runtime surface: UniGaze is the
only gaze vector the project wants to reason about and render.

The durable repair is at two boundaries:

- `src/chess_gaze/frame_observation.py` owns default per-frame observer
  semantics and now maps UniGaze directly into schema-compatible gaze fields.
- `src/chess_gaze/visualization.py` owns processed-frame rendering and now has a
  single gaze-vector drawing path for UniGaze.

During verification, `tests/chess_gaze/test_gaze_observation_real_video.py`
failed collection because it still imported the removed pupil/recommendation
helpers. That test was updated to validate UniGaze-only real-video evidence
without reintroducing geometric synthesis.

## Review

Subagent reviews were run after the test task, implementation task, review-fix
passes, and final branch diff. Important findings were fixed:

- real-video fixture still used `GAZE_ESTIMATORS_DISAGREE`;
- missing-face records kept old gaze-field semantics;
- visualization fixes had removed unrelated non-gaze overlays;
- the outlined UniGaze arrow was shifted off the face-center anchor;
- the closeout was missing before final documentation.

The final code review found no runtime/code issues after those repairs.

## Verification

RED evidence before implementation:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q
```

Result before production changes: `8 failed, 24 passed, 18 warnings`. Failures
covered removed helper exports, old geometric/recommended observer behavior, and
extra processed-frame gaze overlays/label.

Focused final gate:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_visualization.py -q
```

Result: `33 passed, 18 warnings in 1.62s`.

Real-video visualization smoke:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_visualization_real_video.py -q
```

Result: `1 passed in 0.35s`.

Real-video gaze collection check:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation_real_video.py --collect-only -q
```

Result: `2 tests collected in 0.84s`.

Available broad local suite:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q -m "not native_mediapipe and not local_socket" --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py
```

Result: `414 passed, 12 deselected, 18 warnings in 4.33s`.

Static gates:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check .
```

Result: `All checks passed!`.

```sh
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
```

Result: `71 files already formatted`.

```sh
UV_CACHE_DIR=.uv-cache uv run mypy
```

Result: `Success: no issues found in 71 source files`.

## Residual Risk

The full unfiltered pytest suite was not run in this managed sandbox. Native
MediaPipe tests are marked `native_mediapipe` and are documented as requiring
unsandboxed macOS GL/Metal access; local viewer server tests are marked
`local_socket` and require loopback socket permission. Heavy real-video contract
tests were excluded by the active plan from the broad local gate.

The repeated warnings are existing Torch deprecation warnings from
`torch.jit.script`; they do not come from this change.
