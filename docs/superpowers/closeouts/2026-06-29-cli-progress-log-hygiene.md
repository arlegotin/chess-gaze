# CLI Progress And Native Log Hygiene Closeout

Date: 2026-06-29

## Summary

The noisy long-run terminal output was not evidence that the reported
`nepo_2.mp4` analysis artifacts were corrupt. It was a mix of dependency-owned
native stderr/stdout messages plus an actual UX gap: `chess-gaze analyze` did
not expose committed-frame progress or ETA during long videos.

This closeout verifies the repair committed in:

- `eb7fcb5` `docs: plan cli progress log hygiene`
- `1298808` `feat: report committed analysis progress`
- `9571b44` `feat: show analyze progress on stderr`
- `2c3facd` `fix: filter known native analysis log noise`
- `1269e79` `fix: satisfy progress log hygiene gates`

## Root Cause

The user-visible terminal problem had four separate causes:

- `chess_gaze.cli` eagerly imported the analysis pipeline, so commands and
  preflight failures could load PyAV, OpenCV, MediaPipe, and UniGaze before
  analysis actually started.
- PyAV and OpenCV each ship their own FFmpeg/AVFoundation-linked native
  libraries. Loading both in one macOS process emits Objective-C duplicate class
  warnings for `AVFFrameReceiver` and `AVFAudioReceiver`. The warning is real
  packaging risk, but it is not itself proof of bad output.
- MediaPipe/TFLite emit startup and Clearcut telemetry uploader messages
  directly to native stderr. A direct local probe showed ordinary Python/Abseil
  log-threshold environment variables did not suppress this installed wheel's
  startup output.
- The pipeline only reported completion after the whole run, leaving long
  videos such as `nepo_2.mp4` with no reliable ETA despite having a known total
  decoded-frame count.

## Durable Surface Changed

The durable runtime surfaces changed were:

- `src/chess_gaze/pipeline.py`: added committed-frame progress events after the
  analysis state is durably advanced.
- `src/chess_gaze/cli.py`: added lazy analysis imports, `--progress
  {auto,on,off}`, and a tqdm progress renderer on stderr.
- `src/chess_gaze/native_log_filter.py`: added a narrow fd-2 native stderr
  filter for known dependency chatter during `analyze`.
- `src/chess_gaze/gaze_observation.py`: suppressed the UniGaze package's local
  weight-load print without suppressing repository-owned errors.
- `pyproject.toml` and `uv.lock`: recorded `tqdm` as a direct runtime
  dependency.

## Regression Coverage

Added or extended tests cover:

- pipeline progress events on fresh and resumed runs;
- CLI `--progress auto/on/off` wiring;
- analyze lazy imports for missing-input/preflight paths;
- native stderr filtering for known AVFoundation, MediaPipe/TFLite, and
  Clearcut messages while preserving unknown stderr;
- UniGaze local weight loading not writing to stdout;
- CLI progress using the original stderr stream while native filtering is
  active.

## Real-Video Evidence

### Baseline reproduction

Before the fix, this command completed on `artifacts/input/nakamura_short.mp4`
but printed the same classes of native startup noise and no progress bar:

```sh
UV_CACHE_DIR=.uv-cache env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 \
  --output-root /private/tmp/chess-gaze-noise-before \
  --models-root models \
  --unigaze-device mps \
  --unigaze-batch-size 7 \
  --save-frames \
  --no-resume
```

Run validation:

```text
qa_summary.final_status: complete
decoded_frames: 180
frame_records: 180
scene_frame_records: 180
scene_summary.valid_monitor_hit_frames: 180
errors_by_code: {"GAZE_ESTIMATORS_DISAGREE": 53}
```

Frames `0`, `90`, and `179` were visually inspected. The overlays were
coherent; frame `179` carried an expected model-disagreement warning rather than
an artifact failure.

### Fixed smoke run

After the fix, the required real verification command was:

```sh
UV_CACHE_DIR=.uv-cache env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 \
  --output-root /private/tmp/chess-gaze-progress-after-20260629T1622 \
  --models-root models \
  --unigaze-device mps \
  --unigaze-batch-size 7 \
  --save-frames \
  --no-resume \
  --progress on
```

Observed result:

```text
/private/tmp/chess-gaze-progress-after-20260629T1622/nakamura_short/runs/20260629T162442Z-0fdbecc1
viewer: /private/tmp/chess-gaze-progress-after-20260629T1622/nakamura_short/runs/20260629T162442Z-0fdbecc1/viewer/index.html
```

Stdout contained only the run directory and viewer path. Stderr contained a
tqdm progress bar reaching `180/180`; it did not contain the known noisy tokens:

```text
AVFFrameReceiver: false
AVFAudioReceiver: false
portable_clearcut_uploader: false
Failed to send to clearcut: false
init-domain.cc: false
face_landmarker_graph.cc: false
gl_context.cc: false
TensorFlow Lite XNNPACK: false
inference_feedback_manager.cc: false
```

Run validation:

```text
qa_summary.final_status: complete
decoded_frames: 180
frame_records: 180
processed_frames: 180
raw_frames: 180
scene_frame_records: 180
crop_files: 360
errors_by_code: {"GAZE_ESTIMATORS_DISAGREE": 53}
status_transitions: ["created", "processing", "revalidating", "complete"]
scene_summary.valid_eye_midpoint_frames: 180
scene_summary.valid_unigaze_ray_frames: 180
scene_summary.valid_monitor_hit_frames: 180
```

Frames `0`, `90`, and `179` were visually inspected from
`processed_frames/`. The overlays remained coherent after the progress/log
hygiene changes.

## Third-Party Docs Checked

Verified on 2026-06-29:

- tqdm docs, https://tqdm.github.io/docs/tqdm/: selected for `total`,
  `initial`, `unit`, `file`, terminal refresh, and ETA behavior.
- Rich progress docs, https://rich.readthedocs.io/en/latest/progress.html:
  rejected as unnecessary extra surface for a single scalar progress bar.
- MediaPipe Face Landmarker Python docs,
  https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python:
  confirmed the task remains in `IMAGE` mode and does not change model
  semantics.
- MediaPipe BaseOptions docs,
  https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions:
  checked the CPU/GPU delegate API while investigating the sandbox-only native
  abort. Explicit `Delegate.CPU` did not prevent this wheel's face-landmarker
  graph from initializing an internal macOS Metal helper.
- PyAV container docs, https://pyav.org/docs/stable/api/container.html:
  rechecked decode/container ownership boundaries; progress is intentionally
  tied to committed frame records, not raw decode iteration.
- OpenCV Python headless package,
  https://pypi.org/project/opencv-python-headless/: confirmed the repo uses one
  OpenCV Python provider, but OpenCV and PyAV still bundle separate FFmpeg
  native libraries.

## Gate Evidence

Focused and static gates:

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py -q
35 passed

UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py -q
17 passed

UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_native_log_filter.py tests/chess_gaze/test_gaze_observation.py -q
21 passed

UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_native_log_filter.py tests/chess_gaze/test_gaze_observation.py -q
73 passed

UV_CACHE_DIR=.uv-cache uv run ruff check .
All checks passed.

UV_CACHE_DIR=.uv-cache uv run ruff format --check .
69 files already formatted.

UV_CACHE_DIR=.uv-cache uv run mypy
Success: no issues found
```

Full suite:

```text
UV_CACHE_DIR=.uv-cache uv run pytest -q --disable-warnings
395 passed, 18 warnings in 53.30s
```

The full suite must run with native runtime access on macOS. The same command
inside the managed filesystem sandbox aborted inside MediaPipe native code while
creating `FaceLandmarker` with:

```text
gl_context_nsgl.cc:80 failed to create pixel format; trying without acceleration
graph_service.h:139 Check failed: service_ Service is unavailable
-[DrishtiMetalHelper initWithCalculatorContext:]
```

The focused real-video MediaPipe test passed outside the sandbox:

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_eye_observation_real_video.py -q --disable-warnings
1 passed in 3.28s
```

This is a native environment constraint, not a Python assertion failure. The
current `chess-gaze analyze` terminal path was verified outside the sandbox
using the required `nakamura_short.mp4` smoke run above.

## Residual Risk

The underlying PyAV/OpenCV duplicate FFmpeg/AVFoundation packaging conflict is
not eliminated. The implemented filter prevents known dependency chatter from
misleading users during `analyze`, but it does not remove the native collision.
If future failures occur inside AVFoundation, FFmpeg, or MediaPipe native
symbols, the durable architecture options are dependency isolation in a worker
process or eliminating one bundled FFmpeg provider from the runtime.

MediaPipe Face Landmarker also depends on macOS graphics services even when its
documented `BaseOptions.Delegate.CPU` is selected. Headless or sandboxed
environments can abort the process before Python can catch an exception. This
task documents that gate constraint and verifies the real user-facing terminal
path; it does not redesign MediaPipe into an isolated worker process.
