# Frame Gaze Analysis Pipeline Closeout

Date: 2026-06-25

## Request Summary

Execute the frame-gaze analysis implementation plan with Superpowers and
subagents, making meaningful commits along the way. The requested pipeline
decodes local videos, preserves raw evidence, records strict per-frame
face/eye/head/gaze observations, writes processed visualizations, and produces
QA artifacts without temporal smoothing.

## Spec And Plan Status

The active spec and plan were followed for the implemented pipeline surface.
Tasks 1-13 are implemented and verified, including the previously blocked
real-model Tasks 7-10. The default `chess-gaze analyze` path now validates
local model assets, builds the real observer bundle, runs MediaPipe face
landmarks, derives eye/iris and head-pose evidence, runs the local UniGaze
checkpoint, writes processed overlays, and revalidates artifacts.

The full default CLI run over `artifacts/input/test_2.mp4` completed
successfully after the initial closeout, covering all 1,973 decoded frames. A
bounded real-model entrypoint smoke also used a one-frame lossless clip from
`test_1.mp4` and verified the full default observer path on a frame with face,
both eyes, head pose, UniGaze appearance gaze, recommended gaze, eye crops, and
QA summary. The full default CLI run over `test_1.mp4` was not run because it
contains 3,613 frames and the default pipeline intentionally processes every
frame.

## Task Summary

- Added strict CLI error handling and project gates.
- Added strict Pydantic frame, error, calibration, manifest, and QA schemas.
- Added atomic image writes and explicit RGB/OpenCV boundaries.
- Added config loading, `.env` parsing for setup-only values, and a committed
  model registry as the trust root.
- Added PyAV video inspection and deterministic frame decoding.
- Added calibration defaults with provenance and percentile policy.
- Added face-candidate selection and a lazy MediaPipe adapter surface; real
  MediaPipe verification now passes with the committed local asset checksum.
- Added eye and iris observation with independent left/right crop evidence.
- Added head-pose estimation preserving MediaPipe transform evidence and OpenCV
  PnP output.
- Added UniGaze local checkpoint loading with offline Hugging Face boundaries,
  model-level appearance gaze, geometric per-eye gaze, and recommended-gaze
  synthesis.
- Added processed-frame visualization for face, eye, iris, gaze, head-pose,
  status, timestamp, and error overlays.
- Added `analyze_video()` orchestration with strict run layout, raw frames,
  processed frames, `frames.jsonl`, `errors.jsonl`, model-free observer
  injection for tests, and the default model-backed observer bundle.
- Added QA summary revalidation from disk, including count checks, schema
  failure evidence, deterministic samples, representative failures, byte counts,
  status transitions, and final status.
- Updated README with usage, artifact layout, and model policy.

## Dependency And Model Status

Analysis does not download models and does not require `HF_TOKEN`. Optional
setup-time tokens may live in ignored `.env` for explicit prefetch work only.
Model binaries stay under ignored `models/`; `src/chess_gaze/model_registry.json`
is the committed trust root.

The registry records UniGaze `unigaze_h14_joint` with `MG-NC-RAI-2.0` intended
use approved by the repo owner on 2026-06-25. The MediaPipe Face Landmarker
entry records Google AI Edge Terms and `IMAGE` running mode.

OpenCV dependency resolution is guarded so only `opencv-python-headless` is
installed.

Required real-model assets are present locally under ignored `models/` and have
committed registry checksums:

```text
models/mediapipe/face_landmarker.task
  sha256: 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
models/unigaze/unigaze_h14_joint.safetensors
  sha256: a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f
```

MediaPipe real tests and default real analysis require unsandboxed execution in
the managed macOS agent environment. Inside the sandbox, MediaPipe native
initialization aborts at GL/Metal graph services; unsandboxed runs pass.

## TDD And Review Evidence

Each task was implemented test-first where behavior changed. Representative
evidence:

```text
Task 10 RED:
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py -q
ModuleNotFoundError: No module named 'chess_gaze.gaze_observation'
```

```text
Default real observer RED:
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py::test_config_models_root_controls_default_model_observer_factory -q
ModuleNotFoundError: No module named 'chess_gaze.frame_observation'
```

Task 8 and Task 9 were implemented by subagents and integrated after local
verification. Task 10 and the default real observer boundary were implemented
test-first in the main session.

## Full Gate Outputs

```text
UV_CACHE_DIR=.uv-cache uv run pytest -q
116 passed, 18 warnings in 1649.66s (0:27:29)
```

```text
UV_CACHE_DIR=.uv-cache uv run ruff check src tests
All checks passed!
```

```text
UV_CACHE_DIR=.uv-cache uv run ruff format --check src tests
44 files already formatted
```

```text
UV_CACHE_DIR=.uv-cache uv run mypy src tests
Success: no issues found in 44 source files
```

```text
UV_CACHE_DIR=.uv-cache uv lock --check
Resolved 86 packages in 0.42ms
```

```text
UV_CACHE_DIR=.uv-cache uv run python -c 'import importlib.metadata as m; providers=[d.metadata["Name"] for d in m.distributions() if d.metadata["Name"].lower().startswith("opencv-python")]; print(providers); raise SystemExit(0 if providers == ["opencv-python-headless"] else 1)'
['opencv-python-headless']
```

## Real-Video Smoke Status

The mandatory verification videos are present:

```text
/Volumes/git/legotin/chess-gaze/artifacts/input/test_1.mp4
/Volumes/git/legotin/chess-gaze/artifacts/input/test_2.mp4
```

Real-video observer gates now pass with real local assets:

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_eye_observation_real_video.py tests/chess_gaze/test_head_pose_real_video.py tests/chess_gaze/test_gaze_observation_real_video.py -q
4 passed, 18 warnings in 26.68s
```

The default real CLI path was smoke-tested through the actual entrypoint on a
one-frame lossless clip extracted from `artifacts/input/test_1.mp4` at the
sampled frame-300 timestamp:

```text
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze /private/tmp/chess_gaze_test1_frame300_lossless.mp4 --output-root /private/tmp/chess-gaze-smoke-output --models-root models
/private/tmp/chess-gaze-smoke-output/chess_gaze_test1_frame300_lossless/runs/20260625T165520Z-f4966cb9
```

The generated record had `status=OK`, face present, left and right eyes present,
head pose valid, geometric gaze valid, UniGaze appearance gaze valid,
recommended gaze valid, no errors, two eye crops, and QA summary `final_status`
`complete`.

The default real CLI path was then run end-to-end on the full smaller mandatory
verification video:

```text
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/test_2.mp4 --output-root artifacts/output --models-root models
artifacts/output/test_2/runs/20260625T173257Z-d9d9ce3f
```

Fresh artifact verification for that run:

```text
final_status complete
decoded_frames 1973
frame_records 1973
raw_frames 1973
processed_frames 1973
crop_files 2915
schema_validation_passed True
counts_match True
face_present_rate 0.7384693360364927
both_eyes_present_rate 0.7384693360364927
head_pose_valid_rate 0.6644703497212366
face_gaze_valid_rate 0.7384693360364927
recommended_gaze_valid_rate 0.3882412569690826
```

## Manual QA Sample Notes

QA summaries now provide deterministic sample IDs, worst blur/exposure frame
IDs, representative failure frame IDs, byte counts, and detection rates for both
model-free and model-backed runs. The bounded real-model smoke wrote a processed
overlay and both eye crops for manual inspection.

## Remaining Limitations

- Full default `chess-gaze analyze` over `artifacts/input/test_1.mp4` has not
  been run yet. The smaller mandatory video, `artifacts/input/test_2.mp4`,
  completed end to end with schema-valid artifacts.
- Board target mapping is still absent from the current schema, so recommended
  gaze target fields remain null.
- MediaPipe/OpenCV/PyAV runs emit non-fatal duplicate AVFoundation class
  warnings from packaged native libraries.
- Disk-space preflight is a free-space availability check; QA closeout records
  completed run size as the estimate field.
