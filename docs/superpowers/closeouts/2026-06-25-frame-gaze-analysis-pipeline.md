# Frame Gaze Analysis Pipeline Closeout

Date: 2026-06-25

## Request Summary

Execute the frame-gaze analysis implementation plan with Superpowers and
subagents, making meaningful commits along the way. The requested pipeline
decodes local videos, preserves raw evidence, records strict per-frame
face/eye/head/gaze observations, writes processed visualizations, and produces
QA artifacts without temporal smoothing.

## Spec And Plan Status

The active spec and plan were followed for the implemented, model-free pipeline
surface. Tasks 1-6 and 11-13 were implemented, committed, reviewed, and verified.
Task 7's pure face-candidate selection and lazy MediaPipe adapter surface were
committed and reviewed, but its real MediaPipe verification remains blocked.
Tasks 8-10 remain blocked and are not claimed complete because the required
local MediaPipe and UniGaze assets are absent, so real face/eye/head/gaze
observer verification cannot run.

The default CLI real-model path validates local model assets and currently stops
with `PIPELINE_NOT_IMPLEMENTED` after that gate. The testable observer-injection
pipeline path is covered by synthetic and mandatory real-video model-free
contracts.

## Task Summary

- Added strict CLI error handling and project gates.
- Added strict Pydantic frame, error, calibration, manifest, and QA schemas.
- Added atomic image writes and explicit RGB/OpenCV boundaries.
- Added config loading, `.env` parsing for setup-only values, and a committed
  model registry as the trust root.
- Added PyAV video inspection and deterministic frame decoding.
- Added calibration defaults with provenance and percentile policy.
- Added face-candidate selection and a lazy MediaPipe adapter surface; real
  MediaPipe verification is blocked by the missing `.task` asset.
- Added processed-frame visualization for face, eye, iris, gaze, head-pose,
  status, timestamp, and error overlays.
- Added `analyze_video()` orchestration with strict run layout, raw frames,
  processed frames, `frames.jsonl`, `errors.jsonl`, and model-free observer
  injection for tests.
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

Missing required real-model assets:

```text
/Volumes/git/legotin/chess-gaze/models/mediapipe/face_landmarker.task
/Volumes/git/legotin/chess-gaze/models/unigaze/unigaze_h14_joint.safetensors
```

## TDD And Review Evidence

Each task was implemented test-first where behavior changed. Representative
evidence:

```text
Task 13 RED:
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_qa_summary.py -q
ModuleNotFoundError: No module named 'chess_gaze.qa_summary'
```

```text
Task 13 post-review focused gate:
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_video_decode.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_pipeline_contract.py -q
18 passed in 0.47s
```

Subagent reviews were run after major task commits. Task 13 required two review
fixes: the QA summary now uses independent decoded-frame evidence from
`video_manifest.json`, and malformed JSONL including invalid UTF-8 leaves failed
QA summary evidence instead of escaping before persistence.

## Full Gate Outputs

```text
UV_CACHE_DIR=.uv-cache uv run pytest -q
92 passed, 1 skipped in 1707.40s (0:28:27)
```

The skip is the expected blocked real MediaPipe asset test:

```text
SKIPPED [1] tests/chess_gaze/test_face_observation_real_video.py:41: BLOCKED: missing mandatory MediaPipe Face Landmarker task asset: /Volumes/git/legotin/chess-gaze/models/mediapipe/face_landmarker.task
```

```text
UV_CACHE_DIR=.uv-cache uv run ruff check .
All checks passed!
```

```text
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
33 files already formatted
```

```text
UV_CACHE_DIR=.uv-cache uv run mypy
Success: no issues found in 33 source files
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

Model-free real-video contract tests passed before the final UTF-8 review fix
and were re-covered by the final full pytest gate:

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py -q
4 passed in 976.18s
```

Full real-model smoke is blocked because the two required model files listed
above are absent. The next unblock action is to place the approved local model
assets under `models/`, record checksums in `src/chess_gaze/model_registry.json`,
then implement and verify Tasks 8-10 before rerunning:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/test_1.mp4
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/test_2.mp4
```

No real-model manual QA is claimed.

## Manual QA Sample Notes

QA summaries now provide deterministic sample IDs, worst blur/exposure frame
IDs, and representative failure frame IDs for model-free real-video runs. Manual
inspection of model-backed overlays remains blocked until real observers and
local model assets are available.

## Remaining Limitations

- Tasks 8-10 are not complete.
- Default `chess-gaze analyze` cannot complete real-model analysis yet.
- Model registry production checksums are still unset because the local model
  files are absent.
- MediaPipe and UniGaze quality, fidelity, and real-video smoke claims remain
  unverified.
- Disk-space preflight is a free-space availability check; QA closeout records
  completed run size as the estimate field.
