# chess-gaze

Local Python pipeline for per-frame video evidence used by chess gaze analysis.

The implemented pipeline decodes video, writes strict run artifacts, preserves raw
and processed frame evidence, and revalidates artifacts into `qa_summary.json`.
The default real-model CLI path is still blocked until local model assets are
installed and the real face/eye/head/gaze observers are completed.

## Setup

Use `uv` from the repository root:

```sh
uv sync
```

Run local gates:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Format code when needed:

```sh
uv run ruff format .
```

## Analyze

Run analysis from the repository root:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4
```

Useful options:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4 --output-root artifacts/output
uv run chess-gaze analyze artifacts/input/test_1.mp4 --models-root models
uv run chess-gaze analyze artifacts/input/test_1.mp4 --config analysis.json
```

Runs are written under:

```text
artifacts/output/<video-stem>/runs/<run-id>/
```

Each completed model-free test run contains:

- `run_manifest.json`
- `calibration.json`
- `video_manifest.json`
- `raw_frames/`
- `processed_frames/`
- `records/frames.jsonl`
- `records/errors.jsonl`
- `qa_summary.json`

## Model Policy

Model binaries stay under ignored `models/`. The committed trust root is
`src/chess_gaze/model_registry.json`; local manifests cannot add or override
registry entries.

The optional setup-time `HF_TOKEN` may live in ignored `.env` for explicit model
prefetch work. Analysis does not download models, does not require network
access, and does not read `HF_TOKEN` for analysis-time access.

UniGaze `unigaze_h14_joint` uses the `MG-NC-RAI-2.0` license. The repo owner's
intended-use approval was granted on 2026-06-25 and is recorded as registry
metadata, not as a secret.

Real-model smoke requires these local files with registry-approved metadata:

```text
models/mediapipe/face_landmarker.task
models/unigaze/unigaze_h14_joint.safetensors
```

## Repository Shape

- `src/chess_gaze/` contains the importable Python package.
- `tests/` contains behavior and real-data contract tests.
- `artifacts/input/` contains local verification videos when present.
- `artifacts/output/` contains ignored analysis output.
- `models/` contains ignored local model binaries.
- `docs/development/` contains canonical development guidance.
- `docs/superpowers/` contains active specs, implementation plans, and closeouts
  produced by Superpowers workflows.
