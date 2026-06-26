# chess-gaze

Local Python pipeline for per-frame video evidence and 3D scene artifacts used
by chess gaze analysis.

The implemented pipeline decodes video, writes strict run artifacts, preserves raw
and processed frame evidence, and revalidates artifacts into `qa_summary.json`.
The default CLI path validates local model checksums, runs MediaPipe face
landmarks, derives eye/iris and head-pose evidence, runs the local UniGaze
checkpoint, records strict per-frame gaze outputs, builds pseudo-metric 3D scene
artifacts, and generates a local browser viewer.

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

Each completed run contains:

- `run_manifest.json`
- `calibration.json`
- `video_manifest.json`
- `raw_frames/`
- `processed_frames/`
- `records/frames.jsonl`
- `records/errors.jsonl`
- `records/scene_frames.jsonl`
- `scene/scene_manifest.json`
- `scene/scene_summary.json`
- `viewer/index.html`
- `viewer/scene-data.json`
- `qa_summary.json`

`chess-gaze analyze` prints the run directory and then the generated viewer
entry point:

```text
artifacts/output/<video-stem>/runs/<run-id>
viewer: artifacts/output/<video-stem>/runs/<run-id>/viewer/index.html
```

Open the viewer through the localhost-only static server:

```sh
uv run chess-gaze view artifacts/output/<video-stem>/runs/<run-id>
```

The command prints a local URL and serves only files under that run's
`viewer/` directory. It binds to loopback hosts only.

## Scene Artifacts

Scene units are pseudo-metric. Eye depth is inferred from the adult-male
interpupillary-distance assumption unless future calibration supplies measured
scale, so absolute distances are useful for reconstruction/debugging but should
not be treated as measured room geometry.

Persisted scene assumptions include:

- adult-male interpupillary distance: `0.063 m`
- main-monitor distance from the robust eye-midpoint scene center: `0.700 m`
- physical monitor size: `0.600 m x 0.340 m`
- extended monitor plane scale: `3.0`
- head ellipsoid radii: `0.090 m, 0.120 m, 0.100 m`
- eye sphere radius: `0.012 m`

Every decoded frame produces one `records/scene_frames.jsonl` record. Every
valid forward ray-plane intersection produces one persisted gaze hit point.
Hit points are not merged, sampled, smoothed, clamped to the physical monitor,
clustered, or replaced by a heatmap.

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
  sha256: 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
models/unigaze/unigaze_h14_joint.safetensors
  sha256: a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f
```

In the managed agent sandbox, MediaPipe's native macOS GL/Metal initialization
fails. Run real-model gates and `chess-gaze analyze` unsandboxed with local
assets.

## Repository Shape

- `src/chess_gaze/` contains the importable Python package.
- `tests/` contains behavior and real-data contract tests.
- `artifacts/input/` contains local verification videos when present.
- `artifacts/output/` contains ignored analysis output.
- `models/` contains ignored local model binaries.
- `docs/development/` contains canonical development guidance.
- `docs/superpowers/` contains active specs, implementation plans, and closeouts
  produced by Superpowers workflows.
