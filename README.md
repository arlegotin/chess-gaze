# chess-gaze

Local Python pipeline for per-frame video evidence and 3D scene artifacts used
by chess gaze analysis.

The implemented pipeline decodes video, writes strict run artifacts, keeps raw
and processed frame images only when explicitly requested, and revalidates
artifacts into `qa_summary.json`.
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
uv run chess-gaze analyze artifacts/input/test_1.mp4 --save-frames
```

By default, UniGaze runs on Apple Silicon MPS with batch size 7:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 \
  --output-root artifacts/output \
  --models-root models
```

For accepted MPS runs, leave `PYTORCH_ENABLE_MPS_FALLBACK`,
`PYTORCH_MPS_FAST_MATH`, and `PYTORCH_MPS_PREFER_METAL` unset. The MPS path
preflights the verified local UniGaze checkpoint on batch shape
`(7, 3, 224, 224)` before creating a run directory, then records runtime
metadata in
`run_manifest.json`.

For CPU compatibility or non-MPS machines, opt in explicitly:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4 \
  --unigaze-device cpu \
  --unigaze-batch-size 1
```

The selected batch size comes from the full Nakamura benchmark report at
`artifacts/output/benchmarks/2026-06-26-unigaze-mps-batching.json`: MPS batch 7
was the fastest passing MPS batch above 1 on this Apple M3 Max run. The
benchmark keeps frame processing independent and compares MPS outputs to the
CPU batch-1 flow with the approved tolerances.

Runs are written under:

```text
artifacts/output/<video-stem>/runs/<run-id>/
```

By default, rerunning the same `chess-gaze analyze <video>` command resumes the
newest compatible interrupted run for that video. Compatibility is checked
against the input video path and hash, video manifest, calibration, and inference
runtime metadata. Completed runs are never resumed; a completed rerun creates a
new run directory.

Use `--no-resume` to force a fresh run even when a compatible partial run
exists:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4 --no-resume
```

By default, completed runs do not retain decoded raw PNGs or processed overlay
JPEGs. The analyzer keeps decoded frames in memory only as long as needed for
observation and visualization data generation. Use `--save-frames` when a run
must retain `raw_frames/*.png` and `processed_frames/*.jpg` for visual debugging
or external QA.

Each completed run contains:

- `run_manifest.json`
- `calibration.json`
- `video_manifest.json`
- `analysis_state.json`
- `raw_frames/` (empty by default; populated by `--save-frames`)
- `processed_frames/` (empty by default; populated by `--save-frames`)
- `records/frames.jsonl`
- `records/errors.jsonl`
- `records/scene_frames.jsonl`
- `scene/scene_manifest.json`
- `scene/scene_summary.json`
- `viewer/index.html`
- `viewer/served.html`
- `viewer/scene-data.json`
- `qa_summary.json`

`chess-gaze analyze` prints the run directory and then the generated viewer
entry point:

```text
artifacts/output/<video-stem>/runs/<run-id>
viewer: artifacts/output/<video-stem>/runs/<run-id>/viewer/index.html
```

Open `viewer/index.html` directly in a browser. For large runs, prefer the
localhost-only static server:

```sh
uv run chess-gaze view artifacts/output/<video-stem>/runs/<run-id>
```

The command prints a local URL, serves `viewer/served.html` at `/`, and keeps
`viewer/index.html` available as the direct file-url artifact. It serves only
files under that run's `viewer/` directory and binds to loopback hosts only.

The viewer keeps run artifacts local, but it loads Three.js `0.185.0` from
pinned jsDelivr module URLs when the page renders. Expected remote module
requests are `three.module.js`, its transitive `three.core.js`, and
`OrbitControls.js` for the same pinned version. Project viewer code does not
upload scene JSON, frames, crops, or model data, but the remote modules execute
in the same page as embedded scene data and must be trusted. Offline viewing
requires those pinned modules to already be present in the browser cache.

The viewer also includes a `Hit Area` layer. It keeps the hit point as the point
estimate and overlays translucent angular-error patches on the monitor plane.
In `Accumulated` mode, hit-area patches accumulate like hit points but remain
controlled by the separate `Hit Area` toggle. The default typical angular error
is 8 degrees and can be adjusted from 0 to 12 degrees in the viewer. Hit-area
opacity defaults to 24% and is adjustable in the same control group. This is a
display assumption, not per-frame UniGaze confidence.

## Scene Artifacts

Scene units are pseudo-metric. Eye depth is inferred from the adult-male
interpupillary-distance assumption unless future calibration supplies measured
scale, so absolute distances are useful for reconstruction/debugging but should
not be treated as measured room geometry.

Scene coordinates are human-centered for the frontal desktop webcam assumption:
`+X` is the streamer's anatomical right, `+Y` is up, and `+Z` is the
streamer's back. OpenCV camera `+X` is still image-right, so when a face is
toward the camera, image-right gaze is the streamer's left and maps to negative
scene X. Monitor-directed gaze points toward negative scene Z.

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

For sandboxed test runs, exclude the tests that need native MediaPipe or
loopback socket permission:

```sh
uv run pytest -m "not native_mediapipe and not local_socket"
```

Before landing model-runtime or viewer-server changes, run the native and socket
gates unsandboxed:

```sh
uv run pytest -m native_mediapipe
uv run pytest -m local_socket
uv run pytest
```

If MediaPipe aborts with `gl_context_nsgl.cc`, `graph_service.h:139`, or
`DrishtiMetalHelper`, treat that as missing native runtime access. Do not hide
that stderr; it is diagnostic evidence.

## Repository Shape

- `src/chess_gaze/` contains the importable Python package.
- `tests/` contains behavior and real-data contract tests.
- `artifacts/input/` contains local verification videos when present.
- `artifacts/output/` contains ignored analysis output.
- `models/` contains ignored local model binaries.
- `docs/development/` contains canonical development guidance.
- `docs/superpowers/` contains active specs, implementation plans, and closeouts
  produced by Superpowers workflows.
