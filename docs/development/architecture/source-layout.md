# Source Layout

This repository uses a small `src/` layout so tests exercise the installed
package instead of accidentally importing Python files from the repository root.

## Current Map

- `pyproject.toml` is the canonical project and tool configuration.
- `uv.lock` records the resolved local development environment.
- `src/chess_gaze/` is the importable package. Active modules are named after
  the behavior they own:
  - `artifact_runs.py` owns run directory layout and artifact-relative paths.
  - `calibration.py` and `configuration.py` own analysis configuration records.
  - `video_decode.py`, `image_io.py`, and `visualization.py` own frame IO and
    processed-frame rendering.
  - `model_assets.py` and `model_registry.json` own local model trust and
    checksum validation.
  - `face_landmark_indices.py` owns named MediaPipe face landmark indices that
    encode anatomical left/right semantics.
  - `face_observation.py`, `eye_observation.py`, `head_pose.py`,
    `gaze_observation.py`, and `frame_observation.py` own per-frame evidence
    extraction.
  - `unigaze_runtime.py` owns UniGaze device/batch runtime validation, MPS
    preflight, synchronization, and inference metadata assembly.
  - `frame_records.py`, `errors.py`, and `geometry.py` own strict shared record
    and primitive geometry contracts.
  - `scene_calibration.py` owns persisted adult-male, monitor, and robust
    estimator assumptions.
  - `scene_records.py` owns strict scene, viewer, and summary schemas.
  - `scene_geometry.py` owns pseudo-metric back-projection, robust estimators,
    scene axes, monitor-plane construction, transforms, and ray intersections.
  - `scene_artifacts.py` owns reading validated frame artifacts and writing
    scene JSON/JSONL artifacts.
  - `scene_viewer.py` owns viewer-data generation, packaged static asset
    copying, and localhost-only static serving.
  - `run_equivalence.py` owns strict run-to-run artifact equivalence checks for
    CPU/MPS optimization validation.
  - `unigaze_batch_benchmark.py` owns the Nakamura UniGaze device/batch
    benchmark harness and selected-batch report schema.
  - `pipeline.py`, `qa_summary.py`, and `cli.py` own orchestration, validation,
    and command-line entry points.
- `src/chess_gaze/viewer_assets/` contains package resources for the generated
  local viewer: HTML, CSS, JavaScript, and pinned remote Three.js dependency
  metadata. The app assets are copied into each run's `viewer/` directory; the
  generated page loads Three.js `0.185.0` from pinned jsDelivr npm module URLs
  at render time per ADR-0003. There is still no frontend build tree.
- `tests/` contains behavior tests for code in `src/chess_gaze/`, with
  package-path-mirroring tests under `tests/chess_gaze/` and repository
  packaging checks at top level.
- `docs/development/architecture/` contains current architecture guidance.
- `docs/development/decisions/` contains ADRs for architecture-significant
  decisions.
- `docs/superpowers/specs/` contains active design specs.
- `docs/superpowers/plans/` contains executable implementation plans.
- `docs/superpowers/closeouts/` contains completed-work closeouts.

## Ownership Rules

Add implementation modules only when they have meaningful behavior or protect a
real invariant. Avoid empty `core`, `services`, `adapters`, `engine`, `domain`,
or similarly generic packages until a concrete runtime seam exists.

Name modules after the concept they own. Prefer one deeper module with a stable
interface over several pass-through files that merely forward calls.

Keep tests near the behavior they verify by mirroring package paths under
`tests/` once modules exist. Configuration and documentation checks may use
top-level tests when they verify repository behavior rather than package
behavior.

If a source file grows past about 800 lines or starts owning three distinct
runtime responsibilities, perform a source-layout review before adding more.
If it grows past about 1,500 lines, write a split plan or document why the file
is intentionally deep.

## Local Development

Use `uv` for all project commands:

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Do not install development dependencies into a global Python environment.
