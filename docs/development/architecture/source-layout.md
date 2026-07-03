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
  - `video_decode.py`, `image_io.py`, and `visualization.py` own frame IO,
    optional debug-image writes, and processed-frame rendering.
  - `model_assets.py` and `model_registry.json` own local model trust and
    checksum validation.
  - `face_landmark_indices.py` owns named MediaPipe face landmark indices that
    encode anatomical left/right semantics.
  - `face_observation.py`, `eye_observation.py`, `head_pose.py`,
    `gaze_observation.py`, and `frame_observation.py` own per-frame evidence
    extraction, including in-memory crop geometry and optional retained eye
    crop debug artifacts.
  - `unigaze_runtime.py` owns UniGaze device/batch runtime validation, MPS
    preflight, synchronization, and inference metadata assembly.
  - `frame_records.py`, `errors.py`, and `geometry.py` own strict shared record,
    frame/crop image retention policies, and primitive geometry contracts.
  - `analysis_resume.py` owns interrupted-run discovery, compatible-run
    validation, retention-policy matching, committed frame-journal repair,
    checkpoint state, and cleanup of uncommitted or derived artifacts before
    resumed analysis rebuilds them.
  - `scene_calibration.py` owns persisted adult-male, gaze-sphere, and robust
    estimator assumptions.
  - `scene_records.py` owns strict scene, viewer, and summary schemas.
  - `scene_geometry.py` owns pseudo-metric back-projection, robust estimators,
    scene axes, and camera/scene transforms.
  - `sphere_projection.py` owns ray-to-gaze-sphere projection math, angular
    hit coordinates, and invalid sphere-intersection reasons.
  - `scene_artifacts.py` owns reading validated frame artifacts and writing
    scene JSON/JSONL artifacts, including sphere hit summaries.
  - `scene_viewer.py` owns viewer-data generation, packaged static asset
    copying, and localhost-only static serving.
  - `run_equivalence.py` owns strict run-to-run artifact equivalence checks for
    CPU/MPS optimization validation.
  - `unigaze_batch_benchmark.py` owns the Nakamura UniGaze device/batch
    benchmark harness and selected-batch report schema.
  - `pipeline.py`, `qa_summary.py`, and `cli.py` own orchestration,
    policy-aware artifact validation, and command-line entry points.

Source-layout review, 2026-07-03: `qa_summary.py` is intentionally deep at
1,093 lines after the streaming closeout repair. It still owns one cohesive
run-closeout boundary: loading durable run artifacts, validating their schemas,
aggregating QA counters, stabilizing byte counts, and writing the completion
seal. The custom viewer-data scanner is kept with the QA validator because its
only purpose is preserving that seal without materializing duplicate scene
frames. If the file grows toward 1,500 lines or adds non-QA artifact repair,
split JSONL streaming summaries and viewer-envelope scanning into named modules
with explicit interface tests.

Source-layout review, 2026-06-29: `pipeline.py` is intentionally deep at 839
lines after adding crop image retention plumbing. It still owns one cohesive
analysis orchestration boundary: request resolution, model/runtime preparation,
run creation/resume matching, frame processing, scene/viewer generation, and QA
closeout all coordinate the same completed-run contract. The crop-retention
change only threads an existing kind of artifact policy through this boundary.
If the file grows toward 1,500 lines or adds a second independent workflow,
split request resolution, observer construction, and artifact closeout into
named modules with explicit interface tests.

Source-layout review, 2026-07-03: `pipeline.py` remains intentionally deep at
866 lines after adding the `revalidating` closeout state. The new state is part
of the existing orchestration boundary rather than an independent workflow:
frame processing, derived scene/viewer artifacts, QA validation, and completion
seal ordering must stay coordinated. The previous split candidates remain the
right ones if this file grows further.

Source-layout review, 2026-06-29: `face_observation.py` is intentionally deep
at 1,203 lines, still below the 1,500-line split-plan trigger. It owns one
cohesive MediaPipe face-observation boundary: region definition, crop-to-full-
frame coordinate remapping, candidate arbitration, and observer result assembly
all change together and protect the same per-frame selection invariants. Before
further behavior expansion, the first split candidate is the arbitration and
crop-region policy helpers into separate named modules with explicit interface
tests.

Source-layout review, 2026-06-27: `unigaze_batch_benchmark.py` is intentionally
deep despite crossing the 800-line review trigger. The module is a finite
CLI-only benchmark harness for one optimization workflow, and keeping report
schema, candidate execution, equivalence writing, and artifact-retention policy
together keeps the benchmark contract auditable. If it grows toward 1,500 lines
or becomes a reusable benchmark framework, split report/schema, subprocess
runner, forward-timing, and retention helpers into separate concept modules.
- `src/chess_gaze/viewer_assets/` contains package resources for the generated
  local viewer: HTML, CSS, JavaScript, and pinned remote Three.js dependency
  metadata. The app assets are copied into each run's `viewer/` directory; the
  generated page loads Three.js `0.185.0` from pinned jsDelivr npm module URLs
  at render time per ADR-0003. There is still no frontend build tree.

Source-layout review, 2026-06-28: `viewer_assets/scene_viewer.js` is
intentionally deep after the large-run performance repair despite crossing the
800-line review trigger. It is a single generated browser app asset without a
frontend build tree; splitting it now would either add new browser module
loading constraints to direct `file://` artifacts or create pass-through helper
files copied beside every viewer. Keep the file together while it owns one
cohesive viewer surface: DOM bindings, current-frame rendering, accumulated
geometry caches, and render scheduling. If it grows toward 1,500 lines, adds a
second independently testable viewer mode, or the repo adds a frontend build
pipeline, split geometry-cache construction, hit-area math, and render-loop
scheduling into separate named viewer modules with explicit file-url packaging
tests.
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
