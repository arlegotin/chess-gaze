# Hit-Area-Only Viewer Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Hit Points from the viewer data contract while preserving Hit Area from per-frame sphere-hit records.

**Architecture:** Delete the duplicated `valid_hit_points` viewer payload and schema branch. Keep `frames[*].sphere_hit` as the canonical source for hit-area geometry, valid-hit counts, QA validation, and run equivalence.

**Tech Stack:** Python 3.12, Pydantic strict schemas, pytest, standard-library streaming JSON scanner in `qa_summary.py`, generated static HTML/CSS/JS viewer assets.

## Global Constraints

- Work directly on `main`.
- Make meaningful commits along the way.
- Use test-first development.
- Do not remove `frames[*].sphere_hit`; Hit Area depends on it.
- Do not add dependencies or a frontend build system.
- Keep QA viewer-data validation streaming; do not materialize large viewer JSON.
- Keep direct `file://` and served viewer behavior from the prior visualization redo.
- Keep summary counts named after sphere hits, not Hit Points.

---

### Task 1: Lock The Slim Viewer Data Contract In Tests

**Files:**

- Modify: `tests/chess_gaze/test_scene_records.py`
- Modify: `tests/chess_gaze/test_scene_artifacts.py`
- Modify: `tests/chess_gaze/test_scene_viewer.py`
- Modify: `tests/chess_gaze/test_qa_summary.py`
- Modify: `tests/chess_gaze/test_run_equivalence.py`

**Interfaces:**

- Consumes: current `ViewerSceneData`, `build_viewer_scene_data()`, QA summary
  fixture runs, and run-equivalence fixtures.
- Produces: failing tests that require `valid_hit_points` to be absent and
  rejected.

- [ ] Update viewer schema tests to build `ViewerSceneData` without
  `valid_hit_points`.
- [ ] Add a strict-schema assertion that a payload containing
  `valid_hit_points` raises an unexpected-key validation error.
- [ ] Update scene-artifact tests to compare valid sphere-hit frames from
  `viewer_data.frames`, not duplicated hit-point records.
- [ ] Update generated viewer tests to assert serialized `scene-data.json` and
  embedded file-url JSON omit top-level `valid_hit_points`.
- [ ] Replace the QA malformed-hit-point test with a test that
  `valid_hit_points` is rejected as an unexpected top-level key.
- [ ] Update run-equivalence fixture payloads to omit `valid_hit_points`.
- [ ] Run the focused tests and confirm RED failures are caused by the old
  required/emitted hit-point data.

### Task 2: Remove Hit-Point Data From Production Code

**Files:**

- Modify: `src/chess_gaze/scene_records.py`
- Modify: `src/chess_gaze/scene_artifacts.py`
- Modify: `src/chess_gaze/qa_summary.py`

**Interfaces:**

- Consumes: RED tests from Task 1.
- Produces: `ViewerSceneData` without duplicated hit-point data.

- [ ] Delete `ViewerHitPoint` from `scene_records.py`.
- [ ] Delete `ViewerSceneData.valid_hit_points`.
- [ ] Remove `_valid_hit_points()` and the `valid_hit_points=` constructor
  argument in `scene_artifacts.py`.
- [ ] Remove `_ViewerSceneDataEnvelope.valid_hit_points_count`.
- [ ] Make `_scan_viewer_scene_data_payload()` count and validate only the
  `frames` array.
- [ ] Make `valid_hit_points` fail through the existing unexpected top-level
  key path.
- [ ] Run the focused tests and iterate to green.

### Task 3: Update Active Documentation And Close Out

**Files:**

- Modify: `README.md`
- Modify: `docs/development/decisions/0006-stream-qa-closeout-artifacts.md`
- Modify: `docs/superpowers/specs/2026-07-04-visualization-redo-design.md`
- Modify: `docs/superpowers/plans/2026-07-04-visualization-redo.md`
- Modify: `docs/superpowers/closeouts/2026-07-04-visualization-redo.md`
- Create: `docs/superpowers/closeouts/2026-07-04-hit-area-only-viewer-data.md`

**Interfaces:**

- Consumes: implemented data-contract change and verification evidence.
- Produces: non-conflicting active guidance and closeout evidence.

- [ ] Remove active docs language that says viewer hit-point data remains.
- [ ] Update ADR-0006 to describe streaming validation of viewer frames and the
  small envelope, not viewer hit points.
- [ ] Write a closeout with root cause, durable surface changed, focused and
  broad test evidence, and residual blockers.
- [ ] Run a subagent review and address material findings.
- [ ] Commit the final implementation and documentation changes.

## Self-Review

- Spec coverage: the plan covers schema, producer, streaming QA validation,
  fixtures, docs, closeout, and verification.
- Placeholder scan: no placeholder work remains; each task has concrete files
  and required assertions.
- Type/name consistency: `valid_hit_points` is removed only from viewer data;
  `sphere_hit` remains the canonical per-frame source for Hit Area.

