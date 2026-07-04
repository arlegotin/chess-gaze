# Streaming QA Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent completed analysis runs from appearing stuck and exhausting memory during final QA closeout.

**Architecture:** Keep `qa_summary.json` as the completion seal, but make QA closeout stream large artifacts instead of materializing full run records and viewer data. Add a nonterminal `revalidating` analysis state so interrupted closeout cannot leave `analysis_state.json` claiming `complete` before the QA seal exists.

**Tech Stack:** Python 3.12, Pydantic v2 model validation, stdlib JSON/path IO, PyAV video fixtures, pytest, uv.

## Global Constraints

- Preserve `qa_summary.json` as the only completion seal.
- Do not drop, sample, or downsample scene/viewer frame data.
- Do not add a new streaming JSON dependency for this repair.
- Validate `artifacts/input/nakamura_short.mp4` through the real-video pipeline contract.
- Keep CLI success stdout limited to the run directory and `viewer: <path>`.
- Treat whole-file reads of `frames.jsonl`, `errors.jsonl`, `scene_frames.jsonl`, and `viewer/scene-data.json` as a regression.

---

## Evidence Summary

- The interrupted run `artifacts/output/nepo_2/runs/20260630T082559Z-f865e2af` has 28,141 frame and scene records through `f000028140`, scene artifacts, and viewer artifacts.
- The same run has no `qa_summary.json`, but `analysis_state.json` says `status="complete"`.
- A read-only `build_qa_summary()` over that run took about 50 seconds and reached `8,392,245,248` bytes maximum resident set size.
- The root-cause surface is final QA validation in `src/chess_gaze/qa_summary.py`, which uses `Path.read_text().splitlines()` and full Pydantic materialization for large JSONL files and full `ViewerSceneData`.
- Primary documentation checked:
  - Python `Path.read_text()` returns the decoded file contents as a string.
  - Python `json.load()` deserializes a `.read()`-supporting file into a Python object.
  - Pydantic recommends `model_validate_json()` over `model_validate(json.loads(...))`, reusing validators when needed, and avoiding validation work that is not needed.
  - PyAV containers are already used through context managers in this code path.

## File Structure

- Modify `src/chess_gaze/qa_summary.py`: replace whole-run artifact materialization with streaming validation and aggregation.
- Modify `src/chess_gaze/pipeline.py`: write `analysis_state.status="revalidating"` before QA closeout and write the final `complete` or `failed` state only after `qa_summary.json` is durable.
- Modify `src/chess_gaze/analysis_resume.py`: allow `AnalysisState.status="revalidating"` and keep completion detection based on `qa_summary.json`.
- Modify `tests/chess_gaze/test_qa_summary.py`: regression for streaming large artifacts without whole-file reads.
- Modify `tests/chess_gaze/test_pipeline_contract.py`: regression for interrupted QA write not leaving `analysis_state.status="complete"` without `qa_summary.json`.
- Modify `docs/superpowers/specs/2026-06-28-resumable-analysis-design.md`: document the `revalidating` state and closeout ordering.
- Add `docs/superpowers/closeouts/2026-07-03-streaming-qa-closeout.md`: record root cause, durable surface, validation, and residual risk.

### Task 1: Closeout Regressions

**Files:**
- Modify: `tests/chess_gaze/test_qa_summary.py`
- Modify: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Consumes: `build_qa_summary(run_layout)`, `analyze_video(request, observers=...)`, `AnalysisState`.
- Produces: failing tests that prove the current whole-file QA reads and premature complete state.

- [x] **Step 1: Add a streaming-read regression**

Add a test that builds an existing fixture run, monkeypatches `Path.read_text` to raise for:

```python
{"frames.jsonl", "errors.jsonl", "scene_frames.jsonl", "scene-data.json"}
```

and asserts `build_qa_summary(layout)` still returns a complete summary.

- [x] **Step 2: Run the new QA regression and verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_qa_summary.py::test_build_qa_summary_streams_large_artifacts_without_whole_file_reads -q
```

Expected: FAIL from the monkeypatched `Path.read_text` on `frames.jsonl` or `scene-data.json`.

- [x] **Step 3: Add a premature-complete regression**

Add a pipeline test that monkeypatches `pipeline.write_qa_summary` to raise before writing `qa_summary.json`, runs a tiny fake-observer analysis, and asserts the surviving `analysis_state.json` status is `revalidating`, not `complete`.

- [x] **Step 4: Run the new pipeline regression and verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_does_not_mark_complete_before_qa_summary_exists -q
```

Expected: FAIL because the current pipeline writes `analysis_state.status="complete"` before `qa_summary.json`.

### Task 2: Streaming QA Summary

**Files:**
- Modify: `src/chess_gaze/qa_summary.py`

**Interfaces:**
- Consumes: strict Pydantic artifact models and run layout paths.
- Produces: `build_qa_summary(run_layout) -> QASummary` with the same public schema and low peak memory.

- [x] **Step 1: Replace JSONL list loading with streaming stats**

Create internal dataclasses for frame, error, and scene-frame summaries. Iterate each JSONL file line by line, validate each nonblank line with `model_validate_json(line)`, and aggregate counts, rates, samples, representative failures, and contiguity evidence without storing full record objects.

- [x] **Step 2: Add a streaming viewer-data envelope validator**

Use a stdlib byte scanner over `mmap` or buffered file IO to validate `viewer/scene-data.json` as a top-level JSON object, count the top-level `frames` and `valid_hit_points` arrays structurally, and Pydantic-validate only the envelope fields needed for cross-artifact consistency.

Historical note, 2026-07-04: the hit-area-only viewer-data follow-up removed
top-level `valid_hit_points`. Current streaming QA validation counts
`frames` and valid `frames[*].sphere_hit` records in
`gaze-scene-viewer-data-v3`.

- [x] **Step 3: Preserve artifact validation semantics**

Keep strict validation for `run_manifest.json`, `calibration.json`, `video_manifest.json`, `scene_manifest.json`, `scene_summary.json`, every frame JSONL line, every error JSONL line, and every scene-frame JSONL line. Treat malformed viewer JSON, count mismatches, missing viewer index, or envelope mismatches as schema validation failures.

- [x] **Step 4: Verify focused QA tests GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_qa_summary.py -q
```

Expected: PASS.

### Task 3: Completion-State Ordering

**Files:**
- Modify: `src/chess_gaze/analysis_resume.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `tests/chess_gaze/test_analysis_resume.py`
- Modify: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Consumes: `AnalysisState`, `write_analysis_state`, `write_qa_summary`.
- Produces: interrupted closeout state that is explicitly nonterminal until the QA seal exists.

- [x] **Step 1: Add `revalidating` to `AnalysisState.status`**

Extend the status literal to:

```python
Literal["processing", "revalidating", "complete", "failed"]
```

- [x] **Step 2: Write `revalidating` before QA closeout**

After scene and viewer generation, update `analysis_state.next_frame_index` to the decoded frame count and `status="revalidating"` before `build_qa_summary()`.

- [x] **Step 3: Avoid rebuilding QA during write**

Allow `write_qa_summary(run_layout, qa_summary_path, qa_summary=qa_summary)` to stabilize byte counts and atomically write the already-built summary instead of reparsing the run a second time.

- [x] **Step 4: Write terminal state before the QA seal and revert on QA-write failure**

After `QASummary` is built in memory, write
`analysis_state.status=qa_summary.final_status`, then atomically write the
already-built `qa_summary.json`. If the QA write fails in-process, revert
`analysis_state.status` to `revalidating` so no failed write leaves a durable
`complete` status without the completion seal. This preserves the existing
final-state-before-seal regression while removing the long high-memory window
between state completion and QA seal creation.

- [x] **Step 5: Verify focused pipeline/resume tests GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_analysis_resume.py -q
```

Expected: PASS.

### Task 4: Real-Video and Resource Verification

**Files:**
- Modify: `docs/superpowers/closeouts/2026-07-03-streaming-qa-closeout.md`
- Modify: `docs/superpowers/specs/2026-06-28-resumable-analysis-design.md`

**Interfaces:**
- Consumes: `artifacts/input/nakamura_short.mp4`.
- Produces: verification evidence and updated canonical docs.

- [x] **Step 1: Run sandbox-safe focused tests**

Run:

```sh
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py::test_real_video_model_free_pipeline_writes_complete_artifact_contract -q
```

Expected: PASS using `artifacts/input/nakamura_short.mp4`.

- [x] **Step 2: Run native real-model smoke if local runtime allows**

Run unsandboxed if required:

```sh
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py::test_nakamura_short_default_model_pipeline_does_not_create_crop_directory -q
```

Expected: PASS, or record exact native runtime failure.

- [x] **Step 3: Measure fixed QA closeout on the stopped run**

Run:

```sh
/usr/bin/time -l uv run python -c "from pathlib import Path; from chess_gaze.artifact_runs import run_layout_from_dir; from chess_gaze.qa_summary import build_qa_summary; layout=run_layout_from_dir(Path('artifacts/output/nepo_2/runs/20260630T082559Z-f865e2af')); summary=build_qa_summary(layout); print(summary.final_status, summary.counts.frame_records, summary.counts.scene_frame_records, summary.byte_counts.total_run_bytes)"
```

Expected: max RSS materially below the captured `8,392,245,248` bytes baseline.

- [x] **Step 4: Run local gates**

Run:

```sh
uv run pytest -m "not native_mediapipe and not local_socket"
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: PASS.

- [x] **Step 5: Write closeout and commit**

Record root cause, durable surface changed, third-party guidance, test evidence, remaining limitations, and exact commands. Commit docs, tests, and implementation in meaningful commits.
