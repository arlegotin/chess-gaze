# Resumable Analysis Closeout

Date: 2026-06-28

## Summary

`chess-gaze analyze <video>` now resumes the newest compatible interrupted run
by default. Repeating the same analyze command continues from the committed
`records/frames.jsonl` prefix instead of starting a new run and redoing model
work from frame zero.

Completed runs are not resumed. Use `--no-resume` to force a fresh run even when
a compatible partial run exists.

## Root Cause

Before this work, each analyze invocation created a new timestamped run
directory. An interrupted run could leave a valid prefix in `records/frames.jsonl`
plus extra raw/crop artifacts from an uncommitted in-flight batch, but there was
no durable runtime boundary that let the next invocation distinguish committed
analysis from uncommitted artifacts.

The durable boundary is now the frame journal:

- committed frames are contiguous valid `FrameRecord` lines in
  `records/frames.jsonl`;
- records and errors are flushed and fsynced after each committed batch;
- `analysis_state.json` records the next frame index for observability;
- `qa_summary.json` remains the completion seal and is written only after final
  `analysis_state.json` is durable.

## Design Decisions

- Resume discovery chooses the newest incomplete run whose input path, input
  video hash and manifest, calibration, and inference metadata match the current
  request.
- Resume repairs the committed JSONL prefix, rebuilds `errors.jsonl` from that
  prefix, deletes uncommitted frame artifacts, and deletes derived scene/viewer/QA
  artifacts before continuing.
- The decoder still starts from frame zero and skips frames whose index is below
  the committed prefix. This avoids relying on frame-index seeking.
- `analysis_resume.py` owns compatible-run discovery, committed-prefix repair,
  checkpoint state, and cleanup safety checks.
- `qa_summary.py` owns final QA writing and in-memory byte-count stabilization so
  complete `qa_summary.json` is written once as the last durable completion seal.
- `pipeline.py` remains orchestration-only and is 758 lines after the refactor,
  below the 800-line source-layout review trigger.

## Dependency Evidence

Verified on 2026-06-28:

- Installed `av==17.1.0`: `InputContainer.seek()` documents timestamp/keyframe
  seeking, not exact frame-index seeking. The resume design therefore decodes
  from the start and skips committed frame indices.
- PyAV docs: `https://pyav.org/docs/stable/api/container.html`.
- Installed `pydantic==2.13.4`: resume and QA artifacts use strict Pydantic
  model validation, `model_validate_json()`, and `model_copy()` for nested
  immutable-style updates.
- Pydantic docs: `https://docs.pydantic.dev/latest/concepts/models/`.
- JSON Lines convention: one JSON value per line lets a partial tail be discarded
  while preserving the valid prefix.
- JSON Lines docs: `https://jsonlines.org/`.
- Python durable-write primitives: `os.fsync()` for committed JSONL handles and
  atomic replacement through the existing `atomic_write_bytes()` helper.
- Python docs: `https://docs.python.org/3/library/os.html#os.fsync` and
  `https://docs.python.org/3/library/os.html#os.replace`.
- The real analyzer still uses the existing MediaPipe and PyTorch/MPS runtime
  paths; no model, checkpoint, or inference library was changed.

## Implementation

Changed files:

- `src/chess_gaze/analysis_resume.py`
- `src/chess_gaze/artifact_runs.py`
- `src/chess_gaze/pipeline.py`
- `src/chess_gaze/qa_summary.py`
- `src/chess_gaze/cli.py`
- `tests/chess_gaze/test_analysis_resume.py`
- `tests/chess_gaze/test_artifact_runs.py`
- `tests/chess_gaze/test_pipeline_contract.py`
- `tests/chess_gaze/test_cli.py`
- `README.md`
- `docs/development/architecture/source-layout.md`
- `docs/superpowers/specs/2026-06-28-resumable-analysis-design.md`
- `docs/superpowers/plans/2026-06-28-resumable-analysis.md`

Commits:

- `6758354 docs: plan resumable analysis`
- `9e04dee feat: add analysis resume recovery utilities`
- `27c51e1 fix: enforce analysis resume cleanup boundary`
- `50d9f22 fix: reject symlinked resume cleanup paths`
- `44b53ad fix: preflight resume cleanup paths`
- `9311246 fix: skip malformed resume candidates`
- `82ca3d9 fix: skip cleanup-invalid resume candidates`
- `528c195 feat: resume interrupted analysis runs`
- `df0aa22 refactor: keep resume lifecycle out of pipeline`
- `966ed42 docs: document resumable analyze runs`
- `9596472 fix: seal resumed analysis after final state`
- `370a2bf test: cover resume edge cases`

## Review Findings

Subagent review of `82ca3d9..df0aa22` found one medium issue:

- `qa_summary.json` could be written complete before final `analysis_state.json`
  was durable. If interrupted in that window, the next run could treat the run as
  complete and skip repair.

Fix:

- added a regression that simulates final-state write failure and asserts no
  complete QA seal is left behind;
- changed finalization to build QA in memory first, write final
  `analysis_state.json`, then atomically write the final `qa_summary.json`;
- moved final QA byte-count stabilization into `qa_summary.py` so no intermediate
  complete seal is exposed.

Review also called out gaps for batched resume and `resume=False`; both are now
covered in `tests/chess_gaze/test_pipeline_contract.py`.

## Real Video Evidence

Final-code real verification used:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 \
  --output-root /private/tmp/chess-gaze-resume-final.FyX6DN \
  --models-root models
```

The first run was interrupted with `KeyboardInterrupt` during real batch
preparation/inference. Partial evidence:

- run: `/private/tmp/chess-gaze-resume-final.FyX6DN/nakamura_short/runs/20260628T123339Z-094f0c49`
- `records/frames.jsonl`: 63 committed records
- `analysis_state.json`: `next_frame_index=63`, `status=processing`
- no `qa_summary.json` completion seal

Rerunning the same command resumed the same run directory and completed:

- `analysis_state.json`: `next_frame_index=180`, `status=complete`
- `qa_summary.json`: `final_status=complete`
- decoded frames: 180
- frame records: 180
- raw frames: 180
- processed frames: 180
- scene frame records: 180
- `schema_validation_passed=True`
- `counts_match=True`
- `qa_summary.byte_counts.total_run_bytes=273057041`
- actual run file bytes: `273057041`

Observed native runtime warnings:

- macOS Objective-C duplicate `AVFFrameReceiver` / `AVFAudioReceiver` class
  warnings from `cv2` and `av` shared libraries.
- MediaPipe/TFLite informational warnings during face-landmarker startup.

These warnings did not prevent successful completion.

## Verification Commands

Targeted RED evidence:

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py::test_final_state_write_failure_does_not_leave_complete_qa_summary -q
```

Initial result: `1 failed`; old code left `qa_summary.json` after simulated
final-state write failure.

Focused verification after fixes:

```sh
uv run pytest tests/chess_gaze/test_analysis_resume.py tests/chess_gaze/test_artifact_runs.py::test_run_layout_from_existing_run_dir tests/chess_gaze/test_cli.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py -q
```

Result: `65 passed in 2.09s`.

Full local gates:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run mypy
```

Results:

- `ruff check`: `All checks passed!`
- `ruff format --check`: `67 files already formatted`
- `pytest`: `367 passed, 18 warnings in 120.05s`
- `mypy`: `Success: no issues found in 67 source files`

## Residual Risk

The implementation does not preserve in-flight model work from an interrupted
batch. It resumes from the last committed frame record, so the interrupted batch
is redone. This is intentional: the committed JSONL prefix is the durable
boundary, and redoing an uncommitted batch is safer than trusting artifacts that
may have been written before an exception or signal.
