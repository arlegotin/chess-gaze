# Streaming QA Closeout Closeout

Date: 2026-07-03

## Summary

Fixed the end-of-run memory spike that made a completed-looking analysis appear
stuck during final QA closeout.

The observed run
`artifacts/output/nepo_2/runs/20260630T082559Z-f865e2af` had complete frame,
scene, and viewer artifacts, but no `qa_summary.json`. `analysis_state.json`
said `status="complete"`, which made the interrupted closeout look finished
even though the completion seal was missing.

## Root Cause

`build_qa_summary()` reparsed and materialized the entire run:

- `records/frames.jsonl` with 28,141 records and about 901 MB
- `records/scene_frames.jsonl` with 28,141 records and about 84 MB
- `viewer/scene-data.json` at about 139 MB
- full Pydantic lists for frame records, scene records, and viewer scene data

Read-only reproduction before the fix:

```sh
/usr/bin/time -l uv run python -c "from pathlib import Path; from chess_gaze.artifact_runs import run_layout_from_dir; from chess_gaze.qa_summary import build_qa_summary; layout=run_layout_from_dir(Path('artifacts/output/nepo_2/runs/20260630T082559Z-f865e2af')); summary=build_qa_summary(layout); print(summary.final_status, summary.counts.frame_records, summary.counts.scene_frame_records, summary.byte_counts.total_run_bytes)"
```

Result:

```text
complete 28141 28141 1278665712
50.49 real
8392245248 maximum resident set size
```

The durable runtime surface was QA closeout, not MediaPipe, PyAV, the viewer
server, or a progress-thread leak.

## Changes

- `qa_summary.py` now streams `frames.jsonl`, `errors.jsonl`, and
  `scene_frames.jsonl` line by line, validating each record with Pydantic and
  aggregating only QA counters.
- `viewer/scene-data.json` is validated with a stdlib structural scanner that
  rejects unexpected top-level keys, validates each `frames` item as a
  `SceneFrameRecord`, validates each `valid_hit_points` item as a
  `ViewerHitPoint`, validates the small envelope with Pydantic, and cross-checks
  it against `run_manifest`, `video_manifest`, and `scene_summary`.
- `write_qa_summary()` accepts an already-built `QASummary` so pipeline closeout
  no longer reparses the whole run a second time.
- `AnalysisState.status` now includes `revalidating`.
- `pipeline.py` writes `revalidating` before QA closeout and reverts to it if QA
  writing fails in-process, while preserving the previous final-state-before-QA
  seal ordering.
- Added ADR-0006 and updated the resumable-analysis spec and source-layout
  review.

## Third-Party Guidance Used

- Python `Path.read_text()` returns the entire decoded file contents as a
  string, so it is inappropriate for multi-hundred-MB JSONL closeout:
  https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text
- Python `json.load()` deserializes a readable file into a Python object, which
  would still materialize large viewer data:
  https://docs.python.org/3/library/json.html#json.load
- Pydantic recommends direct `model_validate_json()` and avoiding validation
  work that is not needed:
  https://docs.pydantic.dev/latest/concepts/performance/
- PyAV container lifetime was checked and is already handled with context
  managers in `video_decode.py`:
  https://pyav.org/docs/stable/api/container.html

## Regression Coverage

- `tests/chess_gaze/test_qa_summary.py::test_build_qa_summary_streams_large_artifacts_without_whole_file_reads`
  prevents whole-file reads of `frames.jsonl`, `errors.jsonl`,
  `scene_frames.jsonl`, and `viewer/scene-data.json`.
- `tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_does_not_mark_complete_before_qa_summary_exists`
  proves QA write failures leave `analysis_state.status="revalidating"` and no
  `qa_summary.json`.

## Verification

Focused:

```sh
uv run pytest tests/chess_gaze/test_qa_summary.py -q
uv run pytest tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_analysis_resume.py -q
```

Result:

```text
17 passed
49 passed
```

Real-video required by the task:

```sh
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py::test_real_video_model_free_pipeline_writes_complete_artifact_contract -q
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py::test_nakamura_short_default_model_pipeline_does_not_create_crop_directory -q
```

Result:

```text
1 passed
1 passed, 18 warnings in 65.62s
```

Fixed large-run closeout measurement:

```text
complete 28141 28141 1278665712
73.00 real
224313344 maximum resident set size
```

Local gates:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Result:

```text
All checks passed.
71 files already formatted.
Success: no issues found in 71 source files.
```

## Residual Risk

- Closeout still scans large artifacts, and restoring strict viewer array
  validation increased elapsed time on the `nepo_2` run from the 50-second
  baseline to 73 seconds. The critical memory failure is fixed; a later
  performance improvement should use persisted incremental counters or a vetted
  streaming JSON parser.
- A process kill can still happen in the small window after terminal
  `analysis_state.json` is written and before the atomic `qa_summary.json`
  write completes. The long high-memory window is removed, and a subsequent run
  still treats missing `qa_summary.json` as incomplete.
