# Optional QA Task 4 Report: QA-Dependent Tooling And Real-Video Contracts

## Scope

Implemented Task 4 only from `.superpowers/sdd/task-4-brief.md`.

## What Changed

- Updated `tests/chess_gaze/test_unigaze_batch_benchmark.py` so the benchmark
  command-capture test now requires every `chess-gaze analyze` subprocess
  command to include `--qa-summary`.
- Updated `src/chess_gaze/unigaze_batch_benchmark.py` so
  `_run_analysis_subprocess()` always adds `--qa-summary` to the benchmark
  analyze command.
- Split the default real-video completion helper in
  `tests/chess_gaze/test_pipeline_real_video_contract.py` away from QA summary
  assumptions. The default contract now proves:
  - `result.qa_summary_path is None`
  - `validated_record_count is None`
  - `validated_error_count is None`
  - `qa_summary.json` is absent for the default `nakamura_short.mp4` path
  - analysis state still seals as `status == "complete"`
- Updated `tests/chess_gaze/test_qa_summary_real_video_contract.py` so the QA
  real-video contract explicitly opts into `AnalyzeRequest(generate_qa_summary=True)`
  and asserts `result.qa_summary_path == qa_summary_path`.
- Checked `tests/chess_gaze/test_run_equivalence.py`; no fixture changes were
  required because its fixtures already provide QA evidence.

## Red / Green Evidence

1. RED command:

   - `uv run pytest tests/chess_gaze/test_unigaze_batch_benchmark.py::test_benchmark_cli_writes_candidate_rows_and_removes_mps_env -q`
   - outcome: `1 failed`
   - failure point: `assert "--qa-summary" in command`

2. Run-equivalence fixture check:

   - `uv run pytest tests/chess_gaze/test_run_equivalence.py -q`
   - outcome: `8 passed in 0.13s`

3. Focused GREEN command:

   - `uv run pytest tests/chess_gaze/test_unigaze_batch_benchmark.py tests/chess_gaze/test_pipeline_real_video_contract.py::test_real_video_model_free_pipeline_writes_complete_artifact_contract tests/chess_gaze/test_qa_summary_real_video_contract.py::test_real_video_model_free_pipeline_writes_qa_summary_revalidation tests/chess_gaze/test_run_equivalence.py -q`
   - outcome: `22 passed in 2.76s`

## Files Changed

- `src/chess_gaze/unigaze_batch_benchmark.py`
- `tests/chess_gaze/test_unigaze_batch_benchmark.py`
- `tests/chess_gaze/test_pipeline_real_video_contract.py`
- `tests/chess_gaze/test_qa_summary_real_video_contract.py`
- `.superpowers/sdd/optional-qa-task-4-report.md`

## Notes

- `tests/chess_gaze/test_run_equivalence.py` was inspected and verified by the
  required command, but it did not need source edits.
- No Task 5+ work was implemented.

## Review Fix Evidence

Follow-up commit:
- `c779816 Fix Task 4 real-video contract tests`

Fix scope:
- Reworked the default no-QA real-video helper to use lightweight JSONL line
  inspection instead of Pydantic-validating every frame record.
- Kept targeted sanity validation in the model-free test by validating only the
  first few records.
- Added a regression test proving the default helper does not call
  `FrameRecord.model_validate_json`.
- Replaced stale deleted-helper calls in native no-QA tests with the default
  no-QA helper.

Fix tests:
- `pytest -q tests/chess_gaze/test_pipeline_real_video_contract.py -k "model_free_pipeline or avoids_full_record_validation"`
  - Result: `2 passed, 2 deselected in 2.71s`
- `pytest -q tests/chess_gaze/test_pipeline_real_video_contract.py -k "not native_mediapipe"`
  - Result: `2 passed, 2 deselected in 2.43s`
- `pytest -q tests/chess_gaze/test_pipeline_real_video_contract.py -k "default_model_pipeline"`
  - Result: failed with `Fatal Python error: Aborted` inside MediaPipe
    `face_landmarker.py` during `analyze_video(...)`, before the test reached
    contract assertions.

Remaining concern:
- Native default-model verification was blocked by a MediaPipe abort in this
  environment. The stale-helper `NameError` is removed from the test file.

## Review Blocker Follow-Up

Follow-up commit:
- `01dee49 test: bound no-QA real-video contract checks`

Fix scope:
- Reworked the default no-QA helper in
  `tests/chess_gaze/test_pipeline_real_video_contract.py` so it no longer
  reads or counts all `frames.jsonl` lines to prove completion.
- Completion sanity now uses `result.decoded_frame_count`,
  `analysis_state.next_frame_index`, completion status, QA absence, and
  required scene/viewer artifacts.
- Frame-record sanity now reads only the first JSONL line and the last JSONL
  line, parsing just those two records with `json.loads`.
- Strengthened the regression so it fails if the helper touches
  `frames.jsonl` through `Path.read_text()`, calls
  `FrameRecord.model_validate_json()`, or tries to materialize/iterate the
  whole file through `read()`, `readlines()`, or file iteration.

Fix tests:
- `uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py -k "model_free_pipeline or avoids_full_record_validation"`
  - Result: `2 passed, 2 deselected in 3.70s`
- `uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py -k "not native_mediapipe"`
  - Result: `2 passed, 2 deselected in 3.70s`

Notes:
- This follow-up touched only the requested test file plus this report.
- No production code changed.
- No Task 5 work or broader verification was added.
