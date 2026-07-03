# Optional QA Summary Closeout

Date: 2026-07-03

## Summary

Default analysis no longer generates or writes `qa_summary.json`. QA summary
generation is available through `--qa-summary` or
`AnalyzeRequest(generate_qa_summary=True)`.

## Root Cause

`qa_summary.json` had been treated as the universal completion seal even though
it is an expensive audit artifact, not a necessary artifact for a healthy
viewer-ready run. That made a successful default analysis continue into strict
record revalidation and summary construction, which was unnecessary for the
default path and could keep consuming memory after useful artifacts had already
been produced.

## Durable Surface Changed

- Run manifests now persist QA closeout policy through
  `qa_summary_policy.generate_qa_summary`.
- Resume discovery includes QA policy compatibility so default no-QA and
  explicit QA runs do not resume into each other.
- No-QA runs complete through `analysis_state.json` plus required derived
  artifact existence checks.
- QA-requested and legacy QA-required runs keep strict streamed QA validation
  and `qa_summary.json` closeout.
- QA-dependent tooling, including benchmark subprocess analysis, explicitly
  requests `--qa-summary`.
- Canonical docs now describe `qa_summary.json` as optional by default and as
  the strict QA seal only for QA-requested or legacy QA-required runs.

## Regression Coverage

Feature-level tests added or changed across this plan:

- CLI default and `--qa-summary` request plumbing in
  `tests/chess_gaze/test_cli.py`.
- Run manifest QA policy serialization and legacy default parsing in
  `tests/chess_gaze/test_frame_records.py`.
- Pipeline default no-QA behavior, explicit QA behavior, failure handling, and
  no default `build_qa_summary()` / `write_qa_summary()` calls in
  `tests/chess_gaze/test_pipeline_contract.py`.
- Resume policy matching and cheap no-QA completion classification in
  `tests/chess_gaze/test_analysis_resume.py`.
- Benchmark subprocess opt-in in
  `tests/chess_gaze/test_unigaze_batch_benchmark.py`.
- Real-video default no-QA and explicit QA contracts using
  `artifacts/input/nakamura_short.mp4` in
  `tests/chess_gaze/test_pipeline_real_video_contract.py` and
  `tests/chess_gaze/test_qa_summary_real_video_contract.py`.
- Bounded JSONL boundary checks in
  `tests/chess_gaze/test_pipeline_real_video_contract.py`, including a
  regression for a last record larger than the previous fixed tail chunk.

Task 6 added or changed:

- `tests/chess_gaze/test_pipeline_real_video_contract.py`
  - `test_read_last_jsonl_record_handles_last_line_larger_than_tail_chunk`
  - `test_default_completed_artifact_contract_avoids_full_record_validation`
- `tests/chess_gaze/test_pipeline_contract.py`
  - `test_analyze_video_retains_raw_and_processed_frame_images_when_requested`
  - `test_analyze_video_resumes_latest_compatible_partial_run`
  - `test_analyze_video_requested_qa_fails_on_counts_mismatch`

Task 6 supporting helpers changed:

- `tests/chess_gaze/test_pipeline_real_video_contract.py`
  - `_read_last_jsonl_record`
  - `_assert_default_completed_artifact_contract`

## Verification

```sh
uv run pytest tests/chess_gaze/test_cli.py tests/chess_gaze/test_analysis_resume.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q
```

Result: passed, `105 passed in 1.64s`.

```sh
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py::test_real_video_model_free_pipeline_writes_complete_artifact_contract -q
```

Result: passed, `1 passed in 1.57s`.

```sh
uv run pytest tests/chess_gaze/test_qa_summary_real_video_contract.py::test_real_video_model_free_pipeline_writes_qa_summary_revalidation -q
```

Result: passed, `1 passed in 1.74s`.

```sh
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py::test_nakamura_short_default_model_pipeline_does_not_create_crop_directory -q
```

Result: passed, `1 passed, 18 warnings in 47.35s`.

```sh
uv run ruff check .
```

Result: passed, `All checks passed!`.

```sh
uv run ruff format --check .
```

Result: passed, `71 files already formatted`.

```sh
uv run mypy
```

Result: passed, `Success: no issues found in 71 source files`.

```sh
uv run pytest -m "not native_mediapipe and not local_socket"
```

Result: passed, `421 passed, 14 deselected, 18 warnings in 7.26s`.

```sh
uv run pytest -m local_socket
```

Result: passed, `2 passed, 433 deselected in 1.02s`.

```sh
uv run python -c 'from pathlib import Path
from tempfile import TemporaryDirectory
from chess_gaze.pipeline import AnalyzeRequest, ObserverBundle, analyze_video
from tests.chess_gaze.test_pipeline_contract import _fake_record
with TemporaryDirectory() as tmp:
    result = analyze_video(
        AnalyzeRequest(
            video_path=Path("artifacts/input/nakamura_short.mp4"),
            output_root=Path(tmp) / "output",
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )
    print(result.qa_summary_path)
    print((result.layout.run_dir / "qa_summary.json").exists())
    print((result.layout.run_dir / "analysis_state.json").read_text())'
```

Result: passed. Output included:

```text
None
False
{"schema_version":"analysis-state-v1","run_id":"20260703T183636Z-6c3f66cb","input_path":"artifacts/input/nakamura_short.mp4","source_video_sha256":"6364e160934c7a8de4318095172edeaf457f008f07a57f4266b2882225b5cb88","frame_count_decoded":180,"next_frame_index":180,"status":"complete","updated_at_utc":"2026-07-03T18:36:37Z"}
```

## Residual Risk

- Native default-model smoke passed locally; no native smoke was skipped.
- Native/default-model tests emitted 18 torch `torch.jit.script` deprecation
  warnings. They are pre-existing third-party/runtime warnings and did not
  fail the gate.
- The direct no-QA inspection emitted Objective-C duplicate-class warnings from
  both `cv2` and `av` vendored `libavdevice` libraries:
  `AVFFrameReceiver` and `AVFAudioReceiver` are implemented in both packages.
  The command still exited 0 and confirmed `qa_summary_path=None`,
  `qa_summary.json` absent, and `analysis_state.status="complete"`.
