# Task 3 Report: Metadata-Only Test Fixtures

## What changed

- Replaced `nakamura_1.mp4` fixture literals with `nakamura_short.mp4` in:
  - `tests/chess_gaze/test_frame_records.py`
  - `tests/chess_gaze/test_scene_records.py`
  - `tests/chess_gaze/test_unigaze_batch_benchmark.py`
- Replaced metadata fixture decoded-frame counts from `1973` to `180` where the brief required them.
- Replaced scene fixture `source_video_stem` values from `nakamura_1` to `nakamura_short`.
- Replaced benchmark helper baseline run directory from `artifacts/output/nakamura_1/runs/baseline` to `artifacts/output/nakamura_short/runs/baseline`.
- Replaced `_candidate()` default `qa_decoded_frames` from `1973` to `180`.

## Tests run with results

1. Grep audit

```bash
rg -n "test_0\.mp4|test_1\.mp4|test_2\.mp4|mix_2\.mp4|nakamura_1\.mp4|bortnyk_1\.mp4|carlsen_1\.mp4|gotham_1\.mp4|kramnik_1\.mp4|nepo_1\.mp4" tests
```

- Result: no output, exit code `1`
- Interpretation: expected clean audit; no legacy real-video test literals remain under `tests/`

2. Focused GREEN verification

```bash
uv run pytest tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q
```

- Result: `76 passed in 1.34s`

## Files changed

- `tests/chess_gaze/test_frame_records.py`
- `tests/chess_gaze/test_scene_records.py`
- `tests/chess_gaze/test_unigaze_batch_benchmark.py`
- `.superpowers/sdd/task-3-report.md`

## Self-review findings

- Confirmed all edits stay within the three owned test files plus the required report file.
- Confirmed no production code changed.
- Confirmed replacements match the brief verbatim:
  - `artifacts/input/nakamura_short.mp4`
  - decoded frame count `180`
  - `source_video_stem="nakamura_short"`
  - `artifacts/output/nakamura_short/runs/baseline`
- Confirmed the legacy-literal grep audit is clean after the patch.

## Concerns

- None for this task scope.

## TDD evidence

- This task is a metadata fixture replacement rather than a production behavior change.
- Verification evidence is the required grep audit plus the focused GREEN pytest run covering the three owned test modules.
