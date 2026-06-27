# Task 1 Report: Model-Free Real-Video Contract Tests

## What changed

- Rebased `tests/chess_gaze/test_video_decode_real_video.py` onto:
  - `NAKAMURA_SHORT_VIDEO = Path("artifacts/input/nakamura_short.mp4")`
  - `NAKAMURA_SHORT_FRAME_COUNT = 180`
- Rebased `tests/chess_gaze/test_pipeline_real_video_contract.py` onto the same single-video, 180-frame contract.
- Rebased `tests/chess_gaze/test_qa_summary_real_video_contract.py` onto the same single-video, 180-frame contract.
- Updated `tests/chess_gaze/test_visualization_real_video.py` so `MANDATORY_VIDEO_PATHS` contains only `Path("artifacts/input/nakamura_short.mp4")`.
- Rebased `tests/chess_gaze/test_scene_artifacts_real_video_contract.py` from `artifacts/input/nakamura_1.mp4` to `artifacts/input/nakamura_short.mp4`, introduced `expected_frame_count = 180`, replaced the hard-coded `1973` expectations with `expected_frame_count`, and changed the last-frame assertion to `expected_frame_count - 1`.
- Refreshed the QA-summary real-video assertion to match the current repository contract: `summary.source_artifacts` now matches `summary.artifact_validation.source_artifacts` and includes the scene/viewer artifact keys already produced by the pipeline.

## RED and GREEN evidence

### RED

Command:

```sh
uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py -q
```

Result:

```text
FFFFFFF                                                                  [100%]
7 failed in 1.32s
```

Relevant failure evidence:

```text
AssertionError: missing mandatory real-data video: artifacts/input/test_1.mp4
AssertionError: missing mandatory real-data video: artifacts/input/test_2.mp4
AssertionError: missing mandatory real-data video(s): [PosixPath('artifacts/input/test_1.mp4'), PosixPath('artifacts/input/test_2.mp4')]
```

### GREEN

Command:

```sh
uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Result:

```text
.....                                                                    [100%]
5 passed in 117.37s (0:01:57)
```

## Tests run with results

- `uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py -q`
  - RED, `7 failed in 1.32s`
- `uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q`
  - first post-edit run exposed one stale QA-summary assertion
  - final GREEN run passed, `5 passed in 117.37s (0:01:57)`

## Files changed

- `tests/chess_gaze/test_video_decode_real_video.py`
- `tests/chess_gaze/test_pipeline_real_video_contract.py`
- `tests/chess_gaze/test_qa_summary_real_video_contract.py`
- `tests/chess_gaze/test_visualization_real_video.py`
- `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`
- `.superpowers/sdd/task-1-report.md`

## Self-review findings

- No production files changed.
- The real-video contract tests now point at the single mandatory short clip and assert the 180-frame contract consistently across decode, pipeline, QA-summary, visualization, and scene-artifact coverage.
- The QA-summary real-video test had an outdated exact `source_artifacts` map. Aligning it with the current repository-wide QA-summary contract removed the remaining false failure without weakening the core assertions.
- The focused verification command from the task brief passed fresh after the updates.

## Concerns

- The task brief described only fixture-input/count changes, but the current repository also required one owned-test assertion refresh in `test_qa_summary_real_video_contract.py` because scene/viewer artifacts are now part of the emitted QA-summary contract.
- I ran only the focused five-file pytest command required by the task, not the broader suite.
