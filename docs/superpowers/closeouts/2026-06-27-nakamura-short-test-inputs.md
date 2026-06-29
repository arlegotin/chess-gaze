# Nakamura Short Test Inputs Closeout

> Fixture note, 2026-06-29: `artifacts/input/nakamura_short.mp4` was replaced
> after this closeout. Current fixture expectations and digest are in
> [2026-06-29-nakamura-short-video-refresh.md](2026-06-29-nakamura-short-video-refresh.md).
> Historical values below describe the prior clip.

## Summary

- Replaced tests that consumed removed or slower `artifacts/input/*.mp4` real
  videos with `artifacts/input/nakamura_short.mp4`.
- Preserved model-free decode, pipeline, QA summary, visualization, scene
  artifact, MediaPipe, eye, head-pose, UniGaze, schema, and benchmark fixture
  coverage.
- Untracked stray `.superpowers/sdd` handoff artifacts so workflow scratch files
  stay out of repository history.

## Root Cause

Tests encoded mandatory dependencies on legacy local videos that are no longer
present or are slower than the short canonical clip.

## Durable Surface Changed

The test suite now has one canonical real-video fixture for local real-video
coverage: `artifacts/input/nakamura_short.mp4`.

## Regression Evidence

- Initial focused RED:
  `uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py -q`
  failed with 7 missing `artifacts/input/test_1.mp4` and
  `artifacts/input/test_2.mp4` assertions.
- `nakamura_short.mp4` was verified locally as sha256
  `4f4f4f0035dacd0e469e50ed1c8f78d4de93964474f3a5656117cc4d13383d6e`,
  decoded frame count `180`, and size `1920x1080`.

## Verification

- Changed-file suite:
  `uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_eye_observation_real_video.py tests/chess_gaze/test_head_pose_real_video.py tests/chess_gaze/test_gaze_observation_real_video.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q`
  passed: `89 passed, 18 warnings in 140.43s (0:02:20)`.
- Full suite: `uv run pytest -q` passed:
  `347 passed, 18 warnings in 209.32s (0:03:29)`.
- Lint: `uv run ruff check .` passed: `All checks passed!`.
- Format: `uv run ruff format --check .` passed:
  `65 files already formatted`.
- Types: `uv run mypy` passed:
  `Success: no issues found in 65 source files`.
- Legacy-video test audit:
  `rg -n "test_0\\.mp4|test_1\\.mp4|test_2\\.mp4|mix_2\\.mp4|nakamura_1\\.mp4|bortnyk_1\\.mp4|carlsen_1\\.mp4|gotham_1\\.mp4|kramnik_1\\.mp4|nepo_1\\.mp4" tests`
  produced no output.
- Repository hygiene: `git ls-files .superpowers/sdd` produced no output.

## Residual Risk

- The passing pytest runs still emit 18 `torch.jit.script` deprecation warnings
  from the installed UniGaze dependency path. They are not introduced by this
  test-input repair and do not affect pass/fail behavior.
- Real-video diversity is intentionally reduced to the available canonical short
  clip per task scope; removed long-clip/test-clip scenario diversity is no
  longer asserted by tests.
