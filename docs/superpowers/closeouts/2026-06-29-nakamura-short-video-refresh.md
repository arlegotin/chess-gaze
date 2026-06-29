# Nakamura Short Video Refresh Closeout

## Summary

- Refreshed real-video fixture expectations for the updated
  `artifacts/input/nakamura_short.mp4`.
- Preserved the existing decode contract: sha256
  `6364e160934c7a8de4318095172edeaf457f008f07a57f4266b2882225b5cb88`,
  decoded frame count `180`, and size `1920x1080`.
- Updated only content-specific assertions: MediaPipe face boxes, head-pose
  pitch-sign samples, and model-backed frames that produce fully valid
  recommended gaze.

## Root Cause

The replacement video kept the same container dimensions and decoded frame count
but changed visible frame content. Tests that asserted exact sampled face boxes,
all sampled frames looking downward, and three fully OK model-backed gaze frames
were still describing the previous clip.

## Durable Surface Changed

The real-video contract remains `artifacts/input/nakamura_short.mp4`; the test
fixtures now describe the current local file's observed per-frame evidence.

## Regression Evidence

- Metadata check:
  `UV_CACHE_DIR=.uv-cache uv run python -c 'from pathlib import Path; from chess_gaze.video_decode import inspect_video; p=Path("artifacts/input/nakamura_short.mp4"); i=inspect_video(p); print(i.video_manifest.model_dump_json(indent=2)); print("frame_count_expected", i.frame_count_expected); print("decoded", i.frame_count_decoded)'`
  reported sha256
  `6364e160934c7a8de4318095172edeaf457f008f07a57f4266b2882225b5cb88`,
  `frame_count_expected 180`, and `decoded 180`.
- Focused RED:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_eye_observation_real_video.py tests/chess_gaze/test_head_pose_real_video.py tests/chess_gaze/test_gaze_observation_real_video.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q`
  passed 9 tests and failed 4 content-specific assertions.
- Diagnostic run found new sampled face boxes, negative-pitch frames
  `(0, 30, 60, 90, 179)`, and fully OK model-backed frames including
  `(0, 90, 170)`.

## Verification

- Focused real-video gate:
  `UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_video_decode_real_video.py tests/chess_gaze/test_face_observation_real_video.py tests/chess_gaze/test_eye_observation_real_video.py tests/chess_gaze/test_head_pose_real_video.py tests/chess_gaze/test_gaze_observation_real_video.py tests/chess_gaze/test_visualization_real_video.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q`
  passed: `13 passed, 18 warnings in 153.34s (0:02:33)`.
- Full suite: `UV_CACHE_DIR=.uv-cache uv run pytest -q` passed:
  `372 passed, 18 warnings in 152.62s (0:02:32)`.
- Lint: `UV_CACHE_DIR=.uv-cache uv run ruff check .` passed:
  `All checks passed!`.
- Format: `UV_CACHE_DIR=.uv-cache uv run ruff format --check .` passed:
  `67 files already formatted`.
- Types: `UV_CACHE_DIR=.uv-cache uv run mypy` passed:
  `Success: no issues found in 67 source files`.

## Residual Risk

- Real-video tests that initialize MediaPipe must run outside this coding
  sandbox on macOS. Inside the sandbox, MediaPipe aborts while creating its GL
  context with `Check failed: service_ Service is unavailable`; the same minimal
  landmarker creation succeeds outside the sandbox.
