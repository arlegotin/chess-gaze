# Gaze Precision Improvement Closeout

Date: 2026-07-05

## Summary

This branch improves the strongest measurable precision surface available in the
repo: the UniGaze input contract. The default UniGaze preprocessing is now
`reference_face2x_imagenet`, which expands the face crop to `2.0x` and applies
RGB ImageNet mean/std normalization. The old tight RGB `/255` crop remains
available as `legacy_bbox_rgb01` for rollback and benchmarking.

The branch also adds the two precision surfaces needed for true point-of-gaze
work:

- calibrated target-plane projection from camera-space gaze rays to normalized
  screen/board coordinates;
- an affine per-person/per-video gaze calibrator with held-out evaluation.

No claim is made that `nakamura_short.mp4` now has measured point-of-gaze
accuracy. That video has no target-plane calibration and no ground-truth gaze
labels, so the real-video benchmark reports proxy stability/coverage metrics.

## Research

Primary sources checked on 2026-07-05:

- UniGaze repository and video inference script:
  https://github.com/ut-vision/UniGaze and
  https://github.com/ut-vision/UniGaze/blob/main/unigaze/predict_gaze_video.py
- UniGaze paper: https://arxiv.org/abs/2502.02307
- UniGaze model files and model card:
  https://huggingface.co/UniGaze/UniGaze-models
- UniGaze PyPI package: https://pypi.org/project/unigaze/
- OpenCV camera calibration/PnP reference:
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- MediaPipe Face Landmarker docs:
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- Webcam gaze calibration papers:
  https://arxiv.org/abs/1905.04451,
  https://arxiv.org/abs/2001.09284,
  https://arxiv.org/abs/2009.01270

Main finding: `unigaze==0.1.3` exposes the model loader but not the full
official `gazelib` geometric-normalization pipeline used by the UniGaze video
script. The strongest safe repo-local change was therefore to make preprocessing
explicit and benchmarkable, then move the default closer to the official script
where assets are available: enlarged face ROI and ImageNet transform.

## Changes

- Added `src/chess_gaze/unigaze_preprocessing.py`.
- Persisted preprocessing profile, crop scale, and RGB normalization constants
  in `calibration.json`.
- Added `--unigaze-preprocessing-profile` with profiles:
  `legacy_bbox_rgb01` and `reference_face2x_imagenet`.
- Updated frame observation, batch benchmark crop replay, and legacy calibration
  parsing to use the persisted preprocessing contract.
- Added `src/chess_gaze/target_plane.py` and optional target-plane config under
  `target_plane` in `config.json`.
- Added optional `target_plane` and per-frame `target_plane_hit` scene/viewer
  records; the Three.js viewer now draws a configured plane and current-frame
  plane hit marker.
- Added `src/chess_gaze/gaze_calibration.py` for affine calibration:
  `(yaw, pitch, head_yaw, head_pitch) -> (target_x, target_y)`.
- Added `src/chess_gaze/gaze_precision_benchmark.py` for run-to-run proxy
  precision reports.

## Benchmark

Input: `artifacts/input/nakamura_short.mp4`, current decoded frame count `1200`.

MPS note: sandboxed `uv run python` reported MPS unavailable, but the repo venv
reported MPS available. Real inference/benchmark runs were therefore executed
with `.venv/bin/python`.

Fresh runs:

- Legacy baseline:
  `artifacts/output/precision-benchmarks/nakamura_short/runs/20260705T114842Z-331fdf3e`
- Reference preprocessing:
  `artifacts/output/precision-benchmarks/nakamura_short/runs/20260705T115106Z-90712506`
- Comparison report:
  `artifacts/output/precision-benchmarks/nakamura_short/gaze_precision_comparison.json`

Results:

| Metric | Legacy | Reference | Change |
|---|---:|---:|---:|
| decoded frames | 1200 | 1200 | 0 |
| valid appearance gaze rate | 1.000 | 1.000 | 0 |
| valid sphere hits | 1182 | 1185 | +3 |
| median frame-step ray angle | 0.066635 rad | 0.042889 rad | -35.64% |
| p95 frame-step ray angle | 0.261756 rad | 0.211610 rad | -19.16% |
| p99 frame-step ray angle | 0.401026 rad | 0.327892 rad | -18.24% |

Interpretation: on this video, the reference UniGaze preprocessing substantially
reduces frame-to-frame ray jitter without reducing valid gaze coverage. This is
a precision proxy, not ground-truth accuracy.

Target-plane and calibrator benchmark status:

- Target plane: unit and scene integration tests validate ray-plane
  intersection, normalized coordinates, mirroring, schema persistence, QA
  validation, and viewer rendering. `nakamura_short.mp4` has no calibrated
  physical screen/board plane, so the real-video comparison has
  `valid_target_plane_hit_frames = 0`.
- Affine calibrator: unit tests validate exact affine recovery, held-out error
  reporting, insufficient-sample rejection, and non-finite input rejection.
  There are no calibration labels for `nakamura_short.mp4`, so no honest
  held-out point-of-gaze accuracy number can be reported for that video.

## Verification

- Focused preprocessing/calibration tests:
  `13 passed`.
- Target-plane non-socket suite:
  `122 passed, 2 deselected`.
- Local socket viewer tests outside sandbox:
  `2 passed, 34 deselected`.
- Gaze precision benchmark tests:
  `3 passed`.
- Broad suite:
  `UV_CACHE_DIR=.uv-cache uv run pytest -q -m 'not native_mediapipe and not local_socket' --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py`
  -> `439 passed, 12 deselected, 18 warnings`.
- Lint:
  `UV_CACHE_DIR=.uv-cache uv run ruff check .`
  -> `All checks passed!`.
- Type check:
  `UV_CACHE_DIR=.uv-cache uv run mypy`
  -> `Success: no issues found in 78 source files`.
- Fresh MPS QA summaries:
  both benchmark runs had `final_status=complete`, `decoded_frames=1200`,
  `face_gaze_valid_rate=1.0`, `schema_validation_passed=true`, and
  `counts_match=true`.

## Residual Risk

- Full official UniGaze geometric normalization is still not implemented because
  the installed package does not include the required official video-inference
  geometry helpers/assets. This branch makes the repo’s preprocessing explicit
  and closer to the reference script, but not identical to the full official
  denormalization path.
- Target-plane precision requires real calibration fields in `config.json`.
  Without physical camera/screen/board calibration, the repo should not present
  a webcam ray as a chessboard point.
- Calibrator quality requires labeled calibration samples and held-out
  evaluation. Training error on calibration points is not an accuracy claim.
