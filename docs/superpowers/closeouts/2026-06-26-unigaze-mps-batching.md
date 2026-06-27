# UniGaze MPS Batching Closeout

Date: 2026-06-27

## Summary

UniGaze inference now supports explicit `unigaze_device="mps"` and
`unigaze_batch_size > 1` without changing the frame-level evidence semantics.
The compatibility default remains CPU batch size 1. Batching is limited to the
UniGaze tensor inference boundary; decoded frames, raw/processed frame writes,
frame records, scene records, QA, and viewer artifacts remain one-to-one with
decoder frame order.

The corrected full Nakamura benchmark selected `mps` batch size `7` on the
Apple M3 Max: `667.389 s` wall time for 1973 frames (`2.956 fps`), versus the
current CPU/1 flow at `1089.510 s` (`1.811 fps`). That is a `1.63x` full-analysis
speedup and `422.122 s` saved for this video.

## Root Cause

The previous slow path loaded UniGaze on CPU and called the model one crop at a
time. UniGaze itself accepts batched tensors and preserves batch dimension in
`pred_gaze`, but the repository wrapper enforced a single-row output and the
pipeline had no batch observer seam.

During benchmark execution, the first full benchmark report was invalid because
the benchmark parser trusted the first non-empty stdout line from
`chess-gaze analyze`; UniGaze model loading also prints to stdout before the run
directory. Commit `37c7cc7` fixed `_parse_run_dir()` to select a real run
directory containing `run_manifest.json` and `records/`.

## Implementation

- Added strict config and CLI overrides for `unigaze_device` and
  `unigaze_batch_size`; defaults remain `cpu` and `1`.
- Added `unigaze_runtime.py` for MPS availability/env validation, checkpoint
  construction, requested-batch dummy preflight, synchronization, and persisted
  inference metadata.
- Added `UniGazeModel.predict_batch()` while preserving `predict()` as a
  single-row compatibility wrapper.
- Added a batch observer seam in the pipeline and `ModelBackedFrameObserver`;
  only selected-face UniGaze crops are stacked, and outputs are mapped back to
  original frame order.
- Added strict artifact equivalence comparison in `run_equivalence.py`.
- Added `unigaze_batch_benchmark.py` to benchmark CPU/MPS batch sizes
  `1, 2, 4, 7, 8, 16, 32, 64`, record failures as rows, retain only necessary
  full run directories, and select the fastest passing MPS batch above 1.
- After final code review, added a narrow `ModelInferenceError` boundary so
  post-preflight UniGaze batch inference contract failures become
  `PipelineError(USAGE)` in the analysis pipeline, while per-row non-finite
  outputs still mark only the originating frame invalid.
- After final code review, made benchmark preflight failures first-class
  candidate failures: rows are marked `preflight_failed` and the full analysis
  subprocess is skipped for that candidate.
- Updated README and source-layout documentation.

## Benchmark Matrix

Benchmark report:
`artifacts/output/benchmarks/2026-06-26-unigaze-mps-batching.json`

Source video:
`artifacts/input/nakamura_1.mp4`

Source video sha256:
`eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45`

Current-flow CPU/1 baseline:
`artifacts/output/nakamura_1/runs/20260627T030933Z-681f237e`

Fresh benchmark CPU/1 equivalence baseline:
`/private/tmp/chess-gaze-benchmarks/nakamura_1/runs/20260627T102547Z-9763f47f`

| Device | Batch | Status | Wall s | FPS | UniGaze forward median s | Max pitch/yaw delta rad | Max scene ray delta | Max monitor UV delta m | Error |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cpu | 1 | passed | 1089.510 | 1.811 | 31.696 | 0.000 | 0.000 | 0.000 |  |
| cpu | 2 | passed | 970.302 | 2.033 | 27.315 | 2.384e-07 | 1.891e-07 | 7.706e-07 |  |
| cpu | 4 | equivalence_failed | 916.688 | 2.152 | 25.567 | 1.669e-06 | 1.257e-06 | 7.731e-06 | EQUIVALENCE_FAILED |
| cpu | 7 | equivalence_failed | 961.380 | 2.052 | 23.069 | 1.669e-06 | 1.257e-06 | 7.740e-06 | EQUIVALENCE_FAILED |
| cpu | 8 | equivalence_failed | 984.743 | 2.004 | 23.534 | 1.669e-06 | 1.257e-06 | 7.731e-06 | EQUIVALENCE_FAILED |
| cpu | 16 | equivalence_failed | 965.946 | 2.043 | 24.265 | 2.444e-06 | 1.747e-06 | 7.041e-06 | EQUIVALENCE_FAILED |
| cpu | 32 | equivalence_failed | 1086.957 | 1.815 | 23.346 | 2.444e-06 | 1.747e-06 | 7.041e-06 | EQUIVALENCE_FAILED |
| cpu | 64 | equivalence_failed | 2267.212 | 0.870 | 69.407 | 2.444e-06 | 1.747e-06 | 7.041e-06 | EQUIVALENCE_FAILED |
| mps | 1 | passed | 1179.676 | 1.672 | 53.263 | 3.457e-06 | 2.249e-06 | 1.031e-05 |  |
| mps | 2 | passed | 774.778 | 2.547 | 10.314 | 3.457e-06 | 2.249e-06 | 1.031e-05 |  |
| mps | 4 | passed | 795.687 | 2.480 | 9.813 | 3.457e-06 | 2.249e-06 | 1.031e-05 |  |
| mps | 7 | passed | 667.389 | 2.956 | 7.542 | 3.457e-06 | 2.249e-06 | 1.031e-05 |  |
| mps | 8 | passed | 686.279 | 2.875 | 7.201 | 3.457e-06 | 2.249e-06 | 1.031e-05 |  |
| mps | 16 | passed | 707.084 | 2.790 | 7.786 | 3.457e-06 | 2.249e-06 | 1.031e-05 |  |
| mps | 32 | passed | 715.083 | 2.759 | 7.211 | 3.457e-06 | 2.249e-06 | 1.031e-05 |  |
| mps | 64 | passed | 750.165 | 2.630 | 7.786 | 3.457e-06 | 2.249e-06 | 1.031e-05 |  |

No valid benchmark candidate failed with OOM, unsupported MPS ops, or analysis
failure. CPU batch sizes 4 and above were rejected only by the strict CPU
`1e-6` equivalence tolerance. All MPS candidates passed the approved MPS
tolerances; MPS batch 7 was selected because it had the lowest full-analysis
wall time, even though some larger batches had slightly lower isolated UniGaze
forward medians.

The benchmark report stores every isolated UniGaze forward repetition and the
median. Forward min/max values are derivable from the repetition arrays; they
are not duplicated as explicit fields in the v1 report schema.

## Selected Optimized Profile

Use this explicit profile on this Apple M3 Max with the verified local models:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK \
  -u PYTORCH_MPS_FAST_MATH \
  -u PYTORCH_MPS_PREFER_METAL \
  .venv/bin/chess-gaze analyze artifacts/input/nakamura_1.mp4 \
  --output-root artifacts/output \
  --models-root models \
  --unigaze-device mps \
  --unigaze-batch-size 7
```

The default CLI behavior remains CPU/1; MPS is intentionally opt-in because CPU
and MPS floating-point outputs are not bitwise identical.

## Equivalence Evidence

Benchmark MPS tolerance:

- appearance pitch/yaw: `1e-3 rad`
- scene ray component: `1e-3`
- monitor U/V: `2e-3 m`
- exact frame identity, validity/status/error, and artifact-count contracts

Final optimized run compared to the fresh benchmark CPU/1 baseline:

- report:
  `artifacts/output/benchmarks/2026-06-26-final-mps7-benchmark_fresh_cpu1-equivalence.json`
- passed: `true`
- exact mismatches: `0`
- numeric mismatches: `0`
- max pitch/yaw delta: `3.4570693969726562e-06`
- max scene ray component delta: `2.249202009863005e-06`
- max monitor U/V delta: `1.0314956934598385e-05 m`

Final optimized run compared to the current-flow CPU/1 baseline:

- report:
  `artifacts/output/benchmarks/2026-06-26-final-mps7-current_flow_cpu1-equivalence.json`
- passed: `true`
- exact mismatches: `0`
- numeric mismatches: `0`
- max pitch/yaw delta: `3.4570693969726562e-06`
- max scene ray component delta: `2.249202009863005e-06`
- max monitor U/V delta: `1.0314956934598385e-05 m`

## Real Nakamura Verification

Final optimized run:
`artifacts/output/nakamura_1/runs/20260627T170119Z-edc98c89`

Viewer:
`artifacts/output/nakamura_1/runs/20260627T170119Z-edc98c89/viewer/index.html`

Observed run facts:

- `qa_summary.final_status`: `complete`
- `artifact_validation.counts_match`: `true`
- `artifact_validation.schema_validation_passed`: `true`
- decoded frames: `1973`
- frame records: `1973`
- scene frame records: `1973`
- raw frames: `1973`
- processed frames: `1973`
- crop files: `3946`
- viewer frames: `1973`
- total run bytes: `3013824563`
- source video sha256:
  `eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45`

Run manifest inference metadata:

- `observer_source`: `default_model_observer`
- `unigaze_model_id`: `unigaze-h14-joint`
- `unigaze_device`: `mps`
- `unigaze_batch_size`: `7`
- `torch_version`: `2.12.1`
- `torch_mps_available`: `true`
- `mps_fallback_env`: `unset`
- `mps_fast_math_env`: `unset`
- `mps_prefer_metal_env`: `unset`
- `mps_preflight_passed`: `true`

## Verification Commands

Commands that used real MediaPipe/MPS/macOS native paths were run unsandboxed.

- Focused implementation suite:
  `.venv/bin/pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_run_equivalence.py tests/chess_gaze/test_unigaze_runtime.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q`
  - result after final review fixes and benchmark wording cleanup:
    `143 passed, 18 warnings in 3.80s`
- Scene focused suite:
  `.venv/bin/pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py -q`
  - result: `49 passed in 1.96s`
- Required real-video smoke:
  `.venv/bin/pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_rejects_nakamura_overexpanded_faces -q`
  - result after final review fixes: `1 passed in 2.61s`
- Full suite:
  `.venv/bin/pytest`
  - result after final review fixes:
    `7 failed, 330 passed, 7 skipped, 18 warnings in 455.69s`
  - failure reason: all seven failures asserted missing ignored legacy media
    `artifacts/input/test_1.mp4` or `artifacts/input/test_2.mp4`; those files
    are not present locally. No failure referenced `nakamura_1.mp4`, MPS,
    batching, manifest metadata, equivalence, or benchmark behavior.
- Broadest available subset excluding only absent-media tests:
  `.venv/bin/pytest --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py`
  - result after final review fixes:
    `330 passed, 7 skipped, 18 warnings in 454.42s`
- Ruff lint:
  `.venv/bin/ruff check .`
  - result: `All checks passed!`
- Ruff format:
  `.venv/bin/ruff format --check .`
  - result: `65 files already formatted`
- Mypy:
  `.venv/bin/mypy`
  - result: `Success: no issues found in 65 source files`
- Benchmark selector:
  `env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL .venv/bin/python -m chess_gaze.unigaze_batch_benchmark --report artifacts/output/benchmarks/2026-06-26-unigaze-mps-batching.json --print-selected-batch-size`
  - result: printed `7` and exited `0`; emitted the known duplicate FFmpeg
    Objective-C class warning from importing cv2/PyAV.
- Post-review focused regressions:
  `.venv/bin/pytest tests/chess_gaze/test_pipeline_contract.py::test_default_model_batch_inference_failure_returns_usage_error tests/chess_gaze/test_frame_observation.py::test_model_backed_frame_observer_batch_propagates_model_contract_errors -q`
  - result: `2 passed in 1.26s`
- Post-review benchmark regressions:
  `.venv/bin/pytest tests/chess_gaze/test_unigaze_batch_benchmark.py::test_preflight_failure_skips_full_candidate_run tests/chess_gaze/test_unigaze_batch_benchmark.py::test_forward_timing_failure_does_not_skip_full_candidate_run -q`
  - result: `2 passed in 1.15s`
- Post-review benchmark suite after phase-specific error-code cleanup:
  `.venv/bin/pytest tests/chess_gaze/test_unigaze_batch_benchmark.py -q`
  - result: `12 passed in 2.10s`
- Post-review affected suite:
  `.venv/bin/pytest tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q`
  - result: `46 passed in 1.80s`
- Post-review Ruff lint:
  `.venv/bin/ruff check .`
  - result: `All checks passed!`
- Post-review Ruff format:
  `.venv/bin/ruff format --check .`
  - result: `65 files already formatted`
- Post-review mypy:
  `.venv/bin/mypy`
  - result: `Success: no issues found in 65 source files`

## Residual Risk

- The selected batch size is empirical for this Apple M3 Max, this dependency
  set, this model checkpoint, and `artifacts/input/nakamura_1.mp4`. Other Apple
  Silicon devices or future dependency versions should rerun the benchmark.
- The full suite cannot be completely green in the current checkout because
  ignored legacy videos `test_1.mp4` and `test_2.mp4` are absent. The plan's
  required fallback subset passed after excluding only the absent-media tests.
- PyTorch MPS and CPU are not bitwise identical. The accepted contract is
  artifact equivalence under the approved tolerances, plus exact frame identity,
  validity/status/error, and artifact-count matching.
- `unigaze_batch_benchmark.py` is intentionally deep for this finite CLI-only
  benchmark harness. `docs/development/architecture/source-layout.md` records
  the source-layout review and the split trigger if it grows or becomes a
  reusable benchmark framework.
- The native runtime still emits the known duplicate FFmpeg Objective-C class
  warning when cv2 and PyAV are imported in the same process.
