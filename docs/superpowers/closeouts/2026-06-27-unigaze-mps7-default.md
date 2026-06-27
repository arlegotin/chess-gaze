# UniGaze MPS/7 Default Closeout

Date: 2026-06-27

## Summary

The no-override UniGaze runtime now defaults to `unigaze_device="mps"` and
`unigaze_batch_size=7` through the canonical `AnalysisConfig`. CLI parser
defaults remain `None`, so bare CLI runs inherit the config default while
explicit CPU/1 compatibility remains available through config or CLI overrides.

This change does not alter model checkpoint selection, crop geometry,
normalization, frame independence, batch transport semantics, benchmark
candidate logic, CPU/1 baseline semantics, or equivalence tolerances.

## Implementation

- Changed `AnalysisConfig` defaults from CPU/1 to MPS/7.
- Added config and pipeline regressions proving no-override model-backed runs
  resolve to MPS/7.
- Added default preflight regressions proving default MPS fails before run
  layout when MPS is unavailable or when `PYTORCH_ENABLE_MPS_FALLBACK`,
  `PYTORCH_MPS_FAST_MATH`, or `PYTORCH_MPS_PREFER_METAL` is enabled.
- Kept external-observer manifests on `external_observer` /
  `not_applicable` metadata.
- Kept CLI runtime flags unset by default so config resolution remains the
  single source of runtime defaults.
- Updated README current guidance and marked the old CPU/1 default decision in
  historical docs as superseded.

## Verification

Focused RED before production change:

- `uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_pipeline_contract.py::test_default_mps_unavailable_fails_before_run_layout tests/chess_gaze/test_pipeline_contract.py::test_default_mps_rejects_unsafe_env_before_run_layout tests/chess_gaze/test_pipeline_contract.py::test_default_model_observer_manifest_records_unigaze_runtime tests/chess_gaze/test_pipeline_contract.py::test_explicit_cpu_batch_one_override_reaches_default_model_runtime tests/chess_gaze/test_pipeline_contract.py::test_config_models_root_controls_default_model_observer_factory -q`
- Result before the default change: failed on old CPU/1 behavior as expected.

Focused and broad gates after implementation:

- `uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_pipeline_contract.py -q`
  - `41 passed`
- `uv run pytest 'tests/chess_gaze/test_unigaze_runtime.py::test_prepare_unigaze_runtime_mps_rejects_unsafe_env_before_model_load[PYTORCH_MPS_PREFER_METAL]' 'tests/chess_gaze/test_pipeline_contract.py::test_default_mps_rejects_unsafe_env_before_run_layout[PYTORCH_MPS_PREFER_METAL]' -q`
  - `2 passed` after confirming these tests failed before the runtime fix
- `uv run pytest tests/chess_gaze/test_unigaze_runtime.py tests/chess_gaze/test_pipeline_contract.py -q`
  - `32 passed`
- `uv run pytest tests/chess_gaze/test_unigaze_runtime.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q`
  - `62 passed` before the final `PYTORCH_MPS_PREFER_METAL` review fix
- `uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q`
  - `74 passed`
- `uv run ruff format --check .`
  - `65 files already formatted`
- `uv run ruff check .`
  - `All checks passed!`
- `uv run mypy`
  - `Success: no issues found in 65 source files`
- `uv run pytest -q`
  - `333 passed, 7 failed, 7 skipped`
  - The seven failures are all absent legacy mandatory media:
    `artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4`.
- `uv run pytest -q --ignore=tests/chess_gaze/test_pipeline_real_video_contract.py --ignore=tests/chess_gaze/test_qa_summary_real_video_contract.py --ignore=tests/chess_gaze/test_video_decode_real_video.py --ignore=tests/chess_gaze/test_visualization_real_video.py`
  - `337 passed, 7 skipped`
  - This was run after the final `PYTORCH_MPS_PREFER_METAL` review fix and
    excludes only the known absent legacy-video failure modules.

Real no-override smoke on the required Nakamura clip:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK \
  -u PYTORCH_MPS_FAST_MATH \
  -u PYTORCH_MPS_PREFER_METAL \
  .venv/bin/chess-gaze analyze artifacts/input/nakamura_1.mp4 \
  --output-root artifacts/output/default-mps7-smoke \
  --models-root models
```

Result:

- exit code `0`
- run:
  `artifacts/output/default-mps7-smoke/nakamura_1/runs/20260627T184859Z-280eb656`
- `run_manifest.json` records:
  - `unigaze_device`: `mps`
  - `unigaze_batch_size`: `7`
  - `torch_mps_available`: `true`
  - `mps_preflight_passed`: `true`
  - MPS env states: `unset`
- artifact counts:
  - `records/frames.jsonl`: `1973`
  - `records/scene_frames.jsonl`: `1973`
  - raw frames: `1973`
  - processed frames: `1973`
  - `qa_summary.json` final status: `complete`

## Residual Risk

Bare model-backed analysis now requires Apple MPS availability. Non-MPS
machines must explicitly use `--unigaze-device cpu --unigaze-batch-size 1` or
equivalent config fields.

The full local test suite still cannot be fully green until the ignored legacy
videos `artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4` are
restored.
