# UniGaze MPS/7 Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MPS batch 7 the default UniGaze runtime everywhere config defaults are resolved, while preserving explicit CPU/1 compatibility and all batching invariants.

**Architecture:** Change only the canonical `AnalysisConfig` defaults and let existing CLI/config precedence propagate the new runtime. Add focused regression tests for default MPS/7 resolution, default preflight failure behavior, and explicit CPU/1 override preservation. Update current docs so user-facing guidance no longer says CPU/1 is the no-override path.

**Tech Stack:** Python 3.12, uv, pytest, Pydantic v2, existing PyTorch MPS UniGaze runtime.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Use installed Superpowers skills for implementation flow.
- Approved spec: `docs/superpowers/specs/2026-06-27-unigaze-mps7-default-design.md`.
- Canonical no-override defaults are `unigaze_device="mps"` and `unigaze_batch_size=7`.
- CLI parser defaults for `--unigaze-device` and `--unigaze-batch-size` remain `None`; the resolved config owns defaults.
- Explicit CPU/1 compatibility must remain available through config files and CLI overrides.
- Default model-backed MPS runs must fail before run directory creation when MPS is unavailable or unsafe MPS env vars are enabled.
- External-observer manifests continue to record `external_observer` and `not_applicable` UniGaze fields.
- Do not change model checkpoint, inference math, frame batching semantics, benchmark candidate grid, benchmark CPU/1 baseline semantics, or equivalence tolerances.

---

## File Structure

- Modify `src/chess_gaze/configuration.py` for canonical MPS/7 defaults.
- Modify `tests/chess_gaze/test_configuration.py` for config default and explicit CPU override coverage.
- Modify `tests/chess_gaze/test_pipeline_contract.py` for default model-backed MPS preflight coverage and default manifest expectations.
- Modify `README.md` for current run guidance.
- Modify `docs/superpowers/closeouts/2026-06-26-unigaze-mps-batching.md` only to add a supersession note for the old default statement.

---

### Task 1: Default Runtime Contract

**Files:**
- Modify: `tests/chess_gaze/test_configuration.py`
- Modify: `tests/chess_gaze/test_pipeline_contract.py`
- Modify: `src/chess_gaze/configuration.py`
- Modify: `README.md`
- Modify: `docs/superpowers/closeouts/2026-06-26-unigaze-mps-batching.md`

**Interfaces:**
- Consumes: `AnalysisConfig`, `load_config(None)`, `apply_analysis_overrides()`, `AnalyzeRequest`, `prepare_unigaze_runtime()`.
- Produces: no-override resolved runtime `("mps", 7)`; explicit CPU/1 override remains valid.

- [ ] **Step 1: Write failing configuration tests**

Update `tests/chess_gaze/test_configuration.py` so the runtime default test expects MPS/7 and add explicit CPU/1 override preservation:

```python
def test_load_config_uses_unigaze_runtime_defaults() -> None:
    config = load_config(None)

    assert config.unigaze_device == "mps"
    assert config.unigaze_batch_size == 7


def test_load_config_accepts_explicit_cpu_batch_one_runtime(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"unigaze_device": "cpu", "unigaze_batch_size": 1}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.unigaze_device == "cpu"
    assert config.unigaze_batch_size == 1
```

- [ ] **Step 2: Run configuration tests and verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_configuration.py -q
```

Expected before production change: failure showing `config.unigaze_device` is
still `"cpu"` and/or `config.unigaze_batch_size` is still `1`.

- [ ] **Step 3: Write failing pipeline default tests**

In `tests/chess_gaze/test_pipeline_contract.py`:

1. Change `test_explicit_mps_unavailable_fails_before_run_layout` into a
   default-path test by removing explicit `unigaze_device` and
   `unigaze_batch_size` from `AnalyzeRequest`.
2. Change `test_explicit_mps_rejects_unsafe_env_before_run_layout` into a
   default-path test the same way.
3. In `test_default_model_observer_manifest_records_unigaze_runtime`, call
   `AnalyzeRequest` without UniGaze overrides and assert the manifest records
   `"unigaze_device": "mps"` and `"unigaze_batch_size": 7`.
4. Add a separate explicit CPU/1 override assertion that `prepare_unigaze_runtime`
   receives `device="cpu"` and `batch_size=1`.

Use this helper shape for the explicit CPU/1 assertion:

```python
captured_runtime_requests: list[tuple[str, int]] = []

def fake_prepare_unigaze_runtime(
    asset: object, *, device: str, batch_size: int, input_size_px: int
) -> object:
    del asset, input_size_px
    captured_runtime_requests.append((device, batch_size))
    return SimpleNamespace(
        model=object(),
        inference=InferenceRuntimeRecord(
            observer_source="default_model_observer",
            unigaze_model_id="unigaze-h14-joint",
            unigaze_device=cast(Any, device),
            unigaze_batch_size=batch_size,
            torch_version="test-torch",
            torch_mps_available=True,
            mps_fallback_env="unset",
            mps_fast_math_env="unset",
            mps_prefer_metal_env="unset",
            mps_preflight_passed=None,
        ),
    )
```

- [ ] **Step 4: Run pipeline default tests and verify RED**

Run:

```sh
uv run pytest \
  tests/chess_gaze/test_pipeline_contract.py::test_default_mps_unavailable_fails_before_run_layout \
  tests/chess_gaze/test_pipeline_contract.py::test_default_mps_rejects_unsafe_env_before_run_layout \
  tests/chess_gaze/test_pipeline_contract.py::test_default_model_observer_manifest_records_unigaze_runtime \
  tests/chess_gaze/test_pipeline_contract.py::test_explicit_cpu_batch_one_override_reaches_default_model_runtime \
  -q
```

Expected before production change: at least the default MPS failure tests fail
because the default runtime is still CPU/1.

- [ ] **Step 5: Change canonical defaults**

In `src/chess_gaze/configuration.py`, change only these fields:

```python
unigaze_device: Literal["cpu", "mps"] = "mps"
unigaze_batch_size: int = 7
```

Do not change CLI parser defaults or pipeline override precedence.

- [ ] **Step 6: Update current docs**

In `README.md`, replace the old default paragraph with:

```markdown
By default, UniGaze runs on Apple Silicon MPS with batch size 7:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 \
  --output-root artifacts/output \
  --models-root models
```

For accepted MPS runs, leave `PYTORCH_ENABLE_MPS_FALLBACK`,
`PYTORCH_MPS_FAST_MATH`, and `PYTORCH_MPS_PREFER_METAL` unset. The MPS path
preflights the verified local UniGaze checkpoint on batch shape
`(7, 3, 224, 224)` before creating a run directory, then records runtime
metadata in `run_manifest.json`.

For CPU compatibility or non-MPS machines, opt in explicitly:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4 \
  --unigaze-device cpu \
  --unigaze-batch-size 1
```
```

In `docs/superpowers/closeouts/2026-06-26-unigaze-mps-batching.md`, add a short
note after the summary that the CPU/1 default statement was superseded on
2026-06-27 by the approved MPS/7 default spec.

- [ ] **Step 7: Verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_pipeline_contract.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Run broader gates**

Run:

```sh
uv run pytest tests/chess_gaze/test_unigaze_runtime.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: all commands pass. If full `uv run pytest` is run, the known absent
legacy videos `artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4` may
still cause unrelated real-media failures; report exact failures instead of
hiding them.
