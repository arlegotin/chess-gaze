# UniGaze MPS/7 Default Design

Date: 2026-06-27

## Status

Approved by the user on 2026-06-27.

This spec supersedes only the default-runtime decision in
`docs/superpowers/specs/2026-06-26-unigaze-mps-batching-design.md`. The prior
benchmark matrix, equivalence tolerances, frame-independence contract, model
selection, and runtime-safety requirements remain binding.

The executable plan is
`docs/superpowers/plans/2026-06-27-unigaze-mps7-default.md`.

## Goal

Make the no-override UniGaze runtime default `unigaze_device="mps"` and
`unigaze_batch_size=7` for runs, tests, and config resolution on the Apple M3
Max development target.

## Requirements

- `AnalysisConfig()` and `load_config(None)` must resolve to MPS batch 7.
- Bare CLI analysis must inherit MPS batch 7 through config resolution.
- CLI and config overrides must continue to support explicit CPU/1 runs.
- Default model-backed runs must still preflight MPS before run directory
  creation and fail with `USAGE` if MPS is unavailable or unsafe MPS env knobs
  are enabled.
- External-observer runs must keep `external_observer` inference metadata with
  `not_applicable` UniGaze fields, even when the resolved config default is
  MPS/7.
- Existing frame independence, artifact counts, per-frame calculations, model
  checkpoint, crop geometry, normalization, yaw convention, scene convention,
  and benchmark/equivalence logic must not change.

## Design

Move the default in exactly one canonical place:

```python
class AnalysisConfig(BaseModel):
    unigaze_device: Literal["cpu", "mps"] = "mps"
    unigaze_batch_size: int = 7
```

Keep CLI parser defaults as `None`. This preserves the existing precedence
model: config defaults first, config-file values next, CLI overrides last.

Do not change the benchmark module's CPU/1 baseline semantics. CPU/1 remains the
comparison and rollback profile, but it is no longer the no-override runtime.

## Documentation Impact

Update current user-facing documentation to say bare analysis uses MPS/7 and
that CPU/1 requires explicit override:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4 \
  --unigaze-device cpu \
  --unigaze-batch-size 1
```

Historical closeouts may keep their original benchmark evidence, but any
current guidance that says the default remains CPU/1 must be amended or clearly
marked as superseded by this spec.

## Risks

- Bare model-backed runs now require Apple MPS availability. This is accepted by
  the user for this repository target.
- Non-MPS environments must pass explicit CPU/1 overrides for model-backed
  analysis.
- Tests that instantiate default model-backed analysis must mock or satisfy MPS
  runtime preflight.
