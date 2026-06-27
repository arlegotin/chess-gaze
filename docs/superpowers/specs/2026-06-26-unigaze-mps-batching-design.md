# UniGaze MPS Batching Design

Date: 2026-06-26

## Status

Approved by the user on 2026-06-26. The executable implementation plan is
`docs/superpowers/plans/2026-06-26-unigaze-mps-batching.md`.

This spec depends on the active frame-level pipeline design:
`docs/superpowers/specs/2026-06-24-frame-gaze-analysis-pipeline-design.md`.

Supersession note, 2026-06-27: the default-runtime decision in this spec was
superseded by `2026-06-27-unigaze-mps7-default-design.md`. The current
no-override default is MPS batch size 7. CPU/1 remains the explicit
compatibility and benchmark-baseline profile.

## Goal

Make the existing UniGaze inference path run correctly on Apple Silicon MPS and
support `unigaze_batch_size > 1`, while preserving the current per-frame
analysis contract.

The optimization is transport-level batching for UniGaze only. Frames remain
semantically independent:

- no frame is skipped, sampled, smoothed, interpolated, tracked, averaged, or
  reused as evidence for another frame;
- face, eye, iris, head-pose, geometric gaze, appearance gaze, recommended gaze,
  scene, QA, and visualization calculations keep their current meanings;
- batch order maps exactly back to decoder-emission frame order;
- every decoded frame still writes one raw frame, one processed frame, one
  `records/frames.jsonl` line, and one `records/scene_frames.jsonl` line.

The implementation must benchmark the current CPU batch-1 flow and every
candidate CPU/MPS batch option defined in this spec on
`artifacts/input/nakamura_1.mp4`, then choose the fastest passing MPS
`batch_size > 1` as the recommended optimized profile.

## Non-Goals

- Do not change the selected model checkpoint, model family, crop geometry,
  resize interpolation, channel order, normalization, yaw sign convention,
  scene-ray convention, or recommendation logic.
- Do not introduce temporal MediaPipe tracking, frame reuse, prefetch downloads,
  remote model loading, mixed precision, `torch.compile`, MPS fast math, or CPU
  fallback as part of the accepted path.
- Do not claim MPS and CPU outputs are bitwise identical. They must be
  behaviorally equivalent under the tolerances in this spec.
- Do not weaken any existing tests, schema validation, artifact counts, or
  real-video regressions.

## Verified Current State

Verified locally on 2026-06-26.

| Area | Evidence |
| --- | --- |
| Default device | `src/chess_gaze/pipeline.py:322` hardcodes `UniGazeModel.from_local_asset(gaze_asset, device="cpu")`. |
| Single-row wrapper | `src/chess_gaze/gaze_observation.py:88` accepts a tensor shaped `(batch, 3, H, W)`, but `src/chess_gaze/gaze_observation.py:98` requires `pred_gaze.shape == (1, 2)` and reads only row `0`. |
| Single-frame caller | `src/chess_gaze/frame_observation.py:80` builds one `FrameRecord` at a time. `src/chess_gaze/frame_observation.py:195` normalizes one selected face crop and calls `gaze_model.predict(normalized.tensor)`. |
| Crop tensor | `src/chess_gaze/gaze_observation.py:144` returns a CPU tensor shaped `(1, 3, 224, 224)`. A model on MPS therefore also needs explicit input transfer at the UniGaze boundary. |
| CLI/config | `src/chess_gaze/cli.py:44` exposes `analyze <video_path>`, `--output-root`, `--models-root`, and `--config`. `src/chess_gaze/configuration.py:15` has no inference device or batch fields and forbids unknown config keys. |
| Per-frame invariant | `docs/superpowers/specs/2026-06-24-frame-gaze-analysis-pipeline-design.md:1028` forbids temporal smoothing, tracking, interpolation, or across-frame averaging. |
| Scene invariant | `src/chess_gaze/scene_geometry.py:431` uses `appearance_gaze` as the UniGaze source for scene rays, not `recommended_gaze`. |
| Package versions | Local runtime reports Python on macOS arm64, `torch 2.12.1`, `torchvision 0.27.1`, `unigaze 0.1.3`, `timm 0.3.2`, and `safetensors 0.8.0`. `torch.backends.mps.is_built()` and `torch.backends.mps.is_available()` are both `True` on this machine. |
| Local media | `artifacts/input/nakamura_1.mp4` exists, has sha256 `eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45`, and is documented as 1920x1080 with 1973 decoded frames. |
| Local models | `models/mediapipe/face_landmarker.task` sha256 is `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`; `models/unigaze/unigaze_h14_joint.safetensors` sha256 is `a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f`. |
| Sandbox caveat | README documents that real-model gates and `chess-gaze analyze` may need unsandboxed execution on macOS because MediaPipe native GL/Metal initialization can fail inside the managed sandbox. |

## Dependency And Runtime Evidence Matrix

All external-source claims were verified on 2026-06-26. User-provided candidate:
MPS plus `batch_size > 1` for the existing UniGaze pipeline.

| Candidate | Task fit | Primary sources and verification date | Published metrics or direct evidence | Checkpoint or package availability | License and intended-use constraints | Maintenance status | Runtime and platform fit | Integration cost and reproducibility risk | Known caveats | User provided? | Decision | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Current CPU, `batch_size=1`, UniGaze H14 Joint | Correct baseline and rollback target. Does not meet performance goal. | Local code and lockfile, 2026-06-26. UniGaze repo and model card: https://github.com/ut-vision/UniGaze and https://huggingface.co/UniGaze/UniGaze-models. | Existing repo real-video closeouts show complete Nakamura runs with 1973 records. | Installed `unigaze==0.1.3`; local safetensors checkpoint exists and matches registry sha256. | UniGaze model card and repo registry use MG-NC-RAI style non-commercial/research license approval already recorded by repo owner. | Package is pinned in this repo; upstream has an official repo/model card. | Works everywhere supported by PyTorch CPU, but slow for H14. | Low risk, already implemented. | Current wrapper only accepts `(1, 2)` output and calls per frame. | No, this is existing flow. | Keep as default compatibility baseline and benchmark baseline. | High. |
| PyTorch MPS with explicit `device="mps"` and UniGaze batching | Directly satisfies task: Apple M3 Max acceleration and `batch_size > 1` while keeping the same model/checkpoint. | PyTorch MPS docs: https://docs.pytorch.org/docs/2.12/notes/mps.html. MPS sync docs: https://docs.pytorch.org/docs/2.12/generated/torch.mps.synchronize.html. Numerical accuracy docs: https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html. Verified 2026-06-26. | Local runtime reports MPS built and available. UniGaze installed model code returns `pred_gaze = self.gaze_fc(features)`, so output naturally preserves batch dimension `(N, 2)`. | Existing PyTorch 2.12.1 and UniGaze 0.1.3 package are already installed by `uv.lock`; no new dependency or checkpoint. | Same local model license policy as current flow. | PyTorch MPS is official; dependency is already locked. | Best fit for Apple Silicon. Requires model and inputs on same MPS device. | Medium risk: device transfer, async timing, memory pressure, and CPU/MPS numerical differences must be tested. | Unsupported MPS ops must fail visibly; no hidden CPU fallback. Batched and sliced computations may differ numerically. | Yes. | Select. Implement as explicit runtime option and recommended optimized profile after benchmarks. | Medium-high, pending real benchmark/equivalence gates. |
| PyTorch MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1` | May avoid unsupported-op crashes but weakens the claim that inference is on MPS. | PyTorch MPS environment docs: https://docs.pytorch.org/docs/2.12/mps_environment_variables.html. Verified 2026-06-26. | No repo evidence that UniGaze needs fallback. | No new package. | Same model license. | Official env knob. | Could silently run unsupported ops on CPU. | High reproducibility risk because performance and device claims become ambiguous. | A run could be labeled MPS while part of inference ran on CPU. | No. | Reject for accepted gates. If present during official benchmark, fail the candidate. | High. |
| PyTorch MPS fast math or preferred Metal matmul knobs | Might improve speed but changes numerical contract. | PyTorch MPS env docs: https://docs.pytorch.org/docs/2.12/mps_environment_variables.html. Verified 2026-06-26. | No repo evidence proving no decision changes. | No new package. | Same model license. | Official env knobs. | Apple Silicon only. | Medium integration effort, high validation burden. | Could shift gaze angles near thresholds. | No. | Reject for this task. Must be unset for acceptance benchmarks. | Medium-high. |
| `unigaze.load(..., device=...)` easy path | Official UniGaze loader supports a device parameter, but it downloads/caches from Hugging Face. | UniGaze repo and PyPI: https://github.com/ut-vision/UniGaze and https://pypi.org/project/unigaze/. Verified 2026-06-26. | Package source shows `hf_hub_download()` in `unigaze.loader.load()`. | Installed package is present; remote model path is not allowed during analysis. | Same model license but would bypass this repo's checksum-first local model policy. | Official package path. | Could run on MPS but conflicts with analysis-time local-only rule. | High reproducibility risk because it can reach network/cache outside registry trust. | Violates "analysis does not download models". | No. | Reject. Continue using `build_unigaze_model()` plus verified local safetensors. | High. |
| MediaPipe `VIDEO` mode or temporal tracking to improve speed | Could reduce face-landmark cost, but it changes per-frame independence. | MediaPipe Face Landmarker Python docs: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python. Verified 2026-06-26. | Docs describe distinct IMAGE/VIDEO/LIVE_STREAM modes and tracking behavior. | Existing `.task` asset is present. | Google AI Edge Terms, already accepted by registry. | Official. | Works locally but changes semantics. | High risk to current invariants and face arbitration regressions. | Tracking can hide frame-local failures. | No. | Reject for this task. Keep IMAGE mode. | High. |
| New gaze model or checkpoint | Could be a future accuracy project, but the request is to optimize existing UniGaze inference. | AGENTS.md requires a separate primary-source matrix for model changes. Verified 2026-06-26. | No new model evaluated in this task. | No new checkpoint selected. | Unknown. | Unknown. | Unknown. | High scope expansion and quality risk. | Would confound optimization with model-quality changes. | No. | Out of scope. | High. |

## Design Decision

Use a batch-aware analysis path that batches only UniGaze tensor inference.

Historical decision, superseded on 2026-06-27: the default compatibility
behavior remained `unigaze_device="cpu"` and `unigaze_batch_size=1`, and the
optimized behavior was explicit:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 \
  --models-root models \
  --unigaze-device mps \
  --unigaze-batch-size "$SELECTED_MPS_BATCH_SIZE"
```

After implementation benchmarks complete, the README may recommend the fastest
passing MPS `batch_size > 1` for Apple Silicon. The default should not switch to
MPS in this task because that would make ordinary runs machine-dependent and
would produce small, expected numerical differences from CPU.

`SELECTED_MPS_BATCH_SIZE` is not a placeholder in the implementation. It must be
replaced by the concrete integer chosen from the benchmark report before any
closeout command is recorded.

## Configuration And CLI Contract

Add strict runtime fields to `AnalysisConfig`:

```python
unigaze_device: Literal["cpu", "mps"] = "cpu"
unigaze_batch_size: int = 1
```

Validation requirements:

- `unigaze_batch_size >= 1`;
- config unknown keys remain rejected;
- CLI supports `--unigaze-device {cpu,mps}` and `--unigaze-batch-size INT`;
- CLI values override config-file values;
- explicit `mps` request fails before run directory creation when
  `torch.backends.mps.is_available()` is false;
- explicit `mps` request fails before run directory creation when
  `PYTORCH_ENABLE_MPS_FALLBACK=1` or `PYTORCH_MPS_FAST_MATH=1` is present for
  the accepted analysis path;
- `PYTORCH_MPS_PREFER_METAL` must be recorded in benchmark metadata if present
  and must be unset for official acceptance benchmarks unless a later approved
  spec expands the matrix.

Use `CliErrorCode.USAGE` for invalid runtime configuration unless an existing
more specific code applies. Do not add a new CLI error enum unless tests show
that `USAGE` is ambiguous for callers.

## Runtime Metadata

Every new run must persist enough inference metadata for later agents to know
what produced the values. Add a strict runtime inference section to
`run_manifest.json`, for example:

```json
"inference": {
  "unigaze_model_id": "unigaze-h14-joint",
  "unigaze_device": "mps",
  "unigaze_batch_size": 16,
  "torch_version": "2.12.1",
  "torch_mps_available": true,
  "mps_fallback_env": "unset",
  "mps_fast_math_env": "unset",
  "mps_preflight_passed": true
}
```

This metadata must not include secrets, absolute cache paths, Hugging Face
tokens, or network-derived values. Update schema and tests in the same task that
adds the manifest field.

## UniGaze Model Contract

Keep `UniGazeModel.predict()` as a single-item compatibility wrapper, and add a
batch API:

```python
def predict_batch(self, normalized_batch: torch.Tensor) -> tuple[FaceModelGaze, ...]:
    ...
```

Required behavior:

- accepts a CPU or model-device tensor shaped `(N, 3, H, W)` where `N >= 1`;
- moves the input tensor to the model's resolved `torch.device` inside the
  model wrapper, so callers cannot accidentally send CPU inputs to an MPS model;
- runs in `torch.inference_mode()` or `torch.no_grad()` with `eval()` already
  applied;
- does not use autocast, mixed precision, fast math, or compile;
- requires `pred_gaze` to be a tensor shaped `(N, 2)`;
- maps row `i` to exactly one `FaceModelGaze`;
- preserves the existing pitch/yaw contract:
  `pred_gaze[:, 0]` is pitch radians, `pred_gaze[:, 1]` is UniGaze yaw radians,
  and repo yaw is `-pred_gaze[:, 1]` because frame records use image-right
  positive yaw;
- validates every returned pitch/yaw as finite before building a valid
  `FaceModelGaze`;
- keeps `confidence=None` and
  `confidence_source="not_provided_by_unigaze"`;
- never calls `unigaze.load()` and never downloads model assets.

`predict()` must call `predict_batch()` and require exactly one returned item.
This preserves current tests and callers while making the batch path explicit.

## Pipeline And Observer Contract

Add an optional batch observer surface without removing the existing single-frame
observer surface:

```python
class FrameBatchRecordObserver(Protocol):
    def __call__(self, frames: Sequence[ObserverFrame]) -> Sequence[FrameRecord]: ...

@dataclass(frozen=True)
class ObserverBundle:
    frame_observer: FrameRecordObserver
    frame_batch_observer: FrameBatchRecordObserver | None = None
    close: Callable[[], None] | None = None
```

Pipeline behavior:

- decode frames in existing order;
- group decoded frames into batches of size `unigaze_batch_size`;
- for bundles without `frame_batch_observer`, preserve the current single-frame
  `_process_frame()` path;
- for bundles with `frame_batch_observer`, write raw frames per decoded frame,
  build `ObserverFrame` objects, call the batch observer once, validate
  returned `frame_id` and `frame_index` identity for every row, render processed
  frames in the same order, write errors, and append `frames.jsonl` in the same
  order;
- flush a final partial batch at EOF;
- support `unigaze_batch_size > decoded_frame_count`;
- do not reorder frames by face validity or crop availability.

`ModelBackedFrameObserver` may remain the deep module that owns frame evidence
construction. Refactor it internally so the batch method can reuse the same
per-frame logic as `__call__`:

- per-frame step: observe face, selected face, eyes, head pose, left/right
  geometric gaze, normalized face crop if a selected face exists, and accumulated
  errors;
- batched step: stack only the valid normalized face crop tensors in original
  frame order and call `gaze_model.predict_batch()`;
- record step: pair each batch output back to its originating frame, synthesize
  recommended gaze with the same existing function, and build `FrameRecord`.

Frames with no selected face keep the existing missing-face behavior and do not
enter the UniGaze tensor batch. Frames with eye/head-pose errors may still enter
the UniGaze batch if they have a selected face crop, matching the current
appearance-gaze independence.

If batch model inference raises unexpectedly after preflight, the implementation
must fail the run as a runtime configuration error instead of writing misleading
completed artifacts. Silent device fallback is forbidden. Per-row non-finite
model outputs are different: mark only that originating frame as
`GAZE_MODEL_FAILED` and keep other finite rows valid.

## MPS Preflight

For `unigaze_device="mps"`, preflight must occur before creating a run
directory:

1. verify `torch.backends.mps.is_available()`;
2. verify fallback/fast-math env vars are not enabled for accepted runs;
3. construct the UniGaze model from the verified local checkpoint;
4. move model to MPS;
5. run one dummy tensor with shape
   `(unigaze_batch_size, 3, unigaze_input_size_px, unigaze_input_size_px)`;
6. call `torch.mps.synchronize()` after the dummy run;
7. fail with a stable `USAGE` pipeline error if availability, memory, or
   unsupported-operation checks fail.

The preflight must not download from Hugging Face and must not create
`artifacts/output/.../runs/...` on failure.

## Benchmark Plan

Benchmarks must be run after implementation and before closeout. They must use
`artifacts/input/nakamura_1.mp4` and local model assets under `models/`.

Candidate grid:

```text
devices: cpu, mps
batch_sizes: 1, 2, 4, 7, 8, 16, 32, 64
optional mps extension: 128 only if 64 passes with clear memory headroom
```

This finite grid is the meaning of "all options" for this task. It includes the
current flow (`cpu`, `1`) and a non-dividing batch size (`7`) to force tail-batch
coverage because Nakamura has 1973 frames.

For each device/batch candidate:

- run device preflight and record pass/fail;
- run a pure UniGaze forward benchmark on real normalized face crops collected
  from Nakamura frames;
- run a full `chess-gaze analyze` benchmark for candidates that pass preflight;
- record failures such as unavailable MPS, OOM, unsupported op, schema failure,
  or equivalence failure as candidate results, not as skipped rows.

Benchmark timing rules:

- before the first implementation edit, capture a cold current-flow baseline
  from the current `cpu/1` code, then repeat post-change `cpu/1`;
- perform warmup before timing UniGaze forward passes;
- call `torch.mps.synchronize()` before and after timed MPS regions;
- report wall-clock end-to-end analysis time because the user's target is video
  analysis time;
- also report UniGaze-only time so future agents know whether non-UniGaze stages
  dominate;
- report median, min, max, and at least three repetitions for pure UniGaze
  timing;
- one full end-to-end analysis run per passing candidate is acceptable because
  it writes large artifacts; if repeated full runs are affordable, report median
  over three runs;
- capture peak MPS memory when PyTorch exposes it, and record "unavailable" when
  not available instead of inventing a value.

Benchmark report fields:

```json
{
  "git_revision": "...",
  "source_video": "artifacts/input/nakamura_1.mp4",
  "source_video_sha256": "eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45",
  "decoded_frame_count": 1973,
  "platform": "macOS arm64 Apple M3 Max",
  "torch_version": "2.12.1",
  "unigaze_version": "0.1.3",
  "mps_available": true,
  "mps_fallback_env": "unset",
  "mps_fast_math_env": "unset",
  "model_asset_sha256": "a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f",
  "candidate_results": [
    {
      "device": "mps",
      "batch_size": 16,
      "preflight": "passed",
      "full_run_dir": "artifacts/output/nakamura_1/runs/...",
      "analysis_wall_seconds": 123.456,
      "frames_per_second": 15.977,
      "unigaze_forward_median_seconds": 12.345,
      "equivalence_status": "passed",
      "peak_mps_memory_bytes": "unavailable"
    }
  ],
  "selected_optimized_profile": {
    "device": "mps",
    "batch_size": 16,
    "selection_reason": "fastest passing MPS candidate with batch_size > 1"
  }
}
```

The implementation may store benchmark JSON under ignored
`artifacts/output/benchmarks/` and summarize the final report in the closeout.

## Equivalence Gates

All comparisons use a fresh `cpu/1` post-change run as the canonical
compatibility baseline. Normalize run-specific fields before comparing:
run IDs, creation timestamps, absolute paths, generated viewer paths, byte
counts, and benchmark-only metadata.

Exact requirements for every candidate:

- decoded frame count is 1973;
- frame IDs and frame indexes are identical and ordered from `f000000000` to
  `f000001972`;
- counts of raw frames, processed frames, frame records, error records, scene
  frame records, viewer frames, and QA decoded frames match;
- `qa_summary.final_status == "complete"`;
- `qa_summary.artifact_validation.schema_validation_passed is true`;
- `qa_summary.artifact_validation.counts_match is true`;
- face presence, selected face validity, eye presence, head-pose validity,
  gaze validity booleans, frame status, error codes, and invalid reasons match
  exactly;
- scene/viewer schemas validate with no unknown fields, NaN, or Infinity.

Numeric tolerances:

- CPU batch candidates vs CPU batch-1: UniGaze-derived pitch/yaw max absolute
  delta must be `<= 1e-6` radians; scene ray unit-vector component delta must be
  `<= 1e-6`; monitor hit U/V delta must be `<= 1e-6` pseudo-meters.
- MPS candidates vs CPU batch-1: UniGaze-derived pitch/yaw max absolute delta
  must be `<= 1e-3` radians; scene ray unit-vector component delta must be
  `<= 1e-3`; monitor hit U/V delta must be `<= 2e-3` pseudo-meters.
- If a tolerance passes but changes any validity boolean, invalid reason, frame
  status, or warning/error classification, the candidate fails.

Processed JPEG bytes may differ for MPS candidates because gaze overlays can
shift slightly within the allowed numeric tolerance. The count and schema
contracts are binding; bitwise processed-image equality is not.

## Test Requirements

Use test-first development for every behavior change.

Required unit tests:

- config rejects `unigaze_batch_size < 1` and unsupported devices;
- CLI accepts `--unigaze-device` and `--unigaze-batch-size` and passes them into
  `AnalyzeRequest`;
- explicit MPS unavailable is a stable pre-run failure using monkeypatching;
- fallback/fast-math env vars are rejected for explicit MPS;
- `UniGazeModel.predict_batch()` accepts fake output shaped `(N, 2)` and returns
  one `FaceModelGaze` per row with the current yaw sign conversion;
- `UniGazeModel.predict()` still works for one row and rejects multi-row use
  only through its single-item wrapper contract;
- `UniGazeModel.predict_batch()` rejects wrong rank, wrong channel count,
  empty batch, missing `pred_gaze`, and wrong output row count;
- input tensors are moved to the model device at the wrapper boundary;
- `ModelBackedFrameObserver` batch mode maps distinct model rows back to the
  correct frame IDs;
- mixed valid and missing-face frames preserve current missing-face records and
  batch only valid face crops;
- final partial batches flush at EOF;
- `unigaze_batch_size > decoded_frame_count` still writes one record per frame.

Required integration tests:

- tiny synthetic video with fake batch observer, `batch_size=2`, and 5 frames
  writes 5 raw frames, 5 processed frames, 5 frame records, and ordered yaw
  values from the fake batch model;
- existing fake-observer pipeline tests pass without providing a batch observer;
- existing scene geometry and scene artifact tests pass unchanged in meaning;
- real Nakamura face arbitration regression still passes:
  `tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_rejects_nakamura_overexpanded_faces`;
- real model Nakamura analyze must be run with the selected optimized MPS
  profile and complete successfully.

Required local gates:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_rejects_nakamura_overexpanded_faces -q
SELECTED_MPS_BATCH_SIZE=<integer from benchmark report>
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models --unigaze-device mps --unigaze-batch-size "$SELECTED_MPS_BATCH_SIZE"
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

If full `pytest` fails only because ignored legacy media such as
`artifacts/input/test_1.mp4` or `artifacts/input/test_2.mp4` is absent, the
closeout must record the exact missing-file failures and rerun the broadest
meaningful subset available locally. Do not mark missing media as passing.

## Documentation And Closeout Requirements

Update these docs after implementation:

- `README.md`: document `--unigaze-device`, `--unigaze-batch-size`, explicit MPS
  caveats, local-only model loading, and the selected Apple Silicon recommended
  profile.
- `docs/development/architecture/source-layout.md`: update package ownership if
  new benchmark/runtime modules are added.
- `docs/superpowers/closeouts/2026-06-26-unigaze-mps-batching.md`: record root
  cause, files changed, benchmark matrix, selected batch size, exact run
  directories, verification commands, output snippets, failures, and residual
  uncertainty.

Closeout must include:

- current-flow CPU batch-1 baseline;
- post-change CPU batch-1 equivalence;
- all candidate benchmark rows from this spec;
- selected MPS `batch_size > 1`;
- Nakamura optimized run directory and viewer path;
- QA summary counts and schema status for the optimized run;
- max observed CPU/MPS numeric deltas;
- any candidates rejected for OOM, MPS unsupported ops, equivalence drift, or no
  speed improvement;
- whether any command required unsandboxed execution and why.

## Approval Checklist

Implementation may begin only after this spec is approved.

The approved implementation is complete only when:

- `device=mps` works on this Apple M3 Max without CPU fallback;
- at least one MPS `batch_size > 1` candidate completes the full Nakamura run;
- the fastest passing MPS `batch_size > 1` is selected and documented;
- the optimized run preserves the current per-frame artifact and calculation
  contracts under the exact/tolerance gates above;
- no existing logic, calculation, schema validation, or real-video regression is
  weakened;
- focused tests, real Nakamura verification, broad local gates, benchmark
  report, and closeout evidence are all fresh.

## Remaining Uncertainty

The optimal batch size is intentionally unknown before implementation. It must
be determined empirically on this local Apple M3 Max using the benchmark grid.
MPS numerical drift is also unknown until the real UniGaze H14 checkpoint is run
through the candidate matrix. These uncertainties block implementation defaults,
not the spec: the spec defines how to measure and reject unsafe candidates.
