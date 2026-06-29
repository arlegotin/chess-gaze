# UniGaze MPS Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make UniGaze inference work on Apple Silicon MPS with `unigaze_batch_size > 1`, benchmark the approved CPU/MPS grid on `artifacts/input/nakamura_1.mp4`, and preserve all existing per-frame calculations and artifact contracts.

**Architecture:** Keep the current frame evidence model intact and add batching only at the UniGaze tensor-inference boundary. Add strict runtime config, explicit MPS preflight before run creation, batch-aware observer and pipeline surfaces, a first-class run-equivalence harness, and a benchmark module that selects the fastest passing MPS batch size. The original historical plan kept default behavior as `unigaze_device="cpu"` and `unigaze_batch_size=1`.

> Supersession note, 2026-06-27: the default-runtime decision in this historical
> plan was superseded by
> `docs/superpowers/specs/2026-06-27-unigaze-mps7-default-design.md`. The
> current no-override default is MPS batch size 7; CPU/1 remains an explicit
> compatibility and benchmark-baseline profile.

> Supersession note, 2026-06-29: this historical plan's raw/processed frame
> image retention assumptions are superseded by
> `docs/superpowers/specs/2026-06-29-frame-image-retention-design.md` and
> ADR-0004. Current default runs do not retain raw or processed frame image
> files unless `--save-frames` or `save_frame_images=True` is used.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, mypy, PyTorch 2.12.1 MPS, UniGaze 0.1.3, MediaPipe Face Landmarker IMAGE mode, NumPy, Pydantic v2, existing PyAV/OpenCV/Pillow pipeline.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Use installed Superpowers skills for implementation flow; this plan is written for subagent-driven development.
- Approved spec: `docs/superpowers/specs/2026-06-26-unigaze-mps-batching-design.md`.
- Capture the cold current `cpu/1` Nakamura baseline before the first implementation edit.
- Historical default constraint, superseded on 2026-06-27: defaults stayed
  `unigaze_device="cpu"` and `unigaze_batch_size=1`.
- Optimized MPS behavior must be explicit through config or CLI.
- Do not change the selected model checkpoint, model family, crop geometry, resize interpolation, channel order, normalization, yaw sign convention, scene-ray convention, or recommendation logic.
- Do not introduce temporal MediaPipe tracking, frame reuse, prefetch downloads, remote model loading, mixed precision, `torch.compile`, MPS fast math, or CPU fallback as part of the accepted path.
- No frame may be skipped, sampled, smoothed, interpolated, tracked, averaged, or reused as evidence for another frame.
- Batch order must map exactly back to decoder-emission frame order.
- Every decoded frame still writes one raw frame, one processed frame, one `records/frames.jsonl` line, and one `records/scene_frames.jsonl` line.
- Explicit `mps` request fails before run directory creation when `torch.backends.mps.is_available()` is false.
- Explicit `mps` request fails before run directory creation when `PYTORCH_ENABLE_MPS_FALLBACK=1` or `PYTORCH_MPS_FAST_MATH=1` is present.
- `PYTORCH_MPS_PREFER_METAL` must be unset for official acceptance benchmarks unless a later approved spec expands the matrix.
- MPS preflight must construct UniGaze from the verified local checkpoint, move it to MPS, run a dummy tensor shaped `(unigaze_batch_size, 3, 224, 224)`, synchronize, and fail with `USAGE` if unavailable, OOM, or unsupported.
- The analysis path must never call `unigaze.load()` and must never download model assets.
- CPU batch candidates compare to CPU batch-1 with UniGaze-derived numeric tolerance `<= 1e-6`; MPS candidates compare to CPU batch-1 with pitch/yaw and scene-ray component tolerance `<= 1e-3`, monitor U/V tolerance `<= 2e-3 m`, and exact validity/status/error matching.
- Use `artifacts/input/nakamura_1.mp4` for real verification and benchmark selection.
- Local verified `nakamura_1.mp4` sha256 is `eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45`; expected decoded frame count is `1973`.
- Local verified UniGaze checkpoint sha256 is `a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f`.
- Local verified MediaPipe task sha256 is `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
- Real-model gates and `chess-gaze analyze` may need unsandboxed execution on this machine because README documents MediaPipe native macOS GL/Metal failures in the managed sandbox.

---

## File Structure

- Modify `docs/superpowers/specs/2026-06-26-unigaze-mps-batching-design.md` only for approval-status bookkeeping already done before this plan.
- Create `docs/superpowers/plans/2026-06-26-unigaze-mps-batching.md` for this implementation plan.
- Create `src/chess_gaze/unigaze_runtime.py` for UniGaze runtime configuration validation, MPS env checks, MPS preflight, runtime metadata assembly, and device synchronization helpers.
- Create `src/chess_gaze/run_equivalence.py` for strict CPU/MPS artifact comparison.
- Create `src/chess_gaze/unigaze_batch_benchmark.py` for the benchmark CLI and report schema.
- Modify `src/chess_gaze/configuration.py` for strict `unigaze_device` and `unigaze_batch_size` config fields.
- Modify `src/chess_gaze/cli.py` for CLI flags and CLI-over-config request fields.
- Modify `src/chess_gaze/frame_records.py` for strict inference runtime metadata in `RunManifest`.
- Modify `src/chess_gaze/gaze_observation.py` for batch-aware UniGaze prediction and model-device ownership.
- Modify `src/chess_gaze/frame_observation.py` for batch-aware `ModelBackedFrameObserver`.
- Modify `src/chess_gaze/pipeline.py` for request resolution, pre-run MPS preflight, inference metadata writing, batch transport, default observer wiring, and no-run-dir preflight failures.
- Modify `README.md` for MPS/batch usage, defaults, caveats, and selected profile after benchmark.
- Modify `docs/development/architecture/source-layout.md` to add `unigaze_runtime.py`, `run_equivalence.py`, and `unigaze_batch_benchmark.py` ownership.
- Create `docs/superpowers/closeouts/2026-06-26-unigaze-mps-batching.md` during final verification.
- Modify tests under `tests/chess_gaze/` as named in each task.

---

## Shared Interfaces

These interfaces are binding for later tasks.

`src/chess_gaze/configuration.py`:

```python
from typing import Literal

class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: Path = Path("artifacts/output")
    models_root: Path = Path("models")
    raw_frame_image_format: str = "png"
    processed_frame_image_format: str = "jpg"
    processed_frame_jpeg_quality: int = 95
    unigaze_device: Literal["cpu", "mps"] = "cpu"
    unigaze_batch_size: int = 1

    @field_validator("unigaze_batch_size")
    @classmethod
    def validate_unigaze_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("unigaze_batch_size must be >= 1")
        return value
```

`src/chess_gaze/frame_records.py`:

```python
from typing import Literal

class InferenceRuntimeRecord(StrictSchemaModel):
    schema_version: Literal["inference-runtime-v1"] = "inference-runtime-v1"
    observer_source: Literal["default_model_observer", "external_observer"]
    unigaze_model_id: str | None
    unigaze_device: Literal["cpu", "mps", "not_applicable"]
    unigaze_batch_size: int | None
    torch_version: str | None
    torch_mps_available: bool | None
    mps_fallback_env: str
    mps_fast_math_env: str
    mps_prefer_metal_env: str
    mps_preflight_passed: bool | None

class RunManifest(StrictSchemaModel):
    run_id: str
    created_at_utc: str
    input_path: str
    video: VideoManifest
    inference: InferenceRuntimeRecord
```

`src/chess_gaze/unigaze_runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from chess_gaze.frame_records import InferenceRuntimeRecord
from chess_gaze.gaze_observation import UNIGAZE_MODEL_ID, UniGazeModel
from chess_gaze.model_assets import ResolvedModelAsset

UniGazeDevice = Literal["cpu", "mps"]

@dataclass(frozen=True)
class PreparedUniGazeRuntime:
    model: UniGazeModel
    inference: InferenceRuntimeRecord

def prepare_unigaze_runtime(
    asset: ResolvedModelAsset,
    *,
    device: UniGazeDevice,
    batch_size: int,
    input_size_px: int,
) -> PreparedUniGazeRuntime

def external_observer_inference_record() -> InferenceRuntimeRecord

def synchronize_if_needed(device: str) -> None
```

Task 6 implements the bodies for these signatures.

`src/chess_gaze/gaze_observation.py`:

```python
class UniGazeModel:
    @classmethod
    def from_local_asset(
        cls, asset: ResolvedModelAsset, *, device: str
    ) -> UniGazeModel

    @property
    def device(self) -> torch.device

    def predict_batch(
        self, normalized_batch: torch.Tensor
    ) -> tuple[FaceModelGaze, ...]

    def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze
```

`src/chess_gaze/pipeline.py`:

```python
class FrameBatchRecordObserver(Protocol):
    def __call__(self, frames: Sequence[ObserverFrame]) -> Sequence[FrameRecord]

@dataclass(frozen=True)
class ObserverBundle:
    frame_observer: FrameRecordObserver
    frame_batch_observer: FrameBatchRecordObserver | None = None
    close: Callable[[], None] | None = None

@dataclass(frozen=True)
class AnalyzeRequest:
    video_path: Path
    output_root: Path | None = None
    models_root: Path | None = None
    config_path: Path | None = None
    unigaze_device: str | None = None
    unigaze_batch_size: int | None = None
    model_registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH
    run_suffix: str | None = None
    clock: Clock = utc_now
```

`src/chess_gaze/run_equivalence.py`:

```python
class EquivalenceTolerances(StrictSchemaModel):
    appearance_pitch_yaw_radians: float
    scene_ray_component: float
    monitor_uv_m: float

class EquivalenceReport(StrictSchemaModel):
    schema_version: Literal["run-equivalence-v1"] = "run-equivalence-v1"
    baseline_run_dir: str
    candidate_run_dir: str
    passed: bool
    exact_mismatch_count: int
    numeric_mismatch_count: int
    validation_errors: list[str]
    max_appearance_pitch_yaw_delta_radians: float
    max_scene_ray_component_delta: float
    max_monitor_uv_delta_m: float

def compare_runs(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    *,
    tolerances: EquivalenceTolerances,
) -> EquivalenceReport
```

`src/chess_gaze/unigaze_batch_benchmark.py`:

```python
class BenchmarkCandidateResult(StrictSchemaModel):
    device: Literal["cpu", "mps"]
    batch_size: int
    status: Literal[
        "passed",
        "preflight_failed",
        "analyze_failed",
        "equivalence_failed",
        "oom",
        "unsupported_op",
    ]
    preflight_seconds: float | None
    analysis_wall_seconds: float | None
    frames_per_second: float | None
    unigaze_forward_repetitions_seconds: list[float]
    unigaze_forward_median_seconds: float | None
    full_run_dir: str | None
    qa_final_status: str | None
    qa_decoded_frames: int | None
    qa_counts_match: bool | None
    qa_schema_validation_passed: bool | None
    equivalence_report_path: str | None
    max_appearance_pitch_yaw_delta_radians: float | None
    max_scene_ray_component_delta: float | None
    max_monitor_uv_delta_m: float | None
    peak_mps_memory_bytes: int | None
    error_code: str | None
    error_message: str | None

class UniGazeBatchBenchmarkReport(StrictSchemaModel):
    schema_version: Literal["unigaze-batch-benchmark-v1"] = (
        "unigaze-batch-benchmark-v1"
    )
    git_revision: str
    source_video: str
    source_video_sha256: str
    decoded_frame_count: int
    torch_version: str
    unigaze_version: str
    mps_available: bool
    mps_fallback_env: str
    mps_fast_math_env: str
    mps_prefer_metal_env: str
    model_asset_sha256: str
    baseline_run_dir: str
    candidate_results: list[BenchmarkCandidateResult]
    selected_device: Literal["mps"] | None
    selected_batch_size: int | None
    selected_reason: str | None
```

---

### Task 0: Pre-Edit Baseline Capture

**Files:**
- Read: `docs/superpowers/specs/2026-06-26-unigaze-mps-batching-design.md`
- Read: `README.md`
- Read: `src/chess_gaze/model_registry.json`
- Write ignored artifact: `artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.txt`
- Write ignored artifact: `artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.json`

**Interfaces:**
- Consumes: existing unmodified `chess-gaze analyze` CPU batch-1 behavior.
- Produces: baseline run directory, wall-clock timing, QA counts, checksums, and environment evidence used by Task 9.

- [x] **Step 1: Verify clean implementation state**

Run:

```sh
git status --short
```

Expected: only planning docs may be modified or untracked. No source or test implementation files may be modified before this task finishes.

- [x] **Step 2: Verify required local inputs**

Run:

```sh
shasum -a 256 artifacts/input/nakamura_1.mp4 models/unigaze/unigaze_h14_joint.safetensors models/mediapipe/face_landmarker.task
```

Expected output contains exactly:

```text
eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45  artifacts/input/nakamura_1.mp4
a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f  models/unigaze/unigaze_h14_joint.safetensors
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  models/mediapipe/face_landmarker.task
```

- [x] **Step 3: Run current CPU batch-1 Nakamura baseline**

Run unsandboxed if MediaPipe native initialization fails in the managed sandbox:

```sh
mkdir -p artifacts/output/benchmarks
UV_CACHE_DIR=.uv-cache /usr/bin/time -lp uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models 2>&1 | tee artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.txt
```

Expected: command exits `0`, prints a fresh run directory under `artifacts/output/nakamura_1/runs/`, and prints a second line beginning with `viewer: artifacts/output/nakamura_1/runs/`.

- [x] **Step 4: Extract baseline artifact facts**

Extract the run directory printed in Step 3 and run:

```sh
RUN_DIR=$(awk '/^artifacts\/output\/nakamura_1\/runs\// {print $1; exit}' artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.txt)
test -n "$RUN_DIR"
UV_CACHE_DIR=.uv-cache uv run python - "$RUN_DIR" <<'PY'
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import unigaze

from chess_gaze.model_assets import sha256_file
from chess_gaze.qa_summary import QASummary
from chess_gaze.scene_records import SceneSummary

run_dir = Path(sys.argv[1])
qa = QASummary.model_validate_json((run_dir / "qa_summary.json").read_text())
scene = SceneSummary.model_validate_json(
    (run_dir / "scene" / "scene_summary.json").read_text()
)
payload = {
    "schema_version": "current-flow-baseline-v1",
    "git_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "run_dir": str(run_dir),
    "viewer_index_path": str(run_dir / "viewer" / "index.html"),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "torch_version": torch.__version__,
    "unigaze_version": unigaze.__version__,
    "mps_available": torch.backends.mps.is_available(),
    "source_video_sha256": sha256_file(Path("artifacts/input/nakamura_1.mp4")),
    "unigaze_sha256": sha256_file(
        Path("models/unigaze/unigaze_h14_joint.safetensors")
    ),
    "mediapipe_sha256": sha256_file(Path("models/mediapipe/face_landmarker.task")),
    "qa_final_status": qa.final_status,
    "qa_decoded_frames": qa.counts.decoded_frames,
    "qa_frame_records": qa.counts.frame_records,
    "qa_scene_frame_records": qa.counts.scene_frame_records,
    "qa_raw_frames": qa.counts.raw_frames,
    "qa_processed_frames": qa.counts.processed_frames,
    "qa_counts_match": qa.artifact_validation.counts_match,
    "qa_schema_validation_passed": qa.artifact_validation.schema_validation_passed,
    "scene_valid_unigaze_ray_frames": scene.valid_unigaze_ray_frames,
    "scene_valid_monitor_hit_frames": scene.valid_monitor_hit_frames,
}
Path("artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
```

Expected: JSON reports `qa_final_status: "complete"`, `qa_decoded_frames: 1973`, `qa_counts_match: true`, and `qa_schema_validation_passed: true`.

- [x] **Step 5: Do not commit ignored baseline artifacts**

Run:

```sh
git status --short
```

Expected: `artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.txt` and `artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.json` do not appear because `artifacts/output/` is ignored. Planning docs may still appear.

---

### Task 1: Runtime Config And CLI Contract

**Files:**
- Modify: `src/chess_gaze/configuration.py`
- Modify: `src/chess_gaze/cli.py`
- Modify: `src/chess_gaze/pipeline.py`
- Test: `tests/chess_gaze/test_configuration.py`
- Test: `tests/chess_gaze/test_cli.py`

**Interfaces:**
- Consumes: existing `AnalysisConfig`, `AnalyzeRequest`, and CLI parser.
- Produces: `AnalyzeRequest.unigaze_device: str | None`, `AnalyzeRequest.unigaze_batch_size: int | None`, resolved runtime values in `_ResolvedRequest`, and CLI-over-config behavior used by later tasks.

- [x] **Step 1: Write failing config tests**

Add to `tests/chess_gaze/test_configuration.py`:

```python
def test_load_config_uses_unigaze_runtime_defaults() -> None:
    config = load_config(None)

    assert config.unigaze_device == "cpu"
    assert config.unigaze_batch_size == 1


def test_load_config_accepts_unigaze_runtime_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"unigaze_device": "mps", "unigaze_batch_size": 7}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.unigaze_device == "mps"
    assert config.unigaze_batch_size == 7


@pytest.mark.parametrize("batch_size", [0, -1])
def test_load_config_rejects_invalid_unigaze_batch_size(
    tmp_path: Path, batch_size: int
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        f'{{"unigaze_batch_size": {batch_size}}}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unigaze_batch_size") as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_INVALID"


def test_load_config_rejects_unsupported_unigaze_device(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"unigaze_device": "cuda"}', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unigaze_device") as exc_info:
        load_config(config_path)

    assert exc_info.value.code == "CONFIG_LOAD_INVALID"
```

- [x] **Step 2: Write failing CLI tests**

Add to `tests/chess_gaze/test_cli.py`:

```python
def test_analyze_passes_unigaze_cli_overrides(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    viewer_index_path = run_dir / "viewer" / "index.html"
    make_tiny_video(video_path)
    captured_requests: list[object] = []

    def fake_analyze_video(request: object) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=viewer_index_path,
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    exit_code = main(
        [
            "analyze",
            str(video_path),
            "--unigaze-device",
            "mps",
            "--unigaze-batch-size",
            "7",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().err == ""
    [request] = captured_requests
    assert request.unigaze_device == "mps"
    assert request.unigaze_batch_size == 7
```

- [x] **Step 3: Run tests to verify they fail**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_configuration.py::test_load_config_uses_unigaze_runtime_defaults tests/chess_gaze/test_cli.py::test_analyze_passes_unigaze_cli_overrides -q
```

Expected: FAIL because the config fields and CLI flags do not exist.

- [x] **Step 4: Implement config fields**

Modify `src/chess_gaze/configuration.py`:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
```

Update `AnalysisConfig` with the shared-interface fields and validator.

- [x] **Step 5: Implement request fields and resolution**

Modify `src/chess_gaze/pipeline.py`:

```python
@dataclass(frozen=True)
class AnalyzeRequest:
    video_path: Path
    output_root: Path | None = None
    models_root: Path | None = None
    config_path: Path | None = None
    unigaze_device: str | None = None
    unigaze_batch_size: int | None = None
    model_registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH
    run_suffix: str | None = None
    clock: Clock = utc_now
```

Update `_ResolvedRequest`:

```python
@dataclass(frozen=True)
class _ResolvedRequest:
    video_path: Path
    output_root: Path
    models_root: Path
    raw_frame_image_format: str
    processed_frame_image_format: str
    processed_frame_jpeg_quality: int
    unigaze_device: str
    unigaze_batch_size: int
```

Update `_resolve_request()` so CLI values override config only when not `None`:

```python
return _ResolvedRequest(
    video_path=request.video_path,
    output_root=request.output_root or config.output_root,
    models_root=request.models_root or config.models_root,
    raw_frame_image_format=config.raw_frame_image_format,
    processed_frame_image_format=config.processed_frame_image_format,
    processed_frame_jpeg_quality=config.processed_frame_jpeg_quality,
    unigaze_device=(
        request.unigaze_device
        if request.unigaze_device is not None
        else config.unigaze_device
    ),
    unigaze_batch_size=(
        request.unigaze_batch_size
        if request.unigaze_batch_size is not None
        else config.unigaze_batch_size
    ),
)
```

- [x] **Step 6: Implement CLI flags**

Modify `src/chess_gaze/cli.py` parser:

```python
analyze.add_argument("--unigaze-device", choices=("cpu", "mps"), default=None)
analyze.add_argument("--unigaze-batch-size", type=int, default=None)
```

Pass values into `AnalyzeRequest`:

```python
unigaze_device=args.unigaze_device,
unigaze_batch_size=args.unigaze_batch_size,
```

- [x] **Step 7: Run focused tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py -q
```

Expected: PASS.

- [x] **Step 8: Commit**

```sh
git add src/chess_gaze/configuration.py src/chess_gaze/cli.py src/chess_gaze/pipeline.py tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py
git commit -m "feat: expose unigaze runtime options"
```

---

### Task 2: Inference Runtime Manifest Semantics

**Files:**
- Modify: `src/chess_gaze/frame_records.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `tests/chess_gaze/test_frame_records.py`
- Modify: `tests/chess_gaze/test_pipeline_contract.py`
- Modify: `tests/chess_gaze/test_qa_summary.py`
- Modify: `tests/chess_gaze/test_scene_artifacts.py`
- Modify: `tests/chess_gaze/test_scene_viewer.py`

**Interfaces:**
- Consumes: resolved `unigaze_device` and `unigaze_batch_size` from Task 1.
- Produces: `InferenceRuntimeRecord`, required `RunManifest.inference`, and truthful `external_observer` metadata for injected observers.

- [x] **Step 1: Write failing frame-record schema tests**

Add to `tests/chess_gaze/test_frame_records.py`:

```python
from chess_gaze.frame_records import InferenceRuntimeRecord, RunManifest, VideoManifest


def test_inference_runtime_record_accepts_default_model_observer() -> None:
    record = InferenceRuntimeRecord(
        observer_source="default_model_observer",
        unigaze_model_id="unigaze-h14-joint",
        unigaze_device="mps",
        unigaze_batch_size=16,
        torch_version="2.12.1",
        torch_mps_available=True,
        mps_fallback_env="unset",
        mps_fast_math_env="unset",
        mps_prefer_metal_env="unset",
        mps_preflight_passed=True,
    )

    assert record.schema_version == "inference-runtime-v1"
    assert record.unigaze_device == "mps"


def test_inference_runtime_record_accepts_external_observer() -> None:
    record = InferenceRuntimeRecord(
        observer_source="external_observer",
        unigaze_model_id=None,
        unigaze_device="not_applicable",
        unigaze_batch_size=None,
        torch_version=None,
        torch_mps_available=None,
        mps_fallback_env="not_applicable",
        mps_fast_math_env="not_applicable",
        mps_prefer_metal_env="not_applicable",
        mps_preflight_passed=None,
    )

    assert record.observer_source == "external_observer"
    assert record.unigaze_model_id is None


def test_run_manifest_requires_inference_runtime_record() -> None:
    manifest = RunManifest(
        run_id="run-1",
        created_at_utc="2026-06-26T00:00:00Z",
        input_path="artifacts/input/nakamura_1.mp4",
        video=VideoManifest(
            source_path="artifacts/input/nakamura_1.mp4",
            source_sha256="0" * 64,
            frame_width=1920,
            frame_height=1080,
            frame_count_decoded=1973,
        ),
        inference=InferenceRuntimeRecord(
            observer_source="external_observer",
            unigaze_model_id=None,
            unigaze_device="not_applicable",
            unigaze_batch_size=None,
            torch_version=None,
            torch_mps_available=None,
            mps_fallback_env="not_applicable",
            mps_fast_math_env="not_applicable",
            mps_prefer_metal_env="not_applicable",
            mps_preflight_passed=None,
        ),
    )

    assert manifest.inference.observer_source == "external_observer"
```

- [x] **Step 2: Write failing pipeline manifest test**

Add to `tests/chess_gaze/test_pipeline_contract.py`:

```python
def test_model_free_observer_run_manifest_records_external_observer(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=1)

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            unigaze_device="mps",
            unigaze_batch_size=7,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["inference"] == {
        "schema_version": "inference-runtime-v1",
        "observer_source": "external_observer",
        "unigaze_model_id": None,
        "unigaze_device": "not_applicable",
        "unigaze_batch_size": None,
        "torch_version": None,
        "torch_mps_available": None,
        "mps_fallback_env": "not_applicable",
        "mps_fast_math_env": "not_applicable",
        "mps_prefer_metal_env": "not_applicable",
        "mps_preflight_passed": None,
    }
```

- [x] **Step 3: Run tests to verify they fail**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_accepts_default_model_observer tests/chess_gaze/test_pipeline_contract.py::test_model_free_observer_run_manifest_records_external_observer -q
```

Expected: FAIL because `InferenceRuntimeRecord` and `RunManifest.inference` do not exist.

- [x] **Step 4: Implement strict manifest records**

Modify `src/chess_gaze/frame_records.py` with the shared-interface `InferenceRuntimeRecord`, then add `inference: InferenceRuntimeRecord` to `RunManifest`.

- [x] **Step 5: Write external-observer metadata in pipeline**

In `src/chess_gaze/pipeline.py`, import `InferenceRuntimeRecord`.

Add a private helper:

```python
def _external_observer_inference_record() -> InferenceRuntimeRecord:
    return InferenceRuntimeRecord(
        observer_source="external_observer",
        unigaze_model_id=None,
        unigaze_device="not_applicable",
        unigaze_batch_size=None,
        torch_version=None,
        torch_mps_available=None,
        mps_fallback_env="not_applicable",
        mps_fast_math_env="not_applicable",
        mps_prefer_metal_env="not_applicable",
        mps_preflight_passed=None,
    )
```

Before writing `run_manifest.json`, set:

```python
inference = _external_observer_inference_record()
```

Then pass `inference=inference` into `RunManifest(...)`.

Task 6 will replace this value for default model-backed runs.

- [x] **Step 6: Update test fixture constructors**

Update every direct `RunManifest(...)` construction in tests to include `inference=_external_observer_inference_record_for_tests()`.

Use this helper in affected test files:

```python
def _external_observer_inference_record() -> InferenceRuntimeRecord:
    return InferenceRuntimeRecord(
        observer_source="external_observer",
        unigaze_model_id=None,
        unigaze_device="not_applicable",
        unigaze_batch_size=None,
        torch_version=None,
        torch_mps_available=None,
        mps_fallback_env="not_applicable",
        mps_fast_math_env="not_applicable",
        mps_prefer_metal_env="not_applicable",
        mps_preflight_passed=None,
    )
```

Affected files found during planning:

- `tests/chess_gaze/test_qa_summary.py`
- `tests/chess_gaze/test_scene_artifacts.py`
- `tests/chess_gaze/test_scene_viewer.py`

- [x] **Step 7: Run focused tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py -q
```

Expected: PASS.

- [x] **Step 8: Commit**

```sh
git add src/chess_gaze/frame_records.py src/chess_gaze/pipeline.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py
git commit -m "feat: record inference runtime metadata"
```

---

### Task 3: UniGaze Batch Prediction Wrapper

**Files:**
- Modify: `src/chess_gaze/gaze_observation.py`
- Test: `tests/chess_gaze/test_gaze_observation.py`

**Interfaces:**
- Consumes: existing `UniGazeModel.from_local_asset()` and `FaceModelGaze`.
- Produces: `UniGazeModel.device`, `UniGazeModel.predict_batch()`, and single-item `predict()` wrapper.

- [x] **Step 1: Update fake backend for batches**

In `tests/chess_gaze/test_gaze_observation.py`, replace `FakeUniGazeBackend.__call__` with:

```python
def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    assert batch.ndim == 4
    assert batch.shape[1:] == (3, 224, 224)
    rows = []
    for index in range(batch.shape[0]):
        rows.append([0.125 + index, -0.25 - index])
    return {"pred_gaze": torch.tensor(rows, dtype=torch.float32, device=batch.device)}
```

- [x] **Step 2: Write failing batch prediction tests**

Add:

```python
def test_unigaze_predict_batch_maps_each_output_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    fake_backend = FakeUniGazeBackend()
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: fake_backend
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    gazes = model.predict_batch(torch.zeros((3, 3, 224, 224), dtype=torch.float32))

    assert [gaze.pitch_radians for gaze in gazes] == pytest.approx(
        [0.125, 1.125, 2.125]
    )
    assert [gaze.yaw_radians for gaze in gazes] == pytest.approx(
        [0.25, 1.25, 2.25]
    )
    assert all(gaze.valid for gaze in gazes)


def test_unigaze_predict_batch_rejects_empty_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: FakeUniGazeBackend()
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    with pytest.raises(ValueError, match="non-empty"):
        model.predict_batch(torch.zeros((0, 3, 224, 224), dtype=torch.float32))


def test_unigaze_predict_batch_rejects_output_row_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadBackend(FakeUniGazeBackend):
        def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            del batch
            return {"pred_gaze": torch.zeros((1, 2), dtype=torch.float32)}

    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(unigaze_loader, "build_unigaze_model", lambda _key: BadBackend())
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    with pytest.raises(ValueError, match="shape"):
        model.predict_batch(torch.zeros((2, 3, 224, 224), dtype=torch.float32))


def test_unigaze_predict_batch_marks_non_finite_row_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NonFiniteBackend(FakeUniGazeBackend):
        def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            del batch
            return {
                "pred_gaze": torch.tensor(
                    [[0.1, -0.2], [float("nan"), -0.3]], dtype=torch.float32
                )
            }

    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(
        unigaze_loader, "build_unigaze_model", lambda _key: NonFiniteBackend()
    )
    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")

    valid_gaze, invalid_gaze = model.predict_batch(
        torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    )

    assert valid_gaze.valid is True
    assert invalid_gaze.valid is False
    assert invalid_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
```

- [x] **Step 3: Add a device-transfer test with a backend spy**

Add:

```python
def test_unigaze_predict_batch_moves_input_to_model_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DeviceSpyBackend(FakeUniGazeBackend):
        observed_device: torch.device | None = None

        def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            self.observed_device = batch.device
            return {"pred_gaze": torch.tensor([[0.1, -0.2]], dtype=torch.float32)}

    asset_path = tmp_path / "unigaze_h14_joint.safetensors"
    asset_path.write_bytes(b"weights")
    backend = DeviceSpyBackend()
    unigaze_loader = importlib.import_module("unigaze.loader")
    monkeypatch.setattr(unigaze_loader, "build_unigaze_model", lambda _key: backend)

    model = UniGazeModel.from_local_asset(_asset(asset_path), device="cpu")
    model.predict_batch(torch.zeros((1, 3, 224, 224), dtype=torch.float32))

    assert backend.observed_device == torch.device("cpu")
```

- [x] **Step 4: Run tests to verify failure**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py::test_unigaze_predict_batch_maps_each_output_row tests/chess_gaze/test_gaze_observation.py::test_unigaze_predict_batch_marks_non_finite_row_invalid -q
```

Expected: FAIL because `predict_batch()` does not exist and `predict()` still assumes one row.

- [x] **Step 5: Implement model device ownership**

Modify `src/chess_gaze/gaze_observation.py`:

```python
class UniGazeModel:
    def __init__(self, backend: Any, *, device: str) -> None:
        self._backend = backend
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device
```

Update `from_local_asset()` return:

```python
return cls(backend, device=device)
```

- [x] **Step 6: Implement `predict_batch()` and wrapper `predict()`**

Use this behavior:

```python
def predict_batch(self, normalized_batch: torch.Tensor) -> tuple[FaceModelGaze, ...]:
    if normalized_batch.ndim != 4 or normalized_batch.shape[1] != 3:
        raise ValueError("normalized_batch must have shape (batch, 3, H, W)")
    if normalized_batch.shape[0] < 1:
        raise ValueError("normalized_batch must contain a non-empty batch")

    normalized_batch = normalized_batch.to(self._device)
    with torch.inference_mode():
        output = self._backend(normalized_batch)

    pred_gaze = output.get("pred_gaze") if isinstance(output, dict) else None
    if not isinstance(pred_gaze, torch.Tensor):
        raise ValueError("UniGaze output must contain tensor pred_gaze")
    if (
        pred_gaze.ndim != 2
        or pred_gaze.shape[0] != normalized_batch.shape[0]
        or pred_gaze.shape[1] != 2
    ):
        raise ValueError(
            "UniGaze pred_gaze must have shape (batch, 2) matching input batch"
        )

    pred_gaze_cpu = pred_gaze.detach().cpu()
    return tuple(_face_model_gaze_from_pred_row(pred_gaze_cpu[index]) for index in range(pred_gaze_cpu.shape[0]))
```

Add helper:

```python
def _face_model_gaze_from_pred_row(pred_row: torch.Tensor) -> FaceModelGaze:
    pitch_radians = float(pred_row[0])
    yaw_radians = -float(pred_row[1])
    if not math.isfinite(pitch_radians) or not math.isfinite(yaw_radians):
        return FaceModelGaze(
            valid=False,
            method=UNIGAZE_METHOD,
            pitch_radians=None,
            yaw_radians=None,
            unit_vector=None,
            confidence=None,
            confidence_source=UNIGAZE_CONFIDENCE_SOURCE,
            reason_invalid=ErrorCode.GAZE_MODEL_FAILED,
        )
    return FaceModelGaze(
        valid=True,
        method=UNIGAZE_METHOD,
        pitch_radians=pitch_radians,
        yaw_radians=yaw_radians,
        unit_vector=pitch_yaw_to_unit_vector(
            pitch_radians=pitch_radians, yaw_radians=yaw_radians
        ),
        confidence=None,
        confidence_source=UNIGAZE_CONFIDENCE_SOURCE,
        reason_invalid=None,
    )
```

Keep `predict()` as:

```python
def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze:
    gazes = self.predict_batch(normalized_batch)
    if len(gazes) != 1:
        raise ValueError("UniGaze predict() requires exactly one batch row")
    return gazes[0]
```

- [x] **Step 7: Run focused tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_gaze_observation.py -q
```

Expected: PASS.

- [x] **Step 8: Commit**

```sh
git add src/chess_gaze/gaze_observation.py tests/chess_gaze/test_gaze_observation.py
git commit -m "feat: support batched unigaze predictions"
```

---

### Task 4: Pipeline Batch Transport

**Files:**
- Modify: `src/chess_gaze/pipeline.py`
- Test: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Consumes: resolved `unigaze_batch_size` from Task 1.
- Produces: optional `FrameBatchRecordObserver`, batched `_process_frame_batch()`, ordered JSONL writes, final partial-batch flush, and single-observer fallback.

- [x] **Step 1: Write fake batch observer test**

Add to `tests/chess_gaze/test_pipeline_contract.py`:

```python
def test_analyze_video_uses_batch_observer_without_reordering_frames(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=5)
    observed_batches: list[list[str]] = []

    def fake_batch_record(frames: list[ObserverFrame]) -> list[FrameRecord]:
        observed_batches.append([frame.frame_id for frame in frames])
        return [_fake_record(frame) for frame in frames]

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            unigaze_batch_size=2,
        ),
        observers=ObserverBundle(
            frame_observer=_fake_record,
            frame_batch_observer=fake_batch_record,
        ),
    )

    records = _records_from(result.frames_jsonl_path)
    assert observed_batches == [
        ["f000000000", "f000000001"],
        ["f000000002", "f000000003"],
        ["f000000004"],
    ]
    assert [record.frame_id for record in records] == [
        "f000000000",
        "f000000001",
        "f000000002",
        "f000000003",
        "f000000004",
    ]
    assert len(list(result.layout.raw_frames_dir.glob("*.png"))) == 5
    assert len(list(result.layout.processed_frames_dir.glob("*.jpg"))) == 5
    assert result.decoded_frame_count == 5
```

Add identity mismatch coverage:

```python
def test_batch_observer_identity_mismatch_fails_schema_validation(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=2)

    def wrong_batch_record(frames: list[ObserverFrame]) -> list[FrameRecord]:
        records = [_fake_record(frame) for frame in frames]
        payload = records[0].model_dump(mode="python")
        payload["frame_index"] = 99
        return [FrameRecord.model_validate(payload), records[1]]

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=tmp_path / "output",
                unigaze_batch_size=2,
            ),
            observers=ObserverBundle(
                frame_observer=_fake_record,
                frame_batch_observer=wrong_batch_record,
            ),
        )

    assert exc_info.value.code is CliErrorCode.SCHEMA_VALIDATION_FAILED
```

- [x] **Step 2: Run tests to verify failure**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_uses_batch_observer_without_reordering_frames -q
```

Expected: FAIL because `ObserverBundle.frame_batch_observer` does not exist.

- [x] **Step 3: Add batch observer protocol**

Modify imports in `src/chess_gaze/pipeline.py`:

```python
from collections.abc import Callable, Sequence
```

Add:

```python
class FrameBatchRecordObserver(Protocol):
    def __call__(self, frames: Sequence[ObserverFrame]) -> Sequence[FrameRecord]:
        raise NotImplementedError
```

Update `ObserverBundle` as shown in Shared Interfaces.

- [x] **Step 4: Refactor per-frame raw/write pieces into reusable helpers**

Add private helper:

```python
@dataclass(frozen=True)
class _PreparedDecodedFrame:
    decoded_frame: DecodedFrame
    observer_frame: ObserverFrame
    raw_frame_errors: list[ErrorRecord]
```

Add:

```python
def _prepare_decoded_frame(
    decoded_frame: DecodedFrame,
    resolved: _ResolvedRequest,
    layout: RunLayout,
) -> _PreparedDecodedFrame:
    frame_errors: list[ErrorRecord] = []
    raw_path = layout.raw_frames_dir / f"{decoded_frame.frame_id}.png"
    try:
        _validate_image_format(resolved.raw_frame_image_format, "png")
        raw_frame_writer(raw_path, decoded_frame.rgb)
    except Exception as exc:
        frame_errors.append(
            ErrorRecord(
                code=ErrorCode.RAW_FRAME_WRITE_FAILED,
                message=f"Raw frame write failed: {exc}",
            )
        )
    observer_frame = ObserverFrame(
        frame_id=decoded_frame.frame_id,
        frame_index=decoded_frame.frame_index,
        timestamp_seconds=_timestamp_seconds(decoded_frame),
        rgb=decoded_frame.rgb,
        pts=decoded_frame.pts,
        pts_seconds=decoded_frame.pts_seconds,
        duration_seconds=decoded_frame.duration_seconds,
    )
    return _PreparedDecodedFrame(
        decoded_frame=decoded_frame,
        observer_frame=observer_frame,
        raw_frame_errors=frame_errors,
    )
```

Add rendering helper:

```python
def _render_processed_frame_and_collect_errors(
    decoded_frame: DecodedFrame,
    record: FrameRecord,
    resolved: _ResolvedRequest,
    layout: RunLayout,
) -> tuple[FrameRecord, list[ErrorRecord]]:
    frame_errors: list[ErrorRecord] = []
    processed_path = layout.processed_frames_dir / f"{decoded_frame.frame_id}.jpg"
    try:
        _validate_image_format(resolved.processed_frame_image_format, "jpg")
        render_processed_frame(
            decoded_frame.rgb,
            record,
            processed_path,
            resolved.processed_frame_jpeg_quality,
        )
    except Exception as exc:
        frame_errors.append(
            ErrorRecord(
                code=ErrorCode.PROCESSED_FRAME_WRITE_FAILED,
                message=f"Processed frame write failed: {exc}",
            )
        )
        record = _record_with_errors(record, frame_errors)
    return record, frame_errors
```

Update `_process_frame()` to use these helpers and preserve existing behavior.

- [x] **Step 5: Implement `_process_frame_batch()`**

Add:

```python
def _process_frame_batch(
    decoded_frames: Sequence[DecodedFrame],
    observers: ObserverBundle,
    resolved: _ResolvedRequest,
    layout: RunLayout,
    *,
    errors_handle: TextIO,
) -> list[tuple[FrameRecord, list[ErrorRecord]]]:
    prepared = [
        _prepare_decoded_frame(decoded_frame, resolved, layout)
        for decoded_frame in decoded_frames
    ]
    if observers.frame_batch_observer is None:
        records = [
            observers.frame_observer(item.observer_frame)
            for item in prepared
        ]
    else:
        records = list(
            observers.frame_batch_observer([item.observer_frame for item in prepared])
        )
    if len(records) != len(prepared):
        raise PipelineError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            (
                "Batch observer returned a different record count: "
                f"{len(records)} != {len(prepared)}"
            ),
        )

    processed: list[tuple[FrameRecord, list[ErrorRecord]]] = []
    for item, record in zip(prepared, records, strict=True):
        _validate_observer_record_identity(record, item.decoded_frame)
        record = _record_with_errors(record, item.raw_frame_errors)
        record, processed_errors = _render_processed_frame_and_collect_errors(
            item.decoded_frame, record, resolved, layout
        )
        frame_errors = item.raw_frame_errors + processed_errors
        frame_error_writer(errors_handle, record)
        processed.append((record, frame_errors))
    return processed
```

- [x] **Step 6: Use accumulator in `analyze_video()`**

Replace the frame loop with a batch accumulator:

```python
pending_batch: list[DecodedFrame] = []
for decoded_frame in iter_decoded_frames(resolved.video_path):
    decoded_frame_count += 1
    pending_batch.append(decoded_frame)
    if len(pending_batch) < resolved.unigaze_batch_size:
        continue
    for record, frame_errors in _process_frame_batch(
        pending_batch,
        observers,
        resolved,
        layout,
        errors_handle=errors_handle,
    ):
        frame_error_count += len(frame_errors)
        frames_handle.write(record.model_dump_json() + "\n")
    pending_batch = []

if pending_batch:
    for record, frame_errors in _process_frame_batch(
        pending_batch,
        observers,
        resolved,
        layout,
        errors_handle=errors_handle,
    ):
        frame_error_count += len(frame_errors)
        frames_handle.write(record.model_dump_json() + "\n")
```

This uses the same path for single observers and batch observers.

- [x] **Step 7: Run focused pipeline tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py -q
```

Expected: PASS.

- [x] **Step 8: Commit**

```sh
git add src/chess_gaze/pipeline.py tests/chess_gaze/test_pipeline_contract.py
git commit -m "feat: add frame batch transport"
```

---

### Task 5: ModelBackedFrameObserver Batch Mode

**Files:**
- Modify: `src/chess_gaze/frame_observation.py`
- Test: `tests/chess_gaze/test_frame_observation.py`

**Interfaces:**
- Consumes: `FaceGazeModel.predict_batch()` from Task 3 and batch transport from Task 4.
- Produces: `ModelBackedFrameObserver.__call__(frame)` compatibility plus `ModelBackedFrameObserver.observe_batch(frames)` for default model-backed pipeline.

- [x] **Step 1: Extend fake gaze model to support batches**

In `tests/chess_gaze/test_frame_observation.py`, update `_FakeGazeModel`:

```python
class _FakeGazeModel:
    def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze:
        return self.predict_batch(normalized_batch)[0]

    def predict_batch(self, normalized_batch: torch.Tensor) -> tuple[FaceModelGaze, ...]:
        assert tuple(normalized_batch.shape[1:]) == (3, 224, 224)
        return tuple(
            FaceModelGaze(
                valid=True,
                method="fake_unigaze",
                pitch_radians=0.02 + index,
                yaw_radians=0.01 + index,
                unit_vector=pitch_yaw_to_unit_vector(
                    pitch_radians=0.02 + index,
                    yaw_radians=0.01 + index,
                ),
                confidence=None,
                confidence_source="not_provided_by_unigaze",
                reason_invalid=None,
            )
            for index in range(normalized_batch.shape[0])
        )
```

Add equivalent `predict_batch()` to `_DisagreeingGazeModel`.

- [x] **Step 2: Write batch row-to-frame mapping test**

Add:

```python
def test_model_backed_frame_observer_batch_maps_model_rows_to_frames(
    tmp_path: Path,
) -> None:
    run_layout = _run_layout(tmp_path)
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_face_observation(candidate)),
        gaze_model=_FakeGazeModel(),
        calibration=default_calibration(),
        run_layout=run_layout,
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )
    frames = [
        _observer_frame(),
        ObserverFrame(
            frame_id="f000000001",
            frame_index=1,
            timestamp_seconds=1.0,
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            pts=None,
            pts_seconds=None,
            duration_seconds=None,
        ),
    ]

    records = observer.observe_batch(frames)

    assert [record.frame_id for record in records] == ["f000000000", "f000000001"]
    assert records[0].appearance_gaze.pitch_radians == 0.02
    assert records[1].appearance_gaze.pitch_radians == 1.02
    assert records[0].recommended_gaze.valid is True
    assert records[1].recommended_gaze.valid is False
    assert records[1].recommended_gaze.reason_invalid is ErrorCode.GAZE_ESTIMATORS_DISAGREE
```

Before adding this test, factor the repeated `RunLayout(...)` construction into:

```python
def _run_layout(tmp_path: Path) -> RunLayout:
    return RunLayout(
        run_dir=tmp_path,
        raw_frames_dir=tmp_path / "raw_frames",
        processed_frames_dir=tmp_path / "processed_frames",
        crops_dir=tmp_path / "crops",
        face_crops_dir=tmp_path / "crops" / "face",
        eyes_crops_dir=tmp_path / "crops" / "eyes",
        left_eye_crops_dir=tmp_path / "crops" / "eyes" / "left",
        right_eye_crops_dir=tmp_path / "crops" / "eyes" / "right",
        records_dir=tmp_path / "records",
    )
```

- [x] **Step 3: Write missing-face batch test**

Add:

```python
def test_model_backed_frame_observer_batch_preserves_missing_face_record(
    tmp_path: Path,
) -> None:
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_missing_face_observation()),
        gaze_model=_FakeGazeModel(),
        calibration=default_calibration(),
        run_layout=_run_layout(tmp_path),
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )

    [record] = observer.observe_batch([_observer_frame()])

    assert record.face.present is False
    assert record.appearance_gaze.valid is False
    assert record.appearance_gaze.reason_invalid is ErrorCode.GAZE_MODEL_FAILED
    assert ErrorCode.FACE_NOT_FOUND in {error.code for error in record.errors}
```

- [x] **Step 4: Write per-row invalid output test**

Add a fake model:

```python
class _OneInvalidRowGazeModel:
    def predict(self, normalized_batch: torch.Tensor) -> FaceModelGaze:
        return self.predict_batch(normalized_batch)[0]

    def predict_batch(self, normalized_batch: torch.Tensor) -> tuple[FaceModelGaze, ...]:
        return (
            FaceModelGaze(
                valid=True,
                method="fake_unigaze",
                pitch_radians=0.02,
                yaw_radians=0.01,
                unit_vector=pitch_yaw_to_unit_vector(
                    pitch_radians=0.02, yaw_radians=0.01
                ),
                confidence=None,
                confidence_source="not_provided_by_unigaze",
                reason_invalid=None,
            ),
            FaceModelGaze(
                valid=False,
                method="fake_unigaze",
                pitch_radians=None,
                yaw_radians=None,
                unit_vector=None,
                confidence=None,
                confidence_source="not_provided_by_unigaze",
                reason_invalid=ErrorCode.GAZE_MODEL_FAILED,
            ),
        )
```

Add test:

```python
def test_model_backed_frame_observer_batch_marks_only_invalid_model_row(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    observer = ModelBackedFrameObserver(
        face_observer=_FakeFaceObserver(_face_observation(candidate)),
        gaze_model=_OneInvalidRowGazeModel(),
        calibration=default_calibration(),
        run_layout=_run_layout(tmp_path),
        eye_observer=_observe_eyes,
        head_pose_estimator=_estimate_head_pose,
        face_crop_normalizer=_normalize_face_crop,
    )
    frames = [
        _observer_frame(),
        ObserverFrame(
            frame_id="f000000001",
            frame_index=1,
            timestamp_seconds=1.0,
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            pts=None,
            pts_seconds=None,
            duration_seconds=None,
        ),
    ]

    first, second = observer.observe_batch(frames)

    assert first.appearance_gaze.valid is True
    assert second.appearance_gaze.valid is False
    assert second.status is FrameStatus.ERROR
    assert ErrorCode.GAZE_MODEL_FAILED in {error.code for error in second.errors}
```

- [x] **Step 5: Run tests to verify failure**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_observation.py::test_model_backed_frame_observer_batch_maps_model_rows_to_frames -q
```

Expected: FAIL because `observe_batch()` does not exist.

- [x] **Step 6: Implement batch model protocol**

Modify `FaceGazeModel` in `src/chess_gaze/frame_observation.py`:

```python
class FaceGazeModel(Protocol):
    def predict(self, normalized_batch: Any) -> FaceModelGaze:
        raise NotImplementedError

    def predict_batch(self, normalized_batch: Any) -> tuple[FaceModelGaze, ...]:
        raise NotImplementedError
```

- [x] **Step 7: Refactor observer into evidence and assembly helpers**

Add private dataclass:

```python
@dataclass(frozen=True)
class _FrameEvidence:
    frame: Any
    face_observation: FaceObservation | None
    errors: list[ErrorRecord]
    face_record: FaceRecord
    selected_face: FaceCandidate | None
    left_eye: EyeRecord
    right_eye: EyeRecord
    head_pose_record: HeadPoseRecord
    left_geometric: GazeAngles
    right_geometric: GazeAngles
    geometric_gaze: GazeAngles
    normalized_face_crop: NormalizedFaceCrop | None
```

Implement `_collect_frame_evidence(frame: Any) -> _FrameEvidence` by moving the existing face/eye/head/geometric/crop work out of `__call__`. It must return `_missing_face_record()` through a separate assembly path when `selected_face is None`; do not run eye/head/crop without a selected face.

Implement:

```python
def _record_from_evidence(
    self,
    evidence: _FrameEvidence,
    appearance_gaze: FaceModelGaze,
) -> FrameRecord:
```

This helper must use the existing `synthesize_recommended_gaze()`, `_face_model_gaze_record()`, and `_frame_status()` logic.

- [x] **Step 8: Implement `observe_batch()`**

Add:

```python
def observe_batch(self, frames: Sequence[Any]) -> list[FrameRecord]:
    evidence_items = [self._collect_frame_evidence(frame) for frame in frames]
    crop_items = [
        (index, evidence.normalized_face_crop.tensor)
        for index, evidence in enumerate(evidence_items)
        if evidence.selected_face is not None and evidence.normalized_face_crop is not None
    ]
    appearance_by_index: dict[int, FaceModelGaze] = {}
    if crop_items:
        batch = torch.cat([tensor for _index, tensor in crop_items], dim=0)
        gazes = self.gaze_model.predict_batch(batch)
        if len(gazes) != len(crop_items):
            raise ValueError(
                "Appearance gaze model returned a different number of rows"
            )
        for (index, _tensor), gaze in zip(crop_items, gazes, strict=True):
            appearance_by_index[index] = gaze

    records: list[FrameRecord] = []
    for index, evidence in enumerate(evidence_items):
        if evidence.selected_face is None:
            if evidence.face_observation is None:
                raise AssertionError("missing face evidence requires face observation")
            records.append(
                _missing_face_record(
                    evidence.frame,
                    evidence.face_observation,
                    evidence.errors,
                )
            )
            continue
        appearance_gaze = appearance_by_index.get(index)
        if appearance_gaze is None:
            appearance_gaze = _invalid_face_model_gaze()
        if not appearance_gaze.valid and appearance_gaze.reason_invalid is not None:
            _append_error_once(
                evidence.errors,
                ErrorRecord(
                    code=appearance_gaze.reason_invalid,
                    message="Appearance gaze model failed: non-finite UniGaze output.",
                ),
            )
        records.append(self._record_from_evidence(evidence, appearance_gaze))
    return records
```

The actual implementation must contain explicit calls and values from the current single-frame code. Do not leave placeholder bodies in source.

Update `__call__`:

```python
def __call__(self, frame: Any) -> FrameRecord:
    return self.observe_batch([frame])[0]
```

- [x] **Step 9: Run focused tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_observation.py -q
```

Expected: PASS.

- [x] **Step 10: Commit**

```sh
git add src/chess_gaze/frame_observation.py tests/chess_gaze/test_frame_observation.py
git commit -m "feat: batch model-backed frame observation"
```

---

### Task 6: MPS Preflight And Default Integration

**Files:**
- Create: `src/chess_gaze/unigaze_runtime.py`
- Modify: `src/chess_gaze/pipeline.py`
- Test: `tests/chess_gaze/test_pipeline_contract.py`
- Test: `tests/chess_gaze/test_frame_records.py`

**Interfaces:**
- Consumes: `InferenceRuntimeRecord` from Task 2, `UniGazeModel.predict_batch()` from Task 3, batch observer from Tasks 4 and 5.
- Produces: pre-run MPS failure behavior, default model-backed inference metadata, and default observer `frame_batch_observer=observer.observe_batch`.

- [x] **Step 1: Write MPS env/preflight tests**

Add to `tests/chess_gaze/test_pipeline_contract.py`:

```python
def test_explicit_mps_unavailable_fails_before_run_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "models"
    registry_path = tmp_path / "model_registry.json"
    make_tiny_video(video_path)
    _write_model_registry_with_assets(models_root, registry_path)

    from chess_gaze import unigaze_runtime

    monkeypatch.setattr(unigaze_runtime.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                models_root=models_root,
                model_registry_path=registry_path,
                unigaze_device="mps",
                unigaze_batch_size=2,
            )
        )

    assert exc_info.value.code is CliErrorCode.USAGE
    assert not output_root.exists()
```

Add:

```python
@pytest.mark.parametrize("env_name", ["PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"])
def test_explicit_mps_rejects_unsafe_env_before_run_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "models"
    registry_path = tmp_path / "model_registry.json"
    make_tiny_video(video_path)
    _write_model_registry_with_assets(models_root, registry_path)
    monkeypatch.setenv(env_name, "1")

    with pytest.raises(PipelineError) as exc_info:
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                models_root=models_root,
                model_registry_path=registry_path,
                unigaze_device="mps",
                unigaze_batch_size=2,
            )
        )

    assert exc_info.value.code is CliErrorCode.USAGE
    assert env_name in str(exc_info.value)
    assert not output_root.exists()
```

Add metadata assertion for default model runs using monkeypatch:

```python
def test_default_model_observer_manifest_records_unigaze_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    models_root = tmp_path / "models"
    registry_path = tmp_path / "model_registry.json"
    make_tiny_video(video_path, frame_count=1)
    _write_model_registry_with_assets(models_root, registry_path)

    from chess_gaze import pipeline
    from chess_gaze.frame_records import InferenceRuntimeRecord

    def fake_prepare_unigaze_runtime(asset: object, *, device: str, batch_size: int, input_size_px: int) -> object:
        del asset, input_size_px
        return SimpleNamespace(
            model=object(),
            inference=InferenceRuntimeRecord(
                observer_source="default_model_observer",
                unigaze_model_id="unigaze-h14-joint",
                unigaze_device=device,
                unigaze_batch_size=batch_size,
                torch_version="test-torch",
                torch_mps_available=True,
                mps_fallback_env="unset",
                mps_fast_math_env="unset",
                mps_prefer_metal_env="unset",
                mps_preflight_passed=True,
            ),
        )

    def fake_default_observer_bundle_factory(
        resolved_assets: list[Any],
        calibration: object,
        run_layout: object,
        gaze_model: object,
    ) -> ObserverBundle:
        del resolved_assets, calibration, run_layout, gaze_model
        return ObserverBundle(frame_observer=_fake_record)

    monkeypatch.setattr(
        pipeline, "prepare_unigaze_runtime", fake_prepare_unigaze_runtime
    )
    monkeypatch.setattr(
        pipeline,
        "default_observer_bundle_factory",
        fake_default_observer_bundle_factory,
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=output_root,
            models_root=models_root,
            model_registry_path=registry_path,
            unigaze_device="mps",
            unigaze_batch_size=7,
        )
    )

    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["inference"]["observer_source"] == "default_model_observer"
    assert manifest["inference"]["unigaze_device"] == "mps"
    assert manifest["inference"]["unigaze_batch_size"] == 7
```

- [x] **Step 2: Run tests to verify failure**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py::test_explicit_mps_unavailable_fails_before_run_layout tests/chess_gaze/test_pipeline_contract.py::test_default_model_observer_manifest_records_unigaze_runtime -q
```

Expected: FAIL because `unigaze_runtime.py` and default runtime preparation do not exist.

- [x] **Step 3: Implement `unigaze_runtime.py`**

Create `src/chess_gaze/unigaze_runtime.py` using the shared interfaces. Use these helper rules:

```python
def _env_state(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return "unset"
    return value


def _env_enabled(name: str) -> bool:
    return _env_state(name).lower() not in {"unset", "0", "false", "no"}
```

`prepare_unigaze_runtime()` behavior:

- validates `batch_size >= 1`;
- rejects `device` not in `{"cpu", "mps"}`;
- for `device == "mps"`:
  - rejects unavailable `torch.backends.mps.is_available()`;
  - rejects enabled fallback and fast-math env vars;
  - constructs `UniGazeModel.from_local_asset(asset, device="mps")`;
  - runs `model.predict_batch(torch.zeros((batch_size, 3, input_size_px, input_size_px), dtype=torch.float32))`;
  - calls `torch.mps.synchronize()`;
  - raises `UniGazeRuntimeError` with the original exception message when MPS dummy inference fails because of memory pressure or unsupported operations.
- for `device == "cpu"`:
  - constructs `UniGazeModel.from_local_asset(asset, device="cpu")`;
  - no dummy run is required;
  - sets `mps_preflight_passed=None`.

Because `unigaze_runtime.py` must not import `PipelineError` from `pipeline.py`, define local `UniGazeRuntimeError(RuntimeError)` with `message: str`; catch it in `pipeline.py` and raise `PipelineError(CliErrorCode.USAGE, str(exc))`.

- [x] **Step 4: Integrate runtime preparation before run layout**

Modify `src/chess_gaze/pipeline.py`:

```python
from chess_gaze.unigaze_runtime import (
    PreparedUniGazeRuntime,
    UniGazeRuntimeError,
    external_observer_inference_record,
    prepare_unigaze_runtime,
)
```

Before `_estimate_disk_space()` and `create_run_layout()`, prepare default runtime only when `observers is None`:

```python
prepared_unigaze_runtime: PreparedUniGazeRuntime | None = None
if observers is None:
    if resolved_model_assets is None:
        raise AssertionError("resolved_model_assets must be set for default run")
    gaze_asset = _asset_by_id(resolved_model_assets, UNIGAZE_MODEL_ID)
    try:
        prepared_unigaze_runtime = prepare_unigaze_runtime(
            gaze_asset,
            device=resolved.unigaze_device,
            batch_size=resolved.unigaze_batch_size,
            input_size_px=calibration.unigaze_input_size_px,
        )
    except UniGazeRuntimeError as exc:
        raise PipelineError(CliErrorCode.USAGE, str(exc)) from exc
```

Ensure this occurs before `create_run_layout()`. Move `calibration = default_calibration()` above runtime preparation because preflight needs `unigaze_input_size_px`.

Set manifest inference:

```python
inference = (
    prepared_unigaze_runtime.inference
    if prepared_unigaze_runtime is not None
    else external_observer_inference_record()
)
```

- [x] **Step 5: Update default observer factory signature**

Change `DefaultObserverBundleFactory`:

```python
DefaultObserverBundleFactory = Callable[
    [list[ResolvedModelAsset], CalibrationRecord, RunLayout, object], "ObserverBundle"
]
```

Change `_default_observer_bundle_factory` to accept `gaze_model: object`, remove the internal `UniGazeModel.from_local_asset()` call, and return:

```python
return ObserverBundle(
    frame_observer=observer,
    frame_batch_observer=observer.observe_batch,
    close=observer.close,
)
```

Call it with `prepared_unigaze_runtime.model`.

- [x] **Step 6: Run focused tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_frame_records.py -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```sh
git add src/chess_gaze/unigaze_runtime.py src/chess_gaze/pipeline.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_frame_records.py
git commit -m "feat: preflight unigaze runtime"
```

---

### Task 7: Run Equivalence Harness

**Files:**
- Create: `src/chess_gaze/run_equivalence.py`
- Test: `tests/chess_gaze/test_run_equivalence.py`

**Interfaces:**
- Consumes: completed run directories containing `records/frames.jsonl`, `records/scene_frames.jsonl`, `qa_summary.json`, and viewer data.
- Produces: `compare_runs()` and `EquivalenceReport` used by the benchmark module and closeout.

- [x] **Step 1: Write run-equivalence tests**

Create `tests/chess_gaze/test_run_equivalence.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

from chess_gaze.run_equivalence import EquivalenceTolerances, compare_runs


def _write_minimal_run(run_dir: Path, *, yaw: float, scene_x: float, monitor_u: float) -> None:
    records_dir = run_dir / "records"
    viewer_dir = run_dir / "viewer"
    records_dir.mkdir(parents=True)
    viewer_dir.mkdir(parents=True)
    frame = {
        "frame_id": "f000000000",
        "frame_index": 0,
        "status": "OK",
        "timestamp_seconds": 0.0,
        "face": {"present": True, "bounding_box": {"space": "image_px", "x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0}, "landmarks": [{"space": "image_px", "x": 1.0, "y": 1.0}], "reason_invalid": None},
        "left_eye": {"present": True, "bounding_box": {"space": "image_px", "x_min": 1.0, "y_min": 1.0, "x_max": 2.0, "y_max": 2.0}, "pupil_center": {"space": "image_px", "x": 1.5, "y": 1.5}, "iris_landmarks": [{"space": "image_px", "x": 1.5, "y": 1.5}], "reason_invalid": None},
        "right_eye": {"present": True, "bounding_box": {"space": "image_px", "x_min": 3.0, "y_min": 1.0, "x_max": 4.0, "y_max": 2.0}, "pupil_center": {"space": "image_px", "x": 3.5, "y": 1.5}, "iris_landmarks": [{"space": "image_px", "x": 3.5, "y": 1.5}], "reason_invalid": None},
        "head_pose": {"valid": True, "yaw_radians": 0.0, "pitch_radians": 0.0, "roll_radians": 0.0, "reason_invalid": None},
        "geometric_gaze": {"valid": True, "yaw_radians": 0.0, "pitch_radians": 0.0, "reason_invalid": None},
        "appearance_gaze": {"valid": True, "yaw_radians": yaw, "pitch_radians": 0.0, "reason_invalid": None},
        "recommended_gaze": {"valid": True, "yaw_radians": yaw, "pitch_radians": 0.0, "reason_invalid": None},
        "errors": [],
    }
    scene_frame = {
        "frame_id": "f000000000",
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "eye_midpoint": {"valid": True, "position_camera_m": {"frame": "camera_opencv_pseudo_m", "x": 0.0, "y": 0.0, "z": 1.0}, "position_scene_m": {"frame": "scene_pseudo_m", "x": 0.0, "y": 0.0, "z": 0.0}, "reason_invalid": None},
        "unigaze_ray": {"valid": True, "origin_scene_m": {"frame": "scene_pseudo_m", "x": 0.0, "y": 0.0, "z": 0.0}, "direction_camera": {"frame": "camera_opencv_pseudo_m", "x": scene_x, "y": 0.0, "z": -1.0}, "direction_scene": {"frame": "scene_pseudo_m", "x": scene_x, "y": 0.0, "z": -1.0}, "direction_source": "appearance_gaze_unigaze_pitch_yaw", "reason_invalid": None, "source_reason_invalid": None},
        "monitor_hit": {"valid": True, "point_scene_m": {"frame": "scene_pseudo_m", "x": monitor_u, "y": 0.0, "z": -0.7}, "plane_uv_m": {"frame": "monitor_plane_pseudo_m", "u": monitor_u, "v": 0.0}, "inside_physical_monitor": True, "inside_extended_plane": True, "reason_invalid": None},
        "warnings": [],
    }
    qa_summary = {
        "final_status": "complete",
        "counts": {"decoded_frames": 1, "frame_records": 1, "scene_frame_records": 1, "raw_frames": 1, "processed_frames": 1, "crop_files": 0},
        "artifact_validation": {"schema_validation_passed": True, "counts_match": True},
    }
    viewer_data = {"frame_count": 1}
    (records_dir / "frames.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
    (records_dir / "scene_frames.jsonl").write_text(json.dumps(scene_frame) + "\n", encoding="utf-8")
    (run_dir / "qa_summary.json").write_text(json.dumps(qa_summary), encoding="utf-8")
    (viewer_dir / "scene-data.json").write_text(json.dumps(viewer_data), encoding="utf-8")


def test_compare_runs_accepts_numeric_deltas_within_tolerance(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(baseline, yaw=0.1000, scene_x=0.2000, monitor_u=0.3000)
    _write_minimal_run(candidate, yaw=0.1005, scene_x=0.2005, monitor_u=0.3010)

    report = compare_runs(
        baseline,
        candidate,
        tolerances=EquivalenceTolerances(
            appearance_pitch_yaw_radians=1e-3,
            scene_ray_component=1e-3,
            monitor_uv_m=2e-3,
        ),
    )

    assert report.passed is True
    assert report.exact_mismatch_count == 0
    assert report.numeric_mismatch_count == 0


def test_compare_runs_rejects_status_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(baseline, yaw=0.1, scene_x=0.2, monitor_u=0.3)
    _write_minimal_run(candidate, yaw=0.1, scene_x=0.2, monitor_u=0.3)
    frame = json.loads((candidate / "records" / "frames.jsonl").read_text())
    frame["status"] = "ERROR"
    (candidate / "records" / "frames.jsonl").write_text(json.dumps(frame) + "\n")

    report = compare_runs(
        baseline,
        candidate,
        tolerances=EquivalenceTolerances(
            appearance_pitch_yaw_radians=1e-6,
            scene_ray_component=1e-6,
            monitor_uv_m=1e-6,
        ),
    )

    assert report.passed is False
    assert report.exact_mismatch_count == 1
```

- [x] **Step 2: Run tests to verify failure**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_run_equivalence.py -q
```

Expected: FAIL because `run_equivalence.py` does not exist.

- [x] **Step 3: Implement `run_equivalence.py`**

Implement:

- JSONL loading for `frames.jsonl` and `scene_frames.jsonl`;
- JSON loading for `qa_summary.json` and `viewer/scene-data.json`;
- exact comparison of frame IDs/indexes, frame status, face/eye/head/gaze validity booleans, invalid reasons, and error code lists;
- count comparison for raw frames, processed frames, frame records, scene frames, viewer frame count, QA decoded frames, QA counts match, and QA schema validation;
- numeric comparison for:
  - `FrameRecord.appearance_gaze.pitch_radians`;
  - `FrameRecord.appearance_gaze.yaw_radians`;
  - `SceneFrameRecord.unigaze_ray.direction_camera.{x,y,z}`;
  - `SceneFrameRecord.unigaze_ray.direction_scene.{x,y,z}`;
  - `SceneFrameRecord.monitor_hit.plane_uv_m.{u,v}`.

When a numeric field is absent or invalid in both runs, require the corresponding validity and invalid reason to match exactly and do not count it as a numeric delta.

- [x] **Step 4: Run focused tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_run_equivalence.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```sh
git add src/chess_gaze/run_equivalence.py tests/chess_gaze/test_run_equivalence.py
git commit -m "feat: compare analysis runs for equivalence"
```

---

### Task 8: Benchmark Harness

**Files:**
- Create: `src/chess_gaze/unigaze_batch_benchmark.py`
- Test: `tests/chess_gaze/test_unigaze_batch_benchmark.py`

**Interfaces:**
- Consumes: `compare_runs()` from Task 7 and `chess-gaze analyze` CLI with runtime options from prior tasks.
- Produces: benchmark report JSON, one result row per approved candidate, selected fastest passing MPS `batch_size > 1`, and `--print-selected-batch-size`.

- [x] **Step 1: Write benchmark report schema tests**

Create `tests/chess_gaze/test_unigaze_batch_benchmark.py`:

```python
from __future__ import annotations

from chess_gaze.unigaze_batch_benchmark import (
    BenchmarkCandidateResult,
    UniGazeBatchBenchmarkReport,
    selected_mps_batch_size,
)


def test_benchmark_report_selects_fastest_passing_mps_batch_size() -> None:
    report = UniGazeBatchBenchmarkReport(
        git_revision="abc123",
        source_video="artifacts/input/nakamura_1.mp4",
        source_video_sha256="e" * 64,
        decoded_frame_count=1973,
        torch_version="2.12.1",
        unigaze_version="0.1.3",
        mps_available=True,
        mps_fallback_env="unset",
        mps_fast_math_env="unset",
        mps_prefer_metal_env="unset",
        model_asset_sha256="a" * 64,
        baseline_run_dir="artifacts/output/nakamura_1/runs/baseline",
        candidate_results=[
            BenchmarkCandidateResult(
                device="cpu",
                batch_size=1,
                status="passed",
                preflight_seconds=None,
                analysis_wall_seconds=100.0,
                frames_per_second=19.73,
                unigaze_forward_repetitions_seconds=[10.0, 10.1, 9.9],
                unigaze_forward_median_seconds=10.0,
                full_run_dir="cpu1",
                qa_final_status="complete",
                qa_decoded_frames=1973,
                qa_counts_match=True,
                qa_schema_validation_passed=True,
                equivalence_report_path="cpu1-equivalence.json",
                max_appearance_pitch_yaw_delta_radians=0.0,
                max_scene_ray_component_delta=0.0,
                max_monitor_uv_delta_m=0.0,
                peak_mps_memory_bytes=None,
                error_code=None,
                error_message=None,
            ),
            BenchmarkCandidateResult(
                device="mps",
                batch_size=8,
                status="passed",
                preflight_seconds=1.0,
                analysis_wall_seconds=80.0,
                frames_per_second=24.66,
                unigaze_forward_repetitions_seconds=[8.0, 8.1, 7.9],
                unigaze_forward_median_seconds=8.0,
                full_run_dir="mps8",
                qa_final_status="complete",
                qa_decoded_frames=1973,
                qa_counts_match=True,
                qa_schema_validation_passed=True,
                equivalence_report_path="mps8-equivalence.json",
                max_appearance_pitch_yaw_delta_radians=0.0005,
                max_scene_ray_component_delta=0.0005,
                max_monitor_uv_delta_m=0.001,
                peak_mps_memory_bytes=123,
                error_code=None,
                error_message=None,
            ),
            BenchmarkCandidateResult(
                device="mps",
                batch_size=16,
                status="passed",
                preflight_seconds=1.0,
                analysis_wall_seconds=70.0,
                frames_per_second=28.19,
                unigaze_forward_repetitions_seconds=[7.0, 7.1, 6.9],
                unigaze_forward_median_seconds=7.0,
                full_run_dir="mps16",
                qa_final_status="complete",
                qa_decoded_frames=1973,
                qa_counts_match=True,
                qa_schema_validation_passed=True,
                equivalence_report_path="mps16-equivalence.json",
                max_appearance_pitch_yaw_delta_radians=0.0005,
                max_scene_ray_component_delta=0.0005,
                max_monitor_uv_delta_m=0.001,
                peak_mps_memory_bytes=456,
                error_code=None,
                error_message=None,
            ),
        ],
        selected_device=None,
        selected_batch_size=None,
        selected_reason=None,
    )

    assert selected_mps_batch_size(report) == 16


def test_benchmark_report_ignores_failed_mps_candidates() -> None:
    report = UniGazeBatchBenchmarkReport(
        git_revision="abc123",
        source_video="artifacts/input/nakamura_1.mp4",
        source_video_sha256="e" * 64,
        decoded_frame_count=1973,
        torch_version="2.12.1",
        unigaze_version="0.1.3",
        mps_available=True,
        mps_fallback_env="unset",
        mps_fast_math_env="unset",
        mps_prefer_metal_env="unset",
        model_asset_sha256="a" * 64,
        baseline_run_dir="baseline",
        candidate_results=[
            BenchmarkCandidateResult(
                device="mps",
                batch_size=32,
                status="oom",
                preflight_seconds=None,
                analysis_wall_seconds=None,
                frames_per_second=None,
                unigaze_forward_repetitions_seconds=[],
                unigaze_forward_median_seconds=None,
                full_run_dir=None,
                qa_final_status=None,
                qa_decoded_frames=None,
                qa_counts_match=None,
                qa_schema_validation_passed=None,
                equivalence_report_path=None,
                max_appearance_pitch_yaw_delta_radians=None,
                max_scene_ray_component_delta=None,
                max_monitor_uv_delta_m=None,
                peak_mps_memory_bytes=None,
                error_code="OOM",
                error_message="simulated",
            )
        ],
        selected_device=None,
        selected_batch_size=None,
        selected_reason=None,
    )

    assert selected_mps_batch_size(report) is None
```

- [x] **Step 2: Run tests to verify failure**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_unigaze_batch_benchmark.py -q
```

Expected: FAIL because benchmark module does not exist.

- [x] **Step 3: Implement report models and selection**

Create `src/chess_gaze/unigaze_batch_benchmark.py` with the shared-interface report models.

Implement:

```python
def selected_mps_batch_size(report: UniGazeBatchBenchmarkReport) -> int | None:
    passing = [
        result
        for result in report.candidate_results
        if result.device == "mps"
        and result.batch_size > 1
        and result.status == "passed"
        and result.analysis_wall_seconds is not None
    ]
    if not passing:
        return None
    return min(passing, key=lambda item: item.analysis_wall_seconds).batch_size
```

- [x] **Step 4: Implement benchmark CLI**

Support:

```sh
uv run python -m chess_gaze.unigaze_batch_benchmark --video artifacts/input/nakamura_1.mp4 --models-root models --output artifacts/output/benchmarks/2026-06-26-unigaze-mps-batching.json
uv run python -m chess_gaze.unigaze_batch_benchmark --report artifacts/output/benchmarks/2026-06-26-unigaze-mps-batching.json --print-selected-batch-size
```

Candidate grid:

```python
DEVICES = ("cpu", "mps")
BATCH_SIZES = (1, 2, 4, 7, 8, 16, 32, 64)
OPTIONAL_MPS_EXTENSION_BATCH_SIZE = 128
```

Implementation behavior:

- read baseline run dir from `artifacts/output/benchmarks/2026-06-26-current-cpu1-baseline.json`;
- run post-change `cpu/1` first and compare to baseline-compatible artifacts where possible;
- for each candidate:
  - run `chess-gaze analyze` in a subprocess with env vars `PYTORCH_ENABLE_MPS_FALLBACK`, `PYTORCH_MPS_FAST_MATH`, and `PYTORCH_MPS_PREFER_METAL` removed;
  - measure wall seconds with `time.perf_counter()`;
  - parse run directory and viewer path from stdout;
  - load `qa_summary.json`;
  - run `compare_runs()` against fresh post-change `cpu/1`;
  - write equivalence report next to benchmark JSON;
  - record failures as rows instead of aborting the whole matrix.
- include MPS `128` only when `64` passed and `peak_mps_memory_bytes` is available or no memory warning occurred.

Pure UniGaze forward microbenchmark can initially use real normalized crops harvested during candidate analysis by reading the same model-backed path. If that proves too invasive, use the full-run wall time as binding selection and record `unigaze_forward_repetitions_seconds=[]` with `unigaze_forward_median_seconds=None`; that is an explicit residual limitation to record in closeout. Do not invent microbenchmark values.

- [x] **Step 5: Run focused tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_unigaze_batch_benchmark.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```sh
git add src/chess_gaze/unigaze_batch_benchmark.py tests/chess_gaze/test_unigaze_batch_benchmark.py
git commit -m "feat: benchmark unigaze batch profiles"
```

---

### Task 9: Docs, Benchmark Execution, And Closeout

**Files:**
- Modify: `README.md`
- Modify: `docs/development/architecture/source-layout.md`
- Create: `docs/superpowers/closeouts/2026-06-26-unigaze-mps-batching.md`

**Interfaces:**
- Consumes: all implementation tasks and benchmark module.
- Produces: selected MPS `batch_size > 1`, documented usage, final run evidence, benchmark report, and closeout.

- [x] **Step 1: Run focused unit and integration gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_gaze_observation.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_run_equivalence.py tests/chess_gaze/test_unigaze_batch_benchmark.py -q
```

Expected: PASS.

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected: PASS or skip only for explicitly missing local media.

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_rejects_nakamura_overexpanded_faces -q
```

Expected: PASS when `artifacts/input/nakamura_1.mp4` and local MediaPipe model exist.

- [x] **Step 2: Run benchmark grid**

Run unsandboxed if MediaPipe native initialization fails in the managed sandbox:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL UV_CACHE_DIR=.uv-cache uv run python -m chess_gaze.unigaze_batch_benchmark --video artifacts/input/nakamura_1.mp4 --models-root models --output artifacts/output/benchmarks/2026-06-26-unigaze-mps-batching.json
```

Expected: benchmark report JSON exists, includes candidate rows for `cpu` and `mps` batch sizes `1, 2, 4, 7, 8, 16, 32, 64`, records failures as rows, and includes `selected_batch_size` with an integer greater than `1` if any MPS batch candidate passed.

- [x] **Step 3: Extract selected MPS batch size**

Run:

```sh
SELECTED_MPS_BATCH_SIZE=$(UV_CACHE_DIR=.uv-cache uv run python -m chess_gaze.unigaze_batch_benchmark --report artifacts/output/benchmarks/2026-06-26-unigaze-mps-batching.json --print-selected-batch-size)
printf '%s\n' "$SELECTED_MPS_BATCH_SIZE"
```

Expected: prints a concrete integer greater than `1`. If it prints nothing, stop and report that no passing MPS `batch_size > 1` exists.

- [x] **Step 4: Run final optimized Nakamura analysis**

Run:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models --unigaze-device mps --unigaze-batch-size "$SELECTED_MPS_BATCH_SIZE"
```

Expected: command exits `0`, prints a fresh run dir and viewer path, and the fresh run has `qa_summary.final_status == "complete"`, `counts.decoded_frames == 1973`, `counts.frame_records == 1973`, `counts.scene_frame_records == 1973`, `artifact_validation.schema_validation_passed == true`, and `artifact_validation.counts_match == true`.

- [x] **Step 5: Update README**

Add to the Analyze section:

```markdown
UniGaze runtime options default to the compatibility profile:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4 \
  --unigaze-device cpu \
  --unigaze-batch-size 1
```

On Apple Silicon with local models and MPS available, use the concrete
benchmarked profile recorded in the latest UniGaze MPS batching closeout. This
benchmark selected batch size `7`, so the command is:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 \
  --models-root models \
  --unigaze-device mps \
  --unigaze-batch-size 7
```

MPS runs are explicit because PyTorch does not guarantee bitwise-identical CPU
and MPS floating-point results. Analysis rejects `PYTORCH_ENABLE_MPS_FALLBACK=1`
and `PYTORCH_MPS_FAST_MATH=1` for accepted MPS runs so unsupported operations or
fast-math drift cannot silently change the runtime contract.
```

If Step 3 prints a different integer, write that exact integer instead of `16`
before committing README changes. Do not commit symbolic batch-size text.

- [x] **Step 6: Update source layout docs**

Add under `src/chess_gaze/` map in `docs/development/architecture/source-layout.md`:

```markdown
  - `unigaze_runtime.py` owns UniGaze device/runtime validation, MPS preflight,
    synchronization helpers, and inference metadata assembly.
  - `run_equivalence.py` owns strict artifact-level comparison between analysis
    runs for CPU/MPS and batch-size validation.
  - `unigaze_batch_benchmark.py` owns the finite UniGaze batch benchmark grid
    and benchmark report schema.
```

- [x] **Step 7: Write closeout**

Create `docs/superpowers/closeouts/2026-06-26-unigaze-mps-batching.md` with sections:

```markdown
# UniGaze MPS Batching Closeout

Date: 2026-06-26

## Summary

## Root Cause

## Implementation

## Benchmark Matrix

## Selected Optimized Profile

## Equivalence Evidence

## Real Nakamura Verification

## Verification Commands

## Residual Risk
```

Include:

- current-flow CPU batch-1 baseline run dir and timing;
- post-change CPU batch-1 equivalence;
- every candidate row from `artifacts/output/benchmarks/2026-06-26-unigaze-mps-batching.json`;
- selected MPS `batch_size > 1`;
- optimized run dir and viewer path;
- QA summary counts and schema status;
- max observed CPU/MPS numeric deltas;
- candidates rejected for OOM, unsupported ops, equivalence drift, analysis failure, or no speed improvement;
- exact commands and whether unsandboxed execution was required.

- [x] **Step 8: Run broad local gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Expected: PASS. If full `pytest` fails only because ignored legacy media such as `artifacts/input/test_1.mp4` or `artifacts/input/test_2.mp4` is absent, record exact failures in the closeout and rerun the broadest available subset that excludes only absent-media tests.

- [x] **Step 9: Commit docs and closeout**

```sh
git add README.md docs/development/architecture/source-layout.md docs/superpowers/closeouts/2026-06-26-unigaze-mps-batching.md
git commit -m "docs: close out unigaze mps batching"
```

---

## Plan Self-Review Notes

- Spec coverage: every approved spec section maps to at least one task. Config/CLI is Task 1, manifest semantics are Task 2, model wrapper is Task 3, pipeline batching is Task 4, model-backed batching is Task 5, MPS preflight is Task 6, equivalence is Task 7, benchmark grid is Task 8, docs/closeout/final gates are Task 9.
- Red-flag scan: there are no TBD/TODO/fill-in implementation decisions in the plan. The README closeout step records the measured winner from Task 8.
- Type consistency: `unigaze_batch_size` is an `int` in config, request, resolved request, manifest, and benchmark report. Device values are `"cpu"` and `"mps"` for actual UniGaze runs, and `"not_applicable"` only for external-observer manifests.
- Risk called out by subagents: semantic equivalence is a first-class harness in Task 7; QA summary alone is not treated as sufficient. Preflight happens before run creation in Task 6. External-observer manifests are explicitly truthful in Task 2.
