# Task 2 Report: Inference Runtime Manifest Semantics

## What I implemented

- Added `InferenceRuntimeRecord` to `src/chess_gaze/frame_records.py` as a strict manifest record with schema version `inference-runtime-v1`.
- Made `RunManifest.inference` required.
- Added `_external_observer_inference_record()` in `src/chess_gaze/pipeline.py` and wrote that metadata into `run_manifest.json` for injected observer runs.
- Added the required RED tests in:
  - `tests/chess_gaze/test_frame_records.py`
  - `tests/chess_gaze/test_pipeline_contract.py`
- Updated direct `RunManifest(...)` fixture builders in:
  - `tests/chess_gaze/test_qa_summary.py`
  - `tests/chess_gaze/test_scene_artifacts.py`
  - `tests/chess_gaze/test_scene_viewer.py`
  so they include external-observer inference metadata.
- Kept Task 6 work out of scope: no default-model MPS execution or preflight wiring was added.

## RED evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_accepts_default_model_observer tests/chess_gaze/test_pipeline_contract.py::test_model_free_observer_run_manifest_records_external_observer -q
```

Relevant output:

```text
ERROR: found no collectors for /Volumes/git/legotin/chess-gaze/tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_accepts_default_model_observer

ImportError while importing test module '/Volumes/git/legotin/chess-gaze/tests/chess_gaze/test_frame_records.py'.
E   ImportError: cannot import name 'InferenceRuntimeRecord' from 'chess_gaze.frame_records'
```

Interpretation:

- RED was valid: the new test surface failed because `InferenceRuntimeRecord` did not exist yet.

## GREEN evidence

Narrow recheck after implementation:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_accepts_default_model_observer tests/chess_gaze/test_pipeline_contract.py::test_model_free_observer_run_manifest_records_external_observer -q
```

Output:

```text
..                                                                       [100%]
2 passed in 1.39s
```

Focused suite from the brief:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py -q
```

Sandboxed run relevant output:

```text
FAILED tests/chess_gaze/test_scene_viewer.py::test_static_server_serves_viewer_files
FAILED tests/chess_gaze/test_scene_viewer.py::test_static_server_does_not_escape_viewer_root
E   PermissionError: [Errno 1] Operation not permitted
```

Rerun unsandboxed for accurate verification because those viewer tests bind a loopback socket:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py -q
```

Output:

```text
....................................................................     [100%]
68 passed in 2.06s
```

## Files changed

- `src/chess_gaze/frame_records.py`
- `src/chess_gaze/pipeline.py`
- `tests/chess_gaze/test_frame_records.py`
- `tests/chess_gaze/test_pipeline_contract.py`
- `tests/chess_gaze/test_qa_summary.py`
- `tests/chess_gaze/test_scene_artifacts.py`
- `tests/chess_gaze/test_scene_viewer.py`

## Commit

- `56e3f85` - `feat: record inference runtime metadata`

## Self-review and concerns

- The change stays within the Task 2 file scope plus the requested report file.
- The runtime manifest now records truthful external-observer metadata for injected observers.
- `RunManifest` is stricter now, and the manifest-consuming fixture builders were updated to match.
- No Task 6 behavior was pulled forward.
- Concern: the brief’s focused suite requires unsandboxed execution in this environment because two existing viewer tests open a local server socket; the code itself passed once verified outside the sandbox restriction.

---

## Follow-up fix after review

### Reviewer findings addressed

- Fixed the false metadata bug in `analyze_video()`: injected observers still record `external_observer`, while default model-backed runs now record truthful `default_model_observer` runtime metadata.
- Added semantic validation for `InferenceRuntimeRecord` so contradictory cross-field combinations are rejected instead of silently accepted.
- Kept Task 6 out of scope: no MPS preflight implementation, no model loading/device changes, no batching changes beyond truthfully recording the current effective behavior.

### RED evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_rejects_default_model_observer_contradictions tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_rejects_external_observer_contradictions tests/chess_gaze/test_pipeline_contract.py::test_default_model_run_manifest_records_truthful_current_runtime -q
```

Relevant output:

```text
FAILED tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_rejects_default_model_observer_contradictions[overrides0]
E   Failed: DID NOT RAISE ValidationError

FAILED tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_rejects_external_observer_contradictions[overrides0]
E   Failed: DID NOT RAISE ValidationError

FAILED tests/chess_gaze/test_pipeline_contract.py::test_default_model_run_manifest_records_truthful_current_runtime
E   AssertionError: assert {'mps_fallback_env': 'not_applicable', ... 'observer_source': 'external_observer', ...} == {'mps_fallback_env': '1', ... 'observer_source': 'default_model_observer', ...}
```

Interpretation:

- RED confirmed both review findings: the schema accepted contradictory runtime records, and default model-backed runs still wrote external-observer metadata.

### GREEN evidence

Narrow rerun after implementation:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_rejects_default_model_observer_contradictions tests/chess_gaze/test_frame_records.py::test_inference_runtime_record_rejects_external_observer_contradictions tests/chess_gaze/test_pipeline_contract.py::test_default_model_run_manifest_records_truthful_current_runtime -q
```

Output:

```text
...............                                                          [100%]
15 passed in 3.50s
```

Focused Task 2 suite in sandbox:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py -q
```

Relevant output:

```text
FAILED tests/chess_gaze/test_scene_viewer.py::test_static_server_serves_viewer_files
FAILED tests/chess_gaze/test_scene_viewer.py::test_static_server_does_not_escape_viewer_root
E   PermissionError: [Errno 1] Operation not permitted
2 failed, 81 passed in 1.51s
```

Focused Task 2 suite rerun unsandboxed for accurate verification because those two viewer tests bind a loopback socket:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_viewer.py -q
```

Output:

```text
........................................................................ [ 86%]
...........                                                              [100%]
83 passed in 2.04s
```

### Files changed

- `src/chess_gaze/frame_records.py`
- `src/chess_gaze/pipeline.py`
- `tests/chess_gaze/test_frame_records.py`
- `tests/chess_gaze/test_pipeline_contract.py`
- `.superpowers/sdd/task-2-report.md`

### Commit

- `28d64f7` - `fix: make Task 2 inference runtime metadata truthful`

### Concerns

- The focused suite still needs unsandboxed execution in this environment because the two viewer server tests bind a loopback socket.
- `mps_preflight_passed=False` is an explicit record that no MPS preflight has been implemented or passed yet; that is intentional Task 6 deferral, not an MPS execution claim.
