# CLI Progress And Native Log Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trustworthy `analyze` progress and suppress known dependency-owned native noise without hiding real chess-gaze failures.

**Architecture:** Keep progress as a pipeline callback emitted after durable frame commits. Keep CLI rendering in `cli.py` and write progress to stderr. Add a small native stderr filter module that suppresses only known PyAV/OpenCV/MediaPipe/TFLite chatter during analysis, while progress writes to the original stderr file descriptor.

**Tech Stack:** Python 3.12, argparse, tqdm 4.68.3, PyAV 17.1.0, OpenCV Python headless 4.13.0.92, MediaPipe 0.10.35, PyTorch/UniGaze.

## Global Constraints

- Do not change frame decoding, model semantics, artifact schemas, scene geometry, or run layout.
- Stdout for successful `chess-gaze analyze` remains exactly the run directory and `viewer: <path>`.
- Progress output goes to stderr and updates only after committed frame records plus `analysis_state.json` advancement.
- Known native dependency chatter is suppressible; repo-owned errors and unknown dependency stderr remain visible.
- `artifacts/input/nakamura_short.mp4` is the required real verification input.
- Add `tqdm` as a direct runtime dependency.

---

### Task 1: Pipeline Progress Contract

**Files:**
- Modify: `src/chess_gaze/pipeline.py`
- Test: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Produces: `AnalysisProgressEvent(run_dir: Path, completed_frames: int, total_frames: int)` and `AnalyzeRequest.progress_callback: Callable[[AnalysisProgressEvent], None] | None`.
- Emits: one initial event after run layout is selected and `analysis_state.json` is written; one event after each durable commit/checkpoint.

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
def test_analyze_video_reports_committed_progress_after_each_batch(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=5)
    events = []

    def progress_callback(event: AnalysisProgressEvent) -> None:
        events.append((event.completed_frames, event.total_frames))

    analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            unigaze_batch_size=2,
            progress_callback=progress_callback,
        ),
        observers=ObserverBundle(
            frame_observer=_fake_record,
            frame_batch_observer=lambda frames: [_fake_record(frame) for frame in frames],
        ),
    )

    assert events == [(0, 5), (2, 5), (4, 5), (5, 5)]
```

Also assert resumed runs start at the committed count:

```python
assert resumed_events[0] == (2, 5)
```

- [ ] **Step 2: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_reports_committed_progress_after_each_batch -q
```

Expected: import/name failure for `AnalysisProgressEvent` or missing callback behavior.

- [ ] **Step 3: Implement**

Add the dataclass and callback field, call the callback after the initial
state write, and call it after each post-commit `write_analysis_state()`.

- [ ] **Step 4: Run GREEN**

Run the focused pipeline progress tests and then:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_pipeline_contract.py -q
```

- [ ] **Step 5: Commit**

```sh
git add src/chess_gaze/pipeline.py tests/chess_gaze/test_pipeline_contract.py
git commit -m "feat: report committed analysis progress"
```

### Task 2: CLI Progress Rendering And Lazy Imports

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/chess_gaze/cli.py`
- Test: `tests/chess_gaze/test_cli.py`

**Interfaces:**
- Consumes: `AnalysisProgressEvent` and `AnalyzeRequest.progress_callback`.
- Produces: `--progress {auto,on,off}` and a tqdm-backed stderr progress bar.

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
assert main(["analyze", str(video_path), "--progress", "off"]) == 0
assert captured_request.progress_callback is None
```

```python
assert main(["analyze", str(video_path), "--progress", "on"]) == 0
assert captured_request.progress_callback is not None
```

Add lazy-import protections by monkeypatching `chess_gaze.cli.analyze_video`
absence no longer being required at module import; the fake analyze can be
installed by monkeypatching the lazily imported function seam.

- [ ] **Step 2: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py -q
```

Expected: `--progress` is unrecognized and callback assertions fail.

- [ ] **Step 3: Implement**

Add direct `tqdm>=4.68.3` dependency. Move analyze-only imports inside the
analyze branch or a helper. Build a progress callback that creates `tqdm` with
`total`, `initial`, `unit="frame"`, `dynamic_ncols=True`, and `file` pointing to
the original stderr stream when native filtering is active.

- [ ] **Step 4: Run GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py -q
UV_CACHE_DIR=.uv-cache uv lock --check
```

- [ ] **Step 5: Commit**

```sh
git add pyproject.toml uv.lock src/chess_gaze/cli.py tests/chess_gaze/test_cli.py
git commit -m "feat: show analyze progress on stderr"
```

### Task 3: Native Log Filter And UniGaze Stdout Suppression

**Files:**
- Create: `src/chess_gaze/native_log_filter.py`
- Modify: `src/chess_gaze/cli.py`
- Modify: `src/chess_gaze/gaze_observation.py`
- Create: `tests/chess_gaze/test_native_log_filter.py`
- Modify: `tests/chess_gaze/test_gaze_observation.py`

**Interfaces:**
- Produces: `suppress_known_native_analysis_logs()` context manager with `stderr`
  stream for progress.
- Produces: `suppress_stdout` around `load_unigaze_weights()`.

- [ ] **Step 1: Write failing tests**

Test the line filter directly:

```python
assert _should_suppress_native_stderr_line("objc[1]: Class AVFFrameReceiver ...")
assert _should_suppress_native_stderr_line("INFO: Created TensorFlow Lite XNNPACK delegate for CPU.")
assert not _should_suppress_native_stderr_line("real error")
```

Test a Clearcut block suppresses the source trace lines after the first
Clearcut line.

Test `UniGazeModel.from_local_asset()` does not leak a backend
`load_unigaze_weights()` print to stdout.

- [ ] **Step 2: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_native_log_filter.py tests/chess_gaze/test_gaze_observation.py::test_from_local_asset_suppresses_backend_weight_load_stdout -q
```

Expected: missing module/test failure.

- [ ] **Step 3: Implement**

Implement a fd-2 filter that saves the original stderr fd, redirects native
stderr through a pipe, forwards non-suppressed bytes to the original fd, and
exposes an original-stderr text stream for `tqdm`. Keep the filter active only
around `analyze_video()`.

Wrap only `backend.load_unigaze_weights(str(asset.resolved_path))` with
`contextlib.redirect_stdout(io.StringIO())`.

- [ ] **Step 4: Run GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_native_log_filter.py tests/chess_gaze/test_gaze_observation.py -q
```

- [ ] **Step 5: Commit**

```sh
git add src/chess_gaze/native_log_filter.py src/chess_gaze/cli.py src/chess_gaze/gaze_observation.py tests/chess_gaze/test_native_log_filter.py tests/chess_gaze/test_gaze_observation.py
git commit -m "fix: filter known native analysis log noise"
```

### Task 4: Real Verification And Closeout

**Files:**
- Create: `docs/superpowers/closeouts/2026-06-29-cli-progress-log-hygiene.md`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verification evidence and residual-risk record.

- [ ] **Step 1: Run focused gates**

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_native_log_filter.py tests/chess_gaze/test_gaze_observation.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

- [ ] **Step 2: Run required real-video verification**

```sh
UV_CACHE_DIR=.uv-cache env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 \
  --output-root /private/tmp/chess-gaze-progress-after \
  --models-root models \
  --unigaze-device mps \
  --unigaze-batch-size 7 \
  --save-frames \
  --no-resume \
  --progress on
```

Verify:

- stderr contains a progress bar and no known AVFoundation/MediaPipe/Clearcut noise;
- stdout contains only run dir and viewer path;
- `qa_summary.final_status == "complete"`;
- `counts.decoded_frames == counts.frame_records == counts.scene_frame_records == 180`;
- representative processed frames are visually coherent.

- [ ] **Step 3: Write closeout and commit**

```sh
git add docs/superpowers/closeouts/2026-06-29-cli-progress-log-hygiene.md
git commit -m "docs: close out cli progress log hygiene"
```
