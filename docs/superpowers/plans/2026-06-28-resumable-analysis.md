# Resumable Analysis Implementation Plan

Supersession note, 2026-06-29: this plan's raw/processed frame image count
examples are superseded by
`docs/superpowers/specs/2026-06-29-frame-image-retention-design.md` and
ADR-0004. Current default runs validate zero raw and processed frame image
files; explicit save-frame runs validate decoded-frame-count image files.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `chess-gaze analyze <video>` resume the latest compatible interrupted run by default, while preserving fresh-run behavior for completed runs and `--no-resume`.

**Architecture:** Add a focused `analysis_resume.py` module that owns compatible-run discovery, committed-prefix repair, checkpoint state, and derived-artifact cleanup. Keep `pipeline.py` as orchestrator: resolve/validate inputs, choose existing-or-new layout, skip committed decoded frames before inference, then rebuild whole-run scene/viewer/QA artifacts.

**Tech Stack:** Python 3.12, PyAV 17.1.0, Pydantic 2.13.4, stdlib filesystem primitives, existing chess-gaze artifact schemas.

## Global Constraints

- Do not add a new third-party dependency.
- Default `chess-gaze analyze <video>` must resume the newest compatible incomplete run under `<output-root>/<video-stem>/runs/`.
- `--no-resume` must force a new run directory.
- `qa_summary.json` with `final_status=complete`, `schema_validation_passed=true`, and `counts_match=true` is the completion seal; run directory existence and `run_manifest.json` are not completion signals.
- Resume compatibility requires matching input path string, source SHA256, decoded frame count, frame dimensions, inference runtime record, and calibration record.
- `records/frames.jsonl` is the committed frame journal; raw frame counts, processed frame counts, crop counts, and `analysis_state.json` are not authoritative.
- Resume must repair `frames.jsonl` to its valid contiguous leading prefix and rebuild `errors.jsonl` from committed frame records before appending.
- Resume must clear whole-run derived artifacts before final rebuild: `records/scene_frames.jsonl`, `scene/`, `viewer/`, and `qa_summary.json`.
- Resume must not use direct frame-index seeking; decode from zero and discard committed frames before observer/model inference.
- Use `artifacts/input/nakamura_short.mp4` for real interruption/resume verification.

---

## File Structure

- Create `src/chess_gaze/analysis_resume.py`: run discovery, compatibility checks, frame-journal prefix recovery, checkpoint state, cleanup of uncommitted/derived artifacts, and JSONL fsync helpers.
- Modify `src/chess_gaze/artifact_runs.py`: add `run_layout_from_dir()` so existing run directories can be addressed with the same `RunLayout` contract as new runs.
- Modify `src/chess_gaze/pipeline.py`: add `AnalyzeRequest.resume`, select resume layout or fresh layout, skip committed decoded frames, flush checkpoints after committed batches, and write final analysis state.
- Modify `src/chess_gaze/cli.py`: add `--no-resume` and pass `resume` into `AnalyzeRequest`.
- Modify `README.md` and `docs/development/architecture/source-layout.md`: document default resume behavior and the new module.
- Add tests in `tests/chess_gaze/test_analysis_resume.py`, `tests/chess_gaze/test_pipeline_contract.py`, and `tests/chess_gaze/test_cli.py`.
- Add closeout `docs/superpowers/closeouts/2026-06-28-resumable-analysis.md`.

---

### Task 1: Resume Layout and Recovery Utilities

**Files:**
- Modify: `src/chess_gaze/artifact_runs.py`
- Create: `src/chess_gaze/analysis_resume.py`
- Create: `tests/chess_gaze/test_analysis_resume.py`
- Modify: `tests/chess_gaze/test_artifact_runs.py`

**Interfaces:**
- Consumes: existing `RunLayout`, `frame_id()`, `RunManifest`, `VideoManifest`, `CalibrationRecord`, `FrameRecord`, `InferenceRuntimeRecord`, `QASummary`, and `atomic_write_bytes()`.
- Produces:
  - `run_layout_from_dir(run_dir: Path) -> RunLayout`
  - `AnalysisState` Pydantic model
  - `find_latest_resumable_run(runs_root: Path, video_path: Path, video_manifest: VideoManifest, calibration: CalibrationRecord, inference: InferenceRuntimeRecord) -> RunLayout | None`
  - `prepare_resume_run(layout: RunLayout, video_manifest: VideoManifest, *, clock: Callable[[], datetime]) -> ResumePreparation`
  - `write_analysis_state(layout: RunLayout, state: AnalysisState) -> Path`
  - `flush_jsonl_checkpoint(*handles: TextIO) -> None`

- [ ] **Step 1: Write failing run-layout test**

Add to `tests/chess_gaze/test_artifact_runs.py`:

```python
def test_run_layout_from_existing_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "clip" / "runs" / "run-1"
    layout = run_layout_from_dir(run_dir)

    assert layout.run_dir == run_dir
    assert layout.raw_frames_dir == run_dir / "raw_frames"
    assert layout.processed_frames_dir == run_dir / "processed_frames"
    assert layout.face_crops_dir == run_dir / "crops" / "face"
    assert layout.left_eye_crops_dir == run_dir / "crops" / "eyes" / "left"
    assert layout.right_eye_crops_dir == run_dir / "crops" / "eyes" / "right"
    assert layout.records_dir == run_dir / "records"
    assert layout.scene_dir == run_dir / "scene"
    assert layout.viewer_dir == run_dir / "viewer"
```

Update imports:

```python
from chess_gaze.artifact_runs import create_run_layout, frame_id, run_layout_from_dir
```

- [ ] **Step 2: Run test to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_artifact_runs.py::test_run_layout_from_existing_run_dir -q
```

Expected: FAIL because `run_layout_from_dir` is not defined.

- [ ] **Step 3: Implement existing layout helper**

Add to `src/chess_gaze/artifact_runs.py`:

```python
def run_layout_from_dir(run_dir: Path) -> RunLayout:
    crops_dir = run_dir / "crops"
    eyes_crops_dir = crops_dir / "eyes"
    return RunLayout(
        run_dir=run_dir,
        raw_frames_dir=run_dir / "raw_frames",
        processed_frames_dir=run_dir / "processed_frames",
        crops_dir=crops_dir,
        face_crops_dir=crops_dir / "face",
        eyes_crops_dir=eyes_crops_dir,
        left_eye_crops_dir=eyes_crops_dir / "left",
        right_eye_crops_dir=eyes_crops_dir / "right",
        records_dir=run_dir / "records",
    )
```

- [ ] **Step 4: Run layout test to verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_artifact_runs.py::test_run_layout_from_existing_run_dir -q
```

Expected: PASS.

- [ ] **Step 5: Write failing recovery tests**

Create `tests/chess_gaze/test_analysis_resume.py` with fixtures that build a
minimal compatible run. Include these tests:

```python
def test_prepare_resume_run_repairs_committed_prefix_and_rebuilds_errors(
    tmp_path: Path,
) -> None:
    layout = _make_resume_layout(tmp_path, frame_count=4)
    records = [_fake_frame_record(0), _fake_frame_record(1, with_error=True)]
    (layout.records_dir / "frames.jsonl").write_text(
        records[0].model_dump_json()
        + "\n"
        + records[1].model_dump_json()
        + "\n"
        + '{"frame_id":"f000000999","frame_index":999}\n',
        encoding="utf-8",
    )
    (layout.records_dir / "errors.jsonl").write_text(
        '{"frame_id":"f000000002","frame_index":2,"code":"FACE_NOT_FOUND","message":"stale"}\n',
        encoding="utf-8",
    )
    (layout.raw_frames_dir / "f000000000.png").write_bytes(b"raw0")
    (layout.raw_frames_dir / "f000000002.png").write_bytes(b"raw2")
    (layout.processed_frames_dir / "f000000001.jpg").write_bytes(b"processed1")
    (layout.processed_frames_dir / "f000000002.jpg").write_bytes(b"processed2")
    (layout.left_eye_crops_dir / "f000000002.png").write_bytes(b"crop")
    (layout.records_dir / "scene_frames.jsonl").write_text("stale\n", encoding="utf-8")
    (layout.scene_dir / "scene_manifest.json").write_text("{}", encoding="utf-8")
    (layout.viewer_dir / "index.html").write_text("stale", encoding="utf-8")
    (layout.run_dir / "qa_summary.json").write_text("{}", encoding="utf-8")

    preparation = prepare_resume_run(
        layout,
        _video_manifest(frame_count=4),
        clock=lambda: datetime(2026, 6, 28, 12, 0, tzinfo=UTC),
    )

    assert preparation.next_frame_index == 2
    assert [record.frame_index for record in preparation.committed_records] == [0, 1]
    repaired_lines = (layout.records_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(repaired_lines) == 2
    error_lines = (layout.records_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(error_lines) == 1
    assert json.loads(error_lines[0])["frame_id"] == "f000000001"
    assert not (layout.raw_frames_dir / "f000000002.png").exists()
    assert not (layout.processed_frames_dir / "f000000002.jpg").exists()
    assert not (layout.left_eye_crops_dir / "f000000002.png").exists()
    assert not (layout.records_dir / "scene_frames.jsonl").exists()
    assert not (layout.scene_dir / "scene_manifest.json").exists()
    assert not (layout.viewer_dir / "index.html").exists()
    assert not (layout.run_dir / "qa_summary.json").exists()
```

Also include:

```python
def test_find_latest_resumable_run_ignores_complete_and_incompatible_runs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "output" / "clip" / "runs"
    compatible = _make_compatible_run(runs_root / "20260628T100000Z-good")
    complete = _make_compatible_run(runs_root / "20260628T110000Z-complete")
    incompatible = _make_compatible_run(
        runs_root / "20260628T120000Z-wrong",
        source_sha256="wrong",
    )
    _write_complete_qa_summary(complete)

    result = find_latest_resumable_run(
        runs_root,
        Path("artifacts/input/clip.mp4"),
        _video_manifest(frame_count=4),
        default_calibration(),
        external_observer_inference_record(),
    )

    assert result == compatible
```

- [ ] **Step 6: Run recovery tests to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_analysis_resume.py -q
```

Expected: FAIL because `analysis_resume.py` does not exist.

- [ ] **Step 7: Implement recovery module**

Implement `analysis_resume.py` with:

```python
class AnalysisState(StrictSchemaModel):
    schema_version: Literal["analysis-state-v1"] = "analysis-state-v1"
    run_id: str
    input_path: str
    source_video_sha256: str
    frame_count_decoded: int
    next_frame_index: int
    status: Literal["processing", "complete", "failed"]
    updated_at_utc: str
```

Use `FrameRecord.model_validate_json()` and
`FrameErrorRecord(...).model_dump_json()` when repairing JSONL files. Use
`atomic_write_bytes()` for rewritten JSON/JSONL artifacts. Use `os.fsync()` in
`flush_jsonl_checkpoint()`. Delete only files under the selected run layout.

- [ ] **Step 8: Run focused tests to verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_artifact_runs.py::test_run_layout_from_existing_run_dir tests/chess_gaze/test_analysis_resume.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```sh
git add src/chess_gaze/artifact_runs.py src/chess_gaze/analysis_resume.py tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_analysis_resume.py
git commit -m "feat: add analysis resume recovery utilities"
```

---

### Task 2: Pipeline Resume Orchestration

**Files:**
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Consumes Task 1 functions.
- Produces `AnalyzeRequest.resume: bool = True`.
- Preserves existing `AnalyzeResult` fields and adds `analysis_state_path: Path`.

- [ ] **Step 1: Write failing interrupted-run regression**

Add to `tests/chess_gaze/test_pipeline_contract.py`:

```python
def test_analyze_video_resumes_latest_compatible_partial_run(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=5)
    observed_frames: list[int] = []

    def interrupt_after_two(frame: ObserverFrame) -> FrameRecord:
        observed_frames.append(frame.frame_index)
        if frame.frame_index == 2:
            raise RuntimeError("simulated interruption")
        return _fake_record(frame)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        analyze_video(
            AnalyzeRequest(video_path=video_path, output_root=output_root),
            observers=ObserverBundle(frame_observer=interrupt_after_two),
        )

    [run_dir] = (output_root / "tiny" / "runs").iterdir()
    assert observed_frames == [0, 1, 2]
    assert len(_records_from(run_dir / "records" / "frames.jsonl")) == 2

    resumed_observed_frames: list[int] = []

    def resumed_observer(frame: ObserverFrame) -> FrameRecord:
        resumed_observed_frames.append(frame.frame_index)
        return _fake_record(frame)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=output_root),
        observers=ObserverBundle(frame_observer=resumed_observer),
    )

    assert result.layout.run_dir == run_dir
    assert resumed_observed_frames == [2, 3, 4]
    assert [record.frame_index for record in _records_from(result.frames_jsonl_path)] == [0, 1, 2, 3, 4]
    summary = QASummary.model_validate_json(result.qa_summary_path.read_text(encoding="utf-8"))
    assert summary.final_status == "complete"
    assert summary.counts.frame_records == 5
```

- [ ] **Step 2: Write failing completed-run regression**

Add:

```python
def test_analyze_video_does_not_resume_complete_run(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=2)

    first = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=output_root),
        observers=ObserverBundle(frame_observer=_fake_record),
    )
    second = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=output_root),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert second.layout.run_dir != first.layout.run_dir
```

- [ ] **Step 3: Run pipeline tests to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_resumes_latest_compatible_partial_run tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_does_not_resume_complete_run -q
```

Expected: first test FAILS because the second call creates a fresh run.

- [ ] **Step 4: Implement pipeline integration**

Modify `AnalyzeRequest`:

```python
resume: bool = True
```

Modify `AnalyzeResult`:

```python
analysis_state_path: Path
```

In `analyze_video()`:

- after inspection, calibration, and inference are resolved, call
  `find_latest_resumable_run()` when `request.resume` is true;
- if a layout is returned, call `prepare_resume_run()` and initialize
  `resume_next_frame_index` and committed error count from its records;
- otherwise create a new layout and write initial manifests as today;
- write `analysis_state.json` with `status="processing"`;
- in the decode loop, skip frames whose `frame_index < resume_next_frame_index`
  before observer/model inference;
- after each committed frame or batch, call `flush_jsonl_checkpoint()` and
  update `analysis_state.json` to the next frame index;
- after QA summary is written, update `analysis_state.json` to `complete` or
  `failed` based on `qa_summary.final_status`.

- [ ] **Step 5: Run pipeline tests to verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_resumes_latest_compatible_partial_run tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_does_not_resume_complete_run -q
```

Expected: PASS.

- [ ] **Step 6: Run focused pipeline suite**

Run:

```sh
uv run pytest tests/chess_gaze/test_analysis_resume.py tests/chess_gaze/test_pipeline_contract.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```sh
git add src/chess_gaze/pipeline.py tests/chess_gaze/test_pipeline_contract.py
git commit -m "feat: resume interrupted analysis runs"
```

---

### Task 3: CLI and Documentation

**Files:**
- Modify: `src/chess_gaze/cli.py`
- Modify: `tests/chess_gaze/test_cli.py`
- Modify: `README.md`
- Modify: `docs/development/architecture/source-layout.md`

**Interfaces:**
- Consumes `AnalyzeRequest.resume`.
- Produces `--no-resume` CLI override.

- [ ] **Step 1: Write failing CLI tests**

Add to `tests/chess_gaze/test_cli.py`:

```python
def test_analyze_enables_resume_by_default(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    viewer_index_path = run_dir / "viewer" / "index.html"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=viewer_index_path,
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path)]) == 0
    capsys.readouterr()
    assert captured_requests[0].resume is True
```

Add:

```python
def test_analyze_no_resume_forces_fresh_run_request(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    viewer_index_path = run_dir / "viewer" / "index.html"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=viewer_index_path,
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path), "--no-resume"]) == 0
    capsys.readouterr()
    assert captured_requests[0].resume is False
```

- [ ] **Step 2: Run CLI tests to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_cli.py::test_analyze_enables_resume_by_default tests/chess_gaze/test_cli.py::test_analyze_no_resume_forces_fresh_run_request -q
```

Expected: FAIL because `AnalyzeRequest` has no resume field or CLI flag yet.

- [ ] **Step 3: Implement CLI flag**

In `build_parser()`:

```python
analyze.add_argument(
    "--no-resume",
    action="store_false",
    dest="resume",
    default=True,
    help="start a new run instead of resuming a compatible interrupted run",
)
```

Pass `resume=args.resume` into `AnalyzeRequest`.

- [ ] **Step 4: Update docs**

In `README.md`, add an Analyze subsection:

```markdown
By default, rerunning the same analyze command resumes the newest compatible
incomplete run under the video's output directory. Completed runs still remain
immutable; rerunning after a completed `qa_summary.json` creates a new run.
Use `--no-resume` to force a fresh run even when a compatible partial run
exists.
```

Add `analysis_state.json` to the completed run artifact list.

In `docs/development/architecture/source-layout.md`, add:

```markdown
- `analysis_resume.py` owns interrupted-run discovery, committed frame journal
  repair, resume checkpoint state, and cleanup of derived artifacts before a
  resumed run rebuilds them.
```

- [ ] **Step 5: Run CLI tests to verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_cli.py::test_analyze_enables_resume_by_default tests/chess_gaze/test_cli.py::test_analyze_no_resume_forces_fresh_run_request -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```sh
git add src/chess_gaze/cli.py tests/chess_gaze/test_cli.py README.md docs/development/architecture/source-layout.md
git commit -m "docs: document resumable analyze runs"
```

---

### Task 4: Real Verification, Review, and Closeout

**Files:**
- Create: `docs/superpowers/closeouts/2026-06-28-resumable-analysis.md`

**Interfaces:**
- Consumes the completed implementation.
- Produces verification evidence and final review findings.

- [ ] **Step 1: Run focused tests**

Run:

```sh
uv run pytest tests/chess_gaze/test_analysis_resume.py tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_pipeline_contract.py -q
```

Expected: PASS.

- [ ] **Step 2: Run static gates**

Run:

```sh
uv run ruff format --check src tests
uv run ruff check .
uv run mypy
```

Expected: PASS.

- [ ] **Step 3: Verify real resume on `nakamura_short.mp4`**

Use the real video and real model path required by the user:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --output-root artifacts/output --models-root models
```

Interrupt the first run after frame records are present if it does not stop
naturally through a test hook. Then run the exact same command again. Verify the
same run directory completes.

Inspect final QA:

```sh
uv run python - <<'PY'
from pathlib import Path
from chess_gaze.qa_summary import QASummary
run_dir = sorted((Path("artifacts/output") / "nakamura_short" / "runs").iterdir())[-1]
summary = QASummary.model_validate_json((run_dir / "qa_summary.json").read_text())
print(run_dir)
print(summary.final_status)
print(summary.counts.model_dump())
print(summary.artifact_validation.schema_validation_passed)
print(summary.artifact_validation.counts_match)
PY
```

Expected:

```text
complete
decoded_frames=180
frame_records=180
raw_frames=180
processed_frames=180
scene_frame_records=180
schema_validation_passed=True
counts_match=True
```

- [ ] **Step 4: Run broader tests**

Run:

```sh
uv run pytest -q
```

If local real-video tests requiring unavailable legacy inputs fail, record the
exact failures and run the broad non-missing-input subset.

- [ ] **Step 5: Request final code review**

Dispatch a reviewer subagent using the full branch diff from the pre-feature
base to `HEAD`. Fix Critical and Important findings before closeout.

- [ ] **Step 6: Write closeout**

Create `docs/superpowers/closeouts/2026-06-28-resumable-analysis.md` with:

- request summary;
- root cause;
- durable surface changed;
- tests and gates run with exact results;
- `nakamura_short.mp4` interruption/resume evidence;
- residual risk.

- [ ] **Step 7: Commit Task 4**

```sh
git add docs/superpowers/closeouts/2026-06-28-resumable-analysis.md
git commit -m "docs: close out resumable analysis"
```
