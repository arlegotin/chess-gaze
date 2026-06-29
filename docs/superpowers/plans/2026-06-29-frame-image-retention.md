# Default Frame Image Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `chess-gaze analyze` avoid retaining raw and processed frame images by default, while preserving them when explicitly requested with `--save-frames` or `save_frame_images=True`.

**Architecture:** Persist a frame-image retention policy in `run_manifest.json`, resolve it from config/request/CLI, and enforce it at the two image-write boundaries in `pipeline.py`. `qa_summary.py` validates raw/processed frame counts from the persisted policy so completion seals stay truthful.

**Tech Stack:** Python 3.12, Pydantic 2.13.4, PyAV 17.1.0, pytest, uv, existing chess-gaze artifact schemas.

## Global Constraints

- Do not add a new third-party dependency.
- Default `chess-gaze analyze <video>` must retain zero raw frame PNG files and zero processed frame JPEG files.
- `--save-frames` must be the explicit CLI opt-in that retains raw and processed frame images.
- Programmatic callers opt in with `AnalyzeRequest(save_frame_images=True)`.
- JSON config may set `save_frame_images: true`; request/CLI overrides win over config.
- The run manifest must persist `frame_image_retention.schema_version="frame-image-retention-v1"` and `frame_image_retention.save_frame_images`.
- Legacy run manifests without `frame_image_retention` must read as `save_frame_images=true`.
- QA validation must expect raw/processed counts of `0` when `save_frame_images=false`, and decoded-frame count when `save_frame_images=true`.
- Resume compatibility must require matching frame-image retention policy.
- Do not remove or weaken frame records, scene records, viewer artifacts, model-runtime metadata, or analysis-state checkpoint behavior.
- Keep source layout focused; if `pipeline.py` crosses the 800-line review trigger, record a source-layout review or reduce the change.

---

## File Structure

- Modify `src/chess_gaze/frame_records.py`: add `FrameImageRetentionPolicy` and `RunManifest.frame_image_retention` with legacy default.
- Modify `src/chess_gaze/configuration.py`: add `save_frame_images` config and request override support.
- Modify `src/chess_gaze/analysis_resume.py`: persist policy on new runs and compare it during resumable-run discovery.
- Modify `src/chess_gaze/pipeline.py`: carry resolved policy and skip raw/processed image writes unless saving is enabled.
- Modify `src/chess_gaze/qa_summary.py`: validate raw/processed counts against policy.
- Modify `src/chess_gaze/cli.py`: add `--save-frames`.
- Modify focused tests in `tests/chess_gaze/test_configuration.py`, `tests/chess_gaze/test_cli.py`, `tests/chess_gaze/test_pipeline_contract.py`, `tests/chess_gaze/test_qa_summary.py`, and real-video contract tests.
- Update `README.md`, `docs/development/architecture/source-layout.md`, and add ADR/closeout docs.

---

### Task 1: Persist and Resolve Frame Image Retention Policy

**Files:**
- Modify: `src/chess_gaze/frame_records.py`
- Modify: `src/chess_gaze/configuration.py`
- Modify: `src/chess_gaze/analysis_resume.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/cli.py`
- Modify: `tests/chess_gaze/test_configuration.py`
- Modify: `tests/chess_gaze/test_cli.py`
- Modify: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Produces: `FrameImageRetentionPolicy(save_frame_images: bool, schema_version: Literal["frame-image-retention-v1"])`
- Produces: `AnalysisConfig.save_frame_images: bool`
- Produces: `AnalyzeRequest.save_frame_images: bool | None`
- Produces: `_ResolvedRequest.save_frame_images: bool`
- Produces: CLI flag `--save-frames`

- [ ] **Step 1: Write failing configuration and CLI tests**

Add assertions to `test_load_config_uses_task_4_defaults()`:

```python
assert config.save_frame_images is False
```

Add a config test:

```python
def test_load_config_accepts_save_frame_images(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"save_frame_images": true}', encoding="utf-8")

    config = load_config(config_path)

    assert config.save_frame_images is True
```

Add to `test_analyze_prints_run_dir_and_viewer_path()`:

```python
assert request.save_frame_images is None
```

Add a CLI forwarding test:

```python
def test_analyze_save_frames_flag_requests_frame_image_retention(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=run_dir / "viewer" / "index.html",
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path), "--save-frames"]) == 0

    [request] = captured_requests
    assert request.save_frame_images is True
```

Add a manifest assertion to a pipeline test:

```python
manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
assert manifest["frame_image_retention"] == {
    "schema_version": "frame-image-retention-v1",
    "save_frame_images": False,
}
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_configuration.py::test_load_config_uses_task_4_defaults tests/chess_gaze/test_configuration.py::test_load_config_accepts_save_frame_images tests/chess_gaze/test_cli.py::test_analyze_save_frames_flag_requests_frame_image_retention -q
```

Expected: FAIL because the config field, request field, and CLI flag do not exist.

- [ ] **Step 3: Implement schema/config/request/CLI policy**

Add `FrameImageRetentionPolicy` to `src/chess_gaze/frame_records.py`:

```python
class FrameImageRetentionPolicy(StrictSchemaModel):
    schema_version: Literal["frame-image-retention-v1"] = (
        "frame-image-retention-v1"
    )
    save_frame_images: bool
```

Add to `RunManifest`:

```python
frame_image_retention: FrameImageRetentionPolicy = Field(
    default_factory=lambda: FrameImageRetentionPolicy(save_frame_images=True)
)
```

Import `Field` from `pydantic`.

Add to `AnalysisConfig`:

```python
save_frame_images: bool = False
```

Add `save_frame_images: bool | None = None` to `apply_analysis_overrides()` and
only place it into the payload when not `None`.

Add `save_frame_images: bool | None = None` to `AnalyzeRequest`, and
`save_frame_images: bool` to `_ResolvedRequest`. Pass it from
`apply_analysis_overrides()`.

In `analysis_resume.write_initial_run_artifacts()`, add a required
`frame_image_retention: FrameImageRetentionPolicy` parameter and write it into
`RunManifest`.

In `pipeline.analyze_video()`, pass
`FrameImageRetentionPolicy(save_frame_images=resolved.save_frame_images)` to
`write_initial_run_artifacts()`.

In `cli.build_parser()`, add:

```python
analyze.add_argument(
    "--save-frames",
    action="store_true",
    default=None,
    dest="save_frame_images",
    help="retain raw decoded PNGs and processed overlay JPEGs",
)
```

Pass `save_frame_images=args.save_frame_images` into `AnalyzeRequest`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py -q
```

Expected: PASS.

---

### Task 2: Enforce Policy in Pipeline and QA

**Files:**
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/qa_summary.py`
- Modify: `src/chess_gaze/analysis_resume.py`
- Modify: `tests/chess_gaze/test_pipeline_contract.py`
- Modify: `tests/chess_gaze/test_qa_summary.py`
- Modify: `tests/chess_gaze/test_pipeline_real_video_contract.py`
- Modify: `tests/chess_gaze/test_qa_summary_real_video_contract.py`

**Interfaces:**
- Consumes: `_ResolvedRequest.save_frame_images`
- Consumes: `RunManifest.frame_image_retention.save_frame_images`
- Produces: default runs with empty `raw_frames/` and `processed_frames/`
- Produces: explicit-save runs with one raw and one processed image per decoded frame

- [ ] **Step 1: Write failing default/no-save pipeline regression**

Replace the old default retained-frame assertion in
`tests/chess_gaze/test_pipeline_contract.py` with:

```python
def test_analyze_video_does_not_retain_raw_or_processed_frame_images_by_default(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=4)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.decoded_frame_count == 4
    assert list(result.layout.raw_frames_dir.glob("*.png")) == []
    assert list(result.layout.processed_frames_dir.glob("*.jpg")) == []
    records = _records_from(result.frames_jsonl_path)
    assert len(records) == 4
    summary = QASummary.model_validate_json(
        result.qa_summary_path.read_text(encoding="utf-8")
    )
    assert summary.counts.raw_frames == 0
    assert summary.counts.processed_frames == 0
    assert summary.artifact_validation.counts_match is True
    assert summary.final_status == "complete"
```

- [ ] **Step 2: Write failing explicit-save pipeline regression**

Add:

```python
def test_analyze_video_retains_raw_and_processed_frame_images_when_requested(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=3)

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            save_frame_images=True,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert len(list(result.layout.raw_frames_dir.glob("*.png"))) == 3
    assert len(list(result.layout.processed_frames_dir.glob("*.jpg"))) == 3
    summary = QASummary.model_validate_json(
        result.qa_summary_path.read_text(encoding="utf-8")
    )
    assert summary.counts.raw_frames == 3
    assert summary.counts.processed_frames == 3
    assert summary.artifact_validation.counts_match is True
```

- [ ] **Step 3: Write failing QA policy regression**

Add to `tests/chess_gaze/test_qa_summary.py`:

```python
def test_validate_run_artifacts_accepts_unretained_frame_images_when_policy_disables_saving(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(
        tmp_path,
        frame_count=3,
        save_frame_images=False,
        write_frame_images=False,
    )

    result = validate_run_artifacts(layout)
    summary = build_qa_summary(layout)

    assert result.counts.raw_frames == 0
    assert result.counts.processed_frames == 0
    assert result.counts_match is True
    assert summary.final_status == "complete"
```

Add:

```python
def test_validate_run_artifacts_rejects_stray_frame_images_when_policy_disables_saving(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(
        tmp_path,
        frame_count=2,
        save_frame_images=False,
        write_frame_images=False,
    )
    (layout.raw_frames_dir / "f000000000.png").write_bytes(b"stray")

    result = validate_run_artifacts(layout)

    assert result.counts_match is False
    assert result.final_status == "failed"
    assert result.validation_errors == [
        "raw frame count does not match frame image retention policy: 1 != 0"
    ]
```

- [ ] **Step 4: Run tests to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py -k 'does_not_retain or retains_raw' -q
uv run pytest tests/chess_gaze/test_qa_summary.py -k 'retention_policy or unretained_frame_images or stray_frame_images' -q
```

Expected: FAIL because the pipeline still writes frames and QA still requires decoded-frame counts.

- [ ] **Step 5: Implement pipeline and QA policy behavior**

In `_prepare_decoded_frame()`, wrap only the raw image write:

```python
if resolved.save_frame_images:
    raw_path = layout.raw_frames_dir / f"{decoded_frame.frame_id}.png"
    try:
        _validate_image_format(resolved.raw_frame_image_format, "png")
        raw_frame_writer(raw_path, decoded_frame.rgb)
    except Exception as exc:
        frame_errors.append(...)
```

Always build and return the same `ObserverFrame`.

In `_render_processed_frame_and_collect_errors()`, return `(record, [])` before
constructing/writing the JPEG if `not resolved.save_frame_images`.

In `qa_summary._count_validation_errors()`, accept `frame_image_retention` and
compute:

```python
expected_frame_images = counts.decoded_frames if save_frame_images else 0
```

Validate raw and processed counts against `expected_frame_images`. Include
policy-specific error messages.

In `_validate_loaded_run_artifacts()`, pass
`loaded.run_manifest.frame_image_retention` to `_count_validation_errors()`.

In `analysis_resume.find_latest_resumable_run()` and `_run_matches()`, require
matching `FrameImageRetentionPolicy`.

- [ ] **Step 6: Adjust affected existing tests**

Update pipeline tests whose purpose is not frame retention to expect default
zero frame images or to opt in with `save_frame_images=True` when they test
image write errors.

Required explicit-save changes:

- `test_raw_frame_write_failure_records_partial_status_and_error_evidence`
- `test_processed_frame_write_failure_records_error_evidence`

Required default-no-save changes:

- batched pipeline default count assertions;
- no-face fake observer default processed-frame assertion;
- real-video model-free pipeline and QA default contracts.

- [ ] **Step 7: Run focused GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_analysis_resume.py -q
```

Expected: PASS.

---

### Task 3: Documentation, ADR, Closeout, and Broad Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/development/architecture/source-layout.md`
- Add: `docs/development/decisions/0004-default-frame-image-retention.md`
- Add: `docs/superpowers/closeouts/2026-06-29-frame-image-retention.md`

**Interfaces:**
- Produces: durable documentation that current default runs do not retain frame images.
- Produces: closeout with root cause, changed boundary, tests, and residual risk.

- [ ] **Step 1: Update README behavior text**

Change the README introduction from preserving raw/processed frame evidence by
default to strict records plus optional frame image retention. Add `--save-frames`
to useful options. In the completed-run artifact list, describe `raw_frames/` and
`processed_frames/` as empty by default and populated only with `--save-frames`.

- [ ] **Step 2: Update source layout**

In `docs/development/architecture/source-layout.md`, update the `pipeline.py`,
`qa_summary.py`, and `frame_records.py` ownership bullets to mention persisted
frame-image retention policy and policy-aware QA validation.

- [ ] **Step 3: Add ADR**

Create `docs/development/decisions/0004-default-frame-image-retention.md` with:

- context: raw/processed frames dominate disk use and are derived/debug outputs;
- decision: default no retention, explicit opt-in, persisted policy;
- alternatives: end-of-run deletion, cleanup command only, persisted write gate;
- consequences: lower default disk usage, empty blur/exposure QA samples unless
  frames are saved, legacy manifests treated as saved-frame runs;
- verification: tests and run manifest inspection.

- [ ] **Step 4: Run broad verification**

Run:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: PASS for all commands.

- [ ] **Step 5: Write closeout**

Create `docs/superpowers/closeouts/2026-06-29-frame-image-retention.md` with:

- root cause: raw/processed frame image retention was an unconditional pipeline side effect and QA invariant;
- durable surface changed: persisted run policy, pipeline write gates, QA count validation, CLI/config request;
- regression tests added;
- broad verification output;
- residual risk: no raw-frame blur/exposure ranking exists unless frames are explicitly saved.

