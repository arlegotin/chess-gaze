# Optional QA Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `chess-gaze analyze` skip QA summary generation by default and generate `qa_summary.json` only when `--qa-summary` or `AnalyzeRequest(generate_qa_summary=True)` is used.

**Architecture:** Persist the QA closeout policy in `run_manifest.json`, defaulting legacy manifests to QA-required so old complete-without-QA runs are not reclassified. Default successful runs complete through `analysis_state.json` after scene and viewer artifacts exist; QA-requested runs keep the existing strict `revalidating` and `qa_summary.json` closeout path. Downstream tools that require QA evidence explicitly request or require it.

**Tech Stack:** Python 3.12 dataclasses, Pydantic strict artifact models, argparse CLI, pytest, existing repo-local pipeline/resume/QA modules. No new runtime dependency.

## Global Constraints

- Default `chess-gaze analyze <video>` must not call `build_qa_summary()`.
- Default `chess-gaze analyze <video>` must not call `write_qa_summary()`.
- Default `chess-gaze analyze <video>` must not create `qa_summary.json`.
- Explicit `chess-gaze analyze --qa-summary <video>` must keep the current strict QA closeout.
- Missing QA policy in a legacy manifest means `generate_qa_summary=true`.
- No-QA completion checks must stay cheap and must not count or validate all large records.
- Resume compatibility must include the persisted QA summary policy.
- Real-video verification must use `artifacts/input/nakamura_short.mp4`.
- Use TDD: write each behavior test, verify it fails for the intended reason, then implement.
- Work in the current branch and make meaningful commits.

---

## File Structure

- Modify `src/chess_gaze/frame_records.py`
  - Owns strict persisted artifact records.
  - Add `QASummaryPolicy` and `RunManifest.qa_summary_policy`.
  - Legacy default is `generate_qa_summary=True`.
- Modify `src/chess_gaze/analysis_resume.py`
  - Owns compatible-run discovery and completed-run classification.
  - Add QA policy matching and cheap no-QA completion detection.
- Modify `src/chess_gaze/pipeline.py`
  - Owns analysis orchestration.
  - Add `AnalyzeRequest.generate_qa_summary`, optional QA result fields, persisted policy, and branch closeout.
- Modify `src/chess_gaze/cli.py`
  - Owns command-line entry points.
  - Add `--qa-summary` and pass it into `AnalyzeRequest`.
- Modify `src/chess_gaze/unigaze_batch_benchmark.py`
  - Owns benchmark subprocess analysis.
  - Pass `--qa-summary` because benchmark candidate acceptance requires QA counts and schema validation.
- Modify tests under `tests/chess_gaze/`
  - Keep package-path-mirroring coverage.
  - Split default no-QA expectations from explicit QA expectations.
- Modify docs:
  - `README.md`
  - `docs/superpowers/specs/2026-06-28-resumable-analysis-design.md`
  - `docs/development/decisions/0006-stream-qa-closeout-artifacts.md`
  - `docs/development/architecture/source-layout.md` only if touched file line counts cross review text.

---

### Task 1: Persist QA Summary Policy And CLI/API Plumbing

**Files:**
- Modify: `src/chess_gaze/frame_records.py`
- Modify: `src/chess_gaze/analysis_resume.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/cli.py`
- Test: `tests/chess_gaze/test_cli.py`
- Test: `tests/chess_gaze/test_frame_records.py`
- Test: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Consumes: existing `RunManifest`, `AnalyzeRequest`, `_ResolvedRequest`, `write_initial_run_artifacts()`, and `build_parser()`.
- Produces:
  - `QASummaryPolicy(generate_qa_summary: bool)` in `src/chess_gaze/frame_records.py`.
  - `RunManifest.qa_summary_policy: QASummaryPolicy`.
  - `AnalyzeRequest.generate_qa_summary: bool = False`.
  - `_ResolvedRequest.generate_qa_summary: bool`.
  - CLI flag `--qa-summary`.
  - `write_initial_run_artifacts` accepts the new keyword parameter `qa_summary_policy: QASummaryPolicy`.

- [ ] **Step 1: Write failing CLI default test**

In `tests/chess_gaze/test_cli.py`, extend `test_analyze_prints_run_dir_and_viewer_path` after the existing request assertions:

```python
    assert request.generate_qa_summary is False
```

- [ ] **Step 2: Write failing CLI opt-in test**

In `tests/chess_gaze/test_cli.py`, add this test after `test_analyze_no_resume_forces_fresh_run_request`:

```python
def test_analyze_qa_summary_flag_requests_qa_closeout(
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

    assert main(["analyze", str(video_path), "--qa-summary"]) == 0

    [request] = captured_requests
    assert request.generate_qa_summary is True
```

- [ ] **Step 3: Write failing manifest policy test**

In `tests/chess_gaze/test_pipeline_contract.py`, add this test near the retention-policy manifest tests:

```python
def test_analyze_video_persists_default_no_qa_summary_policy(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=1)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["qa_summary_policy"] == {
        "schema_version": "qa-summary-policy-v1",
        "generate_qa_summary": False,
    }
```

- [ ] **Step 4: Write failing manifest model tests**

In `tests/chess_gaze/test_frame_records.py`, add `QASummaryPolicy` to the import from `chess_gaze.frame_records`.

Add these tests near the existing `RunManifest` tests:

```python
def test_run_manifest_records_explicit_no_qa_summary_policy() -> None:
    manifest = RunManifest(
        run_id="run-1",
        created_at_utc="2026-06-26T00:00:00Z",
        input_path="artifacts/input/nakamura_short.mp4",
        video=VideoManifest(
            source_path="artifacts/input/nakamura_short.mp4",
            source_sha256="0" * 64,
            frame_width=1920,
            frame_height=1080,
            frame_count_decoded=180,
        ),
        inference=InferenceRuntimeRecord(**_external_observer_inference_payload()),
        qa_summary_policy=QASummaryPolicy(generate_qa_summary=False),
    )

    assert manifest.qa_summary_policy.generate_qa_summary is False
    assert manifest.model_dump(mode="json")["qa_summary_policy"] == {
        "schema_version": "qa-summary-policy-v1",
        "generate_qa_summary": False,
    }


def test_run_manifest_defaults_missing_qa_summary_policy_to_legacy_generate() -> None:
    manifest = RunManifest.model_validate(
        {
            "run_id": "run-1",
            "created_at_utc": "2026-06-26T00:00:00Z",
            "input_path": "artifacts/input/nakamura_short.mp4",
            "video": {
                "source_path": "artifacts/input/nakamura_short.mp4",
                "source_sha256": "0" * 64,
                "frame_width": 1920,
                "frame_height": 1080,
                "frame_count_decoded": 180,
            },
            "inference": _external_observer_inference_payload(),
        }
    )

    assert manifest.qa_summary_policy.generate_qa_summary is True
```

- [ ] **Step 5: Run the new tests and verify RED**

Run:

```bash
uv run pytest \
  tests/chess_gaze/test_cli.py::test_analyze_prints_run_dir_and_viewer_path \
  tests/chess_gaze/test_cli.py::test_analyze_qa_summary_flag_requests_qa_closeout \
  tests/chess_gaze/test_frame_records.py::test_run_manifest_records_explicit_no_qa_summary_policy \
  tests/chess_gaze/test_frame_records.py::test_run_manifest_defaults_missing_qa_summary_policy_to_legacy_generate \
  tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_persists_default_no_qa_summary_policy \
  -q
```

Expected: failures because `AnalyzeRequest` has no `generate_qa_summary`, argparse has no `--qa-summary`, and `run_manifest.json` has no `qa_summary_policy`.

- [ ] **Step 6: Add the strict persisted policy model**

In `src/chess_gaze/frame_records.py`, add this class next to the retention policy records:

```python
class QASummaryPolicy(StrictSchemaModel):
    schema_version: Literal["qa-summary-policy-v1"] = "qa-summary-policy-v1"
    generate_qa_summary: bool
```

Then add this field to `RunManifest` after `crop_image_retention`:

```python
    qa_summary_policy: QASummaryPolicy = Field(
        default_factory=lambda: QASummaryPolicy(generate_qa_summary=True)
    )
```

This default is for legacy manifest parsing only. New run creation must pass an explicit policy.

- [ ] **Step 7: Thread the request and resolved request fields**

In `src/chess_gaze/pipeline.py`, import `QASummaryPolicy` from `chess_gaze.frame_records`.

Add this field to `AnalyzeRequest` after `save_crop_images`:

```python
    generate_qa_summary: bool = False
```

Add this field to `_ResolvedRequest` after `save_crop_images`:

```python
    generate_qa_summary: bool
```

In `_resolve_request()`, return the request value directly:

```python
        generate_qa_summary=request.generate_qa_summary,
```

- [ ] **Step 8: Persist the policy when creating a run**

In `src/chess_gaze/analysis_resume.py`, import `QASummaryPolicy`.

Change `write_initial_run_artifacts()` to accept:

```python
    qa_summary_policy: QASummaryPolicy,
```

Add the field to the `RunManifest` construction:

```python
            qa_summary_policy=qa_summary_policy,
```

In `src/chess_gaze/pipeline.py`, pass the explicit policy when calling `write_initial_run_artifacts()`:

```python
            qa_summary_policy=QASummaryPolicy(
                generate_qa_summary=resolved.generate_qa_summary
            ),
```

- [ ] **Step 9: Add the CLI flag**

In `src/chess_gaze/cli.py`, add this argument after `--no-resume`:

```python
    analyze.add_argument(
        "--qa-summary",
        action="store_true",
        default=False,
        dest="generate_qa_summary",
        help="run strict QA closeout and write qa_summary.json",
    )
```

Pass it into the `AnalyzeRequestType` constructor:

```python
                    generate_qa_summary=args.generate_qa_summary,
```

- [ ] **Step 10: Run Task 1 tests and verify GREEN**

Run:

```bash
uv run pytest \
  tests/chess_gaze/test_cli.py::test_analyze_prints_run_dir_and_viewer_path \
  tests/chess_gaze/test_cli.py::test_analyze_qa_summary_flag_requests_qa_closeout \
  tests/chess_gaze/test_frame_records.py::test_run_manifest_records_explicit_no_qa_summary_policy \
  tests/chess_gaze/test_frame_records.py::test_run_manifest_defaults_missing_qa_summary_policy_to_legacy_generate \
  tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_persists_default_no_qa_summary_policy \
  -q
```

Expected: selected tests pass.

- [ ] **Step 11: Run affected CLI and manifest tests**

Run:

```bash
uv run pytest tests/chess_gaze/test_cli.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_pipeline_contract.py -q
```

Expected: failures still allowed only where tests assert default `qa_summary.json` behavior. Fixing those belongs to Task 3.

- [ ] **Step 12: Commit Task 1**

```bash
git add src/chess_gaze/frame_records.py src/chess_gaze/analysis_resume.py src/chess_gaze/pipeline.py src/chess_gaze/cli.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_frame_records.py tests/chess_gaze/test_pipeline_contract.py
git commit -m "feat: add QA summary closeout policy"
```

---

### Task 2: Change Resume Completion Semantics For No-QA Runs

**Files:**
- Modify: `src/chess_gaze/analysis_resume.py`
- Test: `tests/chess_gaze/test_analysis_resume.py`

**Interfaces:**
- Consumes: `RunManifest.qa_summary_policy`, `AnalysisState`, existing `_run_is_complete(run_dir: Path) -> bool`.
- Produces:
  - `find_latest_resumable_run` accepts `qa_summary_policy: QASummaryPolicy` and returns `RunLayout | None`.
  - `_run_matches` accepts `expected_qa_summary_policy: QASummaryPolicy` and returns `bool`.
  - Cheap no-QA completion check based on `analysis_state.json` and required artifact file existence.

- [ ] **Step 1: Write failing no-QA complete discovery test**

In `tests/chess_gaze/test_analysis_resume.py`, import `QASummaryPolicy` from `chess_gaze.frame_records`.

Add helpers near `_write_complete_qa_summary()`:

```python
def _write_complete_analysis_state(layout: RunLayout, *, frame_count: int = 4) -> None:
    state = AnalysisState(
        run_id=layout.run_dir.name,
        input_path="artifacts/input/clip.mp4",
        source_video_sha256="a" * 64,
        frame_count_decoded=frame_count,
        next_frame_index=frame_count,
        status="complete",
        updated_at_utc="2026-06-28T12:00:00Z",
    )
    (layout.run_dir / "analysis_state.json").write_text(
        state.model_dump_json(),
        encoding="utf-8",
    )


def _write_basic_complete_derived_artifacts(layout: RunLayout) -> None:
    (layout.records_dir / "frames.jsonl").write_text("", encoding="utf-8")
    (layout.records_dir / "errors.jsonl").write_text("", encoding="utf-8")
    (layout.records_dir / "scene_frames.jsonl").write_text("", encoding="utf-8")
    (layout.scene_dir / "scene_manifest.json").write_text("{}", encoding="utf-8")
    (layout.scene_dir / "scene_summary.json").write_text("{}", encoding="utf-8")
    (layout.viewer_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (layout.viewer_dir / "scene-data.json").write_text("{}", encoding="utf-8")
```

Add this test after `test_find_latest_resumable_run_ignores_complete_and_incompatible_runs`:

```python
def test_find_latest_resumable_run_ignores_complete_no_qa_run(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "output" / "clip" / "runs"
    complete_no_qa = _make_compatible_run(
        runs_root / "20260628T110000Z-complete-no-qa",
        generate_qa_summary=False,
    )
    _write_complete_analysis_state(complete_no_qa, frame_count=4)
    _write_basic_complete_derived_artifacts(complete_no_qa)

    result = find_latest_resumable_run(
        runs_root,
        Path("artifacts/input/clip.mp4"),
        _video_manifest(frame_count=4),
        default_calibration(),
        external_observer_inference_record(),
        FrameImageRetentionPolicy(save_frame_images=True),
        CropImageRetentionPolicy(save_crop_images=True),
        QASummaryPolicy(generate_qa_summary=False),
    )

    assert result is None
```

- [ ] **Step 2: Write failing legacy protection test**

Add this test after the no-QA complete test:

```python
def test_find_latest_resumable_run_keeps_legacy_complete_state_without_qa_resumable(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "output" / "clip" / "runs"
    legacy = _make_compatible_run(
        runs_root / "20260628T110000Z-legacy-without-qa-policy"
    )
    manifest = json.loads((legacy.run_dir / "run_manifest.json").read_text())
    manifest.pop("qa_summary_policy", None)
    (legacy.run_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    _write_complete_analysis_state(legacy, frame_count=4)
    _write_basic_complete_derived_artifacts(legacy)

    result = find_latest_resumable_run(
        runs_root,
        Path("artifacts/input/clip.mp4"),
        _video_manifest(frame_count=4),
        default_calibration(),
        external_observer_inference_record(),
        FrameImageRetentionPolicy(save_frame_images=True),
        CropImageRetentionPolicy(save_crop_images=True),
        QASummaryPolicy(generate_qa_summary=True),
    )

    assert result == legacy
```

- [ ] **Step 3: Write failing policy compatibility test**

Add this test after the legacy protection test:

```python
def test_find_latest_resumable_run_requires_matching_qa_summary_policy(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "output" / "clip" / "runs"
    _make_compatible_run(
        runs_root / "20260628T100000Z-no-qa-partial",
        generate_qa_summary=False,
    )

    result = find_latest_resumable_run(
        runs_root,
        Path("artifacts/input/clip.mp4"),
        _video_manifest(frame_count=4),
        default_calibration(),
        external_observer_inference_record(),
        FrameImageRetentionPolicy(save_frame_images=True),
        CropImageRetentionPolicy(save_crop_images=True),
        QASummaryPolicy(generate_qa_summary=True),
    )

    assert result is None
```

- [ ] **Step 4: Extend test fixture metadata helpers**

In `tests/chess_gaze/test_analysis_resume.py`, update `_make_compatible_run()` and `_write_run_metadata()` signatures:

```python
    generate_qa_summary: bool = True,
```

Pass the value through `_make_compatible_run()` into `_write_run_metadata()`.

In `_write_run_metadata()`, add this field to the `RunManifest` construction:

```python
        qa_summary_policy=QASummaryPolicy(generate_qa_summary=generate_qa_summary),
```

- [ ] **Step 5: Run the new resume tests and verify RED**

Run:

```bash
uv run pytest \
  tests/chess_gaze/test_analysis_resume.py::test_find_latest_resumable_run_ignores_complete_no_qa_run \
  tests/chess_gaze/test_analysis_resume.py::test_find_latest_resumable_run_keeps_legacy_complete_state_without_qa_resumable \
  tests/chess_gaze/test_analysis_resume.py::test_find_latest_resumable_run_requires_matching_qa_summary_policy \
  -q
```

Expected: failures because `find_latest_resumable_run()` does not accept QA policy and `_run_is_complete()` only trusts `qa_summary.json`.

- [ ] **Step 6: Update resume function signatures**

In `src/chess_gaze/analysis_resume.py`, import `QASummaryPolicy`.

Change `find_latest_resumable_run()` signature to include:

```python
    qa_summary_policy: QASummaryPolicy,
```

Pass it into `_run_matches()`:

```python
            expected_qa_summary_policy=qa_summary_policy,
```

Change `_run_matches()` signature to include:

```python
    expected_qa_summary_policy: QASummaryPolicy,
```

Add this comparison to its return expression:

```python
        and run_manifest.qa_summary_policy == expected_qa_summary_policy
```

In `src/chess_gaze/pipeline.py`, update the `find_latest_resumable_run()` call to pass:

```python
            QASummaryPolicy(generate_qa_summary=resolved.generate_qa_summary),
```

- [ ] **Step 7: Implement cheap no-QA completion detection**

In `src/chess_gaze/analysis_resume.py`, replace `_run_is_complete()` with this structure:

```python
def _run_is_complete(run_dir: Path) -> bool:
    if _run_has_complete_qa_summary(run_dir):
        return True

    try:
        run_manifest = read_run_manifest_artifact_json(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return False

    if run_manifest.qa_summary_policy.generate_qa_summary:
        return False

    return _run_has_complete_no_qa_state(run_dir)
```

Add these helpers below `_run_is_complete()`:

```python
def _run_has_complete_qa_summary(run_dir: Path) -> bool:
    qa_summary_path = run_dir / "qa_summary.json"
    if not qa_summary_path.exists():
        return False

    try:
        qa_summary = QASummary.model_validate_json(
            qa_summary_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return False

    validation = qa_summary.artifact_validation
    return (
        qa_summary.final_status == "complete"
        and validation.final_status == "complete"
        and validation.schema_validation_passed
        and validation.counts_match
    )


def _run_has_complete_no_qa_state(run_dir: Path) -> bool:
    try:
        state = AnalysisState.model_validate_json(
            (run_dir / "analysis_state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError):
        return False

    return (
        state.status == "complete"
        and state.next_frame_index == state.frame_count_decoded
        and all(path.is_file() for path in _required_no_qa_completion_artifacts(run_dir))
    )


def _required_no_qa_completion_artifacts(run_dir: Path) -> list[Path]:
    return [
        run_dir / "records" / "frames.jsonl",
        run_dir / "records" / "errors.jsonl",
        run_dir / "records" / "scene_frames.jsonl",
        run_dir / "scene" / "scene_manifest.json",
        run_dir / "scene" / "scene_summary.json",
        run_dir / "viewer" / "index.html",
        run_dir / "viewer" / "scene-data.json",
    ]
```

- [ ] **Step 8: Run Task 2 tests and verify GREEN**

Run:

```bash
uv run pytest tests/chess_gaze/test_analysis_resume.py -q
```

Expected: all analysis-resume tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/chess_gaze/analysis_resume.py src/chess_gaze/pipeline.py tests/chess_gaze/test_analysis_resume.py
git commit -m "fix: classify completed no-QA runs"
```

---

### Task 3: Make Pipeline QA Closeout Optional

**Files:**
- Modify: `src/chess_gaze/pipeline.py`
- Test: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Consumes: `AnalyzeRequest.generate_qa_summary`, `_ResolvedRequest.generate_qa_summary`, `QASummaryPolicy`, resume completion semantics from Task 2.
- Produces:
  - `AnalyzeResult.qa_summary_path: Path | None`.
  - `AnalyzeResult.validated_record_count: int | None`.
  - `AnalyzeResult.validated_error_count: int | None`.
  - Default branch that marks `analysis_state.status="complete"` without building or writing QA.
  - Explicit QA branch retaining existing `revalidating` behavior.

- [ ] **Step 1: Write failing default no-QA pipeline test**

In `tests/chess_gaze/test_pipeline_contract.py`, update `test_analyze_video_does_not_retain_raw_or_processed_frame_images_by_default` by replacing the QA summary assertions with:

```python
    assert result.qa_summary_path is None
    assert result.validated_record_count is None
    assert result.validated_error_count is None
    assert not (result.layout.run_dir / "qa_summary.json").exists()
    analysis_state = json.loads(result.analysis_state_path.read_text(encoding="utf-8"))
    assert analysis_state["status"] == "complete"
    assert analysis_state["next_frame_index"] == 4
```

Keep the existing frame, manifest, crop, scene, and viewer assertions.

- [ ] **Step 2: Write failing default no-QA rerun immutability test**

Add this test after `test_analyze_video_does_not_resume_complete_run`:

```python
def test_analyze_video_does_not_resume_complete_no_qa_run(
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

    assert first.qa_summary_path is None
    assert second.qa_summary_path is None
    assert second.layout.run_dir != first.layout.run_dir
    assert not (first.layout.run_dir / "qa_summary.json").exists()
    assert not (second.layout.run_dir / "qa_summary.json").exists()
```

- [ ] **Step 3: Write failing explicit QA pipeline test**

Add this test near the existing QA summary contract tests:

```python
def test_analyze_video_writes_qa_summary_when_requested(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=3)

    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            generate_qa_summary=True,
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.qa_summary_path == result.layout.run_dir / "qa_summary.json"
    assert result.qa_summary_path.is_file()
    summary = QASummary.model_validate_json(
        result.qa_summary_path.read_text(encoding="utf-8")
    )
    assert summary.final_status == "complete"
    assert summary.artifact_validation.counts_match is True
    assert result.validated_record_count == 3
    assert result.validated_error_count == sum(summary.errors_by_code.values())
    analysis_state = json.loads(result.analysis_state_path.read_text(encoding="utf-8"))
    assert analysis_state["status"] == "complete"
```

- [ ] **Step 4: Write failing default no-QA builder guard test**

Add this test near the default no-QA pipeline tests:

```python
def test_analyze_video_default_does_not_build_or_write_qa_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    make_tiny_video(video_path, frame_count=2)

    from chess_gaze import pipeline

    def fail_build_qa_summary(*args: Any, **kwargs: Any) -> QASummary:
        del args, kwargs
        raise AssertionError("default analyze must not build QA summary")

    def fail_write_qa_summary(*args: Any, **kwargs: Any) -> QASummary:
        del args, kwargs
        raise AssertionError("default analyze must not write QA summary")

    monkeypatch.setattr(pipeline, "build_qa_summary", fail_build_qa_summary)
    monkeypatch.setattr(pipeline, "write_qa_summary", fail_write_qa_summary)

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_fake_record),
    )

    assert result.qa_summary_path is None
    assert not (result.layout.run_dir / "qa_summary.json").exists()
```

- [ ] **Step 5: Write failing explicit QA validation failure test**

Add this test near `test_analyze_video_does_not_mark_complete_before_qa_summary_exists`:

```python
def test_analyze_video_requested_qa_fails_on_counts_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path, frame_count=2)

    from chess_gaze import pipeline

    real_build_qa_summary = pipeline.build_qa_summary

    def build_failed_summary(layout: object) -> QASummary:
        summary = real_build_qa_summary(layout)
        validation = summary.artifact_validation.model_copy(
            update={
                "counts_match": False,
                "final_status": "failed",
                "validation_errors": ["simulated counts mismatch"],
            }
        )
        return summary.model_copy(
            update={
                "final_status": "failed",
                "artifact_validation": validation,
            }
        )

    monkeypatch.setattr(pipeline, "build_qa_summary", build_failed_summary)

    with pytest.raises(PipelineError, match="Run artifact validation failed"):
        analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                generate_qa_summary=True,
            ),
            observers=ObserverBundle(frame_observer=_fake_record),
        )

    [run_dir] = (output_root / "tiny" / "runs").iterdir()
    state = json.loads((run_dir / "analysis_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
```

- [ ] **Step 6: Run Task 3 tests and verify RED**

Run:

```bash
uv run pytest \
  tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_does_not_retain_raw_or_processed_frame_images_by_default \
  tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_does_not_resume_complete_no_qa_run \
  tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_writes_qa_summary_when_requested \
  tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_default_does_not_build_or_write_qa_summary \
  tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_requested_qa_fails_on_counts_mismatch \
  -q
```

Expected: failures because pipeline still always writes QA by default, result fields are mandatory, and requested QA failure currently checks schema failure but not every failed final status.

- [ ] **Step 7: Make result QA fields optional**

In `src/chess_gaze/pipeline.py`, change `AnalyzeResult` fields:

```python
    qa_summary_path: Path | None
    decoded_frame_count: int
    validated_record_count: int | None
    validated_error_count: int | None
```

- [ ] **Step 8: Add the default no-QA closeout branch**

In `src/chess_gaze/pipeline.py`, after the existing `viewer_result = build_scene_viewer(layout, scene_result)` call and before the current `revalidating` block, add:

```python
    if not resolved.generate_qa_summary:
        analysis_state = update_analysis_state(
            analysis_state,
            next_frame_index=inspection.frame_count_decoded,
            status="complete",
            clock=request.clock,
        )
        analysis_state_path = write_analysis_state(layout, analysis_state)
        return AnalyzeResult(
            layout=layout,
            run_manifest_path=run_manifest_path,
            calibration_path=calibration_path,
            video_manifest_path=video_manifest_path,
            analysis_state_path=analysis_state_path,
            frames_jsonl_path=frames_jsonl_path,
            errors_jsonl_path=errors_jsonl_path,
            scene_manifest_path=scene_result.paths.scene_manifest_path,
            scene_summary_path=scene_result.paths.scene_summary_path,
            scene_frames_jsonl_path=scene_result.paths.scene_frames_jsonl_path,
            viewer_index_path=viewer_result.index_path,
            viewer_scene_data_path=viewer_result.scene_data_path,
            qa_summary_path=None,
            decoded_frame_count=decoded_frame_count,
            validated_record_count=None,
            validated_error_count=None,
            frame_error_count=frame_error_count,
            valid_scene_frame_count=scene_result.scene_frame_count,
            valid_sphere_hit_count=scene_result.valid_sphere_hit_count,
        )
```

Leave the existing `revalidating`, `build_qa_summary()`, and `write_qa_summary()` block as the explicit QA path.

- [ ] **Step 9: Fail explicit QA on any failed QA validation result**

In `src/chess_gaze/pipeline.py`, after the existing `write_qa_summary(layout, qa_summary_path, qa_summary=qa_summary)` call succeeds, replace the schema-only check with:

```python
    validation = qa_summary.artifact_validation
    if (
        qa_summary.final_status != "complete"
        or validation.final_status != "complete"
        or not validation.schema_validation_passed
        or not validation.counts_match
    ):
        analysis_state = update_analysis_state(
            analysis_state,
            status="failed",
            clock=request.clock,
        )
        analysis_state_path = write_analysis_state(layout, analysis_state)
        raise PipelineError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            f"Run artifact validation failed; see {qa_summary_path}",
        )
```

This preserves the existing error code and message while making strict QA failure cover count mismatches and failed final status.

- [ ] **Step 10: Keep explicit QA return values current**

In the existing QA return block, keep:

```python
        qa_summary_path=qa_summary_path,
        validated_record_count=qa_summary.counts.frame_records,
        validated_error_count=sum(qa_summary.errors_by_code.values()),
```

Do not set these fields for default no-QA runs.

- [ ] **Step 11: Update existing tests that still assume default QA**

In `tests/chess_gaze/test_pipeline_contract.py`, convert existing tests that read `result.qa_summary_path` on default requests to either:

1. pass `generate_qa_summary=True` when the test is about QA validation; or
2. assert no QA file when the test is about default artifact production.

Use this exact request pattern for QA-specific tests:

```python
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            generate_qa_summary=True,
        ),
```

Use this exact default assertion for default behavior tests:

```python
    assert result.qa_summary_path is None
    assert not (result.layout.run_dir / "qa_summary.json").exists()
```

Also update `test_analyze_video_does_not_mark_complete_before_qa_summary_exists` so its request opts into QA:

```python
            AnalyzeRequest(
                video_path=video_path,
                output_root=output_root,
                generate_qa_summary=True,
            ),
```

- [ ] **Step 12: Run Task 3 focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/chess_gaze/test_pipeline_contract.py -q
```

Expected: all pipeline contract tests pass.

- [ ] **Step 13: Commit Task 3**

```bash
git add src/chess_gaze/pipeline.py tests/chess_gaze/test_pipeline_contract.py
git commit -m "fix: skip QA summary by default"
```

---

### Task 4: Update QA-Dependent Tooling And Real-Video Contracts

**Files:**
- Modify: `src/chess_gaze/unigaze_batch_benchmark.py`
- Modify: `tests/chess_gaze/test_unigaze_batch_benchmark.py`
- Modify: `tests/chess_gaze/test_pipeline_real_video_contract.py`
- Modify: `tests/chess_gaze/test_qa_summary_real_video_contract.py`
- Modify as needed: `tests/chess_gaze/test_run_equivalence.py`

**Interfaces:**
- Consumes: optional QA result fields and `AnalyzeRequest(generate_qa_summary=True)`.
- Produces:
  - Benchmark subprocess analyze command includes `--qa-summary`.
  - Default real-video pipeline test proves no QA file exists for `nakamura_short.mp4`.
  - QA real-video test opts into QA and validates `qa_summary.json`.
  - Run-equivalence tests still use fixtures with QA summaries because equivalence requires QA evidence.

- [ ] **Step 1: Write failing benchmark command test**

In `tests/chess_gaze/test_unigaze_batch_benchmark.py`, in `test_benchmark_cli_writes_candidate_rows_and_removes_mps_env`, extend the existing command assertions so every analyze command contains `--qa-summary`.

Use this assertion after commands are captured:

```python
    assert commands
    for command in commands:
        assert "--qa-summary" in command
```

- [ ] **Step 2: Run benchmark test and verify RED**

Run:

```bash
uv run pytest tests/chess_gaze/test_unigaze_batch_benchmark.py::test_benchmark_cli_writes_candidate_rows_and_removes_mps_env -q
```

Expected: failure because `_run_analysis_subprocess()` does not include `--qa-summary`.

- [ ] **Step 3: Add `--qa-summary` to benchmark subprocess analysis**

In `src/chess_gaze/unigaze_batch_benchmark.py`, add this argument to the command list in `_run_analysis_subprocess()` after `"analyze"`:

```python
        "--qa-summary",
```

The command begins:

```python
    command = [
        "chess-gaze",
        "analyze",
        "--qa-summary",
        str(video_path),
```

- [ ] **Step 4: Update default real-video pipeline contract**

In `tests/chess_gaze/test_pipeline_real_video_contract.py`, change `_assert_completed_artifact_contract()` so it no longer reads a QA summary. Replace it with:

```python
def _assert_default_completed_artifact_contract(
    result: AnalyzeResult, *, expected_count: int
) -> list[FrameRecord]:
    records = _records(result.frames_jsonl_path)

    assert result.decoded_frame_count == expected_count
    assert result.qa_summary_path is None
    assert result.validated_record_count is None
    assert result.validated_error_count is None
    assert not (result.layout.run_dir / "qa_summary.json").exists()
    assert len(records) == expected_count
    assert records[0].frame_id == "f000000000"
    assert records[-1].frame_index == expected_count - 1
    assert result.scene_manifest_path.is_file()
    assert result.scene_summary_path.is_file()
    assert result.scene_frames_jsonl_path.is_file()
    assert result.viewer_index_path.is_file()
    assert result.viewer_scene_data_path.is_file()
    state = json.loads(result.analysis_state_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["next_frame_index"] == expected_count
    return records
```

Update `test_real_video_model_free_pipeline_writes_complete_artifact_contract()`:

```python
    records = _assert_default_completed_artifact_contract(
        result, expected_count=expected_count
    )
```

Remove assertions that read `summary.counts.crop_files` and `summary.byte_counts.crops_bytes` from this default test.

- [ ] **Step 5: Update QA real-video contract to opt in**

In `tests/chess_gaze/test_qa_summary_real_video_contract.py`, change the request to:

```python
    result = analyze_video(
        AnalyzeRequest(
            video_path=video_path,
            output_root=tmp_path / "output",
            generate_qa_summary=True,
        ),
        observers=ObserverBundle(frame_observer=_deterministic_real_video_record),
    )
```

Add:

```python
    assert result.qa_summary_path == qa_summary_path
```

- [ ] **Step 6: Check run-equivalence fixtures**

Run:

```bash
uv run pytest tests/chess_gaze/test_run_equivalence.py -q
```

Expected: pass without production-code changes. If it fails because a fixture now creates a run through `analyze_video()` without QA, update only that fixture request to pass `generate_qa_summary=True`.

- [ ] **Step 7: Run Task 4 focused tests and verify GREEN**

Run:

```bash
uv run pytest \
  tests/chess_gaze/test_unigaze_batch_benchmark.py \
  tests/chess_gaze/test_pipeline_real_video_contract.py::test_real_video_model_free_pipeline_writes_complete_artifact_contract \
  tests/chess_gaze/test_qa_summary_real_video_contract.py::test_real_video_model_free_pipeline_writes_qa_summary_revalidation \
  tests/chess_gaze/test_run_equivalence.py \
  -q
```

Expected: selected tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/chess_gaze/unigaze_batch_benchmark.py tests/chess_gaze/test_unigaze_batch_benchmark.py tests/chess_gaze/test_pipeline_real_video_contract.py tests/chess_gaze/test_qa_summary_real_video_contract.py tests/chess_gaze/test_run_equivalence.py
git commit -m "test: opt QA-dependent flows into QA summary"
```

---

### Task 5: Update Canonical Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-28-resumable-analysis-design.md`
- Modify: `docs/development/decisions/0006-stream-qa-closeout-artifacts.md`
- Modify if needed: `docs/development/architecture/source-layout.md`

**Interfaces:**
- Consumes: implementation from Tasks 1-4.
- Produces: docs that no longer claim `qa_summary.json` is always generated or always the completion seal.

- [ ] **Step 1: Search stale QA completion claims**

Run:

```bash
rg -n "qa_summary.json|completion seal|revalidating|--qa-summary|QA summary" README.md docs/development docs/superpowers/specs
```

Expected: matches include the resumable-analysis spec and ADR-0006.

- [ ] **Step 2: Update README usage text**

In `README.md`, ensure analyze usage says:

```markdown
Default analysis writes frame records, scene artifacts, and viewer artifacts.
It does not write `qa_summary.json`. Use `--qa-summary` when you want strict
run-level artifact validation and a persisted QA report.
```

If README has a command block for analysis flags, include:

```sh
uv run chess-gaze analyze --qa-summary artifacts/input/nakamura_short.mp4
```

- [ ] **Step 3: Annotate the resumable-analysis design**

In `docs/superpowers/specs/2026-06-28-resumable-analysis-design.md`, add a supersession note near the existing supersession note:

```markdown
Supersession note, 2026-07-03: `qa_summary.json` is no longer the universal
completion seal for new runs. `docs/superpowers/specs/2026-07-03-optional-qa-summary-design.md`
supersedes the completion-seal sections for runs whose manifest explicitly
sets `qa_summary_policy.generate_qa_summary=false`. Legacy manifests without
that policy still require a valid complete `qa_summary.json`.
```

In the `New Artifact` section, replace the absolute sentence "`qa_summary.json` remains the completion seal." with:

```markdown
For legacy and QA-requested runs, `qa_summary.json` remains the QA completion
seal. For new no-QA runs, `analysis_state.json` plus the required derived
artifact files form the cheap completion signal described in the optional QA
summary design.
```

- [ ] **Step 4: Update ADR-0006**

In `docs/development/decisions/0006-stream-qa-closeout-artifacts.md`, add this note under `## Status`:

```markdown
Superseded in part by `docs/superpowers/specs/2026-07-03-optional-qa-summary-design.md`:
streaming QA closeout still applies when QA summary generation is requested,
but default analysis no longer builds or writes `qa_summary.json`.
```

In `## Consequences`, replace "`qa_summary.json` remains the completion seal." with:

```markdown
`qa_summary.json` remains the strict QA validation seal only for QA-requested
and legacy QA-required runs.
```

- [ ] **Step 5: Update source-layout review if line counts require it**

Run:

```bash
wc -l src/chess_gaze/pipeline.py src/chess_gaze/analysis_resume.py src/chess_gaze/frame_records.py
```

If `pipeline.py` or another touched source file crosses a new source-layout review threshold, add a concise dated review note to `docs/development/architecture/source-layout.md` explaining why the added QA policy branching belongs to the existing orchestration/resume boundary.

- [ ] **Step 6: Run docs stale-claim check**

Run:

```bash
rg -n "qa_summary.json remains the completion seal|universal completion seal|always writes.*qa_summary|always.*qa_summary.json" README.md docs/development docs/superpowers/specs
```

Expected: no stale absolute claim remains outside historical context with a supersession note.

- [ ] **Step 7: Commit Task 5**

```bash
git add README.md docs/superpowers/specs/2026-06-28-resumable-analysis-design.md docs/development/decisions/0006-stream-qa-closeout-artifacts.md docs/development/architecture/source-layout.md
git commit -m "docs: document optional QA summary behavior"
```

---

### Task 6: Full Verification And Closeout

**Files:**
- Create: `docs/superpowers/closeouts/2026-07-03-optional-qa-summary.md`
- Modify if test evidence requires: files from prior tasks only

**Interfaces:**
- Consumes: completed implementation and docs.
- Produces: fresh verification evidence and a closeout document.

- [ ] **Step 1: Run focused QA/default test groups**

Run:

```bash
uv run pytest \
  tests/chess_gaze/test_cli.py \
  tests/chess_gaze/test_analysis_resume.py \
  tests/chess_gaze/test_pipeline_contract.py \
  tests/chess_gaze/test_qa_summary.py \
  tests/chess_gaze/test_unigaze_batch_benchmark.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run real-video default no-QA verification**

Run:

```bash
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py::test_real_video_model_free_pipeline_writes_complete_artifact_contract -q
```

Expected: pass; test verifies `artifacts/input/nakamura_short.mp4` produces complete state and viewer artifacts without `qa_summary.json`.

- [ ] **Step 3: Run real-video QA opt-in verification**

Run:

```bash
uv run pytest tests/chess_gaze/test_qa_summary_real_video_contract.py::test_real_video_model_free_pipeline_writes_qa_summary_revalidation -q
```

Expected: pass; test verifies `artifacts/input/nakamura_short.mp4` writes a valid complete `qa_summary.json` only when requested.

- [ ] **Step 4: Run native default-model smoke if available**

Run:

```bash
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py::test_nakamura_short_default_model_pipeline_does_not_create_crop_directory -q
```

Expected: pass on machines with the local model assets required by this test. If it fails because model assets are absent, record the exact failure and do not claim native smoke coverage.

- [ ] **Step 5: Run static gates**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: all commands exit 0.

- [ ] **Step 6: Run broad local gates**

Run:

```bash
uv run pytest -m "not native_mediapipe and not local_socket"
uv run pytest -m local_socket
```

Expected: both commands exit 0.

- [ ] **Step 7: Confirm no default QA summary was generated in a real run**

Run a model-free real-video analysis through pytest evidence from Step 2, then inspect the temp run path printed by pytest only if the test logs it. If the path is not printed, run this direct command:

```bash
uv run python -c 'from pathlib import Path
from tempfile import TemporaryDirectory
from chess_gaze.pipeline import AnalyzeRequest, ObserverBundle, ObserverFrame, analyze_video
from tests.chess_gaze.test_pipeline_contract import _fake_record
with TemporaryDirectory() as tmp:
    result = analyze_video(
        AnalyzeRequest(
            video_path=Path("artifacts/input/nakamura_short.mp4"),
            output_root=Path(tmp) / "output",
        ),
        observers=ObserverBundle(frame_observer=_fake_record),
    )
    print(result.qa_summary_path)
    print((result.layout.run_dir / "qa_summary.json").exists())
    print((result.layout.run_dir / "analysis_state.json").read_text())'
```

Expected output includes:

```text
None
False
```

and `analysis_state.json` contains `"status":"complete"` or `"status": "complete"`.

- [ ] **Step 8: Write closeout**

Create `docs/superpowers/closeouts/2026-07-03-optional-qa-summary.md` with:

```markdown
# Optional QA Summary Closeout

Date: 2026-07-03

## Summary

Default analysis no longer generates or writes `qa_summary.json`. QA summary
generation is available through `--qa-summary` or
`AnalyzeRequest(generate_qa_summary=True)`.

## Root Cause

`qa_summary.json` had been treated as the universal completion seal even though
it is an expensive audit artifact, not a necessary artifact for a healthy
viewer-ready run.

## Durable Surface Changed

- Run manifests now persist QA closeout policy.
- Resume discovery includes QA policy compatibility.
- No-QA runs complete through `analysis_state.json` plus cheap required artifact
  existence checks.
- QA-requested runs keep strict streamed QA validation.

## Regression Tests

List the exact tests added or changed.

## Verification

Paste exact commands and pass/fail results from this task.

## Residual Risk

Record any skipped native test or known warning exactly.
```

- [ ] **Step 9: Commit closeout if all required verification has run**

```bash
git add docs/superpowers/closeouts/2026-07-03-optional-qa-summary.md
git commit -m "docs: close optional QA summary change"
```

- [ ] **Step 10: Final status check**

Run:

```bash
git status --short
git log --oneline -8
```

Expected: working tree clean and recent commits include the task commits from this plan.
