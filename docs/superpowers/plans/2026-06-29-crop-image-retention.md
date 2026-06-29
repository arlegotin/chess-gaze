# Default Crop Image Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make default analysis avoid retaining `crops/**/*.png`, while preserving crop files when explicitly requested with `--save-crops` or `save_crop_images=True`.

**Architecture:** Persist a crop-image retention policy in `run_manifest.json`, resolve it from config/request/CLI, and enforce it at the eye crop PNG write boundary. QA validates zero crop files when saving is disabled and reports crop counts without exact-count enforcement when saving is enabled because current durable records do not encode every write-eligible crop.

**Tech Stack:** Python 3.12, Pydantic 2.13.4, PyAV 17.1.0, MediaPipe 0.10.35, NumPy 2.5.0, Pillow 12.2.0, pytest, uv.

## Global Constraints

- Do not add a new third-party dependency.
- Default `chess-gaze analyze <video>` must retain zero crop image files under `crops/`.
- `--save-crops` must be the explicit CLI opt-in that retains eye crop PNGs.
- `--save-frames` must continue to control only raw decoded frame PNGs and processed overlay JPEGs.
- Programmatic callers opt in with `AnalyzeRequest(save_crop_images=True)`.
- JSON config may set `save_crop_images: true`; request/CLI overrides win over config.
- The run manifest must persist `crop_image_retention.schema_version="crop-image-retention-v1"` and `crop_image_retention.save_crop_images`.
- Legacy run manifests without `crop_image_retention` must read as `save_crop_images=true`.
- QA validation must expect `crop_files=0` when `save_crop_images=false`; when `save_crop_images=true`, QA reports crop counts but does not assert an exact crop count.
- Resume compatibility must require matching crop-image retention policy.
- Do not remove or weaken frame records, scene records, viewer artifacts, model-runtime metadata, in-memory crop geometry, crop transforms, or analysis-state checkpoint behavior.
- During verification, run `artifacts/input/nakamura_short.mp4` through the default model-backed pipeline and confirm zero crop files.

---

## File Structure

- Modify `src/chess_gaze/frame_records.py`: add `CropImageRetentionPolicy` and `RunManifest.crop_image_retention` with legacy default.
- Modify `src/chess_gaze/configuration.py`: add `save_crop_images` config and request override support.
- Modify `src/chess_gaze/analysis_resume.py`: persist policy on new runs and compare it during resumable-run discovery.
- Modify `src/chess_gaze/pipeline.py`: carry resolved policy into the default observer.
- Modify `src/chess_gaze/frame_observation.py`: add `ModelBackedFrameObserver.save_crop_images` and pass it to `observe_eyes`.
- Modify `src/chess_gaze/eye_observation.py`: compute crop geometry independently from optional crop PNG writes.
- Modify `src/chess_gaze/qa_summary.py`: validate crop counts against policy.
- Modify `src/chess_gaze/cli.py`: add `--save-crops`.
- Modify focused tests in `tests/chess_gaze/test_configuration.py`, `tests/chess_gaze/test_cli.py`, `tests/chess_gaze/test_eye_observation.py`, `tests/chess_gaze/test_pipeline_contract.py`, `tests/chess_gaze/test_qa_summary.py`, `tests/chess_gaze/test_analysis_resume.py`, and real-video contract tests.
- Update `README.md`, `docs/development/architecture/source-layout.md`, and add ADR/closeout docs.

---

### Task 1: Persist and Resolve Crop Image Retention Policy

**Files:**
- Modify: `src/chess_gaze/frame_records.py`
- Modify: `src/chess_gaze/configuration.py`
- Modify: `src/chess_gaze/analysis_resume.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/cli.py`
- Test: `tests/chess_gaze/test_configuration.py`
- Test: `tests/chess_gaze/test_cli.py`
- Test: `tests/chess_gaze/test_pipeline_contract.py`
- Test: `tests/chess_gaze/test_analysis_resume.py`

**Interfaces:**
- Produces: `CropImageRetentionPolicy(save_crop_images: bool, schema_version: Literal["crop-image-retention-v1"])`
- Produces: `AnalysisConfig.save_crop_images: bool`
- Produces: `AnalyzeRequest.save_crop_images: bool | None`
- Produces: `_ResolvedRequest.save_crop_images: bool`
- Produces: CLI flag `--save-crops`

- [ ] **Step 1: Write failing configuration, CLI, manifest, and resume tests**

Add these assertions/tests:

```python
assert config.save_crop_images is False
```

```python
def test_load_config_accepts_save_crop_images(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"save_crop_images": true}', encoding="utf-8")

    config = load_config(config_path)

    assert config.save_crop_images is True
```

```python
def test_analyze_save_crops_flag_requests_crop_image_retention(
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

    assert main(["analyze", str(video_path), "--save-crops"]) == 0

    [request] = captured_requests
    assert request.save_crop_images is True
```

In a pipeline manifest test:

```python
manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
assert manifest["crop_image_retention"] == {
    "schema_version": "crop-image-retention-v1",
    "save_crop_images": False,
}
```

Add a resume compatibility test that creates an interrupted run with
`crop_image_retention.save_crop_images=True`, then reruns with the default
request and asserts a new run is created instead of resuming the old one.

- [ ] **Step 2: Run tests to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_configuration.py::test_load_config_accepts_save_crop_images tests/chess_gaze/test_cli.py::test_analyze_save_crops_flag_requests_crop_image_retention -q
```

Expected: FAIL because the config field, request field, and CLI flag do not
exist.

- [ ] **Step 3: Implement schema/config/request/CLI policy**

Add `CropImageRetentionPolicy` to `src/chess_gaze/frame_records.py`:

```python
class CropImageRetentionPolicy(StrictSchemaModel):
    schema_version: Literal["crop-image-retention-v1"] = "crop-image-retention-v1"
    save_crop_images: bool
```

Add to `RunManifest`:

```python
crop_image_retention: CropImageRetentionPolicy = Field(
    default_factory=lambda: CropImageRetentionPolicy(save_crop_images=True)
)
```

Add to `AnalysisConfig`:

```python
save_crop_images: bool = False
```

Add `save_crop_images: bool | None = None` to `apply_analysis_overrides()` and
only place it into the payload when not `None`.

Add `save_crop_images: bool | None = None` to `AnalyzeRequest`, and
`save_crop_images: bool` to `_ResolvedRequest`. Pass it from
`apply_analysis_overrides()`.

In `analysis_resume.write_initial_run_artifacts()`, add a required
`crop_image_retention: CropImageRetentionPolicy` parameter and write it into
`RunManifest`.

In `find_latest_resumable_run()` and `_run_matches()`, add a required
`crop_image_retention` parameter and compare it to the manifest.

In `pipeline.analyze_video()`, create
`CropImageRetentionPolicy(save_crop_images=resolved.save_crop_images)` and pass
it to run creation and resume lookup.

In `cli.build_parser()`, add:

```python
analyze.add_argument(
    "--save-crops",
    action="store_true",
    default=None,
    dest="save_crop_images",
    help="retain eye crop PNGs under crops/",
)
```

Pass `save_crop_images=args.save_crop_images` into `AnalyzeRequest`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_configuration.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_analysis_resume.py -q
```

Expected: PASS.

---

### Task 2: Gate Eye Crop PNG Writes Without Removing Crop Geometry

**Files:**
- Modify: `src/chess_gaze/eye_observation.py`
- Modify: `src/chess_gaze/frame_observation.py`
- Modify: `src/chess_gaze/pipeline.py`
- Test: `tests/chess_gaze/test_eye_observation.py`
- Test: `tests/chess_gaze/test_pipeline_contract.py`

**Interfaces:**
- Consumes: `observe_eyes(..., save_crop_images: bool = False)`
- Consumes: `ModelBackedFrameObserver.save_crop_images`
- Produces: no crop files by default while keeping `crop_bbox_image_px` and `eye_crop_transform_to_image_px`
- Produces: retained crop paths and hashes when `save_crop_images=True`

- [ ] **Step 1: Write failing eye-observation tests**

Change the existing default eye-observation test to assert no crop file paths or
files by default:

```python
assert observation.left.eye_crop_path is None
assert observation.left.eye_crop_sha256 is None
assert observation.left.crop_bbox_image_px is not None
assert observation.left.eye_crop_transform_to_image_px is not None
assert list(run_layout.crops_dir.rglob("*.png")) == []
```

Add an explicit retention test:

```python
def test_observe_eyes_retains_crop_files_when_requested(tmp_path: Path) -> None:
    run_layout = make_run_layout(tmp_path)
    face = make_face_candidate(...)

    observation = observe_eyes(
        face,
        gradient_rgb_frame(),
        run_layout,
        frame_id="f000000047",
        save_crop_images=True,
    )

    assert observation.left.eye_crop_path == Path("crops/eyes/left/f000000047.png")
    assert observation.left.eye_crop_sha256 is not None
    assert (run_layout.run_dir / observation.left.eye_crop_path).is_file()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_eye_observation.py -q
```

Expected: FAIL because crop files are currently written by default and
`save_crop_images` is not accepted.

- [ ] **Step 3: Implement optional crop write**

Add `save_crop_images: bool = False` to `observe_eyes()` and `_observe_eye()`.

Rename `_save_eye_crop()` to `_eye_crop_record()` and make it compute crop
bounds, bbox, and transform before any optional write. Return `path=None` and
`sha256=None` when saving is disabled.

Update `_CropRecord`:

```python
@dataclass(frozen=True)
class _CropRecord:
    path: Path | None
    sha256: str | None
    bbox_image_px: BBox
    transform_to_image_px: CropTransformToImagePx
```

In `ModelBackedFrameObserver`, add:

```python
save_crop_images: bool = False
```

Pass it to `self.eye_observer(..., save_crop_images=self.save_crop_images)`.

In `pipeline._default_observer_bundle_factory()`, accept `save_crop_images` and
construct `ModelBackedFrameObserver(save_crop_images=save_crop_images, ...)`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_eye_observation.py tests/chess_gaze/test_pipeline_contract.py -q
```

Expected: PASS.

---

### Task 3: Make QA Policy-Aware for Crops

**Files:**
- Modify: `src/chess_gaze/qa_summary.py`
- Test: `tests/chess_gaze/test_qa_summary.py`
- Test: `tests/chess_gaze/test_qa_summary_real_video_contract.py`

**Interfaces:**
- Consumes: `RunManifest.crop_image_retention.save_crop_images`
- Produces: failed QA validation when `save_crop_images=false` and crop files exist
- Produces: complete QA validation when `save_crop_images=false` and crop files are absent

- [ ] **Step 1: Write failing QA policy tests**

Extend fixture helper with `save_crop_images: bool = True` and
`write_crop_images: bool = True`.

Add:

```python
def test_validate_run_artifacts_accepts_unretained_crop_images_when_disabled(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(
        tmp_path,
        frame_count=3,
        save_frame_images=False,
        write_frame_images=False,
        save_crop_images=False,
        write_crop_images=False,
    )

    result = validate_run_artifacts(layout)
    summary = build_qa_summary(layout)

    assert result.counts.crop_files == 0
    assert result.counts_match is True
    assert summary.byte_counts.crops_bytes == 0
    assert summary.final_status == "complete"
```

Add:

```python
def test_validate_run_artifacts_rejects_stray_crop_images_when_policy_disables_saving(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(
        tmp_path,
        frame_count=2,
        save_frame_images=False,
        write_frame_images=False,
        save_crop_images=False,
        write_crop_images=False,
    )
    (layout.left_eye_crops_dir / "f000000000.png").write_bytes(b"stray")

    result = validate_run_artifacts(layout)

    assert result.counts_match is False
    assert result.final_status == "failed"
    assert result.validation_errors == [
        "crop file count does not match crop image retention policy: 1 != 0"
    ]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_qa_summary.py -k 'crop_images or crop_files' -q
```

Expected: FAIL because QA does not validate crop policy.

- [ ] **Step 3: Implement crop policy validation**

Pass `loaded.run_manifest.crop_image_retention` into `_count_validation_errors()`.

Add:

```python
if not crop_image_retention.save_crop_images and counts.crop_files != 0:
    errors.append(
        "crop file count does not match crop image retention policy: "
        f"{counts.crop_files} != 0"
    )
```

Do not enforce an exact count when saving is enabled.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```sh
uv run pytest tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_qa_summary_real_video_contract.py -q
```

Expected: PASS.

---

### Task 4: Real Video Verification and Documentation

**Files:**
- Modify: `tests/chess_gaze/test_pipeline_real_video_contract.py`
- Modify: `README.md`
- Modify: `docs/development/architecture/source-layout.md`
- Create: `docs/development/decisions/0005-default-crop-image-retention.md`
- Create: `docs/superpowers/closeouts/2026-06-29-crop-image-retention.md`

**Interfaces:**
- Consumes: `artifacts/input/nakamura_short.mp4`
- Produces: default real-video analysis with zero crop files
- Produces: explicit crop-retaining real-video analysis with crop files

- [ ] **Step 1: Write or update real-video contract tests**

In `test_real_video_model_free_pipeline_writes_complete_artifact_contract()`,
assert `crop_count == 0` for the default external-observer path if no model
observer writes crops.

Add or update a model-backed real-video contract using
`artifacts/input/nakamura_short.mp4`:

```python
def test_nakamura_short_default_model_pipeline_does_not_retain_crop_files(
    tmp_path: Path,
) -> None:
    result = analyze_video(
        AnalyzeRequest(
            video_path=NAKAMURA_SHORT_VIDEO,
            output_root=tmp_path / "output",
            unigaze_device="cpu",
            unigaze_batch_size=7,
        )
    )

    crop_count = len(list(result.layout.crops_dir.rglob("*.png")))
    summary = QASummary.model_validate_json(
        result.qa_summary_path.read_text(encoding="utf-8")
    )

    assert result.decoded_frame_count == NAKAMURA_SHORT_FRAME_COUNT
    assert crop_count == 0
    assert summary.counts.crop_files == 0
    assert summary.final_status == "complete"
```

If this test is too slow for the default suite, mark it consistently with the
repo's existing real-video/native-test marker policy and still run the command
manually during verification.

- [ ] **Step 2: Run real-video test to verify RED**

Run:

```sh
uv run pytest tests/chess_gaze/test_pipeline_real_video_contract.py -k 'crop or artifact_contract' -q
```

Expected before Task 2 implementation: FAIL because default model-backed runs
write crop files.

- [ ] **Step 3: Update docs and closeout**

Document:

- `--save-crops` in README usage/options;
- `crops/` empty by default and populated by `--save-crops`;
- `--save-frames` and `--save-crops` are separate debug-retention flags;
- ADR-0005 with selected approach, alternatives, compatibility, and
  verification;
- source-layout ownership for crop retention policy;
- closeout with root cause, durable surface, real-video evidence, tests run,
  and residual limitations.

- [ ] **Step 4: Run focused real-video verification**

Run:

```sh
uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --no-resume --output-root /private/tmp/chess-gaze-crop-repro-after --progress off
```

Then verify:

```sh
find /private/tmp/chess-gaze-crop-repro-after/nakamura_short/runs -path '*/crops/*' -type f | wc -l
jq '.counts.crop_files, .byte_counts.crops_bytes, .final_status, .artifact_validation.counts_match' /private/tmp/chess-gaze-crop-repro-after/nakamura_short/runs/*/qa_summary.json
```

Expected: `0`, `0`, `"complete"`, and `true`.

- [ ] **Step 5: Run broad gates**

Run:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: PASS.
