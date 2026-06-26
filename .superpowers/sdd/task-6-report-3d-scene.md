# Task 6 Report: Run Layout, Pipeline, And QA Summary Integration

Date: 2026-06-26

## Scope

Implemented Task 6 integration in:

- `src/chess_gaze/artifact_runs.py`
- `src/chess_gaze/pipeline.py`
- `src/chess_gaze/qa_summary.py`
- `tests/chess_gaze/test_artifact_runs.py`
- `tests/chess_gaze/test_pipeline_contract.py`
- `tests/chess_gaze/test_qa_summary.py`

Report file:

- `.superpowers/sdd/task-6-report-3d-scene.md`

No full viewer UI/assets were implemented. `viewer/index.html` is a minimal
placeholder so QA can validate the declared `viewer_index` source artifact.

## RED Evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py -q
```

Output:

```text
.F...FF..........FF.....                                                 [100%]
5 failed, 19 passed in 1.39s
```

Expected failures:

- `RunLayout` lacked `scene_dir`.
- `AnalyzeResult` lacked `scene_manifest_path`.
- Pipeline did not fail when scene artifact validation was corrupted.
- QA fixture code could not write `viewer/scene-data.json` through `RunLayout.viewer_dir`.

## GREEN Evidence

Focused Task 6 command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py -q
```

Final output:

```text
........................                                                 [100%]
24 passed in 1.30s
```

Real-video model-free checkpoint:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Output:

```text
.                                                                        [100%]
1 passed in 424.25s (0:07:04)
```

Focused Ruff:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/artifact_runs.py src/chess_gaze/pipeline.py src/chess_gaze/qa_summary.py tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_frame_observation.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py
```

Output:

```text
All checks passed!
```

## Design Notes

- `RunLayout.scene_dir` and `RunLayout.viewer_dir` are derived from `run_dir`,
  preserving existing direct `RunLayout` fixtures while `create_run_layout()`
  creates both directories for real runs.
- `analyze_video()` now builds scene artifacts after frame artifact handles and
  observers are closed and before QA summary generation.
- `viewer/scene-data.json` is written from approved
  `build_viewer_scene_data()` using aliases and `allow_nan=False`.
- QA validates `SceneManifest`, `SceneSummary`, `SceneFrameRecord` JSONL, and
  `ViewerSceneData` through strict scene models.
- QA source artifacts include `scene_manifest`, `scene_summary`,
  `scene_frames_jsonl`, `viewer_index`, and `viewer_scene_data`, and
  `QASummary.source_artifacts` is copied from
  `ArtifactValidationResult.source_artifacts`.
- `ArtifactCounts.scene_frame_records` remains separate from the original
  `frame_records` count, and `validated_record_count` still reports original
  frame records.
- `ByteCounts` now separates original JSONL bytes, scene JSONL bytes, scene
  directory bytes, viewer bytes, and total run bytes.

## Residual Risks

- The placeholder `viewer/index.html` is intentionally not the final viewer.
  Task 7/8 still need packaged static assets and the interactive UI.
- QA currently validates viewer index existence only; strict viewer HTML asset
  validation belongs with the viewer asset tasks.
- `build_scene_artifacts()` still writes `SceneSummary.artifact_validation`
  before viewer generation, so the persisted scene summary may report
  `viewer_exists=False` until a later task updates scene summary semantics.
