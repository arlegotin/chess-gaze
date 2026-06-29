# Default Frame Image Retention Design

Date: 2026-06-29

## Summary

`chess-gaze analyze` must stop retaining decoded raw frame PNGs and processed
overlay JPEGs by default. Frame images are large debugging artifacts, not the
analysis source of truth. The default run must keep strict frame records, scene
records, viewer data, QA summaries, and model/runtime metadata, while leaving
`raw_frames/` and `processed_frames/` empty unless the caller explicitly opts in.

The explicit opt-in is `--save-frames` on the CLI and `save_frame_images=True`
for programmatic `AnalyzeRequest` callers. JSON config may also set
`save_frame_images: true`.

## Requirements

- Default analysis must not persist raw decoded frames or processed frame
  visualizations.
- Default analysis must not compromise analysis correctness. Observers must
  still receive the decoded RGB frame in memory, and scene/viewer/QA artifacts
  must still be built from committed `records/frames.jsonl` and
  `records/scene_frames.jsonl`.
- Raw and processed frame images must be retained only when an explicit flag or
  request/config option asks for them.
- The run's artifact policy must be persisted so QA validation and resume
  behavior do not infer retention from incidental files.
- QA validation must treat frame image counts according to the persisted policy:
  `0` raw and `0` processed files when saving is disabled, decoded-frame count
  for both when saving is enabled.
- Legacy run manifests that predate the policy must continue to validate as
  frame-saving runs, because historical completed runs retained frame images.
- Resume must not mix runs with different frame-image retention policy.
- The old raw/processed frame directories may still exist as empty run-layout
  directories for compatibility, but image files must not be kept there by
  default.

## Design

Add a persisted frame-image retention policy to `RunManifest`:

```json
"frame_image_retention": {
  "schema_version": "frame-image-retention-v1",
  "save_frame_images": false
}
```

`RunManifest` will default missing legacy policy to `save_frame_images=true`.
New run creation will always write the resolved current policy explicitly.

`AnalysisConfig` gains `save_frame_images: bool = false`. `AnalyzeRequest`
accepts `save_frame_images: bool | None = None`; `None` means use config, while
`True` or `False` explicitly override config. The CLI adds `--save-frames` and
passes `True` only when supplied.

The pipeline enforces the policy at the write boundaries:

- `_prepare_decoded_frame()` constructs the in-memory `ObserverFrame` exactly as
  before, but writes `raw_frames/<frame_id>.png` only when `save_frame_images` is
  true.
- `_render_processed_frame_and_collect_errors()` writes
  `processed_frames/<frame_id>.jpg` only when `save_frame_images` is true.
- When saving is disabled, no raw or processed image writer is called, so no
  image write error can be introduced by storage pressure.

`qa_summary.py` reads `run_manifest.frame_image_retention.save_frame_images` and
validates raw/processed counts against the expected count for that policy.
Quality samples that depend on raw frame images, such as worst blur and exposure
frame IDs, remain empty when frames are not retained.

Resume compatibility adds the frame-image retention policy to the existing
matching criteria. This keeps the completion seal internally consistent: a run
started with saved frame images finishes under that same policy, and a default
no-save run cannot accidentally inherit retained-frame expectations.

## Alternatives Considered

| Approach | Result | Reason Rejected |
| --- | --- | --- |
| Delete raw/processed images after QA writes | Lower final disk use | Creates a stale QA completion seal unless QA is rebuilt after deletion, and still consumes disk during analysis. |
| Add a cleanup command only | Manual disk recovery | Does not satisfy default behavior and leaves analysis runs large unless users remember the command. |
| Gate writes and persist policy | Selected | Avoids unnecessary disk writes, keeps QA/resume consistent, and makes the retention contract auditable. |

## Testing

Focused tests must prove:

- default fake-video runs keep frame records, scene records, viewer data, and QA
  but retain zero raw and processed frame image files;
- `AnalyzeRequest(save_frame_images=True)` retains one raw and one processed
  image per decoded frame;
- `--save-frames` forwards the opt-in to `AnalyzeRequest`;
- config accepts `save_frame_images`;
- QA accepts zero frame images only when policy says saving is disabled;
- QA still reports a count mismatch if a disabled-retention run has stray frame
  images;
- image write failure regressions still work under explicit frame saving;
- real-video model-free contracts follow the new default no-retention behavior.

Broad verification must include:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Documentation

This spec supersedes earlier frame-pipeline guidance that said every completed
run always contains one raw and one processed frame image per decoded frame.
The durable contract is now policy-dependent:

- default completed runs retain zero raw/processed frame image files;
- explicit `--save-frames` completed runs retain decoded-frame-count raw and
  processed frame image files.

