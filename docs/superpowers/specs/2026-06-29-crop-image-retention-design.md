# Default Crop Image Retention Design

Date: 2026-06-29

## Summary

`chess-gaze analyze` must stop retaining eye crop PNGs by default, matching the
current default behavior for decoded raw frames and processed overlays. Crop
files are visual/debug artifacts. They are not the durable analysis source of
truth, and the default pipeline must keep frame records, scene records, viewer
data, QA summaries, runtime metadata, and in-memory crop computation intact
without writing `crops/**/*.png`.

The explicit crop-retention opt-in is `--save-crops` on the CLI and
`save_crop_images=True` for programmatic `AnalyzeRequest` callers. JSON config
may also set `save_crop_images: true`.

This is intentionally separate from `--save-frames`. `--save-frames` continues
to mean raw decoded PNGs and processed overlay JPEGs only.

## Evidence and Research

- Reproduction on 2026-06-29:
  `uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --no-resume --output-root /private/tmp/chess-gaze-crop-repro-before --progress off`
  produced a complete default run with `0` raw frame files, `0` processed frame
  files, and `360` crop files for `180` decoded frames.
- The representative crop at
  `/private/tmp/chess-gaze-crop-repro-before/nakamura_short/runs/20260629T171227Z-5b4ddce6/crops/eyes/left/f000000000.png`
  is a small eye-region debug image. The matching first video frame shows the
  streamer face region and chess board; the crop is not a unique source input.
- Existing large artifacts show the cost class: the local Carlsen run contains
  `31,920` crop files using about `125M`, while raw/processed frame retention is
  already handled separately.
- Code evidence shows the durable records do not persist crop paths or crop
  hashes. `EyeRecord` persists bounding boxes, pupil centers, iris landmarks,
  and invalid reasons. Scene artifacts consume `records/frames.jsonl`, not crop
  files.
- Third-party-library behavior relevant to this change:
  - PyAV decodes video frames into arrays before observers run; the change does
    not alter decode cadence or frame identity.
  - MediaPipe Face Landmarker `IMAGE` mode and UniGaze face normalization use
    in-memory image data and landmark tensors; no persisted crop file is an
    inference input.
  - NumPy slicing can produce in-memory crop views without requiring disk I/O.
  - Pillow writes an image only when code calls an image save path, so gating
    `save_rgb_png()` is the durable storage boundary.
  - Pydantic v2 `default_factory` is the established repo pattern for legacy
    manifest defaults.

## Requirements

- Default analysis must not persist any crop image files under `crops/`.
- Default analysis must not compromise analysis correctness. Eye crop geometry,
  crop transforms, face-region probing, UniGaze normalized face crops, frame
  records, scene records, viewer artifacts, and QA artifacts must still be
  produced from in-memory data and committed records.
- Crop image files must be retained only when an explicit flag or request/config
  option asks for them.
- The run's crop retention policy must be persisted in `run_manifest.json`.
- QA validation must treat crop file counts according to the persisted policy:
  `0` crop files when saving is disabled, and no exact count assertion when
  saving is enabled because crop count is data-dependent and not reconstructible
  from current `FrameRecord` fields.
- Legacy run manifests that predate crop retention must continue to validate as
  crop-saving runs because historical completed runs retained crop images.
- Resume must not mix runs with different crop-retention policy.
- `--save-frames` must not silently retain crops. Users who want all debug
  images must pass both `--save-frames` and `--save-crops`.
- The old `crops/` directory tree may still exist as an empty layout directory
  for compatibility.

## Design

Add a persisted crop-image retention policy to `RunManifest`:

```json
"crop_image_retention": {
  "schema_version": "crop-image-retention-v1",
  "save_crop_images": false
}
```

`RunManifest` will default missing legacy policy to `save_crop_images=true`.
New run creation will always write the resolved current policy explicitly.

`AnalysisConfig` gains `save_crop_images: bool = false`. `AnalyzeRequest`
accepts `save_crop_images: bool | None = None`; `None` means use config, while
`True` or `False` explicitly override config. The CLI adds `--save-crops` and
passes `True` only when supplied.

The pipeline carries the resolved policy into default model observers by
constructing `ModelBackedFrameObserver(save_crop_images=resolved.save_crop_images)`.
Tests and external observer callers are unaffected unless they use the model
observer path.

`eye_observation.py` enforces the policy at the only production crop-file write
boundary:

- `observe_eyes(..., save_crop_images=False)` still computes crop bounds,
  `crop_bbox_image_px`, and `eye_crop_transform_to_image_px`, but does not call
  `save_rgb_png()` and returns `eye_crop_path=None` and `eye_crop_sha256=None`.
- `observe_eyes(..., save_crop_images=True)` preserves the current file path,
  SHA-256, bbox, and transform behavior.

`qa_summary.py` reads `run_manifest.crop_image_retention.save_crop_images`.
When saving is disabled, any file under `crops/` is a validation failure. When
saving is enabled, QA reports crop counts and bytes but does not assert an exact
count because current durable frame records do not encode every write-eligible
eye crop.

Resume compatibility adds the crop-image retention policy to existing matching
criteria. Cleanup continues to delete uncommitted crop files inside the run root
for crop-retaining partial runs.

## Alternatives Considered

| Approach | Result | Reason Rejected |
| --- | --- | --- |
| Reuse `--save-frames` for crops | One fewer flag | Rejected because current docs and CLI semantics define `--save-frames` as raw/processed frames only. Silent broadening would surprise users. |
| Delete crops after QA | Lower final disk use | Rejected because it still consumes disk during analysis and can stale QA summaries unless rebuilt after deletion. |
| Stop computing eye crops entirely | Fewer operations | Rejected because crop geometry and transforms are analysis evidence, and future observers may need in-memory crops. |
| Gate writes and persist crop policy | Selected | Avoids unnecessary disk writes, keeps QA/resume consistent, preserves analysis, and makes the retention contract auditable. |

## Testing

Focused tests must prove:

- `observe_eyes()` defaults to no crop files while preserving crop geometry and
  transforms.
- `observe_eyes(save_crop_images=True)` retains existing crop file, path, hash,
  bbox, and transform behavior.
- config accepts `save_crop_images`;
- CLI `--save-crops` forwards the opt-in to `AnalyzeRequest`;
- default fake-video pipeline runs persist crop policy and keep zero crop files;
- default model-backed real-video runs on `artifacts/input/nakamura_short.mp4`
  keep zero crop files while retaining complete records, scene artifacts, and
  QA status;
- explicit crop-saving real-video runs retain crop files;
- QA accepts zero crop files only when policy says saving is disabled;
- QA rejects stray crop files when policy disables saving;
- legacy run manifests without `crop_image_retention` read as crop-saving runs;
- resume compatibility rejects runs with different crop-retention policy.

Broad verification must include:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Documentation

This spec supersedes earlier crop-pipeline guidance that implied completed runs
always contain eye crop image files. The durable contract is now
policy-dependent:

- default completed runs retain zero crop image files;
- explicit `--save-crops` completed runs retain eye crop debug PNGs for
  write-eligible eyes.
