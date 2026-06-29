# ADR-0005: Default Crop Image Retention

Date: 2026-06-29

## Status

Accepted

## Context

Eye crop PNGs are debugging artifacts, not the durable analysis source of truth.
The durable contract is the committed frame journal, scene records, viewer data,
runtime metadata, and QA summary. Existing local artifacts showed the cost: one
historical Carlsen run retained `31,920` crop PNGs, and a default
`nakamura_short.mp4` run retained `360` crop PNGs even though `raw_frames/` and
`processed_frames/` were already empty by policy.

Code inspection showed persisted `FrameRecord` eye data contains bounding boxes,
pupil centers, iris landmarks, and invalid reasons, but no crop path, hash, or
transform. Scene artifacts are built from `records/frames.jsonl`, not crop
files. MediaPipe region crops and UniGaze normalized face crops are in-memory
operations.

## Alternatives and Evidence

| Alternative | Evidence | Decision |
| --- | --- | --- |
| Reuse `--save-frames` for crop files | `--save-frames` is documented and implemented for raw decoded frames and processed overlays. | Rejected because silently broadening the flag would surprise users and mix distinct artifact classes. |
| Delete crop files after QA | Would reduce final disk use. | Rejected because it still consumes disk during analysis and can stale QA unless rebuilt after deletion. |
| Stop computing crop geometry | Would remove crop work, not only storage. | Rejected because crop geometry and transforms remain useful evidence and do not require persisted PNGs. |
| Persist a crop-retention policy and gate writes | Matches ADR-0004's policy-driven artifact contract while keeping the flag separate. | Selected. |

## Decision

Default analysis does not persist crop image files under `crops/`.

Crop image retention is opt-in:

- CLI: `--save-crops`
- Programmatic request: `AnalyzeRequest(save_crop_images=True)`
- JSON config: `"save_crop_images": true`

Every new run manifest persists:

```json
"crop_image_retention": {
  "schema_version": "crop-image-retention-v1",
  "save_crop_images": false
}
```

Legacy run manifests without `crop_image_retention` are interpreted as
`save_crop_images=true`, matching historical completed runs.

QA validation expects `crop_files=0` when saving is disabled. When saving is
enabled, QA reports crop counts and bytes but does not assert an exact crop
count because current frame records do not encode every write-eligible eye crop.
Resume compatibility requires matching crop-retention policy.

`--save-frames` remains scoped to `raw_frames/*.png` and
`processed_frames/*.jpg`. Users who want all debug images must pass both
`--save-frames` and `--save-crops`.

## Consequences

- Default runs retain no crop PNGs and use less disk space.
- The `crops/` directory tree remains present but empty for layout
  compatibility.
- Eye crop geometry and transforms are still computed in memory.
- Crop write failures are only possible in explicit `--save-crops` runs.
- Historical run artifacts remain readable under the legacy crop-saving policy.

## Verification

Future agents can verify the decision with:

```sh
uv run pytest tests/chess_gaze/test_eye_observation.py
uv run pytest tests/chess_gaze/test_qa_summary.py -k 'crop_images or crop_files'
uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --no-resume --output-root /private/tmp/chess-gaze-crop-check --progress off
find /private/tmp/chess-gaze-crop-check/nakamura_short/runs -path '*/crops/*' -type f | wc -l
```

The default real-video run should report zero crop files and a complete
`qa_summary.json`.
