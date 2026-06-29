# ADR-0004: Default Frame Image Retention

Date: 2026-06-29

## Status

Accepted

## Context

Raw decoded frame PNGs and processed overlay JPEGs dominate run disk usage.
They are useful visual debugging artifacts, but they are not the durable
analysis source of truth. The durable analysis contract is the committed frame
journal, scene records, viewer data, runtime metadata, and QA summary.

The previous pipeline treated raw and processed frame image files as mandatory
completed-run artifacts. That made normal analysis expensive in disk space and
made QA validation depend on retaining derived images even when no later step
needed them.

## Alternatives and Evidence

| Alternative | Evidence | Decision |
| --- | --- | --- |
| Keep unconditional image writes | Existing implementation and QA invariants required decoded-frame-count raw and processed images. | Rejected because it keeps the disk-pressure failure mode. |
| Write images and delete them after QA | Would reduce final run size. | Rejected because the run consumes disk during analysis and QA can become stale unless rebuilt after deletion. |
| Add a manual cleanup command | Lets users recover disk after the fact. | Rejected because default analysis would still keep unnecessary frame files. |
| Persist a retention policy and gate writes | Makes retention explicit in `run_manifest.json`, lets QA validate the correct count, and avoids writes when disabled. | Selected. |

## Decision

Default analysis does not persist raw or processed frame image files.

Frame image retention is opt-in:

- CLI: `--save-frames`
- Programmatic request: `AnalyzeRequest(save_frame_images=True)`
- JSON config: `"save_frame_images": true`

Crop image retention is governed separately by
[ADR-0005](0005-default-crop-image-retention.md). `--save-frames` does not
retain crop PNGs.

Every new run manifest persists:

```json
"frame_image_retention": {
  "schema_version": "frame-image-retention-v1",
  "save_frame_images": false
}
```

Legacy run manifests without `frame_image_retention` are interpreted as
`save_frame_images=true`, matching historical completed runs.

QA validation expects raw and processed frame image counts of `0` when saving is
disabled, and decoded-frame count when saving is enabled. Resume compatibility
requires matching retention policy.

## Consequences

- Default runs use substantially less disk space.
- Default `raw_frames/` and `processed_frames/` directories remain present but
  empty for path-layout compatibility.
- QA blur and exposure rankings are empty unless raw frames are explicitly
  saved.
- Image writer failures are only possible in explicit save-frame runs.
- Historical run artifacts remain readable under the legacy saved-frame policy.

## Verification

Future agents can verify the decision with:

```sh
uv run pytest tests/chess_gaze/test_pipeline_contract.py -k 'frame_images or save'
uv run pytest tests/chess_gaze/test_qa_summary.py -k 'frame_images'
uv run pytest
```

Inspect a new default run's `run_manifest.json` and confirm
`frame_image_retention.save_frame_images` is `false`, with zero raw and processed
frame image files.
