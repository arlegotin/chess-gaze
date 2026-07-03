# Resumable Analysis Design

Date: 2026-06-28

## Status

Approved for implementation by the user's 2026-06-28 task request. The
executable plan is `docs/superpowers/plans/2026-06-28-resumable-analysis.md`.

Supersession note, 2026-06-29: this design's raw/processed frame image count
examples are superseded by
`docs/superpowers/specs/2026-06-29-frame-image-retention-design.md` and
ADR-0004. Current default runs validate zero raw and processed frame image
files; explicit save-frame runs validate decoded-frame-count image files.

Supersession note, 2026-07-03: `qa_summary.json` is no longer the universal
completion seal for new runs.
`docs/superpowers/specs/2026-07-03-optional-qa-summary-design.md` supersedes
the completion-seal sections for runs whose manifest explicitly sets
`qa_summary_policy.generate_qa_summary=false`. Legacy manifests without that
policy still require a valid complete `qa_summary.json`.

## Goal

Make `uv run chess-gaze analyze <video>` resume the latest compatible
interrupted analysis run instead of starting frame inference from zero again.

The concrete motivating case is:

```sh
uv run chess-gaze analyze artifacts/input/carlsen_1.mp4
```

The local checkout contains an interrupted Carlsen run at
`artifacts/output/carlsen_1/runs/20260628T101348Z-e546cf6a`. It has a manifest
for `artifacts/input/carlsen_1.mp4`, no `qa_summary.json`, 9,016 committed
frame records out of 16,050 decoded frames, 9,023 raw PNGs, 9,016 processed
JPGs, 17,972 eye crop PNGs, and no scene/viewer artifacts. That evidence shows
the current failure mode: per-frame artifacts are not transactional as a group,
so resume must treat `records/frames.jsonl` as the committed frame journal and
not infer progress from raw frame counts.

## Current Behavior

`chess-gaze analyze` always calls `analyze_video()`, and `analyze_video()` always
calls `create_run_layout()` to create a new timestamped run directory under:

```text
artifacts/output/<video-stem>/runs/<run-id>/
```

Initial manifests are written before the frame loop:

- `run_manifest.json`
- `calibration.json`
- `video_manifest.json`
- empty `records/frames.jsonl`
- empty `records/errors.jsonl`

Per-frame processing writes raw images, eye crops, processed images, frame
errors, and then frame records. Scene artifacts, viewer files, and
`qa_summary.json` are whole-run derived artifacts written after the frame loop.

There is no existing resume flag, checkpoint, or skip behavior.

## Behavior

Default behavior changes for interrupted runs only:

- Running `chess-gaze analyze <video>` searches
  `<output-root>/<video-stem>/runs/` for the newest compatible incomplete run.
- If one exists, analysis resumes in that same run directory from the first
  uncommitted frame.
- If the newest compatible run is already complete, a rerun still creates a new
  immutable run directory, preserving the historical completed-run behavior.
- If no compatible incomplete run exists, analysis creates a new run directory.
- `--no-resume` disables this lookup and forces a new run directory.

Compatibility is strict:

- `run_manifest.json` and `video_manifest.json` must validate.
- `run_manifest.input_path` must equal the current requested video path string.
- Source SHA256, decoded frame count, width, and height must match the current
  video inspection.
- The persisted inference runtime record must match the current resolved
  runtime record.
- The persisted calibration record must match the current default calibration
  record.
- `qa_summary.json` with `final_status=complete`, schema validation passed, and
  count validation passed seals a completed run and prevents resume.

If a partial run is incompatible, the analyzer leaves it untouched and looks for
an older compatible incomplete run. If none exists, it starts a fresh run.

## Resume Boundary

`records/frames.jsonl` is the source of truth for committed frame progress.
The analyzer reads the valid leading prefix only:

- each non-empty line must validate as `FrameRecord`;
- `frame_index` must be contiguous from zero;
- `frame_id` must equal the canonical zero-padded frame id for the index;
- records beyond the current inspected decoded frame count are uncommitted.

On resume, the analyzer atomically rewrites `frames.jsonl` to that valid prefix
and atomically rebuilds `errors.jsonl` from the committed frame records. It then
removes uncommitted raw/processed frame files and eye crop files whose frame
index is at or beyond the next frame index. Scene artifacts, viewer artifacts,
`records/scene_frames.jsonl`, and `qa_summary.json` are removed or overwritten
because they are derived from the whole final frame journal.

The frame loop still decodes from frame zero. Frames with `frame_index` below
the next uncommitted index are decoded and discarded before model inference.
This is intentionally conservative: PyAV seek is timestamp/keyframe based and
the installed PyAV 17.1.0 `InputContainer.seek()` documentation says frame-index
offsets are unsupported and decoded packets after seek correspond only
"roughly" to the requested position.

After every processed batch, the analyzer flushes and fsyncs the JSONL handles
before writing an atomic `analysis_state.json` checkpoint. The checkpoint is
informational and recoverable; `frames.jsonl` remains the commit log.

## New Artifact

`analysis_state.json` lives at the run root. It is not the completion seal.
For legacy and QA-requested runs, `qa_summary.json` remains the QA completion
seal. For new no-QA runs, `analysis_state.json` plus the required derived
artifact files form the cheap completion signal described in the optional QA
summary design.

Schema:

```json
{
  "schema_version": "analysis-state-v1",
  "run_id": "20260628T101348Z-e546cf6a",
  "input_path": "artifacts/input/carlsen_1.mp4",
  "source_video_sha256": "...",
  "frame_count_decoded": 16050,
  "next_frame_index": 9016,
  "status": "processing",
  "updated_at_utc": "2026-06-28T12:00:00Z"
}
```

Allowed statuses are `processing`, `revalidating`, `complete`, and `failed`.
`revalidating` means frame processing and derived scene/viewer artifacts are
done, but a required `qa_summary.json` completion seal is not yet durable. For
legacy manifests and QA-requested runs, `analysis_state.status == "complete"`
still requires a valid complete `qa_summary.json`; without it, the run remains
incomplete and must stay resumable from the frame journal. For new no-QA runs
whose manifest sets `qa_summary_policy.generate_qa_summary=false`,
`analysis_state.status == "complete"` plus the required derived artifact
presence is the cheap completion signal, so no `qa_summary.json` is required.
The analyzer may write terminal `complete` or `failed` state before atomically
writing `qa_summary.json`, but if that QA write fails in-process for a
QA-required run it must revert the state to `revalidating` so the run is not
reported as complete without the seal.

## Non-Goals

- No direct frame-index seeking.
- No new third-party dependencies.
- No attempt to make all per-frame artifacts transactional as one filesystem
  unit.
- No attempt to resume if the input video, inference runtime, or calibration
  changed.
- No automatic deletion of incompatible partial runs.
- No batch command or multi-video queue.

## Third-Party Documentation and Practices

No new external library is selected for this feature. Existing dependencies are
used with stricter recovery boundaries.

| Concern | Library / API | Source checked on 2026-06-28 | Relevant finding | Decision |
| --- | --- | --- | --- | --- |
| Video resume position | PyAV 17.1.0 `InputContainer.seek()` installed docstring and PyAV container docs, https://pyav.org/docs/stable/api/container.html | Seek offsets are timestamp based; frame-index offsets are unsupported; decode should usually restart from the previous keyframe. | Decode from zero and discard already committed frames for correctness. |
| JSON artifact validation | Pydantic 2.13.4 docs, https://docs.pydantic.dev/latest/concepts/models/ | `model_validate_json()` validates JSON bytes/strings directly against strict models. | Use Pydantic validation on resume; never trust JSONL line count alone. |
| Whole-file atomic replacement | Python `os.replace`, `tempfile.NamedTemporaryFile`, and `os.fsync` docs, https://docs.python.org/3/library/os.html and https://docs.python.org/3/library/tempfile.html | Same-directory temp file plus replacement is the right primitive for whole-file artifacts; fsync improves crash durability. | Keep atomic writes for JSON and rebuilt JSONL; fsync append handles before checkpoint advance. |
| JSONL commit journal | JSON Lines spec, https://jsonlines.org/ | Each line is one valid JSON value; blank lines are not JSON values. | Treat only complete, valid, contiguous lines as committed. |
| Model runtimes | MediaPipe Face Landmarker docs, https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker and UniGaze model source, https://huggingface.co/UniGaze/UniGaze-models | Model/runtime choices are unchanged. | Resume reuses the existing validated runtime path and compares persisted runtime metadata before reusing a run. |

Practices adopted:

- Use an explicit completion seal (`qa_summary.json`) rather than run directory
  existence.
- Use a committed prefix and discard/repair any invalid tail.
- Rebuild derived whole-run artifacts from the recovered frame journal.
- Prefer idempotent repair over appending to potentially skewed
  `errors.jsonl`.

Mistakes avoided:

- Do not use raw frame counts or crop counts as progress.
- Do not trust `run_manifest.json` as completion.
- Do not seek by decoded frame number.
- Do not rebuild scene artifacts from non-contiguous frame records.
- Do not bypass schema validation in recovery paths.

## Testing Strategy

Unit and contract tests:

- CLI default passes `resume=True`; `--no-resume` passes `resume=False`.
- Interrupted runs resume instead of creating a new run.
- Completed runs do not resume and still create a new run.
- Incompatible partial runs are left untouched.
- Malformed or non-contiguous JSONL tails are repaired by truncating to the
  valid committed prefix.
- `errors.jsonl` is rebuilt from committed frame records.
- Surplus uncommitted raw/processed/crop files do not poison the resumed run.

Real verification:

- Use `artifacts/input/nakamura_short.mp4`.
- Create a partial run by interrupting or simulating interruption after a
  committed prefix.
- Run the same `chess-gaze analyze artifacts/input/nakamura_short.mp4` command
  again and verify the same run directory completes with:
  - `qa_summary.final_status == "complete"`;
  - `counts.decoded_frames == 180`;
  - `counts.frame_records == 180`;
  - `counts.raw_frames == 180`;
  - `counts.processed_frames == 180`;
  - `counts.scene_frame_records == 180`.

## Closeout Requirements

The closeout must record:

- the exact root cause;
- the durable surface changed;
- the regression tests added;
- focused and broad gate results;
- the exact `nakamura_short.mp4` interruption/resume evidence;
- whether the existing Carlsen partial run was actually resumed during
  verification or only used as forensic evidence.
