# ADR-0006: Stream QA Closeout Artifacts

Date: 2026-07-03

## Status

Accepted

Superseded in part by `docs/superpowers/specs/2026-07-03-optional-qa-summary-design.md`:
streaming QA closeout still applies when QA summary generation is requested,
but default analysis no longer builds or writes `qa_summary.json`.

## Context

The run `artifacts/output/nepo_2/runs/20260630T082559Z-f865e2af` produced frame,
scene, and viewer artifacts, then was stopped during final closeout. It had
28,141 frame records and scene-frame records and no `qa_summary.json`, while
`analysis_state.json` said `status="complete"`.

Read-only reproduction of `build_qa_summary()` on that run completed but peaked
at `8,392,245,248` bytes maximum resident set size. The root cause was QA
closeout materializing large artifacts at once:

- `records/frames.jsonl`: about 901 MB
- `records/scene_frames.jsonl`: about 84 MB
- `viewer/scene-data.json`: about 139 MB
- full Pydantic `ViewerSceneData` and per-line record lists in memory

Primary sources checked on 2026-07-03:

- Python `pathlib.Path.read_text()` returns the decoded contents of the whole
  file as a string: https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text
- Python `json.load()` deserializes a readable file into a Python object:
  https://docs.python.org/3/library/json.html#json.load
- Pydantic recommends `model_validate_json()` for direct JSON validation and
  avoiding validation work that is not needed:
  https://docs.pydantic.dev/latest/concepts/performance/
- PyAV container usage in this repository already follows context-manager
  lifetime handling:
  https://pyav.org/docs/stable/api/container.html

## Alternatives and Evidence

| Alternative | Evidence | Decision |
| --- | --- | --- |
| Keep whole-run Pydantic materialization | Directly measured 8.39 GB max RSS on the interrupted run. It preserves strict validation but makes closeout fragile for large runs. | Rejected. |
| Add a streaming JSON dependency such as `ijson` | Would reduce custom scanner code, but adds a core dependency solely for closeout and triggers dependency-selection overhead. `ijson` is not already installed locally. | Rejected for this repair. |
| Stream JSONL records and structurally validate viewer data while validating each large array element incrementally | Keeps strict validation for manifests, frame JSONL, error JSONL, scene-frame JSONL, viewer frames, and viewer hit points while avoiding full `ViewerSceneData` materialization. | Accepted. |
| Mark `analysis_state` complete only after `qa_summary.json` exists | Avoids complete state without seal, but violates the prior final-state-before-seal regression and can leave a complete seal with stale state if interrupted. | Rejected. |
| Add `revalidating` state and revert on in-process QA write failure | Preserves `qa_summary.json` as the completion seal, reduces the complete-without-seal window to the final atomic write path, and makes in-process write failures visibly nonterminal. | Accepted. |

## Decision

QA closeout must stream large JSONL artifacts and aggregate only the fields
needed for `QASummary`. It must not call `Path.read_text()` or equivalent
whole-file reads for:

- `records/frames.jsonl`
- `records/errors.jsonl`
- `records/scene_frames.jsonl`
- `viewer/scene-data.json`

`viewer/scene-data.json` is validated with a standard-library structural
scanner. The scanner rejects unexpected top-level keys, validates every `frames`
item as `SceneFrameRecord`, validates every `valid_hit_points` item as
`ViewerHitPoint`, validates the small envelope with Pydantic, and cross-checks
it against `run_manifest`, `video_manifest`, and `scene_summary`.

`AnalysisState.status` includes `revalidating`. The analyzer writes
`revalidating` before QA closeout, writes the terminal state before the QA seal,
and reverts to `revalidating` if the QA write fails in-process.

## Consequences

Benefits:

- Peak closeout memory on the stopped run dropped from 8.39 GB to about 224 MB.
- `qa_summary.json` remains the strict QA validation seal only for
  QA-requested and legacy QA-required runs.
- Frame, error, and scene-frame schemas remain validated record by record.
- The existing final-state-before-seal regression remains honored.

Costs:

- `qa_summary.py` now owns a small JSON structural scanner and is intentionally
  deep.
- Closeout now validates viewer scene-data frames in addition to scene-frame
  JSONL records, so elapsed time increased from about 50 seconds at baseline to
  about 73 seconds while fixing the memory failure.

Follow-up work:

- If closeout time becomes a problem, persist incremental QA counters during
  frame processing or add a well-vetted streaming JSON parser through the model
  and library selection process.
- If `qa_summary.py` approaches 1,500 lines, split JSONL streaming summaries and
  viewer-envelope scanning into named modules with explicit interface tests.

## Verification

Future agents should verify:

- `tests/chess_gaze/test_qa_summary.py::test_build_qa_summary_streams_large_artifacts_without_whole_file_reads`
- `tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_does_not_mark_complete_before_qa_summary_exists`
- A timed read-only `build_qa_summary()` on the `nepo_2` run or an equivalent
  large run, checking that max RSS stays far below the 8.39 GB baseline.
