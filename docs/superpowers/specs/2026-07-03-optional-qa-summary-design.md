# Optional QA Summary Design

Date: 2026-07-03

## Status

Approved by the user's 2026-07-03 confirmation: default analysis must not
generate or write `qa_summary.json`.

## Goal

Make normal analysis complete without generating `qa_summary.json` by default,
while keeping strict QA summary generation available behind an explicit analyze
flag.

## Current Behavior

`chess-gaze analyze <video>` always performs the expensive QA closeout:

- builds scene artifacts;
- builds viewer artifacts;
- streams and validates the full run into `QASummary`;
- writes `qa_summary.json`;
- treats a valid complete `qa_summary.json` as the completed-run seal.

That default is too expensive for normal successful runs. The June 30 Nepo run
showed that even after the memory repair, QA closeout remains a separate
whole-run validation phase rather than a necessary part of producing usable
records, scene artifacts, and viewer artifacts.

## Behavior

Default analyze behavior changes:

- `chess-gaze analyze <video>` does not call `build_qa_summary()`.
- It does not call `write_qa_summary()`.
- It does not create `qa_summary.json`.
- It marks the run complete after frame processing, scene artifact generation,
  and viewer artifact generation succeed.

Explicit QA behavior:

- `chess-gaze analyze --qa-summary <video>` keeps the current strict QA closeout.
- It writes `qa_summary.json`.
- It uses the existing `revalidating` state while the QA seal is being built.
- It fails the run if strict QA artifact validation fails.

Downstream tooling that requires QA evidence must opt in explicitly or require
an existing QA summary. In particular, benchmark commands that launch analysis
must pass `--qa-summary`, and run-equivalence checks may continue to reject run
directories that do not contain a valid `qa_summary.json`.

## Completion Semantics

`qa_summary.json` is no longer the universal completion seal. It is an optional
validation artifact.

New runs must persist their QA closeout policy in `run_manifest.json`, using a
strict record with:

```json
{
  "schema_version": "qa-summary-policy-v1",
  "generate_qa_summary": false
}
```

Missing policy in a legacy manifest means `generate_qa_summary=true`. This is
intentional: old runs such as the stopped Nepo run may have
`analysis_state.status == "complete"` without a durable QA file, and they must
not be silently reclassified as completed no-QA runs.

A run is complete if either condition holds:

1. It has a valid `qa_summary.json` with `final_status="complete"`,
   `artifact_validation.final_status="complete"`,
   `schema_validation_passed=true`, and `counts_match=true`.
2. Its manifest explicitly says `generate_qa_summary=false`, its
   `analysis_state.json` has `status="complete"` and
   `next_frame_index == frame_count_decoded`, and the basic complete-run
   artifacts exist:
   - `records/frames.jsonl`
   - `records/errors.jsonl`
   - `records/scene_frames.jsonl`
   - `scene/scene_manifest.json`
   - `scene/scene_summary.json`
   - `viewer/index.html`
   - `viewer/scene-data.json`

The no-QA completion check must stay cheap. It must not count or validate all
large records, because that would recreate the QA closeout cost under a
different name.

## API And CLI

`AnalyzeRequest` gains:

```python
generate_qa_summary: bool = False
```

The CLI gains:

```text
--qa-summary
```

`AnalyzeResult` must stop implying that a QA file exists for every successful
run. The QA-specific fields become optional:

```python
qa_summary_path: Path | None
validated_record_count: int | None
validated_error_count: int | None
```

For default no-QA runs, those fields are `None`. For `--qa-summary` runs, they
retain their current meanings.

## Resume Behavior

Resume compatibility includes the persisted QA summary policy. A default no-QA
partial run is resumed only by another no-QA request. A QA-requested partial run
is resumed only by another QA-requested request.

This keeps run manifests truthful and avoids changing a run's closeout contract
mid-run. If a user wants QA output for a no-QA run, this change supports rerun
analysis with `--qa-summary`; it does not add a separate post-hoc QA command.

Resume discovery treats default no-QA completed runs as complete using the
cheap completion check above, so rerunning the same default command creates a
new immutable run instead of rewriting the latest completed run.

## Testing Strategy

Add red/green tests for:

- CLI default passes `generate_qa_summary=False`.
- CLI `--qa-summary` passes `generate_qa_summary=True`.
- Default `analyze_video()` does not create `qa_summary.json`.
- Default successful runs write `analysis_state.status == "complete"`.
- Default successful runs are not selected for resume on a rerun.
- `AnalyzeResult` QA-specific fields are `None` by default.
- `AnalyzeRequest(generate_qa_summary=True)` still writes a valid complete
  `qa_summary.json`.
- Legacy manifests without the policy and no QA file are still treated as
  incomplete.
- New no-QA manifests with complete state and basic artifacts are treated as
  complete.
- Benchmark subprocess analysis includes `--qa-summary`.

Real-video verification must use `artifacts/input/nakamura_short.mp4` for both:

- default run: no `qa_summary.json`, complete state, viewer exists;
- `--qa-summary` run: valid complete `qa_summary.json`.

## Documentation

Update canonical docs that currently call `qa_summary.json` the universal
completion seal:

- `docs/superpowers/specs/2026-06-28-resumable-analysis-design.md`
- `docs/development/decisions/0006-stream-qa-closeout-artifacts.md`
- `docs/development/architecture/source-layout.md` if line-count or ownership
  notes change
- README usage text if it claims QA summary is always produced

Historical specs may keep their original requirements if linked or annotated as
superseded by this design.
