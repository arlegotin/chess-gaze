# Real-Data Verification Documentation Update Closeout

Date: 2026-06-25

Supersession note: this closeout is historical. Later on 2026-06-25, the
MediaPipe and UniGaze assets were downloaded, checksummed, and verified; see
`docs/superpowers/closeouts/2026-06-25-frame-gaze-analysis-pipeline.md` for the
current implementation and remaining limitations.

## Request Summary

Update the active frame-gaze spec and plan so `artifacts/input/test_1.mp4` and
`artifacts/input/test_2.mp4` are treated as mandatory real-data verification
inputs, not examples. Every subsystem that can be tested with real data must be
verified as early as possible before later work builds on it.

## Documentation Summary

- Updated the active frame-gaze design spec to name the two local videos as
  mandatory real-data verification inputs.
- Removed stale spec wording that said no implementation plan existed.
- Made acceptance explicit: theoretical progress, synthetic-only tests, and
  skipped smoke checks cannot be claimed as complete real-data verification.
- Updated the implementation plan global constraints with a real-data checkpoint
  rule before task commits.
- Added concrete early real-video gates for decode, face observation, eye/iris,
  head pose, UniGaze, visualization, model-free pipeline orchestration, QA
  summary revalidation, and final real-model smoke.
- Required exact blocker recording when ignored local videos or model assets are
  unavailable.

## Subagent Review Evidence

- Canonical-target review identified the active frame-gaze spec and plan as the
  correct documents, plus stale spec status text and ambiguous optional smoke
  wording.
- Wording review confirmed `artifacts/input/test_1.mp4` and
  `artifacts/input/test_2.mp4` are present locally, while `models/` is absent.
- Diff review found blocking gaps in Tasks 7, 8, 9, 10, and 13 where synthetic
  evidence could still be accepted before real-data verification. Those gaps
  were repaired by adding task-local real-video gates or explicit blockers.
- A final review subagent did not return after interruption and was closed; the
  completed blocking findings from the prior review were incorporated.

## Verification Evidence

Initial sandboxed pytest was blocked by package-resolution network access for
`hatchling`. It was rerun with network access and passed.

Fresh local gates after documentation edits:

```text
UV_CACHE_DIR=.uv-cache uv run pytest
2 passed in 0.01s
```

```text
UV_CACHE_DIR=.uv-cache uv run ruff check .
All checks passed!
```

```text
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
2 files already formatted
```

```text
UV_CACHE_DIR=.uv-cache uv run mypy
Success: no issues found in 2 source files
```

```text
git diff --check
```

No whitespace errors were reported.

## Remaining Limitations

- This change updates documentation only; it does not implement the frame-gaze
  pipeline or real-video tests.
- Full real-model smoke is still blocked until the required local model assets
  exist under `models/` with committed registry checksums.
