# Task 1 Report: Face Candidate Fallback Arbitration Regression

## Scope

Implemented only Task 1 from `.superpowers/sdd/task-1-brief.md`.

Files changed:

- `tests/chess_gaze/test_face_observation_region_arbitration.py`
- `.superpowers/sdd/task-1-report.md`

No production code was modified.

## Regression Added

Added
`test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate`.

The fake MediaPipe result sequence follows the current observer region order:

1. `full_frame`: no detections.
2. `left_half`: real face.
3. `right_half`: no detections.
4. `left_top`: no detections.
5. `right_top`: no detections.
6. `left_upper_band`: same real face plus a larger false positive.
7. `right_upper_band`: no detections.
8. `right_upper_middle`: no detections.

The expected primary candidate is the cross-region-supported real face with
full-image pixel bounds `(360, 216)` to `(540, 432)`, not the larger
single-region false positive.

## TDD RED Evidence

Command:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate -q
```

Observed output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate _

E       AssertionError: assert 'larger_singl...alse_positive' == 'cross_region_real_face'
E
E         - cross_region_real_face
E         + larger_single_region_false_positive

tests/chess_gaze/test_face_observation_region_arbitration.py:409: AssertionError
=========================== short test summary info ============================
FAILED tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate
1 failed in 0.16s
```

The failure is the intended RED failure: current fallback arbitration selects
the `left_upper_band` primary candidate, which is the larger single-region false
positive, instead of the real face seen in both `left_half` and
`left_upper_band`.

## Additional Checks

Focused lint:

```sh
uv run ruff check tests/chess_gaze/test_face_observation_region_arbitration.py
```

Observed output:

```text
All checks passed!
```

Diff whitespace check:

```sh
git diff --check
```

Observed result: exit code `0`, no output.

## Self-Review

- The test uses the existing fake MediaPipe sequence helper and tests the real
  `MediaPipeFaceObserver.observe()` behavior at the observer seam.
- The fake sequence length and detected image shapes lock the region order so
  the regression cannot pass accidentally by shifting fake results across
  regions.
- The assertion resolves `selection.primary_candidate_id` before checking the
  blendshape label, avoiding a false pass when the selected region contains
  multiple candidates.
- The expected coordinates are full-image pixel coordinates derived from the
  same real-face landmarks in `left_half` and `left_upper_band`.
- The function name keeps the exact brief-specified pytest node id; a narrow
  `# noqa: E501` is used because the required name exceeds the Ruff line limit.

## Concerns

- This task is intentionally RED-only, so the focused pytest command fails until
  the production arbitration is repaired in a later task.
- I did not run full local gates because the active regression is expected to
  fail and Task 1 is scoped to committing the failing test.
