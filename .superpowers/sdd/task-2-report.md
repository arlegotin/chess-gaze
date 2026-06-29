# Task 2 Report: Repair Fallback Candidate Arbitration

## Summary

Implemented fallback arbitration in `src/chess_gaze/face_observation.py` so fallback scoring evaluates every valid, non-seam candidate from focused regions instead of only each region selection's primary candidate.

The fallback score now applies a cross-region overlap multiplier when a candidate overlaps a valid non-seam candidate from another deterministic focused region. When no cross-region overlap evidence exists, fallback eligibility is narrowed back to each region's existing primary candidate to preserve prior single-region behavior.

## TDD RED Evidence

Command:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate -q
```

Result before implementation:

```text
F                                                                        [100%]
FAILED tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate
AssertionError: assert 'larger_single_region_false_positive' == 'cross_region_real_face'
1 failed in 0.14s
```

Root cause: `_select_fallback_face()` scored only `_primary_candidate(region_selection.selection)`. In the regression, the real face was a secondary candidate in `left_upper_band` and also appeared in `left_half`; the larger false positive was the `left_upper_band` primary, so fallback arbitration could not see the cross-region consensus candidate.

## GREEN Evidence

Command:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation_region_arbitration.py::test_mediapipe_observer_prefers_cross_region_consensus_over_larger_single_region_candidate -q
```

Result after implementation:

```text
.                                                                        [100%]
1 passed in 0.13s
```

## Focused Suite Evidence

Command:

```sh
MPLCONFIGDIR=/private/tmp/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_face_observation.py tests/chess_gaze/test_face_observation_region_arbitration.py -q
```

Result:

```text
.............................                                            [100%]
29 passed in 0.10s
```

## Files Changed

- `src/chess_gaze/face_observation.py`
- `.superpowers/sdd/task-2-report.md`

`tests/chess_gaze/test_face_observation_region_arbitration.py` already contained the Task 1 failing regression and was not modified.

## Self-Review

- Confirmed the selected fallback candidate is drawn from the originating region selection, preserving crop-to-full-frame coordinates already produced by `_candidates_from_mediapipe_result()`.
- Consensus matching ignores full-frame candidates, same-region candidates, invalid candidates, and candidates clipped by focused-region seams.
- Added a no-consensus fallback pool that keeps the previous primary-candidate-only behavior when there is no overlap evidence.
- Kept the change scoped to fallback arbitration; existing full-frame refinement scoring is unchanged.

## Concerns and Residual Risk

- Only the required focused suites were run, not the entire repository test suite.
- `src/chess_gaze/face_observation.py` is already over the source-layout review threshold, but this task's write scope did not allow a split or documentation change.
