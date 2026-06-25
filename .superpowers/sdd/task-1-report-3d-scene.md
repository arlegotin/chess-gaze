# Task 1 Report: Scene Constants And Strict Schemas

## What I implemented

- Added `src/chess_gaze/scene_calibration.py` with the task-required scene constants, `SceneAssumptionRecord`, frozen/strict `SceneAssumptions`, `default_scene_assumptions()`, and explicit finite validation for tuple-valued assumptions.
- Added `src/chess_gaze/scene_records.py` with the task-required enums and strict Pydantic schemas for scene frames, manifests, summaries, and viewer payloads.
- Added focused validators for:
  - finite vector/list-adjacent values;
  - unit-vector norm bounds;
  - valid/invalid record completeness rules;
  - frame-level dependency rules between eyes, midpoint, ray, and monitor hit;
  - persisted scene invalid reasons as scene diagnostic strings and scene enums only.
- Added task-owned tests for exact constants, frozen/strict assumptions, enum rejection, strict nested schema behavior, and schema-version serialization.

## RED command and failing output summary

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py -q
```

Summary:

- Pytest failed during collection as expected.
- Both new test modules raised `ModuleNotFoundError: No module named 'chess_gaze.scene_calibration'`.
- This confirmed the tests were exercising the missing Task 1 scene modules before implementation.

## GREEN command and passing output summary

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py -q
```

Summary:

- `16 passed in 0.07s`

Additional focused hygiene check:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_calibration.py src/chess_gaze/scene_records.py tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py
```

- `All checks passed!`

## Files changed

- `src/chess_gaze/scene_calibration.py`
- `src/chess_gaze/scene_records.py`
- `tests/chess_gaze/test_scene_calibration.py`
- `tests/chess_gaze/test_scene_records.py`
- `.superpowers/sdd/task-1-report-3d-scene.md`

## Self-review findings

- The implementation stays inside the Task 1 ownership boundary and leaves `FrameRecord` unchanged.
- Shared public names and schema-version literals from the task brief and binding plan sections are preserved.
- Enum parsing accepts known persisted strings via explicit coercion and still rejects unknown strings under strict validation.
- Tuple-valued assumption and radii fields now reject non-finite values explicitly, covering the gap that `StrictSchemaModel` does not cover for nested/list-like fields by itself.
- Cross-record validation is intentionally minimal and aligned to the brief: valid midpoint requires valid eyes; valid monitor hit requires a valid unigaze ray.

## Concerns, if any

- No functional concerns for Task 1.
- I did not run the broader repository test suite; verification was limited to the task-specified focused pytest command and focused Ruff checks.

## Review Fixes

### What changed

- Replaced free-form nested dicts in scene records with strict structured models for frame diagnostics, manifest source artifacts, coordinate frames, robust estimator metadata, viewer metadata, summary hit bounds, and artifact validation.
- Added spec-level scene frame fields: `source_frame_status`, `valid_for_scene_center`, `valid_for_main_monitor_direction`, and structured `camera`.
- Added alias-backed persisted names so scene payloads can serialize spec-aligned keys such as `camera_m`, `scene_m`, `ray_t_m`, `scene_axes_camera`, `main_monitor_plane`, and `viewer`.
- Fixed the camera model policy literal to `estimated_pinhole_from_image_size`.
- Added source diagnostic string fields for midpoint, head, unigaze ray, and monitor hit records.
- Added axis-basis validation for anti-parallel `back_camera`/`forward_camera` and determinant near `+1`.
- Split persisted head-radius assumption records into independent `X`, `Y`, and `Z` constant names.

### RED evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py -q
```

Summary:

- `7 failed, 13 passed in 0.13s`
- Failures matched the review findings:
  - missing separate head-radius assumption names;
  - no axis-basis invariant enforcement;
  - missing spec-level frame and manifest fields;
  - free-form diagnostics/manifest metadata drift;
  - missing structured summary fields.

### GREEN evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py -q
```

Summary:

- `20 passed in 0.07s`

Focused Ruff:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_calibration.py src/chess_gaze/scene_records.py tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py
```

- `All checks passed!`

### Files changed

- `src/chess_gaze/scene_calibration.py`
- `src/chess_gaze/scene_records.py`
- `tests/chess_gaze/test_scene_calibration.py`
- `tests/chess_gaze/test_scene_records.py`
- `.superpowers/sdd/task-1-report-3d-scene.md`

## Review Fixes Round 2

### What changed

- Added JSON round-trip coverage for a full `SceneFrameRecord`, including `head.ellipsoid_radii_m` supplied as a JSON array.
- Updated `SceneHeadRecord` radii validation to accept JSON-style lists, coerce valid length-3 input into the strict tuple shape, and still reject wrong-length and non-finite values.
- Tightened `SceneAxisBasisRecord` validation to compute the determinant from the actual `right/up/back` vectors, reject degenerate bases even when a fake determinant is supplied, and reject material mismatch between supplied and computed determinant values.
- Preserved existing anti-parallel `back_camera`/`forward_camera` validation.

### RED evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py -q
```

Summary:

- `2 failed, 20 passed in 0.09s`
- Failures matched the remaining review findings:
  - computed-degenerate axis basis was incorrectly accepted;
  - `SceneFrameRecord.model_validate_json(...)` rejected JSON-array `ellipsoid_radii_m`.

### GREEN evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py -q
```

Summary:

- `23 passed in 0.06s`

Focused Ruff:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/scene_calibration.py src/chess_gaze/scene_records.py tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py
```

- `All checks passed!`

### Files changed

- `src/chess_gaze/scene_records.py`
- `tests/chess_gaze/test_scene_records.py`
- `.superpowers/sdd/task-1-report-3d-scene.md`
