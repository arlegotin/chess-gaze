# 3D Scene Artifact Viewer Closeout

Date: 2026-06-26

## Summary

Implemented strict 3D scene artifacts and a local 3D viewer for completed
`chess-gaze analyze` runs.

Major behavior now shipped:

- every decoded frame writes one `records/scene_frames.jsonl` record;
- `scene/scene_manifest.json` persists scene assumptions, camera model, robust
  estimator diagnostics, axis basis, monitor plane, source artifact refs, and
  vendored viewer dependency metadata;
- `scene/scene_summary.json` validates scene-frame counts and monitor-hit
  bounds;
- `viewer/scene-data.json` preserves every valid gaze hit point without
  merging, smoothing, clustering, sampling, or clamping to the physical monitor;
- generated `viewer/index.html` renders the head ellipsoid, eyes, UniGaze ray,
  monitor plane, extended plane, axes, current hit, and accumulated hit points;
- `chess-gaze analyze` prints the viewer path;
- `chess-gaze view <run-dir>` serves the viewer from `viewer/` on loopback-only
  localhost.

Primary implementation files:

- `src/chess_gaze/scene_calibration.py`
- `src/chess_gaze/scene_records.py`
- `src/chess_gaze/scene_geometry.py`
- `src/chess_gaze/scene_artifacts.py`
- `src/chess_gaze/scene_viewer.py`
- `src/chess_gaze/viewer_assets/`
- `src/chess_gaze/artifact_runs.py`
- `src/chess_gaze/pipeline.py`
- `src/chess_gaze/qa_summary.py`
- `src/chess_gaze/cli.py`

Docs updated:

- `README.md`
- `docs/development/architecture/source-layout.md`

## Default-Model Run

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

Run directory:

```text
artifacts/output/nakamura_1/runs/20260626T042921Z-d0f9cfa2
```

Viewer:

```text
artifacts/output/nakamura_1/runs/20260626T042921Z-d0f9cfa2/viewer/index.html
```

Observed native runtime notes:

- MediaPipe initialized successfully only in unsandboxed execution.
- Native MediaPipe periodically logged Clearcut upload failures. The analysis
  still exited `0`; the generated viewer does not make remote requests.
- cv2/PyAV duplicate Objective-C AVFoundation class warnings appeared during
  model and viewer commands.

Counts and validation:

- decoded frames: `1973`
- `records/frames.jsonl`: `1973`
- `records/scene_frames.jsonl`: `1973`
- valid eye midpoint frames: `1973`
- valid UniGaze ray frames: `1973`
- valid monitor hit frames: `1958`
- invalid monitor hit reasons: `RAY_INTERSECTION_BEHIND_ORIGIN: 15`
- `viewer/scene-data.json` frames: `1973`
- `viewer/scene-data.json` valid hit points: `1958`
- `scene_summary.artifact_validation`: all true
- `qa_summary.artifact_validation.schema_validation_passed`: true
- `qa_summary.artifact_validation.counts_match`: true

Axis convention evidence:

- `scene_axes_camera.convention`: `right_up_back_columns_right_handed`
- `scene_axes_camera.determinant_right_up_back`: `1.0000000000000002`
- persisted forward vector remains semantic streamer-to-monitor direction;
  transform basis persists right/up/back columns for determinant `+1`.

Robust-estimator diagnostics from the default run:

- scene center candidates: `1973`
- scene center finite candidates: `1973`
- scene center inliers: `1535`
- scene center MAD: `[0.01532704000722196, 0.011037823957597365, 0.08910786306212204]`
- scene center thresholds: `[0.05364464002527686, 0.038632383851590776, 0.31187752071742714]`
- scene center iterations: `22`
- scene center convergence tolerance: `1e-06`
- main direction candidates: `1973`
- main direction finite candidates: `1973`
- main direction inliers: `1076`
- main direction angular threshold: `0.35`
- main direction median residual: `0.14491992375700796`
- main direction residual percentiles:
  `p50=0.14491992375700796`, `p75=0.19398085703095397`,
  `p90=0.2732289373669492`, `p95=0.29896335758909287`

Monitor hit bounds:

- `u_min_m`: `-305.72883116521473`
- `u_max_m`: `0.27410863938333296`
- `v_min_m`: `-98.7621214604876`
- `v_max_m`: `1.465727504425273`
- valid hits outside the physical monitor rectangle: `911`
- valid hits outside the extended plane: `319`

This confirms hit points are persisted honestly even when far outside the
physical monitor rectangle.

## Browser Smoke

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view artifacts/output/nakamura_1/runs/20260626T042921Z-d0f9cfa2 --host 127.0.0.1 --port 0
```

URL opened:

```text
http://127.0.0.1:57503/
```

Observed:

- local network requests only: `/`, `styles.css`, `scene_viewer.js`,
  `vendor/three.module.js`, `vendor/OrbitControls.js`, `vendor/three.core.js`,
  `scene-data.json`, all `200`;
- console had no errors, warnings, or issues;
- initial canvas was nonblank (`toDataURL` length `275650`);
- hit counter showed `1958`;
- slider to frame index `1000` showed `Frame 1001 of 1973`;
- accumulated mode at frame index `1000` showed `1001 of 1958`;
- instant mode at frame index `1000` showed the current hit without the
  accumulated point cloud;
- wheel zoom changed the rendered image and kept console clean;
- mobile viewport had no horizontal overflow.

Screenshot evidence:

- desktop accumulated:
  `/private/tmp/chess-gaze-task10-default-desktop-accumulated.png`
  sha256 `68daa036bff95a087793afe61720d190858a683028cb85057a99acef0168696e`
- desktop instant:
  `/private/tmp/chess-gaze-task10-default-desktop-instant.png`
  sha256 `aa67b7fdbd44e087038ab13b829677f313cfbb0971c2709b82912968c472631e`
- wheel before:
  `/private/tmp/chess-gaze-task10-default-wheel-before.png`
  sha256 `aa67b7fdbd44e087038ab13b829677f313cfbb0971c2709b82912968c472631e`
- wheel after:
  `/private/tmp/chess-gaze-task10-default-wheel-after.png`
  sha256 `471c71f906d80b45bcf20c6e71b0bd2bd4b9e10aebb58dea712ee90475148b39`
- mobile instant:
  `/private/tmp/chess-gaze-task10-default-mobile-instant.png`
  sha256 `8f93fdafe4fb1d091c1bad2eb6ec516aa6f3f550c25c8e7a7c4c0932d01da173`

## Verification

Focused Task 10 suite before the final diagnostics repair:

```text
124 passed in 524.39s (0:08:44)
```

Targeted robust-diagnostics regression after repair:

```text
64 passed in 0.84s
```

Final full pytest:

```text
7 failed, 227 passed, 7 skipped, 18 warnings in 460.57s (0:07:40)
```

All 7 failures are absent legacy media blockers for:

- `artifacts/input/test_1.mp4`
- `artifacts/input/test_2.mp4`

Final broad available subset excluding those absent-media files:

```text
227 passed, 7 skipped, 18 warnings in 463.34s (0:07:43)
```

Static gates:

```text
UV_CACHE_DIR=.uv-cache uv run ruff check .
All checks passed!

UV_CACHE_DIR=.uv-cache uv run ruff format --check .
55 files already formatted

UV_CACHE_DIR=.uv-cache uv run mypy
Success: no issues found in 55 source files
```

## Residual Risk

- Scene coordinates are pseudo-metric, not measured room geometry. Absolute
  scale is dominated by the adult-male interpupillary-distance assumption and
  default monitor distance.
- Monitor plane distance and size are explicit constants, not inferred from a
  calibrated real monitor.
- Default-model MediaPipe emitted native Clearcut upload-failure logs during
  analysis. The command completed and the generated viewer made no remote
  requests, but the native library behavior should remain visible in future
  privacy reviews.
- Legacy real-video tests for `test_0.mp4`, `test_1.mp4`, and `test_2.mp4`
  remain blocked/skipped/failed when those local ignored media files are absent.
