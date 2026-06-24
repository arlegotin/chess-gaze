# Frame-Level Gaze Analysis Pipeline Design

Date: 2026-06-24

## Status

This is the active design spec for the first real runtime feature in
`chess-gaze`. It intentionally stops before an implementation plan. The next
step, after user review, is a separate Superpowers implementation plan.

## Goal

Build a local Python command-line app that analyzes one chess-stream video at a
time. The app must decode every video frame, preserve the raw frame evidence,
estimate face, head, eye, iris, and gaze observations independently per frame,
write a machine-readable record for every decoded frame, and generate annotated
visualization frames for human QA.

The output must be rich enough to support later 3D reconstruction of the
streamer's head, eyes, and apparent gaze through time. The first implementation
does not render a 3D scene and does not require temporal smoothing, tracking, or
cross-frame inference.

## Stakeholders

- Primary user: the repo owner running local analysis on chess-stream clips.
- Future implementers: coding agents that must add models, tests, calibration,
  and visualization without guessing output contracts.
- Future consumers: 3D reconstruction and gaze-to-board/screen mapping tools
  that need raw evidence, coordinate systems, confidence, and invalidity
  reasons, not just final labels.

## Current State

The repository is a private, uv-managed Python 3.12 project with a metadata-only
package. Runtime dependencies are currently empty. Local gates are:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

`artifacts/`, `models/`, and common checkpoint formats are gitignored. The two
local sample videos exist under `artifacts/input/`, but they are ignored local
inputs, not committed test fixtures.

Observed sample media:

| Video | Resolution | FPS | Duration | Frames | Video codec | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `artifacts/input/test_1.mp4` | 1262x720 | 60 | 60.202995s | 3613 | AV1 | Board on left, streamer large on right, hand occlusions, downcast gaze. |
| `artifacts/input/test_2.mp4` | 1920x1080 | 60 | 32.883333s | 1973 | AV1 | Streamer lower-left, board on right, headphones, microphone, profile/down-right gaze. |

Manual frame sampling found that the streamer is visible in the sampled frames,
but face location, scale, pose, and occlusion differ materially between videos.
The app must not assume a fixed facecam quadrant, a fixed board location, or an
unobstructed chessboard.

## Primary Interface

The public command is:

```sh
uv run chess-gaze analyze artifacts/input/test_1.mp4
```

Required behavior:

- Accept one video path.
- Default output root to `artifacts/output/`.
- Default model root to `models/`.
- Reject missing, unreadable, or unsupported video inputs before creating a
  partial run directory.
- Verify required local model assets and their checksums before processing.
- Never download model assets implicitly during analysis.
- Exit nonzero with a clear message when required model assets are missing.

Allowed initial optional arguments:

```sh
uv run chess-gaze analyze <video_path> \
  --output-root artifacts/output \
  --models-root models \
  --config path/to/config.json
```

No batch mode is required in this spec.

## Output Layout

Use one immutable run directory per analysis attempt. The output path must inherit
the input file stem while still avoiding destructive reruns:

```text
artifacts/
  input/
    test_1.mp4
    test_2.mp4
  output/
    test_1/
      runs/
        20260624T215933Z-a1b2c3d4/
          run_manifest.json
          calibration.json
          video_manifest.json
          qa_summary.json
          raw_frames/
            f000000000.png
            f000000001.png
          processed_frames/
            f000000000.jpg
            f000000001.jpg
          crops/
            face/
              f000000000.png
            eyes/
              left/
                f000000000.png
              right/
                f000000000.png
          records/
            frames.jsonl
            errors.jsonl
```

Hard invariants:

- `raw_frames/` contains unannotated decoded full frames.
- Raw frames are lossless PNG by default, not JPEG.
- `processed_frames/` contains annotated visualization frames.
- Processed frames are JPEG by default with explicit quality stored in
  `run_manifest.json`.
- Frame IDs are zero-padded decode-order IDs: `f000000000`, `f000000001`, ...
- Every decoded source frame has exactly one JSONL record in
  `records/frames.jsonl`.
- Failed observations still produce a frame record and a processed frame.
- Relative artifact paths inside records are relative to the run directory.
- A run directory is never overwritten. Re-running the same video creates a new
  run ID.

## Coordinate Spaces

Every coordinate-bearing field must declare or imply one of these named spaces:

| Space | Meaning |
| --- | --- |
| `image_px` | Source decoded image pixels, origin top-left, x right, y down. |
| `image_norm` | Source image coordinates normalized to `[0.0, 1.0]`. |
| `face_crop_px` | Pixels in the saved face crop after crop transform. |
| `left_eye_crop_px` | Pixels in the saved left-eye crop after crop transform. |
| `right_eye_crop_px` | Pixels in the saved right-eye crop after crop transform. |
| `camera_3d_m` | Right-handed estimated camera space in meters. |
| `head_3d_m` | Estimated head-local 3D space. |
| `board_norm` | Future normalized chessboard plane coordinates. |
| `board_square` | Future algebraic square if board mapping is known. |

The first implementation may leave `board_norm` and `board_square` null. It must
not fake board intersections without calibration.

## Selected Technical Stack

The first implementation should use the best available local, reproducible stack
for Python 3.12 on this repo. Exact versions are resolved and locked by `uv`
during implementation.

| Concern | Selected tool | Reason |
| --- | --- | --- |
| Video decode and timestamps | `av` / PyAV | Direct FFmpeg access to containers, streams, frames, PTS, time base, and frame metadata. Better fit than `cv2.VideoCapture` for faithful frame records. |
| Dense face, eye, iris landmarks | MediaPipe Face Landmarker | Maintained Python package, current Python 3.12 support, 478 3D face landmarks, blendshapes, and facial transform matrices. |
| Per-frame no-smoothing mode | MediaPipe Face Landmarker `IMAGE` mode | Avoids MediaPipe video/live tracking behavior and aligns with frame-independent analysis. |
| Geometry, PnP, overlays | `opencv-python-headless` | `solvePnP`, reprojection, drawing, color conversion, and image encoding without GUI dependencies. Install exactly one OpenCV wheel flavor. |
| Arrays | `numpy` | Common data representation for PyAV, MediaPipe, OpenCV, and model preprocessing. |
| Schema validation | `pydantic` v2 | Strict typed manifests and records at artifact boundaries. |
| Learned apparent gaze | L2CS-Net with Gaze360 weights | Practical open gaze estimator with published MPIIGaze/Gaze360 evaluation and usable PyTorch inference path. |
| Learned gaze runtime | `torch` and `torchvision` | Required for the initial L2CS-Net implementation and available for Python 3.12. |

Model assets are local artifacts under `models/`, not committed source:

```text
models/
  manifest.json
  mediapipe/
    face_landmarker.task
  l2cs/
    L2CSNet_gaze360.pkl
```

`models/manifest.json` must store model IDs, source URLs, local paths,
checksums, license notes, expected input shapes, and model task names. Analysis
must fail before frame processing if any required model file is missing or has a
checksum mismatch.

## Library Findings and Sources

High-impact findings verified on 2026-06-24:

- MediaPipe Face Landmarker outputs dense face landmarks, optional blendshapes,
  and optional facial transformation matrices. Its current overview documents
  478 3D landmarks and the Python package supports Python 3.12. Sources:
  `https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker`
  and `https://pypi.org/project/mediapipe/`.
- MediaPipe video/live modes use tracking to reduce model invocations. For this
  app, measurement passes must use `IMAGE` mode to avoid hiding per-frame
  failures. Source:
  `https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python`.
- MediaPipe Iris documentation explicitly says iris tracking does not infer
  where people are looking. Therefore iris landmarks are evidence for per-eye
  geometry, not a standalone true fixation model. Source:
  `https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/`.
- OpenCV `solvePnP` estimates rotation and translation from 3D object points and
  2D image projections by minimizing reprojection error for supported methods.
  Source: `https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html`.
- PyAV exposes direct FFmpeg access to containers, streams, packets, codecs, and
  frames, with Python 3.12 classifier support. Source:
  `https://pypi.org/project/av/`.
- L2CS-Net is the official PyTorch implementation for fine-grained gaze
  estimation and tracking, with Gaze360 weights. Sources:
  `https://github.com/Ahmednull/L2CS-Net` and
  `https://arxiv.org/abs/2203.03339`.

## Rejected or Deferred Alternatives

| Alternative | Decision | Reason |
| --- | --- | --- |
| OpenFace 2.2 as core | Reject for first implementation | Strong all-in-one facial behavior toolkit, but last release is 2019, build/tooling burden is high, and licensing is non-commercial research oriented. |
| OpenFace 3.0 | Defer | Promising 2025 research direction, but not yet a clearer Python 3.12 local dependency than MediaPipe plus L2CS-Net for this repo. |
| InsightFace SCRFD as core | Defer and license-gate | Excellent face detection/alignment. Code is MIT, but pretrained model licensing is explicitly non-commercial research for common model packs. Use only as optional fallback if license is acceptable. |
| dlib 68-point landmarks | Reject | Not enough eye/iris detail, weaker fit for occlusion and profile gaze, build friction on current Python/macOS. |
| MTCNN or face-only detectors | Reject as core | Face boxes and 5 landmarks are insufficient for iris, per-eye metrics, and head/eye geometry. |
| `cv2.VideoCapture` as decoder | Reject as primary | Too lossy for frame identity, PTS, time-base, and variable-frame-rate evidence. OpenCV remains useful for geometry and drawing. |
| Temporal smoothing or tracking | Reject for this spec | User requirement is independent frame analysis and preservation of true per-frame gaze evidence. |

## Constants Policy

No magic numbers are allowed in runtime code. Any threshold, model setting,
camera assumption, crop padding, encoding quality, or anatomical prior must be a
named configuration field or named constant persisted into `calibration.json` or
`run_manifest.json`.

Initial named defaults:

| Constant | Initial value | Stored in | Reason |
| --- | ---: | --- | --- |
| `raw_frame_image_format` | `png` | `run_manifest.json` | Lossless full-frame evidence. |
| `processed_frame_image_format` | `jpg` | `run_manifest.json` | Compact annotated QA output. |
| `processed_frame_jpeg_quality` | `95` | `run_manifest.json` | Explicit lossy visualization quality. |
| `max_face_candidates` | `4` | `calibration.json` | Preserve ambiguity and avoid a hidden single-face assumption. |
| `candidate_face_confidence_min` | `0.25` | `calibration.json` | Keep low-confidence difficult faces for review. |
| `usable_face_confidence_min` | `0.50` | `calibration.json` | Initial status threshold, tuned by evidence later. |
| `usable_eye_confidence_min` | `0.50` | `calibration.json` | Initial eye status threshold, tuned by evidence later. |
| `default_iris_diameter_mm` | `11.7` | `calibration.json` | Published MediaPipe Iris depth prior. |
| `default_iris_diameter_uncertainty_mm` | `0.5` | `calibration.json` | Published population variation range for iris diameter. |
| `camera_intrinsics_policy` | `estimate_with_explicit_uncertainty` | `calibration.json` | Avoid pretending uncalibrated camera geometry is exact. |

The implementation may adjust default thresholds only with documented smoke
evidence. The names and persistence are mandatory.

## Run-Level Calibration and Setup Constants

The streamer setup is assumed to be static within a video. The app may derive
run-level setup constants from all successfully observed frames, but these
constants are calibration metadata, not smoothing inputs. They must not rewrite,
average, or otherwise alter per-frame face, eye, iris, head pose, or gaze
measurements.

Allowed derived setup constants:

- median and percentile range of selected face bbox size in `image_px`
- median and percentile range of inter-pupil distance in `image_px`
- median and percentile range of iris diameter per eye in `image_px`
- estimated camera intrinsics policy and uncertainty
- estimated streamer facecam region, marked as `derived_roi`, for QA only
- detected mirror policy if there is clear evidence, otherwise `unknown`

Each derived constant must store:

- value
- unit
- coordinate space
- derivation method
- number of contributing frames
- confidence or uncertainty
- whether the value is used for measurement, QA only, or future use

Derived constants are allowed to help future 3D reconstruction. They are not
allowed to hide frame-level errors, interpolate missing eyes, or correct gaze
directions across time in this spec.

## Pipeline Behavior

### 1. Input Verification

Before frame decoding:

- Resolve the input video path.
- Compute `source_sha256`.
- Read container, stream, duration, frame count when available, dimensions,
  nominal FPS, time base, codec, color range, and rotation metadata.
- Verify required models and checksums.
- Create a new immutable run directory only after input and model validation.

### 2. Frame Decode

For each decoded video frame, in decode order:

- Assign `frame_index` and `frame_id`.
- Preserve `pts`, `pts_seconds`, `duration_seconds` when available.
- Convert to full-frame RGB for model use.
- Save a raw full-frame PNG.
- Compute `raw_frame_sha256`.
- Compute frame quality metrics:
  - blur score
  - exposure score
  - image dimensions
  - color conversion metadata
  - decode warnings

No frame sampling, skipping, dropping, or duplicate suppression is allowed in the
first implementation.

### 3. Face and Landmark Observation

For each frame, run MediaPipe Face Landmarker in `IMAGE` mode.

Required outputs:

- All face candidates up to `max_face_candidates`.
- Candidate bbox in `image_px` and `image_norm`.
- Candidate confidence scores.
- MediaPipe 478 landmark list for each candidate, in normalized coordinates and
  converted pixel coordinates.
- Optional blendshapes when provided.
- Optional facial transformation matrix when provided.
- Primary face selection result and reason.

Primary face selection is per-frame only:

1. If there is one candidate, select it.
2. If there are multiple candidates, score by explicit formula:
   `candidate_confidence * candidate_area_fraction`.
3. Store every candidate and the selected candidate ID.
4. If no candidate passes `candidate_face_confidence_min`, mark `face.present`
   false and write `FACE_NOT_FOUND`.

The implementation must never silently discard alternate face candidates.

### 4. Eye and Iris Observation

For the selected face, derive both eyes independently.

Required outputs per eye:

- `present`
- `confidence`
- `reason_missing`
- eye landmark group in `image_px` and `image_norm`
- iris landmark group in `image_px` and `image_norm`
- iris center in `image_px`
- iris radius or diameter estimate in pixels
- eyelid/eye contour bbox in `image_px`
- eye crop path
- eye crop transform back to `image_px`
- normalized iris offset inside the eye aperture
- eye aspect ratio or equivalent open/closed metric
- occlusion state: `none`, `partial`, `severe`, or `unknown`

Left and right eye records are independent. If one eye is missing or occluded,
the other eye may still be valid. A frame must not be globally failed merely
because one eye is invalid.

### 5. Head Pose Observation

The app must preserve both model-provided and geometry-derived pose evidence:

- MediaPipe facial transformation matrix when available.
- OpenCV `solvePnP` result from named stable landmark correspondences.
- Rotation as matrix, quaternion, and yaw/pitch/roll radians when valid.
- Translation vector only with intrinsics source and uncertainty.
- Reprojection error for the PnP fit.
- Pose validity and invalidity reason.

If camera intrinsics are estimated rather than calibrated, records must say so.
No metric 3D translation may be treated as ground truth without calibrated
camera intrinsics.

### 6. Gaze Observation

The first implementation must distinguish three gaze evidence layers:

1. `per_eye_geometric_gaze`
   - Computed independently for left and right eyes from iris center, eye
     aperture geometry, iris size, and head pose.
   - Represents per-eye apparent gaze proxy and optional estimated ray, not a
     guaranteed screen fixation point.
2. `face_model_gaze`
   - Computed by L2CS-Net on the selected face crop.
   - Represents face-level apparent gaze yaw/pitch.
   - Must not be copied into left and right eye fields as if it were independent
     per-eye evidence.
3. `recommended_gaze`
   - A per-frame synthesized apparent gaze result.
   - Valid only when required per-eye geometry and face-level model evidence are
     present and their disagreement is within named thresholds.
   - If evidence disagrees, store both measurements and mark
     `recommended_gaze.valid=false` with reason `GAZE_ESTIMATORS_DISAGREE`.

The first implementation must not claim true board square, screen coordinate,
or real-world fixation point. Those require calibration and are explicitly
future work.

### 7. Visualization

Every decoded frame must have a processed visualization frame.

Visualization overlays must include, when available:

- selected face bbox
- alternate face candidate bboxes in a different style
- face confidence
- major face landmarks
- left and right eye contours
- left and right iris centers
- per-eye geometric gaze vectors
- face-level L2CS gaze vector
- head pose axes
- frame status and error code summary

When face or eyes are not found, the visualization must still exist and must show
the failure status rather than silently omitting the frame.

Visualization frames are QA artifacts only. They must never be used as model
input or source evidence.

## Record Schemas

Schema validation is part of the feature. Each schema has a version string so
future changes can be migrated intentionally.

### `run_manifest.json`

```json
{
  "schema_version": "gaze-run-v1",
  "run_id": "20260624T215933Z-a1b2c3d4",
  "created_at": "2026-06-24T21:59:33Z",
  "source_video_path": "artifacts/input/test_1.mp4",
  "artifact_root": "artifacts/output/test_1/runs/20260624T215933Z-a1b2c3d4",
  "video_manifest_path": "video_manifest.json",
  "calibration_path": "calibration.json",
  "frames_jsonl_path": "records/frames.jsonl",
  "errors_jsonl_path": "records/errors.jsonl",
  "qa_summary_path": "qa_summary.json",
  "raw_frame_image_format": "png",
  "processed_frame_image_format": "jpg",
  "processed_frame_jpeg_quality": 95,
  "models": [
    {
      "model_id": "mediapipe-face-landmarker",
      "task": "face_eye_iris_landmarks",
      "path": "models/mediapipe/face_landmarker.task",
      "sha256": "..."
    },
    {
      "model_id": "l2cs-gaze360",
      "task": "face_level_gaze",
      "path": "models/l2cs/L2CSNet_gaze360.pkl",
      "sha256": "..."
    }
  ],
  "environment": {
    "python": "3.12.x",
    "platform": "...",
    "package_version": "0.1.0",
    "git_commit": "..."
  },
  "status": "complete"
}
```

### `video_manifest.json`

```json
{
  "schema_version": "gaze-video-v1",
  "video_id": "test_1",
  "source_path": "artifacts/input/test_1.mp4",
  "source_sha256": "...",
  "container": "mp4",
  "video_codec": "av1",
  "width_px": 1262,
  "height_px": 720,
  "rotation_degrees": 0,
  "duration_seconds": 60.202995,
  "nominal_fps": 60.0,
  "time_base": "...",
  "frame_count_expected": 3613,
  "frame_count_decoded": 3613,
  "color_space": "bt709",
  "decode_warnings": []
}
```

### `calibration.json`

```json
{
  "schema_version": "gaze-calibration-v1",
  "calibration_id": "default-uncalibrated-v1",
  "coordinate_spaces": [
    "image_px",
    "image_norm",
    "face_crop_px",
    "left_eye_crop_px",
    "right_eye_crop_px",
    "camera_3d_m",
    "head_3d_m",
    "board_norm",
    "board_square"
  ],
  "face_selection": {
    "max_face_candidates": 4,
    "candidate_face_confidence_min": 0.25,
    "usable_face_confidence_min": 0.5,
    "selection_formula": "candidate_confidence * candidate_area_fraction"
  },
  "eyes": {
    "usable_eye_confidence_min": 0.5,
    "default_iris_diameter_mm": 11.7,
    "default_iris_diameter_uncertainty_mm": 0.5
  },
  "camera": {
    "intrinsics": null,
    "intrinsics_source": "estimated",
    "intrinsics_uncertainty": "high",
    "distortion_coefficients": null
  },
  "derived_setup_constants": {
    "inter_pupil_distance_px": null,
    "iris_diameter_left_px": null,
    "iris_diameter_right_px": null,
    "derived_facecam_roi_image_px": null,
    "mirror_policy": "unknown",
    "measurement_usage": "qa_and_future_reconstruction_only"
  },
  "board": {
    "mapping_status": "not_in_scope",
    "orientation": null,
    "homography_image_to_board": null
  }
}
```

### `records/frames.jsonl`

Each line is one frame record:

```json
{
  "schema_version": "gaze-frame-v1",
  "run_id": "20260624T215933Z-a1b2c3d4",
  "video_id": "test_1",
  "frame_id": "f000000000",
  "frame_index": 0,
  "pts": 0,
  "pts_seconds": 0.0,
  "duration_seconds": 0.0166666667,
  "source": {
    "raw_frame_path": "raw_frames/f000000000.png",
    "raw_frame_sha256": "...",
    "processed_frame_path": "processed_frames/f000000000.jpg"
  },
  "status": "ok",
  "errors": [],
  "quality": {
    "blur_score": 0.0,
    "exposure_score": 0.0,
    "occlusion": "none",
    "usable_for_gaze": true
  },
  "face": {
    "present": true,
    "primary_candidate_id": "face_0",
    "selection_reason": "single_candidate",
    "candidates": [
      {
        "candidate_id": "face_0",
        "confidence": 0.98,
        "bbox_image_px": [720, 120, 1180, 690],
        "bbox_image_norm": [0.5705, 0.1667, 0.9350, 0.9583],
        "landmarks_image_px": [],
        "landmarks_image_norm": [],
        "blendshapes": [],
        "facial_transformation_matrix": null
      }
    ],
    "head_pose": {
      "available": true,
      "method": "mediapipe_transform_and_solvepnp",
      "rotation_radians": {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0
      },
      "rotation_matrix": [],
      "quaternion": [],
      "translation_camera_3d_m": null,
      "intrinsics_source": "estimated",
      "reprojection_error_px": null,
      "reason_invalid": null
    }
  },
  "eyes": {
    "left": {
      "present": true,
      "confidence": 0.91,
      "reason_missing": null,
      "eye_crop_path": "crops/eyes/left/f000000000.png",
      "bbox_image_px": [0, 0, 0, 0],
      "landmarks_image_px": [],
      "iris_landmarks_image_px": [],
      "iris_center_image_px": [0.0, 0.0],
      "iris_diameter_px": 0.0,
      "normalized_iris_offset_xy": [0.0, 0.0],
      "eye_open_metric": 0.0,
      "occlusion": "none",
      "per_eye_geometric_gaze": {
        "valid": true,
        "yaw_radians": 0.0,
        "pitch_radians": 0.0,
        "ray_camera_3d": null,
        "confidence": 0.0,
        "reason_invalid": null
      }
    },
    "right": {
      "present": true,
      "confidence": 0.89,
      "reason_missing": null,
      "eye_crop_path": "crops/eyes/right/f000000000.png",
      "bbox_image_px": [0, 0, 0, 0],
      "landmarks_image_px": [],
      "iris_landmarks_image_px": [],
      "iris_center_image_px": [0.0, 0.0],
      "iris_diameter_px": 0.0,
      "normalized_iris_offset_xy": [0.0, 0.0],
      "eye_open_metric": 0.0,
      "occlusion": "none",
      "per_eye_geometric_gaze": {
        "valid": true,
        "yaw_radians": 0.0,
        "pitch_radians": 0.0,
        "ray_camera_3d": null,
        "confidence": 0.0,
        "reason_invalid": null
      }
    }
  },
  "gaze": {
    "face_model_gaze": {
      "valid": true,
      "method": "l2cs_gaze360",
      "yaw_radians": 0.0,
      "pitch_radians": 0.0,
      "confidence": 0.0,
      "reason_invalid": null
    },
    "recommended_gaze": {
      "valid": true,
      "yaw_radians": 0.0,
      "pitch_radians": 0.0,
      "target_image_px": null,
      "target_board_norm": null,
      "target_square": null,
      "confidence": 0.0,
      "reason_invalid": null
    }
  },
  "provenance": {
    "models": [
      "mediapipe-face-landmarker",
      "l2cs-gaze360"
    ],
    "calibration_id": "default-uncalibrated-v1",
    "record_created_at": "2026-06-24T21:59:33Z"
  }
}
```

The placeholder empty arrays above represent schema shape only. Real records
must contain actual landmarks or explicit invalidity reasons.

### `records/errors.jsonl`

Each line is a machine-readable warning/error:

```json
{
  "schema_version": "gaze-error-v1",
  "run_id": "20260624T215933Z-a1b2c3d4",
  "video_id": "test_1",
  "frame_id": "f000000120",
  "severity": "warning",
  "surface": "face_observation",
  "code": "FACE_NOT_FOUND",
  "message": "No face candidate passed candidate_face_confidence_min",
  "recoverable": true,
  "artifact_refs": [
    "processed_frames/f000000120.jpg"
  ],
  "evidence": {
    "best_candidate_confidence": 0.12
  }
}
```

Required error codes:

- `INPUT_NOT_FOUND`
- `UNSUPPORTED_VIDEO`
- `MODEL_ASSET_MISSING`
- `MODEL_ASSET_CHECKSUM_MISMATCH`
- `FRAME_DECODE_FAILED`
- `RAW_FRAME_WRITE_FAILED`
- `FACE_NOT_FOUND`
- `MULTIPLE_FACE_CANDIDATES`
- `PRIMARY_FACE_LOW_CONFIDENCE`
- `LEFT_EYE_NOT_FOUND`
- `RIGHT_EYE_NOT_FOUND`
- `LEFT_IRIS_NOT_FOUND`
- `RIGHT_IRIS_NOT_FOUND`
- `HEAD_POSE_INVALID`
- `GAZE_MODEL_FAILED`
- `GAZE_ESTIMATORS_DISAGREE`
- `SCHEMA_VALIDATION_FAILED`

### `qa_summary.json`

The QA summary must aggregate run-level trust signals:

- source hash, frame counts, decode warnings
- record count equals decoded frame count
- raw frame file count equals decoded frame count
- processed frame file count equals decoded frame count
- JSON schema validation pass/fail
- face present rate
- both-eyes-present rate
- left-eye-only and right-eye-only rates
- iris present rate per eye
- head pose valid rate
- face-level gaze valid rate
- recommended gaze valid rate
- error counts by code and severity
- worst 20 frames by blur score
- worst 20 frames by exposure score
- representative failure frame IDs
- 30 deterministic QA sample frame IDs per video when available

## Acceptance Criteria

1. `uv run chess-gaze analyze artifacts/input/test_1.mp4` completes and writes a
   new immutable run directory under `artifacts/output/test_1/runs/`.
2. `uv run chess-gaze analyze artifacts/input/test_2.mp4` completes and writes a
   new immutable run directory under `artifacts/output/test_2/runs/`.
3. For `test_1.mp4`, `video_manifest.json` reports 3613 decoded frames unless
   PyAV decode evidence proves a different count from the local file.
4. For `test_2.mp4`, `video_manifest.json` reports 1973 decoded frames unless
   PyAV decode evidence proves a different count from the local file.
5. For every decoded frame, exactly one raw frame exists.
6. For every decoded frame, exactly one processed frame exists.
7. For every decoded frame, exactly one `frames.jsonl` line exists.
8. A frame with no face candidate still has a frame record, processed frame, and
   `FACE_NOT_FOUND` error record.
9. A frame with one missing eye still preserves the other eye's independent
   measurements when available.
10. MediaPipe is run in `IMAGE` mode for measurement output.
11. No temporal smoothing, tracking, interpolation, or across-frame averaging is
    applied to face, eye, iris, head pose, or gaze outputs.
12. L2CS-Net face-level gaze is stored separately from independent per-eye
    geometric gaze. It is not copied into left/right eye fields.
13. Records never claim `target_square` or calibrated screen target unless a
    calibration file provides the needed mapping.
14. Every coordinate-bearing field is in a named coordinate space.
15. Every threshold and anatomical/camera assumption is named and persisted in a
    manifest or calibration file.
16. `qa_summary.json` validates that artifact counts and JSONL line counts match
    decoded frame count.
17. On the two local sample videos, `FACE_NOT_FOUND` is no more than 1 percent of
    decoded frames unless manual visualization review proves the face is absent
    or fully hidden in those frames.
18. On the two local sample videos, frames with both eyes visibly open in the
    processed visualization should have both iris centers marked. Misses must be
    listed in `qa_summary.json` representative failures.
19. The app runs fully locally and does not send video frames, crops, metadata,
    or model inputs to a remote service.
20. The standard repo gates pass after implementation:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Testing Strategy

Implementation must be test-first under the repo contract.

Required automated tests:

| Layer | Required checks |
| --- | --- |
| Unit | Path-to-run layout, frame ID formatting, constants persistence, coordinate-space validation, schema validation, error-code serialization. |
| Unit | Primary face selection formula preserves all candidates and explains selection. |
| Unit | Missing face, missing left eye, missing right eye, missing iris, and gaze disagreement all produce valid records. |
| Integration | Decode a tiny synthetic video and verify raw frame count, processed frame count, and JSONL line count match decoded frames. |
| Integration | Run with fake/model-stub observers to prove the artifact contract without heavyweight ML dependencies. |
| Smoke | Run real analysis on `artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4` when model assets are present locally. |
| Manual QA | Inspect deterministic processed-frame samples from each local test video. |

Tests must not require committing ignored video files or model weights. Real
model smoke tests may be opt-in if model assets are absent, but the absence must
be reported clearly rather than hidden.

## Best Practices and Mistakes to Avoid

Do:

- Preserve raw full-frame evidence before analysis.
- Treat every frame independently.
- Store invalidity reasons instead of silently dropping hard frames.
- Keep left and right eye observations independent.
- Store model versions, checksums, and configuration.
- Use explicit coordinate spaces and crop transforms.
- Record estimator disagreement rather than hiding it through fusion.
- Keep artifacts gitignored.
- Use local deterministic output naming.

Avoid:

- Using MediaPipe `VIDEO` or `LIVE_STREAM` mode for measurement records.
- Treating MediaPipe iris landmarks alone as true gaze target prediction.
- Treating L2CS-Net face-level output as independent per-eye gaze.
- Using only the final inferred square or a single gaze label.
- Assuming the streamer is always in one screen quadrant.
- Assuming the largest face is always correct without recording candidates.
- Assuming the chessboard is visible, unobstructed, or needed for gaze.
- Auto-downloading unpinned model weights during analysis.
- Writing magic thresholds directly inside processing code.
- Overwriting prior runs.
- Using visualization frames as source evidence.

## Source Layout Direction

The implementation should add modules only where they own meaningful behavior.
Avoid generic `core`, `services`, `adapters`, `engine`, or `domain` packages.

Likely domain module names:

- `video_decode` for source video inspection and frame iteration.
- `artifact_runs` for run-directory creation and artifact paths.
- `frame_records` for schema models and JSONL writing.
- `calibration` for constants, camera assumptions, and named thresholds.
- `face_observation` for MediaPipe face/landmark observation.
- `eye_observation` for eye/iris measurements.
- `head_pose` for transform and PnP pose estimates.
- `gaze_observation` for per-eye geometry and L2CS face-level gaze.
- `visualization` for processed frame overlays.
- `qa_summary` for aggregate validation and sampling.

If any source file crosses roughly 800 lines or three runtime responsibilities,
stop and perform a source-layout review before adding more behavior.

## Rollback

This feature writes only gitignored artifacts and adds source/config/docs to the
repo. Rollback is a normal git revert for source/config/docs plus deleting any
ignored run directories under `artifacts/output/` if desired. The app must never
modify input videos.

## Open Assumptions

- The streamer is a single adult male in the webcam view, but the app must still
  record multiple face candidates if detected.
- The webcam setup is static within a video, but the first implementation does
  not use temporal calibration or smoothing.
- The first implementation estimates apparent gaze from monocular video. It
  does not prove the exact screen pixel, monitor, chess square, or real-world
  target being viewed.
- Model assets can live in local ignored `models/` with checksums documented in
  `models/manifest.json`.
- Local smoke videos are available to the repo owner but are not committed
  fixtures.

## Out of Scope

- 3D scene rendering.
- Chessboard detection, board orientation, move parsing, or square-level gaze.
- Screen, monitor, camera, or room calibration UI.
- Temporal smoothing, tracking, or across-frame gaze correction.
- Training or fine-tuning models.
- Cloud inference.
- Batch video processing.
- Real-time livestream ingestion.
- GUI/web UI.
- Committing input videos, output artifacts, or model weights.

## Plan Gate

Do not write the implementation plan until this spec is reviewed. The plan must
convert this design into test-first implementation steps and must preserve the
no-smoothing, per-frame artifact, and independent per-eye requirements.
