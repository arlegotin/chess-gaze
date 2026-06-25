# Frame-Level Gaze Analysis Pipeline Design

Date: 2026-06-24

Updated: 2026-06-25

## Status

This is the active design spec for the first real runtime feature in
`chess-gaze`. The executable Superpowers implementation plan is
`docs/superpowers/plans/2026-06-25-frame-gaze-analysis-pipeline.md`.

2026-06-25 correction: the initial model choice did not meet the project's
accuracy-first standard because it selected L2CS-Net without a current,
candidate-complete comparison against UniGaze. This revision makes UniGaze
`unigaze_h14_joint` the primary learned gaze model, records license/runtime
caveats explicitly, and tightens other dependency and artifact decisions found
during the follow-up audit.

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
local verification videos exist under `artifacts/input/`, but they are ignored
local inputs, not committed test fixtures.

These two local verification videos are mandatory real-data verification inputs
for this spec. They are not illustrative examples. Every subsystem that can be
exercised with a real video must be checked against
`artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4` as soon as that
subsystem is executable, before later tasks build on it. Synthetic fixtures,
fakes, and unit tests may define deterministic contracts, but they do not
replace required real-data verification for any subsystem that can consume real
video data.

Observed mandatory verification media:

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

## Secrets and Environment

`.env` is a local ignored secrets file. `HF_TOKEN` may be read from `.env` or
the process environment only by explicit setup/prefetch tooling that downloads
model assets into `models/` before analysis. The analysis command itself must
not require, read for network access, log, persist, or transmit `HF_TOKEN`.

`HF_TOKEN` is not a substitute for the local-model policy. Any model fetched with
the token must still be checksum-verified, recorded in the ignored local
`models/manifest.json`, matched against the committed model registry, and loaded
from local disk during `chess-gaze analyze`.

No other secret or credential is required by this spec. The repo owner granted
license/use approval for UniGaze `unigaze_h14_joint` under MG-NC-RAI-2.0 on
2026-06-25. This approval is a local policy decision and must be recorded as
configuration or registry metadata, not treated as a secret.

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
- Raw and processed image writes use temp-then-rename atomic writes.
- Frame IDs are zero-padded decoder-emission IDs in presentation order:
  `f000000000`, `f000000001`, ...
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
| `camera_ray` | Unitless right-handed camera-relative direction vectors. |
| `camera_3d_m` | Right-handed metric camera space in meters; valid only when calibrated intrinsics and a scale source are available. |
| `head_3d_m` | Estimated head-local 3D space. |
| `board_norm` | Future normalized chessboard plane coordinates. |
| `board_square` | Future algebraic square if board mapping is known. |

The first implementation may leave `board_norm` and `board_square` null. It must
not fake board intersections without calibration.

The first implementation must not populate metric `camera_3d_m` translations
when camera intrinsics or scale are uncalibrated. In that case, store null metric
translation, preserve head/gaze angles and `camera_ray` direction vectors, and
record the intrinsics state as unavailable or estimated-with-high-uncertainty.

## Model and Library Selection Matrix

High-impact dependency choices were re-audited on 2026-06-25 using primary
sources. The table records decisions for the first implementation; unresolved
items become implementation gates, not hidden assumptions.

| Candidate | Task fit | Primary sources | Availability | License | Runtime/platform fit | Caveats | User-provided | Decision | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniGaze `unigaze_h14_joint` | Selected fit for quality-first face-level apparent gaze: current universal gaze estimator, UniGaze-H backbone, joint-dataset training, video inference script, PyPI loader. | GitHub README, PyPI, Hugging Face model card, arXiv/project page, verified 2026-06-25. | PyPI `unigaze` 0.1.3; HF repo provides `unigaze_h14_joint.safetensors` around 2.53 GB. | ModelGo Attribution-NonCommercial-ResponsibleAI v2.0. License-gate before use. | Python package supports Python >=3.8, but upstream quickstart pins older PyTorch. Use Python 3.12-compatible PyTorch only after local smoke. GPU expected for H14. | Loader auto-downloads from HF by default; analysis must use local pre-fetched weights. Input is normalized `(B, 3, 224, 224)`, output is `pred_gaze` `(pitch, yaw)`, no confidence. Requires UniGaze-compatible normalization, not an arbitrary face crop. | Yes | Select as primary learned gaze model. | High for model choice, medium for local integration until smoke-tested. |
| L2CS-Net with Gaze360 weights | Usable older apparent-gaze baseline, but no longer the strongest known candidate for this accuracy-first task. | Official repo and paper, verified in initial spec. | Public repo/checkpoints exist. | Must be checked per checkpoint before use. | PyTorch inference path is practical. | Prior selection was based on practicality and familiarity rather than a current model comparison. Face-level only, not per-eye. | No | Reject as primary; may be added later only as a benchmark comparator. | High |
| MediaPipe Face Landmarker | Strong fit for local face, iris, landmark, blendshape, and transform evidence. | Google Face Landmarker docs and PyPI, verified 2026-06-24. | Python package and `.task` model available. | Check model task license before asset registry acceptance. | Python 3.12 support; use `IMAGE` mode for independent frames. | Do not assume per-candidate confidence is exposed. Store nullable model scores and deterministic fallback selection provenance. | No | Keep. | High |
| MediaPipe Iris landmarks as gaze estimator | Good iris evidence source, poor standalone gaze estimator. | Google MediaPipe Iris blog/docs, verified 2026-06-24. | Included via face landmarker outputs. | Same asset gate as MediaPipe model. | Local, lightweight compared with learned gaze. | Docs state iris tracking does not infer where people look. | No | Keep only as geometric evidence, not true gaze prediction. | High |
| OpenFace 2.2 / 3.0 as core | Useful reference/alternative all-in-one facial behavior stack. | OpenFace release/research sources, verified in initial spec. | 2.2 is older and build-heavy; 3.0 remains research/dependency evaluation work. | License constraints must be checked before any use. | Less direct fit for Python 3.12 local pipeline. | Higher integration cost than MediaPipe plus UniGaze for first implementation. | No | Reject 2.2 as core; defer 3.0. | Medium |
| PyAV decoder | Best fit for faithful frame identity and metadata. | PyAV package/docs, verified 2026-06-24. | Python package available. | BSD-style project license; verify package metadata during lock. | Python 3.12 classifier. | PTS/duration/frame-count may be absent or misleading; store nullable values and FFmpeg/PyAV versions. | No | Keep. | High |
| `cv2.VideoCapture` as decoder | Convenient but weaker evidence model. | OpenCV docs and known API behavior. | Available via OpenCV wheels. | OpenCV license acceptable for library, verify wheel metadata. | Easy install. | Loses too much timing/container evidence for this project. | No | Reject as primary decoder. | High |
| `opencv-python-headless` for geometry/overlays | Good fit for PnP, image transforms, drawing, color conversion, and encoding. | OpenCV docs, verified 2026-06-24. | Python wheels available. | Verify package metadata during lock. | Install exactly one OpenCV provider. | RGB/BGR boundary must be explicit to avoid color-swapped artifacts. | No | Keep. | High |
| Pydantic v2 schemas | Good fit for strict artifact-boundary validation. | Pydantic docs/package metadata to verify during implementation. | Python package available. | Verify package metadata during lock. | Python 3.12 support. | Must use strict models, finite floats, `extra="forbid"`, and post-run artifact re-read validation. Avoid hot-loop overhead where boundary validation is enough. | No | Keep with strict usage rules. | High |

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
| Learned apparent gaze | UniGaze `unigaze_h14_joint` | Quality-first current gaze estimator selected after comparison. Use the H14 joint-dataset checkpoint for face-level apparent gaze. |
| Learned gaze runtime | `torch`, `torchvision`, `timm`, `unigaze`, `safetensors`, `huggingface_hub` | Required to run UniGaze locally. Use Python 3.12-compatible PyTorch versions after smoke validation; do not blindly install upstream training/demo requirements. |

Model binaries are local artifacts under `models/`, not committed source:

```text
models/
  manifest.json
  mediapipe/
    face_landmarker.task
  unigaze/
    unigaze_h14_joint.safetensors
```

A committed model registry, initially `src/chess_gaze/model_registry.json`, must
be the trust root for model IDs, source URLs, expected filenames, checksums,
license notes, expected input/output shapes, and model task names. The ignored
`models/manifest.json` may only map installed local paths to committed registry
entries and record local verification results. Analysis must fail before frame
processing if any required model file is missing, has a checksum mismatch, or is
not license-approved for the intended use.

For UniGaze `unigaze_h14_joint`, intended-use approval has been granted by the
repo owner for this implementation. The committed registry should record that
approval explicitly with the model ID, license name, approver, and approval date.

The analysis command must not use a code path that downloads UniGaze weights on
first use. Weights must be prefetched intentionally, checksum-verified, and
loaded from the local `models/unigaze/unigaze_h14_joint.safetensors` path.
That prefetch step may use `HF_TOKEN` from `.env`; the subsequent analysis step
must work without network access.

## Library Findings and Sources

High-impact findings verified on 2026-06-24 and updated on 2026-06-25:

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
- UniGaze is the official PyTorch implementation of "UniGaze: Towards Universal
  Gaze Estimation via Large-scale Pre-Training" and lists `unigaze_h14_joint`
  as a UniGaze-H model trained on joint datasets. The README shows
  `unigaze.load("unigaze_h14_joint", device="cuda")` and warns by example that
  the easy loader downloads weights from Hugging Face on first use. Source:
  `https://github.com/ut-vision/UniGaze`.
- PyPI `unigaze` 0.1.3 is a lightweight loader. Its quickstart documents input
  tensors as `(B, 3, 224, 224)` and output as `{"pred_gaze": (B, 2)}` in
  `(pitch, yaw)` order. Source: `https://pypi.org/project/unigaze/`.
- UniGaze easy-loader source maps `unigaze_h14_joint` to
  `unigaze_h14_joint.safetensors`, uses `hf_hub_download`, and then calls
  `load_unigaze_weights(path)`. The model source returns only
  `output_dict["pred_gaze"]`; it does not return confidence. Gaze utility source
  converts pitch/yaw with sine, cosine, `asin`, and `atan2`, supporting radians
  as the stored angle unit. Sources:
  `https://raw.githubusercontent.com/ut-vision/UniGaze/main/unigaze_easy/src/unigaze/loader.py`,
  `https://raw.githubusercontent.com/ut-vision/UniGaze/main/unigaze_easy/src/unigaze/models/mae_gaze.py`,
  and
  `https://raw.githubusercontent.com/ut-vision/UniGaze/main/unigaze/gazelib/gaze/gaze_utils.py`.
- Hugging Face `UniGaze/UniGaze-models` lists `unigaze_h14_joint.safetensors`
  around 2.53 GB and marks the model license as `mg-by-nc-rai`. Source:
  `https://huggingface.co/UniGaze/UniGaze-models`.
- UniGaze's README and Hugging Face model card state the model is licensed under
  ModelGo Attribution-NonCommercial-ResponsibleAI License v2.0. The first
  implementation must therefore keep a license gate and must not assume
  commercial permissibility beyond the repo owner's 2026-06-25 approval for this
  project. Sources: `https://github.com/ut-vision/UniGaze` and
  `https://huggingface.co/UniGaze/UniGaze-models`.
- L2CS-Net is an older practical gaze estimator and remains a possible future
  comparison baseline, but it is not the first implementation's primary gaze
  model after the 2026-06-25 candidate audit. Sources:
  `https://github.com/Ahmednull/L2CS-Net` and
  `https://arxiv.org/abs/2203.03339`.

## Rejected or Deferred Alternatives

| Alternative | Decision | Reason |
| --- | --- | --- |
| OpenFace 2.2 as core | Reject for first implementation | Strong all-in-one facial behavior toolkit, but last release is 2019, build/tooling burden is high, and licensing is non-commercial research oriented. |
| OpenFace 3.0 | Defer | Promising 2025 research direction, but not yet a clearer Python 3.12 local dependency than MediaPipe plus UniGaze for this repo. |
| InsightFace SCRFD as core | Defer and license-gate | Excellent face detection/alignment. Code is MIT, but pretrained model licensing is explicitly non-commercial research for common model packs. Use only as optional fallback if license is acceptable. |
| dlib 68-point landmarks | Reject | Not enough eye/iris detail, weaker fit for occlusion and profile gaze, build friction on current Python/macOS. |
| MTCNN or face-only detectors | Reject as core | Face boxes and 5 landmarks are insufficient for iris, per-eye metrics, and head/eye geometry. |
| `cv2.VideoCapture` as decoder | Reject as primary | Too lossy for frame identity, PTS, time-base, and variable-frame-rate evidence. OpenCV remains useful for geometry and drawing. |
| Temporal smoothing or tracking | Reject for this spec | User requirement is independent frame analysis and preservation of true per-frame gaze evidence. |
| L2CS-Net with Gaze360 as primary learned gaze model | Reject for first implementation | It is a practical older apparent-gaze estimator, but UniGaze `unigaze_h14_joint` is the better-supported current quality-first choice for this task. L2CS may be added later only as a benchmark comparator with its own evidence gate. |

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
| `candidate_face_score_min` | `0.25` | `calibration.json` | Initial threshold only when a real detector score is available; otherwise null and provenance are stored. |
| `usable_face_score_min` | `0.50` | `calibration.json` | Initial status threshold for explicit detector/quality scores, tuned by evidence later. |
| `usable_eye_confidence_min` | `0.50` | `calibration.json` | Initial eye status threshold, tuned by evidence later. |
| `default_iris_diameter_mm` | `11.7` | `calibration.json` | Published MediaPipe Iris depth prior. |
| `default_iris_diameter_uncertainty_mm` | `0.5` | `calibration.json` | Published population variation range for iris diameter. |
| `camera_intrinsics_policy` | `estimate_with_explicit_uncertainty` | `calibration.json` | Avoid pretending uncalibrated camera geometry is exact. |
| `unigaze_input_size_px` | `224` | `run_manifest.json` | UniGaze loader contract for normalized square input tensors. |
| `unigaze_output_order` | `pitch_yaw_radians` | `run_manifest.json` | Prevent yaw/pitch order reversal. |
| `face_landmarker_running_mode` | `IMAGE` | `run_manifest.json` | Preserve frame-independent analysis. |

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
- estimated intrinsics are recorded only as an uncertainty-bearing aid; they do
  not authorize metric `camera_3d_m` translation without a calibrated scale
  source
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
  nominal FPS, time base, codec, pixel format, color range, color space,
  rotation metadata, PyAV version, and FFmpeg library versions.
- Verify required models and checksums.
- Create a new immutable run directory only after input, model, license, and
  disk-space preflight validation.

Pre-run validation failures do not create a run directory and therefore do not
write `records/errors.jsonl`. They must exit nonzero with a stable CLI error
code on stderr. Frame-time recoverable failures are written to `errors.jsonl`.

### 2. Frame Decode

For each decoded video frame, in decoder emission/presentation order:

- Assign `frame_index` and `frame_id`.
- Preserve `pts`, `pts_seconds`, `duration_seconds`, and packet/frame metadata
  when available; these fields are nullable because containers may omit or
  misreport them.
- Convert to full-frame RGB for model use.
- Save a raw full-frame PNG through an atomic temp-then-rename write.
- Compute `raw_frame_sha256`.
- Compute frame quality metrics:
  - blur score
  - exposure score
  - image dimensions
  - source pixel format and RGB conversion policy
  - color conversion metadata
  - decode warnings

No frame sampling, skipping, dropping, or duplicate suppression is allowed in the
first implementation.

### 3. Face and Landmark Observation

For each frame, run MediaPipe Face Landmarker in `IMAGE` mode.

Required Face Landmarker options:

- running mode: `IMAGE`
- `num_faces`: `max_face_candidates`
- `min_face_detection_confidence`: named config value
- `min_face_presence_confidence`: named config value
- `min_tracking_confidence`: unset or ignored for `IMAGE` mode, with provenance
  recorded if the package requires a value
- output face blendshapes: enabled
- output facial transformation matrices: enabled

Required outputs:

- All face candidates up to `max_face_candidates`.
- Candidate bbox in `image_px` and `image_norm`.
- Candidate detector/presence scores when exposed by the API; otherwise null
  with `score_source="not_exposed_by_mediapipe_face_landmarker"`.
- MediaPipe 478 landmark list for each candidate, in normalized coordinates and
  converted pixel coordinates.
- Optional blendshapes when provided.
- Optional facial transformation matrix when provided.
- Primary face selection result and reason.

Primary face selection is per-frame only:

1. If there is one candidate, select it.
2. If there are multiple candidates and a real detector/presence score is
   available, score by explicit formula:
   `candidate_score * candidate_area_fraction`.
3. If candidate score is not exposed, use deterministic fallback selection:
   highest `candidate_area_fraction`, tie-broken by lowest candidate ID, and
   store `selection_score_source="area_only_no_model_score"`.
4. Store every candidate and the selected candidate ID.
5. If a real score is available and no candidate passes `candidate_face_score_min`,
   mark `face.present` false and write `FACE_NOT_FOUND`.
6. If no real score is available, do not invent confidence. Mark selected face
   `present=true` only when required landmarks exist and geometry sanity checks
   pass; otherwise write `FACE_NOT_FOUND` with the invalidity reason.

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

Initial PnP correspondences must be explicit and stored in calibration metadata.
Use these MediaPipe landmark indices unless implementation evidence proves a
better set:

| Name | MediaPipe index | Role |
| --- | ---: | --- |
| `nose_tip` | 1 | central forward anchor |
| `chin` | 152 | lower face anchor |
| `left_eye_outer` | 33 | left canthus anchor |
| `right_eye_outer` | 263 | right canthus anchor |
| `left_eye_inner` | 133 | left inner eye anchor |
| `right_eye_inner` | 362 | right inner eye anchor |
| `left_mouth_corner` | 61 | lower lateral anchor |
| `right_mouth_corner` | 291 | lower lateral anchor |

Canonical 3D coordinates for these points must not be hard-coded as anonymous
numbers. The implementation must either:

- use MediaPipe's provided facial transformation matrix as the primary pose
  source; or
- load the canonical 3D points from a named committed resource generated from a
  documented canonical face model, with units and source recorded.

PnP validity requires named thresholds for minimum point count, reprojection
error, matrix finite-ness, and plausible rotation range. When those checks fail,
store `HEAD_POSE_INVALID` and keep all raw landmarks.

### 6. Gaze Observation

The first implementation must distinguish three gaze evidence layers:

1. `per_eye_geometric_gaze`
   - Computed independently for left and right eyes from iris center, eye
     aperture geometry, iris size, and head pose.
   - Represents per-eye apparent gaze proxy and optional estimated ray, not a
     guaranteed screen fixation point.
2. `face_model_gaze`
   - Computed by UniGaze `unigaze_h14_joint` on a UniGaze-compatible normalized
     224x224 face/head crop, not an arbitrary face bbox crop.
   - Represents face-level apparent gaze pitch/yaw and optional derived
     direction vector.
   - Stores the normalization transform needed to map the prediction back to
     source image/camera evidence.
   - Stores `confidence=null` and `confidence_source="not_provided_by_unigaze"`
     unless a future wrapper adds a calibrated confidence model.
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
- face score and score provenance when available
- major face landmarks
- left and right eye contours
- left and right iris centers
- per-eye geometric gaze vectors
- face-level UniGaze vector
- head pose axes
- frame status and error code summary

When face or eyes are not found, the visualization must still exist and must show
the failure status rather than silently omitting the frame.

Visualization frames are QA artifacts only. They must never be used as model
input or source evidence.

## Record Schemas

Schema validation is part of the feature. Each schema has a version string so
future changes can be migrated intentionally.

Pydantic models must be strict at artifact boundaries:

- `extra="forbid"`
- finite floats only; no NaN or infinity
- no silent string-to-number coercion
- typed bbox, point, landmark, and transform models instead of anonymous arrays
  in production schemas
- impossible states rejected, such as `present=true` with no required landmarks
  or `valid=true` with missing yaw/pitch
- invalid records carry explicit reason codes
- after a run completes or fails, manifests and JSONL records are re-read and
  validated from disk

The JSON examples below show representative shape. Implementation schemas must
replace placeholder arrays with typed structures and invariants.

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
      "model_id": "unigaze-h14-joint",
      "task": "face_level_gaze",
      "path": "models/unigaze/unigaze_h14_joint.safetensors",
      "sha256": "...",
      "license": "MG-NC-RAI-2.0",
      "license_approved_for_run": true,
      "input_shape": [1, 3, 224, 224],
      "output_contract": "pred_gaze_pitch_yaw_radians"
    }
  ],
  "model_registry_path": "src/chess_gaze/model_registry.json",
  "environment": {
    "python": "3.12.x",
    "platform": "...",
    "pyav": "...",
    "ffmpeg": "...",
    "opencv_provider": "opencv-python-headless",
    "package_version": "0.1.0",
    "git_commit": "..."
  },
  "image_io": {
    "decode_order": "decoder_emission_presentation_order",
    "rgb_boundary": "model_inputs_are_rgb_numpy_arrays",
    "opencv_boundary": "opencv_drawing_and_encoding_use_bgr_arrays"
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
  "frame_count_source": "container_metadata_hint",
  "pixel_format": "yuv420p",
  "color_space": "bt709",
  "color_range": "tv",
  "pyav_version": "...",
  "ffmpeg_versions": {},
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
    "camera_ray",
    "camera_3d_m",
    "head_3d_m",
    "board_norm",
    "board_square"
  ],
  "face_selection": {
    "max_face_candidates": 4,
    "candidate_face_score_min": 0.25,
    "usable_face_score_min": 0.5,
    "selection_formula_when_score_available": "candidate_score * candidate_area_fraction",
    "selection_formula_when_score_unavailable": "candidate_area_fraction_then_candidate_id",
    "score_null_policy": "do_not_invent_model_confidence"
  },
  "eyes": {
    "usable_eye_confidence_min": 0.5,
    "default_iris_diameter_mm": 11.7,
    "default_iris_diameter_uncertainty_mm": 0.5
  },
  "camera": {
    "intrinsics": null,
    "intrinsics_source": "unavailable",
    "intrinsics_uncertainty": "high",
    "distortion_coefficients": null,
    "metric_translation_allowed": false
  },
  "head_pose": {
    "pnp_landmark_indices": {
      "nose_tip": 1,
      "chin": 152,
      "left_eye_outer": 33,
      "right_eye_outer": 263,
      "left_eye_inner": 133,
      "right_eye_inner": 362,
      "left_mouth_corner": 61,
      "right_mouth_corner": 291
    },
    "canonical_points_source": "committed_named_resource_or_mediapipe_transform",
    "pnp_method": "solvepnp_iterative",
    "validity_thresholds": {}
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
        "candidate_score": null,
        "score_source": "not_exposed_by_mediapipe_face_landmarker",
        "selection_score": 0.26,
        "selection_score_source": "area_only_no_model_score",
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
      "intrinsics_source": "unavailable",
      "metric_translation_allowed": false,
      "reprojection_error_px": null,
      "reason_invalid": null
    }
  },
  "eyes": {
    "left": {
      "present": true,
      "confidence": 0.91,
      "confidence_source": "derived_landmark_geometry",
      "reason_missing": null,
      "eye_crop_path": "crops/eyes/left/f000000000.png",
      "eye_crop_transform_to_image_px": [],
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
        "ray_camera": null,
        "confidence": 0.0,
        "reason_invalid": null
      }
    },
    "right": {
      "present": true,
      "confidence": 0.89,
      "confidence_source": "derived_landmark_geometry",
      "reason_missing": null,
      "eye_crop_path": "crops/eyes/right/f000000000.png",
      "eye_crop_transform_to_image_px": [],
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
        "ray_camera": null,
        "confidence": 0.0,
        "reason_invalid": null
      }
    }
  },
  "gaze": {
    "face_model_gaze": {
      "valid": true,
      "method": "unigaze_h14_joint",
      "input_crop_path": "crops/face/f000000000.png",
      "input_normalization_transform": [],
      "model_output_order": "pitch_yaw",
      "yaw_radians": 0.0,
      "pitch_radians": 0.0,
      "gaze_vector_camera": null,
      "confidence": null,
      "confidence_source": "not_provided_by_unigaze",
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
      "unigaze-h14-joint"
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
  "message": "No face candidate passed candidate_face_score_min",
  "recoverable": true,
  "artifact_refs": [
    "processed_frames/f000000120.jpg"
  ],
  "evidence": {
    "best_candidate_score": 0.12,
    "score_source": "detector_score"
  }
}
```

Pre-run CLI error codes, emitted on stderr before a run directory exists:

- `INPUT_NOT_FOUND`
- `UNSUPPORTED_VIDEO`
- `MODEL_ASSET_MISSING`
- `MODEL_ASSET_CHECKSUM_MISMATCH`
- `MODEL_LICENSE_NOT_APPROVED`

Required frame-time `errors.jsonl` codes:

- `FRAME_DECODE_FAILED`
- `RAW_FRAME_WRITE_FAILED`
- `PROCESSED_FRAME_WRITE_FAILED`
- `FACE_NOT_FOUND`
- `MULTIPLE_FACE_CANDIDATES`
- `PRIMARY_FACE_LOW_SCORE`
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
- run status transitions and final status
- record count equals decoded frame count
- raw frame file count equals decoded frame count
- processed frame file count equals decoded frame count
- byte counts for raw frames, processed frames, crops, JSONL, and total run size
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
- disk-space preflight estimate and remaining-space measurement at closeout

## Acceptance Criteria

Hard artifact-contract criteria:

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
10. MediaPipe Face Landmarker is run in `IMAGE` mode with explicit options
    persisted in `run_manifest.json`.
11. No temporal smoothing, tracking, interpolation, or across-frame averaging is
    applied to face, eye, iris, head pose, or gaze outputs.
12. UniGaze `unigaze_h14_joint` face-level gaze is stored separately from
    independent per-eye geometric gaze. It is not copied into left/right eye
    fields.
13. Records never claim `target_square` or calibrated screen target unless a
    calibration file provides the needed mapping.
14. Every coordinate-bearing field is in a named coordinate space.
15. Every threshold and anatomical/camera assumption is named and persisted in a
    manifest or calibration file.
16. `qa_summary.json` validates that artifact counts and JSONL line counts match
    decoded frame count.
17. The committed model registry is the trust root. Ignored local
    `models/manifest.json` cannot introduce or override a model not present in
    the committed registry.
18. Missing model files, checksum mismatch, or unapproved model license exits
    nonzero before creating a run directory.
19. UniGaze weights are loaded from local verified artifacts; analysis does not
    trigger Hugging Face or other network downloads.
20. Exactly one OpenCV provider is installed after dependency sync.
21. Schema validation rejects impossible states, NaN/infinite floats, unknown
    fields, and silent type coercion at artifact boundaries.
22. A partial run caused by a frame-time write or schema failure records final
    status, status transition, and recoverable/nonrecoverable error evidence.
23. The app runs fully locally and does not send video frames, crops, metadata,
    or model inputs to a remote service.
24. The standard repo gates pass after implementation:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Real-model smoke targets:

- On the two local verification videos, `FACE_NOT_FOUND` should be no more than 1
  percent of decoded frames unless manual visualization review proves the face is
  absent or fully hidden in those frames. Until labeled expected-output frames
  exist, this is a smoke warning and triage target, not an automated hard gate.
- On the two local verification videos, frames with both eyes visibly open in the
  processed visualization should have both iris centers marked. Misses must be
  listed in `qa_summary.json` representative failures. Until labeled frames
  exist, this is manual QA evidence, not a deterministic pass/fail threshold.

Real-data verification is mandatory for acceptance. If either local video or a
required local model asset is unavailable, the exact missing path, affected
subsystem, blocked verification, and next unblock action must be recorded in
the closeout. Theoretical progress, synthetic-only tests, or skipped smoke
checks must not be described as complete real-data verification.

## Testing Strategy

Implementation must be test-first under the repo contract.

Required automated tests:

| Layer | Required checks |
| --- | --- |
| Unit | Path-to-run layout, frame ID formatting, constants persistence, coordinate-space validation, schema validation, error-code serialization. |
| Unit | Primary face selection preserves all candidates, handles nullable detector scores, and explains selection provenance. |
| Unit | Missing face, missing left eye, missing right eye, missing iris, and gaze disagreement all produce valid records. |
| Unit | UniGaze wrapper maps `pred_gaze[:, 0]` to pitch and `pred_gaze[:, 1]` to yaw, stores no fabricated confidence, and refuses network auto-download paths. |
| Unit | Model registry validation rejects missing local files, checksum mismatch, unapproved license, and local manifest entries absent from the committed registry. |
| Unit | Strict schemas reject unknown fields, impossible valid/present states, NaN, infinity, and silent type coercion. |
| Integration | Decode a tiny synthetic video and verify raw frame count, processed frame count, and JSONL line count match decoded frames. |
| Integration | Run with fake/model-stub observers to prove the artifact contract without heavyweight ML dependencies. |
| Integration | Verify atomic temp-then-rename writes and partial-run status when a write failure is injected. |
| Integration | Verify exactly one `cv2` provider is importable after dependency sync. |
| Smoke | Run real analysis on `artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4`, or record the exact blocker when required local model assets are absent. |
| Manual QA | Inspect deterministic processed-frame samples from each local verification video. |

Tests must not require committing ignored video files or model weights. When
ignored assets are unavailable, tests may skip with explicit messages, but the
skip is a recorded blocker for the affected real-data requirement, not a pass.
Video-only checks must still run when `test_1.mp4` and `test_2.mp4` are
present, even if model assets are absent.

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
- Keep the committed model registry as the authority for ignored local model
  files.
- Treat UniGaze as a normalized-head/face gaze model with a strict input/output
  contract, not as a raw bbox-crop classifier.
- Record null when a package does not provide confidence.

Avoid:

- Using MediaPipe `VIDEO` or `LIVE_STREAM` mode for measurement records.
- Treating MediaPipe iris landmarks alone as true gaze target prediction.
- Treating UniGaze face-level output as independent per-eye gaze.
- Letting `unigaze.load()` or any other helper download weights during analysis.
- Using only the final inferred square or a single gaze label.
- Assuming the streamer is always in one screen quadrant.
- Assuming the largest face is correct without recording candidates and
  selection provenance.
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
- `model_assets` for committed registry validation and local model checks.
- `image_io` for RGB/BGR boundaries and atomic image writes.
- `face_observation` for MediaPipe face/landmark observation.
- `eye_observation` for eye/iris measurements.
- `head_pose` for transform and PnP pose estimates.
- `gaze_observation` for per-eye geometry and UniGaze face-level gaze.
- `visualization` for processed frame overlays.
- `qa_summary` for aggregate validation and sampling.

These are allowed module names, not mandatory scaffolding. Add a module only
when it owns real invariants or meaningful behavior.

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
- Model binaries can live in local ignored `models/`, but expected IDs, URLs,
  checksums, licenses, and input/output contracts must be in a committed model
  registry. The ignored `models/manifest.json` is not authoritative.
- The repo owner granted intended-use approval for UniGaze's non-commercial
  responsible-AI license on 2026-06-25. The implementation must record that
  approval in the committed model registry or config metadata. If future use
  changes materially, the implementation must stop for a new model-selection or
  license decision rather than silently falling back to L2CS-Net.
- H14 runtime performance on local hardware is unverified. If it is too slow or
  too memory-heavy, the next decision must compare UniGaze B/L/H variants and
  any other current models using the same evidence rules.
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
