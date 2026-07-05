# Gaze Precision Improvement Design

Date: 2026-07-05

## Goal

Improve default gaze precision on real chess-stream videos by fixing the
highest-confidence UniGaze input mismatch, while adding benchmark and calibration
surfaces that make future precision claims measurable instead of visual-only.

The immediate real-video target is `artifacts/input/nakamura_short.mp4`.

## Root Cause

The current default observer sends UniGaze a tight selected-face bounding box,
resized to `224x224`, scaled to `[0, 1]`, and then interprets returned
pitch/yaw as camera-space after only a yaw sign flip.

Primary-source UniGaze inference does not use that image contract. The official
video script expands the detected landmark face box, estimates head pose,
normalizes the face with camera geometry, applies an ImageNet-style model
transform at `224x224`, predicts gaze in normalized space, then denormalizes the
predicted vector with the inverse normalization rotation.

The installed `unigaze==0.1.3` package exposes the model loader only. It does
not ship the `gazelib` normalization code, `face_model.txt`, or the training
dataset helper transform used by the full repository script. Therefore the
durable default fix in this repository is the strongest directly supportable
contract repair:

- make the UniGaze model input use an explicit preprocessing profile;
- default to a reference-like profile that expands the face crop by `2.0` and
  applies ImageNet RGB channel normalization;
- preserve a legacy profile for one-by-one benchmarking and rollback;
- record the profile and numeric transform constants in `calibration.json`;
- keep denormalization and target-plane calibration separate until their
  required geometric inputs are available.

## Evidence, Verified 2026-07-05

| Candidate / source | Primary evidence | Decision |
| --- | --- | --- |
| Current `unigaze_h14_joint` checkpoint and `unigaze==0.1.3` loader | UniGaze official repo lists `unigaze_h14_joint`; Hugging Face model card provides `unigaze_h14_joint.safetensors` under MG-NC-RAI; PyPI `unigaze 0.1.3` exposes a lightweight loader and model output `pred_gaze` shaped `(B, 2)`. | Keep. No model switch in this task. |
| Current tight bbox + RGB `/255` preprocessing | Local code and tests show a direct crop/resize/scale path. It lacks the reference script's expanded crop and model transform. | Reject as default; keep as `legacy_bbox_rgb01` for benchmark. |
| Reference-like crop + ImageNet transform | UniGaze reference video script uses a scaled face box and `wrap_transforms('basic_imagenet', image_size=224)`. The exact helper is not packaged locally, but its standard contract is testable and does not add dependencies. | Select as default `reference_face2x_imagenet`. |
| Full official geometric normalization/denormalization | UniGaze reference script depends on `gazelib.normalize`, `estimateHeadPose`, a 68-point face model, and denormalizes with `R_inv`. Those assets are not present in `unigaze==0.1.3`. | Defer; implement only after adding verified assets or vendored code through ADR-0002. |
| Camera/target-plane calibration | OpenCV docs define camera intrinsics/distortion and projection geometry; point-of-gaze on a screen/board requires a calibrated target plane or empirical mapping. | Add optional math surfaces and benchmark requirements, but do not claim default screen/board accuracy without calibration data. |
| Person/session calibration | Calibration papers show person/session bias calibration can reduce appearance-gaze error. Labels are required for held-out evaluation. | Add affine calibrator API and tests; do not apply automatically without labels. |

Primary URLs:

- UniGaze repo: https://github.com/ut-vision/UniGaze
- UniGaze video inference script: https://github.com/ut-vision/UniGaze/blob/main/unigaze/predict_gaze_video.py
- UniGaze paper: https://arxiv.org/abs/2502.02307
- UniGaze model card/files: https://huggingface.co/UniGaze/UniGaze-models
- UniGaze PyPI: https://pypi.org/project/unigaze/
- OpenCV camera calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- MediaPipe Face Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- Calibration papers: https://arxiv.org/abs/1905.04451,
  https://arxiv.org/abs/2001.09284, https://arxiv.org/abs/2009.01270

## Baseline

Fresh baseline command:

```sh
.venv/bin/python -c "from chess_gaze.cli import main; raise SystemExit(main(['analyze','artifacts/input/nakamura_short.mp4','--no-resume','--progress','off','--qa-summary']))"
```

Baseline run:
`artifacts/output/nakamura_short/runs/20260705T111919Z-792e42a8`

Observed baseline metrics:

- decoded frames: `1200`
- face/eye/head/appearance-gaze valid rates: `1.0`
- valid sphere hits: `1182 / 1200`
- median raw UniGaze yaw: `0.8152058124542236` radians
- median raw UniGaze pitch: `-0.34289664030075073` radians
- median frame-to-frame ray angle: `0.06663548703654452` radians
- p95 frame-to-frame ray angle: `0.2615590868301171` radians

These are not ground-truth point-of-gaze accuracy numbers. They are the local
before/after runtime and stability anchors for this clip.

## Approach

### 1. UniGaze Preprocessing

Add an explicit `unigaze_preprocessing_profile` to analysis configuration and
calibration artifacts.

Profiles:

- `legacy_bbox_rgb01`: current behavior for reproducible baseline comparisons.
- `reference_face2x_imagenet`: default behavior. Expand the selected face box
  by `2.0` around its center, clamp to the image, resize to `224x224`, scale RGB
  to `[0, 1]`, then normalize channels with ImageNet RGB mean/std:
  mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.

The frame record schema remains unchanged because downstream consumers already
store the resulting pitch/yaw. `calibration.json` records the preprocessing
contract so resumed runs cannot silently mix profiles.

### 2. Target-Plane Calibration

Point-of-gaze precision on a chessboard, screen, or overlay cannot be inferred
from the webcam ray alone. A target plane needs calibrated camera-space geometry:

```text
t = -(n . O + b) / (n . d)
X_hit = O + t d
```

This task keeps the default viewer's gaze-sphere visualization intact and adds a
small tested geometry surface for target-plane intersections. The scene/viewer
schema should not claim board/screen hits unless a target plane is configured.

### 3. Person / Video Calibration

Add a simple affine calibrator that maps raw gaze/head features to normalized
target coordinates. The first supported feature order is:

```text
1, yaw, pitch, head_yaw, head_pitch
```

The calibrator must support ridge regularization and held-out evaluation. It is
not applied by default. A calibration model without held-out labels cannot be
reported as a precision improvement.

## Benchmark Design

The benchmark must compare ideas independently:

1. **Preprocessing A/B**: run `nakamura_short.mp4` with
   `legacy_bbox_rgb01` and `reference_face2x_imagenet`, same model/device/batch.
   Compare valid rates, sphere-hit counts, yaw/pitch distribution, frame-to-frame
   angular stability, runtime, and artifact validity.
2. **Target-plane geometry**: use deterministic synthetic rays/planes with known
   intersections. For real videos, report target-plane metrics only when a
   target-plane config exists.
3. **Affine calibration**: use synthetic and labeled datasets with held-out
   split. Report train error separately from held-out error and never use fit
   error as accuracy.

If no real labels exist, final reporting must say that true point-of-gaze
accuracy is unmeasured. It may still report proxy/stability metrics.

## Integration Requirements

- Default `chess-gaze analyze` uses `reference_face2x_imagenet`.
- `--unigaze-preprocessing-profile legacy_bbox_rgb01` restores old behavior.
- Existing MPS default remains `mps` batch `7`.
- The benchmark must use `.venv/bin/python` in this environment for MPS because
  sandboxed `uv run` cannot see MPS while the repo venv can.
- Existing 3D sphere visualization must keep working.
- No invented UniGaze confidence values.
- No silent coordinate sign changes without tests.
- No default screen/board precision claims without calibration data.

## Validation

Required gates:

- focused preprocessing tests;
- focused frame observer tests;
- focused target-plane geometry and affine-calibration tests;
- focused scene/viewer tests if schema or viewer data changes;
- `nakamura_short.mp4` MPS baseline and changed runs;
- `uv run pytest` or documented narrower gate if native real-video tests require
  unavailable inputs;
- `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`.

## Residual Risk

The selected default fixes a confirmed model input mismatch, but it is not full
official UniGaze geometric normalization/denormalization. True board/screen
accuracy still requires target-plane calibration and held-out labels.
