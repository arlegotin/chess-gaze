# ADR-0007: Restore UniGaze Geometric Normalization

Date: 2026-07-13

## Status

Accepted

## Context

The repository's prior default, `reference_face2x_imagenet`, expanded a face
bounding box, resized it directly to 224 x 224, and applied ImageNet channel
normalization. The pinned UniGaze video path instead estimates head pose from
six face landmarks, applies a camera-normalization homography, predicts a gaze
model vector in normalized coordinates, and applies the inverse normalization
rotation before camera-space use. Direct resize therefore did not implement the
checkpoint's published geometric input/output contract.

H1 was initially blocked because `face_model.txt` had no file-level license
notice. On 2026-07-13 the repository owner approved its use and redistribution
under `MG-NC-RAI-2.0`. The registered asset is:

- model ID: `unigaze-face-model-v1`;
- local path: `models/unigaze/face_model.txt`;
- upstream revision: `9c240fbe33f3d6146970a77b7c8fa06a7e60019e`;
- source: <https://raw.githubusercontent.com/ut-vision/UniGaze/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/unigaze/data/face_model.txt>;
- SHA-256: `0c943d1d48627d97038b64f9a73816b9ab80a002ce81a8f04d532da2f4c337d7`;
- approval: `repo_owner`, 2026-07-13, `MG-NC-RAI-2.0`;
- contract: finite 50 x 3 `xyz` points, selecting rows
  `[20,23,26,29,15,19]` for MediaPipe landmarks
  `[33,133,362,263,98,327]`.

This ADR records the repository owner's authorization decision; it does not
offer an independent legal conclusion about the asset.

The implementation and decision use the official repository at the pinned
revision, the pinned
[video inference path](https://github.com/ut-vision/UniGaze/blob/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/unigaze/predict_gaze_video.py),
the pinned
[normalizer](https://github.com/ut-vision/UniGaze/blob/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/unigaze/gazelib/gaze/normalize.py),
the pinned
[training-label preparation](https://github.com/ut-vision/UniGaze/blob/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/gazedata_preparation/normalize_xgaze.py),
the pinned
[root license](https://github.com/ut-vision/UniGaze/blob/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/LICENSE.txt),
the [UniGaze model revision](https://huggingface.co/UniGaze/UniGaze-models/tree/d3f8335cd4b7d249adbc32389986ce49b52f6f72),
its pinned
[model card](https://huggingface.co/UniGaze/UniGaze-models/blob/d3f8335cd4b7d249adbc32389986ce49b52f6f72/README.md),
and the [WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/papers/Qin_UniGaze_Towards_Universal_Gaze_Estimation_via_Large-scale_Pre-Training_WACV_2026_paper.pdf).
These primary sources and local assets were reverified on 2026-07-13. The local
`unigaze-h14-joint` checkpoint remains pinned at SHA-256
`a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f`.

The three allowed `*_short.mp4` files have no gaze target or fixation labels.
They can verify provenance, coverage, runtime behavior, and descriptive motion
proxies, but cannot establish angular or point-of-gaze accuracy.

## Alternatives and Evidence

| Alternative | Contract and evidence | Decision |
| --- | --- | --- |
| `official_geometric_v1` | Reproduces the pinned six-point pose solve, `W = K_norm * S * R * K_crop^-1` perspective warp, ImageNet normalization, and row-aligned `R^-1` output conversion. Independent equation, sign, flip, and native fixed-frame oracles passed. All three paired campaigns passed QA with zero valid-gaze coverage loss. | Accepted as the analysis default. |
| `reference_face2x_imagenet` | Existing 2x direct-resize profile. It preserves full paired-campaign coverage but omits pose normalization and inverse rotation. | Retained only as an explicit rollback and comparison profile. |
| `legacy_bbox_rgb01` | Original 1x RGB `[0,1]` crop. It omits both ImageNet normalization and geometric normalization. Earlier proxy results were mixed and it does not match the pinned contract. | Retained only for historical artifact compatibility, not as a default or H1 comparator. |
| Keep H1 blocked and remove geometric support | This was required while asset permission was unresolved. The repository owner's dated license approval, pinned checksum, reviewed six-point mapping, and independent oracles now satisfy that prerequisite. | Rejected after the prerequisite was resolved. |

No alternate gaze model or inference library is selected by this ADR. UniGaze
0.1.3, PyTorch 2.12.1, OpenCV 4.13, MediaPipe 0.10.35, and the existing
checkpoint remain in place; no production dependency was added.

## Decision

`official_geometric_v1` is the default UniGaze preprocessing profile for normal
analysis. It:

1. uses the expanded face crop and the reviewed six MediaPipe landmarks;
2. validates and loads the pinned face-model asset only when this profile is
   selected;
3. solves head pose with the six corresponding face-model rows;
4. persists the profile and face-model ID/checksum in calibration metadata;
5. maps cropped-image pixels to normalized 224 x 224 pixels with
   `W = K_norm * S * R * K_crop^-1`, where `R` is
   `normalized_from_camera_rotation`; and
6. applies the row's `camera_from_normalized_rotation = R^-1` to the UniGaze
   model vector before converting it to repository pitch/yaw.

The output-sign repair is part of the contract, not a display tweak. UniGaze
training stores `vector_to_pitchyaw(-gaze_norm)`, and the pinned video path
projects the physical ray as `-(R^-1 @ model_vector_normalized)`. If
`model_vector_camera = R^-1 @ model_vector_normalized`, the repository stores:

```text
pitch = asin(model_vector_camera.y)
yaw   = atan2(-model_vector_camera.x, model_vector_camera.z)
```

The existing scene boundary maps that stored angle vector to `(x, -y, -z)`,
which reconstructs the physical camera ray `-model_vector_camera`. The original
H1 plan treated the predicted vector too directly as the physical ray; pinned
training and video evidence exposed that sign error, and vector/oracle tests now
protect the corrected composition.

The model output contract remains two radians in `[pitch,yaw]` order and has no
confidence value. The runtime must not invent confidence or silently fall back
to direct resize when requested geometry is invalid. A geometric failure
invalidates that frame.

Operators can roll back explicitly with:

```sh
uv run chess-gaze analyze video.mp4 \
  --unigaze-preprocessing-profile reference_face2x_imagenet
```

## Consequences

- Normal analysis requires the pinned `models/unigaze/face_model.txt` in
  addition to the existing MediaPipe and UniGaze assets. Reference and legacy
  profiles validate only the assets they use.
- The retained six-run campaign executed on an Apple Silicon host with MPS,
  batch size 7, PyTorch 2.12.1, MPS preflight passing, and the three MPS
  fallback/fast-math/prefer-Metal environment variables unset.
- Valid-gaze coverage was identical: Carlsen 600/600, Nakamura 1,200/1,200,
  and Nepomniachtchi 1,190/1,200 for both profiles; aggregate coverage was
  2,990/3,000 for both, so coverage loss was zero.
- Median and p95 ray-speed proxies decreased on all three clips. Those values
  mix real eye/head motion with estimator noise and are descriptive only. With
  no target or fixation truth, this ADR makes no gaze-accuracy or fixation-
  precision improvement claim.
- `reference_face2x_imagenet` remains the rollback profile, so older calibrated
  runs and explicit A/B comparisons remain reproducible.
- The forward-only crop replay in `unigaze_batch_benchmark.py` intentionally
  remains a historical reference-tensor device/batch benchmark. Although frame
  records persist face landmarks, its loader does not supply those landmarks
  and face-model points to preprocessing or preserve per-row inverse rotations.
  The harness therefore explicitly pins `reference_face2x_imagenet` instead of
  inheriting the runtime default. Upgrading it would be a separate experiment:
  load the face asset, reconstruct row geometry, and decide whether inverse
  rotation belongs in a forward-only timing/equivalence metric.
- The six run artifacts do not embed a Git tree/dirty fingerprint. Before any
  retention/default edits, operator process evidence froze HEAD
  `023a8e57bca4df17ec166fda0aa1f94a2cbd5f59` and repeatedly hashed
  `git diff --binary HEAD -- src/chess_gaze pyproject.toml uv.lock` as
  `2f57202d87d25be3a6cc8c673c66d5d44ea277f343bbae417ad0e4ea6eafcd8b`,
  with no untracked runtime files. This must not be represented as embedded
  artifact provenance.

## Verification

Future agents should verify the following independently:

- the registry metadata and local SHA-256 values for all three required assets;
- equation parity, degeneracy failures, sign/rotation round trips, flip
  equivariance, row alignment, selective asset validation, and persisted
  calibration provenance in the focused tests;
- the native fixed-frame oracle
  `test_official_geometric_normalization_matches_pinned_short_video_frames` on
  frame 0 of each exact approved clip; and
- a paired all-clip campaign whose comparator allows only
  `unigaze_preprocessing` to differ and whose coverage gate allows at most 2
  aggregate percentage points and 5 per-clip points of loss.

The retained 2026-07-13 evidence is:

| Clip | Reference run | Official run | Comparator report / SHA-256 |
| --- | --- | --- | --- |
| Carlsen | `artifacts/output/gaze-hypotheses/h1/reference/carlsen_short/runs/20260713T060135Z-b85b647c` | `artifacts/output/gaze-hypotheses/h1/official/carlsen_short/runs/20260713T061256Z-2cfab469` | `artifacts/output/gaze-hypotheses/h1/carlsen_short-comparison.json` / `97fe260b76ca2dd92fd607ce09a09bcc7c914dd1d801a91dd80cc560a488ecb9` |
| Nakamura | `artifacts/output/gaze-hypotheses/h1/reference/nakamura_short/runs/20260713T060354Z-1d7a1750` | `artifacts/output/gaze-hypotheses/h1/official/nakamura_short/runs/20260713T061520Z-2013413b` | `artifacts/output/gaze-hypotheses/h1/nakamura_short-comparison.json` / `5dd0b7541b57cbf2f5268db20e89c4a17b43e887dce0058afbb98c1b1a1bcb16` |
| Nepomniachtchi | `artifacts/output/gaze-hypotheses/h1/reference/nepo_short/runs/20260713T060833Z-2e32fec2` | `artifacts/output/gaze-hypotheses/h1/official/nepo_short/runs/20260713T061857Z-58672092` | `artifacts/output/gaze-hypotheses/h1/nepo_short-comparison.json` / `c00a06a0861bc1d162849657feb0c5b0b6a835ff992ae834b84d8a1914736c62` |

`artifacts/output/gaze-hypotheses/h1/coverage-gate.json` has SHA-256
`92074c83627691a91657fb8c12d6a150e8903ae55c627c981d6bc84abb09a1e3`
and records a passing zero-loss result. Each run's `qa_summary.json` records
`final_status="complete"`, schema validation passing, and count agreement.
