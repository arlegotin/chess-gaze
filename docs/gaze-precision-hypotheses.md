# Gaze precision improvement hypotheses

Research and repository review date: 2026-07-12

This is an experiment backlog for future coding agents, not a claim that any
proposed change is guaranteed to improve accuracy. The shortest sensible path is:

1. make the existing benchmark capable of measuring accuracy;
2. restore UniGaze's published geometric input/output contract;
3. use the existing lightweight calibrator;
4. only then spend time on tracking, filtering, content priors, or another model.

No paid hardware, paid software, labeled chess-streamer dataset, large training
job, or new production dependency is required for the first eight hypotheses.

## The important distinction: precision is not accuracy

The project currently estimates a 3D apparent-gaze direction from a face image.
Several different qualities must not be collapsed into one number:

| Term | Meaning | Honest metric |
| --- | --- | --- |
| Angular accuracy | Closeness to a known true gaze direction | Median and p95 angular error in degrees |
| Point-of-gaze accuracy | Closeness to a known screen point | Pixels, fraction of screen diagonal, or centimetres |
| Fixation precision | Scatter while the person holds one target | Median and p95 deviation from the fixation mean |
| Coverage | How often the system returns an accepted estimate | Accepted frames / scored frames |
| Temporal response | How quickly output follows a real gaze change | Target-switch delay in milliseconds or frames |
| Chess focus | Board/off-board, region, or square classification | Balanced accuracy, exact-square accuracy, Manhattan square error |

A constant but wrong vector has perfect stability and zero accuracy. A filter can
improve precision while making accuracy and latency worse. A quality gate can
improve error only by discarding most frames. Every experiment below therefore
keeps these metrics separate.

Exact focus for an archived third-party streamer is not identifiable from the
facecam alone. The physical relationship among camera, eyes, monitor, and board
is unknown, and the broadcast layout may not match the streamer's monitor. For
those videos the honest outputs are apparent gaze, uncertainty, and possibly
coarse focus. Exact screen or board-square claims need a controlled calibration
or equivalent target evidence.

At a viewing distance of 60 cm, `1°` is about `10.5 mm`, `3°` is `31.4 mm`, and
`5°` is `52.5 mm` (`D * tan(angle)`). A square's centre-to-edge allowance is
approximately `atan(board_width / (16 * viewing_distance))`: only `1.43°` for a
24 cm board and `2.39°` for a 40 cm board at 60 cm. Board-versus-off-board and
coarse regions should therefore precede exact-square claims.

## What the repository does today

The active path is:

```text
PyAV decode
  -> MediaPipe Face Landmarker in independent IMAGE mode
  -> per-frame primary-face selection, sometimes across ten fixed regions
  -> eye landmarks and head pose
  -> 2x face bbox, direct 224x224 resize, ImageNet channel normalization
  -> UniGaze-H14 joint checkpoint
  -> hard-coded yaw sign flip
  -> apparent-gaze ray
  -> optional pseudo-metric 3D scene / configured target plane
```

The most important findings are:

- The current crop performs only bbox expansion, resize, and channel
  normalization
  ([`gaze_observation.py:119`](../src/chess_gaze/gaze_observation.py#L119)).
  UniGaze's current official video path also estimates head pose, applies a
  perspective normalization warp, predicts in normalized coordinates, and
  applies the inverse normalization rotation before using the vector in camera
  coordinates. The repository does not do those geometric steps.
- Head pose is computed but not passed into UniGaze preprocessing. UniGaze angles
  are treated as camera-space after only a drawing-derived yaw negation
  ([`gaze_observation.py:222`](../src/chess_gaze/gaze_observation.py#L222),
  [`scene_geometry.py:483`](../src/chess_gaze/scene_geometry.py#L483)).
- Face detection is forced to stateless `IMAGE` mode with up to four faces
  ([`face_observation.py:155`](../src/chess_gaze/face_observation.py#L155)). The
  model does not expose candidate scores through this API, so selection falls
  back to area and hand-written geometry. If full-frame detection needs help,
  nine additional fixed regions are tried
  ([`face_observation.py:946`](../src/chess_gaze/face_observation.py#L946)).
- Eye confidence is not model confidence. It is a discrete derived value of
  `0.5` or `1.0`; the configured minimum is not consumed in production. Iris
  offsets are computed but not used by inference or calibration
  ([`eye_observation.py:214`](../src/chess_gaze/eye_observation.py#L214),
  [`eye_observation.py:484`](../src/chess_gaze/eye_observation.py#L484)).
- The existing affine calibrator is isolated from the runtime. Raw appearance
  gaze is copied directly to recommended gaze
  ([`frame_observation.py:260`](../src/chess_gaze/frame_observation.py#L260)).
  Its unregularized normal-equation solve can fail on rank-deficient real
  samples, and ridge currently penalizes the intercept
  ([`gaze_calibration.py:52`](../src/chess_gaze/gaze_calibration.py#L52)).
- The current benchmark measures coverage and radians per frame, not target
  accuracy. It ignores timestamps
  ([`gaze_precision_benchmark.py:172`](../src/chess_gaze/gaze_precision_benchmark.py#L172))
  and does not reject comparisons made from different source videos.
- The benchmark's target-plane count includes valid mathematical intersections
  outside the configured plane bounds
  ([`scene_artifacts.py:597`](../src/chess_gaze/scene_artifacts.py#L597)). It is
  not board/screen coverage.
- Decode records rotation metadata but does not explicitly apply an orientation
  transform. If PTS is unavailable, the pipeline stores raw frame index as
  `timestamp_seconds`, which must not be used as seconds
  ([`video_decode.py:94`](../src/chess_gaze/video_decode.py#L94),
  [`pipeline.py:866`](../src/chess_gaze/pipeline.py#L866)).
- The 3D viewer's camera and depth are explicitly approximate: focal length is
  `max(frame_width, frame_height)` and depth assumes a 63 mm adult-male IPD
  ([`scene_geometry.py:69`](../src/chess_gaze/scene_geometry.py#L69),
  [`scene_calibration.py:10`](../src/chess_gaze/scene_calibration.py#L10)). Keep
  this useful visualization, but do not use it as screen truth.

### Fresh descriptive measurements

Fresh inference completed on two current 20-second local clips; the third run
was interrupted after 12.47 seconds. These values mix real gaze/head movement
with estimator noise and have no target labels. They are descriptive baselines,
not accuracy results.

| Clip | Frames / fps | Valid gaze | Median step | p95 step | p95 speed | Median face bbox |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nakamura | 1200 / 60 | 100.00% | 2.457° | 12.124° | 727.5°/s | 158 x 189 px |
| Carlsen | 600 / 30 | 100.00% | 1.617° | 7.456° | 223.7°/s | 170 x 184 px |
| Nepomniachtchi, partial | 749 / 60 | 98.66% | 1.808° | 9.463° | 567.8°/s | 182 x 218 px |

Median detected eye boxes are only about `16-35 px` wide and `4-9 px` high in
these clips. That makes raw iris offsets and eyelid geometry noisy even when
MediaPipe returns subpixel coordinates.

The historical 2026-07-05 A/B did find that the current 2x + ImageNet profile
reduced Nakamura median frame-step angle from `0.066635` to `0.042889` radians
(`-35.64%`), p95 by `19.16%`, and p99 by `18.24%`, without losing gaze coverage
([closeout](superpowers/closeouts/2026-07-05-gaze-precision-improvement.md#L88)).
That is good evidence that model input matters, but it remains a jitter proxy.

There is also local-input drift: `artifacts/` is ignored, the current Nakamura
file has 1200 frames and SHA-256
`6524928897505e614a0eae419a1b7bd0e2a8dff25ffed22db2706d02bbf909bc`, while
two real-video tests still expect 180 frames
([decode test](../tests/chess_gaze/test_video_decode_real_video.py#L9),
[pipeline test](../tests/chess_gaze/test_pipeline_real_video_contract.py#L21)).
This is another reason to repair benchmark identity checks before interpreting
future deltas.

## Dependency and model audit

The resolved local versions below were checked against their official docs and
actual imports. A newer version is not, by itself, a precision improvement.

| Direct dependency | Current role and precision relevance | Recommendation |
| --- | --- | --- |
| [`av 17.1.0`](https://pyav.basswood-io.com/docs/stable/) | Faithful decode, PTS, time base, dimensions, rotation metadata. Essential for frame identity and time-normalized metrics. | Keep. Add timestamp-validity and orientation tests; do not replace with `cv2.VideoCapture`. |
| [`huggingface-hub 1.21.0`](https://huggingface.co/docs/huggingface_hub/) | Used by the UniGaze loader/download path; local runs use a checked-in registry entry and local checkpoint. Precision-neutral. | Keep while UniGaze requires it; no online download during analysis. |
| [`mediapipe 0.10.35`](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) | Face, iris, blendshape, and transform evidence. Running mode and temporal identity can materially affect crop stability. | Keep. Test explicit identity locking and `VIDEO` mode; do not add another detector first. |
| [`numpy 2.5.0`](https://numpy.org/doc/stable/reference/routines.linalg.html) | Geometry, metrics, calibration, robust statistics. | Keep. It already provides least squares, filtering primitives, and bootstrap sampling. |
| [`opencv-python-headless 4.13.0.92`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) | Resize, perspective warp, PnP, projection, drawing. It contains everything needed for full geometric normalization. | Keep. No SciPy or extra vision package is needed for the proposed work. |
| [`pillow 12.2.0`](https://pillow.readthedocs.io/) | Image artifact and report I/O, not model inference. | Keep unless a separate cleanup proves it unused; changing it cannot improve gaze. |
| [`pydantic 2.13.4`](https://docs.pydantic.dev/latest/) | Runtime manifests and validation. Precision-neutral but protects experiment identity. | Keep; extend the smallest existing manifest rather than build a new experiment framework. |
| [`safetensors 0.8.0`](https://huggingface.co/docs/safetensors/) | UniGaze checkpoint loading through upstream code. | Keep as required by UniGaze. |
| [`timm 0.3.2`](https://huggingface.co/docs/timm/) | UniGaze architecture dependency. The installed UniGaze package requires this old exact version. | Keep pinned by the model; do not upgrade independently. |
| [`torch 2.12.1`](https://docs.pytorch.org/docs/stable/) | UniGaze inference and MPS batching. | Keep. Upstream examples use older Torch, but this version has already run the local checkpoint on MPS; do not downgrade without an equivalence benchmark. |
| [`torchvision 0.27.1`](https://docs.pytorch.org/vision/stable/) | Upstream model/transform dependency, not directly imported by project source. | Keep while required by UniGaze/Torch compatibility. |
| [`tqdm 4.68.3`](https://tqdm.github.io/) | Progress display only. | Precision-neutral; no change. |
| [`unigaze 0.1.3`](https://pypi.org/project/unigaze/) | Loads `unigaze_h14_joint` and returns `(pitch, yaw)` without confidence. The package does not ship the full official video normalizer. | Keep the checkpoint and loader; supply the missing geometric contract locally and benchmark it. |

The dev tools (`pytest 9.1.1`, `ruff 0.15.19`, and `mypy 2.1`) affect confidence
in changes, not gaze precision. The build backend is likewise irrelevant.

Relevant dependency caveats:

- MediaPipe normally declares `opencv-contrib-python`; this repository overrides
  it and supplies headless OpenCV. The current native smoke works, so dependency
  churn would add risk without a precision benefit.
- `huggingface-hub`, `safetensors`, `timm`, `torchvision`, and `unigaze` are
  upstream/dynamic model dependencies even where project source has no direct
  import. Do not remove them merely from an import grep.
- PyAV and the OpenCV wheel bundle different FFmpeg libraries on macOS and can
  emit duplicate Objective-C class warnings. Treat a reproduced crash as a
  reliability bug; it is not evidence of inaccurate gaze.
- The local model registry pins the MediaPipe task SHA-256 to
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
  and UniGaze-H14 to
  `a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f`
  ([registry](../src/chess_gaze/model_registry.json)). Preserve these exact
  assets for paired tests.

## Ranked hypotheses

| Rank | Hypothesis | Expected measurable effect | Cost |
| ---: | --- | --- | --- |
| 0 | Repair measurement and record one controlled clip | No intrinsic accuracy gain; prevents false wins | Small |
| 1 | Restore full UniGaze geometric normalization and inverse rotation | Largest likely accuracy gain; especially across head pose | Medium |
| 2 | Apply simple per-person/session calibration to 2D screen coordinates | Remove person, camera, and setup bias | Small-medium |
| 3 | Lock the streamer face temporally | Remove wrong-face and crop jumps; improve tail precision | Small-medium |
| 4 | Make mirroring and orientation explicit | Prevent complete horizontal inversion | Small |
| 5 | Gate bad frames using transparent quality evidence | Better accuracy/precision at measured coverage | Small-medium |
| 6 | Add a minimal temporal filter while preserving raw gaze | Lower within-fixation scatter, with bounded lag | Small |
| 7 | Test existing iris offsets as calibration/quality features | Possible small residual improvement | Small |
| 8 | Add chess content only as a coarse prior or weak label | Better focus classification, not necessarily raw gaze | Medium |
| 9 | Bake off ST-Gaze and L2CS only after the benchmark is honest | Possible model-level gain if UniGaze still plateaus | Medium-high |
| 10 | Stabilize or ensemble crops only if geometric normalization stalls | Possible jitter reduction at extra compute | Small-medium |

### Hypothesis 0: an honest benchmark will prevent optimizing the wrong thing

This does not directly improve gaze. It makes every claimed improvement
falsifiable and is therefore the first code change.

Why it should be done:

- The current comparator can compare different source videos and call the result
  a preprocessing delta.
- Radians per frame makes 30 fps and 60 fps incomparable.
- Consecutive-frame motion includes real saccades and head motion.
- Calibration training error is already reported, but it is not held-out error.

Minimum code change:

- Extend `gaze_precision_benchmark.py`; do not build another harness.
- Require an explicitly declared experimental variable. Source SHA, dimensions,
  decoded-frame count, and usable PTS sequence must always match; model SHA and
  other settings must match unless that exact field is the declared variable.
- Add degrees/second from positive timestamp deltas. Mark the frame-index
  timestamp fallback as unusable for time metrics.
- With a target file, add per-fixation held-out error, dispersion, coverage, and
  switch lag. Rename the current calibration `mean_absolute_error` in reporting
  or state clearly that it is mean Euclidean target-coordinate distance.
- Count in-bounds target-plane hits separately from merely valid intersections.

Smallest controlled recording:

1. Use the ordinary webcam and intended streamer layout.
2. Put the target and facecam into the same recorded video so synchronization is
   automatic.
3. Show nine known screen targets for two seconds each, in five randomized
   passes: about 90 seconds total.
4. Discard the first 500 ms after each target change; score the remaining
   1500 ms.
5. Put four targets inside the displayed chessboard and five outside, or vice
   versa.
6. Use passes 1-2 for calibration, pass 3 for selecting filter/ridge parameters,
   and passes 4-5 once for final evaluation.
7. Aggregate and resample by fixation, not frame. Otherwise 60 fps receives
   twice the statistical weight of 30 fps and autocorrelation creates false
   confidence.
8. If a default/generalization claim matters, repeat once on another day or with
   a changed camera position. One session is still enough to rank ideas for that
   person and setup.

Starting decision gates, to be adjusted if baseline test-retest noise is larger:

| Surface | Provisional gate |
| --- | --- |
| Input identity | Exact source/frame/timestamp alignment and equality of every non-declared variable; otherwise abort comparison |
| Median accuracy | At least 10% relative and 0.5° absolute gain, or 0.01 screen diagonal when angle is unavailable |
| Tail accuracy | p95 may not worsen by more than 0.25° or 0.005 screen diagonal |
| Fixation precision | At least 15% and 0.25° lower p95 dispersion |
| Coverage | No more than 2 percentage points aggregate loss or 5 points on one clip |
| Quality rejection | Compare candidates at fixed 95% and 90% retained coverage |
| Added lag | Median at most one source frame; p95 at most two frames |
| Board focus | At least 80% held-out balanced accuracy and 75% recall for both classes, with at least 40 held-out fixations |
| Exact square | At least 10 percentage points higher exact accuracy and 0.25-square lower mean Manhattan error |

For a close result, use a fixed-seed paired bootstrap of whole fixations. Ten
thousand resamples is cheap with NumPy; obvious failures do not need statistical
ceremony.

### Hypothesis 1: full geometric normalization will reduce head-pose-dependent bias

This is the strongest code hypothesis because it fixes a known model-contract
mismatch rather than adding a heuristic.

The official UniGaze video inference currently:

- expands a landmark face box by 2x;
- uses dummy intrinsics with focal length `crop_width * 4` when real camera
  intrinsics are unavailable;
- estimates head pose from six landmarks and `face_model.txt`;
- warps the face to 224 x 224 with normalized focal length 960 and normalized
  distance 600;
- predicts gaze in normalized coordinates; and
- applies `R^-1` to the predicted 3D vector before camera-space use.

The released normalizer computes `W = K_norm * S * R * K^-1`. See the
[official inference script](https://github.com/ut-vision/UniGaze/blob/main/unigaze/predict_gaze_video.py),
[normalizer](https://github.com/ut-vision/UniGaze/blob/main/unigaze/gazelib/gaze/normalize.py),
and [WACV paper](https://openaccess.thecvf.com/content/WACV2026/papers/Qin_UniGaze_Towards_Universal_Gaze_Estimation_via_Large-scale_Pre-Training_WACV_2026_paper.pdf).

Minimum implementation direction:

- Add one explicit `official_geometric_v1` preprocessing profile. Preserve the
  current profile only for paired rollback.
- Reuse current MediaPipe landmarks, current head-pose evidence, NumPy, and
  OpenCV. Do not add `face_alignment` as a production dependency.
- First reproduce the official crop/rotation on a handful of fixed frames as an
  offline oracle. Then map the required eye/nose landmarks to MediaPipe.
- Persist the normalization matrix/rotation or enough parameters to reproduce
  it. Convert model angles to a unit vector and apply `R^-1` before any repo
  coordinate convention.
- Re-derive yaw/pitch signs from vector tests. Do not preserve the current
  drawing-derived yaw negation by assumption.
- Test a synthetic known camera/pose round trip, left/centre/right directions,
  horizontal flip equivariance, and a real fixed-frame reference.

Expected effect:

- The 2018 study reports that its modified normalization formulation improved
  over the prior normalization by `9.5-32.7%` in its own evaluation settings.
  That is evidence that normalization details matter, not an expected gain for
  this checkpoint or these videos.
- The earlier local partial-contract fix improved median jitter by `35.64%`, but
  that also is not an accuracy forecast.
- Accept this hypothesis only on held-out targets using the gates above. A
  meaningful starting result is at least 10% / 0.5° lower median error without
  worse tail error or more than two points of coverage loss.

Licensing matters but does not require new process. The UniGaze model is
MG-NC-RAI-2.0. The released normalizer is CC BY-NC-SA 4.0 and requests citation.
If code or the face model is copied, preserve those terms visibly. An independent
implementation of the published equations should still cite the paper.

### Hypothesis 2: a tiny calibration will remove more bias than another generic model

Person anatomy, camera placement, sitting position, and screen geometry create
systematic bias that a person-independent model cannot observe. Published
few-shot results support calibration as a high-leverage direction, but their
numbers are model-specific:

- [FAZE](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.html)
  uses at most nine calibration samples, reports gains with three, and reports
  3.18° / 19% over prior work on GazeCapture.
- [On-device few-shot personalization](https://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/He_On-Device_Few-Shot_Personalization_for_Real-Time_Gaze_Estimation_ICCVW_2019_paper.html)
  reports 24-26% mean-error improvement with at most five points on phones.
- [Offset calibration](https://openaccess.thecvf.com/content_WACV_2020/html/Chen_Offset_Calibration_for_Appearance-Based_Gaze_Estimation_via_Gaze_Decomposition_WACV_2020_paper.html)
  shows that even one/few targets can correct subject bias, with up to 35.6% in
  its setup.

Do not promise those percentages for this affine map.

Minimum implementation direction:

- Start with an offset-only baseline. It has two parameters and can use one or a
  few targets without pretending to learn a full mapping.
- Then use the existing affine features
  `[1, gaze_yaw, gaze_pitch, head_yaw, head_pitch]` when nine spatially diverse
  targets and repeated fixations exist.
- Fit one median feature vector per fixation, not hundreds of correlated frames.
- For zero ridge, use `np.linalg.lstsq`. For ridge, do not penalize the intercept.
- Save coefficients and their target coordinate system. Keep raw UniGaze output
  unchanged and put calibrated output in a distinct field.
- Prefer direct normalized-screen prediction for board/UI focus. Do not route it
  through the pseudo-metric IPD scene unless real camera/screen geometry exists.

Expected effect and test:

- Evaluate offset-only versus affine on held-out passes 4-5. Calibration fit
  error is never the result.
- Use the main 10% / 0.5° accuracy gate. If affine does not beat the offset-only
  baseline outside the fitting session, keep the simpler offset.
- This is directly usable for the owner's controlled recordings. It cannot be
  honestly fitted for an archived streamer without target evidence; do not
  manufacture labels to force it.

### Hypothesis 3: temporal face identity will remove tail errors and simplify detection

The current code solves a video problem independently on every frame, then adds
region heuristics when a small face is hard to find. The smallest durable change
is to remember which face was selected.

Minimum implementation direction:

- Reuse the existing bbox-IoU helper. Prefer a candidate close to the previous
  bbox/landmark centroid; use current full arbitration only to acquire or
  reacquire after a small number of misses.
- Once acquired, try a moderately expanded prior facecam ROI before the ten-region
  search. This can remove work as well as improve stability.
- A/B MediaPipe `VIDEO` mode using monotonically increasing PTS and
  `detect_for_video`. Test it; do not assume it provides identity.
- MediaPipe documents landmark smoothing only for `num_faces=1`, and it does not
  promise stable multi-face order. A useful variant is one tracked face inside
  the acquired ROI, with the existing multi-region logic only for reacquisition.
- Do not add SORT, DeepSORT, a re-identification network, or another detector
  before this simple continuity rule fails.

Expected effect and test:

- Review the frames around the largest bbox and gaze discontinuities, not every
  frame. Count wrong-person selections and reacquisition time.
- Report bbox-centre/scale step, p95 fixation dispersion, p95 gaze step, and
  coverage.
- A good result removes observed identity swaps, reduces p95 fixation dispersion
  at least 15%, and stays within the coverage/lag gates. It may leave median
  target accuracy unchanged; that is still useful if catastrophic tail errors
  disappear.

### Hypothesis 4: explicit mirror/orientation policy will prevent sign catastrophes

The current hard-coded UniGaze yaw negation is based on reference drawing
direction, while the derived facecam mirror policy is explicitly unknown.
Target-plane horizontal mirroring occurs later and cannot repair mirrored model
input or head pose. A mirrored facecam can reverse every horizontal conclusion.

Minimum implementation direction:

- Add one per-video `facecam_mirrored` value with `true`, `false`, and unknown
  represented explicitly. Do not guess silently.
- Define one coordinate boundary where mirroring is removed or accounted for;
  do not apply independent sign fixes in model, overlay, and target plane.
- Make decode orientation explicit and test 0/90/180/270-degree metadata.
- On unknown archived footage, retain both horizontal interpretations or mark
  horizontal focus ambiguous. Text/logos can be supporting evidence but are not
  a robust automatic mirror detector.

Expected effect and test:

- Record left/centre/right targets and run the original plus horizontally flipped
  frames. After mapping back, unit vectors should be flip-equivariant within the
  normal model noise.
- If the policy was wrong, horizontal error should improve dramatically; use a
  provisional 50% x-error reduction as the signal. If it was already correct,
  expect no accuracy change and require no regression.

### Hypothesis 5: transparent quality gating will improve error at known coverage

UniGaze supplies no confidence. The project should not turn heuristic eye values
into a fake probability. It can, however, expose evidence that predicts bad
frames.

Start with measurements already available or cheap to compute:

- face and eye pixel dimensions;
- iris availability and left/right consistency;
- crop blur or compression score, normalized within a video rather than a global
  magic threshold;
- head-pose range and PnP reprojection error;
- bbox/landmark continuity; and
- disagreement under a small crop perturbation or horizontal-flip round trip.

Minimum implementation direction:

- Record the individual components first. Do not introduce a learned quality
  model or an abstract scoring framework.
- Rank each component against held-out target error. Keep only components with a
  monotonic accuracy-versus-coverage relationship.
- If a combined score is useful, use a small documented weighted sum or the
  maximum badness percentile. Name it `quality_score`, not confidence/probability.
- Keep raw estimates and rejection reasons.

Expected effect and test:

- Compare every approach at exactly 100%, 95%, and 90% retained coverage.
- A useful gate should reduce median error by at least 10% or p95 error/dispersion
  by 20% at 90% coverage. A result obtained only by retaining easy centre targets
  is a failure; report per-target coverage.
- Crop/flip consistency costs extra inference. Keep it optional unless its error
  separation pays for the roughly 2x compute.

### Hypothesis 6: a three-frame vector filter will reduce jitter without hiding errors

Filtering can improve only temporal precision. It should follow normalization
and calibration so it does not hide their failure modes.

Minimum implementation direction:

- Preserve raw gaze.
- Convert valid gaze to unit vectors, take a centred three-frame component-wise
  median, and renormalize. For offline video this is simpler than adding a new
  filtering dependency.
- Break the window across invalid frames or large timestamp gaps.
- Only try a simple adaptive/One-Euro low-pass if the three-frame median passes
  precision but fails motion response.

Expected effect and test:

- Require at least 15% / 0.25° lower p95 within-fixation dispersion.
- Median added target-switch delay must be at most one source frame and p95 at
  most two. Report seconds as well as frames because the clips use 30 and 60 fps.
- Held-out target accuracy and coverage must still pass. Never claim that lower
  radians/frame proves a more accurate gaze model.

### Hypothesis 7: existing iris offsets may explain residual eye motion

UniGaze sees the whole face, so explicit iris geometry may be redundant. But the
repo already computes all necessary geometry, making this a cheap nested test.

Minimum implementation direction:

- Recompute normalized iris/pupil offsets from existing `EyeRecord` pupil centres
  and eye bboxes; no schema change is required.
- Start with two features: mean left/right horizontal offset and mean vertical
  offset. Use left-right disagreement as quality evidence, not another regressor
  input at first.
- Add them only to the calibrated model, with ridge and more held-out fixations
  than coefficients.

Expected effect and test:

- Compare the same calibration splits with and without the two features.
- Keep them only for a repeatable held-out improvement of at least 5% without
  unstable coefficients or worse cross-session error.
- Given the observed 4-9 px eye heights, no improvement is a likely and useful
  result. Do not build a second handcrafted gaze engine around these landmarks.

### Hypothesis 8: chess content can refine coarse focus, but must not overwrite raw gaze

Chess video contains unusually strong semantic structure: board, clocks, engine
line, chat, camera, and move changes. This can help classify focus after gaze has
been mapped to screen coordinates.

The [EVE paper](https://arxiv.org/abs/2007.13120) reports up to 28% point-of-gaze
improvement, from `3.48°` to `2.49°`, when combining screen content, offset
augmentation, and temporal refinement. It also reports `3.85 -> 2.75 cm` and
`132.56 -> 95.59 px`. That is an upper piece of evidence from synchronized user
screens, not an expected chess result; temporal-only gains were smaller/mixed.

Minimum implementation direction:

- First support manually supplied board/UI rectangles or four board corners for
  a controlled recording. This is a few values, not a labeled dataset.
- Classify board/off-board or broad regions from the calibrated fixation median.
  Preserve the raw point and uncertainty.
- For archived streams, OpenCV frame differences in an 8x8 board ROI can identify
  likely move-change squares. Treat source/destination squares around a move as
  noisy weak evidence only; players do not necessarily look there at that time.
- Do not snap predictions to the nearest square before scoring. Do not train a
  neural content model until the rectangle/weak-label experiment wins.

Expected effect and test:

- Start with board/off-board balanced accuracy. Require at least five percentage
  points over gaze-only classification while both class recalls remain at least
  75%.
- Then test coarse board quadrants. Attempt exact squares only after the physical
  angular limits and held-out exact-square gate are met.
- If the broadcast layout is not the streamer's actual screen, label this output
  as a content prior, not measured attention.

### Hypothesis 9: another model may help after the current model is used correctly

Do not switch models before hypotheses 0-2. Otherwise a new model can win merely
because its demo performs normalization that the current integration omitted.

Primary-source candidate matrix, verified 2026-07-12:

| Candidate | Checkpoint / license | Relevant evidence | Integration risk and decision |
| --- | --- | --- | --- |
| Current UniGaze-H14 joint | Local 2.4 GB safetensors; MG-NC-RAI-2.0, approved for this noncommercial project | Final within-dataset H errors: 3.96° XGaze, 4.07° MPII, 3.01° GazeCapture, 4.34° EYEDIAP, 9.44° Gaze360. Joint H: 4.46/5.08/3.20/5.16/9.07°. | Already runs on local MPS. Keep as baseline and fix its geometric contract. |
| [ST-Gaze](https://u0172623.pages.gitlab.kuleuven.be/ST-Gaze/) | MIT code and released approximately 83 MiB checkpoint | EVE: 2.58° / 2.87 cm; no-GRU ablation 2.88°, pool-before-GRU 2.79°. 21M parameters, 6.39 GFLOPs, reported 105 fps / 800 MB on RTX 4090. | Serious later candidate. Needs separate 128x128 eye/face crops, temporal state, EVE preprocessing; upstream requirements are broad/CUDA-oriented and RTX speed does not predict MPS. Test in isolation; do not copy its whole environment into this project. |
| [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) | MIT code; Gaze360 weights hosted externally without a model card/checksum or clear separate weight terms | Paper reports 3.92° MPII and 10.41° Gaze360: mixed, not categorically worse than UniGaze across datasets. | Legitimate independent sanity model. Official packaging pulls GUI OpenCV and a Git face detector, conflicting with the lean headless stack. Isolate a minimal checkpoint adapter and license/checksum it before use. |
| [3DGazeNet](https://github.com/eververas/3DGazeNet) | Public code and Drive checkpoint, but no root license | Paper/project claim up to 23% generalization gain. | Not verified open source; old Python 3.8/CUDA environment and GPU-only extras. Exclude under this project's FOSS/simple constraints unless licensing is clarified. |

Model bakeoff rules:

- Use identical decoded frames and each model's documented normalization.
- Record checkpoint SHA, license, input/output convention, model version, device,
  and preprocessing.
- Compare raw and calibrated variants. A model that only wins after seeing held-out
  targets has leaked the test.
- Require the same accuracy, tail, coverage, and latency gates as other changes.
- Do not add a candidate to `pyproject.toml` until an isolated experiment wins.

Public generic datasets are allowed but not necessary for this first bakeoff.
EVE is especially relevant to webcam + screen + time, but contains 54 people,
12,308,334 frames, and about 105 hours; access and dataset use are noncommercial
and request-based. Its released code targets an old Torch environment. A small
validation slice may later test generalization; downloading/retraining the whole
dataset is not the next move and cannot replace the controlled chess-layout clip.

### Hypothesis 10: crop stabilization or small ensembles may reduce residual jitter

This is a fallback if full geometric normalization is blocked or leaves a clear
crop-sensitivity error.

Minimum implementation direction:

- Median-smooth bbox centre and log-scale over three to five frames, or anchor a
  square crop to stable eye/nose landmarks.
- Alternatively run the same frame with tiny crop translations/scales and take
  the normalized mean/median unit vector. Start with three variants, not a large
  test-time-augmentation framework.
- Preserve reacquisition behavior and never let smoothing clip the face after a
  real movement.

Expected effect and test:

- Measure crop-transform step, held-out target error, fixation dispersion,
  coverage, and reacquisition delay.
- Continue only for at least 5% lower held-out error or 15% lower p95 dispersion
  within the lag/coverage gates.
- Three variants cost about 3x model inference. Reject them if a one-pass
  normalized crop performs similarly.

Do not add super-resolution as the default. The current expanded face crops are
mostly downscaled to 224 x 224, while source eye height remains only a few pixels.
Super-resolution can make plausible-looking but invented iris texture and should
not be treated as recovered gaze evidence.

## What not to build now

- Do not train UniGaze, ST-Gaze, GA3CE, or another large network from scratch.
- Do not download a huge dataset before the 90-second controlled benchmark can
  rank the current pipeline.
- Do not add another face detector/tracker before a previous-bbox continuity rule
  and one-face MediaPipe video ROI are tested.
- Do not use same-scene gaze-target models such as Gazelle/Gaze-LLE as if a
  facecam sees the streamer's monitor; it does not.
- Do not add broad source refactors, experiment registries, dashboards, ADRs, PR
  ceremonies, or services for this work. Extend the existing benchmark and
  calibration modules.
- Do not keep adding streamer-layout-specific detection regions. Temporal
  acquisition/reacquisition should replace that direction if it works.
- Do not call stability accuracy, valid intersections board hits, or heuristic
  quality a probability.
- Do not snap gaze to a board square and then use the snapped result to claim
  square accuracy.

## Recommended execution order for future agents

1. Patch the existing comparator's identity/timestamp checks and target metrics.
   Record the one 90-second clip.
2. Implement `official_geometric_v1` with inverse rotation and run the paired
   held-out comparison.
3. Repair and integrate offset-only plus affine calibration; keep the simplest
   model that wins held-out.
4. Add explicit mirror/orientation handling and previous-face continuity.
5. Evaluate individual quality signals, then the three-frame vector filter.
6. Try iris features and coarse chess regions only if the earlier steps plateau.
7. Run isolated ST-Gaze and L2CS bakeoffs only if the correctly normalized and
   calibrated UniGaze result is still insufficient.

Change one experimental variable at a time and retain raw outputs. Stop an idea
as soon as it misses its decision gate; a negative result is cheaper and more
useful than permanent speculative machinery.

## Primary sources

All were checked on 2026-07-12.

- UniGaze [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Qin_UniGaze_Towards_Universal_Gaze_Estimation_via_Large-scale_Pre-Training_WACV_2026_paper.pdf),
  [repository](https://github.com/ut-vision/UniGaze),
  [video inference](https://github.com/ut-vision/UniGaze/blob/main/unigaze/predict_gaze_video.py),
  and [normalizer](https://github.com/ut-vision/UniGaze/blob/main/unigaze/gazelib/gaze/normalize.py)
- [Revisiting Data Normalization for Appearance-Based Gaze Estimation](https://www.collaborative-ai.org/publications/zhang18_etra/)
- MediaPipe [Face Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
  and [options reference](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions)
- [Few-Shot Adaptive Gaze Estimation](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.html)
- [On-Device Few-Shot Personalization](https://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/He_On-Device_Few-Shot_Personalization_for_Real-Time_Gaze_Estimation_ICCVW_2019_paper.html)
- [Offset Calibration](https://openaccess.thecvf.com/content_WACV_2020/html/Chen_Offset_Calibration_for_Appearance-Based_Gaze_Estimation_via_Gaze_Decomposition_WACV_2020_paper.html)
- [MTGLS](https://openaccess.thecvf.com/content/WACV2022/papers/Ghosh_MTGLS_Multi-Task_Gaze_Estimation_With_Limited_Supervision_WACV_2022_paper.pdf)
- EVE [paper](https://arxiv.org/abs/2007.13120),
  [project](https://ait.ethz.ch/eve), and [code](https://github.com/swook/EVE)
- ST-Gaze [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Personnic_Learning_Spatio-temporal_Feature_Representations_for_Video-based_Gaze_Estimation_WACV_2026_paper.pdf)
  and [official project/code](https://u0172623.pages.gitlab.kuleuven.be/ST-Gaze/)
- L2CS-Net [paper](https://arxiv.org/abs/2203.03339) and
  [official repository](https://github.com/Ahmednull/L2CS-Net)
- 3DGazeNet [paper/project](https://eververas.github.io/3DGazeNet/) and
  [repository](https://github.com/eververas/3DGazeNet)
