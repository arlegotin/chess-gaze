# Gaze Precision Hypothesis Evaluation Design

Date: 2026-07-12

Status: approved

Implementation update, 2026-07-13: H1 resumed after `repo_owner` approved the
pinned `face_model.txt` asset under `MG-NC-RAI-2.0`. The equation, sign, native-
frame, asset, QA, comparator, and coverage gates passed, so
`official_geometric_v1` is retained as the default under
[ADR-0007](../../development/decisions/0007-restore-unigaze-geometric-normalization.md).
The paired short-video motion metrics remain descriptive; no gaze-accuracy or
fixation-precision claim is authorized.

## Goal

Evaluate every hypothesis in `docs/gaze-precision-hypotheses.md` in rank order,
using only local real-video inputs whose names match `*_short.mp4`. Keep only
changes that either pass an honest observable gate or repair an independently
verified contract defect. Remove failed and inconclusive mechanisms from the
production tree.

This task improves both the reliability of future gaze comparisons and any
runtime surface that survives the evidence gates. It does not manufacture gaze
targets for archived streamer footage or relabel stability as accuracy.

## Approved Evidence Policy

The available clips have no known fixation targets, physical screen geometry,
or attention labels. The user approved a strict policy:

- proxy stability, coverage, consistency, or runtime measurements may guide the
  next experiment;
- a proxy cannot make an angular-accuracy, point-of-gaze, calibration, quality,
  filtering, focus-classification, or model-selection hypothesis pass;
- code may still be retained when it fixes a separately reproducible model,
  geometry, provenance, numerical, or identity-selection defect;
- each hypothesis ends as `kept`, `rejected`, `inconclusive`, or `blocked`.

`kept` means the stated gate passed or an independent correctness contract was
verified. `rejected` means a valid experiment failed. `inconclusive` means the
allowed data lacks the truth needed to decide. `blocked` means a documented
prerequisite failed before a valid experiment could run.

## Instruction Conflict Resolution

The hypothesis backlog recommends a new controlled target recording and delays
the alternate-model bakeoff until an honest accuracy benchmark and calibration
exist. The task restricts real-video inputs to `*_short.mp4` and asks to try all
hypotheses. These constraints are reconciled as follows:

- no controlled recording, public gaze dataset, or non-short local video will
  be used;
- every hypothesis receives the smallest experiment that can answer any valid
  part of it;
- an experiment that cannot test its claimed effect is recorded as
  inconclusive or blocked instead of being implemented and scored with a proxy;
- H9 receives a current primary-source, asset, license, and integration
  preflight, but no checkpoint integration while an accuracy bakeoff is
  impossible.

## Allowed Corpus

Fresh PyAV inspection on 2026-07-12 found:

| Input | SHA-256 | Frames / fps | Dimensions | PTS time base | Rotation |
| --- | --- | ---: | ---: | --- | --- |
| `artifacts/input/carlsen_short.mp4` | `48505b38898a843c5b03d9cfa717efda2a915f0c5399c81369be20d316f6fc01` | 600 / 30 | 1920 x 1080 | `1/15360` | none |
| `artifacts/input/nakamura_short.mp4` | `6524928897505e614a0eae419a1b7bd0e2a8dff25ffed22db2706d02bbf909bc` | 1200 / 60 | 1920 x 1080 | `1/15360` | none |
| `artifacts/input/nepo_short.mp4` | `aa24fb658a3a3723d8b953d01c5ddf174d60978b6a5a2312c5c79f4b23c36b8c` | 1200 / 60 | 1246 x 720 | `1/15360` | none |

Tests may use arrays or generated fixtures named `*_short.mp4`. Empirical runs
must discover inputs with `artifacts/input/*_short.mp4` and fail if no files are
found. Test commands that load other recorded videos are excluded and listed in
the closeout.

## Approach

Use staged falsification:

1. repair comparison identity and timing before interpreting a delta;
2. produce fresh baselines for all allowed clips;
3. change one declared experimental variable at a time;
4. run the smallest focused probe before a full-corpus inference pass;
5. keep, remove, or block the idea immediately after its gate;
6. record every result and residual uncertainty in one closeout.

Permanent experiment-framework code is out of scope. The existing benchmark,
runtime modules, schemas, and CLI remain the durable surfaces. Temporary probes
may exist during an experiment but must be removed unless their behavior is
retained.

## Architecture

### Paired comparison boundary

```text
*_short.mp4
  -> baseline and candidate runs from the same source tree
  -> manifest and timestamp identity validation
  -> one declared-variable comparison
  -> focused metric or correctness oracle
  -> keep, remove, or block
  -> closeout evidence
```

`gaze_precision_benchmark.py` remains the comparison owner. It will load the run
manifest, video manifest, calibration, frame records, and scene evidence. A
comparison must abort unless source SHA, dimensions, decoded-frame count, usable
PTS sequence, model asset checksum, and every non-declared setting match.

The comparison API requires one named experimental variable. Supported names
are enumerated beside the comparator and map to exact allowed manifest or
calibration field paths; unknown names and unexpected additional differences
are errors. H0 initially adds the existing UniGaze preprocessing group. A later
retained H3 implementation may add a face-selection group with its exact fields.

The persisted video contract must distinguish usable PTS from the frame-index
fallback and include a deterministic hash of the raw PTS/time-base sequence.
Time-normalized metrics use only positive deltas from runs whose manifest says
the complete PTS sequence is usable. The benchmark reports degrees per second,
not just radians per frame.

Target-plane reporting distinguishes valid infinite-plane intersections from
hits inside the configured finite plane. No target/fixation schema is added
because no allowed input supplies those labels.

### UniGaze geometric normalization

```text
selected face + mapped landmarks + verified camera/face geometry
  -> perspective warp and inverse normalization rotation
  -> normalized UniGaze tensor
  -> normalized UniGaze model vector
  -> inverse rotation into camera coordinates
  -> convert the model vector to the existing physical-ray yaw/pitch convention
```

`unigaze_preprocessing.py` owns the normalization contract and
`gaze_observation.py` owns model input/output conversion. The implementation
uses the installed NumPy and OpenCV rather than a new production dependency.
The equation implementation must match the official normalizer on identical
inputs and must reproduce the official inverse-rotation result before it can be
retained.

UniGaze's training preparation stores `vector_to_pitchyaw(-gaze_norm)`: its
predicted vector is the opposite of the physical eye-to-target ray. The pinned
video inference preserves that convention through `R^-1` and negates the vector
when projecting the physical ray. After inverse rotation, repository pitch is
therefore `asin(model_vector_camera.y)` and repository yaw is
`atan2(-model_vector_camera.x, model_vector_camera.z)`. This composes with the
existing scene conversion `(x, -y, -z)` to recover the physical camera ray
`-model_vector_camera`.

If upstream face-model values are required, they must be registered with a
source URL, license, local checksum, and citation. If those facts or a stable
MediaPipe-to-reference landmark mapping cannot be verified, H1 is blocked and
the experimental code is removed.

That conditional gate resolved on 2026-07-13 with repository-owner approval,
the pinned asset checksum, and 9/9 original-resolution landmark reviews. The
accepted implementation keeps required-asset validation profile-selective so
rollback profiles do not acquire an unused face-model dependency.

### Temporal face identity

```text
previous selected bbox + current candidates
  -> IoU and centroid continuity
  -> select a current candidate or explicitly reacquire
```

The first H3 variant changes only current-candidate selection. A previous bbox
may guide selection but may never be emitted as a successful current
observation. Prior-ROI acquisition and MediaPipe `VIDEO` mode are separate
later variants.

Any retained temporal implementation must restore state from committed records
or replay the required detector history so an interrupted/resumed analysis is
equivalent to an uninterrupted run. A variant that cannot satisfy that boundary
is rejected.

`face_observation.py` already exceeds the source-layout review threshold. If H3
survives, the existing arbitration and temporal-selection behavior will be
split only at the already documented face-selection boundary; unrelated face,
eye, or pipeline refactoring remains out of scope.

## Hypothesis Experiments

| Hypothesis | Minimum experiment | Retention decision |
| --- | --- | --- |
| H0 measurement | Regression tests for source/settings mismatch, declared-variable allowance, PTS validity, degrees/second, and in-bounds hits; fresh baseline on every allowed clip | Keep when invalid comparisons abort and valid paired comparisons are reproducible |
| H1 geometry | Official-equation oracle on fixed frames, synthetic pose/vector round trips, left/centre/right sign checks, flip equivariance, then paired runs on all clips | Keep as corrected default only after oracle parity, resolved provenance, and coverage within the backlog limits; accuracy remains unclaimed |
| H2 calibration | Rank-deficient zero-ridge regression, ridge-intercept regression, and synthetic held-out evaluation | Keep `lstsq` and unpenalized-intercept correctness repairs; do not integrate runtime calibration or offset fitting without labels |
| H3 identity | Manually classify the largest bbox discontinuities; test continuity, prior ROI, and `VIDEO` independently | Keep the smallest variant that removes confirmed wrong selections, preserves coverage, introduces no new swaps, and remains resume-equivalent |
| H4 orientation | Audit all source metadata, test synthetic 0/90/180/270 transforms, and test horizontal-flip equivariance on fixed frames | Keep only an independently demonstrated coordinate/sign correction; otherwise inconclusive |
| H5 quality | Bin cheap existing evidence against confirmed face-selection failures and gaze-step outliers at fixed retained coverage | Record proxy separation but ship no gaze-quality gate without target error |
| H6 filtering | Apply an offline centred three-vector median to every clip, breaking windows across invalid frames and PTS gaps; report jitter and delay relative to raw changes | Record proxy effect; ship no filter without fixation and target-switch truth |
| H7 iris | Recompute mean horizontal/vertical offsets and left-right disagreement; quantify availability, noise, and stability | Ship no calibration feature without held-out targets |
| H8 chess prior | Verify board rectangles and move-change evidence can be extracted from all three layouts | Block focus integration because gaze-to-screen calibration and attention truth are absent |
| H9 models | Refresh primary-source checkpoint, license, preprocessing, runtime, and integration evidence | Block integration because H2 cannot produce an honest held-out accuracy benchmark |
| H10 crops | Run small crop perturbations on stable and worst-discontinuity frames; test a three-crop ensemble only on the sample if sensitivity is material | Ship nothing from consistency or smoothness alone |

H3 independent correctness is observable without gaze labels: a selected bbox
can be manually confirmed as the streamer, another person, or a false crop.
Coverage loss limits remain two percentage points aggregate and five on any
clip. Gaze accuracy, fixation dispersion, and target-switch latency are not
inferred from those identity results.

## Model And Dependency Decision

Primary sources were refreshed on 2026-07-12. The task keeps the existing
runtime stack and does not select a new production model.

| Candidate | Verified availability and license | Task fit and runtime risk | Decision |
| --- | --- | --- | --- |
| UniGaze-H14 joint | Existing local 2.4 GB safetensors, SHA-256 `a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f`; model card says MG-NC-RAI-2.0; `unigaze==0.1.3` is already locked | The design-time MPS path ran locally but omitted official geometric normalization. The 2026-07-13 implementation update above repairs that contract and keeps the pitch/yaw-without-confidence output. | Keep as baseline with the retained, verified geometric contract before any bakeoff. |
| ST-Gaze | Official project exposes code and an EfficientNet-B3 checkpoint; GitLab identifies an MIT license | Requires temporal sequences, eye/face preprocessing, state, and a separate checkpoint. Published EVE conditions do not match the unlabeled streamer clips; RTX evidence does not establish MPS behavior. | Serious later candidate; blocked here by unavailable accuracy truth. |
| L2CS-Net | Official repository is MIT; checkpoint is a separately downloaded `L2CSNet_gaze360.pkl` without a verified local checksum in this repo | ResNet50 pipeline and its own face-processing integration conflict with the existing headless, MediaPipe-owned path. It outputs yaw/pitch but no task-relevant calibrated confidence contract. | Serious independent later candidate; do not download or integrate for a proxy-only comparison. |
| 3DGazeNet | Official repository provides Drive-hosted checkpoints but no root license file or release; no local checksum | Requires a separate old/CUDA-oriented environment, face preprocessing, and 3D eye-mesh output integration. Reuse rights are unverified. | Exclude unless code and weight licensing is clarified. |

Primary URLs:

- UniGaze repository, model card, video inference, and normalizer:
  https://github.com/ut-vision/UniGaze,
  https://huggingface.co/UniGaze/UniGaze-models,
  https://github.com/ut-vision/UniGaze/blob/main/unigaze/predict_gaze_video.py,
  https://github.com/ut-vision/UniGaze/blob/main/unigaze/gazelib/gaze/normalize.py
- ST-Gaze project and repository:
  https://u0172623.pages.gitlab.kuleuven.be/ST-Gaze/,
  https://gitlab.kuleuven.be/u0172623/ST-Gaze
- L2CS-Net repository: https://github.com/Ahmednull/L2CS-Net
- 3DGazeNet project and repository:
  https://eververas.github.io/3DGazeNet/,
  https://github.com/eververas/3DGazeNet
- MediaPipe Face Landmarker Python guide:
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

The existing MediaPipe asset remains pinned at SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The runtime assumption is Apple Silicon MPS through `.venv`, where a fresh
unsandboxed check reports Torch 2.12.1 and MPS available. Sandboxed execution
does not expose MPS. Alternative CUDA/RTX claims are not treated as MPS
evidence.

## Failure Handling

Every production behavior begins with a failing focused test. The test must be
observed failing for the intended reason before implementation, then passing
after the smallest fix. Exploratory code is discarded before production TDD.

An unexpected run or test failure stops speculative changes and triggers:

1. exact reproduction and evidence capture;
2. data-flow tracing to the durable failing boundary;
3. one explicit root-cause hypothesis;
4. one minimal test;
5. one fix and focused/broad verification.

After three failed fixes or repeated shared-state surprises, stop that
hypothesis and reassess the architecture. Failed and inconclusive mechanisms
must not leave dead toggles, schema fields, helpers, dependencies, or partially
used abstractions.

## Verification

Each retained hypothesis requires focused tests and fresh paired evidence.
Completion requires:

- all unit and non-video integration tests;
- native real-video tests whose inputs match `*_short.mp4`;
- explicit exclusion and reporting of tests that load other recorded videos;
- fresh MPS baseline/candidate runs over all three allowed clips for each
  retained runtime change;
- uninterrupted/resumed equivalence for retained temporal behavior;
- `uv run ruff check .`;
- `uv run ruff format --check .`;
- `uv run mypy`;
- independent final code review and resolution of critical/important findings.

Large generated run artifacts remain ignored. The closeout records source
hashes, run paths, exact commands, per-clip metrics, manual-review frame IDs,
outcomes for H0-H10, root causes, durable surfaces, regressions, and residual
uncertainty.

## Commit Strategy

1. Commit the supplied hypothesis backlog unchanged with this approved design.
2. Commit H0 benchmark/provenance repairs after focused verification.
3. Commit each independently retained H1, H2, or H3 result after its own gate.
4. Do not commit failed or inconclusive mechanisms.
5. Commit the final closeout and any required ADR/source-layout corrections
   after broad verification.

No branch switch, worktree, dependency churn, PR creation, or unrelated cleanup
is part of this task.

## Residual Uncertainty

The allowed clips can validate provenance, time metrics, model-contract math,
runtime coverage, crop/identity correctness, and proxy stability. They cannot
establish true gaze accuracy, screen point accuracy, fixation precision,
target-switch latency, board focus, square focus, calibration benefit, quality
ranking by error, or comparative model quality. The closeout must preserve that
boundary even when a proxy improves substantially.
