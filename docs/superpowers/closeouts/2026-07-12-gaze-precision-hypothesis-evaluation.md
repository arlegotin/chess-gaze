# Gaze Precision Hypothesis Evaluation Closeout

Date: 2026-07-13
Branch: `improvements-1`
Plan base: `34d7876`
Implementation HEAD before the closeout commit: `050b82d`

## Result

All H0-H10 proposals in `docs/gaze-precision-hypotheses.md` received the
smallest valid experiment allowed by the three `*_short.mp4` inputs. Four
durable correctness surfaces were kept: experiment provenance/comparability,
finite target-plane accounting, calibration numerics, and declared video
orientation. No proxy was promoted to an accuracy claim.

| Hypothesis | Question tested | Valid gate | Outcome | Retained result |
| --- | --- | --- | --- | --- |
| H0 measurement | Can a paired run prove input/model/timestamp identity and report time-normalized, finite-plane observables? | Invalid pairs abort; one-variable pairs reproduce on every clip | **kept** | PTS/model provenance, strict v2 comparator, finite-plane count, corrected short-video constants |
| H1 geometry | Can the official UniGaze geometric path be reproduced with verified inputs and redistributable geometry? | Revision, digest, six-point mapping, and asset-license gates all pass | **blocked** | None; `face_model.txt` redistribution permission is unresolved |
| H2 calibration | Does the existing affine fit behave correctly on rank-deficient and ridge cases? | Focused numerical regressions | **kept** | `lstsq` at zero ridge; intercept excluded from ridge penalty |
| H3 identity | Does continuity, prior ROI, or MediaPipe VIDEO fix all five confirmed face-selection failures without new swaps or excessive coverage loss? | 5/5 fixed, no new reviewed failure, aggregate loss <=2 pp, per-clip loss <=5 pp, then resume equivalence | **rejected** | None; variants fixed 3/5, 2/5, and 0/5 respectively |
| H4 orientation/mirror | Is there an independently reproducible orientation/sign defect? | Synthetic orientation oracle or labeled horizontal-direction truth | **kept** | Decode/display-orientation correctness fix; mirror policy remains inconclusive |
| H5 quality | Do cheap signals predict error at fixed coverage? | Target error and per-target coverage | **inconclusive** | None; proxy-only separation recorded |
| H6 filtering | Does a centred three-vector median improve fixation precision without harmful lag? | Fixation truth and target-switch timing | **inconclusive** | None; smoothness-only proxy improved |
| H7 iris | Do existing pupil offsets improve held-out calibration error? | Held-out target residuals | **inconclusive** | None; availability/noise/proxy association only |
| H8 chess prior | Can board-local changes be extracted, and can they improve focus classification? | Layout/change feasibility plus calibrated gaze/attention truth | **blocked** | Feasibility passed; focus integration blocked and nothing shipped |
| H9 models | Is an alternate model better under identical held-out accuracy conditions? | Licensed/checksummed candidate and held-out target benchmark | **blocked** | Current primary-source preflight only; no checkpoint selected or downloaded |
| H10 crops | Is crop sensitivity material, and does a sample ensemble provide an observable win? | Materiality first; accuracy truth required for retention | **inconclusive** | Sensitivity was material, but the sample ensemble worsened smoothness proxies at about 3x model cost; nothing shipped |

## Evidence Boundary And Corpus

The clips contain no fixation target, physical screen geometry, attention label,
or gaze-to-screen calibration. Coverage, step size, consistency, runtime, and
board-local changes are proxies. They cannot establish angular accuracy,
point-of-gaze accuracy, fixation precision, chess focus, or comparative model
quality.

Every empirical input discovery used `artifacts/input/*_short.mp4` and asserted
the exact set below. Synthetic MP4 orientation fixtures also ended in
`_short.mp4`.

| Input | Source SHA-256 | Frames / fps | Dimensions | PTS sequence SHA-256 | PTS usable |
| --- | --- | ---: | ---: | --- | --- |
| `artifacts/input/carlsen_short.mp4` | `48505b38898a843c5b03d9cfa717efda2a915f0c5399c81369be20d316f6fc01` | 600 / 30 | 1920x1080 | `00ebaaeaf8b08505a75257042739905b1f2150727bad315c22c522ee6c5c3800` | true |
| `artifacts/input/nakamura_short.mp4` | `6524928897505e614a0eae419a1b7bd0e2a8dff25ffed22db2706d02bbf909bc` | 1,200 / 60 | 1920x1080 | `495d2618252f41334fb230c5ae06d4d2e90039ba0a8c9ef992920e2e9e75a067` | true |
| `artifacts/input/nepo_short.mp4` | `aa24fb658a3a3723d8b953d01c5ddf174d60978b6a5a2312c5c79f4b23c36b8c` | 1,200 / 60 | 1246x720 | `495d2618252f41334fb230c5ae06d4d2e90039ba0a8c9ef992920e2e9e75a067` | true |

The retained H0/H3/H5-H7/H10 records use `unigaze-h14-joint`, model SHA-256
`a336e7234738e9a9517fc6af7a9bc69cee16958388ad648d48c0f6b0df42ac8f`,
and `reference_face2x_imagenet`. Native face work used
`models/mediapipe/face_landmarker.task`, SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

Evidence routing after cleanup:

| Hypotheses | Per-clip inference source | Report path before/after cleanup |
| --- | --- | --- |
| H0 | The six reference/legacy run paths listed below | Three persisted `artifacts/output/gaze-hypotheses/h0/*-comparison.json` reports |
| H1 | Direct frames 0/midpoint/last from each approved source | Deleted `artifacts/experiments/2026-07-12-gaze-precision/h1/`; hashes summarized here |
| H2 | Synthetic calibration samples; no video | Focused tracked tests; no generated report |
| H3, H5, H6, H7 | The exact three H0 reference run paths below | Deleted `h3-review.json`, `h3-variants/`, `h5-quality.json`, `h6-filter.json`, and `h7-iris.json`; hashes summarized here |
| H4 | Direct approved frames 0/midpoint/last plus generated `rotation_*_short.mp4` | Deleted `h4/orientation.json` and `h4/flip-equivariance.json`; hashes summarized here |
| H8 | Direct full decode of each approved source | Deleted `h8-board-events.json` and sheets; hashes/labels summarized here |
| H9 | No video opened | Primary-source preflight in this closeout; no generated report |
| H10 | The exact H0 reference runs and 18 pre-bound approved-source frames | Deleted `h10-crops.json`; hashes/metrics summarized here |

## H0: Honest Measurement

Six fresh MPS runs changed only `unigaze_preprocessing`, used batch size 7 and
`--no-resume`, and completed with schema/count checks passing. The reports are
persisted and remain available after experiment cleanup.

| Clip | Reference run | Legacy run | Comparison report / SHA-256 |
| --- | --- | --- | --- |
| Carlsen | `artifacts/output/gaze-hypotheses/h0/reference/carlsen_short/runs/20260712T171441Z-ccc62c74` | `artifacts/output/gaze-hypotheses/h0/legacy/carlsen_short/runs/20260712T172239Z-fc254e4d` | `artifacts/output/gaze-hypotheses/h0/carlsen_short-comparison.json` / `37d49bfffc135247e08b4f59dfbe628addf3c6b3d6a50a32e04abd34ccaca39b` |
| Nakamura | `artifacts/output/gaze-hypotheses/h0/reference/nakamura_short/runs/20260712T171704Z-d410ce25` | `artifacts/output/gaze-hypotheses/h0/legacy/nakamura_short/runs/20260712T172457Z-9cb5ad56` | `artifacts/output/gaze-hypotheses/h0/nakamura_short-comparison.json` / `a1f3dc2dfff55658120f5d02cf776c7554f5f9acb3dd6695241d140b462334e3` |
| Nepomniachtchi | `artifacts/output/gaze-hypotheses/h0/reference/nepo_short/runs/20260712T171931Z-dfee1df3` | `artifacts/output/gaze-hypotheses/h0/legacy/nepo_short/runs/20260712T172724Z-ba51b246` | `artifacts/output/gaze-hypotheses/h0/nepo_short-comparison.json` / `8d19459bfc089c10dca5d652dd59389219a16c0ffd4ea3b5ac185d0764fd61e3` |

Reference-to-legacy descriptive results:

| Clip | Profile | Valid gaze | Sphere hits | Plane intersections / in-bounds | Median speed deg/s | P95 speed deg/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Carlsen | reference | 1.000000 | 599 | 0 / 0 | 48.4997 | 223.6912 |
| Carlsen | legacy | 1.000000 | 599 | 0 / 0 | 63.0930 | 293.1974 |
| Nakamura | reference | 1.000000 | 1,185 | 0 / 0 | 147.4409 | 727.4605 |
| Nakamura | legacy | 1.000000 | 1,182 | 0 / 0 | 229.0759 | 899.8517 |
| Nepomniachtchi | reference | 0.991667 | 1,188 | 0 / 0 | 102.7479 | 532.8520 |
| Nepomniachtchi | legacy | 0.991667 | 1,189 | 0 / 0 | 33.3078 | 279.4395 |

The reference profile has lower speed proxies for Carlsen and Nakamura but
higher values for Nepomniachtchi. This mixed proxy result does not establish a
preprocessing accuracy winner. A deliberately invalid Carlsen-reference versus
Nakamura-legacy pair aborted and named source, frame, timestamp, and PTS paths.

Retained commits: `2d0cbd8`, `e31f931`, `50d4235`, `04aef4f`, and final
source-bound fixture repair `050b82d`.

## H1: Official UniGaze Geometry

Verified on 2026-07-12 against official UniGaze revision
`9c240fbe33f3d6146970a77b7c8fa06a7e60019e`. The pinned face-model SHA-256 is
`0c943d1d48627d97038b64f9a73816b9ab80a002ce81a8f04d532da2f4c337d7`.
The operational mapping `[33,133,362,263,98,327]` to dlib
`[36,39,42,45,31,35]` passed manual original-resolution review on frames
0/midpoint/last of every clip (9/9):

| Clip | Reviewed frames | Mapping evidence |
| --- | --- | --- |
| Carlsen | 0, 300, 599 | `artifacts/experiments/2026-07-12-gaze-precision/h1/carlsen_short-*-mapping.png` |
| Nakamura | 0, 600, 1,199 | `artifacts/experiments/2026-07-12-gaze-precision/h1/nakamura_short-*-mapping.png` |
| Nepomniachtchi | 0, 600, 1,199 | `artifacts/experiments/2026-07-12-gaze-precision/h1/nepo_short-*-mapping.png` |

The deleted temporary manifest was
`artifacts/experiments/2026-07-12-gaze-precision/h1/mapping-manifest.json`,
SHA-256 `f6412a4db03ea2cdf2cd4c190d7cef88e05566fa973ece17d37adf506b872e83`;
the manual checklist SHA-256 was
`6922829d31e30e14ca21cdc2ad3c7312eed627de2c6d5fd03a0f5de6d82bc389`.

The official root/model license covers the model, while `normalize.py` carries
CC BY-NC-SA 4.0 terms. The numeric `face_model.txt` has no file-level source,
license, citation, or redistribution notice, and the sole addition commit says
only `fixed missing file`. Assigning either neighboring license to that data
would be an unsupported inference. H1 was therefore blocked before production
implementation; planned Tasks 6-7 and ADR-0007 were skipped.

## H2: Calibration Numerics

No video was needed. Two synthetic public-API regressions reproduced the
existing solver defects:

- rank-deficient zero-ridge data raised `numpy.linalg.LinAlgError: Singular matrix`;
- ridge `1_000_000` shrank a true `(2,-3)` intercept to approximately
  `(1e-5,-1.5e-5)`.

The focused GREEN suite passed 6 tests. The retained shared fit boundary now
uses `np.linalg.lstsq(..., rcond=None)` for zero ridge and sets
`regularizer[0,0] = 0` for positive ridge. Runtime calibration, offset fitting,
and accuracy claims were deliberately not retained because the corpus has no
targets. Retained commit: `74310a2`.

## H3: Temporal Face Identity

### Baseline Ground Truth

The deterministic union of per-clip top-15 bbox-centre, bbox-scale, and gaze
steps produced 75 events. All contact sheets were inspected at original
resolution. Ledger path before cleanup:
`artifacts/experiments/2026-07-12-gaze-precision/h3-review.json`, SHA-256
`d6d636293c75346b9e14dc9bbedf09b272c34b3afd58b5b3d78b61455b46c960`.

Every reviewed representative and label:

- Carlsen `streamer`: f120, f125, f131, f211, f219, f222, f276, f299,
  f302, f348, f351, f368, f401, f404, f407, f410, f528, f545, f548.
  `false_crop`: f205 (actual bad f204, wall plaque) and f379 (actual bad f378,
  wall plaque).
- Nakamura `streamer`: f8, f11, f16, f35, f42, f66, f87, f253, f290, f379,
  f608, f611, f615, f628, f637, f727, f915, f931, f936, f939, f942, f1114,
  f1123, f1127, f1152. No incorrect representative.
- Nepomniachtchi `streamer`: f10, f25, f47, f69, f129, f156, f189, f248,
  f308, f459, f462, f469, f474, f494, f499, f504, f509, f516, f526, f529,
  f536, f666, f725, f844, f901, f904, f1093, f1142. `false_crop`: f477;
  actual bad frames f477-f478 select the wall ornament and f479 is
  `missing_visible_face`.

Totals: 72 streamer representatives, 3 false-crop representatives, five exact
bad frames (four false crops and one missed visible face), zero other-person or
ambiguous labels.

### Independent Variant Campaign

The final frozen inference source was
`h3-variants/probe-h3-task10-final-20260712T204900Z-inference.py`, SHA-256
`0db7a7b596bda8ae6e5d8e0a088f02681a20a8a1e604f7df7afc8994a92a19f1`.
The 124 regenerated event IDs and sheet hashes matched the preserved visual
snapshot; final summary SHA-256 was
`9f2837fd3afa85aa52b5075458629852aa9300c9e16de09017d268949703abac`.

Every candidate event label is accounted for below; unqualified IDs are
`streamer`:

- Continuity / Carlsen: f117, f121, f125, f204, f211, f217, f225, f232,
  f290, f300, f303, f315, f323, f342, **f378=`false_crop`**, f379, f402,
  f405, f408. Nakamura: f8, f16, f35, f87, f253, f290, f379, f608, f615,
  f628, f637, f1114, f1123, f1127, f1152. Nepomniachtchi: f10, f47, f457,
  f462, f474, f477, f478, **f479=`missing_visible_face`**, f489, f494, f499,
  f504, f509, f526, f529, f536, f546, f725, f872.
- Prior ROI / Carlsen: f1, f98, f113, f119, f122, f126, f201, f204, f213,
  f217, f288, f299, f302, f341, f378, f399, f402, f405, f408. Nakamura:
  f1, f5, f37, f63, f127, f242, f262, f287, f290, f302, f320, f363, f381,
  f437, f451, f455, f569, f575, f1107. Nepomniachtchi: f454, f459, f462,
  f469, f474, **f477-f479=`false_crop`**, f492, f497, f504, f509, f519.
- MediaPipe VIDEO / Carlsen: **f204 and f378=`missing_visible_face`**.
  Nakamura: f8, f37, f51, f55, f66, f289, f292, f303, f306, f318, f574,
  f669, f1109, f1112, f1115. Nepomniachtchi:
  **f477-f479=`missing_visible_face`**.

| Variant | Candidate faces | Coverage loss | Confirmed failures fixed | New reviewed swaps/false crops | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current-candidate continuity | 2,993 / 3,000 | -0.100 pp | 3 / 5 | 0 | reject |
| Prior ROI | 2,994 / 3,000 | -0.133 pp | 2 / 5 | 0 | reject |
| MediaPipe VIDEO | 1,195 / 3,000 | 59.833 pp | 0 / 5 | 0 | reject |

Continuity left Carlsen f378 and Nepo f479 unresolved. Prior ROI left Nepo
f477-f479 unresolved. VIDEO left all five unresolved and collapsed Carlsen/Nepo
coverage. The planned resume-equivalence task did not run for any variant:
resume verification was gated after correctness/coverage retention, and all
three variants had already failed. No temporal state or resume behavior entered
production.

## H4: Decode Orientation And Mirroring

The approved clips declared no rotation. Their post-fix direct-PyAV and shared
decoder RGB sequence hashes were byte-identical: Carlsen
`d4dc80ee66e3a286d29ed58cca9916f45a6805ad6b2f118ad7bdc74ed7141498`,
Nakamura `027dcdd8eb4a1b30adeaa0b6de73fbedc16c164f6781586dddc1fcae021e68fb`,
and Nepomniachtchi
`af00d3344eef2be3d2560233be60a237a3e7c7dfef60f0b664f65bb54ca3ec4d`.

Generated `rotation_{0,90,180,270}_short.mp4` fixtures encoded one asymmetric
marker with PyAV display matrices. Before the fix, 90/180/270 failed; after the
fix all decoded to upright 192x128 with maximum sampled MPEG-4 channel error 1.

| Declared rotation | Encoded dimensions | Inspection | Marker | Fixture SHA-256 |
| ---: | ---: | ---: | --- | --- |
| 0 | 192x128 | none | pass | `1a664cbcbaa161a36924c0861a5f00f4e5ec2c7c3cdbf78651d2eb0bbf1a7ce5` |
| 90 | 128x192 | 90 | pass | `3ca4d709dde1c4743881663ea799a8ddcc6b48424baba886ac86f72c4a1ad13e` |
| 180 | 192x128 | 180 | pass | `8f5a6d0a92e70f13e7c6e305492e9e1ce13e3c2339f98bdc8e314de48080c408` |
| 270 | 128x192 | 270 | pass | `9ff32629b7686dbea33797fe02dbfb9bbc11eadc605cb2c3591de760dcacf7d8` |

Fixed-frame original/flip MPS inference had 18/18 face/prediction coverage.
After mapping the flip back once with camera `x := -x`, angular disagreement was:

| Clip | frame 0 | midpoint | last |
| --- | ---: | ---: | ---: |
| Carlsen | 10.463847° | 3.618141° | 12.980382° |
| Nakamura | 7.896394° | 9.918045° | 12.059857° |
| Nepomniachtchi | 17.645048° | 14.114197° | 8.316302° |

Median/p95/max were 10.463847° / 16.232708° / 17.645048°. Without labeled
left/centre/right gaze this cannot choose original versus mirrored facecam
semantics. No `facecam_mirrored` field, sign toggle, or mirror detector was
added. Deleted temporary reports were `h4/orientation.json` (SHA-256
`cc00be9c66e6d57c730006c4d0f8625357ffd9cdf5ec42473d088e9ac7b37503`)
and `h4/flip-equivariance.json` (SHA-256
`421203a6721a3090f02a5e07bd2cee0fef9378ef63d1cf8336f52b7615f6ea1a`).
Retained commits: `33a2651` and reviewer follow-up `199c6b0`.

## H5: Transparent Quality Evidence

Twelve signals were recomputed on all 3,000 H0 reference records. The label was
the clip-local p95 appearance-gaze step (150 outliers among 2,983 valid pairs),
plus the five H3 failures. It is explicitly a proxy label.

| Signal | Gaze-step rank correlation | Outlier rate at 95% / 90% coverage | H3 rank correlation |
| --- | ---: | ---: | ---: |
| Face width | -0.0008 | 4.534% / 4.712% | -0.0215 |
| Face height | -0.0054 | 4.534% / 4.490% | ~0 |
| Left-eye width | -0.0231 | 4.536% / 4.232% | -0.0081 |
| Left-eye height | -0.0192 | 5.065% / 4.976% | 0.3572 |
| Right-eye width | -0.0042 | 4.534% / 4.304% | -0.0107 |
| Right-eye height | -0.0651 | 4.889% / 4.642% | 0.3277 |
| Both iris available | constant | 5.065% / 4.976% | -0.0640 |
| Iris horizontal disagreement | 0.0158 | 4.502% / 4.642% | -0.1074 |
| Head-pose magnitude | 0.0548 | 4.571% / 4.417% | 0.2283 |
| Bbox centre step | 0.1935 | 4.035% / 3.370% | 0.2525 |
| Bbox log-area step | 0.1797 | 3.965% / 3.667% | 0.3492 |
| Appearance-gaze step | 0.3752 | 1.263% / 0% | 0.2632 |

Appearance step is circular because it defines the proxy label. Bbox steps are
the clearest non-circular separation but are biased by layout and do not measure
target error. No score, confidence, threshold, gate, or field was retained.
Deleted report: `h5-quality.json`, SHA-256
`49269a15eb8125ea975a91177ba91593b68e8b626d4717105a8afadecd7d99c4`.

## H6: Three-Vector Median

The exact componentwise median plus renormalization was applied offline at
2,979/3,000 positions; raw/filtered validity remained 2,990/3,000. These are
adjacent-step smoothness proxies.

| Clip | Raw median / p95 / p99 rad | Filtered median / p95 / p99 rad | Reduction median / p95 / p99 |
| --- | --- | --- | --- |
| Carlsen | .0282160 / .1301382 / .3015503 | .0159563 / .0876721 / .1659450 | 43.45% / 32.63% / 44.97% |
| Nakamura | .0428888 / .2116097 / .3278919 | .0176316 / .1310103 / .2200150 | 58.89% / 38.09% / 32.90% |
| Nepomniachtchi | .0298881 / .1550004 / .3870902 | .0117908 / .1149286 / .3488238 | 60.55% / 25.85% / 9.89% |
| Pooled | .0333639 / .1795896 / .3478160 | .0152623 / .1156496 / .2472942 | 54.26% / 35.60% / 28.90% |

The 150 raw p95 candidates clustered to 112 events; all had a filtered match
within +/-2 frames. Two amplitude ratios exceeded one (maximum 1.8758), and no
true switch time exists. No filter or runtime setting was retained. Deleted
report: `h6-filter.json`, SHA-256
`4a11286e2f49e61ff8e32ad0d7daf2550df2fadc5cec2624bdd95d5117df2c87`.

## H7: Existing Iris Offsets

Both-eye features were finite on 2,990/3,000 frames; 2,983/2,997 possible
within-clip temporal pairs were valid. Aggregate feature distributions and
their weak association with clip-local p95 gaze steps were:

| Feature | Aggregate median [p05,p95] | Step median / p95 | Proxy rank correlation |
| --- | --- | --- | ---: |
| Left horizontal | -.01075 [-.07485,.31484] | .00574 / .04533 | .1701 |
| Left vertical | -.11766 [-.31111,.03165] | .00996 / .12017 | .1829 |
| Right horizontal | -.01183 [-.10243,.17575] | .00494 / .03906 | .1771 |
| Right vertical | -.13252 [-.28809,-.01116] | .00796 / .10768 | .1751 |
| Mean horizontal | -.01364 [-.08457,.24042] | .00463 / .03759 | .1753 |
| Mean vertical | -.12328 [-.28617,-.01001] | .00827 / .09928 | .1967 |
| Horizontal disagreement | .04138 [.00444,.16708] | .00548 / .03946 | .1850 |

All four measurable H3 false-crop frames had finite features and p95 gaze-step
outliers; Nepo f479 had no selected face. Finite iris evidence can therefore
coexist with a wrong crop. No feature/regressor was retained. Deleted report:
`h7-iris.json`, SHA-256
`c771fa4be02b6dfe033db23af290fe0d58c2864792d384e3870793a47bffe334`.

## H8: Board-Content Feasibility

Board rectangles were fixed before scoring, split by exact rational eighths,
and trimmed 10% per square. The probe processed 3,000 frames and 2,997
consecutive transitions.

| Clip | Fixed rectangle `[x0,y0,x1,y1)` | Reviewed top events |
| --- | --- | --- |
| Carlsen | `[980,87,1885,992)` | f41, f50, f2, f35, f29: all `board_animation` |
| Nakamura | `[953,77,1879,1003)` | f1035, f456, f669: `piece_move`; f612, f601: `board_animation` |
| Nepomniachtchi | `[0,0,720,720)` | f285, f1147: `piece_move`; f653, f290, f663: `board_animation` |

The corresponding two largest square MAD values were Carlsen f41
93.032/64.976, f50 87.250/73.076, f2 86.312/72.995, f35 71.346/0.003,
f29 69.586/0.053; Nakamura f1035 89.340/57.000, f612 66.970/0.015,
f456 62.128/55.991, f601 66.720/0.033, f669 64.772/50.628; and
Nepomniachtchi f285 103.081/51.778, f653 84.707/0.156, f1147
84.053/44.937, f290 73.304/4.208, f663 71.185/13.302.

All three 0/midpoint/last alignment sheets showed stable 8x8 edges; all 15
event sheets were manually reviewed and none was unrelated noise. Feasibility
passed, but no gaze-to-screen calibration or attention truth exists, so focus
integration is blocked. No rectangle, detector, classifier, focus label, or
square snap was retained. Deleted ledger: `h8-board-events.json`, SHA-256
`3680c5769fac1e413a2a8cb7ec14fd6202cf9a3f7ee5c140b1d1333f0625ac91`.

## H9: Alternate-Model Preflight

Primary sources were refreshed on 2026-07-13. No video was opened, checkpoint
downloaded, dependency changed, or adapter written.

| Candidate | Revision / checkpoint | Input contract | Output contract | Confidence / temporal state | Decision |
| --- | --- | --- | --- | --- | --- |
| UniGaze-H14 joint | code `9c240fbe...`; HF `d3f8335...`; local 2,528,232,848-byte checkpoint SHA above | one geometrically normalized 224x224 RGB face | `pred_gaze` radians in implementation-established `[pitch,yaw]`, convertible to 3-vector and inverse-normalized | no gaze confidence; frame-independent | incumbent only |
| ST-Gaze | `43abefaba5c1e92cd5f65f3c2e85ba0a3587cf31`; 86,803,705-byte EfficientNet-B3 checkpoint available remotely | synchronized 128x128 face and left/right eye sequences plus EVE metadata | per-eye `[pitch,yaw]` radians plus GRU hidden state; EVE wrapper negates the vector before screen mapping | no learned gaze confidence; recurrent state required | serious later candidate; weight license and MPS unverified |
| L2CS-Net | `a4d8f7fa5436a2b2b9f088471623b552a85811bd`; external unmanifested Gaze360 checkpoint | independent RetinaFace-selected full-face tensor | separate 90-bin pitch/yaw logits converted to radians; public pipeline also returns detector evidence | no calibrated gaze confidence; frame-independent | serious later candidate; weight license/checksum/MPS unverified |
| 3DGazeNet | `196396c7d00d3bae8fa7ff08b3c79f8286cb5b3a`; external Drive bundles | left eye, right eye, face crops plus detector/template assets | unit gaze vector plus dense eye/face vertices; camera-axis/sign order unverified | no gaze confidence; optional demo-only prior-frame average | excluded until code/weight licensing is explicit |

| Candidate | Code / weight terms | Published labeled condition (not comparable across rows) | Apple MPS evidence |
| --- | --- | --- | --- |
| UniGaze-H14 joint | MG-NC-RAI-2.0 model/checkpoint; normalizer separately CC BY-NC-SA 4.0; face-model data permission unresolved | Joint-H angular error 4.46° ETH-XGaze, 5.08° MPIIFaceGaze, 3.20° GazeCapture, 5.16° EYEDIAP, 9.07° Gaze360 | Existing local model-backed campaigns pass |
| ST-Gaze | MIT code; applicability to committed checkpoint unverified | 2.58° angular and 2.87 cm point-of-gaze on labeled EVE single-view test | None verified; official environment is CUDA/NVIDIA-oriented |
| L2CS-Net | MIT code; external checkpoint has no separate terms, manifest, or checksum | 3.92° MPIIFaceGaze leave-one-person-out; 10.41° Gaze360 | None verified |
| 3DGazeNet | No root code license; external bundles have no verified weight license/checksum | 4.2° ETH-XGaze, 3.3° GazeCapture, 8.8° Gaze360, 4.3° MPIIFaceGaze under its published multi-dataset setup | None verified; official environments are CUDA-oriented |

Published metrics use different labeled datasets and preprocessing, so they do
not rank models on streamer clips. Current local versions were PyAV 17.1.0,
OpenCV-headless 4.13.0.92, MediaPipe 0.10.35, NumPy 2.5.0, Torch 2.12.1,
and UniGaze 0.1.3. A host MPS tensor probe passed, but that says nothing about
uninstalled candidates. H9 remains blocked until a held-out labeled target
benchmark and licensed/checksummed candidate assets exist.

## H10: Crop Sensitivity And Sample Ensemble

Frame selection was frozen before inference: three stable and three Task 9
worst-reviewed representatives per clip (18 total). Five perturbations were
`x=-2%,0,+2%` and centred scales `0.95,1.05`.

| Clip / cohort | Frames | Median spread | Maximum spread |
| --- | --- | ---: | ---: |
| Carlsen / all | 58, 50, 59, 379, 205, 276 | 9.174432° | 11.382896° |
| Nakamura / all | 841, 953, 1173, 16, 379, 931 | 22.268126° | 55.201685° |
| Nepomniachtchi / all | 1002, 187, 994, 477, 156, 474 | 16.469889° | 32.108616° |
| All stable | 9 frames | 14.588023° | 55.201685° |
| All reviewed-worst | 9 frames | 12.917171° | 52.987632° |
| All | 18 frames | 13.752597° | 55.201685° |

Materiality passed. The three-translation sample ensemble used 54 tensors
instead of 18. It changed sampled median/p95 adjacent step from
3.138637°/37.026452° to 9.693001°/40.706465° and took 2.692909 s versus
0.907472 s, ratio 2.967485x.

The first artifact was invalidated: aliased selection dictionaries were mutated
after their in-memory ledger was hashed. Canonicalization moved after decoded
PTS/RGB provenance binding but still before runtime preparation/inference. Two
corrected runs matched in every non-timing field, canonical SHA-256
`71fbf9dedbed8fcfda97f63d72c6593b9417039721c5df69ea4f34800ee7118b`.
Deleted final report: `h10-crops.json`, SHA-256
`a13e8ee512c8b0c89a5a107437fbc2fdf6bd5cf19aa113877033e1cec64b9785`;
selection ledger SHA-256
`95c8a087eea2effe8742a6032060bc248fa9cd27c01a1bac92b87c4acaa042e6`.

This establishes crop sensitivity, not which crop is accurate. The sample
ensemble also failed its weaker smoothness/cost observation, so no production
mechanism was retained.

## Retained Root-Cause Repairs

| Commit / owner | Reproduced failure | Root cause | Durable repair and regression | Focused GREEN | Deliberately not retained |
| --- | --- | --- | --- | --- | --- |
| `2d0cbd8`; `frame_records.py`, `video_decode.py`, `unigaze_runtime.py` | Nine expected missing-field/helper failures | Runs lacked deterministic decoded PTS identity/usability and model asset checksum | Stream/hash PTS+time-base, validate strictly increasing usable time, persist model checksum; legacy defaults remain compatible | 98 passed | New manifest/framework |
| `e31f931`; `scene_records.py`, `scene_artifacts.py` | Fresh summary had no in-bounds count | Aggregation collapsed valid infinite-plane intersections with finite-plane hits | Compatible `in_bounds_target_plane_hit_frames` and shared aggregation regression proving 2 intersections / 1 in-bounds | 94 passed (host loopback gate) | Reinterpretation of historical `valid_target_plane_hit_frames` |
| `50d4235`; `gaze_precision_benchmark.py` | 19 RED failures accepted unrelated sources/settings and lacked speed/finite-hit fields | Comparator did not load full manifests/identity and divided motion by frames | Exact declared-variable allowlist, embedded/standalone validation, source/model/runtime/calibration/frame/PTS equality, deg/s from positive PTS, v2 report | 24 focused; 230 non-socket broad | Permanent experiment service or accuracy metric without labels |
| `04aef4f`; three real-video contract tests | Exact RED gate: 6 failed, 1 passed, each `1200 == 180` | Ignored Nakamura input changed while constants remained stale | Three expected counts corrected to independently inspected 1,200 | 7 passed in exact host-native gate | Runtime-derived/loosened expectation |
| `74310a2`; `gaze_calibration.py` | Singular zero-ridge solve; ridge shrank constant intercept | Normal equations square conditioning; identity penalty included intercept | `lstsq` for zero ridge; slope-only ridge penalty; two public regressions | 6 passed | Runtime calibration or target-accuracy claim |
| `33a2651`, `199c6b0`; `video_decode.py` | 90/180/270 marker failures; later legacy-stream pixel mismatch and explicit 90-to-0 change accepted | Shared decoder read stream metadata but not frame DISPLAYMATRIX consistently; inspection/pixel paths diverged | One `_frames_with_rotation` owner; explicit frame matrix (including 0) overrides stream fallback; right-angle validation; oriented dimensions/RGB | 17 passed including approved-short contracts | Mirror/sign policy or downstream coordinate heuristic |
| `050b82d`; `test_face_observation_real_video.py` | Final native gate passed presence/bounds but failed two exact-box ledgers on all seven Nakamura samples | Expectations were last refreshed for ignored SHA `6364e160...` / 180 frames, while the filename now resolves to approved SHA `65249288...` / 1,200 frames; the exact ledger had no source binding | Assert the approved source SHA before exact comparisons and replace only the seven boxes after original-resolution visual review | Exact two native nodes passed twice, 2/2 in 1.88 s each | Detector/runtime change or loosened geometric bounds |

No default runtime profile changed, no new module or ownership boundary was
introduced, and H1 was blocked. Therefore README, source-layout, and ADR files
did not require updates.

## Temporary-Evidence Cleanup And Leftover Audit

After extracting the paths, hashes, labels, and metrics above, Task 19 removed
`artifacts/experiments/2026-07-12-gaze-precision`. The retained H0 run/report
tree and all three approved input videos were left untouched.

The required source/test audit searched for
`quality_score|facecam_mirrored|three.frame|iris.*feature|crop.*ensemble|ST.Gaze|L2CS`.
It found only the pre-existing `_quality_scores()` helper and its two callers in
`qa_summary.py` at lines 1026, 1036, and 1045. That helper was introduced by
`c662ab9e` on 2026-06-25 and ranks retained raw-frame JPEGs by blur/exposure for
QA sampling; it is not H5's rejected gaze-quality field or gate. No other audit
term matched. `git ls-files` reported no tracked file below the removed tree,
`test ! -e` passed, and `git status --short` showed only this closeout. Fresh
hashes confirmed all three H0 reports and all three approved inputs were
unchanged. No failed/inconclusive experiment left a toggle, field, helper,
test, dependency, or abstraction.

A separate `.mp4` source/test audit found only two tests that can open existing
recorded non-short inputs. Both are excluded by exact node ID:

- `tests/chess_gaze/test_face_observation_real_video.py::test_mediapipe_observer_keeps_carlsen_face_centers_in_visible_person_region`
  (`carlsen_1.mp4`);
- `tests/chess_gaze/test_eye_observation_real_video.py::test_nepo_edge_window_marks_off_frame_eye_missing_without_empty_crop`
  (`nepo_2.mp4`).

Other non-short `.mp4` strings in model-free tests are schema placeholders or
paths for generated temporary fixtures, not reads of empirical recorded input.
No branch change introduced a hidden non-short empirical path.

## Verification

Fresh Task 19 gate results are recorded here after cleanup. Native MediaPipe/MPS
tests are serialized and run separately from the non-native suite.

| Gate | Fresh result |
| --- | --- |
| `UV_CACHE_DIR=.uv-cache uv run pytest -q -m 'not native_mediapipe and not local_socket'` | Final post-review rerun: **485 passed, 14 deselected, 18 warnings** in 29.09 s |
| `UV_CACHE_DIR=.uv-cache uv run pytest -q -m local_socket` | Initial sandbox reproduction: 2 failed only at `socket.bind` with `PermissionError`; final post-review host rerun: **2 passed, 497 deselected** in 1.06 s |
| `UV_CACHE_DIR=.uv-cache uv run ruff check .` | **All checks passed** |
| `UV_CACHE_DIR=.uv-cache uv run mypy` | **Success: no issues in 78 source files** |
| `git diff --check` | **passed**, no output |
| `UV_CACHE_DIR=.uv-cache uv run ruff format --check .` | **known pre-existing failure**: would reformat `src/chess_gaze/cli.py`, `src/chess_gaze/gaze_observation.py`, `src/chess_gaze/unigaze_preprocessing.py`, and `tests/chess_gaze/test_gaze_observation.py`; 74 files already formatted |
| Initial exact approved-input native gate | **8 passed, 2 failed, 18 warnings** in 734.42 s; both failures were stale unbound Nakamura exact-box evidence, while face presence and width/height limits passed |
| Focused source-binding repair gate | The exact two failed native nodes passed twice: **2 passed** in 1.88 s, then **2 passed** in 1.88 s |
| Final post-review exact approved-input native gate | **10 passed, 18 warnings** in 792.63 s; all nodes used only the approved `*_short.mp4` inputs |

None of the four format-drift files changed between plan base `34d7876` and
implementation HEAD `050b82d`; their latest shared source commit is `9a9d45a`
from 2026-07-05. They were not reformatted because that would be unrelated,
untested scope and would obscure the hypothesis diff.

The final native gate ran after commit `050b82d` and after independent review,
serialized without another native/MPS process. It intentionally excluded the
two exact non-short node IDs above. H1 was not retained, so
`tests/chess_gaze/test_unigaze_preprocessing_real_video.py` was also excluded.

The fresh independent final reviewer approved the retained source diff after
134 focused inference-free tests, scoped Ruff, and `git diff --check`. The sole
Important completion finding was the then-pending post-`050b82d` full native
rerun. The 10/10 final native result above remediates it; subsequent changes
were limited to recording final gate evidence in this closeout, so the reviewer
required no further review.

## Residual Risk

- The allowed corpus cannot establish gaze accuracy, point-of-gaze accuracy,
  fixation precision, target-switch lag, calibration benefit, chess focus, or
  comparative model quality.
- H0 run artifacts are ignored local evidence. The closeout preserves exact
  paths and hashes, but a clone without those artifacts retains only the
  summarized results and regression tests.
- H1 remains blocked on asset-level permission, not geometry or mapping parity.
- H3 manual review targets worst events rather than every frame. It is adequate
  to reject the tested variants, not to estimate global face-selection error.
- H4 fixes declared orientation but does not infer subregion mirroring.
- H5-H7 and H10 are proxy-only. Their numerical effects must not become default
  policy without labeled fixation/target evidence.
- H8 proves layout/change extraction only. It does not authorize attention or
  board-square claims.
- H9 upstream metrics are dataset-specific; candidate weight licenses,
  checksums, exact output conventions, and MPS compatibility remain unresolved
  where explicitly noted.

## Primary Sources And Verification Dates

H1 official UniGaze repository, pinned video path, normalizer, face model,
license, README/model card, repository tree, and asset history were checked on
2026-07-12. H9 refreshed official UniGaze, ST-Gaze, L2CS-Net, and 3DGazeNet
repositories/papers/checkpoint metadata on 2026-07-13 at the revisions recorded
above. Local checkpoint and runtime hashes were verified on the same task dates.

Primary URLs:

- UniGaze repository/revision, model card, paper, video inference, and
  normalizer: <https://github.com/ut-vision/UniGaze/tree/9c240fbe33f3d6146970a77b7c8fa06a7e60019e>,
  <https://huggingface.co/UniGaze/UniGaze-models/tree/d3f8335cd4b7d249adbc32389986ce49b52f6f72>,
  <https://openaccess.thecvf.com/content/WACV2026/papers/Qin_UniGaze_Towards_Universal_Gaze_Estimation_via_Large-scale_Pre-Training_WACV_2026_paper.pdf>,
  <https://github.com/ut-vision/UniGaze/blob/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/unigaze/predict_gaze_video.py>,
  <https://github.com/ut-vision/UniGaze/blob/9c240fbe33f3d6146970a77b7c8fa06a7e60019e/unigaze/gazelib/gaze/normalize.py>.
- ST-Gaze project, pinned repository, and paper:
  <https://u0172623.pages.gitlab.kuleuven.be/ST-Gaze/>,
  <https://gitlab.kuleuven.be/u0172623/ST-Gaze/-/tree/43abefaba5c1e92cd5f65f3c2e85ba0a3587cf31>,
  <https://openaccess.thecvf.com/content/WACV2026/papers/Personnic_Learning_Spatio-temporal_Feature_Representations_for_Video-based_Gaze_Estimation_WACV_2026_paper.pdf>.
- L2CS-Net pinned repository and paper:
  <https://github.com/Ahmednull/L2CS-Net/tree/a4d8f7fa5436a2b2b9f088471623b552a85811bd>,
  <https://arxiv.org/abs/2203.03339>.
- 3DGazeNet project and pinned repository:
  <https://eververas.github.io/3DGazeNet/>,
  <https://github.com/eververas/3DGazeNet/tree/196396c7d00d3bae8fa7ff08b3c79f8286cb5b3a>.
