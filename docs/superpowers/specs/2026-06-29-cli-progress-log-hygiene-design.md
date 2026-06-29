# CLI Progress And Native Log Hygiene Design

Date: 2026-06-29

## Status

Approved for implementation by the user's 2026-06-29 task request. The
executable plan is
`docs/superpowers/plans/2026-06-29-cli-progress-log-hygiene.md`.

## Goal

Make long `chess-gaze analyze` runs visibly healthy while preserving the
machine-readable stdout contract.

The motivating command is:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  uv run chess-gaze analyze artifacts/input/nepo_2.mp4 \
  --output-root artifacts/output \
  --models-root models \
  --unigaze-device mps \
  --unigaze-batch-size 7
```

It printed dependency-owned native messages from PyAV/OpenCV/MediaPipe/TFLite
and then no frame progress. On `nepo_2.mp4` this is a serious usability problem:
local inspection shows 28,141 decoded frames and a 469.066 second AV1 video, so
the run can look stuck for a long time even when it is processing correctly.

## Findings

The controlled reproduction on `artifacts/input/nakamura_short.mp4` with the
same MPS/batch settings completed successfully:

- decoded frames: 180;
- `qa_summary.final_status`: `complete`;
- `records/frames.jsonl`: 180 lines;
- `records/scene_frames.jsonl`: 180 lines;
- `scene_summary.valid_monitor_hit_frames`: 180;
- retained visual overlays for frames 0, 90, and 179 looked coherent.

The visible terminal noise is not evidence of corrupted output:

- The Objective-C duplicate class warning is emitted when the same process loads
  both PyAV's bundled FFmpeg `libavdevice.62...dylib` and OpenCV's bundled
  FFmpeg `libavdevice.61...dylib`. Both link AVFoundation and define
  `AVFFrameReceiver` / `AVFAudioReceiver`. The repo already guarantees only one
  OpenCV Python provider (`opencv-python-headless`), but that does not remove
  PyAV's separate FFmpeg build.
- The MediaPipe/TFLite `I0000`/`W0000` startup messages originate from the
  Face Landmarker native runtime. The normal Python/Abseil log-threshold
  environment variables did not suppress these messages in a direct local probe.
- The repeated `portable_clearcut_uploader.cc` messages originate from symbols
  and strings inside the installed MediaPipe native dylib
  (`mediapipe/tasks/c/libmediapipe.dylib`). They are dependency telemetry upload
  failures, not frame-processing failures from this repo.
- UniGaze prints `Loaded UniGaze pretrained weights...` from the installed
  `unigaze==0.1.3` package while loading local weights. This is useful for
  debugging but pollutes `chess-gaze analyze` stdout, whose documented success
  contract is the run directory followed by the viewer path.

## Behavior

`chess-gaze analyze` should:

- keep stdout limited to the run directory and viewer path on success;
- emit a frame progress bar with ETA on stderr for interactive terminals;
- support explicit progress control for tests and scripts:
  `--progress auto`, `--progress on`, and `--progress off`;
- update frame progress only after frame records have been durably committed and
  `analysis_state.json` has been advanced;
- initialize resumed runs at the committed frame count;
- suppress only known dependency-owned native chatter during analysis;
- continue to show repo-owned CLI errors and unexpected dependency errors.

`chess-gaze view` and preflight failures should not import the full analysis
pipeline before they need it. This prevents analyze-only native warnings from
appearing for usage, missing-input, and viewer commands.

## Non-Goals

- Do not change frame decoding, model selection, MediaPipe mode, UniGaze batch
  semantics, scene geometry, or artifact schemas.
- Do not claim the PyAV/OpenCV duplicate FFmpeg class risk is eliminated. This
  task contains the terminal symptom and documents the residual packaging risk.
- Do not suppress all stderr globally. Stable repo-owned errors and unknown
  dependency errors must remain visible.
- Do not add a multi-video queue or persistent progress database.

## Dependency And Practice Matrix

Verified on 2026-06-29.

| Candidate / practice | Evidence | Decision |
| --- | --- | --- |
| `tqdm` progress bar | Installed locally as `tqdm 4.68.3`; official docs expose `total`, `initial`, `unit`, `file`, `disable`, and ETA behavior. Source: https://tqdm.github.io/docs/tqdm/. | Select. Add as a direct dependency so runtime use is explicit. |
| Rich progress | Installed as a transitive dependency, polished but heavier and less necessary for one scalar frame loop. Source: https://rich.readthedocs.io/en/latest/progress.html. | Reject for this narrow CLI task. |
| Custom text progress | No dependency, but would reimplement ETA, terminal refresh, and non-TTY handling. | Reject. Higher bug surface than using `tqdm`. |
| Progress on stdout | Conflicts with README's stdout contract and the benchmark parser that scans stdout for the run directory. | Reject. Use stderr. |
| Progress per decoded frame | Resume design says `records/frames.jsonl` is the source of truth and raw decode counts must not imply progress. | Reject. Update after commit/checkpoint. |
| Python/Abseil log env vars for MediaPipe | Direct local probe with `GLOG_minloglevel=3 ABSL_MIN_LOG_LEVEL=3` still printed MediaPipe/TFLite native startup logs. | Reject as sole fix. |
| Narrow native stderr filter | Captures C/C++ writes to fd 2 and filters only known AVFoundation, MediaPipe/TFLite startup, and Clearcut uploader lines. | Select for analyze scope. Keep Python progress on the original stderr fd. |
| Blanket stderr redirect to `/dev/null` | Would hide real failures and CLI errors. | Reject. |
| Replacing PyAV or OpenCV | Existing ADR/spec work selected PyAV for faithful decode metadata and OpenCV for PnP/drawing/resize; replacing either would be a larger architecture task. | Reject for this task. |

Other relevant primary docs:

- MediaPipe Face Landmarker Python docs:
  https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python.
- PyAV container/decode docs: https://pyav.org/docs/stable/api/container.html.
- OpenCV Python wheel project: https://pypi.org/project/opencv-python-headless/.

## Testing Strategy

Unit and contract tests:

- CLI missing-input and `view` paths should not import analyze-only code.
- `--progress` options should pass the selected progress mode into
  `AnalyzeRequest`.
- Pipeline progress callback should receive initial, committed batch, and final
  counts in order, including resumed runs.
- The native stderr filter should suppress known multi-line Clearcut blocks and
  known startup warnings while passing unknown stderr through.
- UniGaze local weight loading should not write to stdout.

Real verification:

- Run `nakamura_short.mp4` with `--unigaze-device mps --unigaze-batch-size 7`
  and `--progress on`.
- Confirm progress output appears on stderr, stdout remains the run directory
  and viewer path, and known native noise is absent.
- Confirm `qa_summary.final_status == "complete"` and all 180 frames are present.
- Visually inspect representative processed overlays from the real run.

## Residual Risk

The underlying PyAV/OpenCV duplicate FFmpeg/AVFoundation packaging conflict is
still present in the process. If future crashes appear inside AVFoundation or
FFmpeg symbols, the durable architecture fix is dependency isolation or removing
one bundled FFmpeg provider from the runtime. This task keeps the current
validated artifact pipeline and makes the CLI usable and less misleading.
