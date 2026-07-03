from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, TextIO, cast

from chess_gaze.errors import CliErrorCode
from chess_gaze.native_log_filter import suppress_known_native_analysis_logs
from chess_gaze.scene_viewer import ViewerServerError, serve_viewer

AnalyzeVideoCallable = Any
analyze_video: AnalyzeVideoCallable | None = None
AnalyzeRequest: type[Any] | None = None
PipelineError: type[Exception] | None = None

INPUT_NOT_FOUND_EXIT = 10
UNSUPPORTED_VIDEO_EXIT = 11
MODEL_ASSET_MISSING_EXIT = 12
MODEL_ASSET_CHECKSUM_MISMATCH_EXIT = 13
MODEL_LICENSE_NOT_APPROVED_EXIT = 14
SCHEMA_VALIDATION_FAILED_EXIT = 15
PIPELINE_NOT_IMPLEMENTED_EXIT = 16
USAGE_EXIT = 2
GENERAL_FAILURE_EXIT = 1

ERROR_EXIT_CODES = {
    CliErrorCode.USAGE: USAGE_EXIT,
    CliErrorCode.INPUT_NOT_FOUND: INPUT_NOT_FOUND_EXIT,
    CliErrorCode.UNSUPPORTED_VIDEO: UNSUPPORTED_VIDEO_EXIT,
    CliErrorCode.MODEL_ASSET_MISSING: MODEL_ASSET_MISSING_EXIT,
    CliErrorCode.MODEL_ASSET_CHECKSUM_MISMATCH: MODEL_ASSET_CHECKSUM_MISMATCH_EXIT,
    CliErrorCode.MODEL_LICENSE_NOT_APPROVED: MODEL_LICENSE_NOT_APPROVED_EXIT,
    CliErrorCode.SCHEMA_VALIDATION_FAILED: SCHEMA_VALIDATION_FAILED_EXIT,
    CliErrorCode.PIPELINE_NOT_IMPLEMENTED: PIPELINE_NOT_IMPLEMENTED_EXIT,
}


def system_exit_code(exc: SystemExit) -> int:
    if isinstance(exc.code, int):
        return exc.code
    if exc.code is None:
        return 0
    return USAGE_EXIT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-gaze")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("video_path")
    analyze.add_argument("--output-root", default=None)
    analyze.add_argument("--models-root", default=None)
    analyze.add_argument("--config", default=None)
    analyze.add_argument("--unigaze-device", choices=("cpu", "mps"), default=None)
    analyze.add_argument("--unigaze-batch-size", type=int, default=None)
    analyze.add_argument(
        "--progress",
        choices=("auto", "on", "off"),
        default="auto",
        help="show frame progress on stderr: auto, on, or off",
    )
    analyze.add_argument(
        "--save-frames",
        action="store_true",
        default=None,
        dest="save_frame_images",
        help="retain raw decoded PNGs and processed overlay JPEGs",
    )
    analyze.add_argument(
        "--save-crops",
        action="store_true",
        default=None,
        dest="save_crop_images",
        help="retain eye crop PNGs under crops/",
    )
    analyze.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        default=True,
        help="create a fresh run instead of resuming a compatible interrupted run",
    )
    analyze.add_argument(
        "--qa-summary",
        action="store_true",
        default=False,
        dest="generate_qa_summary",
        help="run strict QA closeout and write qa_summary.json",
    )
    view = subparsers.add_parser("view")
    view.add_argument("run_dir")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return system_exit_code(exc)

    if args.command == "view":
        return _run_view(args, parser)

    if args.command != "analyze":
        return _usage_error(parser, "unknown command")

    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"INPUT_NOT_FOUND: {video_path}", file=sys.stderr)
        return INPUT_NOT_FOUND_EXIT

    pipeline_error: Exception | None = None
    result: Any | None = None
    with suppress_known_native_analysis_logs() as native_logs:
        AnalyzeRequestType, PipelineErrorType, analyze_video_func = (
            _pipeline_dependencies()
        )
        progress = _AnalyzeProgressBar(args.progress, native_logs.stderr)
        try:
            result = analyze_video_func(
                AnalyzeRequestType(
                    video_path=video_path,
                    output_root=(
                        Path(args.output_root) if args.output_root is not None else None
                    ),
                    models_root=(
                        Path(args.models_root) if args.models_root is not None else None
                    ),
                    config_path=Path(args.config) if args.config is not None else None,
                    unigaze_device=args.unigaze_device,
                    unigaze_batch_size=args.unigaze_batch_size,
                    save_frame_images=args.save_frame_images,
                    save_crop_images=args.save_crop_images,
                    generate_qa_summary=args.generate_qa_summary,
                    resume=args.resume,
                    progress_callback=(progress.callback if progress.enabled else None),
                )
            )
        except PipelineErrorType as exc:
            pipeline_error = exc
        finally:
            progress.close()

    if pipeline_error is not None:
        code = cast(Any, pipeline_error).code
        print(f"{code.value}: {pipeline_error}", file=sys.stderr)
        return ERROR_EXIT_CODES.get(code, GENERAL_FAILURE_EXIT)

    if result is None:
        raise AssertionError("analyze_video did not return a result")
    print(result.layout.run_dir)
    print(f"viewer: {result.viewer_index_path}")
    return 0


def _run_view(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        server = serve_viewer(Path(args.run_dir), host=args.host, port=args.port)
    except ViewerServerError as exc:
        return _usage_error(parser, str(exc))
    except OSError as exc:
        return _usage_error(parser, f"Could not start viewer server: {exc}")

    try:
        print(server.url, flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
    return 0


def _pipeline_dependencies() -> tuple[type[Any], type[Exception], AnalyzeVideoCallable]:
    global AnalyzeRequest, PipelineError, analyze_video

    if AnalyzeRequest is None or PipelineError is None or analyze_video is None:
        from chess_gaze.pipeline import (
            AnalyzeRequest as LoadedAnalyzeRequest,
        )
        from chess_gaze.pipeline import (
            PipelineError as LoadedPipelineError,
        )
        from chess_gaze.pipeline import (
            analyze_video as loaded_analyze_video,
        )

        AnalyzeRequest = LoadedAnalyzeRequest
        PipelineError = LoadedPipelineError
        if analyze_video is None:
            analyze_video = loaded_analyze_video

    return AnalyzeRequest, PipelineError, analyze_video


class _AnalyzeProgressBar:
    def __init__(self, mode: str, stderr: TextIO) -> None:
        self._enabled = mode == "on" or (mode == "auto" and stderr.isatty())
        self._stderr = stderr
        self._bar: Any | None = None
        self._completed_frames = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def callback(self, event: Any) -> None:
        if not self._enabled:
            return
        if self._bar is None:
            tqdm = cast(Any, import_module("tqdm")).tqdm

            self._bar = tqdm(
                total=event.total_frames,
                initial=event.completed_frames,
                unit="frame",
                desc="analyze frames",
                dynamic_ncols=True,
                file=self._stderr,
            )
            self._completed_frames = event.completed_frames
            if event.completed_frames >= event.total_frames:
                self.close()
            return

        delta = event.completed_frames - self._completed_frames
        if delta > 0:
            self._bar.update(delta)
            self._completed_frames = event.completed_frames
        if event.completed_frames >= event.total_frames:
            self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _usage_error(parser: argparse.ArgumentParser, message: str) -> int:
    parser.print_usage(sys.stderr)
    print(f"{parser.prog}: error: {message}", file=sys.stderr)
    return USAGE_EXIT
