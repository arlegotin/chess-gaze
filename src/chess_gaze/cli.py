from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chess_gaze.errors import CliErrorCode
from chess_gaze.pipeline import AnalyzeRequest, PipelineError, analyze_video

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
    analyze.add_argument("--output-root", default="artifacts/output")
    analyze.add_argument("--models-root", default="models")
    analyze.add_argument("--config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return system_exit_code(exc)

    if args.command != "analyze":
        parser.print_usage(sys.stderr)
        return USAGE_EXIT

    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"INPUT_NOT_FOUND: {video_path}", file=sys.stderr)
        return INPUT_NOT_FOUND_EXIT

    try:
        result = analyze_video(
            AnalyzeRequest(
                video_path=video_path,
                output_root=Path(args.output_root),
                models_root=Path(args.models_root),
                config_path=Path(args.config) if args.config is not None else None,
            )
        )
    except PipelineError as exc:
        print(f"{exc.code.value}: {exc}", file=sys.stderr)
        return ERROR_EXIT_CODES.get(exc.code, GENERAL_FAILURE_EXIT)

    print(result.layout.run_dir)
    return 0
