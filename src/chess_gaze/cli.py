from __future__ import annotations

import argparse
import sys
from pathlib import Path

INPUT_NOT_FOUND_EXIT = 10
USAGE_EXIT = 2


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
        return int(exc.code)

    if args.command != "analyze":
        parser.print_usage(sys.stderr)
        return USAGE_EXIT

    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"INPUT_NOT_FOUND: {video_path}", file=sys.stderr)
        return INPUT_NOT_FOUND_EXIT

    print("Pipeline implementation is not wired yet", file=sys.stderr)
    return 1
