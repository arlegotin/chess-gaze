from __future__ import annotations

import os
import sys
import threading
from types import TracebackType
from typing import BinaryIO, TextIO

_CLEARCUT_SOURCE_TRACE = (
    "wireless/android/play/playlog/cplusplus/portable_clearcut_uploader.cc:180"
)
_KNOWN_MEDIAPIPE_STARTUP_LINES = (
    ("I", " init-domain.cc:", "Fiber init: default domain = pthread"),
    (
        "W",
        " face_landmarker_graph.cc:",
        "Sets FaceBlendshapesGraph acceleration to xnnpack by default.",
    ),
    ("I", " gl_context.cc:", "GL version:"),
    (
        "W",
        " inference_feedback_manager.cc:",
        "Feedback manager requires a model with a single signature inference. "
        "Disabling support for feedback tensors.",
    ),
)


class _NativeStderrLineFilter:
    def __init__(self) -> None:
        self._clearcut_trace_lines_remaining = 0

    def should_suppress(self, line: str) -> bool:
        stripped = line.strip()
        if self._clearcut_trace_lines_remaining > 0:
            if (
                stripped == "=== Source Location Trace: ==="
                or stripped == _CLEARCUT_SOURCE_TRACE
            ):
                self._clearcut_trace_lines_remaining -= 1
                return True
            self._clearcut_trace_lines_remaining = 0

        if _is_duplicate_avfoundation_class_warning(stripped):
            return True
        if _is_mediapipe_startup_line(stripped):
            return True
        if _is_clearcut_upload_failure(stripped):
            self._clearcut_trace_lines_remaining = 2
            return True
        return False


class _NativeAnalysisLogFilter:
    def __init__(self) -> None:
        self.stderr: TextIO = sys.stderr
        self._saved_stderr_fd: int | None = None
        self._read_fd: int | None = None
        self._passthrough_fd: int | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _NativeAnalysisLogFilter:
        sys.stderr.flush()
        self._saved_stderr_fd = os.dup(2)
        self._passthrough_fd = os.dup(self._saved_stderr_fd)
        self.stderr = os.fdopen(os.dup(self._saved_stderr_fd), "w", buffering=1)
        read_fd, write_fd = os.pipe()
        self._read_fd = read_fd
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self._thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        sys.stderr.flush()
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)
            self._saved_stderr_fd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.stderr.close()

    def _pump_stderr(self) -> None:
        if self._read_fd is None:
            return
        if self._passthrough_fd is None:
            return

        line_filter = _NativeStderrLineFilter()
        with (
            os.fdopen(self._read_fd, "rb", closefd=True) as source,
            os.fdopen(self._passthrough_fd, "wb", closefd=True) as target,
        ):
            self._read_fd = None
            self._passthrough_fd = None
            _forward_filtered_stderr(source, target, line_filter)


def suppress_known_native_analysis_logs() -> _NativeAnalysisLogFilter:
    return _NativeAnalysisLogFilter()


def _forward_filtered_stderr(
    source: BinaryIO,
    target: BinaryIO,
    line_filter: _NativeStderrLineFilter,
) -> None:
    pending = b""
    while True:
        chunk = source.read(4096)
        if not chunk:
            break
        pending += chunk
        while b"\n" in pending:
            raw_line, pending = pending.split(b"\n", 1)
            _forward_line(raw_line + b"\n", target, line_filter)
    if pending:
        _forward_line(pending, target, line_filter)


def _forward_line(
    raw_line: bytes,
    target: BinaryIO,
    line_filter: _NativeStderrLineFilter,
) -> None:
    text = raw_line.decode("utf-8", errors="replace").rstrip("\n")
    if line_filter.should_suppress(text):
        return
    target.write(raw_line)
    target.flush()


def _is_duplicate_avfoundation_class_warning(line: str) -> bool:
    return (
        line.startswith("objc[")
        and " is implemented in both " in line
        and ("Class AVFFrameReceiver" in line or "Class AVFAudioReceiver" in line)
    )


def _is_mediapipe_startup_line(line: str) -> bool:
    if line == "INFO: Created TensorFlow Lite XNNPACK delegate for CPU.":
        return True

    return any(
        line.startswith(level) and source in line and message in line
        for level, source, message in _KNOWN_MEDIAPIPE_STARTUP_LINES
    )


def _is_clearcut_upload_failure(line: str) -> bool:
    return (
        " portable_clearcut_uploader.cc:" in line
        and "Failed to send to clearcut:" in line
    )
