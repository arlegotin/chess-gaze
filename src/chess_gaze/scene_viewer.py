from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from importlib.resources.abc import Traversable
from ipaddress import ip_address
from pathlib import Path
from types import TracebackType

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.image_io import atomic_write_bytes
from chess_gaze.scene_artifacts import (
    SceneArtifactResult,
    scene_result_with_viewer_exists,
)
from chess_gaze.scene_records import SceneSummary, ViewerSceneData


@dataclass(frozen=True)
class ViewerBuildResult:
    viewer_dir: Path
    index_path: Path
    scene_data_path: Path


class ViewerServerError(ValueError):
    pass


@dataclass
class ViewerServer:
    url: str
    _httpd: ThreadingHTTPServer
    _thread: threading.Thread
    _closed: threading.Event

    def serve_forever(self) -> None:
        self._closed.wait()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._closed.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)

    def __enter__(self) -> ViewerServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()


def build_scene_viewer(
    run_layout: RunLayout, scene_result: SceneArtifactResult
) -> ViewerBuildResult:
    viewer_dir = run_layout.viewer_dir
    updated_scene_result = scene_result_with_viewer_exists(
        scene_result,
        viewer_exists=True,
    )
    copy_viewer_assets(viewer_dir)
    _write_scene_summary(
        updated_scene_result.paths.scene_summary_path,
        updated_scene_result.summary,
    )
    scene_data_path = write_viewer_scene_data(
        viewer_dir,
        updated_scene_result.viewer_data,
    )
    return ViewerBuildResult(
        viewer_dir=viewer_dir,
        index_path=viewer_dir / "index.html",
        scene_data_path=scene_data_path,
    )


def copy_viewer_assets(viewer_dir: Path) -> None:
    viewer_assets = resources.files("chess_gaze").joinpath("viewer_assets")
    _copy_resource_tree(viewer_assets, viewer_dir)


def write_viewer_scene_data(viewer_dir: Path, data: ViewerSceneData) -> Path:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    path = viewer_dir / "scene-data.json"
    _write_json(path, data.model_dump(mode="json", by_alias=True))
    return path


def serve_viewer(
    run_dir: Path, host: str = "127.0.0.1", port: int = 0
) -> ViewerServer:
    bind_host = _validate_loopback_host(host)
    viewer_root = _validate_viewer_run(run_dir)
    handler = _viewer_request_handler(viewer_root)
    httpd = _ViewerHTTPServer((bind_host, port), handler)
    actual_host, actual_port = httpd.server_address[:2]
    closed = threading.Event()
    started = threading.Event()
    thread = threading.Thread(
        target=_serve_httpd,
        args=(httpd, started),
        name="chess-gaze-viewer",
        daemon=True,
    )
    thread.start()
    started.wait(timeout=2)
    return ViewerServer(
        url=f"http://{_format_url_host(str(actual_host))}:{actual_port}/",
        _httpd=httpd,
        _thread=thread,
        _closed=closed,
    )


def _write_scene_summary(path: Path, summary: SceneSummary) -> None:
    _write_json(path, summary.model_dump(mode="json", by_alias=True))


def _write_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(path, encoded)


def _copy_resource_tree(source: Traversable, destination: Path) -> None:
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_resource_tree(child, destination / child.name)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, source.read_bytes())


def _validate_viewer_run(run_dir: Path) -> Path:
    if not run_dir.is_dir():
        raise ViewerServerError(f"Run directory does not exist: {run_dir}")

    viewer_root = run_dir / "viewer"
    if not viewer_root.is_dir():
        raise ViewerServerError(f"Viewer directory does not exist: {viewer_root}")

    index_path = viewer_root / "index.html"
    if not index_path.is_file():
        raise ViewerServerError(f"Viewer index is missing: {index_path}")

    scene_data_path = viewer_root / "scene-data.json"
    if not scene_data_path.is_file():
        raise ViewerServerError(f"Viewer scene data is missing: {scene_data_path}")

    return viewer_root.resolve()


def _validate_loopback_host(host: str) -> str:
    normalized = host.strip()
    if normalized == "localhost":
        return normalized

    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]

    try:
        address = ip_address(normalized)
    except ValueError as exc:
        raise ViewerServerError(
            f"Viewer host must be a loopback address, not {host!r}"
        ) from exc

    if not address.is_loopback:
        raise ViewerServerError(
            f"Viewer host must be a loopback address, not {host!r}"
        )
    return normalized


class _ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _LockedViewerRequestHandler(SimpleHTTPRequestHandler):
    _viewer_root: Path

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(self._viewer_root), **kwargs)

    def translate_path(self, path: str) -> str:
        candidate = Path(super().translate_path(path)).resolve()
        if _is_relative_to(candidate, self._viewer_root):
            return str(candidate)
        return str(self._viewer_root / ".not-found")

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _viewer_request_handler(
    viewer_root: Path,
) -> type[_LockedViewerRequestHandler]:
    class ViewerRequestHandler(_LockedViewerRequestHandler):
        _viewer_root = viewer_root

    return ViewerRequestHandler


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _format_url_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _serve_httpd(httpd: ThreadingHTTPServer, started: threading.Event) -> None:
    started.set()
    httpd.serve_forever(poll_interval=0.1)
