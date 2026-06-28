from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Mapping
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
from chess_gaze.viewer_dependencies import three_import_map

_INDEX_IMPORT_MAP_MARKER = "    <!-- CHESS_GAZE_IMPORT_MAP -->"
_INDEX_MODULE_TAG = '    <script type="module" src="./scene_viewer.js"></script>'


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
    _remove_stale_local_vendor_assets(viewer_dir)
    copy_viewer_assets(viewer_dir)
    _write_scene_summary(
        updated_scene_result.paths.scene_summary_path,
        updated_scene_result.summary,
    )
    scene_data_path = write_viewer_scene_data(
        viewer_dir,
        updated_scene_result.viewer_data,
    )
    _write_served_index(viewer_dir)
    _write_file_url_compatible_standalone(viewer_dir, updated_scene_result.viewer_data)
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


def _write_served_index(viewer_dir: Path) -> None:
    index_path = viewer_dir / "index.html"
    html = index_path.read_text(encoding="utf-8")
    if _INDEX_IMPORT_MAP_MARKER not in html:
        raise ViewerServerError("viewer index template is missing import map marker")

    html = html.replace(_INDEX_IMPORT_MAP_MARKER, _import_map_script(three_import_map()))
    atomic_write_bytes(index_path, html.encode("utf-8"))


def _write_file_url_compatible_standalone(
    viewer_dir: Path,
    data: ViewerSceneData,
) -> None:
    standalone_path = viewer_dir / "standalone.html"
    html = (viewer_dir / "index.html").read_text(encoding="utf-8")
    if _INDEX_MODULE_TAG not in html:
        raise ViewerServerError("viewer index template is missing module script tag")

    scene_data_json = json.dumps(
        data.model_dump(mode="json", by_alias=True),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    embedded_bootstrap = "\n".join(
        (
            '    <script type="application/json" id="scene-data-json">',
            _safe_script_payload(scene_data_json),
            "    </script>",
            _module_source_script(
                "scene-viewer-source",
                (viewer_dir / "scene_viewer.js").read_text(encoding="utf-8"),
            ),
            "    <script>",
            _file_url_bootstrap_script(),
            "    </script>",
        )
    )
    html = html.replace(_INDEX_MODULE_TAG, embedded_bootstrap)
    atomic_write_bytes(standalone_path, html.encode("utf-8"))


def _module_source_script(script_id: str, source: str) -> str:
    return "\n".join(
        (
            f'    <script type="application/json" id="{script_id}">',
            _safe_script_payload(json.dumps(source, ensure_ascii=True)),
            "    </script>",
        )
    )


def _import_map_script(import_map: Mapping[str, object]) -> str:
    return "\n".join(
        (
            '    <script type="importmap">',
            _safe_script_payload(
                json.dumps(import_map, ensure_ascii=True, separators=(",", ":"))
            ),
            "    </script>",
        )
    )


def _safe_script_payload(payload: str) -> str:
    return payload.replace("</", "<\\/")


def _file_url_bootstrap_script() -> str:
    return r"""(() => {
      const sourceText = (id) => JSON.parse(document.getElementById(id).textContent);
      const moduleUrl = (source) =>
        URL.createObjectURL(new Blob([source], { type: "text/javascript" }));

      window.__CHESS_GAZE_SCENE_DATA__ = JSON.parse(
        document.getElementById("scene-data-json").textContent,
      );

      const sceneViewerUrl = moduleUrl(sourceText("scene-viewer-source"));

      import(sceneViewerUrl).catch((error) => {
        const message = `Scene viewer unavailable: ${error.message}`;
        const frameStatus = document.querySelector('[data-testid="frame-status"]');
        const fallbackStatus = document.querySelector(".fallback-status");
        if (frameStatus) {
          frameStatus.textContent = message;
        }
        if (fallbackStatus) {
          fallbackStatus.textContent = message;
          fallbackStatus.dataset.state = "error";
        }
        console.error(error);
      });
    })();"""


def _remove_stale_local_vendor_assets(viewer_dir: Path) -> None:
    vendor_dir = viewer_dir / "vendor"
    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)


def serve_viewer(run_dir: Path, host: str = "127.0.0.1", port: int = 0) -> ViewerServer:
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
        raise ViewerServerError(f"Viewer host must be a loopback address, not {host!r}")
    return normalized


class _ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _LockedViewerRequestHandler(SimpleHTTPRequestHandler):
    _viewer_root: Path

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(self._viewer_root), **kwargs)  # type: ignore[arg-type]

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
