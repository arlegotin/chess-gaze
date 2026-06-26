# Task 9 Report: Localhost Viewer Command And Static Server

## Scope

Implemented the localhost viewer command and static server slice for the 3D
scene artifact viewer.

Changed files:

- `src/chess_gaze/cli.py`
- `src/chess_gaze/scene_viewer.py`
- `tests/chess_gaze/test_cli.py`
- `tests/chess_gaze/test_scene_viewer.py`
- `.superpowers/sdd/task-9-report-3d-scene.md`

## RED Evidence

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py tests/chess_gaze/test_scene_viewer.py -q
```

Result before implementation:

```text
4 failed, 18 passed in 1.34s
```

Expected failures:

- `test_analyze_prints_run_dir_and_viewer_path` failed because `analyze` printed
  only the run directory.
- `test_view_prints_localhost_url_for_run_viewer` failed because the `view`
  subcommand was not registered.
- `test_static_server_serves_viewer_files` failed because `serve_viewer()` was
  not implemented.
- `test_static_server_does_not_escape_viewer_root` failed because
  `serve_viewer()` was not implemented.

## Server Design

- `serve_viewer(run_dir, host="127.0.0.1", port=0)` validates that `run_dir`
  exists and contains `viewer/index.html` and `viewer/scene-data.json`.
- The HTTP server uses only Python standard library `http.server` primitives.
- The default bind host is `127.0.0.1`.
- The server root is locked to `run_dir/viewer` by resolving every translated
  request path and refusing paths outside the resolved viewer root.
- `ViewerServer.url` exposes the local URL, including the OS-selected port when
  `port=0`.
- `ViewerServer.close()` calls `shutdown()`, `server_close()`, signals waiting
  callers, and joins the daemon server thread with a timeout.
- CLI `chess-gaze view <run-dir>` prints one URL, flushes it, and blocks until
  interrupted. It closes the server in `finally`.

## GREEN Evidence

The sandbox denied localhost socket binding without escalation:

```text
PermissionError: [Errno 1] Operation not permitted
```

Rerun with localhost binding permitted:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py tests/chess_gaze/test_scene_viewer.py -q
```

Result:

```text
22 passed in 1.20s
```

Focused Ruff:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check src/chess_gaze/cli.py src/chess_gaze/scene_viewer.py tests/chess_gaze/test_cli.py tests/chess_gaze/test_scene_viewer.py
```

Result:

```text
All checks passed!
```

## Residual Risks

- Browser smoke was intentionally not run in this slice. The main agent still
  needs to open a generated viewer URL and verify rendering, controls, and
  mobile/desktop layout behavior.
- Unit tests cover traversal attempts and root locking, but they do not exhaust
  every URL encoding variant.
- Real CLI serving was covered through the lower-level server and a monkeypatched
  CLI wait path to avoid hanging the test process.
