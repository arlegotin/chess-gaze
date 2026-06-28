# Task 1 Report: Python Viewer Output Contract

## Scope

Implemented only the Task 1 surface from `.superpowers/sdd/task-1-brief.md`:

- modified `src/chess_gaze/scene_viewer.py`
- modified `src/chess_gaze/viewer_assets/index.html`
- modified `tests/chess_gaze/test_scene_viewer.py`

Did not modify viewer artifact output directories.

## Root Cause and Durable Boundary

- `build_scene_viewer()` copied the packaged viewer assets, wrote
  `scene-data.json`, and then rewrote `viewer/index.html` into the embedded
  file-url bootstrap variant.
- That coupled the served entrypoint to the standalone/file-url contract, so the
  generated `index.html` always carried the full scene payload and inline module
  source.
- The durable fix splits the generated outputs at the builder boundary:
  - `index.html` stays the lightweight served entrypoint with import map plus
    normal module loading;
  - `standalone.html` becomes the file-url-compatible embedded entrypoint;
  - `scene-data.json` remains the shared structured payload.

## RED/GREEN TDD Evidence

### RED: split-contract tests fail before implementation

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py::test_build_scene_viewer_writes_server_and_standalone_indexes tests/chess_gaze/test_scene_viewer.py::test_generated_index_fetches_scene_data_without_embedding_payload tests/chess_gaze/test_scene_viewer.py::test_generated_standalone_embeds_file_url_bootstrap_and_scene_data -q
```

Observed output:

```text
FFF
FAILED test_build_scene_viewer_writes_server_and_standalone_indexes
FAILED test_generated_index_fetches_scene_data_without_embedding_payload
FAILED test_generated_standalone_embeds_file_url_bootstrap_and_scene_data
```

Failure reasons matched the task brief:

- `standalone.html` was missing;
- generated `index.html` still contained the embedded file-url payload.

### GREEN: full viewer test file passes after implementation

Command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

Observed inside the default sandbox:

```text
23 passed, 2 failed
```

The two failures were `test_static_server_serves_viewer_files` and
`test_static_server_does_not_escape_viewer_root`, both caused by sandbox socket
binding denial:

```text
PermissionError: [Errno 1] Operation not permitted
```

Rerun with escalation for loopback socket binding:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

Observed output:

```text
25 passed in 1.87s
```

## Implementation Notes

- Added a served-index write step that injects the pinned import map into the
  copied template while preserving the external module script.
- Renamed the embedded writer boundary from index generation to standalone
  generation and now write the embedded bootstrap to `viewer/standalone.html`.
- Kept the file-url bootstrap scene-data semantics unchanged.
- Updated viewer tests to cover:
  - presence of `index.html`, `standalone.html`, and `scene-data.json`;
  - absence of embedded payload in served `index.html`;
  - presence of embedded bootstrap in `standalone.html`;
  - import map and remote-module constraints across both HTML outputs.

## Verification

Fresh verification command:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

Verified result:

```text
25 passed in 1.87s
```

---

## 2026-06-28 Review-Fix Addendum: restore `viewer/index.html` file contract

### Review finding addressed

Task 1 commit `f5c1b9a` inverted the public viewer entrypoint contract:

- the CLI and artifact manifests still expose `viewer/index.html`;
- the user-reported workflow was direct `file:///.../viewer/index.html`;
- but the implementation had made `index.html` served-only and moved the
  file-url bootstrap to `viewer/standalone.html`.

### Root cause and durable surface changed

- The regression lived at the scene-viewer build boundary in
  `src/chess_gaze/scene_viewer.py`.
- `build_scene_viewer()` rendered the lightweight import-map page into
  `viewer/index.html` and wrote the embedded bootstrap variant into
  `viewer/standalone.html`.
- That broke the existing artifact contract without changing the public CLI or
  result metadata.

Durable fix:

- restore `viewer/index.html` as the embedded file-url-compatible entrypoint;
- generate `viewer/served.html` as the lightweight served entrypoint;
- keep `scene-data.json` and artifact directories unchanged;
- make the localhost server prefer `served.html` for `/` while still serving the
  embedded `index.html` at `/index.html`.

### RED/GREEN evidence

#### RED

Command:

```sh
uv run pytest tests/chess_gaze/test_scene_viewer.py -q -k 'embedded_index_and_served_entrypoint or generated_index_embeds_file_url_bootstrap_and_scene_data or generated_served_html_fetches_scene_data_without_embedding_payload or static_server_serves_viewer_files'
```

Observed output:

```text
FFFF
FAILED test_build_scene_viewer_writes_embedded_index_and_served_entrypoint
FAILED test_generated_index_embeds_file_url_bootstrap_and_scene_data
FAILED test_generated_served_html_fetches_scene_data_without_embedding_payload
FAILED test_static_server_serves_viewer_files
```

Failure reasons matched the review finding:

- `served.html` was not generated;
- `index.html` still had the served/module-script contract instead of embedded
  scene data and bootstrap;
- server root `/` still served `index.html` instead of preferring `served.html`.

#### GREEN

Focused rerun after implementation:

```sh
uv run pytest tests/chess_gaze/test_scene_viewer.py -q -k 'embedded_index_and_served_entrypoint or generated_index_embeds_file_url_bootstrap_and_scene_data or generated_served_html_fetches_scene_data_without_embedding_payload or static_server_serves_viewer_files'
```

Observed output:

```text
4 passed, 21 deselected in 1.49s
```

Full required verification:

```sh
uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

Observed output:

```text
25 passed in 1.40s
```

### Files changed

- `src/chess_gaze/scene_viewer.py`
- `tests/chess_gaze/test_scene_viewer.py`

### Implementation summary

- Replaced the `standalone.html` output contract with `served.html`.
- Rendered both HTML outputs from the same copied template:
  - `served.html` receives the import map and keeps the normal module script;
  - `index.html` receives the same import map plus the embedded
    `scene-data-json`, `scene-viewer-source`, and file-url bootstrap.
- Updated the locked static server so `/` serves `served.html` when present and
  `/index.html` still serves the embedded artifact.
- Updated tests to assert the corrected generated files, payload placement, and
  served-root behavior.
