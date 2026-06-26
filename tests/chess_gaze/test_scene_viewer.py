from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import chess_gaze.scene_viewer as scene_viewer
from chess_gaze.artifact_runs import RunLayout
from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    EyeRecord,
    FaceRecord,
    FrameRecord,
    GazeAngles,
    HeadPoseRecord,
    RunManifest,
    VideoManifest,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.scene_artifacts import build_scene_artifacts
from chess_gaze.scene_records import SceneSummary, ViewerSceneData
from chess_gaze.scene_viewer import (
    build_scene_viewer,
    write_viewer_scene_data,
)

THREE_MODULE_URL = "https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js"
THREE_CORE_URL = "https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.core.js"
THREE_ADDONS_URL = "https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/"
ORBIT_CONTROLS_URL = (
    "https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/controls/OrbitControls.js"
)
APPROVED_REMOTE_MODULE_URLS = {
    THREE_MODULE_URL,
    THREE_CORE_URL,
    THREE_ADDONS_URL,
    ORBIT_CONTROLS_URL,
}
EXPECTED_IMPORT_MAP = {
    "imports": {
        "three": THREE_MODULE_URL,
        "three/addons/": THREE_ADDONS_URL,
    }
}
EXTERNAL_URL_RE = re.compile(r"https?://[^\s\"'<>`]+")
PROTOCOL_RELATIVE_URL_RE = re.compile(
    r"(?<!:)//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s\"'<>`]*"
)


def _layout(run_dir: Path) -> RunLayout:
    return RunLayout(
        run_dir=run_dir,
        raw_frames_dir=run_dir / "raw_frames",
        processed_frames_dir=run_dir / "processed_frames",
        crops_dir=run_dir / "crops",
        face_crops_dir=run_dir / "crops" / "face",
        eyes_crops_dir=run_dir / "crops" / "eyes",
        left_eye_crops_dir=run_dir / "crops" / "eyes" / "left",
        right_eye_crops_dir=run_dir / "crops" / "eyes" / "right",
        records_dir=run_dir / "records",
    )


def _point(x: float, y: float) -> Point2D:
    return Point2D(space=CoordinateSpace.IMAGE_PX, x=x, y=y)


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> BBox:
    return BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def _eye(center_x: float, center_y: float) -> EyeRecord:
    return EyeRecord(
        present=True,
        bounding_box=_box(
            center_x - 8.0,
            center_y - 6.0,
            center_x + 8.0,
            center_y + 6.0,
        ),
        pupil_center=_point(center_x, center_y),
        iris_landmarks=[
            _point(center_x - 3.0, center_y),
            _point(center_x + 3.0, center_y),
            _point(center_x, center_y - 3.0),
            _point(center_x, center_y + 3.0),
        ],
        reason_invalid=None,
    )


def _gaze(
    *, valid: bool, yaw_radians: float | None, pitch_radians: float | None
) -> GazeAngles:
    return GazeAngles(
        valid=valid,
        yaw_radians=yaw_radians,
        pitch_radians=pitch_radians,
        reason_invalid=None if valid else ErrorCode.GAZE_MODEL_FAILED,
    )


def _frame(index: int, *, gaze_valid: bool = True) -> FrameRecord:
    eye_index = 2 if index in (2, 3) else index
    yaw = 0.0 if index in (2, 3) else (index - 2) / 500.0
    pitch = 0.0 if index in (2, 3) else -(index - 2) / 1000.0
    appearance_gaze = _gaze(
        valid=gaze_valid,
        yaw_radians=yaw if gaze_valid else None,
        pitch_radians=pitch if gaze_valid else None,
    )

    return FrameRecord(
        frame_id=f"f{index:09d}",
        frame_index=index,
        status=FrameStatus.OK if gaze_valid else FrameStatus.WARNING,
        timestamp_seconds=index / 30.0,
        face=FaceRecord(
            present=True,
            bounding_box=_box(700.0, 240.0, 1180.0, 900.0),
            landmarks=[
                _point(860.0, 430.0),
                _point(1060.0, 430.0),
                _point(960.0, 600.0),
            ],
            reason_invalid=None,
        ),
        left_eye=_eye(900.0 + eye_index, 540.0),
        right_eye=_eye(1020.0 + eye_index, 540.0),
        head_pose=HeadPoseRecord(
            valid=True,
            yaw_radians=0.0,
            pitch_radians=0.0,
            roll_radians=0.0,
            reason_invalid=None,
        ),
        geometric_gaze=_gaze(valid=True, yaw_radians=yaw, pitch_radians=pitch),
        appearance_gaze=appearance_gaze,
        recommended_gaze=_gaze(valid=True, yaw_radians=0.5, pitch_radians=0.25),
        errors=[],
    )


def _write_minimal_run(run_dir: Path) -> RunLayout:
    layout = _layout(run_dir)
    layout.records_dir.mkdir(parents=True)
    layout.scene_dir.mkdir(parents=True)
    layout.viewer_dir.mkdir(parents=True)

    video = VideoManifest(
        source_path="artifacts/input/synthetic_scene_source.mp4",
        source_sha256="a" * 64,
        frame_width=1920,
        frame_height=1080,
        frame_count_decoded=7,
    )
    run_manifest = RunManifest(
        run_id="20260626T120000Z-scene",
        created_at_utc=datetime(2026, 6, 26, 12, tzinfo=UTC).isoformat(),
        input_path=video.source_path,
        video=video,
    )

    (run_dir / "run_manifest.json").write_text(
        run_manifest.model_dump_json(), encoding="utf-8"
    )
    (run_dir / "video_manifest.json").write_text(
        video.model_dump_json(), encoding="utf-8"
    )
    (layout.records_dir / "frames.jsonl").write_text(
        "".join(
            _frame(index, gaze_valid=index != 6).model_dump_json() + "\n"
            for index in range(7)
        ),
        encoding="utf-8",
    )
    return layout


def _read_viewer_app_assets(viewer_dir: Path) -> dict[str, str]:
    return {
        relative_path: (viewer_dir / relative_path).read_text(encoding="utf-8")
        for relative_path in ("index.html", "scene_viewer.js", "styles.css")
    }


def _external_urls(text: str) -> set[str]:
    return set(EXTERNAL_URL_RE.findall(text))


def _assert_absent(text: str, needle: str, label: str) -> None:
    if needle in text:
        pytest.fail(f"{label} contains forbidden {needle!r}")


def _extract_import_map(html: str) -> dict[str, object]:
    for match in re.finditer(
        r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        if re.search(r"""\btype=["']importmap["']""", match.group("attrs")):
            return cast("dict[str, object]", json.loads(match.group("body")))
    raise AssertionError("generated viewer index is missing an import map")


def _resolve_import_map_specifier(import_map: dict[str, object], specifier: str) -> str:
    imports = import_map["imports"]
    assert isinstance(imports, dict)
    exact = imports.get(specifier)
    if isinstance(exact, str):
        return exact

    prefix_matches = [
        (prefix, target)
        for prefix, target in imports.items()
        if (
            isinstance(prefix, str)
            and isinstance(target, str)
            and prefix.endswith("/")
            and specifier.startswith(prefix)
        )
    ]
    if not prefix_matches:
        raise AssertionError(f"{specifier!r} is not resolvable by the import map")

    prefix, target = max(prefix_matches, key=lambda item: len(item[0]))
    return target + specifier[len(prefix) :]


@pytest.fixture
def built_viewer(tmp_path: Path) -> tuple[RunLayout, ViewerSceneData]:
    layout = _write_minimal_run(tmp_path / "run")
    scene_result = build_scene_artifacts(layout)

    result = build_scene_viewer(layout, scene_result)

    assert result.viewer_dir == layout.viewer_dir
    assert result.index_path == layout.viewer_dir / "index.html"
    assert result.scene_data_path == layout.viewer_dir / "scene-data.json"
    return layout, ViewerSceneData.model_validate_json(
        result.scene_data_path.read_text(encoding="utf-8")
    )


def test_build_scene_viewer_writes_index_and_scene_data(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, viewer_data = built_viewer

    assert (layout.viewer_dir / "index.html").is_file()
    assert (layout.viewer_dir / "scene-data.json").is_file()
    assert viewer_data.schema_version == "gaze-scene-viewer-data-v1"


def test_build_scene_viewer_writes_app_assets_without_local_vendor_modules(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer

    generated_files = {
        path.relative_to(layout.viewer_dir).as_posix()
        for path in layout.viewer_dir.rglob("*")
        if path.is_file()
    }

    assert {
        "index.html",
        "scene-data.json",
        "scene_viewer.js",
        "styles.css",
    } <= generated_files
    assert not (layout.viewer_dir / "vendor").exists()
    assert all(not path.startswith("vendor/") for path in generated_files)


def test_build_scene_viewer_removes_stale_local_vendor_assets(tmp_path: Path) -> None:
    layout = _write_minimal_run(tmp_path / "run")
    scene_result = build_scene_artifacts(layout)
    stale_vendor_dir = layout.viewer_dir / "vendor"
    stale_vendor_dir.mkdir(parents=True)
    (stale_vendor_dir / "three.module.js").write_text(
        "stale local dependency", encoding="utf-8"
    )

    build_scene_viewer(layout, scene_result)

    assert not stale_vendor_dir.exists()


def test_scene_data_is_strict_schema_versioned_viewer_scene_data(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, viewer_data = built_viewer
    payload = json.loads((layout.viewer_dir / "scene-data.json").read_text("utf-8"))

    assert payload["schema_version"] == "gaze-scene-viewer-data-v1"
    assert viewer_data.run_id == "20260626T120000Z-scene"

    with pytest.raises(ValidationError):
        ViewerSceneData.model_validate(payload | {"unknown_field": True})


def test_scene_data_includes_all_frames_in_slider_order(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    _layout, viewer_data = built_viewer

    assert viewer_data.frame_count == 7
    assert [frame.frame_index for frame in viewer_data.frames] == list(range(7))
    assert viewer_data.frames[-1].main_monitor_hit.valid is False


def test_scene_data_keeps_one_hit_identity_per_valid_monitor_hit_frame(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    _layout, viewer_data = built_viewer
    valid_frame_count = sum(
        1 for frame in viewer_data.frames if frame.main_monitor_hit.valid
    )

    assert len(viewer_data.valid_hit_points) == valid_frame_count
    assert [point.frame_index for point in viewer_data.valid_hit_points] == [
        frame.frame_index
        for frame in viewer_data.frames
        if frame.main_monitor_hit.valid
    ]

    duplicate_points = [
        point for point in viewer_data.valid_hit_points if point.frame_index in (2, 3)
    ]
    assert len(duplicate_points) == 2
    assert duplicate_points[0].u_m == duplicate_points[1].u_m
    assert duplicate_points[0].v_m == duplicate_points[1].v_m
    assert duplicate_points[0].frame_id != duplicate_points[1].frame_id


def test_generated_html_includes_required_selectors(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    html = (layout.viewer_dir / "index.html").read_text(encoding="utf-8")

    for selector in (
        'data-testid="scene-canvas"',
        'data-testid="frame-slider"',
        'data-testid="frame-number"',
        'data-testid="frame-label"',
        'data-testid="mode-instant"',
        'data-testid="mode-accumulated"',
        'data-testid="play-pause"',
        'data-testid="step-prev"',
        'data-testid="step-next"',
        'data-testid="toggle-head"',
        'data-testid="toggle-eyes"',
        'data-testid="toggle-ray"',
        'data-testid="toggle-monitor-plane"',
        'data-testid="toggle-monitor-rectangle"',
        'data-testid="toggle-extended-plane"',
        'data-testid="toggle-axes"',
        'data-testid="toggle-hit-points"',
        'data-testid="status-panel"',
    ):
        assert selector in html


def test_generated_html_js_and_css_reference_only_approved_remote_three_modules(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer

    app_assets = _read_viewer_app_assets(layout.viewer_dir)
    combined_assets = "\n".join(app_assets.values())

    _assert_absent(combined_assets, "http://", "generated viewer assets")
    protocol_relative_match = PROTOCOL_RELATIVE_URL_RE.search(combined_assets)
    assert protocol_relative_match is None, (
        "generated viewer assets contain protocol-relative URL "
        f"{protocol_relative_match.group(0)!r}"
    )
    _assert_absent(combined_assets.lower(), "latest", "generated viewer assets")
    _assert_absent(combined_assets.lower(), "telemetry", "generated viewer assets")
    _assert_absent(combined_assets, "./vendor/", "generated viewer assets")
    _assert_absent(combined_assets, "/vendor/", "generated viewer assets")
    unexpected_urls = _external_urls(combined_assets) - APPROVED_REMOTE_MODULE_URLS
    assert unexpected_urls == set()

    html = app_assets["index.html"]
    js = app_assets["scene_viewer.js"]
    css = app_assets["styles.css"]
    import_map = _extract_import_map(html)
    assert import_map == EXPECTED_IMPORT_MAP
    assert THREE_MODULE_URL in _external_urls(html)
    assert THREE_ADDONS_URL in _external_urls(html)
    assert _external_urls(js) == set()
    assert _external_urls(css) == set()

    load_references = re.findall(r"""(?:src|href)=["']([^"']+)["']""", html)
    assert load_references
    assert all(
        not reference.startswith(("http://", "https://", "//"))
        for reference in load_references
    )
    assert 'rel="icon" href="data:,"' in html
    assert 'href="./styles.css"' in html
    assert 'src="./scene_viewer.js"' not in html
    assert 'from "three"' in js
    assert 'from "three/addons/controls/OrbitControls.js"' in js
    assert (
        _resolve_import_map_specifier(
            import_map, "three/addons/controls/OrbitControls.js"
        )
        == ORBIT_CONTROLS_URL
    )


def test_generated_index_embeds_file_url_bootstrap_and_scene_data(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, viewer_data = built_viewer
    html = (layout.viewer_dir / "index.html").read_text(encoding="utf-8")

    assert 'type="module" src="./scene_viewer.js"' not in html
    assert 'id="scene-data-json"' in html
    _assert_absent(html, 'id="three-core-source"', "generated viewer index")
    _assert_absent(html, 'id="three-module-source"', "generated viewer index")
    _assert_absent(html, 'id="orbit-controls-source"', "generated viewer index")
    assert 'id="scene-viewer-source"' in html
    assert "window.__CHESS_GAZE_SCENE_DATA__" in html
    assert viewer_data.run_id in html


def test_generated_js_prefers_embedded_scene_data_for_file_url_viewer(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    js = (layout.viewer_dir / "scene_viewer.js").read_text(encoding="utf-8")

    assert "window.__CHESS_GAZE_SCENE_DATA__" in js
    assert 'fetch("./scene-data.json"' in js


def test_generated_css_uses_light_theme_and_semantic_color_roles(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    css = (layout.viewer_dir / "styles.css").read_text(encoding="utf-8")

    assert "color-scheme: light" in css
    for role in (
        "--color-background:",
        "--color-surface:",
        "--color-text:",
        "--color-head:",
        "--color-left-eye:",
        "--color-right-eye:",
        "--color-unigaze-ray:",
        "--color-current-hit:",
        "--color-accumulated-hit:",
        "--color-monitor-plane:",
        "--color-warning:",
    ):
        assert role in css


def test_generated_js_contains_mode_names(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    js = (layout.viewer_dir / "scene_viewer.js").read_text(encoding="utf-8")

    assert "Instant" in js
    assert "Accumulated" in js


def test_generated_index_import_map_resolves_pinned_three_modules(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer
    html = (layout.viewer_dir / "index.html").read_text(encoding="utf-8")

    import_map = _extract_import_map(html)

    assert import_map == EXPECTED_IMPORT_MAP
    assert _resolve_import_map_specifier(import_map, "three") == THREE_MODULE_URL
    assert (
        _resolve_import_map_specifier(
            import_map, "three/addons/controls/OrbitControls.js"
        )
        == ORBIT_CONTROLS_URL
    )


def test_build_scene_viewer_updates_viewer_exists_summary(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    _layout, viewer_data = built_viewer

    assert viewer_data.summary.artifact_validation.viewer_exists is True


def test_build_scene_viewer_updates_persisted_scene_summary(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")
    scene_result = build_scene_artifacts(layout)
    before_summary = SceneSummary.model_validate_json(
        scene_result.paths.scene_summary_path.read_text(encoding="utf-8")
    )
    assert before_summary.artifact_validation.viewer_exists is False

    result = build_scene_viewer(layout, scene_result)

    after_summary = SceneSummary.model_validate_json(
        scene_result.paths.scene_summary_path.read_text(encoding="utf-8")
    )
    viewer_data = ViewerSceneData.model_validate_json(
        result.scene_data_path.read_text(encoding="utf-8")
    )
    assert after_summary.artifact_validation.viewer_exists is True
    assert viewer_data.summary.artifact_validation.viewer_exists is True
    assert after_summary.artifact_validation.viewer_exists == (
        viewer_data.summary.artifact_validation.viewer_exists
    )


def test_write_viewer_scene_data_writes_supplied_viewer_data(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")
    scene_result = build_scene_artifacts(layout)
    viewer_data = scene_result.viewer_data.model_copy(update={"run_id": "changed"})

    path = write_viewer_scene_data(layout.viewer_dir, viewer_data)

    written_viewer_data = ViewerSceneData.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    assert written_viewer_data.run_id == "changed"


def test_static_server_serves_viewer_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    viewer_dir = run_dir / "viewer"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "index.html").write_text("<!doctype html>viewer", encoding="utf-8")
    (viewer_dir / "scene-data.json").write_text('{"ok": true}', encoding="utf-8")

    server = scene_viewer.serve_viewer(run_dir)
    try:
        assert server.url.startswith("http://127.0.0.1:")

        with urllib.request.urlopen(server.url, timeout=2) as response:
            assert response.status == 200
            assert response.read().decode("utf-8") == "<!doctype html>viewer"

        scene_data_url = f"{server.url}scene-data.json"
        with urllib.request.urlopen(scene_data_url, timeout=2) as response:
            assert response.status == 200
            assert json.loads(response.read().decode("utf-8")) == {"ok": True}
    finally:
        server.close()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.0.25", "example.com"])
def test_static_server_rejects_non_loopback_hosts(tmp_path: Path, host: str) -> None:
    run_dir = tmp_path / "run"
    viewer_dir = run_dir / "viewer"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "index.html").write_text("<!doctype html>viewer", encoding="utf-8")
    (viewer_dir / "scene-data.json").write_text("{}", encoding="utf-8")

    with pytest.raises(scene_viewer.ViewerServerError, match="loopback"):
        scene_viewer.serve_viewer(run_dir, host=host)


def test_static_server_does_not_escape_viewer_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    viewer_dir = run_dir / "viewer"
    viewer_dir.mkdir(parents=True)
    (run_dir / "secret.txt").write_text("outside viewer", encoding="utf-8")
    (viewer_dir / "index.html").write_text("<!doctype html>viewer", encoding="utf-8")
    (viewer_dir / "scene-data.json").write_text("{}", encoding="utf-8")

    server = scene_viewer.serve_viewer(run_dir)
    try:
        for traversal_path in ("../secret.txt", "%2e%2e/secret.txt"):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{server.url}{traversal_path}", timeout=2)
            assert exc_info.value.code in {403, 404}
    finally:
        server.close()
