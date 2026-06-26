from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

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
from chess_gaze.scene_records import ViewerSceneData
from chess_gaze.scene_viewer import build_scene_viewer


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


def test_build_scene_viewer_copies_local_vendor_assets(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer

    for relative_path in (
        "vendor/three.module.js",
        "vendor/three.core.js",
        "vendor/OrbitControls.js",
        "vendor/THREE_LICENSE.txt",
        "vendor/vendor_manifest.json",
    ):
        assert (layout.viewer_dir / relative_path).is_file()


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


def test_generated_html_js_and_css_reference_only_local_assets(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    layout, _viewer_data = built_viewer

    for relative_path in ("index.html", "scene_viewer.js", "styles.css"):
        text = (layout.viewer_dir / relative_path).read_text(encoding="utf-8")
        assert "http://" not in text
        assert "https://" not in text
        assert "cdn" not in text.lower()
        assert "telemetry" not in text.lower()

    html = (layout.viewer_dir / "index.html").read_text(encoding="utf-8")
    js = (layout.viewer_dir / "scene_viewer.js").read_text(encoding="utf-8")
    assert 'href="./styles.css"' in html
    assert 'src="./scene_viewer.js"' in html
    assert 'from "./vendor/three.module.js"' in js
    assert 'from "./vendor/OrbitControls.js"' in js


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


def test_copied_vendor_modules_are_esm_importable_when_node_is_available(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; skipping copied vendored ESM import check.")

    layout, _viewer_data = built_viewer
    three_module_url = (layout.viewer_dir / "vendor" / "three.module.js").as_uri()
    orbit_controls_url = (layout.viewer_dir / "vendor" / "OrbitControls.js").as_uri()
    import_script = """
const modules = process.argv.slice(1);
const results = await Promise.allSettled(modules.map((moduleUrl) => import(moduleUrl)));
const failures = results
  .map((result, index) => ({ result, moduleUrl: modules[index] }))
  .filter(({ result }) => result.status === "rejected");
if (failures.length > 0) {
  for (const { result, moduleUrl } of failures) {
    console.error(`${moduleUrl}: ${result.reason.message}`);
  }
  process.exit(1);
}
"""

    result = subprocess.run(
        [
            node,
            "--experimental-default-type=module",
            "--input-type=module",
            "-e",
            import_script,
            three_module_url,
            orbit_controls_url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_build_scene_viewer_updates_viewer_exists_summary(
    built_viewer: tuple[RunLayout, ViewerSceneData],
) -> None:
    _layout, viewer_data = built_viewer

    assert viewer_data.summary.artifact_validation.viewer_exists is True


def test_build_scene_viewer_writes_scene_result_viewer_data(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")
    scene_result = build_scene_artifacts(layout)
    scene_result = replace(
        scene_result,
        viewer_data=scene_result.viewer_data.model_copy(update={"run_id": "changed"}),
    )

    result = build_scene_viewer(layout, scene_result)

    viewer_data = ViewerSceneData.model_validate_json(
        result.scene_data_path.read_text(encoding="utf-8")
    )
    assert viewer_data.run_id == "changed"
