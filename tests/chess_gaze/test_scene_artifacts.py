from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    ErrorRecord,
    EyeRecord,
    FaceRecord,
    FrameRecord,
    GazeAngles,
    HeadPoseRecord,
    InferenceRuntimeRecord,
    RunManifest,
    VideoManifest,
    read_run_manifest_artifact_json,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.scene_artifacts import (
    build_scene_artifacts,
    build_viewer_scene_data,
    load_scene_frames,
)
from chess_gaze.scene_records import (
    CoordinateFrame3D,
    SceneAssumptionRecord,
    SceneManifest,
    SceneSummary,
)

THREE_VERSION = "0.185.0"
THREE_MODULE_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js"
)
THREE_CORE_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.core.js"
)
THREE_ADDONS_URL = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/"
ORBIT_CONTROLS_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/controls/"
    "OrbitControls.js"
)
EXPECTED_MODULE_URLS = {
    "three": THREE_MODULE_URL,
    "three/core": THREE_CORE_URL,
    "three/addons/": THREE_ADDONS_URL,
    "three/addons/controls/OrbitControls.js": ORBIT_CONTROLS_URL,
}


def _layout(run_dir: Path) -> RunLayout:
    records_dir = run_dir / "records"
    return RunLayout(
        run_dir=run_dir,
        raw_frames_dir=run_dir / "raw_frames",
        processed_frames_dir=run_dir / "processed_frames",
        crops_dir=run_dir / "crops",
        face_crops_dir=run_dir / "crops" / "face",
        eyes_crops_dir=run_dir / "crops" / "eyes",
        left_eye_crops_dir=run_dir / "crops" / "eyes" / "left",
        right_eye_crops_dir=run_dir / "crops" / "eyes" / "right",
        records_dir=records_dir,
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
    recommended_gaze = _gaze(valid=True, yaw_radians=0.5, pitch_radians=0.25)
    geometric_gaze = _gaze(valid=True, yaw_radians=yaw, pitch_radians=pitch)
    errors: list[ErrorRecord] = []
    if not gaze_valid:
        errors.append(
            ErrorRecord(
                code=ErrorCode.GAZE_MODEL_FAILED,
                message="Synthetic appearance gaze intentionally unavailable.",
            )
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
        left_eye=_eye(1020.0 + eye_index, 540.0),
        right_eye=_eye(900.0 + eye_index, 540.0),
        head_pose=HeadPoseRecord(
            valid=True,
            yaw_radians=0.0,
            pitch_radians=0.0,
            roll_radians=0.0,
            reason_invalid=None,
        ),
        geometric_gaze=geometric_gaze,
        appearance_gaze=appearance_gaze,
        recommended_gaze=recommended_gaze,
        errors=errors,
    )


def _external_observer_inference_record() -> InferenceRuntimeRecord:
    return InferenceRuntimeRecord(
        observer_source="external_observer",
        unigaze_model_id=None,
        unigaze_device="not_applicable",
        unigaze_batch_size=None,
        torch_version=None,
        torch_mps_available=None,
        mps_fallback_env="not_applicable",
        mps_fast_math_env="not_applicable",
        mps_prefer_metal_env="not_applicable",
        mps_preflight_passed=None,
    )


def _write_minimal_run(run_dir: Path) -> RunLayout:
    layout = _layout(run_dir)
    layout.records_dir.mkdir(parents=True)

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
        inference=_external_observer_inference_record(),
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


def test_build_scene_artifacts_reads_legacy_run_manifest_without_inference(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")
    run_manifest_path = layout.run_dir / "run_manifest.json"
    legacy_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    legacy_manifest.pop("inference")
    run_manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")

    result = build_scene_artifacts(layout)
    manifest = read_run_manifest_artifact_json(
        run_manifest_path.read_text(encoding="utf-8")
    )

    assert result.manifest.run_id == "20260626T120000Z-scene"
    assert result.scene_frame_count == 7
    assert manifest.inference.observer_source == "legacy_manifest_without_inference"


def test_build_scene_artifacts_writes_strict_manifest_summary_and_frames(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")

    result = build_scene_artifacts(layout)

    assert result.paths.scene_manifest_path == (
        layout.run_dir / "scene" / "scene_manifest.json"
    )
    assert result.paths.scene_summary_path == (
        layout.run_dir / "scene" / "scene_summary.json"
    )
    assert result.paths.scene_frames_jsonl_path == (
        layout.records_dir / "scene_frames.jsonl"
    )
    assert result.paths.scene_manifest_path.is_file()
    assert result.paths.scene_summary_path.is_file()
    assert result.paths.scene_frames_jsonl_path.is_file()

    manifest = SceneManifest.model_validate_json(
        result.paths.scene_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.run_id == "20260626T120000Z-scene"
    assert manifest.source_video_path == "artifacts/input/synthetic_scene_source.mp4"
    assert manifest.source_video_sha256 == "a" * 64
    assert manifest.camera_model.frame_width_px == 1920
    assert manifest.camera_model.fx_px == 1920.0
    assert manifest.source_artifacts.frame_records == "records/frames.jsonl"
    assert manifest.source_artifacts.scene_frame_records == "records/scene_frames.jsonl"
    assert manifest.source_artifacts.scene_summary == "scene/scene_summary.json"
    assert manifest.source_artifacts.viewer == "viewer/index.html"
    assert manifest.assumptions
    assert {
        "DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M",
        "DEFAULT_GAZE_SPHERE_RADIUS_M",
    } <= {assumption.name for assumption in manifest.assumptions}
    assert manifest.robust_estimators.scene_center.candidate_frame_count == 7
    assert manifest.robust_estimators.scene_center.finite_candidate_frame_count == 7
    assert manifest.robust_estimators.scene_center.dropped_non_finite_frame_count == 0
    assert manifest.robust_estimators.scene_center.inlier_frame_count >= 5
    assert manifest.robust_estimators.scene_center.mad_m[0] >= 0.0
    assert manifest.robust_estimators.scene_center.thresholds_m[0] >= 0.015
    assert manifest.robust_estimators.scene_center.iteration_count >= 1
    assert (
        manifest.robust_estimators.scene_center.convergence_tolerance_m
        == pytest.approx(1e-6)
    )
    assert manifest.robust_estimators.scene_center.uncertainty == "medium"
    assert manifest.robust_estimators.main_unigaze_direction.candidate_frame_count == 6
    assert (
        manifest.robust_estimators.main_unigaze_direction.finite_candidate_frame_count
        == 6
    )
    assert (
        manifest.robust_estimators.main_unigaze_direction.median_angular_residual_radians
        is not None
    )
    assert {
        "p50",
        "p75",
        "p90",
        "p95",
    } == set(
        manifest.robust_estimators.main_unigaze_direction.angular_residual_percentiles_radians
    )
    assert manifest.robust_estimators.main_unigaze_direction.uncertainty == "medium"
    assert (
        manifest.robust_estimators.scene_orientation.method
        == "anatomical_frontal_webcam_right_up_back_axes"
    )
    assert manifest.robust_estimators.scene_orientation.candidate_frame_count == 0
    assert manifest.robust_estimators.scene_orientation.fallbacks == []
    assert manifest.gaze_sphere.radius_m == pytest.approx(0.7)
    assert manifest.gaze_sphere.radius_source == "DEFAULT_GAZE_SPHERE_RADIUS_M"
    assert manifest.gaze_sphere.center_source == "robust_scene_center"
    assert manifest.axis_basis.convention == "right_up_back_columns_right_handed"
    assert manifest.axis_basis.determinant_right_up_back > 0.99
    assert manifest.coordinate_frames.math_frame == "camera_opencv_pseudo_m"
    assert (
        manifest.coordinate_frames.projection_frame
        == CoordinateFrame3D.GAZE_SPHERE_PSEUDO_M
    )
    assert manifest.viewer_dependency.library == "three"
    assert manifest.viewer_dependency.version == THREE_VERSION
    assert manifest.viewer_dependency.source == "npm:three"
    assert manifest.viewer_dependency.license == "MIT"
    assert manifest.viewer_dependency.cdn_provider == "cdn.jsdelivr.net"
    assert manifest.viewer_dependency.module_urls == EXPECTED_MODULE_URLS

    summary = SceneSummary.model_validate_json(
        result.paths.scene_summary_path.read_text(encoding="utf-8")
    )
    assert summary.decoded_frames == 7
    assert summary.scene_frame_records == 7
    assert summary.valid_eye_midpoint_frames == 7
    assert summary.valid_unigaze_ray_frames == 6
    assert summary.valid_sphere_hit_frames == 6
    assert summary.invalid_sphere_hit_reasons == {"UNIGAZE_INVALID": 1}
    assert (
        summary.sphere_hit_angle_bounds.theta_min_radians
        <= summary.sphere_hit_angle_bounds.theta_max_radians
    )
    assert (
        summary.sphere_hit_angle_bounds.phi_min_radians
        <= summary.sphere_hit_angle_bounds.phi_max_radians
    )
    assert summary.representative_scene_warning_frame_ids == ["f000000006"]
    assert summary.artifact_validation.scene_frame_count_matches_decoded is True
    assert summary.artifact_validation.scene_manifest_valid is True
    assert summary.artifact_validation.scene_summary_valid is True

    assert result.scene_frame_count == 7
    assert result.valid_sphere_hit_count == 6
    assert result.summary == summary
    assert result.manifest == manifest


def test_scene_frames_preserve_source_identity_invalid_reasons_and_duplicate_hits(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")

    result = build_scene_artifacts(layout)
    records = load_scene_frames(result.paths.scene_frames_jsonl_path)
    raw_jsonl = result.paths.scene_frames_jsonl_path.read_text(encoding="utf-8")

    assert len(records) == 7
    assert [record.frame_id for record in records] == [
        f"f{index:09d}" for index in range(7)
    ]
    assert [record.frame_index for record in records] == list(range(7))
    persisted_lines = result.paths.scene_frames_jsonl_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line)["frame_index"] for line in persisted_lines] == list(
        range(7)
    )

    valid_hits = [record for record in records if record.sphere_hit.valid]
    assert len(valid_hits) == 6
    assert all(record.sphere_hit.point_scene_m is not None for record in valid_hits)
    duplicate_hits = [
        record.sphere_hit for record in records if record.frame_index in (2, 3)
    ]
    assert duplicate_hits[0].point_scene_m == duplicate_hits[1].point_scene_m

    invalid_record = records[6]
    assert invalid_record.source_frame_status == FrameStatus.WARNING
    assert invalid_record.unigaze_ray.valid is False
    assert invalid_record.unigaze_ray.source_reason_invalid == "GAZE_MODEL_FAILED"
    assert invalid_record.sphere_hit.valid is False
    assert invalid_record.sphere_hit.reason_invalid == "UNIGAZE_INVALID"
    assert invalid_record.diagnostics.source_error_codes == ["GAZE_MODEL_FAILED"]
    assert "main_monitor_hit" not in raw_jsonl
    assert "monitor_hit" not in raw_jsonl
    assert "plane_uv_m" not in raw_jsonl

    viewer_data = build_viewer_scene_data(result)
    assert viewer_data == result.viewer_data
    assert viewer_data.run_id == "20260626T120000Z-scene"
    assert viewer_data.source_video_stem == "synthetic_scene_source"
    assert viewer_data.frame_count == 7
    assert len(viewer_data.frames) == 7
    assert viewer_data.gaze_sphere == result.manifest.gaze_sphere
    assert not hasattr(viewer_data, "valid_hit_points")
    valid_viewer_frames = [
        frame for frame in viewer_data.frames if frame.sphere_hit.valid
    ]
    assert len(valid_viewer_frames) == result.valid_sphere_hit_count
    assert [frame.frame_index for frame in valid_viewer_frames] == [
        frame.frame_index for frame in records if frame.sphere_hit.valid
    ]
    assert [
        (frame.frame_id, frame.frame_index)
        for frame in valid_viewer_frames
        if frame.frame_index in (2, 3)
    ] == [("f000000002", 2), ("f000000003", 3)]
    assert all(
        frame.sphere_hit.radius_m == pytest.approx(0.7) for frame in valid_viewer_frames
    )
    assert viewer_data.axis_basis == result.manifest.axis_basis
    assert viewer_data.summary.valid_sphere_hit_frames == 6


def test_scene_frame_direction_maps_positive_pitch_to_scene_up(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    layout = _layout(run_dir)
    layout.records_dir.mkdir(parents=True)
    video = VideoManifest(
        source_path="artifacts/input/synthetic_upward_gaze_source.mp4",
        source_sha256="b" * 64,
        frame_width=1920,
        frame_height=1080,
        frame_count_decoded=7,
    )
    run_manifest = RunManifest(
        run_id="20260626T123000Z-scene-up",
        created_at_utc=datetime(2026, 6, 26, 12, 30, tzinfo=UTC).isoformat(),
        input_path=video.source_path,
        video=video,
        inference=_external_observer_inference_record(),
    )
    neutral_gaze = _gaze(valid=True, yaw_radians=0.0, pitch_radians=0.0)
    upward_gaze = _gaze(valid=True, yaw_radians=0.0, pitch_radians=0.20)
    frames = [
        _frame(index).model_copy(
            update={
                "appearance_gaze": neutral_gaze,
                "geometric_gaze": neutral_gaze,
                "recommended_gaze": neutral_gaze,
            }
        )
        for index in range(6)
    ]
    frames.append(
        _frame(6).model_copy(
            update={
                "appearance_gaze": upward_gaze,
                "geometric_gaze": upward_gaze,
                "recommended_gaze": upward_gaze,
            }
        )
    )
    (run_dir / "run_manifest.json").write_text(
        run_manifest.model_dump_json(), encoding="utf-8"
    )
    (run_dir / "video_manifest.json").write_text(
        video.model_dump_json(), encoding="utf-8"
    )
    (layout.records_dir / "frames.jsonl").write_text(
        "".join(frame.model_dump_json() + "\n" for frame in frames),
        encoding="utf-8",
    )

    result = build_scene_artifacts(layout)
    records = load_scene_frames(result.paths.scene_frames_jsonl_path)
    upward_record = records[6]

    assert upward_record.unigaze_ray.valid is True
    assert upward_record.unigaze_ray.direction_camera is not None
    assert upward_record.unigaze_ray.direction_scene is not None
    assert upward_record.unigaze_ray.direction_camera.y < 0.0
    assert upward_record.unigaze_ray.direction_camera.z < 0.0
    assert upward_record.unigaze_ray.direction_scene.y > 0.0
    assert upward_record.unigaze_ray.direction_scene.z < 0.0


def test_scene_frame_direction_maps_image_right_to_streamer_left(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    layout = _layout(run_dir)
    layout.records_dir.mkdir(parents=True)
    video = VideoManifest(
        source_path="artifacts/input/synthetic_left_right_gaze_source.mp4",
        source_sha256="c" * 64,
        frame_width=1920,
        frame_height=1080,
        frame_count_decoded=7,
    )
    run_manifest = RunManifest(
        run_id="20260626T124000Z-scene-left-right",
        created_at_utc=datetime(2026, 6, 26, 12, 40, tzinfo=UTC).isoformat(),
        input_path=video.source_path,
        video=video,
        inference=_external_observer_inference_record(),
    )
    neutral_gaze = _gaze(valid=True, yaw_radians=0.0, pitch_radians=0.0)
    image_right_gaze = _gaze(valid=True, yaw_radians=0.20, pitch_radians=0.0)
    frames = [
        _frame(index).model_copy(
            update={
                "appearance_gaze": neutral_gaze,
                "geometric_gaze": neutral_gaze,
                "recommended_gaze": neutral_gaze,
            }
        )
        for index in range(6)
    ]
    frames.append(
        _frame(6).model_copy(
            update={
                "appearance_gaze": image_right_gaze,
                "geometric_gaze": image_right_gaze,
                "recommended_gaze": image_right_gaze,
            }
        )
    )
    (run_dir / "run_manifest.json").write_text(
        run_manifest.model_dump_json(), encoding="utf-8"
    )
    (run_dir / "video_manifest.json").write_text(
        video.model_dump_json(), encoding="utf-8"
    )
    (layout.records_dir / "frames.jsonl").write_text(
        "".join(frame.model_dump_json() + "\n" for frame in frames),
        encoding="utf-8",
    )

    result = build_scene_artifacts(layout)
    records = load_scene_frames(result.paths.scene_frames_jsonl_path)
    image_right_record = records[6]

    assert image_right_record.unigaze_ray.valid is True
    assert image_right_record.unigaze_ray.direction_camera is not None
    assert image_right_record.unigaze_ray.direction_scene is not None
    assert image_right_record.unigaze_ray.direction_camera.x > 0.0
    assert image_right_record.unigaze_ray.direction_scene.x < 0.0
    assert image_right_record.unigaze_ray.direction_scene.z < 0.0


def test_scene_frame_places_eyes_on_front_side_of_head(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")

    result = build_scene_artifacts(layout)
    record = result.frames[0]

    assert record.eye_midpoint.scene_point_m is not None
    assert record.head.ellipsoid_center_scene_m is not None
    assert record.head.ellipsoid_center_scene_m.y < record.eye_midpoint.scene_point_m.y
    assert record.eye_midpoint.scene_point_m.z < record.head.ellipsoid_center_scene_m.z


def test_scene_frame_preserves_anatomical_eye_sides_for_frontal_webcam(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")

    result = build_scene_artifacts(layout)
    record = result.frames[0]

    assert record.left_eye.image_px is not None
    assert record.right_eye.image_px is not None
    assert record.left_eye.camera_point_m is not None
    assert record.right_eye.camera_point_m is not None
    assert record.left_eye.scene_point_m is not None
    assert record.right_eye.scene_point_m is not None
    assert record.eye_midpoint.scene_point_m is not None
    assert record.left_eye.image_px.x > record.right_eye.image_px.x
    assert record.left_eye.camera_point_m.x > record.right_eye.camera_point_m.x
    assert record.left_eye.scene_point_m.x < record.eye_midpoint.scene_point_m.x
    assert record.right_eye.scene_point_m.x > record.eye_midpoint.scene_point_m.x


def test_build_viewer_scene_data_uses_result_manifest_assumptions(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")
    result = build_scene_artifacts(layout)
    manifest_assumptions = [
        SceneAssumptionRecord(
            name="REVIEW_SENTINEL_ASSUMPTION",
            value=123.0,
            unit="review_unit",
            source="review_fix",
            uncertainty="low",
        ),
        *reversed(result.manifest.assumptions),
    ]
    modified_result = replace(
        result,
        manifest=result.manifest.model_copy(
            update={"assumptions": manifest_assumptions}
        ),
    )

    viewer_data = build_viewer_scene_data(modified_result)

    assert viewer_data.assumptions == modified_result.manifest.assumptions
    assert viewer_data.assumptions != result.viewer_data.assumptions
