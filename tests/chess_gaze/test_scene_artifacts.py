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
    RunManifest,
    VideoManifest,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.scene_artifacts import (
    build_scene_artifacts,
    build_viewer_scene_data,
    load_scene_frames,
)
from chess_gaze.scene_records import SceneAssumptionRecord, SceneManifest, SceneSummary


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
        left_eye=_eye(900.0 + eye_index, 540.0),
        right_eye=_eye(1020.0 + eye_index, 540.0),
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
    assert manifest.assumptions
    assert {
        "DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M",
        "DEFAULT_MONITOR_DISTANCE_FROM_EYES_M",
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
        manifest.monitor_plane.distance_source == "DEFAULT_MONITOR_DISTANCE_FROM_EYES_M"
    )
    assert manifest.axis_basis.convention == "right_up_back_columns_right_handed"
    assert manifest.axis_basis.determinant_right_up_back > 0.99
    assert manifest.coordinate_frames.math_frame == "camera_opencv_pseudo_m"
    assert manifest.viewer_dependency.library == "three"
    assert manifest.viewer_dependency.version == "0.185.0"

    summary = SceneSummary.model_validate_json(
        result.paths.scene_summary_path.read_text(encoding="utf-8")
    )
    assert summary.decoded_frames == 7
    assert summary.scene_frame_records == 7
    assert summary.valid_eye_midpoint_frames == 7
    assert summary.valid_unigaze_ray_frames == 6
    assert summary.valid_monitor_hit_frames == 6
    assert summary.invalid_monitor_hit_reasons == {"UNIGAZE_INVALID": 1}
    assert summary.monitor_hit_bounds.u_min_m <= summary.monitor_hit_bounds.u_max_m
    assert summary.monitor_hit_bounds.v_min_m <= summary.monitor_hit_bounds.v_max_m
    assert summary.representative_scene_warning_frame_ids == ["f000000006"]
    assert summary.artifact_validation.scene_frame_count_matches_decoded is True
    assert summary.artifact_validation.scene_manifest_valid is True
    assert summary.artifact_validation.scene_summary_valid is True

    assert result.scene_frame_count == 7
    assert result.valid_monitor_hit_count == 6
    assert result.summary == summary
    assert result.manifest == manifest


def test_scene_frames_preserve_source_identity_invalid_reasons_and_duplicate_hits(
    tmp_path: Path,
) -> None:
    layout = _write_minimal_run(tmp_path / "run")

    result = build_scene_artifacts(layout)
    records = load_scene_frames(result.paths.scene_frames_jsonl_path)

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

    valid_hits = [record for record in records if record.main_monitor_hit.valid]
    assert len(valid_hits) == 6
    assert all(
        record.main_monitor_hit.point_scene_m is not None for record in valid_hits
    )
    duplicate_hits = [
        record.main_monitor_hit for record in records if record.frame_index in (2, 3)
    ]
    assert duplicate_hits[0].plane_uv_m == duplicate_hits[1].plane_uv_m

    invalid_record = records[6]
    assert invalid_record.source_frame_status == FrameStatus.WARNING
    assert invalid_record.unigaze_ray.valid is False
    assert invalid_record.unigaze_ray.source_reason_invalid == "GAZE_MODEL_FAILED"
    assert invalid_record.main_monitor_hit.valid is False
    assert invalid_record.main_monitor_hit.reason_invalid == "UNIGAZE_INVALID"
    assert invalid_record.diagnostics.source_error_codes == ["GAZE_MODEL_FAILED"]

    viewer_data = build_viewer_scene_data(result)
    assert viewer_data == result.viewer_data
    assert viewer_data.run_id == "20260626T120000Z-scene"
    assert viewer_data.source_video_stem == "synthetic_scene_source"
    assert viewer_data.frame_count == 7
    assert len(viewer_data.frames) == 7
    assert len(viewer_data.valid_hit_points) == 6
    assert [point.frame_index for point in viewer_data.valid_hit_points] == list(
        range(6)
    )
    assert [
        (point.frame_id, point.frame_index)
        for point in viewer_data.valid_hit_points
        if point.frame_index in (2, 3)
    ] == [("f000000002", 2), ("f000000003", 3)]
    assert viewer_data.valid_hit_points[2].u_m == viewer_data.valid_hit_points[3].u_m
    assert viewer_data.valid_hit_points[2].v_m == viewer_data.valid_hit_points[3].v_m
    assert viewer_data.monitor_plane == result.manifest.monitor_plane
    assert viewer_data.axis_basis == result.manifest.axis_basis
    assert viewer_data.summary.valid_monitor_hit_frames == 6


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
