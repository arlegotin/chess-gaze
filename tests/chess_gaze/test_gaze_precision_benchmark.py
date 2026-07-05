from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chess_gaze.calibration import default_calibration
from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    ErrorRecord,
    EyeRecord,
    FaceRecord,
    FrameRecord,
    GazeAngles,
    HeadPoseRecord,
)
from chess_gaze.gaze_precision_benchmark import (
    build_gaze_precision_run_metrics,
    compare_gaze_precision_runs,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.scene_records import (
    SceneArtifactValidationRecord,
    SceneSphereHitAngleBoundsRecord,
    SceneSummary,
)


def _point(x: float, y: float) -> Point2D:
    return Point2D(space=CoordinateSpace.IMAGE_PX, x=x, y=y)


def _box() -> BBox:
    return BBox(
        space=CoordinateSpace.IMAGE_PX,
        x_min=0.0,
        y_min=0.0,
        x_max=10.0,
        y_max=10.0,
    )


def _eye() -> EyeRecord:
    return EyeRecord(
        present=True,
        bounding_box=_box(),
        pupil_center=_point(5.0, 5.0),
        iris_landmarks=[_point(4.0, 5.0), _point(6.0, 5.0)],
        reason_invalid=None,
    )


def _gaze(valid: bool, yaw: float | None, pitch: float | None) -> GazeAngles:
    return GazeAngles(
        valid=valid,
        yaw_radians=yaw,
        pitch_radians=pitch,
        reason_invalid=None if valid else ErrorCode.GAZE_MODEL_FAILED,
    )


def _frame(index: int, *, yaw: float | None, pitch: float | None) -> FrameRecord:
    gaze_valid = yaw is not None and pitch is not None
    appearance_gaze = _gaze(gaze_valid, yaw, pitch)
    return FrameRecord(
        frame_id=f"f{index:09d}",
        frame_index=index,
        status=FrameStatus.OK if gaze_valid else FrameStatus.WARNING,
        timestamp_seconds=index / 30.0,
        face=FaceRecord(
            present=True,
            bounding_box=_box(),
            landmarks=[_point(2.0, 2.0), _point(8.0, 2.0)],
            reason_invalid=None,
        ),
        left_eye=_eye(),
        right_eye=_eye(),
        head_pose=HeadPoseRecord(
            valid=True,
            yaw_radians=0.0,
            pitch_radians=0.0,
            roll_radians=0.0,
            reason_invalid=None,
        ),
        geometric_gaze=_gaze(False, None, None),
        appearance_gaze=appearance_gaze,
        recommended_gaze=appearance_gaze,
        errors=(
            []
            if gaze_valid
            else [
                ErrorRecord(
                    code=ErrorCode.GAZE_MODEL_FAILED,
                    message="synthetic invalid gaze",
                )
            ]
        ),
    )


def _write_run(
    run_dir: Path,
    *,
    yaws: tuple[float | None, ...],
    preprocessing_profile: str,
    valid_sphere_hit_frames: int,
    valid_target_plane_hit_frames: int,
) -> None:
    (run_dir / "records").mkdir(parents=True)
    calibration = default_calibration(
        unigaze_preprocessing_profile=preprocessing_profile
    )
    (run_dir / "calibration.json").write_text(
        calibration.model_dump_json(), encoding="utf-8"
    )
    frames = [
        _frame(index, yaw=yaw, pitch=0.0 if yaw is not None else None)
        for index, yaw in enumerate(yaws)
    ]
    (run_dir / "records" / "frames.jsonl").write_text(
        "".join(frame.model_dump_json() + "\n" for frame in frames),
        encoding="utf-8",
    )
    summary = SceneSummary(
        run_id=run_dir.name,
        decoded_frames=len(frames),
        scene_frame_records=len(frames),
        valid_eye_midpoint_frames=len(frames),
        valid_unigaze_ray_frames=sum(frame.appearance_gaze.valid for frame in frames),
        valid_sphere_hit_frames=valid_sphere_hit_frames,
        valid_target_plane_hit_frames=valid_target_plane_hit_frames,
        invalid_sphere_hit_reasons={},
        sphere_hit_angle_bounds=SceneSphereHitAngleBoundsRecord(
            theta_min_radians=0.0,
            theta_max_radians=0.0,
            phi_min_radians=0.0,
            phi_max_radians=0.0,
            front_hemisphere_frames=valid_sphere_hit_frames,
            rear_hemisphere_frames=0,
            equator_frames=0,
        ),
        representative_scene_warning_frame_ids=[],
        artifact_validation=SceneArtifactValidationRecord(
            scene_frame_count_matches_decoded=True,
            viewer_exists=True,
            scene_manifest_valid=True,
            scene_summary_valid=True,
        ),
    )
    (run_dir / "scene").mkdir()
    (run_dir / "scene" / "scene_summary.json").write_text(
        summary.model_dump_json(), encoding="utf-8"
    )


def test_build_gaze_precision_run_metrics_reports_validity_and_jitter(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        yaws=(0.0, 0.1, None),
        preprocessing_profile="legacy_bbox_rgb01",
        valid_sphere_hit_frames=2,
        valid_target_plane_hit_frames=0,
    )

    metrics = build_gaze_precision_run_metrics(run_dir)

    assert metrics.run_dir == str(run_dir)
    assert metrics.unigaze_preprocessing_profile == "legacy_bbox_rgb01"
    assert metrics.frame_count == 3
    assert metrics.valid_appearance_gaze_frames == 2
    assert metrics.valid_appearance_gaze_rate == pytest.approx(2 / 3)
    assert metrics.valid_sphere_hit_frames == 2
    assert metrics.valid_target_plane_hit_frames == 0
    assert metrics.yaw_median_radians == pytest.approx(0.05)
    assert metrics.pitch_median_radians == pytest.approx(0.0)
    assert metrics.ray_step_median_radians == pytest.approx(0.1, rel=1e-3)


def test_compare_gaze_precision_runs_reports_candidate_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        yaws=(0.0, 0.2, 0.4),
        preprocessing_profile="legacy_bbox_rgb01",
        valid_sphere_hit_frames=3,
        valid_target_plane_hit_frames=0,
    )
    _write_run(
        candidate,
        yaws=(0.0, 0.1, 0.2),
        preprocessing_profile="reference_face2x_imagenet",
        valid_sphere_hit_frames=3,
        valid_target_plane_hit_frames=2,
    )

    report = compare_gaze_precision_runs(
        baseline,
        candidate,
        generated_at_utc=datetime(2026, 7, 5, 12, tzinfo=UTC),
    )

    assert report.baseline.unigaze_preprocessing_profile == "legacy_bbox_rgb01"
    assert report.candidate.unigaze_preprocessing_profile == (
        "reference_face2x_imagenet"
    )
    assert report.ray_step_median_delta_radians == pytest.approx(-0.1, rel=1e-3)
    assert report.valid_target_plane_hit_delta == 2
    json.loads(report.model_dump_json())
