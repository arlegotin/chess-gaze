from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from _pytest.capture import CaptureFixture

from chess_gaze.calibration import default_calibration
from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    CropImageRetentionPolicy,
    ErrorRecord,
    EyeRecord,
    FaceRecord,
    FrameImageRetentionPolicy,
    FrameRecord,
    GazeAngles,
    HeadPoseRecord,
    InferenceRuntimeRecord,
    QASummaryPolicy,
    RunManifest,
    VideoManifest,
)
from chess_gaze.gaze_precision_benchmark import (
    build_gaze_precision_run_metrics,
    compare_gaze_precision_runs,
    main,
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


def _frame(
    index: int,
    *,
    yaw: float | None,
    pitch: float | None,
    timestamp_seconds: float,
    frame_id_value: str | None = None,
    frame_index_value: int | None = None,
) -> FrameRecord:
    gaze_valid = yaw is not None and pitch is not None
    appearance_gaze = _gaze(gaze_valid, yaw, pitch)
    return FrameRecord(
        frame_id=frame_id_value or f"f{index:09d}",
        frame_index=index if frame_index_value is None else frame_index_value,
        status=FrameStatus.OK if gaze_valid else FrameStatus.WARNING,
        timestamp_seconds=timestamp_seconds,
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
    valid_sphere_hit_frames: int = 0,
    valid_target_plane_hit_frames: int = 0,
    in_bounds_target_plane_hit_frames: int = 0,
    source_sha256: str = "a" * 64,
    frame_width: int = 1920,
    frame_height: int = 1080,
    frame_count_decoded: int | None = None,
    pts_sequence_sha256: str | None = "b" * 64,
    pts_sequence_usable: bool = True,
    timestamps: tuple[float, ...] | None = None,
    frame_ids: tuple[str, ...] | None = None,
    frame_indices: tuple[int, ...] | None = None,
    model_checksum: str | None = "c" * 64,
    unigaze_batch_size: int = 7,
    candidate_face_score_min: float | None = None,
    save_frame_images: bool = False,
    save_crop_images: bool = False,
    generate_qa_summary: bool = True,
    standalone_source_sha256: str | None = None,
) -> None:
    (run_dir / "records").mkdir(parents=True)
    calibration = default_calibration(
        unigaze_preprocessing_profile=preprocessing_profile
    )
    if candidate_face_score_min is not None:
        calibration = calibration.model_copy(
            update={"candidate_face_score_min": candidate_face_score_min}
        )
    (run_dir / "calibration.json").write_text(
        calibration.model_dump_json(), encoding="utf-8"
    )
    resolved_timestamps = timestamps or tuple(index * 0.5 for index in range(len(yaws)))
    if len(resolved_timestamps) != len(yaws):
        raise ValueError("timestamps must align with yaws")
    if frame_ids is not None and len(frame_ids) != len(yaws):
        raise ValueError("frame_ids must align with yaws")
    if frame_indices is not None and len(frame_indices) != len(yaws):
        raise ValueError("frame_indices must align with yaws")
    frames = [
        _frame(
            index,
            yaw=yaw,
            pitch=0.0 if yaw is not None else None,
            timestamp_seconds=resolved_timestamps[index],
            frame_id_value=None if frame_ids is None else frame_ids[index],
            frame_index_value=None if frame_indices is None else frame_indices[index],
        )
        for index, yaw in enumerate(yaws)
    ]
    video = VideoManifest(
        source_path="artifacts/input/synthetic_short.mp4",
        source_sha256=source_sha256,
        frame_width=frame_width,
        frame_height=frame_height,
        frame_count_decoded=(
            len(frames) if frame_count_decoded is None else frame_count_decoded
        ),
        pts_sequence_sha256=pts_sequence_sha256,
        pts_sequence_usable=pts_sequence_usable,
    )
    inference = InferenceRuntimeRecord(
        observer_source="default_model_observer",
        unigaze_model_id="unigaze-h14-joint",
        unigaze_model_checksum_sha256=model_checksum,
        unigaze_device="cpu",
        unigaze_batch_size=unigaze_batch_size,
        torch_version="2.12.1",
        torch_mps_available=True,
        mps_fallback_env="unset",
        mps_fast_math_env="unset",
        mps_prefer_metal_env="unset",
        mps_preflight_passed=None,
    )
    run_manifest = RunManifest(
        run_id=run_dir.name,
        created_at_utc="2026-07-12T00:00:00+00:00",
        input_path=video.source_path,
        video=video,
        inference=inference,
        frame_image_retention=FrameImageRetentionPolicy(
            save_frame_images=save_frame_images
        ),
        crop_image_retention=CropImageRetentionPolicy(
            save_crop_images=save_crop_images
        ),
        qa_summary_policy=QASummaryPolicy(generate_qa_summary=generate_qa_summary),
    )
    (run_dir / "run_manifest.json").write_text(
        run_manifest.model_dump_json(), encoding="utf-8"
    )
    (run_dir / "video_manifest.json").write_text(
        video.model_copy(
            update={"source_sha256": standalone_source_sha256}
            if standalone_source_sha256 is not None
            else {}
        ).model_dump_json(),
        encoding="utf-8",
    )
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
        in_bounds_target_plane_hit_frames=in_bounds_target_plane_hit_frames,
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
        valid_target_plane_hit_frames=2,
        in_bounds_target_plane_hit_frames=1,
        timestamps=(0.0, 0.5, 1.0),
    )

    metrics = build_gaze_precision_run_metrics(run_dir)

    assert metrics.run_dir == str(run_dir)
    assert metrics.unigaze_preprocessing_profile == "legacy_bbox_rgb01"
    assert metrics.frame_count == 3
    assert metrics.valid_appearance_gaze_frames == 2
    assert metrics.valid_appearance_gaze_rate == pytest.approx(2 / 3)
    assert metrics.valid_sphere_hit_frames == 2
    assert metrics.valid_target_plane_hit_frames == 2
    assert metrics.in_bounds_target_plane_hit_frames == 1
    assert metrics.yaw_median_radians == pytest.approx(0.05)
    assert metrics.pitch_median_radians == pytest.approx(0.0)
    assert metrics.ray_step_median_radians == pytest.approx(0.1, rel=1e-3)
    expected_speed = math.degrees(0.1) / 0.5
    assert metrics.ray_speed_median_degrees_per_second == pytest.approx(
        expected_speed, rel=1e-3
    )
    assert metrics.ray_speed_p95_degrees_per_second == pytest.approx(
        expected_speed, rel=1e-3
    )
    assert metrics.ray_speed_p99_degrees_per_second == pytest.approx(
        expected_speed, rel=1e-3
    )


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
        in_bounds_target_plane_hit_frames=1,
    )

    report = compare_gaze_precision_runs(
        baseline,
        candidate,
        experimental_variable="unigaze_preprocessing",
        generated_at_utc=datetime(2026, 7, 5, 12, tzinfo=UTC),
    )

    assert report.baseline.unigaze_preprocessing_profile == "legacy_bbox_rgb01"
    assert report.candidate.unigaze_preprocessing_profile == (
        "reference_face2x_imagenet"
    )
    assert report.ray_step_median_delta_radians == pytest.approx(-0.1, rel=1e-3)
    assert report.valid_target_plane_hit_delta == 2
    assert report.in_bounds_target_plane_hit_delta == 1
    assert report.ray_speed_median_delta_degrees_per_second == pytest.approx(
        -math.degrees(0.1) / 0.5, rel=1e-3
    )
    assert report.experimental_variable == "unigaze_preprocessing"
    assert report.baseline.source_sha256 == "a" * 64
    assert report.baseline.pts_sequence_sha256 == "b" * 64
    assert report.baseline.pts_sequence_usable is True
    assert report.baseline.unigaze_model_checksum_sha256 == "c" * 64
    json.loads(report.model_dump_json())


@pytest.mark.parametrize(
    ("candidate_overrides", "expected_path"),
    [
        ({"source_sha256": "d" * 64}, "video.source_sha256"),
        ({"frame_width": 1280}, "video.frame_width"),
        ({"frame_height": 720}, "video.frame_height"),
        ({"frame_count_decoded": 4}, "video.frame_count_decoded"),
        ({"pts_sequence_sha256": "e" * 64}, "video.pts_sequence_sha256"),
        ({"model_checksum": "f" * 64}, "inference.unigaze_model_checksum_sha256"),
        ({"unigaze_batch_size": 8}, "inference.unigaze_batch_size"),
        ({"candidate_face_score_min": 0.2}, "calibration.candidate_face_score_min"),
        ({"save_frame_images": True}, "frame_image_retention.save_frame_images"),
        ({"save_crop_images": True}, "crop_image_retention.save_crop_images"),
        ({"generate_qa_summary": False}, "qa_summary_policy.generate_qa_summary"),
        ({"frame_ids": ("f000000000", "wrong", "f000000002")}, "frames.frame_id"),
        ({"frame_indices": (0, 2, 1)}, "frames.frame_index"),
        ({"timestamps": (0.0, 0.25, 1.0)}, "frames.timestamp_seconds"),
    ],
)
def test_compare_gaze_precision_runs_rejects_non_declared_differences(
    tmp_path: Path,
    candidate_overrides: dict[str, Any],
    expected_path: str,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    common: dict[str, Any] = {
        "yaws": (0.0, 0.1, 0.2),
        "preprocessing_profile": "legacy_bbox_rgb01",
    }
    _write_run(baseline, **common)
    _write_run(candidate, **common, **candidate_overrides)

    with pytest.raises(ValueError, match=expected_path.replace(".", r"\.")):
        compare_gaze_precision_runs(
            baseline,
            candidate,
            experimental_variable="unigaze_preprocessing",
        )


def test_build_gaze_precision_run_metrics_rejects_embedded_video_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        yaws=(0.0, 0.1),
        preprocessing_profile="legacy_bbox_rgb01",
        standalone_source_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match=r"run_manifest\.video\.source_sha256"):
        build_gaze_precision_run_metrics(run_dir)


def test_compare_gaze_precision_runs_sorts_mismatched_field_paths(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        yaws=(0.0, 0.1),
        preprocessing_profile="legacy_bbox_rgb01",
    )
    _write_run(
        candidate,
        yaws=(0.0, 0.1),
        preprocessing_profile="legacy_bbox_rgb01",
        source_sha256="d" * 64,
        unigaze_batch_size=8,
    )

    with pytest.raises(ValueError) as exc_info:
        compare_gaze_precision_runs(
            baseline,
            candidate,
            experimental_variable="unigaze_preprocessing",
        )

    message = str(exc_info.value)
    assert message.index("inference.unigaze_batch_size") < message.index(
        "video.source_sha256"
    )


def test_compare_gaze_precision_runs_rejects_missing_model_checksum(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        yaws=(0.0, 0.1),
        preprocessing_profile="legacy_bbox_rgb01",
        model_checksum=None,
    )
    _write_run(
        candidate,
        yaws=(0.0, 0.1),
        preprocessing_profile="reference_face2x_imagenet",
        model_checksum=None,
    )

    with pytest.raises(ValueError, match=r"inference\.unigaze_model_checksum_sha256"):
        compare_gaze_precision_runs(
            baseline,
            candidate,
            experimental_variable="unigaze_preprocessing",
        )


def test_compare_gaze_precision_runs_rejects_unknown_experimental_variable(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        yaws=(0.0, 0.1),
        preprocessing_profile="legacy_bbox_rgb01",
    )
    _write_run(
        candidate,
        yaws=(0.0, 0.1),
        preprocessing_profile="reference_face2x_imagenet",
    )

    with pytest.raises(ValueError, match="experimental_variable"):
        compare_gaze_precision_runs(
            baseline,
            candidate,
            experimental_variable=cast(Any, "unknown"),
        )


def test_compare_gaze_precision_runs_requires_declared_difference(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for run_dir in (baseline, candidate):
        _write_run(
            run_dir,
            yaws=(0.0, 0.1),
            preprocessing_profile="legacy_bbox_rgb01",
        )

    with pytest.raises(ValueError, match=r"calibration\.unigaze_preprocessing_profile"):
        compare_gaze_precision_runs(
            baseline,
            candidate,
            experimental_variable="unigaze_preprocessing",
        )


def test_build_gaze_precision_run_metrics_omits_speed_for_unusable_pts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        yaws=(0.0, 0.1, 0.2),
        preprocessing_profile="legacy_bbox_rgb01",
        pts_sequence_usable=False,
    )

    metrics = build_gaze_precision_run_metrics(run_dir)

    assert metrics.ray_step_median_radians is not None
    assert metrics.ray_speed_median_degrees_per_second is None
    assert metrics.ray_speed_p95_degrees_per_second is None
    assert metrics.ray_speed_p99_degrees_per_second is None


def test_build_gaze_precision_run_metrics_rejects_non_positive_usable_pts_delta(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        yaws=(0.0, 0.1, 0.2),
        preprocessing_profile="legacy_bbox_rgb01",
        timestamps=(0.0, 0.5, 0.5),
    )

    with pytest.raises(ValueError, match=r"frames\.timestamp_seconds"):
        build_gaze_precision_run_metrics(run_dir)


def test_main_requires_and_reports_experimental_variable(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        yaws=(0.0, 0.1),
        preprocessing_profile="legacy_bbox_rgb01",
    )
    _write_run(
        candidate,
        yaws=(0.0, 0.1),
        preprocessing_profile="reference_face2x_imagenet",
    )

    with pytest.raises(SystemExit) as exc_info:
        main([str(baseline), str(candidate)])
    assert exc_info.value.code == 2
    assert "--experimental-variable" in capsys.readouterr().err

    assert (
        main(
            [
                str(baseline),
                str(candidate),
                "--experimental-variable",
                "unigaze_preprocessing",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "gaze-precision-comparison-v2"
    assert payload["experimental_variable"] == "unigaze_preprocessing"
