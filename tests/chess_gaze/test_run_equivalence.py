from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from chess_gaze.run_equivalence import EquivalenceTolerances, compare_runs
from chess_gaze.scene_calibration import default_scene_assumptions
from chess_gaze.scene_records import ViewerSceneData


def _write_minimal_run(
    run_dir: Path,
    *,
    appearance_yaw: float | None = 0.1,
    appearance_pitch: float | None = -0.2,
    ray_x: float | None = 0.0,
    ray_y: float | None = 0.0,
    sphere_theta: float | None = 0.3,
    sphere_phi: float | None = -0.1,
    frame_status: str = "OK",
    qa_raw_frames: int = 1,
    qa_crop_files: int = 0,
    viewer_frame_count: int = 1,
) -> None:
    records_dir = run_dir / "records"
    viewer_dir = run_dir / "viewer"
    records_dir.mkdir(parents=True)
    viewer_dir.mkdir(parents=True)

    frame = _frame_payload(
        appearance_yaw=appearance_yaw,
        appearance_pitch=appearance_pitch,
        status=frame_status,
    )
    scene_frame = _scene_frame_payload(
        ray_x=ray_x,
        ray_y=ray_y,
        sphere_theta=sphere_theta,
        sphere_phi=sphere_phi,
        source_frame_status=frame_status,
    )
    qa_summary = {
        "schema_version": "qa-summary-v1",
        "counts": {
            "decoded_frames": 1,
            "frame_records": 1,
            "scene_frame_records": 1,
            "raw_frames": qa_raw_frames,
            "processed_frames": 1,
            "crop_files": qa_crop_files,
        },
        "artifact_validation": {
            "schema_validation_passed": True,
            "counts_match": True,
        },
    }
    viewer_data = {
        **_viewer_data_payload(
            scene_frame=scene_frame,
            viewer_frame_count=viewer_frame_count,
        )
    }

    (records_dir / "frames.jsonl").write_text(
        json.dumps(frame, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (records_dir / "scene_frames.jsonl").write_text(
        json.dumps(scene_frame, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "qa_summary.json").write_text(
        json.dumps(qa_summary, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (viewer_dir / "scene-data.json").write_text(
        json.dumps(viewer_data, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _frame_payload(
    *,
    appearance_yaw: float | None,
    appearance_pitch: float | None,
    status: str,
) -> dict[str, Any]:
    appearance_valid = appearance_yaw is not None and appearance_pitch is not None
    appearance_reason = None if appearance_valid else "GAZE_MODEL_FAILED"
    return {
        "frame_id": "f000000000",
        "frame_index": 0,
        "status": status,
        "timestamp_seconds": 0.0,
        "face": {
            "present": True,
            "bounding_box": {
                "space": "IMAGE_PX",
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 10.0,
                "y_max": 10.0,
            },
            "landmarks": [{"space": "IMAGE_PX", "x": 1.0, "y": 1.0}],
            "reason_invalid": None,
        },
        "left_eye": _eye_payload(1.0),
        "right_eye": _eye_payload(3.0),
        "head_pose": {
            "valid": True,
            "yaw_radians": 0.0,
            "pitch_radians": 0.0,
            "roll_radians": 0.0,
            "reason_invalid": None,
        },
        "geometric_gaze": {
            "valid": True,
            "yaw_radians": 0.0,
            "pitch_radians": 0.0,
            "reason_invalid": None,
        },
        "appearance_gaze": {
            "valid": appearance_valid,
            "yaw_radians": appearance_yaw,
            "pitch_radians": appearance_pitch,
            "reason_invalid": appearance_reason,
        },
        "recommended_gaze": {
            "valid": appearance_valid,
            "yaw_radians": appearance_yaw,
            "pitch_radians": appearance_pitch,
            "reason_invalid": appearance_reason,
        },
        "errors": [],
    }


def _eye_payload(x_min: float) -> dict[str, Any]:
    return {
        "present": True,
        "bounding_box": {
            "space": "IMAGE_PX",
            "x_min": x_min,
            "y_min": 1.0,
            "x_max": x_min + 1.0,
            "y_max": 2.0,
        },
        "pupil_center": {"space": "IMAGE_PX", "x": x_min + 0.5, "y": 1.5},
        "iris_landmarks": [{"space": "IMAGE_PX", "x": x_min + 0.5, "y": 1.5}],
        "reason_invalid": None,
    }


def _scene_frame_payload(
    *,
    ray_x: float | None,
    ray_y: float | None,
    sphere_theta: float | None,
    sphere_phi: float | None,
    source_frame_status: str,
) -> dict[str, Any]:
    ray_valid = ray_x is not None and ray_y is not None
    hit_valid = sphere_theta is not None and sphere_phi is not None and ray_valid
    scene_direction = _direction_payload(ray_x, ray_y, "scene_pseudo_m")
    camera_direction = _direction_payload(ray_x, ray_y, "camera_opencv_pseudo_m")
    return {
        "schema_version": "gaze-scene-frame-v2",
        "frame_id": "f000000000",
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "source_frame_status": source_frame_status,
        "valid_for_scene_center": True,
        "valid_for_sphere_projection": ray_valid,
        "camera": {
            "fx_px": 1920.0,
            "fy_px": 1920.0,
            "cx_px": 960.0,
            "cy_px": 540.0,
            "depth_source": "interpupillary_distance_assumption",
        },
        "left_eye": _scene_eye_payload(1.0),
        "right_eye": _scene_eye_payload(3.0),
        "eye_midpoint": {
            "valid": True,
            "origin_policy": "both_eyes_required",
            "camera_point_m": _vector_payload(
                x=0.0,
                y=0.0,
                z=0.8,
                space="camera_opencv_pseudo_m",
            ),
            "scene_point_m": _vector_payload(
                x=0.0,
                y=0.0,
                z=0.0,
                space="scene_pseudo_m",
            ),
            "pupil_distance_px": 30.0,
            "estimated_depth_m": 0.8,
            "source_reason_invalid": None,
            "reason_invalid": None,
        },
        "head": {
            "valid": True,
            "ellipsoid_center_camera_m": _vector_payload(
                x=0.0,
                y=0.02,
                z=0.72,
                space="camera_opencv_pseudo_m",
            ),
            "ellipsoid_center_scene_m": _vector_payload(
                x=0.0,
                y=0.02,
                z=0.0,
                space="scene_pseudo_m",
            ),
            "radii_m": [0.09, 0.12, 0.10],
            "yaw_radians": 0.0,
            "pitch_radians": 0.0,
            "roll_radians": 0.0,
            "orientation_source": "head_pose_yaw_pitch_roll",
            "source_reason_invalid": None,
            "reason_invalid": None,
        },
        "unigaze_ray": {
            "valid": ray_valid,
            "source": "appearance_gaze",
            "origin_camera_m": _vector_payload(
                x=0.0,
                y=0.0,
                z=0.8,
                space="camera_opencv_pseudo_m",
            )
            if ray_valid
            else None,
            "origin_scene_m": _vector_payload(
                x=0.0,
                y=0.0,
                z=0.0,
                space="scene_pseudo_m",
            )
            if ray_valid
            else None,
            "direction_camera": camera_direction,
            "direction_scene": scene_direction,
            "direction_source": "appearance_gaze_unigaze_pitch_yaw"
            if ray_valid
            else None,
            "pitch_radians": -0.2 if ray_valid else None,
            "yaw_radians": 0.1 if ray_valid else None,
            "source_reason_invalid": None if ray_valid else "GAZE_MODEL_FAILED",
            "reason_invalid": None if ray_valid else "UNIGAZE_INVALID",
        },
        "sphere_hit": {
            "valid": hit_valid,
            "point_scene_m": _vector_payload(
                x=0.0,
                y=0.0,
                z=-0.7,
                space="scene_pseudo_m",
            )
            if hit_valid
            else None,
            "ray_t_m": 0.7 if hit_valid else None,
            "radius_m": 0.7 if hit_valid else None,
            "theta_radians": sphere_theta if hit_valid else None,
            "phi_radians": sphere_phi if hit_valid else None,
            "hemisphere": "front" if hit_valid else None,
            "source_reason_invalid": None if hit_valid else "UNIGAZE_INVALID",
            "reason_invalid": None if hit_valid else "UNIGAZE_INVALID",
        },
        "diagnostics": {
            "warnings": [],
            "source_error_codes": [],
        },
    }


def _scene_eye_payload(x: float) -> dict[str, Any]:
    return {
        "valid": True,
        "image_px": {"space": "IMAGE_PX", "x": x, "y": 1.5},
        "camera_point_m": _vector_payload(
            x=x / 100.0,
            y=0.0,
            z=0.8,
            space="camera_opencv_pseudo_m",
        ),
        "scene_point_m": _vector_payload(
            x=x / 100.0,
            y=0.0,
            z=0.0,
            space="scene_pseudo_m",
        ),
        "source_reason_invalid": None,
        "reason_invalid": None,
    }


def _direction_payload(
    x: float | None,
    y: float | None,
    space: str,
) -> dict[str, Any] | None:
    if x is None or y is None:
        return None
    z = -math.sqrt(max(0.0, 1.0 - (x * x) - (y * y)))
    return _vector_payload(x=x, y=y, z=z, space=space)


def _vector_payload(*, x: float, y: float, z: float, space: str) -> dict[str, Any]:
    return {
        "space": space,
        "x": x,
        "y": y,
        "z": z,
    }


def _viewer_data_payload(
    *,
    scene_frame: dict[str, Any],
    viewer_frame_count: int,
) -> dict[str, Any]:
    sphere_hit = scene_frame["sphere_hit"]
    viewer_data = ViewerSceneData.model_validate_json(
        json.dumps(
            {
                "run_id": "run-123",
                "source_video_stem": "synthetic-scene-source",
                "frame_count": viewer_frame_count,
                "frames": [scene_frame],
                "gaze_sphere": {
                    "center_scene_m": _vector_payload(
                        x=0.0,
                        y=0.0,
                        z=0.0,
                        space="scene_pseudo_m",
                    ),
                    "radius_m": 0.7,
                    "radius_source": "DEFAULT_GAZE_SPHERE_RADIUS_M",
                    "center_source": "robust_scene_center",
                },
                "axis_basis": {
                    "right_camera": _vector_payload(
                        x=-1.0,
                        y=0.0,
                        z=0.0,
                        space="camera_opencv_pseudo_m",
                    ),
                    "up_camera": _vector_payload(
                        x=0.0,
                        y=-1.0,
                        z=0.0,
                        space="camera_opencv_pseudo_m",
                    ),
                    "back_camera": _vector_payload(
                        x=0.0,
                        y=0.0,
                        z=1.0,
                        space="camera_opencv_pseudo_m",
                    ),
                    "forward_camera": _vector_payload(
                        x=0.0,
                        y=0.0,
                        z=-1.0,
                        space="camera_opencv_pseudo_m",
                    ),
                    "determinant_right_up_back": 1.0,
                    "convention": "right_up_back_columns_right_handed",
                    "fallbacks": [],
                },
                "assumptions": [
                    record.model_dump(mode="json")
                    for record in default_scene_assumptions().records
                ],
                "summary": {
                    "run_id": "run-123",
                    "decoded_frames": viewer_frame_count,
                    "scene_frame_records": viewer_frame_count,
                    "valid_eye_midpoint_frames": 1,
                    "valid_unigaze_ray_frames": 1
                    if scene_frame["unigaze_ray"]["valid"]
                    else 0,
                    "valid_sphere_hit_frames": 1 if sphere_hit["valid"] else 0,
                    "invalid_sphere_hit_reasons": (
                        {} if sphere_hit["valid"] else {sphere_hit["reason_invalid"]: 1}
                    ),
                    "sphere_hit_angle_bounds": {
                        "theta_min_radians": sphere_hit["theta_radians"] or 0.0,
                        "theta_max_radians": sphere_hit["theta_radians"] or 0.0,
                        "phi_min_radians": sphere_hit["phi_radians"] or 0.0,
                        "phi_max_radians": sphere_hit["phi_radians"] or 0.0,
                        "front_hemisphere_frames": 1
                        if sphere_hit["hemisphere"] == "front"
                        else 0,
                        "rear_hemisphere_frames": 1
                        if sphere_hit["hemisphere"] == "rear"
                        else 0,
                        "equator_frames": 1
                        if sphere_hit["hemisphere"] == "equator"
                        else 0,
                    },
                    "representative_scene_warning_frame_ids": [],
                    "artifact_validation": {
                        "scene_frame_count_matches_decoded": True,
                        "viewer_exists": True,
                        "scene_manifest_valid": True,
                        "scene_summary_valid": True,
                    },
                },
            },
            allow_nan=False,
        )
    )
    return viewer_data.model_dump(mode="json")


def test_compare_runs_accepts_numeric_deltas_within_tolerance(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(
        baseline,
        appearance_yaw=0.1000,
        appearance_pitch=-0.2000,
        ray_x=0.0100,
        ray_y=0.0200,
        sphere_theta=0.1,
        sphere_phi=-0.2,
    )
    _write_minimal_run(
        candidate,
        appearance_yaw=0.1005,
        appearance_pitch=-0.2004,
        ray_x=0.0105,
        ray_y=0.0205,
        sphere_theta=0.1 + 5e-7,
        sphere_phi=-0.2,
    )

    report = compare_runs(
        baseline,
        candidate,
        tolerances=EquivalenceTolerances(
            appearance_pitch_yaw_radians=1e-3,
            scene_ray_component=1e-3,
            sphere_hit_angle_radians=1e-6,
        ),
    )

    assert report.passed is True
    assert report.exact_mismatch_count == 0
    assert report.numeric_mismatch_count == 0
    assert report.max_appearance_pitch_yaw_delta_radians == pytest.approx(0.0005)
    assert report.max_scene_ray_component_delta == pytest.approx(0.0005)
    assert report.max_sphere_hit_angle_delta_radians == pytest.approx(5e-7)
    assert report.mismatches == []


def test_compare_runs_rejects_status_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(baseline)
    _write_minimal_run(candidate)
    frame = json.loads(
        (candidate / "records" / "frames.jsonl").read_text(encoding="utf-8")
    )
    frame["status"] = "ERROR"
    (candidate / "records" / "frames.jsonl").write_text(
        json.dumps(frame, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = compare_runs(
        baseline,
        candidate,
        tolerances=EquivalenceTolerances(
            appearance_pitch_yaw_radians=1e-6,
            scene_ray_component=1e-6,
            sphere_hit_angle_radians=1e-6,
        ),
    )

    assert report.passed is False
    assert report.exact_mismatch_count == 1
    assert report.numeric_mismatch_count == 0
    assert report.mismatches == [
        "frames[0].status exact mismatch: baseline='OK' candidate='ERROR'"
    ]


def test_compare_runs_rejects_numeric_delta_outside_tolerance(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(baseline, appearance_yaw=0.1000)
    _write_minimal_run(candidate, appearance_yaw=0.1020)

    report = compare_runs(
        baseline,
        candidate,
        tolerances=EquivalenceTolerances(
            appearance_pitch_yaw_radians=1e-3,
            scene_ray_component=1e-3,
            sphere_hit_angle_radians=1e-3,
        ),
    )

    assert report.passed is False
    assert report.exact_mismatch_count == 0
    assert report.numeric_mismatch_count == 1
    assert report.max_appearance_pitch_yaw_delta_radians == pytest.approx(0.002)
    assert report.mismatches == [
        "frames[0].appearance_gaze.yaw_radians numeric mismatch: "
        "baseline=0.1 candidate=0.102 delta=0.002 tolerance=0.001"
    ]


def test_compare_runs_rejects_qa_count_and_viewer_frame_count_mismatches(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(baseline)
    _write_minimal_run(
        candidate,
        qa_raw_frames=2,
        qa_crop_files=1,
        viewer_frame_count=2,
    )

    report = compare_runs(baseline, candidate)

    assert report.passed is False
    assert report.exact_mismatch_count == 3
    assert report.numeric_mismatch_count == 0
    assert report.mismatches == [
        "qa_summary.counts.raw_frames exact mismatch: baseline=1 candidate=2",
        "qa_summary.counts.crop_files exact mismatch: baseline=0 candidate=1",
        "viewer.frame_count exact mismatch: baseline=1 candidate=2",
    ]


def test_compare_runs_rejects_non_v2_viewer_data(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(baseline)
    _write_minimal_run(candidate)

    candidate_viewer_path = candidate / "viewer" / "scene-data.json"
    candidate_viewer_payload = json.loads(candidate_viewer_path.read_text("utf-8"))
    candidate_viewer_path.write_text(
        json.dumps(
            {
                "schema_version": "gaze-scene-viewer-data-v1",
                "frame_count": candidate_viewer_payload["frame_count"],
                "frames": candidate_viewer_payload["frames"],
            },
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = compare_runs(baseline, candidate)

    assert report.passed is False
    assert any(
        "candidate viewer scene data" in error and "schema_version" in error
        for error in report.validation_errors
    )


def test_compare_runs_does_not_count_absent_numeric_fields_when_both_invalid(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(
        baseline,
        appearance_yaw=None,
        appearance_pitch=None,
        ray_x=None,
        ray_y=None,
        sphere_theta=None,
        sphere_phi=None,
    )
    _write_minimal_run(
        candidate,
        appearance_yaw=None,
        appearance_pitch=None,
        ray_x=None,
        ray_y=None,
        sphere_theta=None,
        sphere_phi=None,
    )

    report = compare_runs(baseline, candidate)

    assert report.passed is True
    assert report.exact_mismatch_count == 0
    assert report.numeric_mismatch_count == 0
    assert report.max_appearance_pitch_yaw_delta_radians == 0.0
    assert report.max_scene_ray_component_delta == 0.0
    assert report.max_sphere_hit_angle_delta_radians == 0.0


def test_compare_runs_requires_matching_invalid_reasons_for_absent_numeric_fields(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(
        baseline,
        appearance_yaw=None,
        appearance_pitch=None,
        ray_x=None,
        ray_y=None,
        sphere_theta=None,
        sphere_phi=None,
    )
    _write_minimal_run(
        candidate,
        appearance_yaw=None,
        appearance_pitch=None,
        ray_x=None,
        ray_y=None,
        sphere_theta=None,
        sphere_phi=None,
    )
    frame = json.loads(
        (candidate / "records" / "frames.jsonl").read_text(encoding="utf-8")
    )
    frame["appearance_gaze"] = {
        **deepcopy(frame["appearance_gaze"]),
        "reason_invalid": "FACE_NOT_FOUND",
    }
    (candidate / "records" / "frames.jsonl").write_text(
        json.dumps(frame, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = compare_runs(baseline, candidate)

    assert report.passed is False
    assert report.exact_mismatch_count == 1
    assert report.numeric_mismatch_count == 0
    assert report.mismatches == [
        "frames[0].appearance_gaze.reason_invalid exact mismatch: "
        "baseline='GAZE_MODEL_FAILED' candidate='FACE_NOT_FOUND'"
    ]


def test_run_equivalence_rejects_sphere_hit_angle_delta_above_tolerance(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_run(baseline, sphere_theta=0.1, sphere_phi=-0.2)
    _write_minimal_run(candidate, sphere_theta=0.1 + 2e-6, sphere_phi=-0.2)

    report = compare_runs(
        baseline,
        candidate,
        tolerances=EquivalenceTolerances(
            appearance_pitch_yaw_radians=1e-6,
            scene_ray_component=1e-6,
            sphere_hit_angle_radians=1e-6,
        ),
    )

    assert report.passed is False
    assert report.numeric_mismatch_count == 1
    assert any(
        "scene_frames[0].sphere_hit.theta_radians numeric mismatch" in mismatch
        for mismatch in report.mismatches
    )
