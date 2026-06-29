from __future__ import annotations

import json
import math
from typing import Any

import pytest
from pydantic import ValidationError

from chess_gaze.geometry import CoordinateSpace, Point2D
from chess_gaze.scene_calibration import default_scene_assumptions
from chess_gaze.scene_records import (
    CoordinateFrame3D,
    SceneAxisBasisRecord,
    SceneEyeMidpointRecord,
    SceneEyeRecord,
    SceneFrameRecord,
    SceneInvalidReason,
    SceneManifest,
    SceneSphereHitRecord,
    SceneSummary,
    SceneUniGazeRayRecord,
    UnitVector3D,
    Vector3D,
    ViewerSceneData,
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


def _point2d_payload(x: float = 10.0, y: float = 20.0) -> dict[str, Any]:
    return Point2D(space=CoordinateSpace.IMAGE_PX, x=x, y=y).model_dump()


def _vector_payload(
    *,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    space: str = "camera_opencv_pseudo_m",
) -> dict[str, Any]:
    return {
        "space": space,
        "x": x,
        "y": y,
        "z": z,
    }


def _unit_vector_payload(
    *,
    x: float = 1.0,
    y: float = 0.0,
    z: float = 0.0,
    space: str = "camera_opencv_pseudo_m",
) -> dict[str, Any]:
    return _vector_payload(x=x, y=y, z=z, space=space)


def _eye_payload(valid: bool = True) -> dict[str, Any]:
    return {
        "valid": valid,
        "image_px": _point2d_payload(),
        "camera_point_m": _vector_payload(),
        "source_reason_invalid": None,
        "reason_invalid": None,
    }


def _midpoint_payload(valid: bool = True) -> dict[str, Any]:
    return {
        "valid": valid,
        "camera_point_m": _vector_payload(x=0.0, y=0.0, z=0.8),
        "scene_point_m": _vector_payload(
            x=0.0,
            y=0.0,
            z=0.0,
            space="scene_pseudo_m",
        ),
        "origin_policy": "both_eyes_required",
        "pupil_distance_px": 31.5,
        "estimated_depth_m": 0.8,
        "source_reason_invalid": None,
        "reason_invalid": None,
    }


def _head_payload(valid: bool = True) -> dict[str, Any]:
    return {
        "valid": valid,
        "ellipsoid_center_camera_m": _vector_payload(x=0.0, y=0.025, z=0.72),
        "ellipsoid_center_scene_m": _vector_payload(
            x=0.0,
            y=0.03,
            z=0.02,
            space="scene_pseudo_m",
        ),
        "radii_m": (0.09, 0.12, 0.10),
        "yaw_radians": 0.02,
        "pitch_radians": -0.04,
        "roll_radians": 0.01,
        "orientation_source": "head_pose_yaw_pitch_roll",
        "source_reason_invalid": None,
        "reason_invalid": None,
    }


def _ray_payload(valid: bool = True) -> dict[str, Any]:
    return {
        "valid": valid,
        "source": "appearance_gaze",
        "origin_camera_m": _vector_payload(x=0.0, y=0.0, z=0.8),
        "origin_scene_m": _vector_payload(
            x=0.0,
            y=0.0,
            z=0.0,
            space="scene_pseudo_m",
        ),
        "direction_camera": _unit_vector_payload(),
        "direction_scene": _unit_vector_payload(space="scene_pseudo_m"),
        "direction_source": "appearance_gaze_unigaze_pitch_yaw",
        "pitch_radians": 0.1,
        "yaw_radians": -0.1,
        "source_reason_invalid": None,
        "reason_invalid": None,
    }


def _scene_vector(x: float, y: float, z: float) -> dict[str, Any]:
    return _vector_payload(x=x, y=y, z=z, space="scene_pseudo_m")


def _sphere_hit_payload() -> dict[str, Any]:
    return {
        "valid": True,
        "point_scene_m": _scene_vector(0.0, 0.0, -0.7),
        "ray_t_m": 0.7,
        "radius_m": 0.7,
        "theta_radians": 0.0,
        "phi_radians": 0.0,
        "hemisphere": "front",
        "source_reason_invalid": None,
        "reason_invalid": None,
    }


def _axis_basis_payload() -> dict[str, Any]:
    return {
        "right_camera": _unit_vector_payload(x=-1.0, y=0.0, z=0.0),
        "up_camera": _unit_vector_payload(x=0.0, y=-1.0, z=0.0),
        "back_camera": _unit_vector_payload(x=0.0, y=0.0, z=1.0),
        "forward_camera": _unit_vector_payload(x=0.0, y=0.0, z=-1.0),
        "determinant_right_up_back": 1.0,
        "convention": "right_up_back_columns_right_handed",
        "fallbacks": [],
    }


def _gaze_sphere_payload() -> dict[str, Any]:
    return {
        "center_scene_m": _scene_vector(0.0, 0.0, 0.0),
        "radius_m": 0.7,
        "radius_source": "DEFAULT_GAZE_SPHERE_RADIUS_M",
        "center_source": "robust_scene_center",
    }


def _frame_camera_payload() -> dict[str, Any]:
    return {
        "fx_px": 1920.0,
        "fy_px": 1920.0,
        "cx_px": 960.0,
        "cy_px": 540.0,
        "depth_source": "interpupillary_distance_assumption",
    }


def _manifest_payload() -> dict[str, Any]:
    return {
        "run_id": "run-123",
        "source_video_path": "artifacts/input/nakamura_short.mp4",
        "source_video_sha256": "abc123",
        "source_artifacts": {
            "frame_records": "records/frames.jsonl",
            "scene_frame_records": "records/scene_frames.jsonl",
            "scene_summary": "scene/scene_summary.json",
            "viewer": "viewer/index.html",
        },
        "coordinate_frames": {
            "math_frame": "camera_opencv_pseudo_m",
            "scene_frame": "scene_pseudo_m",
            "projection_frame": "gaze_sphere_pseudo_m",
            "viewer_frame": "three_view",
        },
        "camera_model": {
            "policy": "estimated_pinhole_from_image_size",
            "frame_width_px": 1920,
            "frame_height_px": 1080,
            "fx_px": 960.0,
            "fy_px": 960.0,
            "cx_px": 960.0,
            "cy_px": 540.0,
            "metric_translation_allowed": False,
            "uncertainty": "high",
        },
        "assumptions": [
            record.model_dump() for record in default_scene_assumptions().records
        ],
        "robust_estimators": {
            "scene_center": {
                "method": "geometric_median_after_mad_screen",
                "candidate_frame_count": 1900,
                "finite_candidate_frame_count": 1890,
                "dropped_non_finite_frame_count": 10,
                "inlier_frame_count": 1850,
                "mad_m": (0.012, 0.010, 0.08),
                "thresholds_m": (0.042, 0.035, 0.28),
                "iteration_count": 18,
                "convergence_tolerance_m": 0.000001,
                "fallback_used": False,
                "uncertainty": "medium",
            },
            "main_unigaze_direction": {
                "method": "angular_ransac_then_normalized_inlier_mean",
                "candidate_frame_count": 1800,
                "finite_candidate_frame_count": 1790,
                "inlier_frame_count": 1550,
                "inlier_angle_radians": 0.35,
                "median_angular_residual_radians": 0.11,
                "angular_residual_percentiles_radians": {
                    "p50": 0.11,
                    "p75": 0.18,
                    "p90": 0.27,
                    "p95": 0.31,
                },
                "fallback_used": False,
                "uncertainty": "medium",
            },
            "scene_orientation": {
                "method": "anatomical_frontal_webcam_right_up_back_axes",
                "candidate_frame_count": 0,
                "fallbacks": [],
            },
        },
        "scene_center_camera_m": _vector_payload(x=0.0, y=0.0, z=0.65),
        "scene_axes_camera": _axis_basis_payload(),
        "gaze_sphere": _gaze_sphere_payload(),
        "viewer": {
            "library": "three",
            "version": "0.185.0",
            "source": "npm:three",
            "license": "MIT",
            "dist_integrity": "sha512-test",
        },
        "generated_at_utc": "2026-06-26T12:00:00Z",
    }


def _summary_payload() -> dict[str, Any]:
    return {
        "run_id": "run-123",
        "decoded_frames": 20,
        "scene_frame_records": 20,
        "valid_eye_midpoint_frames": 10,
        "valid_unigaze_ray_frames": 8,
        "valid_sphere_hit_frames": 6,
        "invalid_sphere_hit_reasons": {"UNIGAZE_INVALID": 2},
        "sphere_hit_angle_bounds": {
            "theta_min_radians": -0.42,
            "theta_max_radians": 0.42,
            "phi_min_radians": -0.2,
            "phi_max_radians": 0.2,
            "front_hemisphere_frames": 4,
            "rear_hemisphere_frames": 1,
            "equator_frames": 1,
        },
        "representative_scene_warning_frame_ids": ["frame-0002"],
        "artifact_validation": {
            "scene_frame_count_matches_decoded": True,
            "viewer_exists": True,
            "scene_manifest_valid": True,
            "scene_summary_valid": True,
        },
    }


def _scene_frame_payload() -> dict[str, Any]:
    return {
        "frame_id": "frame-0001",
        "frame_index": 1,
        "timestamp_seconds": 0.1,
        "source_frame_status": "OK",
        "valid_for_scene_center": True,
        "valid_for_sphere_projection": True,
        "camera": _frame_camera_payload(),
        "left_eye": _eye_payload(),
        "right_eye": _eye_payload(),
        "eye_midpoint": _midpoint_payload(),
        "head": _head_payload(),
        "unigaze_ray": _ray_payload(),
        "sphere_hit": _sphere_hit_payload(),
        "diagnostics": {
            "warnings": [],
            "source_error_codes": [],
        },
    }


def test_vector3d_rejects_non_finite_values_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Vector3D.model_validate(_vector_payload(x=math.nan))

    with pytest.raises(ValidationError):
        Vector3D.model_validate(_vector_payload(y=math.inf))

    with pytest.raises(ValidationError):
        Vector3D.model_validate(
            {
                **_vector_payload(),
                "unknown": "field",
            }
        )


def test_unit_vector3d_rejects_norms_outside_tolerance() -> None:
    with pytest.raises(ValidationError):
        UnitVector3D.model_validate(_unit_vector_payload(x=0.998, y=0.0, z=0.0))

    with pytest.raises(ValidationError):
        UnitVector3D.model_validate(_unit_vector_payload(x=1.002, y=0.0, z=0.0))


def test_scene_enums_reject_unknown_strings() -> None:
    with pytest.raises(ValidationError):
        Vector3D.model_validate(_vector_payload(space="camera_pseudo_m"))

    with pytest.raises(ValidationError):
        SceneEyeRecord.model_validate(
            {
                **_eye_payload(valid=False),
                "image_px": None,
                "camera_point_m": None,
                "source_reason_invalid": "unknown eye state",
                "reason_invalid": "UNKNOWN_REASON",
            }
        )


def test_scene_frame_record_rejects_unknown_source_frame_status() -> None:
    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(
            {
                **_scene_frame_payload(),
                "source_frame_status": "NOT_A_REAL_STATUS",
            }
        )


def test_scene_records_enforce_semantic_coordinate_frames() -> None:
    with pytest.raises(ValidationError):
        SceneEyeRecord.model_validate(
            {
                **_eye_payload(),
                "camera_point_m": _vector_payload(space="scene_pseudo_m"),
            }
        )

    with pytest.raises(ValidationError):
        SceneEyeRecord.model_validate(
            {
                **_eye_payload(),
                "scene_point_m": _vector_payload(space="camera_opencv_pseudo_m"),
            }
        )

    with pytest.raises(ValidationError):
        SceneUniGazeRayRecord.model_validate(
            {
                **_ray_payload(),
                "direction_scene": _unit_vector_payload(space="camera_opencv_pseudo_m"),
            }
        )

    with pytest.raises(ValidationError):
        SceneManifest.model_validate(
            {
                **_manifest_payload(),
                "gaze_sphere": {
                    **_gaze_sphere_payload(),
                    "center_scene_m": _vector_payload(space="camera_opencv_pseudo_m"),
                },
            }
        )

    with pytest.raises(ValidationError):
        ViewerSceneData.model_validate(
            {
                "run_id": "run-123",
                "source_video_stem": "nakamura_short",
                "frame_count": 1,
                "frames": [_scene_frame_payload()],
                "valid_hit_points": [
                    {
                        "frame_id": "frame-0001",
                        "frame_index": 1,
                        "point_scene_m": _vector_payload(x=0.2, y=0.1, z=0.3),
                        "radius_m": 0.7,
                        "theta_radians": 0.2,
                        "phi_radians": 0.1,
                        "hemisphere": "front",
                    }
                ],
                "gaze_sphere": _gaze_sphere_payload(),
                "axis_basis": _axis_basis_payload(),
                "assumptions": [
                    record.model_dump()
                    for record in default_scene_assumptions().records
                ],
                "summary": {
                    **_summary_payload(),
                    "decoded_frames": 1,
                    "scene_frame_records": 1,
                    "valid_eye_midpoint_frames": 1,
                    "valid_unigaze_ray_frames": 1,
                    "valid_sphere_hit_frames": 1,
                    "invalid_sphere_hit_reasons": {},
                    "representative_scene_warning_frame_ids": [],
                },
            }
        )


def test_scene_manifest_rejects_wrong_coordinate_frame_mapping() -> None:
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(
            {
                **_manifest_payload(),
                "coordinate_frames": {
                    **_manifest_payload()["coordinate_frames"],
                    "math_frame": "scene_pseudo_m",
                },
            }
        )

    with pytest.raises(ValidationError):
        SceneManifest.model_validate(
            {
                **_manifest_payload(),
                "coordinate_frames": {
                    **_manifest_payload()["coordinate_frames"],
                    "viewer_frame": "camera_opencv_pseudo_m",
                },
            }
        )


def test_scene_axis_basis_rejects_parallel_back_and_negative_determinant() -> None:
    with pytest.raises(ValidationError):
        SceneAxisBasisRecord.model_validate(
            {
                **_axis_basis_payload(),
                "back_camera": _unit_vector_payload(x=0.0, y=0.0, z=-1.0),
            }
        )

    with pytest.raises(ValidationError):
        SceneAxisBasisRecord.model_validate(
            {
                **_axis_basis_payload(),
                "determinant_right_up_back": -1.0,
            }
        )


def test_scene_axis_basis_rejects_computed_degenerate_basis() -> None:
    with pytest.raises(ValidationError):
        SceneAxisBasisRecord.model_validate(
            {
                **_axis_basis_payload(),
                "right_camera": _unit_vector_payload(x=1.0, y=0.0, z=0.0),
                "up_camera": _unit_vector_payload(x=1.0, y=0.0, z=0.0),
                "determinant_right_up_back": 1.0,
            }
        )


def test_valid_scene_eye_requires_image_and_camera_point() -> None:
    with pytest.raises(ValidationError):
        SceneEyeRecord.model_validate(
            {
                **_eye_payload(),
                "image_px": None,
            }
        )

    with pytest.raises(ValidationError):
        SceneEyeRecord.model_validate(
            {
                **_eye_payload(),
                "camera_point_m": None,
            }
        )


def test_invalid_scene_eye_requires_reason_invalid() -> None:
    with pytest.raises(ValidationError):
        SceneEyeRecord.model_validate(
            {
                **_eye_payload(valid=False),
                "image_px": None,
                "camera_point_m": None,
                "source_reason_invalid": "LEFT_EYE_NOT_FOUND",
                "reason_invalid": None,
            }
        )


def test_valid_scene_eye_midpoint_requires_valid_eyes_and_camera_point() -> None:
    with pytest.raises(ValidationError):
        SceneEyeMidpointRecord.model_validate(
            {
                **_midpoint_payload(),
                "camera_point_m": None,
            }
        )

    invalid_frame_payload = _scene_frame_payload()
    invalid_frame_payload["right_eye"] = {
        **_eye_payload(valid=False),
        "image_px": None,
        "camera_point_m": None,
        "source_reason_invalid": "RIGHT_EYE_NOT_FOUND",
        "reason_invalid": "RIGHT_EYE_INVALID",
    }

    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(invalid_frame_payload)


def test_scene_frame_record_rejects_contradictory_estimator_eligibility_flags() -> None:
    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(
            {
                **_scene_frame_payload(),
                "valid_for_scene_center": True,
                "eye_midpoint": {
                    **_midpoint_payload(valid=False),
                    "camera_point_m": None,
                    "scene_point_m": None,
                    "origin_policy": None,
                    "pupil_distance_px": None,
                    "estimated_depth_m": None,
                    "source_reason_invalid": "RIGHT_EYE_INVALID",
                    "reason_invalid": "EYE_MIDPOINT_INVALID",
                },
            }
        )

    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(
            {
                **_scene_frame_payload(),
                "valid_for_sphere_projection": True,
                "unigaze_ray": {
                    **_ray_payload(valid=False),
                    "origin_camera_m": None,
                    "origin_scene_m": None,
                    "direction_camera": None,
                    "direction_scene": None,
                    "direction_source": None,
                    "pitch_radians": None,
                    "yaw_radians": None,
                    "source_reason_invalid": "GAZE_MODEL_FAILED",
                    "reason_invalid": "UNIGAZE_INVALID",
                },
            }
        )


def test_valid_unigaze_ray_requires_origin_and_appearance_source() -> None:
    with pytest.raises(ValidationError):
        SceneUniGazeRayRecord.model_validate(
            {
                **_ray_payload(),
                "origin_camera_m": None,
            }
        )

    with pytest.raises(ValidationError):
        SceneUniGazeRayRecord.model_validate(
            {
                **_ray_payload(),
                "direction_camera": _unit_vector_payload(x=0.5, y=0.0, z=0.0),
            }
        )

    with pytest.raises(ValidationError):
        SceneUniGazeRayRecord.model_validate(
            {
                **_ray_payload(),
                "source": "recommended_gaze",
            }
        )


def test_scene_head_radii_accept_json_style_lists_but_reject_bad_values() -> None:
    head = SceneFrameRecord.model_validate(_scene_frame_payload()).head

    assert head.model_dump(by_alias=True)["ellipsoid_radii_m"] == (0.09, 0.12, 0.10)

    with pytest.raises(ValidationError):
        type(head).model_validate(
            {
                **_head_payload(),
                "radii_m": [0.09, 0.12],
            }
        )

    with pytest.raises(ValidationError):
        type(head).model_validate(
            {
                **_head_payload(),
                "radii_m": [0.09, math.nan, 0.10],
            }
        )


def test_valid_sphere_hit_requires_point_angles_radius_and_forward_t() -> None:
    hit = SceneSphereHitRecord.model_validate(_sphere_hit_payload())

    assert hit.valid is True
    assert hit.point_scene_m is not None
    assert hit.ray_t_m == 0.7
    assert hit.radius_m == 0.7
    assert hit.theta_radians == 0.0
    assert hit.phi_radians == 0.0
    assert hit.hemisphere == "front"

    with pytest.raises(ValidationError):
        SceneSphereHitRecord.model_validate(
            {**_sphere_hit_payload(), "ray_t_m": -0.001}
        )

    with pytest.raises(ValidationError):
        SceneSphereHitRecord.model_validate({**_sphere_hit_payload(), "radius_m": 0.0})


def test_invalid_sphere_hit_requires_explicit_reason() -> None:
    hit = SceneSphereHitRecord.model_validate(
        {
            "valid": False,
            "point_scene_m": None,
            "ray_t_m": None,
            "radius_m": None,
            "theta_radians": None,
            "phi_radians": None,
            "hemisphere": None,
            "source_reason_invalid": "appearance gaze unavailable",
            "reason_invalid": "UNIGAZE_INVALID",
        }
    )

    assert hit.valid is False
    assert hit.reason_invalid == SceneInvalidReason.UNIGAZE_INVALID

    with pytest.raises(ValidationError):
        SceneSphereHitRecord.model_validate({"valid": False, "reason_invalid": None})


def test_frame_record_uses_sphere_hit_and_rejects_monitor_hit() -> None:
    frame = SceneFrameRecord.model_validate(_scene_frame_payload())

    assert frame.schema_version == "gaze-scene-frame-v2"
    assert frame.sphere_hit.valid is True
    assert frame.valid_for_sphere_projection is True
    assert "main_monitor_hit" not in frame.model_dump()

    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(
            {**_scene_frame_payload(), "main_monitor_hit": _sphere_hit_payload()}
        )


def test_invalid_nested_records_retain_explicit_scene_invalid_reasons() -> None:
    frame = SceneFrameRecord.model_validate(
        {
            **_scene_frame_payload(),
            "valid_for_scene_center": False,
            "valid_for_sphere_projection": False,
            "left_eye": {
                **_eye_payload(valid=False),
                "image_px": None,
                "camera_point_m": None,
                "source_reason_invalid": "LEFT_EYE_NOT_FOUND",
                "reason_invalid": "LEFT_EYE_INVALID",
            },
            "right_eye": {
                **_eye_payload(valid=False),
                "image_px": None,
                "camera_point_m": None,
                "source_reason_invalid": "RIGHT_EYE_NOT_FOUND",
                "reason_invalid": "RIGHT_EYE_INVALID",
            },
            "eye_midpoint": {
                **_midpoint_payload(valid=False),
                "camera_point_m": None,
                "scene_point_m": None,
                "origin_policy": None,
                "pupil_distance_px": None,
                "estimated_depth_m": None,
                "source_reason_invalid": "RIGHT_EYE_INVALID",
                "reason_invalid": "EYE_MIDPOINT_INVALID",
            },
            "head": {
                **_head_payload(valid=False),
                "ellipsoid_center_camera_m": None,
                "ellipsoid_center_scene_m": None,
                "orientation_source": None,
                "source_reason_invalid": "HEAD_POSE_INVALID",
                "reason_invalid": "NON_FINITE_INPUT",
            },
            "unigaze_ray": {
                **_ray_payload(valid=False),
                "origin_camera_m": None,
                "origin_scene_m": None,
                "direction_camera": None,
                "direction_scene": None,
                "direction_source": None,
                "pitch_radians": None,
                "yaw_radians": None,
                "source_reason_invalid": "GAZE_MODEL_FAILED",
                "reason_invalid": "UNIGAZE_INVALID",
            },
            "sphere_hit": {
                "valid": False,
                "point_scene_m": None,
                "ray_t_m": None,
                "radius_m": None,
                "theta_radians": None,
                "phi_radians": None,
                "hemisphere": None,
                "source_reason_invalid": "appearance gaze unavailable",
                "reason_invalid": "UNIGAZE_INVALID",
            },
        }
    )

    assert frame.left_eye.reason_invalid == SceneInvalidReason.LEFT_EYE_INVALID
    assert frame.right_eye.reason_invalid == SceneInvalidReason.RIGHT_EYE_INVALID
    assert frame.eye_midpoint.reason_invalid == SceneInvalidReason.EYE_MIDPOINT_INVALID
    assert frame.unigaze_ray.reason_invalid == SceneInvalidReason.UNIGAZE_INVALID
    assert frame.sphere_hit.reason_invalid == SceneInvalidReason.UNIGAZE_INVALID
    assert frame.eye_midpoint.source_reason_invalid == "RIGHT_EYE_INVALID"
    assert frame.head.source_reason_invalid == "HEAD_POSE_INVALID"
    assert frame.unigaze_ray.source_reason_invalid == "GAZE_MODEL_FAILED"
    assert frame.sphere_hit.source_reason_invalid == "appearance gaze unavailable"


def test_scene_frame_record_serializes_spec_alias_fields() -> None:
    frame = SceneFrameRecord.model_validate(_scene_frame_payload())
    payload = frame.model_dump(by_alias=True)

    assert payload["schema_version"] == "gaze-scene-frame-v2"
    assert payload["source_frame_status"] == "OK"
    assert payload["valid_for_scene_center"] is True
    assert payload["valid_for_sphere_projection"] is True
    assert payload["camera"]["depth_source"] == "interpupillary_distance_assumption"
    assert payload["left_eye"]["camera_m"]["space"] == "camera_opencv_pseudo_m"
    assert payload["eye_midpoint"]["camera_m"]["z"] == 0.8
    assert payload["eye_midpoint"]["scene_m"]["space"] == "scene_pseudo_m"
    assert payload["sphere_hit"]["ray_t_m"] == 0.7
    assert payload["diagnostics"] == {"warnings": [], "source_error_codes": []}


def test_scene_frame_record_validates_from_json_text_and_round_trips() -> None:
    payload = _scene_frame_payload()
    payload["head"].pop("radii_m")
    payload["head"]["ellipsoid_radii_m"] = [0.09, 0.12, 0.10]

    frame = SceneFrameRecord.model_validate_json(json.dumps(payload))

    assert frame.model_dump(by_alias=True)["head"]["ellipsoid_radii_m"] == (
        0.09,
        0.12,
        0.10,
    )
    assert SceneFrameRecord.model_validate_json(
        frame.model_dump_json(by_alias=True)
    ).model_dump(by_alias=True)["head"]["ellipsoid_radii_m"] == (0.09, 0.12, 0.10)


def test_scene_camera_model_requires_approved_policy_literal() -> None:
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(
            {
                **_manifest_payload(),
                "camera_model": {
                    **_manifest_payload()["camera_model"],
                    "policy": "estimated_pinhole_from_frame_size",
                },
            }
        )


def test_scene_manifest_serializes_structured_spec_fields() -> None:
    manifest = SceneManifest.model_validate(_manifest_payload())
    payload = manifest.model_dump(by_alias=True)

    assert payload["schema_version"] == "gaze-scene-manifest-v2"
    assert payload["camera_model"]["policy"] == "estimated_pinhole_from_image_size"
    assert (
        payload["source_artifacts"]["scene_frame_records"]
        == "records/scene_frames.jsonl"
    )
    assert payload["coordinate_frames"]["projection_frame"] == "gaze_sphere_pseudo_m"
    assert payload["coordinate_frames"]["viewer_frame"] == "three_view"
    assert payload["scene_axes_camera"]["forward_camera"]["z"] == -1.0
    assert payload["robust_estimators"]["scene_center"]["thresholds_m"] == (
        0.042,
        0.035,
        0.28,
    )
    assert (
        payload["robust_estimators"]["main_unigaze_direction"][
            "angular_residual_percentiles_radians"
        ]["p95"]
        == 0.31
    )
    assert payload["gaze_sphere"]["radius_m"] == 0.7
    assert payload["viewer"]["version"] == "0.185.0"
    assert payload["generated_at_utc"] == "2026-06-26T12:00:00Z"


def test_scene_manifest_accepts_remote_viewer_dependency_provenance() -> None:
    manifest = SceneManifest.model_validate(
        {
            **_manifest_payload(),
            "viewer": {
                **_manifest_payload()["viewer"],
                "cdn_provider": "cdn.jsdelivr.net",
                "module_urls": EXPECTED_MODULE_URLS,
            },
        }
    )

    assert manifest.viewer_dependency.cdn_provider == "cdn.jsdelivr.net"
    assert manifest.viewer_dependency.module_urls == EXPECTED_MODULE_URLS
    payload = manifest.model_dump(by_alias=True)
    assert payload["viewer"]["cdn_provider"] == "cdn.jsdelivr.net"
    assert payload["viewer"]["module_urls"] == EXPECTED_MODULE_URLS


def test_manifest_and_viewer_data_use_gaze_sphere() -> None:
    manifest = SceneManifest.model_validate(_manifest_payload())
    viewer_data = ViewerSceneData.model_validate(
        {
            "run_id": "run-123",
            "source_video_stem": "nakamura_short",
            "frame_count": 1,
            "frames": [_scene_frame_payload()],
            "valid_hit_points": [
                {
                    "frame_id": "frame-0001",
                    "frame_index": 1,
                    "point_scene_m": _scene_vector(0.0, 0.0, -0.7),
                    "radius_m": 0.7,
                    "theta_radians": 0.0,
                    "phi_radians": 0.0,
                    "hemisphere": "front",
                }
            ],
            "gaze_sphere": _gaze_sphere_payload(),
            "axis_basis": _axis_basis_payload(),
            "assumptions": [
                record.model_dump() for record in default_scene_assumptions().records
            ],
            "summary": {
                **_summary_payload(),
                "decoded_frames": 1,
                "scene_frame_records": 1,
                "valid_eye_midpoint_frames": 1,
                "valid_unigaze_ray_frames": 1,
                "valid_sphere_hit_frames": 1,
                "invalid_sphere_hit_reasons": {},
                "representative_scene_warning_frame_ids": [],
            },
        }
    )

    assert manifest.schema_version == "gaze-scene-manifest-v2"
    assert (
        manifest.coordinate_frames.projection_frame
        == CoordinateFrame3D.GAZE_SPHERE_PSEUDO_M
    )
    assert manifest.gaze_sphere.radius_m == 0.7
    assert viewer_data.schema_version == "gaze-scene-viewer-data-v2"
    assert viewer_data.gaze_sphere == manifest.gaze_sphere
    assert "monitor_plane" not in viewer_data.model_dump()


def test_structured_nested_models_reject_non_finite_values_and_unknown_keys() -> None:
    frame_payload = _scene_frame_payload()
    frame_payload["diagnostics"]["unexpected_metric"] = math.nan
    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(frame_payload)

    unknown_frame_payload = _scene_frame_payload()
    unknown_frame_payload["diagnostics"]["unexpected"] = "value"
    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(unknown_frame_payload)

    manifest_payload = _manifest_payload()
    manifest_payload["robust_estimators"]["scene_center"]["candidate_frame_count"] = (
        math.nan
    )
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(manifest_payload)

    unknown_manifest_payload = _manifest_payload()
    unknown_manifest_payload["robust_estimators"]["scene_center"]["unexpected"] = (
        "value"
    )
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(unknown_manifest_payload)

    viewer_manifest_payload = _manifest_payload()
    viewer_manifest_payload["viewer"]["dist_integrity"] = math.nan
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(viewer_manifest_payload)

    unknown_viewer_manifest_payload = _manifest_payload()
    unknown_viewer_manifest_payload["viewer"]["unexpected"] = "value"
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(unknown_viewer_manifest_payload)


def test_scene_manifest_rejects_non_finite_and_unknown_estimator_diagnostics() -> None:
    non_finite_mad_payload = _manifest_payload()
    non_finite_mad_payload["robust_estimators"]["scene_center"]["mad_m"] = (
        0.012,
        math.nan,
        0.08,
    )
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(non_finite_mad_payload)

    non_finite_threshold_payload = _manifest_payload()
    non_finite_threshold_payload["robust_estimators"]["scene_center"][
        "thresholds_m"
    ] = (0.042, math.inf, 0.28)
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(non_finite_threshold_payload)

    non_finite_percentile_payload = _manifest_payload()
    non_finite_percentile_payload["robust_estimators"]["main_unigaze_direction"][
        "angular_residual_percentiles_radians"
    ]["p50"] = math.inf
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(non_finite_percentile_payload)

    missing_percentile_payload = _manifest_payload()
    missing_percentile_payload["robust_estimators"]["main_unigaze_direction"][
        "angular_residual_percentiles_radians"
    ].pop("p95")
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(missing_percentile_payload)

    extra_percentile_payload = _manifest_payload()
    extra_percentile_payload["robust_estimators"]["main_unigaze_direction"][
        "angular_residual_percentiles_radians"
    ]["p99"] = 0.4
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(extra_percentile_payload)


def test_scene_summary_serializes_structured_breakdowns() -> None:
    summary = SceneSummary.model_validate(_summary_payload())
    payload = summary.model_dump()

    assert payload["schema_version"] == "gaze-scene-summary-v2"
    assert payload["invalid_sphere_hit_reasons"] == {"UNIGAZE_INVALID": 2}
    assert payload["sphere_hit_angle_bounds"]["theta_min_radians"] == -0.42
    assert payload["representative_scene_warning_frame_ids"] == ["frame-0002"]
    assert payload["artifact_validation"]["scene_manifest_valid"] is True


def test_summary_reports_sphere_hit_bounds_and_reasons() -> None:
    summary = SceneSummary.model_validate(_summary_payload())
    payload = summary.model_dump()

    assert summary.schema_version == "gaze-scene-summary-v2"
    assert payload["valid_sphere_hit_frames"] == 6
    assert payload["invalid_sphere_hit_reasons"] == {"UNIGAZE_INVALID": 2}
    assert payload["sphere_hit_angle_bounds"]["theta_min_radians"] == -0.42
    assert payload["sphere_hit_angle_bounds"]["theta_max_radians"] == 0.42
    assert payload["sphere_hit_angle_bounds"]["phi_min_radians"] == -0.2
    assert payload["sphere_hit_angle_bounds"]["phi_max_radians"] == 0.2


def test_viewer_scene_data_serializes_schema_version_and_hit_identities() -> None:
    viewer_data = ViewerSceneData.model_validate(
        {
            "run_id": "run-123",
            "source_video_stem": "nakamura_short",
            "frame_count": 1,
            "frames": [_scene_frame_payload()],
            "valid_hit_points": [
                {
                    "frame_id": "frame-0001",
                    "frame_index": 1,
                    "point_scene_m": _scene_vector(0.0, 0.0, -0.7),
                    "radius_m": 0.7,
                    "theta_radians": 0.0,
                    "phi_radians": 0.0,
                    "hemisphere": "front",
                }
            ],
            "gaze_sphere": _gaze_sphere_payload(),
            "axis_basis": _axis_basis_payload(),
            "assumptions": [
                record.model_dump() for record in default_scene_assumptions().records
            ],
            "summary": {
                **_summary_payload(),
                "decoded_frames": 1,
                "scene_frame_records": 1,
                "valid_eye_midpoint_frames": 1,
                "valid_unigaze_ray_frames": 1,
                "valid_sphere_hit_frames": 1,
                "invalid_sphere_hit_reasons": {},
                "representative_scene_warning_frame_ids": [],
            },
        }
    )

    payload = viewer_data.model_dump()

    assert payload["schema_version"] == "gaze-scene-viewer-data-v2"
    assert len(payload["frames"]) == 1
    assert payload["valid_hit_points"] == [
        {
            "frame_id": "frame-0001",
            "frame_index": 1,
            "point_scene_m": {
                "space": "scene_pseudo_m",
                "x": 0.0,
                "y": 0.0,
                "z": -0.7,
            },
            "radius_m": 0.7,
            "theta_radians": 0.0,
            "phi_radians": 0.0,
            "hemisphere": "front",
        }
    ]
