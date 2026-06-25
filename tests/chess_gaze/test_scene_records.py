from __future__ import annotations

import json
import math
from typing import Any

import pytest
from pydantic import ValidationError

from chess_gaze.geometry import CoordinateSpace, Point2D
from chess_gaze.scene_calibration import default_scene_assumptions
from chess_gaze.scene_records import (
    SceneAxisBasisRecord,
    SceneEyeMidpointRecord,
    SceneEyeRecord,
    SceneFrameRecord,
    SceneInvalidReason,
    SceneManifest,
    SceneMonitorHitRecord,
    SceneSummary,
    SceneUniGazeRayRecord,
    UnitVector3D,
    Vector3D,
    ViewerSceneData,
)


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


def _hit_payload(valid: bool = True) -> dict[str, Any]:
    return {
        "valid": valid,
        "point_camera_m": _vector_payload(x=0.2, y=0.1, z=1.0),
        "point_scene_m": _vector_payload(
            x=0.2,
            y=0.1,
            z=0.3,
            space="scene_pseudo_m",
        ),
        "u_m": 0.2,
        "v_m": 0.1,
        "t": 0.3,
        "denominator": -0.6,
        "signed_distance_m": 0.0,
        "within_physical_monitor": True,
        "within_extended_plane": True,
        "source_reason_invalid": None,
        "reason_invalid": None,
    }


def _axis_basis_payload() -> dict[str, Any]:
    return {
        "right_camera": _unit_vector_payload(x=1.0, y=0.0, z=0.0),
        "up_camera": _unit_vector_payload(x=0.0, y=-1.0, z=0.0),
        "back_camera": _unit_vector_payload(x=0.0, y=0.0, z=-1.0),
        "forward_camera": _unit_vector_payload(x=0.0, y=0.0, z=1.0),
        "determinant_right_up_back": 1.0,
        "convention": "right_up_back_columns_right_handed",
        "fallbacks": [],
    }


def _monitor_plane_payload() -> dict[str, Any]:
    return {
        "center_camera_m": _vector_payload(x=0.0, y=0.0, z=1.35),
        "center_scene_m": _vector_payload(
            x=0.0,
            y=0.0,
            z=0.7,
            space="scene_pseudo_m",
        ),
        "normal_camera": _unit_vector_payload(x=0.0, y=0.0, z=-1.0),
        "right_camera": _unit_vector_payload(x=1.0, y=0.0, z=0.0),
        "up_camera": _unit_vector_payload(x=0.0, y=-1.0, z=0.0),
        "width_m": 0.6,
        "height_m": 0.34,
        "extended_width_m": 1.8,
        "extended_height_m": 1.02,
        "distance_from_scene_center_m": 0.7,
        "distance_source": "DEFAULT_MONITOR_DISTANCE_FROM_EYES_M",
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
        "source_video_path": "artifacts/input/nakamura_1.mp4",
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
            "monitor_frame": "monitor_plane_pseudo_m",
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
                "inlier_frame_count": 1850,
                "fallback_used": False,
            },
            "main_unigaze_direction": {
                "method": "angular_ransac_then_normalized_inlier_mean",
                "candidate_frame_count": 1800,
                "inlier_frame_count": 1550,
                "inlier_angle_radians": 0.35,
                "fallback_used": False,
            },
            "scene_orientation": {
                "method": "eye_pair_right_and_head_up_with_camera_axis_fallbacks",
                "candidate_frame_count": 1850,
                "fallbacks": [],
            },
        },
        "scene_center_camera_m": _vector_payload(x=0.0, y=0.0, z=0.65),
        "scene_axes_camera": _axis_basis_payload(),
        "main_monitor_plane": _monitor_plane_payload(),
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
        "valid_monitor_hit_frames": 6,
        "invalid_monitor_hit_reasons": {"UNIGAZE_INVALID": 2},
        "monitor_hit_bounds": {
            "u_min_m": -0.42,
            "u_max_m": 0.38,
            "v_min_m": -0.21,
            "v_max_m": 0.19,
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
        "valid_for_main_monitor_direction": True,
        "camera": _frame_camera_payload(),
        "left_eye": _eye_payload(),
        "right_eye": _eye_payload(),
        "eye_midpoint": _midpoint_payload(),
        "head": _head_payload(),
        "unigaze_ray": _ray_payload(),
        "main_monitor_hit": _hit_payload(),
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
                "direction_scene": _unit_vector_payload(
                    space="camera_opencv_pseudo_m"
                ),
            }
        )

    with pytest.raises(ValidationError):
        SceneManifest.model_validate(
            {
                **_manifest_payload(),
                "main_monitor_plane": {
                    **_monitor_plane_payload(),
                    "normal_camera": _unit_vector_payload(space="scene_pseudo_m"),
                },
            }
        )

    with pytest.raises(ValidationError):
        ViewerSceneData.model_validate(
            {
                "run_id": "run-123",
                "source_video_stem": "nakamura_1",
                "frame_count": 1,
                "frames": [_scene_frame_payload()],
                "valid_hit_points": [
                    {
                        "frame_id": "frame-0001",
                        "frame_index": 1,
                        "point_scene_m": _vector_payload(
                            x=0.2,
                            y=0.1,
                            z=0.3,
                            space="camera_opencv_pseudo_m",
                        ),
                        "u_m": 0.2,
                        "v_m": 0.1,
                        "within_physical_monitor": True,
                        "within_extended_plane": True,
                    }
                ],
                "monitor_plane": _monitor_plane_payload(),
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
                    "valid_monitor_hit_frames": 1,
                    "invalid_monitor_hit_reasons": {},
                    "representative_scene_warning_frame_ids": [],
                },
            }
        )


def test_scene_axis_basis_rejects_parallel_back_and_negative_determinant() -> None:
    with pytest.raises(ValidationError):
        SceneAxisBasisRecord.model_validate(
            {
                **_axis_basis_payload(),
                "back_camera": _unit_vector_payload(x=0.0, y=0.0, z=1.0),
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


def test_valid_monitor_hit_requires_valid_ray_and_finite_forward_intersection() -> None:
    with pytest.raises(ValidationError):
        SceneMonitorHitRecord.model_validate(
            {
                **_hit_payload(),
                "u_m": math.nan,
            }
        )

    with pytest.raises(ValidationError):
        SceneMonitorHitRecord.model_validate(
            {
                **_hit_payload(),
                "t": -0.001,
            }
        )

    invalid_frame_payload = _scene_frame_payload()
    invalid_frame_payload["unigaze_ray"] = {
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
    }

    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(invalid_frame_payload)


def test_scene_monitor_hit_serializes_only_plane_uv_m() -> None:
    hit = SceneMonitorHitRecord.model_validate(_hit_payload())
    payload = hit.model_dump(by_alias=True)

    assert payload["plane_uv_m"] == (0.2, 0.1)
    assert "u_m" not in payload
    assert "v_m" not in payload


def test_invalid_nested_records_retain_explicit_scene_invalid_reasons() -> None:
    frame = SceneFrameRecord.model_validate(
        {
            **_scene_frame_payload(),
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
            "main_monitor_hit": {
                **_hit_payload(valid=False),
                "point_camera_m": None,
                "point_scene_m": None,
                "u_m": None,
                "v_m": None,
                "t": None,
                "denominator": None,
                "signed_distance_m": None,
                "within_physical_monitor": None,
                "within_extended_plane": None,
                "source_reason_invalid": "RAY_PARALLEL_TO_MONITOR",
                "reason_invalid": "RAY_PARALLEL_TO_MONITOR",
            },
        }
    )

    assert frame.left_eye.reason_invalid == SceneInvalidReason.LEFT_EYE_INVALID
    assert frame.right_eye.reason_invalid == SceneInvalidReason.RIGHT_EYE_INVALID
    assert frame.eye_midpoint.reason_invalid == SceneInvalidReason.EYE_MIDPOINT_INVALID
    assert frame.unigaze_ray.reason_invalid == SceneInvalidReason.UNIGAZE_INVALID
    assert (
        frame.main_monitor_hit.reason_invalid
        == SceneInvalidReason.RAY_PARALLEL_TO_MONITOR
    )
    assert frame.eye_midpoint.source_reason_invalid == "RIGHT_EYE_INVALID"
    assert frame.head.source_reason_invalid == "HEAD_POSE_INVALID"
    assert frame.unigaze_ray.source_reason_invalid == "GAZE_MODEL_FAILED"
    assert frame.main_monitor_hit.source_reason_invalid == "RAY_PARALLEL_TO_MONITOR"


def test_scene_frame_record_serializes_spec_alias_fields() -> None:
    frame = SceneFrameRecord.model_validate(_scene_frame_payload())
    payload = frame.model_dump(by_alias=True)

    assert payload["schema_version"] == "gaze-scene-frame-v1"
    assert payload["source_frame_status"] == "OK"
    assert payload["valid_for_scene_center"] is True
    assert payload["valid_for_main_monitor_direction"] is True
    assert payload["camera"]["depth_source"] == "interpupillary_distance_assumption"
    assert payload["left_eye"]["camera_m"]["space"] == "camera_opencv_pseudo_m"
    assert payload["eye_midpoint"]["camera_m"]["z"] == 0.8
    assert payload["eye_midpoint"]["scene_m"]["space"] == "scene_pseudo_m"
    assert payload["main_monitor_hit"]["ray_t_m"] == 0.3
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
    assert (
        SceneFrameRecord.model_validate_json(frame.model_dump_json(by_alias=True))
        .model_dump(by_alias=True)["head"]["ellipsoid_radii_m"]
        == (0.09, 0.12, 0.10)
    )


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

    assert payload["schema_version"] == "gaze-scene-manifest-v1"
    assert payload["camera_model"]["policy"] == "estimated_pinhole_from_image_size"
    assert (
        payload["source_artifacts"]["scene_frame_records"]
        == "records/scene_frames.jsonl"
    )
    assert payload["coordinate_frames"]["viewer_frame"] == "three_view"
    assert payload["scene_axes_camera"]["forward_camera"]["z"] == 1.0
    assert (
        payload["main_monitor_plane"]["distance_source"]
        == "DEFAULT_MONITOR_DISTANCE_FROM_EYES_M"
    )
    assert payload["viewer"]["version"] == "0.185.0"
    assert payload["generated_at_utc"] == "2026-06-26T12:00:00Z"


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
    manifest_payload["robust_estimators"]["scene_center"][
        "candidate_frame_count"
    ] = math.nan
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(manifest_payload)

    unknown_manifest_payload = _manifest_payload()
    unknown_manifest_payload["robust_estimators"]["scene_center"][
        "unexpected"
    ] = "value"
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


def test_scene_summary_serializes_structured_breakdowns() -> None:
    summary = SceneSummary.model_validate(_summary_payload())
    payload = summary.model_dump()

    assert payload["schema_version"] == "gaze-scene-summary-v1"
    assert payload["invalid_monitor_hit_reasons"] == {"UNIGAZE_INVALID": 2}
    assert payload["monitor_hit_bounds"]["u_min_m"] == -0.42
    assert payload["representative_scene_warning_frame_ids"] == ["frame-0002"]
    assert payload["artifact_validation"]["scene_manifest_valid"] is True


def test_viewer_scene_data_serializes_schema_version_and_hit_identities() -> None:
    viewer_data = ViewerSceneData.model_validate(
        {
            "run_id": "run-123",
            "source_video_stem": "nakamura_1",
            "frame_count": 1,
            "frames": [_scene_frame_payload()],
            "valid_hit_points": [
                {
                    "frame_id": "frame-0001",
                    "frame_index": 1,
                    "point_scene_m": _vector_payload(
                        x=0.2,
                        y=0.1,
                        z=0.3,
                        space="scene_pseudo_m",
                    ),
                    "u_m": 0.2,
                    "v_m": 0.1,
                    "within_physical_monitor": True,
                    "within_extended_plane": True,
                }
            ],
            "monitor_plane": _monitor_plane_payload(),
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
                "valid_monitor_hit_frames": 1,
                "invalid_monitor_hit_reasons": {},
                "representative_scene_warning_frame_ids": [],
            },
        }
    )

    payload = viewer_data.model_dump()

    assert payload["schema_version"] == "gaze-scene-viewer-data-v1"
    assert len(payload["frames"]) == 1
    assert payload["valid_hit_points"] == [
        {
            "frame_id": "frame-0001",
            "frame_index": 1,
            "point_scene_m": {
                "space": "scene_pseudo_m",
                "x": 0.2,
                "y": 0.1,
                "z": 0.3,
            },
            "u_m": 0.2,
            "v_m": 0.1,
            "within_physical_monitor": True,
            "within_extended_plane": True,
        }
    ]
