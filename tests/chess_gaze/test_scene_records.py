from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from chess_gaze.geometry import CoordinateSpace, Point2D
from chess_gaze.scene_calibration import default_scene_assumptions
from chess_gaze.scene_records import (
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
        "scene_point_m": _vector_payload(space="scene_pseudo_m"),
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
        "pupil_distance_px": 31.5,
        "estimated_depth_m": 0.8,
        "reason_invalid": None,
    }


def _head_payload(valid: bool = True) -> dict[str, Any]:
    return {
        "valid": valid,
        "ellipsoid_center_scene_m": _vector_payload(
            x=0.0,
            y=0.03,
            z=0.02,
            space="scene_pseudo_m",
        ),
        "radii_m": (0.09, 0.12, 0.10),
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
        "pitch_radians": 0.1,
        "yaw_radians": -0.1,
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
    }


def _scene_frame_payload() -> dict[str, Any]:
    return {
        "frame_id": "frame-0001",
        "frame_index": 1,
        "timestamp_seconds": 0.1,
        "left_eye": _eye_payload(),
        "right_eye": _eye_payload(),
        "eye_midpoint": _midpoint_payload(),
        "head": _head_payload(),
        "unigaze_ray": _ray_payload(),
        "main_monitor_hit": _hit_payload(),
        "diagnostics": {
            "source_frame_status": "OK",
            "left_eye_reason": None,
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
                "scene_point_m": None,
                "reason_invalid": "UNKNOWN_REASON",
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
                "scene_point_m": None,
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
        "scene_point_m": None,
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
        "pitch_radians": None,
        "yaw_radians": None,
        "reason_invalid": "UNIGAZE_INVALID",
    }

    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(invalid_frame_payload)


def test_invalid_nested_records_retain_explicit_scene_invalid_reasons() -> None:
    frame = SceneFrameRecord.model_validate(
        {
            **_scene_frame_payload(),
            "left_eye": {
                **_eye_payload(valid=False),
                "image_px": None,
                "camera_point_m": None,
                "scene_point_m": None,
                "source_reason_invalid": "LEFT_EYE_NOT_FOUND",
                "reason_invalid": "LEFT_EYE_INVALID",
            },
            "right_eye": {
                **_eye_payload(valid=False),
                "image_px": None,
                "camera_point_m": None,
                "scene_point_m": None,
                "source_reason_invalid": "RIGHT_EYE_NOT_FOUND",
                "reason_invalid": "RIGHT_EYE_INVALID",
            },
            "eye_midpoint": {
                **_midpoint_payload(valid=False),
                "camera_point_m": None,
                "scene_point_m": None,
                "pupil_distance_px": None,
                "estimated_depth_m": None,
                "reason_invalid": "EYE_MIDPOINT_INVALID",
            },
            "head": {
                **_head_payload(valid=False),
                "ellipsoid_center_scene_m": None,
                "reason_invalid": "NON_FINITE_INPUT",
            },
            "unigaze_ray": {
                **_ray_payload(valid=False),
                "origin_camera_m": None,
                "origin_scene_m": None,
                "direction_camera": None,
                "direction_scene": None,
                "pitch_radians": None,
                "yaw_radians": None,
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


def test_scene_frame_record_serializes_schema_version() -> None:
    frame = SceneFrameRecord.model_validate(_scene_frame_payload())

    assert frame.model_dump()["schema_version"] == "gaze-scene-frame-v1"


def test_scene_manifest_serializes_schema_version() -> None:
    manifest = SceneManifest.model_validate(
        {
            "run_id": "run-123",
            "source_video_path": "artifacts/input/nakamura_1.mp4",
            "source_video_sha256": "abc123",
            "camera_model": {
                "frame_width_px": 1920,
                "frame_height_px": 1080,
                "fx_px": 960.0,
                "fy_px": 960.0,
                "cx_px": 960.0,
                "cy_px": 540.0,
                "model": "estimated_pinhole_from_frame_size",
            },
            "assumptions": [
                record.model_dump() for record in default_scene_assumptions().records
            ],
            "scene_center_camera_m": _vector_payload(x=0.0, y=0.0, z=0.65),
            "axis_basis": _axis_basis_payload(),
            "monitor_plane": _monitor_plane_payload(),
            "robust_estimators": {"scene_center": {"candidate_count": 7}},
            "viewer_dependency": {"three_js_version": "0.185.0"},
        }
    )

    assert manifest.model_dump()["schema_version"] == "gaze-scene-manifest-v1"


def test_scene_summary_serializes_schema_version() -> None:
    summary = SceneSummary.model_validate(
        {
            "run_id": "run-123",
            "decoded_frames": 20,
            "scene_frame_records": 20,
            "valid_eye_midpoint_frames": 10,
            "valid_unigaze_ray_frames": 8,
            "valid_monitor_hit_frames": 6,
            "invalid_reason_counts": {"UNIGAZE_INVALID": 2},
            "representative_invalid_frame_ids": ["frame-0002"],
            "count_validation_passed": True,
        }
    )

    assert summary.model_dump()["schema_version"] == "gaze-scene-summary-v1"


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
                "run_id": "run-123",
                "decoded_frames": 1,
                "scene_frame_records": 1,
                "valid_eye_midpoint_frames": 1,
                "valid_unigaze_ray_frames": 1,
                "valid_monitor_hit_frames": 1,
                "invalid_reason_counts": {},
                "representative_invalid_frame_ids": [],
                "count_validation_passed": True,
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
