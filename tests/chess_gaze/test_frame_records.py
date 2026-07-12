import math
from typing import Any

import pytest
from pydantic import ValidationError

from chess_gaze.errors import ErrorCode
from chess_gaze.frame_records import (
    CropImageRetentionPolicy,
    FrameImageRetentionPolicy,
    FrameRecord,
    GazeAngles,
    InferenceRuntimeRecord,
    QASummaryPolicy,
    RunManifest,
    VideoManifest,
    read_run_manifest_artifact_json,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D


@pytest.fixture
def valid_frame_record_dict() -> dict[str, Any]:
    return {
        "frame_id": "f000000001",
        "frame_index": 1,
        "status": "ERROR",
        "timestamp_seconds": 0.0,
        "face": {
            "present": False,
            "bounding_box": None,
            "landmarks": None,
            "reason_invalid": "FACE_NOT_FOUND",
        },
        "left_eye": {
            "present": False,
            "bounding_box": None,
            "pupil_center": None,
            "iris_landmarks": None,
            "reason_invalid": "LEFT_EYE_NOT_FOUND",
        },
        "right_eye": {
            "present": False,
            "bounding_box": None,
            "pupil_center": None,
            "iris_landmarks": None,
            "reason_invalid": "RIGHT_EYE_NOT_FOUND",
        },
        "head_pose": {
            "valid": False,
            "yaw_radians": None,
            "pitch_radians": None,
            "roll_radians": None,
            "reason_invalid": "HEAD_POSE_INVALID",
        },
        "geometric_gaze": {
            "valid": False,
            "yaw_radians": None,
            "pitch_radians": None,
            "reason_invalid": "GAZE_ESTIMATORS_DISAGREE",
        },
        "appearance_gaze": {
            "valid": False,
            "yaw_radians": None,
            "pitch_radians": None,
            "reason_invalid": "GAZE_MODEL_FAILED",
        },
        "recommended_gaze": {
            "valid": False,
            "yaw_radians": None,
            "pitch_radians": None,
            "reason_invalid": "GAZE_ESTIMATORS_DISAGREE",
        },
        "errors": [
            {
                "code": "FACE_NOT_FOUND",
                "message": "No face detected in frame.",
            }
        ],
    }


def test_bbox_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValidationError):
        BBox(space=CoordinateSpace.IMAGE_PX, x_min=20, y_min=10, x_max=10, y_max=40)


def test_point_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        Point2D(space=CoordinateSpace.IMAGE_PX, x=math.nan, y=1.0)


def test_gaze_valid_requires_pitch_and_yaw() -> None:
    with pytest.raises(ValidationError):
        GazeAngles(valid=True, yaw_radians=None, pitch_radians=0.1, reason_invalid=None)


def test_gaze_angles_reject_enum_strings_in_direct_validation() -> None:
    invalid_reason: Any = "GAZE_MODEL_FAILED"

    with pytest.raises(ValidationError):
        GazeAngles(
            valid=False,
            yaw_radians=None,
            pitch_radians=None,
            reason_invalid=invalid_reason,
        )


def test_frame_record_accepts_valid_artifact_payload(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    record = FrameRecord.model_validate(valid_frame_record_dict)

    assert record.status.value == "ERROR"
    assert record.face.reason_invalid == ErrorCode.FACE_NOT_FOUND
    assert record.left_eye.reason_invalid == ErrorCode.LEFT_EYE_NOT_FOUND


def test_frame_record_rejects_unknown_fields(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["unknown"] = "rejected"

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_near_miss_enum_strings(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["status"] = "NOT_A_STATUS"
    valid_frame_record_dict["recommended_gaze"]["reason_invalid"] = "GAZE_MODEL_FAILED "

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_invalid_nested_enum_string(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["recommended_gaze"]["reason_invalid"] = "NOT_A_REASON"

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_present_face_without_landmarks(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["face"]["present"] = True
    valid_frame_record_dict["face"]["reason_invalid"] = None

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_present_eye_without_landmarks(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["left_eye"]["present"] = True
    valid_frame_record_dict["left_eye"]["reason_invalid"] = None

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_frame_record_rejects_infinite_head_pose_angle(
    valid_frame_record_dict: dict[str, Any],
) -> None:
    valid_frame_record_dict["head_pose"]["yaw_radians"] = math.inf

    with pytest.raises(ValidationError):
        FrameRecord.model_validate(valid_frame_record_dict)


def test_error_code_names_are_stable() -> None:
    assert ErrorCode.FACE_NOT_FOUND.value == "FACE_NOT_FOUND"
    assert ErrorCode.GAZE_ESTIMATORS_DISAGREE.value == "GAZE_ESTIMATORS_DISAGREE"


def _default_model_inference_payload() -> dict[str, Any]:
    return {
        "observer_source": "default_model_observer",
        "unigaze_model_id": "unigaze-h14-joint",
        "unigaze_device": "mps",
        "unigaze_batch_size": 16,
        "torch_version": "2.12.1",
        "torch_mps_available": True,
        "mps_fallback_env": "unset",
        "mps_fast_math_env": "unset",
        "mps_prefer_metal_env": "unset",
        "mps_preflight_passed": True,
    }


def _external_observer_inference_payload() -> dict[str, Any]:
    return {
        "observer_source": "external_observer",
        "unigaze_model_id": None,
        "unigaze_device": "not_applicable",
        "unigaze_batch_size": None,
        "torch_version": None,
        "torch_mps_available": None,
        "mps_fallback_env": "not_applicable",
        "mps_fast_math_env": "not_applicable",
        "mps_prefer_metal_env": "not_applicable",
        "mps_preflight_passed": None,
    }


def test_inference_runtime_record_accepts_default_model_observer() -> None:
    record = InferenceRuntimeRecord(**_default_model_inference_payload())

    assert record.schema_version == "inference-runtime-v1"
    assert record.unigaze_device == "mps"
    assert record.unigaze_batch_size == 16


def test_inference_runtime_record_accepts_current_cpu_default_model_runtime() -> None:
    payload = _default_model_inference_payload()
    payload.update(
        {
            "unigaze_device": "cpu",
            "unigaze_batch_size": 1,
            "torch_mps_available": False,
            "mps_preflight_passed": None,
        }
    )

    record = InferenceRuntimeRecord(**payload)

    assert record.unigaze_device == "cpu"
    assert record.unigaze_batch_size == 1
    assert record.mps_preflight_passed is None


def test_inference_runtime_record_accepts_external_observer() -> None:
    record = InferenceRuntimeRecord(**_external_observer_inference_payload())

    assert record.observer_source == "external_observer"
    assert record.unigaze_model_id is None


def test_provenance_fields_default_for_legacy_payloads() -> None:
    legacy_video = VideoManifest.model_validate(
        {
            "source_path": "artifacts/input/nakamura_short.mp4",
            "source_sha256": "0" * 64,
            "frame_width": 1920,
            "frame_height": 1080,
            "frame_count_decoded": 180,
        }
    )
    legacy_inference = InferenceRuntimeRecord.model_validate(
        _default_model_inference_payload()
    )

    assert legacy_video.pts_sequence_sha256 is None
    assert legacy_video.pts_sequence_usable is False
    assert legacy_inference.unigaze_model_checksum_sha256 is None


@pytest.mark.parametrize(
    "observer_source",
    ["external_observer", "legacy_manifest_without_inference"],
)
def test_model_free_inference_records_reject_model_checksum(
    observer_source: str,
) -> None:
    payload = _external_observer_inference_payload()
    payload["observer_source"] = observer_source
    payload["unigaze_model_checksum_sha256"] = "abc123"

    with pytest.raises(ValidationError, match=observer_source):
        InferenceRuntimeRecord(**payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unigaze_model_id": None},
        {"unigaze_device": "not_applicable"},
        {"unigaze_batch_size": None},
        {"unigaze_batch_size": 0},
        {"unigaze_batch_size": -1},
        {"torch_version": None},
        {"torch_mps_available": None},
        {"mps_fallback_env": "not_applicable"},
        {"mps_preflight_passed": None},
    ],
)
def test_inference_runtime_record_rejects_default_model_observer_contradictions(
    overrides: dict[str, Any],
) -> None:
    payload = _default_model_inference_payload()
    payload.update(overrides)

    with pytest.raises(ValidationError, match="default_model_observer"):
        InferenceRuntimeRecord(**payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unigaze_device": "cpu", "mps_preflight_passed": False},
        {"unigaze_device": "cpu", "mps_preflight_passed": True},
        {
            "unigaze_device": "mps",
            "torch_mps_available": False,
            "mps_preflight_passed": True,
        },
        {
            "unigaze_device": "mps",
            "torch_mps_available": True,
            "mps_preflight_passed": False,
        },
    ],
)
def test_inference_runtime_record_rejects_default_model_runtime_device_contradictions(
    overrides: dict[str, Any],
) -> None:
    payload = _default_model_inference_payload()
    payload.update(overrides)

    with pytest.raises(ValidationError, match="default_model_observer"):
        InferenceRuntimeRecord(**payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unigaze_model_id": "unigaze-h14-joint"},
        {"unigaze_device": "cpu"},
        {"unigaze_batch_size": 1},
        {"torch_version": "2.12.1"},
        {"torch_mps_available": False},
        {"mps_fallback_env": "unset"},
        {"mps_preflight_passed": False},
    ],
)
def test_inference_runtime_record_rejects_external_observer_contradictions(
    overrides: dict[str, Any],
) -> None:
    payload = _external_observer_inference_payload()
    payload.update(overrides)

    with pytest.raises(ValidationError, match="external_observer"):
        InferenceRuntimeRecord(**payload)


def test_run_manifest_requires_inference_runtime_record() -> None:
    manifest = RunManifest(
        run_id="run-1",
        created_at_utc="2026-06-26T00:00:00Z",
        input_path="artifacts/input/nakamura_short.mp4",
        video=VideoManifest(
            source_path="artifacts/input/nakamura_short.mp4",
            source_sha256="0" * 64,
            frame_width=1920,
            frame_height=1080,
            frame_count_decoded=180,
        ),
        inference=InferenceRuntimeRecord(**_external_observer_inference_payload()),
    )

    assert manifest.inference.observer_source == "external_observer"


def test_run_manifest_defaults_missing_frame_image_retention_to_legacy_save() -> None:
    manifest = RunManifest.model_validate(
        {
            "run_id": "run-1",
            "created_at_utc": "2026-06-26T00:00:00Z",
            "input_path": "artifacts/input/nakamura_short.mp4",
            "video": {
                "source_path": "artifacts/input/nakamura_short.mp4",
                "source_sha256": "0" * 64,
                "frame_width": 1920,
                "frame_height": 1080,
                "frame_count_decoded": 180,
            },
            "inference": _external_observer_inference_payload(),
        }
    )

    assert manifest.frame_image_retention == FrameImageRetentionPolicy(
        save_frame_images=True
    )
    assert manifest.crop_image_retention == CropImageRetentionPolicy(
        save_crop_images=True
    )


def test_run_manifest_records_explicit_no_qa_summary_policy() -> None:
    manifest = RunManifest(
        run_id="run-1",
        created_at_utc="2026-06-26T00:00:00Z",
        input_path="artifacts/input/nakamura_short.mp4",
        video=VideoManifest(
            source_path="artifacts/input/nakamura_short.mp4",
            source_sha256="0" * 64,
            frame_width=1920,
            frame_height=1080,
            frame_count_decoded=180,
        ),
        inference=InferenceRuntimeRecord(**_external_observer_inference_payload()),
        qa_summary_policy=QASummaryPolicy(generate_qa_summary=False),
    )

    assert manifest.qa_summary_policy.generate_qa_summary is False
    assert manifest.model_dump(mode="json")["qa_summary_policy"] == {
        "schema_version": "qa-summary-policy-v1",
        "generate_qa_summary": False,
    }


def test_run_manifest_defaults_missing_qa_summary_policy_to_legacy_generate() -> None:
    manifest = RunManifest.model_validate(
        {
            "run_id": "run-1",
            "created_at_utc": "2026-06-26T00:00:00Z",
            "input_path": "artifacts/input/nakamura_short.mp4",
            "video": {
                "source_path": "artifacts/input/nakamura_short.mp4",
                "source_sha256": "0" * 64,
                "frame_width": 1920,
                "frame_height": 1080,
                "frame_count_decoded": 180,
            },
            "inference": _external_observer_inference_payload(),
        }
    )

    assert manifest.qa_summary_policy.generate_qa_summary is True


def test_run_manifest_direct_validation_rejects_missing_inference() -> None:
    with pytest.raises(ValidationError, match="inference"):
        RunManifest.model_validate(
            {
                "run_id": "run-1",
                "created_at_utc": "2026-06-26T00:00:00Z",
                "input_path": "artifacts/input/nakamura_short.mp4",
                "video": {
                    "source_path": "artifacts/input/nakamura_short.mp4",
                    "source_sha256": "0" * 64,
                    "frame_width": 1920,
                    "frame_height": 1080,
                    "frame_count_decoded": 180,
                },
            }
        )


def test_legacy_manifest_json_gets_compatibility_inference() -> None:
    manifest = read_run_manifest_artifact_json(
        """
        {
          "run_id": "run-1",
          "created_at_utc": "2026-06-26T00:00:00Z",
          "input_path": "artifacts/input/nakamura_short.mp4",
          "video": {
            "source_path": "artifacts/input/nakamura_short.mp4",
            "source_sha256": "%s",
            "frame_width": 1920,
            "frame_height": 1080,
            "frame_count_decoded": 180
          }
        }
        """
        % ("0" * 64)
    )

    assert manifest.inference.observer_source == "legacy_manifest_without_inference"
    assert manifest.inference.unigaze_device == "not_applicable"
    assert manifest.inference.mps_preflight_passed is None
