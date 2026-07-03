from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.calibration import default_calibration
from chess_gaze.errors import CliErrorCode, ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    CropImageRetentionPolicy,
    FrameImageRetentionPolicy,
    FrameRecord,
    InferenceRuntimeRecord,
    RunManifest,
    VideoManifest,
    read_run_manifest_artifact_json,
)
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.image_io import save_rgb_png
from chess_gaze.qa_summary import (
    build_qa_summary,
    validate_run_artifacts,
)
from chess_gaze.scene_artifacts import build_scene_artifacts, build_viewer_scene_data
from chess_gaze.scene_records import SceneSummary, ViewerSceneData


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


def _record(frame_index: int, *, status: FrameStatus = FrameStatus.OK) -> FrameRecord:
    frame_id = f"f{frame_index:09d}"
    face_present = frame_index % 4 != 1
    left_eye_present = frame_index % 5 != 2
    right_eye_present = frame_index % 6 != 3
    head_pose_valid = face_present and frame_index % 7 != 4
    face_gaze_valid = face_present and frame_index % 8 != 5
    recommended_gaze_valid = (
        face_gaze_valid
        and left_eye_present
        and right_eye_present
        and frame_index % 9 != 6
    )
    errors: list[dict[str, object]] = []

    if status is FrameStatus.ERROR:
        face_present = False
        left_eye_present = False
        right_eye_present = False
        head_pose_valid = False
        face_gaze_valid = False
        recommended_gaze_valid = False
        errors.extend(
            [
                {
                    "code": ErrorCode.FACE_NOT_FOUND,
                    "message": "No face detected in frame.",
                },
                {
                    "code": ErrorCode.HEAD_POSE_INVALID,
                    "message": "Head pose unavailable without a face.",
                },
            ]
        )

    gaze = {
        "valid": face_gaze_valid,
        "yaw_radians": 0.01 * frame_index if face_gaze_valid else None,
        "pitch_radians": -0.01 * frame_index if face_gaze_valid else None,
        "reason_invalid": None if face_gaze_valid else ErrorCode.GAZE_MODEL_FAILED,
    }
    recommended_gaze = {
        "valid": recommended_gaze_valid,
        "yaw_radians": 0.01 * frame_index if recommended_gaze_valid else None,
        "pitch_radians": -0.01 * frame_index if recommended_gaze_valid else None,
        "reason_invalid": (
            None if recommended_gaze_valid else ErrorCode.GAZE_ESTIMATORS_DISAGREE
        ),
    }

    return FrameRecord.model_validate(
        {
            "frame_id": frame_id,
            "frame_index": frame_index,
            "status": status,
            "timestamp_seconds": frame_index / 30.0,
            "face": {
                "present": face_present,
                "bounding_box": _box(10.0, 10.0, 30.0, 30.0) if face_present else None,
                "landmarks": [_point(15.0, 15.0), _point(25.0, 15.0)]
                if face_present
                else None,
                "reason_invalid": None if face_present else ErrorCode.FACE_NOT_FOUND,
            },
            "left_eye": _eye_payload(left_eye_present, ErrorCode.LEFT_EYE_NOT_FOUND),
            "right_eye": _eye_payload(right_eye_present, ErrorCode.RIGHT_EYE_NOT_FOUND),
            "head_pose": {
                "valid": head_pose_valid,
                "yaw_radians": 0.01 if head_pose_valid else None,
                "pitch_radians": 0.02 if head_pose_valid else None,
                "roll_radians": 0.03 if head_pose_valid else None,
                "reason_invalid": (
                    None if head_pose_valid else ErrorCode.HEAD_POSE_INVALID
                ),
            },
            "geometric_gaze": recommended_gaze,
            "appearance_gaze": gaze,
            "recommended_gaze": recommended_gaze,
            "errors": errors,
        }
    )


def _eye_payload(present: bool, reason: ErrorCode) -> dict[str, object]:
    if not present:
        return {
            "present": False,
            "bounding_box": None,
            "pupil_center": None,
            "iris_landmarks": None,
            "reason_invalid": reason,
        }
    return {
        "present": True,
        "bounding_box": _box(12.0, 12.0, 22.0, 18.0),
        "pupil_center": _point(17.0, 15.0),
        "iris_landmarks": [
            _point(15.0, 15.0),
            _point(19.0, 15.0),
            _point(17.0, 13.0),
            _point(17.0, 17.0),
        ],
        "reason_invalid": None,
    }


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


def _write_fixture_run(
    tmp_path: Path,
    frame_count: int = 35,
    *,
    save_frame_images: bool = True,
    write_frame_images: bool = True,
    save_crop_images: bool = True,
    write_crop_images: bool = True,
) -> RunLayout:
    layout = _make_layout(tmp_path)
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"source-video")

    run_manifest = RunManifest(
        run_id=layout.run_dir.name,
        created_at_utc="2026-06-25T12:00:00Z",
        input_path=str(video_path),
        video=VideoManifest(
            source_path=str(video_path),
            source_sha256="0" * 64,
            frame_width=8,
            frame_height=8,
            frame_count_decoded=frame_count,
        ),
        inference=_external_observer_inference_record(),
        frame_image_retention=FrameImageRetentionPolicy(
            save_frame_images=save_frame_images
        ),
        crop_image_retention=CropImageRetentionPolicy(
            save_crop_images=save_crop_images
        ),
    )
    (layout.run_dir / "run_manifest.json").write_text(
        run_manifest.model_dump_json(), encoding="utf-8"
    )
    (layout.run_dir / "calibration.json").write_text(
        default_calibration().model_dump_json(), encoding="utf-8"
    )
    (layout.run_dir / "video_manifest.json").write_text(
        run_manifest.video.model_dump_json(), encoding="utf-8"
    )

    frame_records: list[FrameRecord] = []
    for frame_index in range(frame_count):
        status = FrameStatus.ERROR if frame_index in {4, 17, 31} else FrameStatus.OK
        record = _record(frame_index, status=status)
        frame_records.append(record)
        if write_frame_images:
            pixel_value = frame_index * 7 % 255
            image = _solid_rgb(pixel_value)
            save_rgb_png(layout.raw_frames_dir / f"{record.frame_id}.png", image)
            (layout.processed_frames_dir / f"{record.frame_id}.jpg").write_bytes(
                b"processed" + bytes([frame_index])
            )
    if write_crop_images:
        (layout.face_crops_dir / "f000000000.png").write_bytes(b"face-crop")
        (layout.left_eye_crops_dir / "f000000000.png").write_bytes(b"left-crop")
        (layout.right_eye_crops_dir / "f000000000.png").write_bytes(b"right-crop")

    (layout.records_dir / "frames.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in frame_records),
        encoding="utf-8",
    )
    error_lines = [
        {
            "frame_id": "f000000004",
            "frame_index": 4,
            "code": ErrorCode.FACE_NOT_FOUND.value,
            "message": "No face detected in frame.",
        },
        {
            "frame_id": "f000000004",
            "frame_index": 4,
            "code": ErrorCode.RAW_FRAME_WRITE_FAILED.value,
            "message": "Raw frame write failed.",
        },
        {
            "frame_id": "f000000017",
            "frame_index": 17,
            "code": ErrorCode.FACE_NOT_FOUND.value,
            "message": "No face detected in frame.",
        },
        {
            "frame_id": "f000000018",
            "frame_index": 18,
            "code": ErrorCode.PROCESSED_FRAME_WRITE_FAILED.value,
            "message": "Processed frame write failed.",
        },
    ]
    (layout.records_dir / "errors.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in error_lines),
        encoding="utf-8",
    )
    _write_scene_and_viewer_artifacts(layout)
    return layout


def test_build_qa_summary_reads_legacy_run_manifest_without_inference(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path)
    run_manifest_path = layout.run_dir / "run_manifest.json"
    legacy_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    legacy_manifest.pop("inference")
    run_manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")

    summary = build_qa_summary(layout)
    manifest = read_run_manifest_artifact_json(
        run_manifest_path.read_text(encoding="utf-8")
    )

    assert summary.run_id == layout.run_dir.name
    assert summary.source_video_path == str(tmp_path / "source.mp4")
    assert summary.artifact_validation.schema_validation_passed is True
    assert manifest.inference.observer_source == "legacy_manifest_without_inference"


def test_build_qa_summary_streams_large_artifacts_without_whole_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=7)
    blocked_names = {
        "frames.jsonl",
        "errors.jsonl",
        "scene_frames.jsonl",
        "scene-data.json",
    }
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name in blocked_names:
            raise AssertionError(f"whole-file read forbidden for {path.name}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    summary = build_qa_summary(layout)

    assert summary.final_status == "complete"
    assert summary.counts.frame_records == 7
    assert summary.counts.scene_frame_records == 7
    assert summary.artifact_validation.validation_errors == []


def _make_layout(tmp_path: Path) -> RunLayout:
    run_dir = tmp_path / "runs" / "20260625T120000Z-test"
    raw_frames_dir = run_dir / "raw_frames"
    processed_frames_dir = run_dir / "processed_frames"
    crops_dir = run_dir / "crops"
    face_crops_dir = crops_dir / "face"
    eyes_crops_dir = crops_dir / "eyes"
    left_eye_crops_dir = eyes_crops_dir / "left"
    right_eye_crops_dir = eyes_crops_dir / "right"
    records_dir = run_dir / "records"
    scene_dir = run_dir / "scene"
    viewer_dir = run_dir / "viewer"
    for directory in (
        raw_frames_dir,
        processed_frames_dir,
        face_crops_dir,
        left_eye_crops_dir,
        right_eye_crops_dir,
        records_dir,
        scene_dir,
        viewer_dir,
    ):
        directory.mkdir(parents=True)
    return RunLayout(
        run_dir=run_dir,
        raw_frames_dir=raw_frames_dir,
        processed_frames_dir=processed_frames_dir,
        crops_dir=crops_dir,
        face_crops_dir=face_crops_dir,
        eyes_crops_dir=eyes_crops_dir,
        left_eye_crops_dir=left_eye_crops_dir,
        right_eye_crops_dir=right_eye_crops_dir,
        records_dir=records_dir,
    )


def _write_scene_and_viewer_artifacts(layout: RunLayout) -> None:
    scene_result = build_scene_artifacts(layout)
    viewer_data = build_viewer_scene_data(scene_result)
    (layout.viewer_dir / "scene-data.json").write_text(
        json.dumps(
            viewer_data.model_dump(mode="json", by_alias=True),
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (layout.viewer_dir / "index.html").write_text(
        "<!doctype html><title>Chess Gaze Scene Viewer</title>\n",
        encoding="utf-8",
    )


def _solid_rgb(pixel_value: int) -> np.ndarray:
    return np.full((8, 8, 3), pixel_value, dtype=np.uint8)


def test_build_qa_summary_revalidates_counts_rates_errors_and_samples(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path)

    summary = build_qa_summary(layout)

    assert summary.schema_version == "qa-summary-v1"
    assert summary.run_id == "20260625T120000Z-test"
    assert summary.counts.decoded_frames == 35
    assert summary.counts.frame_records == 35
    assert summary.counts.raw_frames == 35
    assert summary.counts.processed_frames == 35
    assert summary.byte_counts.raw_frames_bytes > 0
    assert summary.byte_counts.processed_frames_bytes > 0
    assert summary.byte_counts.crops_bytes > 0
    assert summary.byte_counts.jsonl_bytes > 0
    assert summary.byte_counts.total_run_bytes >= (
        summary.byte_counts.raw_frames_bytes
        + summary.byte_counts.processed_frames_bytes
        + summary.byte_counts.crops_bytes
        + summary.byte_counts.jsonl_bytes
    )
    assert summary.rates.face_present_rate == pytest.approx(24 / 35)
    assert summary.rates.both_eyes_present_rate == pytest.approx(21 / 35)
    assert summary.rates.left_eye_only_rate == pytest.approx(5 / 35)
    assert summary.rates.right_eye_only_rate == pytest.approx(5 / 35)
    assert summary.rates.left_iris_present_rate == pytest.approx(26 / 35)
    assert summary.rates.right_iris_present_rate == pytest.approx(26 / 35)
    assert summary.rates.head_pose_valid_rate == pytest.approx(21 / 35)
    assert summary.rates.face_gaze_valid_rate == pytest.approx(24 / 35)
    assert summary.rates.recommended_gaze_valid_rate == pytest.approx(14 / 35)
    assert summary.errors_by_code == {
        ErrorCode.FACE_NOT_FOUND.value: 2,
        ErrorCode.PROCESSED_FRAME_WRITE_FAILED.value: 1,
        ErrorCode.RAW_FRAME_WRITE_FAILED.value: 1,
    }
    assert summary.errors_by_severity == {"error": 2, "warning": 2}
    assert summary.worst_blur_frame_ids == sorted(summary.worst_blur_frame_ids)
    assert summary.worst_exposure_frame_ids[:3] == [
        "f000000000",
        "f000000001",
        "f000000002",
    ]
    assert len(summary.qa_sample_frame_ids) == 30
    assert summary.qa_sample_frame_ids == [
        "f000000000",
        "f000000001",
        "f000000002",
        "f000000004",
        "f000000005",
        "f000000006",
        "f000000007",
        "f000000008",
        "f000000009",
        "f000000011",
        "f000000012",
        "f000000013",
        "f000000014",
        "f000000015",
        "f000000016",
        "f000000018",
        "f000000019",
        "f000000020",
        "f000000021",
        "f000000022",
        "f000000023",
        "f000000025",
        "f000000026",
        "f000000027",
        "f000000028",
        "f000000029",
        "f000000030",
        "f000000032",
        "f000000033",
        "f000000034",
    ]
    assert summary.representative_failure_frame_ids == [
        "f000000004",
        "f000000017",
        "f000000018",
        "f000000031",
    ]
    assert summary.status_transitions == [
        "created",
        "processing",
        "revalidating",
        "complete",
    ]
    assert summary.final_status == "complete"
    assert summary.disk_space.preflight_estimate_bytes > 0
    assert summary.disk_space.closeout_free_bytes > 0
    assert summary.artifact_validation.schema_validation_passed is True
    assert summary.artifact_validation.counts_match is True
    assert summary.built_from_disk_at_utc.endswith("Z")
    datetime.fromisoformat(summary.built_from_disk_at_utc.replace("Z", "+00:00"))


def test_qa_summary_validates_scene_artifacts_and_counts_scene_bytes(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=5)
    _write_scene_and_viewer_artifacts(layout)

    validation = validate_run_artifacts(layout)
    summary = build_qa_summary(layout)
    scene_summary_json = (layout.scene_dir / "scene_summary.json").read_text(
        encoding="utf-8"
    )
    viewer_data = ViewerSceneData.model_validate_json(
        (layout.viewer_dir / "scene-data.json").read_text(encoding="utf-8")
    )
    scene_summary = SceneSummary.model_validate_json(scene_summary_json)

    assert summary.source_artifacts == validation.source_artifacts
    assert summary.source_artifacts == summary.artifact_validation.source_artifacts
    assert {
        "scene_manifest",
        "scene_summary",
        "scene_frames_jsonl",
        "viewer_index",
        "viewer_scene_data",
    }.issubset(summary.source_artifacts)
    assert summary.counts.decoded_frames == 5
    assert summary.counts.frame_records == 5
    assert summary.counts.scene_frame_records == 5
    assert validation.counts.scene_frame_records == 5
    assert viewer_data.frame_count == 5
    assert scene_summary.valid_sphere_hit_frames >= 0
    assert "valid_sphere_hit_frames" in scene_summary_json
    assert "valid_monitor_hit_frames" not in scene_summary_json
    scene_frame_lines = (
        (layout.records_dir / "scene_frames.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [json.loads(line)["frame_index"] for line in scene_frame_lines] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert (
        summary.byte_counts.scene_jsonl_bytes
        == (layout.records_dir / "scene_frames.jsonl").stat().st_size
    )
    assert summary.byte_counts.scene_bytes == sum(
        path.stat().st_size for path in layout.scene_dir.rglob("*") if path.is_file()
    )
    assert summary.byte_counts.viewer_bytes == sum(
        path.stat().st_size for path in layout.viewer_dir.rglob("*") if path.is_file()
    )
    assert summary.byte_counts.total_run_bytes == sum(
        path.stat().st_size for path in layout.run_dir.rglob("*") if path.is_file()
    )
    assert summary.artifact_validation.schema_validation_passed is True
    assert summary.artifact_validation.counts_match is True


def test_qa_summary_reports_missing_or_malformed_scene_artifacts(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=3)
    _write_scene_and_viewer_artifacts(layout)
    (layout.scene_dir / "scene_manifest.json").unlink()
    scene_frame_lines = (
        (layout.records_dir / "scene_frames.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    (layout.records_dir / "scene_frames.jsonl").write_text(
        scene_frame_lines[1] + "\n",
        encoding="utf-8",
    )
    (layout.viewer_dir / "scene-data.json").write_text(
        "{malformed-json\n",
        encoding="utf-8",
    )

    validation = validate_run_artifacts(layout)
    summary = build_qa_summary(layout)

    assert validation.schema_validation_passed is False
    assert validation.counts_match is False
    assert validation.final_status == "failed"
    assert summary.final_status == "failed"
    assert summary.artifact_validation.validation_errors == validation.validation_errors
    assert any("scene manifest" in error for error in validation.validation_errors)
    assert any(
        "Invalid viewer scene data" in error for error in validation.validation_errors
    )
    assert any(
        "scene frame record count does not match decoded frame count: 1 != 3" in error
        for error in validation.validation_errors
    )
    assert any(
        "scene frame records are not contiguous from decoded frame zero" in error
        for error in validation.validation_errors
    )


def test_qa_summary_rejects_unexpected_viewer_scene_data_top_level_key(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=3)
    viewer_path = layout.viewer_dir / "scene-data.json"
    payload = json.loads(viewer_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    viewer_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_qa_summary(layout)

    assert summary.final_status == "failed"
    assert summary.artifact_validation.schema_validation_passed is False
    assert any(
        "unexpected top-level key" in error
        for error in summary.artifact_validation.validation_errors
    )


def test_qa_summary_rejects_malformed_viewer_hit_points(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path)
    viewer_path = layout.viewer_dir / "scene-data.json"
    payload = json.loads(viewer_path.read_text(encoding="utf-8"))
    payload["valid_hit_points"] = [
        {
            "frame_id": "f000000000",
            "frame_index": 0,
            "point_scene_m": {
                "space": "scene_pseudo_m",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
            "radius_m": -1.0,
            "theta_radians": 0.0,
            "phi_radians": 0.0,
            "hemisphere": "front",
        }
    ]
    viewer_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_qa_summary(layout)

    assert summary.final_status == "failed"
    assert summary.artifact_validation.schema_validation_passed is False
    assert any(
        "viewer hit point" in error
        for error in summary.artifact_validation.validation_errors
    )


def test_qa_summary_rejects_malformed_viewer_frames(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=3)
    viewer_path = layout.viewer_dir / "scene-data.json"
    payload = json.loads(viewer_path.read_text(encoding="utf-8"))
    payload["frames"][0]["frame_index"] = "not-an-int"
    viewer_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_qa_summary(layout)

    assert summary.final_status == "failed"
    assert summary.artifact_validation.schema_validation_passed is False
    assert any(
        "viewer frame" in error
        for error in summary.artifact_validation.validation_errors
    )


def test_build_qa_summary_does_not_treat_warning_only_records_as_failures(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=2)
    warning_payload = _record(0).model_dump(mode="python")
    warning_payload["status"] = FrameStatus.WARNING
    warning_payload["recommended_gaze"] = {
        "valid": False,
        "yaw_radians": None,
        "pitch_radians": None,
        "reason_invalid": ErrorCode.GAZE_ESTIMATORS_DISAGREE,
    }
    warning_payload["errors"] = [
        {
            "code": ErrorCode.GAZE_ESTIMATORS_DISAGREE,
            "message": "Recommended gaze is invalid: GAZE_ESTIMATORS_DISAGREE.",
        }
    ]
    warning_record = FrameRecord.model_validate(warning_payload)
    hard_failure_record = _record(1, status=FrameStatus.ERROR)
    (layout.records_dir / "frames.jsonl").write_text(
        warning_record.model_dump_json()
        + "\n"
        + hard_failure_record.model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (layout.records_dir / "errors.jsonl").write_text(
        json.dumps(
            {
                "frame_id": warning_record.frame_id,
                "frame_index": warning_record.frame_index,
                "code": ErrorCode.GAZE_ESTIMATORS_DISAGREE.value,
                "message": "Recommended gaze is invalid: GAZE_ESTIMATORS_DISAGREE.",
            }
        )
        + "\n"
        + json.dumps(
            {
                "frame_id": hard_failure_record.frame_id,
                "frame_index": hard_failure_record.frame_index,
                "code": ErrorCode.FACE_NOT_FOUND.value,
                "message": "No face detected in frame.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_qa_summary(layout)

    assert summary.errors_by_severity == {"warning": 2}
    assert summary.representative_failure_frame_ids == ["f000000001"]


def test_validate_run_artifacts_reports_count_mismatches_without_hiding_records(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=3)
    (layout.raw_frames_dir / "f000000002.png").unlink()

    result = validate_run_artifacts(layout)

    assert result.schema_validation_passed is True
    assert result.counts_match is False
    assert result.final_status == "failed"
    assert result.validation_errors == [
        "raw frame count does not match decoded frame count: 2 != 3"
    ]
    assert result.counts.decoded_frames == 3
    assert result.counts.raw_frames == 2


def test_validate_run_artifacts_accepts_unretained_frame_images_when_disabled(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(
        tmp_path,
        frame_count=3,
        save_frame_images=False,
        write_frame_images=False,
    )

    result = validate_run_artifacts(layout)
    summary = build_qa_summary(layout)

    assert result.counts.raw_frames == 0
    assert result.counts.processed_frames == 0
    assert result.counts_match is True
    assert summary.final_status == "complete"


def test_validate_run_artifacts_rejects_stray_frame_images_when_policy_disables_saving(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(
        tmp_path,
        frame_count=2,
        save_frame_images=False,
        write_frame_images=False,
    )
    (layout.raw_frames_dir / "f000000000.png").write_bytes(b"stray")

    result = validate_run_artifacts(layout)

    assert result.counts_match is False
    assert result.final_status == "failed"
    assert result.validation_errors == [
        "raw frame count does not match frame image retention policy: 1 != 0"
    ]


def test_validate_run_artifacts_accepts_unretained_crop_images_when_disabled(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(
        tmp_path,
        frame_count=3,
        save_frame_images=False,
        write_frame_images=False,
        save_crop_images=False,
        write_crop_images=False,
    )

    result = validate_run_artifacts(layout)
    summary = build_qa_summary(layout)

    assert result.counts.crop_files == 0
    assert result.counts_match is True
    assert summary.byte_counts.crops_bytes == 0
    assert summary.final_status == "complete"


def test_validate_run_artifacts_rejects_stray_crop_images_when_policy_disables_saving(
    tmp_path: Path,
) -> None:
    layout = _write_fixture_run(
        tmp_path,
        frame_count=2,
        save_frame_images=False,
        write_frame_images=False,
        save_crop_images=False,
        write_crop_images=False,
    )
    (layout.left_eye_crops_dir / "f000000000.png").write_bytes(b"stray")

    result = validate_run_artifacts(layout)

    assert result.counts_match is False
    assert result.final_status == "failed"
    assert result.validation_errors == [
        "crop file count does not match crop image retention policy: 1 != 0"
    ]


def test_tail_truncated_run_uses_video_manifest_decoded_count(tmp_path: Path) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=3)
    frame_lines = (
        (layout.records_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    )
    (layout.records_dir / "frames.jsonl").write_text(
        "\n".join(frame_lines[:2]) + "\n",
        encoding="utf-8",
    )
    (layout.raw_frames_dir / "f000000002.png").unlink()
    (layout.processed_frames_dir / "f000000002.jpg").unlink()

    result = validate_run_artifacts(layout)

    assert result.counts.decoded_frames == 3
    assert result.counts.frame_records == 2
    assert result.counts.raw_frames == 2
    assert result.counts.processed_frames == 2
    assert result.counts_match is False
    assert result.final_status == "failed"


def test_malformed_jsonl_produces_failed_qa_summary(tmp_path: Path) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=2)
    (layout.records_dir / "frames.jsonl").write_text(
        '{"frame_id": "f000000000"}\n{malformed-json\n',
        encoding="utf-8",
    )

    result = validate_run_artifacts(layout)
    summary = build_qa_summary(layout)

    assert result.schema_validation_passed is False
    assert result.final_status == "failed"
    assert any(
        CliErrorCode.SCHEMA_VALIDATION_FAILED.value in error
        for error in result.validation_errors
    )
    assert summary.artifact_validation.schema_validation_passed is False
    assert summary.final_status == "failed"
    assert summary.status_transitions == [
        "created",
        "processing",
        "revalidating",
        "failed",
    ]


def test_invalid_utf8_jsonl_produces_failed_qa_summary(tmp_path: Path) -> None:
    layout = _write_fixture_run(tmp_path, frame_count=2)
    (layout.records_dir / "frames.jsonl").write_bytes(b"\xff\n")

    result = validate_run_artifacts(layout)
    summary = build_qa_summary(layout)

    assert result.schema_validation_passed is False
    assert result.final_status == "failed"
    assert any(
        CliErrorCode.SCHEMA_VALIDATION_FAILED.value in error
        for error in result.validation_errors
    )
    assert summary.artifact_validation.schema_validation_passed is False
    assert summary.final_status == "failed"
