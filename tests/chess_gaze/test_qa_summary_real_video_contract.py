from __future__ import annotations

from pathlib import Path

import pytest

from chess_gaze.errors import FrameStatus
from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.pipeline import (
    AnalyzeRequest,
    ObserverBundle,
    ObserverFrame,
    analyze_video,
)
from chess_gaze.qa_summary import QASummary


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


def _deterministic_real_video_record(frame: ObserverFrame) -> FrameRecord:
    width = float(frame.rgb.shape[1])
    height = float(frame.rgb.shape[0])
    x_shift = float(frame.frame_index % 17)
    y_shift = float(frame.frame_index % 11)
    face_box = _box(
        width * 0.25 + x_shift,
        height * 0.20 + y_shift,
        width * 0.70 + x_shift,
        height * 0.82 + y_shift,
    )
    left_pupil = _point(width * 0.40 + x_shift, height * 0.42 + y_shift)
    right_pupil = _point(width * 0.55 + x_shift, height * 0.42 + y_shift)
    left_eye = {
        "present": True,
        "bounding_box": _box(
            left_pupil.x - 12.0,
            left_pupil.y - 8.0,
            left_pupil.x + 12.0,
            left_pupil.y + 8.0,
        ),
        "pupil_center": left_pupil,
        "iris_landmarks": [
            _point(left_pupil.x - 4.0, left_pupil.y),
            _point(left_pupil.x + 4.0, left_pupil.y),
            _point(left_pupil.x, left_pupil.y - 4.0),
            _point(left_pupil.x, left_pupil.y + 4.0),
        ],
        "reason_invalid": None,
    }
    right_eye = {
        "present": True,
        "bounding_box": _box(
            right_pupil.x - 12.0,
            right_pupil.y - 8.0,
            right_pupil.x + 12.0,
            right_pupil.y + 8.0,
        ),
        "pupil_center": right_pupil,
        "iris_landmarks": [
            _point(right_pupil.x - 4.0, right_pupil.y),
            _point(right_pupil.x + 4.0, right_pupil.y),
            _point(right_pupil.x, right_pupil.y - 4.0),
            _point(right_pupil.x, right_pupil.y + 4.0),
        ],
        "reason_invalid": None,
    }
    yaw = (frame.frame_index % 31) / 300.0
    pitch = -((frame.frame_index % 29) / 300.0)

    return FrameRecord.model_validate(
        {
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "status": FrameStatus.OK,
            "timestamp_seconds": frame.timestamp_seconds,
            "face": {
                "present": True,
                "bounding_box": face_box,
                "landmarks": [
                    _point(width * 0.40 + x_shift, height * 0.38 + y_shift),
                    _point(width * 0.55 + x_shift, height * 0.38 + y_shift),
                    _point(width * 0.48 + x_shift, height * 0.54 + y_shift),
                ],
                "reason_invalid": None,
            },
            "left_eye": left_eye,
            "right_eye": right_eye,
            "head_pose": {
                "valid": True,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "roll_radians": 0.0,
                "reason_invalid": None,
            },
            "geometric_gaze": {
                "valid": True,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "reason_invalid": None,
            },
            "appearance_gaze": {
                "valid": True,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "reason_invalid": None,
            },
            "recommended_gaze": {
                "valid": True,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "reason_invalid": None,
            },
            "errors": [],
        }
    )


@pytest.mark.parametrize(
    ("video_path", "expected_count"),
    [
        (Path("artifacts/input/test_1.mp4"), 3613),
        (Path("artifacts/input/test_2.mp4"), 1973),
    ],
)
def test_real_video_model_free_pipeline_writes_qa_summary_revalidation(
    tmp_path: Path, video_path: Path, expected_count: int
) -> None:
    assert video_path.is_file(), f"missing mandatory real-data video: {video_path}"

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_deterministic_real_video_record),
    )
    qa_summary_path = result.layout.run_dir / "qa_summary.json"
    assert qa_summary_path.is_file()

    summary = QASummary.model_validate_json(qa_summary_path.read_text(encoding="utf-8"))

    assert summary.source_artifacts == {
        "run_manifest": "run_manifest.json",
        "calibration": "calibration.json",
        "video_manifest": "video_manifest.json",
        "frames_jsonl": "records/frames.jsonl",
        "errors_jsonl": "records/errors.jsonl",
        "raw_frames": "raw_frames",
        "processed_frames": "processed_frames",
        "crops": "crops",
    }
    assert summary.counts.decoded_frames == expected_count
    assert summary.counts.raw_frames == expected_count
    assert summary.counts.processed_frames == expected_count
    assert summary.counts.frame_records == expected_count
    assert summary.artifact_validation.schema_validation_passed is True
    assert summary.artifact_validation.counts_match is True
    assert len(summary.qa_sample_frame_ids) == 30
    assert summary.qa_sample_frame_ids == sorted(summary.qa_sample_frame_ids)
    assert summary.byte_counts.raw_frames_bytes > 0
    assert summary.byte_counts.processed_frames_bytes > 0
    assert summary.byte_counts.jsonl_bytes > 0
    assert summary.byte_counts.total_run_bytes > (
        summary.byte_counts.raw_frames_bytes
        + summary.byte_counts.processed_frames_bytes
    )
    assert summary.status_transitions == [
        "created",
        "processing",
        "revalidating",
        "complete",
    ]
    assert isinstance(summary.representative_failure_frame_ids, list)
    assert summary.final_status == "complete"
