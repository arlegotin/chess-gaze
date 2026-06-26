from __future__ import annotations

from pathlib import Path

from chess_gaze.errors import FrameStatus
from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.pipeline import (
    AnalyzeRequest,
    ObserverBundle,
    ObserverFrame,
    analyze_video,
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


def _deterministic_valid_scene_record(frame: ObserverFrame) -> FrameRecord:
    width = float(frame.rgb.shape[1])
    height = float(frame.rgb.shape[0])
    x_shift = float(frame.frame_index % 23)
    y_shift = float(frame.frame_index % 13)
    left_pupil = _point(width * 0.42 + x_shift, height * 0.43 + y_shift)
    right_pupil = _point(width * 0.57 + x_shift, height * 0.43 + y_shift)
    yaw = 0.0
    pitch = 0.0
    gaze = {
        "valid": True,
        "yaw_radians": yaw,
        "pitch_radians": pitch,
        "reason_invalid": None,
    }

    return FrameRecord.model_validate(
        {
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "status": FrameStatus.OK,
            "timestamp_seconds": frame.timestamp_seconds,
            "face": {
                "present": True,
                "bounding_box": _box(
                    width * 0.28 + x_shift,
                    height * 0.20 + y_shift,
                    width * 0.72 + x_shift,
                    height * 0.82 + y_shift,
                ),
                "landmarks": [
                    _point(width * 0.42 + x_shift, height * 0.38 + y_shift),
                    _point(width * 0.57 + x_shift, height * 0.38 + y_shift),
                    _point(width * 0.50 + x_shift, height * 0.54 + y_shift),
                ],
                "reason_invalid": None,
            },
            "left_eye": {
                "present": True,
                "bounding_box": _box(
                    left_pupil.x - 14.0,
                    left_pupil.y - 9.0,
                    left_pupil.x + 14.0,
                    left_pupil.y + 9.0,
                ),
                "pupil_center": left_pupil,
                "iris_landmarks": [
                    _point(left_pupil.x - 4.0, left_pupil.y),
                    _point(left_pupil.x + 4.0, left_pupil.y),
                    _point(left_pupil.x, left_pupil.y - 4.0),
                    _point(left_pupil.x, left_pupil.y + 4.0),
                ],
                "reason_invalid": None,
            },
            "right_eye": {
                "present": True,
                "bounding_box": _box(
                    right_pupil.x - 14.0,
                    right_pupil.y - 9.0,
                    right_pupil.x + 14.0,
                    right_pupil.y + 9.0,
                ),
                "pupil_center": right_pupil,
                "iris_landmarks": [
                    _point(right_pupil.x - 4.0, right_pupil.y),
                    _point(right_pupil.x + 4.0, right_pupil.y),
                    _point(right_pupil.x, right_pupil.y - 4.0),
                    _point(right_pupil.x, right_pupil.y + 4.0),
                ],
                "reason_invalid": None,
            },
            "head_pose": {
                "valid": True,
                "yaw_radians": 0.0,
                "pitch_radians": 0.0,
                "roll_radians": 0.0,
                "reason_invalid": None,
            },
            "geometric_gaze": gaze,
            "appearance_gaze": gaze,
            "recommended_gaze": {
                "valid": True,
                "yaw_radians": 0.75,
                "pitch_radians": -0.25,
                "reason_invalid": None,
            },
            "errors": [],
        }
    )


def test_model_free_nakamura_video_scene_artifact_contract(tmp_path: Path) -> None:
    video_path = Path("artifacts/input/nakamura_1.mp4")
    assert video_path.is_file(), f"missing mandatory real-data video: {video_path}"

    pipeline_result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_deterministic_valid_scene_record),
    )
    generated_viewer_data = ViewerSceneData.model_validate_json(
        pipeline_result.viewer_scene_data_path.read_text(encoding="utf-8")
    )
    scene_result = build_scene_artifacts(pipeline_result.layout)
    viewer_data = build_viewer_scene_data(scene_result)
    summary = SceneSummary.model_validate_json(
        scene_result.paths.scene_summary_path.read_text(encoding="utf-8")
    )

    assert pipeline_result.decoded_frame_count == 1973
    assert pipeline_result.viewer_index_path.is_file()
    assert pipeline_result.viewer_scene_data_path.is_file()
    assert generated_viewer_data.frame_count == 1973
    assert len(generated_viewer_data.frames) == 1973
    assert len(generated_viewer_data.valid_hit_points) == 1973
    assert generated_viewer_data.summary.artifact_validation.viewer_exists is True
    assert scene_result.scene_frame_count == 1973
    assert viewer_data.frame_count == 1973
    assert len(viewer_data.frames) == 1973
    assert len(viewer_data.valid_hit_points) == 1973
    assert [point.frame_index for point in viewer_data.valid_hit_points[:3]] == [
        0,
        1,
        2,
    ]
    assert viewer_data.valid_hit_points[-1].frame_index == 1972
    assert summary.decoded_frames == 1973
    assert summary.scene_frame_records == 1973
    assert summary.valid_monitor_hit_frames == 1973
    assert summary.artifact_validation.scene_frame_count_matches_decoded is True
    assert summary.artifact_validation.scene_manifest_valid is True
    assert summary.artifact_validation.scene_summary_valid is True
