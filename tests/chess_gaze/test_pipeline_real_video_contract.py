from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import IO, Any, NoReturn, cast

import pytest

from chess_gaze.errors import ErrorCode, FrameStatus
from chess_gaze.frame_records import FrameRecord
from chess_gaze.geometry import BBox, CoordinateSpace, Point2D
from chess_gaze.pipeline import (
    AnalyzeRequest,
    AnalyzeResult,
    ObserverBundle,
    ObserverFrame,
    analyze_video,
)

NAKAMURA_SHORT_VIDEO = Path("artifacts/input/nakamura_short.mp4")
NAKAMURA_SHORT_FRAME_COUNT = 180


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
    left_pupil = _point(width * 0.55 + x_shift, height * 0.42 + y_shift)
    right_pupil = _point(width * 0.40 + x_shift, height * 0.42 + y_shift)
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
                    _point(width * 0.55 + x_shift, height * 0.38 + y_shift),
                    _point(width * 0.40 + x_shift, height * 0.38 + y_shift),
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


def _records(path: Path) -> list[FrameRecord]:
    return [
        FrameRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _read_first_jsonl_record(path: Path) -> tuple[str, dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    assert first_line
    return first_line, json.loads(first_line)


def _read_last_jsonl_record(path: Path) -> tuple[str, dict[str, object]]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        assert file_size > 0
        position = file_size

        while position > 0:
            position -= 1
            handle.seek(position)
            if handle.read(1) not in (b"\n", b"\r"):
                break

        while position > 0:
            position -= 1
            handle.seek(position)
            if handle.read(1) == b"\n":
                position += 1
                break

        handle.seek(position)
        last_line = handle.readline().decode("utf-8").rstrip("\r\n")
        assert last_line
        return last_line, json.loads(last_line)

    raise AssertionError(f"expected at least one JSONL record in {path}")


def test_read_last_jsonl_record_handles_last_line_larger_than_tail_chunk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    first = {"frame_id": "f000000000", "frame_index": 0}
    last = {
        "frame_id": "f000000001",
        "frame_index": 1,
        "payload": "x" * 5000,
    }
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(last) + "\n",
        encoding="utf-8",
    )

    last_line, last_record = _read_last_jsonl_record(path)

    assert last_line == json.dumps(last)
    assert last_record == last


def _assert_default_completed_artifact_contract(
    result: AnalyzeResult, *, expected_count: int
) -> tuple[str, str]:
    first_line, first_record = _read_first_jsonl_record(result.frames_jsonl_path)
    last_line, last_record = _read_last_jsonl_record(result.frames_jsonl_path)

    assert result.decoded_frame_count == expected_count
    assert result.qa_summary_path is None
    assert result.validated_record_count is None
    assert result.validated_error_count is None
    assert not (result.layout.run_dir / "qa_summary.json").exists()
    assert first_record["frame_id"] == "f000000000"
    assert first_record["frame_index"] == 0
    assert last_record["frame_id"] == f"f{expected_count - 1:09d}"
    assert last_record["frame_index"] == expected_count - 1
    assert result.scene_manifest_path.is_file()
    assert result.scene_summary_path.is_file()
    assert result.scene_frames_jsonl_path.is_file()
    assert result.viewer_index_path.is_file()
    assert result.viewer_scene_data_path.is_file()
    state = json.loads(result.analysis_state_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["next_frame_index"] == expected_count
    return first_line, last_line


@pytest.mark.parametrize(
    ("video_path", "expected_count"),
    [(NAKAMURA_SHORT_VIDEO, NAKAMURA_SHORT_FRAME_COUNT)],
)
def test_real_video_model_free_pipeline_writes_complete_artifact_contract(
    tmp_path: Path, video_path: Path, expected_count: int
) -> None:
    assert video_path.is_file(), f"missing mandatory real-data video: {video_path}"

    result = analyze_video(
        AnalyzeRequest(video_path=video_path, output_root=tmp_path / "output"),
        observers=ObserverBundle(frame_observer=_deterministic_real_video_record),
    )
    raw_count = len(list(result.layout.raw_frames_dir.glob("*.png")))
    processed_count = len(list(result.layout.processed_frames_dir.glob("*.jpg")))
    crop_count = len(list(result.layout.crops_dir.rglob("*.png")))
    first_line, last_line = _assert_default_completed_artifact_contract(
        result, expected_count=expected_count
    )
    print(
        f"{video_path}: decoded={result.decoded_frame_count} "
        f"raw={raw_count} processed={processed_count} crops={crop_count} "
        f"first={json.loads(first_line)['frame_id']} "
        f"last={json.loads(last_line)['frame_id']}"
    )

    assert raw_count == 0
    assert processed_count == 0
    assert crop_count == 0
    assert not result.layout.crops_dir.exists()
    first_record = FrameRecord.model_validate_json(first_line)
    last_record = FrameRecord.model_validate_json(last_line)
    assert first_record.status is FrameStatus.OK
    assert last_record.status is FrameStatus.OK
    assert all(
        ErrorCode.FACE_NOT_FOUND not in {error.code for error in record.errors}
        for record in (first_record, last_record)
    )


def test_default_completed_artifact_contract_avoids_full_record_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert NAKAMURA_SHORT_VIDEO.is_file(), (
        f"missing mandatory real-data video: {NAKAMURA_SHORT_VIDEO}"
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=NAKAMURA_SHORT_VIDEO, output_root=tmp_path / "output"
        ),
        observers=ObserverBundle(frame_observer=_deterministic_real_video_record),
    )

    original_read_text = Path.read_text
    original_open = Path.open

    def _fail_frame_record_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self == result.frames_jsonl_path:
            raise AssertionError("default no-QA helper must not call Path.read_text()")
        return original_read_text(self, encoding=encoding, errors=errors)

    def _fail_validation(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError(
            "default no-QA helper must not validate frame JSON via "
            "FrameRecord.model_validate_json()"
        )

    class _BoundedFrameReadProxy:
        def __init__(self, wrapped: IO[Any]) -> None:
            self._wrapped = wrapped

        def read(self, size: int = -1) -> bytes | str:
            if size < 0:
                raise AssertionError(
                    "default no-QA helper must not read the whole file"
                )
            return cast(bytes | str, self._wrapped.read(size))

        def readlines(self, *args: object, **kwargs: object) -> NoReturn:
            del args, kwargs
            raise AssertionError(
                "default no-QA helper must not materialize all JSONL lines"
            )

        def __iter__(self) -> NoReturn:
            raise AssertionError(
                "default no-QA helper must not iterate every JSONL record"
            )

        def __enter__(self) -> _BoundedFrameReadProxy:
            self._wrapped.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            return self._wrapped.__exit__(exc_type, exc, traceback)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

    def _guarded_open(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> Any:
        handle = original_open(
            self,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
        if self == result.frames_jsonl_path:
            return _BoundedFrameReadProxy(handle)
        return handle

    monkeypatch.setattr(Path, "read_text", _fail_frame_record_read_text)
    monkeypatch.setattr(Path, "open", _guarded_open)
    monkeypatch.setattr(FrameRecord, "model_validate_json", _fail_validation)

    first_line, last_line = _assert_default_completed_artifact_contract(
        result, expected_count=NAKAMURA_SHORT_FRAME_COUNT
    )

    assert json.loads(first_line)["frame_index"] == 0
    assert json.loads(last_line)["frame_index"] == NAKAMURA_SHORT_FRAME_COUNT - 1


@pytest.mark.native_mediapipe
def test_nakamura_short_default_model_pipeline_does_not_create_crop_directory(
    tmp_path: Path,
) -> None:
    assert NAKAMURA_SHORT_VIDEO.is_file(), (
        f"missing mandatory real-data video: {NAKAMURA_SHORT_VIDEO}"
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=NAKAMURA_SHORT_VIDEO,
            output_root=tmp_path / "output",
            unigaze_device="cpu",
            unigaze_batch_size=7,
        )
    )
    _assert_default_completed_artifact_contract(
        result, expected_count=NAKAMURA_SHORT_FRAME_COUNT
    )

    assert list(result.layout.raw_frames_dir.glob("*.png")) == []
    assert list(result.layout.processed_frames_dir.glob("*.jpg")) == []
    assert not result.layout.crops_dir.exists()
    assert list(result.layout.crops_dir.rglob("*.png")) == []


@pytest.mark.native_mediapipe
def test_nakamura_short_save_crops_retains_crop_images_only(
    tmp_path: Path,
) -> None:
    assert NAKAMURA_SHORT_VIDEO.is_file(), (
        f"missing mandatory real-data video: {NAKAMURA_SHORT_VIDEO}"
    )

    result = analyze_video(
        AnalyzeRequest(
            video_path=NAKAMURA_SHORT_VIDEO,
            output_root=tmp_path / "output",
            unigaze_device="cpu",
            unigaze_batch_size=7,
            save_crop_images=True,
        )
    )
    _assert_default_completed_artifact_contract(
        result, expected_count=NAKAMURA_SHORT_FRAME_COUNT
    )
    crop_paths = list(result.layout.crops_dir.rglob("*.png"))

    assert list(result.layout.raw_frames_dir.glob("*.png")) == []
    assert list(result.layout.processed_frames_dir.glob("*.jpg")) == []
    assert result.layout.crops_dir.is_dir()
    assert len(crop_paths) > 0
