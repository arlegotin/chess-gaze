from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TextIO

import numpy as np
import numpy.typing as npt

from chess_gaze.artifact_runs import RunLayout, create_run_layout
from chess_gaze.calibration import default_calibration
from pydantic import ValidationError

from chess_gaze.configuration import (
    ConfigurationError,
    apply_analysis_overrides,
    load_config,
)
from chess_gaze.errors import CliErrorCode, ErrorCode, FrameStatus
from chess_gaze.frame_records import (
    CalibrationRecord,
    ErrorRecord,
    FrameErrorRecord,
    FrameRecord,
    InferenceRuntimeRecord,
    RunManifest,
)
from chess_gaze.image_io import atomic_write_bytes, save_rgb_png
from chess_gaze.model_assets import (
    ModelAssetError,
    ResolvedModelAsset,
    load_model_registry,
    validate_required_assets,
)
from chess_gaze.qa_summary import ArtifactValidationError, QASummary, build_qa_summary
from chess_gaze.scene_artifacts import build_scene_artifacts
from chess_gaze.scene_viewer import build_scene_viewer
from chess_gaze.video_decode import (
    DecodedFrame,
    VideoDecodeError,
    inspect_video,
    iter_decoded_frames,
)
from chess_gaze.visualization import render_processed_frame

DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")
DEFAULT_APPROVED_LICENSES = frozenset({"MG-NC-RAI-2.0"})
QA_SUMMARY_BYTE_COUNT_STABILIZATION_ATTEMPTS = 5
RawFrameWriter = Callable[[Path, npt.NDArray[np.uint8]], str]
raw_frame_writer: RawFrameWriter = save_rgb_png
FrameErrorWriter = Callable[[TextIO, FrameRecord], None]
DefaultObserverBundleFactory = Callable[
    [list[ResolvedModelAsset], CalibrationRecord, RunLayout], "ObserverBundle"
]


class FrameRecordObserver(Protocol):
    def __call__(self, frame: ObserverFrame) -> FrameRecord: ...


Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ObserverFrame:
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    rgb: npt.NDArray[np.uint8]
    pts: int | None
    pts_seconds: float | None
    duration_seconds: float | None


@dataclass(frozen=True)
class ObserverBundle:
    frame_observer: FrameRecordObserver
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class AnalyzeRequest:
    video_path: Path
    output_root: Path | None = None
    models_root: Path | None = None
    config_path: Path | None = None
    unigaze_device: str | None = None
    unigaze_batch_size: int | None = None
    model_registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH
    run_suffix: str | None = None
    clock: Clock = utc_now


@dataclass(frozen=True)
class AnalyzeResult:
    layout: RunLayout
    run_manifest_path: Path
    calibration_path: Path
    video_manifest_path: Path
    frames_jsonl_path: Path
    errors_jsonl_path: Path
    scene_manifest_path: Path
    scene_summary_path: Path
    scene_frames_jsonl_path: Path
    viewer_index_path: Path
    viewer_scene_data_path: Path
    qa_summary_path: Path
    decoded_frame_count: int
    validated_record_count: int
    validated_error_count: int
    frame_error_count: int
    valid_scene_frame_count: int
    valid_monitor_hit_count: int


class PipelineError(RuntimeError):
    def __init__(self, code: CliErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _ResolvedRequest:
    video_path: Path
    output_root: Path
    models_root: Path
    raw_frame_image_format: str
    processed_frame_image_format: str
    processed_frame_jpeg_quality: int
    unigaze_device: str
    unigaze_batch_size: int


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


def analyze_video(
    request: AnalyzeRequest, observers: ObserverBundle | None = None
) -> AnalyzeResult:
    resolved = _resolve_request(request)
    _validate_input_video(resolved.video_path)

    try:
        inspection = inspect_video(resolved.video_path)
    except VideoDecodeError as exc:
        raise PipelineError(exc.code, str(exc)) from exc

    resolved_model_assets: list[ResolvedModelAsset] | None = None
    if observers is None:
        resolved_model_assets = _validate_model_assets(
            request.model_registry_path, resolved
        )

    _estimate_disk_space(resolved.output_root)

    created_at = request.clock()
    calibration = default_calibration()
    layout = create_run_layout(
        input_path=resolved.video_path,
        output_root=resolved.output_root / resolved.video_path.stem / "runs",
        clock=lambda: created_at,
        run_suffix=request.run_suffix,
    )
    run_manifest_path = layout.run_dir / "run_manifest.json"
    calibration_path = layout.run_dir / "calibration.json"
    video_manifest_path = layout.run_dir / "video_manifest.json"
    frames_jsonl_path = layout.records_dir / "frames.jsonl"
    errors_jsonl_path = layout.records_dir / "errors.jsonl"
    qa_summary_path = layout.run_dir / "qa_summary.json"
    inference = _external_observer_inference_record()

    _write_json(
        run_manifest_path,
        RunManifest(
            run_id=layout.run_dir.name,
            created_at_utc=_format_utc(created_at),
            input_path=str(resolved.video_path),
            video=inspection.video_manifest,
            inference=inference,
        ).model_dump(mode="json"),
    )
    _write_json(calibration_path, calibration.model_dump(mode="json"))
    _write_json(video_manifest_path, inspection.video_manifest.model_dump(mode="json"))
    frames_jsonl_path.touch()
    errors_jsonl_path.touch()

    if observers is None:
        if resolved_model_assets is None:
            raise AssertionError("resolved_model_assets must be set for default run")
        observers = default_observer_bundle_factory(
            resolved_model_assets, calibration, layout
        )

    decoded_frame_count = 0
    frame_error_count = 0
    try:
        with (
            frames_jsonl_path.open("a", encoding="utf-8") as frames_handle,
            errors_jsonl_path.open("a", encoding="utf-8") as errors_handle,
        ):
            for decoded_frame in iter_decoded_frames(resolved.video_path):
                decoded_frame_count += 1
                record, frame_errors = _process_frame(
                    decoded_frame,
                    observers,
                    resolved,
                    layout,
                    errors_handle=errors_handle,
                )
                frame_error_count += len(frame_errors)
                frames_handle.write(record.model_dump_json() + "\n")
    finally:
        if observers.close is not None:
            observers.close()

    try:
        scene_result = build_scene_artifacts(layout)
        viewer_result = build_scene_viewer(layout, scene_result)
    except (OSError, ValueError) as exc:
        raise PipelineError(CliErrorCode.SCHEMA_VALIDATION_FAILED, str(exc)) from exc

    try:
        qa_summary = _build_and_write_qa_summary(layout, qa_summary_path)
    except ArtifactValidationError as exc:
        raise PipelineError(exc.code, str(exc)) from exc
    if not qa_summary.artifact_validation.schema_validation_passed:
        raise PipelineError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            f"Run artifact validation failed; see {qa_summary_path}",
        )

    return AnalyzeResult(
        layout=layout,
        run_manifest_path=run_manifest_path,
        calibration_path=calibration_path,
        video_manifest_path=video_manifest_path,
        frames_jsonl_path=frames_jsonl_path,
        errors_jsonl_path=errors_jsonl_path,
        scene_manifest_path=scene_result.paths.scene_manifest_path,
        scene_summary_path=scene_result.paths.scene_summary_path,
        scene_frames_jsonl_path=scene_result.paths.scene_frames_jsonl_path,
        viewer_index_path=viewer_result.index_path,
        viewer_scene_data_path=viewer_result.scene_data_path,
        qa_summary_path=qa_summary_path,
        decoded_frame_count=decoded_frame_count,
        validated_record_count=qa_summary.counts.frame_records,
        validated_error_count=sum(qa_summary.errors_by_code.values()),
        frame_error_count=frame_error_count,
        valid_scene_frame_count=scene_result.scene_frame_count,
        valid_monitor_hit_count=scene_result.valid_monitor_hit_count,
    )


def _build_and_write_qa_summary(
    run_layout: RunLayout, qa_summary_path: Path
) -> QASummary:
    qa_summary = build_qa_summary(run_layout)
    stable_total_run_bytes: int | None = None

    for _attempt in range(QA_SUMMARY_BYTE_COUNT_STABILIZATION_ATTEMPTS):
        _write_json(qa_summary_path, qa_summary.model_dump(mode="json"))
        refreshed_summary = build_qa_summary(run_layout)
        refreshed_total_run_bytes = refreshed_summary.byte_counts.total_run_bytes
        if refreshed_total_run_bytes == stable_total_run_bytes:
            return qa_summary
        stable_total_run_bytes = refreshed_total_run_bytes
        qa_summary = refreshed_summary

    _write_json(qa_summary_path, qa_summary.model_dump(mode="json"))
    return qa_summary


def _resolve_request(request: AnalyzeRequest) -> _ResolvedRequest:
    try:
        config = load_config(request.config_path)
    except ConfigurationError as exc:
        raise PipelineError(CliErrorCode.USAGE, str(exc)) from exc
    try:
        resolved_config = apply_analysis_overrides(
            config,
            output_root=request.output_root,
            models_root=request.models_root,
            unigaze_device=request.unigaze_device,
            unigaze_batch_size=request.unigaze_batch_size,
        )
    except ValidationError as exc:
        raise PipelineError(CliErrorCode.USAGE, str(exc)) from exc

    return _ResolvedRequest(
        video_path=request.video_path,
        output_root=resolved_config.output_root,
        models_root=resolved_config.models_root,
        raw_frame_image_format=resolved_config.raw_frame_image_format,
        processed_frame_image_format=resolved_config.processed_frame_image_format,
        processed_frame_jpeg_quality=resolved_config.processed_frame_jpeg_quality,
        unigaze_device=resolved_config.unigaze_device,
        unigaze_batch_size=resolved_config.unigaze_batch_size,
    )


def _validate_input_video(video_path: Path) -> None:
    if not video_path.is_file():
        raise PipelineError(
            CliErrorCode.INPUT_NOT_FOUND,
            f"Input video not found: {video_path}",
        )


def _validate_model_assets(
    model_registry_path: Path, resolved: _ResolvedRequest
) -> list[ResolvedModelAsset]:
    registry = load_model_registry(model_registry_path)
    approved_licenses = {
        model.license for model in registry.models if model.license_approved
    } | set(DEFAULT_APPROVED_LICENSES)
    try:
        return validate_required_assets(
            registry,
            resolved.models_root,
            approved_licenses,
        )
    except ModelAssetError as exc:
        raise PipelineError(exc.code, str(exc)) from exc


def _default_observer_bundle_factory(
    resolved_assets: list[ResolvedModelAsset],
    calibration: CalibrationRecord,
    run_layout: RunLayout,
) -> ObserverBundle:
    from chess_gaze.face_observation import MediaPipeFaceObserver
    from chess_gaze.frame_observation import ModelBackedFrameObserver
    from chess_gaze.gaze_observation import UNIGAZE_MODEL_ID, UniGazeModel

    face_asset = _asset_by_id(resolved_assets, "mediapipe-face-landmarker")
    gaze_asset = _asset_by_id(resolved_assets, UNIGAZE_MODEL_ID)
    observer = ModelBackedFrameObserver(
        face_observer=MediaPipeFaceObserver(
            model_asset_path=face_asset.resolved_path,
            calibration=calibration,
        ),
        gaze_model=UniGazeModel.from_local_asset(gaze_asset, device="cpu"),
        calibration=calibration,
        run_layout=run_layout,
    )
    return ObserverBundle(frame_observer=observer, close=observer.close)


def _asset_by_id(
    resolved_assets: list[ResolvedModelAsset], model_id: str
) -> ResolvedModelAsset:
    for asset in resolved_assets:
        if asset.model_id == model_id:
            return asset
    raise PipelineError(
        CliErrorCode.MODEL_ASSET_MISSING,
        f"Required resolved model asset missing for {model_id}",
    )


default_observer_bundle_factory: DefaultObserverBundleFactory = (
    _default_observer_bundle_factory
)


def _estimate_disk_space(output_root: Path) -> None:
    existing_parent = _nearest_existing_parent(output_root)
    disk_usage = shutil.disk_usage(existing_parent)
    if disk_usage.free <= 0:
        raise PipelineError(
            CliErrorCode.USAGE,
            f"No free disk space available under {existing_parent}",
        )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate


def _process_frame(
    decoded_frame: DecodedFrame,
    observers: ObserverBundle,
    resolved: _ResolvedRequest,
    layout: RunLayout,
    *,
    errors_handle: TextIO,
) -> tuple[FrameRecord, list[ErrorRecord]]:
    frame_errors: list[ErrorRecord] = []
    raw_path = layout.raw_frames_dir / f"{decoded_frame.frame_id}.png"
    processed_path = layout.processed_frames_dir / f"{decoded_frame.frame_id}.jpg"

    try:
        _validate_image_format(resolved.raw_frame_image_format, "png")
        raw_frame_writer(raw_path, decoded_frame.rgb)
    except Exception as exc:
        frame_errors.append(
            ErrorRecord(
                code=ErrorCode.RAW_FRAME_WRITE_FAILED,
                message=f"Raw frame write failed: {exc}",
            )
        )

    observer_frame = ObserverFrame(
        frame_id=decoded_frame.frame_id,
        frame_index=decoded_frame.frame_index,
        timestamp_seconds=_timestamp_seconds(decoded_frame),
        rgb=decoded_frame.rgb,
        pts=decoded_frame.pts,
        pts_seconds=decoded_frame.pts_seconds,
        duration_seconds=decoded_frame.duration_seconds,
    )
    record = observers.frame_observer(observer_frame)
    _validate_observer_record_identity(record, decoded_frame)
    record = _record_with_errors(record, frame_errors)

    try:
        _validate_image_format(resolved.processed_frame_image_format, "jpg")
        render_processed_frame(
            decoded_frame.rgb,
            record,
            processed_path,
            resolved.processed_frame_jpeg_quality,
        )
    except Exception as exc:
        frame_errors.append(
            ErrorRecord(
                code=ErrorCode.PROCESSED_FRAME_WRITE_FAILED,
                message=f"Processed frame write failed: {exc}",
            )
        )
        record = _record_with_errors(record, frame_errors)

    frame_error_writer(errors_handle, record)
    return record, frame_errors


def _timestamp_seconds(decoded_frame: DecodedFrame) -> float:
    if decoded_frame.pts_seconds is not None:
        return decoded_frame.pts_seconds
    return float(decoded_frame.frame_index)


def _validate_observer_record_identity(
    record: FrameRecord, decoded_frame: DecodedFrame
) -> None:
    if record.frame_id != decoded_frame.frame_id:
        raise PipelineError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            (
                "Observer returned a frame_id for a different frame: "
                f"{record.frame_id} != {decoded_frame.frame_id}"
            ),
        )
    if record.frame_index != decoded_frame.frame_index:
        raise PipelineError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            (
                "Observer returned a frame_index for a different frame: "
                f"{record.frame_index} != {decoded_frame.frame_index}"
            ),
        )


def _record_with_errors(
    record: FrameRecord, frame_errors: list[ErrorRecord]
) -> FrameRecord:
    if not frame_errors:
        return record

    existing_errors = list(record.errors)
    existing_codes = {(error.code, error.message) for error in existing_errors}
    for frame_error in frame_errors:
        key = (frame_error.code, frame_error.message)
        if key not in existing_codes:
            existing_errors.append(frame_error)
            existing_codes.add(key)

    payload = record.model_dump(mode="python")
    payload["status"] = FrameStatus.ERROR
    payload["errors"] = existing_errors
    return FrameRecord.model_validate(payload)


def _append_frame_errors(errors_handle: TextIO, record: FrameRecord) -> None:
    for error in record.errors:
        errors_handle.write(
            FrameErrorRecord(
                frame_id=record.frame_id,
                frame_index=record.frame_index,
                code=error.code,
                message=error.message,
            ).model_dump_json()
            + "\n"
        )


frame_error_writer: FrameErrorWriter = _append_frame_errors


def _write_json(path: Path, payload: object) -> None:
    data = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(path, data)


def _validate_image_format(actual: str, expected: str) -> None:
    if actual.lower() != expected:
        raise PipelineError(
            CliErrorCode.USAGE,
            f"Unsupported image format {actual!r}; expected {expected!r}",
        )


def _format_utc(value: datetime) -> str:
    return (
        value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
