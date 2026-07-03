from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from chess_gaze.analysis_resume import (
    commit_processed_records,
    find_latest_resumable_run,
    new_analysis_state,
    prepare_resume_run,
    update_analysis_state,
    write_analysis_state,
    write_initial_run_artifacts,
)
from chess_gaze.artifact_runs import RunLayout, create_run_layout
from chess_gaze.calibration import default_calibration
from chess_gaze.configuration import (
    ConfigurationError,
    apply_analysis_overrides,
    load_config,
)
from chess_gaze.errors import CliErrorCode, ErrorCode, FrameStatus
from chess_gaze.frame_observation import ModelInferenceError
from chess_gaze.frame_records import (
    CalibrationRecord,
    CropImageRetentionPolicy,
    ErrorRecord,
    FrameErrorRecord,
    FrameImageRetentionPolicy,
    FrameRecord,
    QASummaryPolicy,
)
from chess_gaze.gaze_observation import UNIGAZE_MODEL_ID
from chess_gaze.image_io import save_rgb_png
from chess_gaze.model_assets import (
    ModelAssetError,
    ResolvedModelAsset,
    load_model_registry,
    validate_required_assets,
)
from chess_gaze.qa_summary import (
    ArtifactValidationError,
    build_qa_summary,
    write_qa_summary,
)
from chess_gaze.scene_artifacts import build_scene_artifacts
from chess_gaze.scene_viewer import build_scene_viewer
from chess_gaze.unigaze_runtime import (
    PreparedUniGazeRuntime,
    UniGazeDevice,
    UniGazeRuntimeError,
    external_observer_inference_record,
    prepare_unigaze_runtime,
)
from chess_gaze.video_decode import (
    DecodedFrame,
    VideoDecodeError,
    inspect_video,
    iter_decoded_frames,
)
from chess_gaze.visualization import render_processed_frame

DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")
DEFAULT_APPROVED_LICENSES = frozenset({"MG-NC-RAI-2.0"})
RawFrameWriter = Callable[[Path, npt.NDArray[np.uint8]], str]
raw_frame_writer: RawFrameWriter = save_rgb_png
FrameErrorWriter = Callable[[TextIO, FrameRecord], None]
DefaultObserverBundleFactory = Callable[
    [list[ResolvedModelAsset], CalibrationRecord, RunLayout, object, bool],
    "ObserverBundle",
]


class FrameRecordObserver(Protocol):
    def __call__(self, frame: ObserverFrame) -> FrameRecord: ...


class FrameBatchRecordObserver(Protocol):
    def __call__(self, frames: Sequence[ObserverFrame]) -> Sequence[FrameRecord]: ...


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
    frame_batch_observer: FrameBatchRecordObserver | None = None
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class AnalysisProgressEvent:
    run_dir: Path
    completed_frames: int
    total_frames: int


ProgressCallback = Callable[[AnalysisProgressEvent], None]


@dataclass(frozen=True)
class AnalyzeRequest:
    video_path: Path
    output_root: Path | None = None
    models_root: Path | None = None
    config_path: Path | None = None
    unigaze_device: str | None = None
    unigaze_batch_size: int | None = None
    save_frame_images: bool | None = None
    save_crop_images: bool | None = None
    generate_qa_summary: bool = False
    model_registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH
    run_suffix: str | None = None
    resume: bool = True
    clock: Clock = utc_now
    progress_callback: ProgressCallback | None = None


@dataclass(frozen=True)
class AnalyzeResult:
    layout: RunLayout
    run_manifest_path: Path
    calibration_path: Path
    video_manifest_path: Path
    analysis_state_path: Path
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
    valid_sphere_hit_count: int


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
    save_frame_images: bool
    save_crop_images: bool
    generate_qa_summary: bool
    unigaze_device: str
    unigaze_batch_size: int


@dataclass(frozen=True)
class _PreparedDecodedFrame:
    decoded_frame: DecodedFrame
    observer_frame: ObserverFrame
    raw_frame_errors: list[ErrorRecord]


def analyze_video(
    request: AnalyzeRequest, observers: ObserverBundle | None = None
) -> AnalyzeResult:
    resolved = _resolve_request(request)
    _validate_input_video(resolved.video_path)

    try:
        inspection = inspect_video(resolved.video_path)
    except VideoDecodeError as exc:
        raise PipelineError(exc.code, str(exc)) from exc

    calibration = default_calibration()
    resolved_model_assets: list[ResolvedModelAsset] | None = None
    prepared_unigaze_runtime: PreparedUniGazeRuntime | None = None
    inference = external_observer_inference_record()
    frame_image_retention = FrameImageRetentionPolicy(
        save_frame_images=resolved.save_frame_images
    )
    crop_image_retention = CropImageRetentionPolicy(
        save_crop_images=resolved.save_crop_images
    )
    if observers is None:
        resolved_model_assets = _validate_model_assets(
            request.model_registry_path, resolved
        )
        gaze_asset = _asset_by_id(resolved_model_assets, UNIGAZE_MODEL_ID)
        try:
            prepared_unigaze_runtime = prepare_unigaze_runtime(
                gaze_asset,
                device=cast(UniGazeDevice, resolved.unigaze_device),
                batch_size=resolved.unigaze_batch_size,
                input_size_px=calibration.unigaze_input_size_px,
            )
        except UniGazeRuntimeError as exc:
            raise PipelineError(CliErrorCode.USAGE, str(exc)) from exc
        inference = prepared_unigaze_runtime.inference

    _estimate_disk_space(resolved.output_root)

    created_at = request.clock()
    runs_root = resolved.output_root / resolved.video_path.stem / "runs"
    layout = (
        find_latest_resumable_run(
            runs_root,
            resolved.video_path,
            inspection.video_manifest,
            calibration,
            inference,
            frame_image_retention,
            crop_image_retention,
            QASummaryPolicy(generate_qa_summary=resolved.generate_qa_summary),
        )
        if request.resume
        else None
    )
    resume_next_frame_index = 0
    frame_error_count = 0

    if layout is None:
        layout = create_run_layout(
            input_path=resolved.video_path,
            output_root=runs_root,
            clock=lambda: created_at,
            run_suffix=request.run_suffix,
        )
        write_initial_run_artifacts(
            layout,
            created_at=created_at,
            input_path=resolved.video_path,
            video_manifest=inspection.video_manifest,
            calibration=calibration,
            inference=inference,
            frame_image_retention=frame_image_retention,
            crop_image_retention=crop_image_retention,
            qa_summary_policy=QASummaryPolicy(
                generate_qa_summary=resolved.generate_qa_summary
            ),
        )
        analysis_state = new_analysis_state(
            layout,
            video_manifest=inspection.video_manifest,
            next_frame_index=0,
            status="processing",
            clock=request.clock,
        )
    else:
        preparation = prepare_resume_run(
            layout,
            inspection.video_manifest,
            clock=request.clock,
        )
        resume_next_frame_index = preparation.next_frame_index
        frame_error_count = sum(
            len(record.errors) for record in preparation.committed_records
        )
        analysis_state = preparation.analysis_state

    run_manifest_path = layout.run_dir / "run_manifest.json"
    calibration_path = layout.run_dir / "calibration.json"
    video_manifest_path = layout.run_dir / "video_manifest.json"
    analysis_state_path = write_analysis_state(layout, analysis_state)
    _report_analysis_progress(
        request,
        layout=layout,
        completed_frames=analysis_state.next_frame_index,
        total_frames=inspection.frame_count_decoded,
    )
    frames_jsonl_path = layout.records_dir / "frames.jsonl"
    errors_jsonl_path = layout.records_dir / "errors.jsonl"
    qa_summary_path = layout.run_dir / "qa_summary.json"

    if observers is None:
        if resolved_model_assets is None:
            raise AssertionError("resolved_model_assets must be set for default run")
        if prepared_unigaze_runtime is None:
            raise AssertionError("prepared_unigaze_runtime must be set for default run")
        observers = default_observer_bundle_factory(
            resolved_model_assets,
            calibration,
            layout,
            prepared_unigaze_runtime.model,
            resolved.save_crop_images,
        )

    try:
        decoded_frame_count = 0
        try:
            with (
                frames_jsonl_path.open("a", encoding="utf-8") as frames_handle,
                errors_jsonl_path.open("a", encoding="utf-8") as errors_handle,
            ):
                use_batch_accumulator = (
                    observers.frame_batch_observer is not None
                    and resolved.unigaze_batch_size > 1
                )
                pending_batch: list[DecodedFrame] = []
                for decoded_frame in iter_decoded_frames(resolved.video_path):
                    decoded_frame_count += 1
                    if decoded_frame.frame_index < resume_next_frame_index:
                        continue
                    if not use_batch_accumulator:
                        processed = [
                            _process_frame(
                                decoded_frame,
                                observers,
                                resolved,
                                layout,
                                errors_handle=errors_handle,
                            )
                        ]
                        committed_next_frame_index, committed_error_count = (
                            commit_processed_records(
                                processed,
                                frames_handle=frames_handle,
                                errors_handle=errors_handle,
                            )
                        )
                        frame_error_count += committed_error_count
                        analysis_state = update_analysis_state(
                            analysis_state,
                            next_frame_index=committed_next_frame_index,
                            status="processing",
                            clock=request.clock,
                        )
                        analysis_state_path = write_analysis_state(
                            layout, analysis_state
                        )
                        _report_analysis_progress(
                            request,
                            layout=layout,
                            completed_frames=committed_next_frame_index,
                            total_frames=inspection.frame_count_decoded,
                        )
                        continue
                    pending_batch.append(decoded_frame)
                    if len(pending_batch) < resolved.unigaze_batch_size:
                        continue
                    processed = _process_frame_batch(
                        pending_batch,
                        observers,
                        resolved,
                        layout,
                        errors_handle=errors_handle,
                    )
                    committed_next_frame_index, committed_error_count = (
                        commit_processed_records(
                            processed,
                            frames_handle=frames_handle,
                            errors_handle=errors_handle,
                        )
                    )
                    frame_error_count += committed_error_count
                    analysis_state = update_analysis_state(
                        analysis_state,
                        next_frame_index=committed_next_frame_index,
                        status="processing",
                        clock=request.clock,
                    )
                    analysis_state_path = write_analysis_state(layout, analysis_state)
                    _report_analysis_progress(
                        request,
                        layout=layout,
                        completed_frames=committed_next_frame_index,
                        total_frames=inspection.frame_count_decoded,
                    )
                    pending_batch = []

                if pending_batch:
                    processed = _process_frame_batch(
                        pending_batch,
                        observers,
                        resolved,
                        layout,
                        errors_handle=errors_handle,
                    )
                    committed_next_frame_index, committed_error_count = (
                        commit_processed_records(
                            processed,
                            frames_handle=frames_handle,
                            errors_handle=errors_handle,
                        )
                    )
                    frame_error_count += committed_error_count
                    analysis_state = update_analysis_state(
                        analysis_state,
                        next_frame_index=committed_next_frame_index,
                        status="processing",
                        clock=request.clock,
                    )
                    analysis_state_path = write_analysis_state(layout, analysis_state)
                    _report_analysis_progress(
                        request,
                        layout=layout,
                        completed_frames=committed_next_frame_index,
                        total_frames=inspection.frame_count_decoded,
                    )
        finally:
            if observers.close is not None:
                observers.close()

        scene_result = build_scene_artifacts(layout)
        viewer_result = build_scene_viewer(layout, scene_result)
    except (OSError, ValueError) as exc:
        analysis_state = update_analysis_state(
            analysis_state,
            status="failed",
            clock=request.clock,
        )
        analysis_state_path = write_analysis_state(layout, analysis_state)
        raise PipelineError(CliErrorCode.SCHEMA_VALIDATION_FAILED, str(exc)) from exc
    except Exception:
        analysis_state = update_analysis_state(
            analysis_state,
            status="failed",
            clock=request.clock,
        )
        analysis_state_path = write_analysis_state(layout, analysis_state)
        raise

    analysis_state = update_analysis_state(
        analysis_state,
        next_frame_index=inspection.frame_count_decoded,
        status="revalidating",
        clock=request.clock,
    )
    analysis_state_path = write_analysis_state(layout, analysis_state)
    try:
        qa_summary = build_qa_summary(layout)
    except ArtifactValidationError as exc:
        analysis_state = update_analysis_state(
            analysis_state,
            status="failed",
            clock=request.clock,
        )
        analysis_state_path = write_analysis_state(layout, analysis_state)
        raise PipelineError(exc.code, str(exc)) from exc
    analysis_state = update_analysis_state(
        analysis_state,
        next_frame_index=qa_summary.counts.decoded_frames,
        status=qa_summary.final_status,
        clock=request.clock,
    )
    analysis_state_path = write_analysis_state(layout, analysis_state)
    try:
        qa_summary = write_qa_summary(layout, qa_summary_path, qa_summary=qa_summary)
    except ArtifactValidationError as exc:
        analysis_state = update_analysis_state(
            analysis_state,
            status="revalidating",
            clock=request.clock,
        )
        analysis_state_path = write_analysis_state(layout, analysis_state)
        raise PipelineError(exc.code, str(exc)) from exc
    except Exception:
        analysis_state = update_analysis_state(
            analysis_state,
            status="revalidating",
            clock=request.clock,
        )
        analysis_state_path = write_analysis_state(layout, analysis_state)
        raise
    if not qa_summary.artifact_validation.schema_validation_passed:
        analysis_state = update_analysis_state(
            analysis_state,
            status="failed",
            clock=request.clock,
        )
        analysis_state_path = write_analysis_state(layout, analysis_state)
        raise PipelineError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            f"Run artifact validation failed; see {qa_summary_path}",
        )

    return AnalyzeResult(
        layout=layout,
        run_manifest_path=run_manifest_path,
        calibration_path=calibration_path,
        video_manifest_path=video_manifest_path,
        analysis_state_path=analysis_state_path,
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
        valid_sphere_hit_count=scene_result.valid_sphere_hit_count,
    )


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
            save_frame_images=request.save_frame_images,
            save_crop_images=request.save_crop_images,
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
        save_frame_images=resolved_config.save_frame_images,
        save_crop_images=resolved_config.save_crop_images,
        generate_qa_summary=request.generate_qa_summary,
        unigaze_device=resolved_config.unigaze_device,
        unigaze_batch_size=resolved_config.unigaze_batch_size,
    )


def _report_analysis_progress(
    request: AnalyzeRequest,
    *,
    layout: RunLayout,
    completed_frames: int,
    total_frames: int,
) -> None:
    if request.progress_callback is None:
        return
    request.progress_callback(
        AnalysisProgressEvent(
            run_dir=layout.run_dir,
            completed_frames=completed_frames,
            total_frames=total_frames,
        )
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
    gaze_model: object,
    save_crop_images: bool,
) -> ObserverBundle:
    from chess_gaze.face_observation import MediaPipeFaceObserver
    from chess_gaze.frame_observation import ModelBackedFrameObserver

    face_asset = _asset_by_id(resolved_assets, "mediapipe-face-landmarker")
    observer = ModelBackedFrameObserver(
        face_observer=MediaPipeFaceObserver(
            model_asset_path=face_asset.resolved_path,
            calibration=calibration,
        ),
        gaze_model=cast(Any, gaze_model),
        calibration=calibration,
        run_layout=run_layout,
        save_crop_images=save_crop_images,
    )
    return ObserverBundle(
        frame_observer=observer,
        frame_batch_observer=observer.observe_batch,
        close=observer.close,
    )


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
    [processed] = _process_frame_batch(
        [decoded_frame],
        observers,
        resolved,
        layout,
        errors_handle=errors_handle,
    )
    return processed


def _prepare_decoded_frame(
    decoded_frame: DecodedFrame,
    resolved: _ResolvedRequest,
    layout: RunLayout,
) -> _PreparedDecodedFrame:
    frame_errors: list[ErrorRecord] = []
    if resolved.save_frame_images:
        raw_path = layout.raw_frames_dir / f"{decoded_frame.frame_id}.png"
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
    return _PreparedDecodedFrame(
        decoded_frame=decoded_frame,
        observer_frame=observer_frame,
        raw_frame_errors=frame_errors,
    )


def _render_processed_frame_and_collect_errors(
    decoded_frame: DecodedFrame,
    record: FrameRecord,
    resolved: _ResolvedRequest,
    layout: RunLayout,
) -> tuple[FrameRecord, list[ErrorRecord]]:
    if not resolved.save_frame_images:
        return record, []

    frame_errors: list[ErrorRecord] = []
    processed_path = layout.processed_frames_dir / f"{decoded_frame.frame_id}.jpg"
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
    return record, frame_errors


def _process_frame_batch(
    decoded_frames: Sequence[DecodedFrame],
    observers: ObserverBundle,
    resolved: _ResolvedRequest,
    layout: RunLayout,
    *,
    errors_handle: TextIO,
) -> list[tuple[FrameRecord, list[ErrorRecord]]]:
    prepared = [
        _prepare_decoded_frame(decoded_frame, resolved, layout)
        for decoded_frame in decoded_frames
    ]
    if observers.frame_batch_observer is None:
        try:
            records = [
                observers.frame_observer(item.observer_frame) for item in prepared
            ]
        except ModelInferenceError as exc:
            raise PipelineError(CliErrorCode.USAGE, str(exc)) from exc
    else:
        try:
            records = list(
                observers.frame_batch_observer(
                    [item.observer_frame for item in prepared]
                )
            )
        except ModelInferenceError as exc:
            raise PipelineError(CliErrorCode.USAGE, str(exc)) from exc
    if len(records) != len(prepared):
        raise PipelineError(
            CliErrorCode.SCHEMA_VALIDATION_FAILED,
            (
                "Batch observer returned a different record count: "
                f"{len(records)} != {len(prepared)}"
            ),
        )

    processed: list[tuple[FrameRecord, list[ErrorRecord]]] = []
    for item, record in zip(prepared, records, strict=True):
        _validate_observer_record_identity(record, item.decoded_frame)
        record = _record_with_errors(record, item.raw_frame_errors)
        record, processed_errors = _render_processed_frame_and_collect_errors(
            item.decoded_frame,
            record,
            resolved,
            layout,
        )
        frame_errors = item.raw_frame_errors + processed_errors
        frame_error_writer(errors_handle, record)
        processed.append((record, frame_errors))
    return processed


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


def _validate_image_format(actual: str, expected: str) -> None:
    if actual.lower() != expected:
        raise PipelineError(
            CliErrorCode.USAGE,
            f"Unsupported image format {actual!r}; expected {expected!r}",
        )
