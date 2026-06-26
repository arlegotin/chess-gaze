from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.frame_records import FrameRecord, RunManifest, VideoManifest
from chess_gaze.image_io import atomic_write_bytes
from chess_gaze.scene_calibration import SceneAssumptions, default_scene_assumptions
from chess_gaze.scene_geometry import (
    RobustDirectionEstimate,
    RobustPointEstimate,
    back_project_eye_points,
    build_monitor_plane,
    build_scene_axis_basis,
    camera_point_to_scene,
    estimated_camera_model,
    intersect_ray_with_monitor,
    robust_main_direction,
    robust_scene_center,
    unigaze_ray_from_frame,
)
from chess_gaze.scene_records import (
    CoordinateFrame3D,
    SceneArtifactValidationRecord,
    SceneAssumptionRecord,
    SceneAxisBasisRecord,
    SceneCameraModel,
    SceneCenterEstimatorRecord,
    SceneCoordinateFramesRecord,
    SceneDirectionEstimatorRecord,
    SceneEyeMidpointRecord,
    SceneEyeRecord,
    SceneFrameCameraRecord,
    SceneFrameDiagnosticsRecord,
    SceneFrameRecord,
    SceneHeadRecord,
    SceneInvalidReason,
    SceneManifest,
    SceneMonitorHitBoundsRecord,
    SceneMonitorHitRecord,
    SceneMonitorPlaneRecord,
    SceneOrientationEstimatorRecord,
    SceneRobustEstimatorsRecord,
    SceneSourceArtifactsRecord,
    SceneSummary,
    SceneUniGazeRayRecord,
    SceneViewerDependencyRecord,
    UnitVector3D,
    Vector3D,
    ViewerHitPoint,
    ViewerSceneData,
)
from chess_gaze.viewer_dependencies import (
    THREE_CDN_PROVIDER,
    THREE_LICENSE,
    THREE_MODULE_URLS,
    THREE_NPM_INTEGRITY,
    THREE_PACKAGE_NAME,
    THREE_SOURCE,
    THREE_VERSION,
)


@dataclass(frozen=True)
class SceneArtifactPaths:
    scene_manifest_path: Path
    scene_summary_path: Path
    scene_frames_jsonl_path: Path


@dataclass(frozen=True)
class SceneArtifactResult:
    paths: SceneArtifactPaths
    scene_frame_count: int
    valid_monitor_hit_count: int
    viewer_data: ViewerSceneData
    manifest: SceneManifest
    summary: SceneSummary
    frames: list[SceneFrameRecord]


@dataclass(frozen=True)
class _FirstPassFrame:
    source: FrameRecord
    eye_pair_right_camera: UnitVector3D | None


def build_scene_artifacts(run_layout: RunLayout) -> SceneArtifactResult:
    paths = SceneArtifactPaths(
        scene_manifest_path=run_layout.run_dir / "scene" / "scene_manifest.json",
        scene_summary_path=run_layout.run_dir / "scene" / "scene_summary.json",
        scene_frames_jsonl_path=run_layout.records_dir / "scene_frames.jsonl",
    )
    source_frames_path = run_layout.records_dir / "frames.jsonl"
    run_manifest = _load_json_model(
        run_layout.run_dir / "run_manifest.json",
        RunManifest,
    )
    video_manifest = _load_json_model(
        run_layout.run_dir / "video_manifest.json",
        VideoManifest,
    )
    source_frames = _load_source_frames(source_frames_path)
    _validate_contiguous_frame_indices(source_frames)

    assumptions = default_scene_assumptions()
    camera = estimated_camera_model(
        frame_width=video_manifest.frame_width,
        frame_height=video_manifest.frame_height,
    )
    first_pass_frames: list[_FirstPassFrame] = []
    midpoint_candidates: list[Vector3D] = []
    ray_candidates: list[SceneUniGazeRayRecord] = []
    eye_pair_right_vectors: list[UnitVector3D] = []

    for source_frame in source_frames:
        projection = back_project_eye_points(source_frame, camera, assumptions)
        ray = unigaze_ray_from_frame(source_frame, projection.midpoint)
        eye_pair_right = _eye_pair_right_camera(
            projection.left_eye,
            projection.right_eye,
        )

        if projection.midpoint.valid and projection.midpoint.camera_point_m is not None:
            midpoint_candidates.append(projection.midpoint.camera_point_m)
        if ray.valid:
            ray_candidates.append(ray)
        if eye_pair_right is not None:
            eye_pair_right_vectors.append(eye_pair_right)
        first_pass_frames.append(
            _FirstPassFrame(
                source=source_frame,
                eye_pair_right_camera=eye_pair_right,
            )
        )

    scene_center = robust_scene_center(midpoint_candidates, assumptions)
    main_direction = robust_main_direction(ray_candidates, assumptions)
    axis_basis = build_scene_axis_basis(
        main_direction,
        eye_pair_right_vectors,
        assumptions,
    )
    monitor_plane = build_monitor_plane(
        scene_center,
        main_direction,
        axis_basis,
        assumptions,
    )

    scene_frames = [
        _build_scene_frame(
            source_frame=first_pass.source,
            camera=camera,
            assumptions=assumptions,
            scene_center=scene_center.point_camera_m,
            axis_basis=axis_basis,
            monitor_plane=monitor_plane,
        )
        for first_pass in first_pass_frames
    ]
    valid_monitor_hit_count = sum(
        1 for scene_frame in scene_frames if scene_frame.main_monitor_hit.valid
    )
    manifest = _build_manifest(
        run_layout=run_layout,
        run_manifest=run_manifest,
        video_manifest=video_manifest,
        camera=camera,
        assumptions=assumptions,
        scene_center=scene_center,
        main_direction=main_direction,
        axis_basis=axis_basis,
        monitor_plane=monitor_plane,
    )
    summary = _build_summary(
        run_id=run_manifest.run_id,
        decoded_frames=video_manifest.frame_count_decoded,
        scene_frames=scene_frames,
        scene_manifest_valid=True,
        scene_summary_valid=True,
        viewer_exists=False,
    )
    viewer_data = _viewer_scene_data_from_parts(
        run_id=run_manifest.run_id,
        source_video_path=video_manifest.source_path,
        frames=scene_frames,
        monitor_plane=monitor_plane,
        axis_basis=axis_basis,
        assumptions=assumptions.records,
        summary=summary,
    )

    _write_json(paths.scene_manifest_path, manifest)
    _write_json(paths.scene_summary_path, summary)
    _write_jsonl(paths.scene_frames_jsonl_path, scene_frames)

    return SceneArtifactResult(
        paths=paths,
        scene_frame_count=len(scene_frames),
        valid_monitor_hit_count=valid_monitor_hit_count,
        viewer_data=viewer_data,
        manifest=manifest,
        summary=summary,
        frames=scene_frames,
    )


def load_scene_frames(path: Path) -> list[SceneFrameRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        SceneFrameRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_viewer_scene_data(result: SceneArtifactResult) -> ViewerSceneData:
    return _viewer_scene_data_from_parts(
        run_id=result.manifest.run_id,
        source_video_path=result.manifest.source_video_path,
        frames=result.frames,
        monitor_plane=result.manifest.monitor_plane,
        axis_basis=result.manifest.axis_basis,
        assumptions=result.manifest.assumptions,
        summary=result.summary,
    )


def scene_result_with_viewer_exists(
    result: SceneArtifactResult, *, viewer_exists: bool
) -> SceneArtifactResult:
    artifact_validation = result.summary.artifact_validation.model_copy(
        update={"viewer_exists": viewer_exists}
    )
    summary = result.summary.model_copy(
        update={"artifact_validation": artifact_validation}
    )
    return replace(
        result,
        summary=summary,
        viewer_data=build_viewer_scene_data(replace(result, summary=summary)),
    )


def _build_scene_frame(
    *,
    source_frame: FrameRecord,
    camera: SceneCameraModel,
    assumptions: SceneAssumptions,
    scene_center: Vector3D,
    axis_basis: SceneAxisBasisRecord,
    monitor_plane: SceneMonitorPlaneRecord,
) -> SceneFrameRecord:
    projection = back_project_eye_points(source_frame, camera, assumptions)
    left_eye = _final_eye_record(
        projection.left_eye,
        scene_center=scene_center,
        axis_basis=axis_basis,
    )
    right_eye = _final_eye_record(
        projection.right_eye,
        scene_center=scene_center,
        axis_basis=axis_basis,
    )
    midpoint = _final_midpoint_record(
        projection.midpoint,
        scene_center=scene_center,
        axis_basis=axis_basis,
    )
    ray = _final_ray_record(
        unigaze_ray_from_frame(source_frame, midpoint),
        scene_center=scene_center,
        axis_basis=axis_basis,
    )
    monitor_hit = intersect_ray_with_monitor(ray, monitor_plane, assumptions)
    head = _head_record(
        source_frame=source_frame,
        midpoint=midpoint,
        assumptions=assumptions,
        scene_center=scene_center,
        axis_basis=axis_basis,
    )
    warnings = _scene_warnings(midpoint, ray, monitor_hit)
    diagnostics = SceneFrameDiagnosticsRecord(
        warnings=warnings,
        source_error_codes=[_enum_value(error.code) for error in source_frame.errors],
    )
    return SceneFrameRecord(
        frame_id=source_frame.frame_id,
        frame_index=source_frame.frame_index,
        timestamp_seconds=source_frame.timestamp_seconds,
        source_frame_status=source_frame.status,
        valid_for_scene_center=midpoint.valid,
        valid_for_main_monitor_direction=ray.valid,
        camera=SceneFrameCameraRecord(
            fx_px=camera.fx_px,
            fy_px=camera.fy_px,
            cx_px=camera.cx_px,
            cy_px=camera.cy_px,
            depth_source="interpupillary_distance_assumption",
        ),
        left_eye=left_eye,
        right_eye=right_eye,
        eye_midpoint=midpoint,
        head=head,
        unigaze_ray=ray,
        main_monitor_hit=monitor_hit,
        diagnostics=diagnostics,
    )


def _final_eye_record(
    eye: SceneEyeRecord,
    *,
    scene_center: Vector3D,
    axis_basis: SceneAxisBasisRecord,
) -> SceneEyeRecord:
    if not eye.valid or eye.camera_point_m is None:
        return eye
    return SceneEyeRecord(
        valid=True,
        image_px=eye.image_px,
        camera_m=eye.camera_point_m,
        scene_m=camera_point_to_scene(
            eye.camera_point_m,
            scene_center,
            axis_basis,
        ),
        source_reason_invalid=None,
        reason_invalid=None,
    )


def _final_midpoint_record(
    midpoint: SceneEyeMidpointRecord,
    *,
    scene_center: Vector3D,
    axis_basis: SceneAxisBasisRecord,
) -> SceneEyeMidpointRecord:
    if not midpoint.valid or midpoint.camera_point_m is None:
        return midpoint
    return SceneEyeMidpointRecord(
        valid=True,
        origin_policy=midpoint.origin_policy,
        camera_m=midpoint.camera_point_m,
        scene_m=camera_point_to_scene(
            midpoint.camera_point_m,
            scene_center,
            axis_basis,
        ),
        pupil_distance_px=midpoint.pupil_distance_px,
        estimated_depth_m=midpoint.estimated_depth_m,
        source_reason_invalid=None,
        reason_invalid=None,
    )


def _final_ray_record(
    ray: SceneUniGazeRayRecord,
    *,
    scene_center: Vector3D,
    axis_basis: SceneAxisBasisRecord,
) -> SceneUniGazeRayRecord:
    if not ray.valid or ray.origin_camera_m is None or ray.direction_camera is None:
        return ray
    return SceneUniGazeRayRecord(
        valid=True,
        source="appearance_gaze",
        origin_camera_m=ray.origin_camera_m,
        scene_m=camera_point_to_scene(
            ray.origin_camera_m,
            scene_center,
            axis_basis,
        ),
        direction_camera=ray.direction_camera,
        direction_scene=_camera_direction_to_scene(ray.direction_camera, axis_basis),
        direction_source=ray.direction_source,
        pitch_radians=ray.pitch_radians,
        yaw_radians=ray.yaw_radians,
        source_reason_invalid=None,
        reason_invalid=None,
    )


def _head_record(
    *,
    source_frame: FrameRecord,
    midpoint: SceneEyeMidpointRecord,
    assumptions: SceneAssumptions,
    scene_center: Vector3D,
    axis_basis: SceneAxisBasisRecord,
) -> SceneHeadRecord:
    if not midpoint.valid or midpoint.camera_point_m is None:
        return SceneHeadRecord(
            valid=False,
            ellipsoid_center_camera_m=None,
            scene_m=None,
            ellipsoid_radii_m=assumptions.head_ellipsoid_radius_m,
            yaw_radians=None,
            pitch_radians=None,
            roll_radians=None,
            orientation_source=None,
            source_reason_invalid=midpoint.source_reason_invalid,
            reason_invalid=midpoint.reason_invalid
            or SceneInvalidReason.EYE_MIDPOINT_INVALID,
        )

    offset_x, offset_y, offset_z = assumptions.head_center_from_eye_midpoint_m
    head_center_camera = Vector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=midpoint.camera_point_m.x + offset_x,
        y=midpoint.camera_point_m.y + offset_y,
        z=midpoint.camera_point_m.z + offset_z,
    )
    head_pose = source_frame.head_pose
    return SceneHeadRecord(
        valid=True,
        ellipsoid_center_camera_m=head_center_camera,
        scene_m=camera_point_to_scene(
            head_center_camera,
            scene_center,
            axis_basis,
        ),
        ellipsoid_radii_m=assumptions.head_ellipsoid_radius_m,
        yaw_radians=head_pose.yaw_radians if head_pose.valid else None,
        pitch_radians=head_pose.pitch_radians if head_pose.valid else None,
        roll_radians=head_pose.roll_radians if head_pose.valid else None,
        orientation_source="source_head_pose" if head_pose.valid else None,
        source_reason_invalid=(
            None if head_pose.valid else _enum_value(head_pose.reason_invalid)
        ),
        reason_invalid=None,
    )


def _build_manifest(
    *,
    run_layout: RunLayout,
    run_manifest: RunManifest,
    video_manifest: VideoManifest,
    camera: SceneCameraModel,
    assumptions: SceneAssumptions,
    scene_center: RobustPointEstimate,
    main_direction: RobustDirectionEstimate,
    axis_basis: SceneAxisBasisRecord,
    monitor_plane: SceneMonitorPlaneRecord,
) -> SceneManifest:
    del run_layout
    return SceneManifest(
        run_id=run_manifest.run_id,
        source_video_path=video_manifest.source_path,
        source_video_sha256=video_manifest.source_sha256,
        source_artifacts=SceneSourceArtifactsRecord(
            frame_records="records/frames.jsonl",
            scene_frame_records="records/scene_frames.jsonl",
            scene_summary="scene/scene_summary.json",
            viewer="viewer/scene-data.json",
        ),
        coordinate_frames=SceneCoordinateFramesRecord(
            math_frame=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
            scene_frame=CoordinateFrame3D.SCENE_PSEUDO_M,
            monitor_frame=CoordinateFrame3D.MONITOR_PLANE_PSEUDO_M,
            viewer_frame=CoordinateFrame3D.THREE_VIEW,
        ),
        camera_model=camera,
        assumptions=assumptions.records,
        robust_estimators=SceneRobustEstimatorsRecord(
            scene_center=SceneCenterEstimatorRecord(
                method="geometric_median_after_mad_screen",
                candidate_frame_count=scene_center.candidate_count,
                finite_candidate_frame_count=scene_center.finite_candidate_count,
                dropped_non_finite_frame_count=(scene_center.dropped_non_finite_count),
                inlier_frame_count=scene_center.inlier_count,
                mad_m=scene_center.mad_m,
                thresholds_m=scene_center.thresholds_m,
                iteration_count=scene_center.iteration_count,
                convergence_tolerance_m=scene_center.convergence_tolerance_m,
                fallback_used=scene_center.fallback_used,
                uncertainty=scene_center.uncertainty,
            ),
            main_unigaze_direction=SceneDirectionEstimatorRecord(
                method="angular_ransac_then_normalized_inlier_mean",
                candidate_frame_count=main_direction.candidate_count,
                finite_candidate_frame_count=main_direction.finite_candidate_count,
                inlier_frame_count=main_direction.inlier_count,
                inlier_angle_radians=main_direction.angle_threshold_radians,
                median_angular_residual_radians=(
                    main_direction.median_angular_residual_radians
                ),
                angular_residual_percentiles_radians=(
                    main_direction.angular_residual_percentiles_radians
                ),
                fallback_used=main_direction.fallback_used,
                uncertainty=main_direction.uncertainty,
            ),
            scene_orientation=SceneOrientationEstimatorRecord(
                method="camera_stable_right_up_back_axes",
                candidate_frame_count=0,
                fallbacks=axis_basis.fallbacks,
            ),
        ),
        scene_center_camera_m=scene_center.point_camera_m,
        scene_axes_camera=axis_basis,
        main_monitor_plane=monitor_plane,
        viewer=SceneViewerDependencyRecord(
            library=THREE_PACKAGE_NAME,
            version=THREE_VERSION,
            source=THREE_SOURCE,
            license=THREE_LICENSE,
            dist_integrity=THREE_NPM_INTEGRITY,
            cdn_provider=THREE_CDN_PROVIDER,
            module_urls=THREE_MODULE_URLS,
        ),
        generated_at_utc=_utc_timestamp(),
    )


def _build_summary(
    *,
    run_id: str,
    decoded_frames: int,
    scene_frames: list[SceneFrameRecord],
    scene_manifest_valid: bool,
    scene_summary_valid: bool,
    viewer_exists: bool,
) -> SceneSummary:
    invalid_monitor_hit_reasons = Counter[str]()
    for scene_frame in scene_frames:
        hit = scene_frame.main_monitor_hit
        if not hit.valid and hit.reason_invalid is not None:
            invalid_monitor_hit_reasons[_enum_value(hit.reason_invalid)] += 1

    return SceneSummary(
        run_id=run_id,
        decoded_frames=decoded_frames,
        scene_frame_records=len(scene_frames),
        valid_eye_midpoint_frames=sum(
            1 for scene_frame in scene_frames if scene_frame.eye_midpoint.valid
        ),
        valid_unigaze_ray_frames=sum(
            1 for scene_frame in scene_frames if scene_frame.unigaze_ray.valid
        ),
        valid_monitor_hit_frames=sum(
            1 for scene_frame in scene_frames if scene_frame.main_monitor_hit.valid
        ),
        invalid_monitor_hit_reasons=dict(sorted(invalid_monitor_hit_reasons.items())),
        monitor_hit_bounds=_monitor_hit_bounds(scene_frames),
        representative_scene_warning_frame_ids=_representative_warning_frame_ids(
            scene_frames
        ),
        artifact_validation=SceneArtifactValidationRecord(
            scene_frame_count_matches_decoded=len(scene_frames) == decoded_frames,
            viewer_exists=viewer_exists,
            scene_manifest_valid=scene_manifest_valid,
            scene_summary_valid=scene_summary_valid,
        ),
    )


def _viewer_scene_data_from_parts(
    *,
    run_id: str,
    source_video_path: str,
    frames: list[SceneFrameRecord],
    monitor_plane: SceneMonitorPlaneRecord,
    axis_basis: SceneAxisBasisRecord,
    assumptions: list[SceneAssumptionRecord],
    summary: SceneSummary,
) -> ViewerSceneData:
    return ViewerSceneData(
        run_id=run_id,
        source_video_stem=Path(source_video_path).stem,
        frame_count=len(frames),
        frames=frames,
        valid_hit_points=_valid_hit_points(frames),
        monitor_plane=monitor_plane,
        axis_basis=axis_basis,
        assumptions=assumptions,
        summary=summary,
    )


def _valid_hit_points(frames: list[SceneFrameRecord]) -> list[ViewerHitPoint]:
    hit_points: list[ViewerHitPoint] = []
    for frame in frames:
        hit = frame.main_monitor_hit
        if not hit.valid:
            continue
        if hit.point_scene_m is None or hit.u_m is None or hit.v_m is None:
            raise ValueError("valid monitor hit is missing persisted viewer fields")
        if hit.within_physical_monitor is None or hit.within_extended_plane is None:
            raise ValueError("valid monitor hit is missing bounds flags")
        hit_points.append(
            ViewerHitPoint(
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                point_scene_m=hit.point_scene_m,
                u_m=hit.u_m,
                v_m=hit.v_m,
                within_physical_monitor=hit.within_physical_monitor,
                within_extended_plane=hit.within_extended_plane,
            )
        )
    return hit_points


def _monitor_hit_bounds(frames: list[SceneFrameRecord]) -> SceneMonitorHitBoundsRecord:
    valid_uv = [
        (hit.u_m, hit.v_m)
        for hit in (frame.main_monitor_hit for frame in frames)
        if hit.valid and hit.u_m is not None and hit.v_m is not None
    ]
    if not valid_uv:
        return SceneMonitorHitBoundsRecord(
            u_min_m=0.0,
            u_max_m=0.0,
            v_min_m=0.0,
            v_max_m=0.0,
        )
    return SceneMonitorHitBoundsRecord(
        u_min_m=min(u for u, _v in valid_uv),
        u_max_m=max(u for u, _v in valid_uv),
        v_min_m=min(v for _u, v in valid_uv),
        v_max_m=max(v for _u, v in valid_uv),
    )


def _representative_warning_frame_ids(
    scene_frames: list[SceneFrameRecord],
) -> list[str]:
    return [
        scene_frame.frame_id
        for scene_frame in scene_frames
        if scene_frame.diagnostics.warnings
    ][:10]


def _scene_warnings(
    midpoint: SceneEyeMidpointRecord,
    ray: SceneUniGazeRayRecord,
    monitor_hit: SceneMonitorHitRecord,
) -> list[str]:
    warnings: list[str] = []
    if not midpoint.valid and midpoint.reason_invalid is not None:
        warnings.append(f"eye_midpoint:{_enum_value(midpoint.reason_invalid)}")
    if not ray.valid and ray.reason_invalid is not None:
        warnings.append(f"unigaze_ray:{_enum_value(ray.reason_invalid)}")
    if not monitor_hit.valid and monitor_hit.reason_invalid is not None:
        warnings.append(f"main_monitor_hit:{_enum_value(monitor_hit.reason_invalid)}")
    return warnings


def _camera_direction_to_scene(
    direction: UnitVector3D,
    axis_basis: SceneAxisBasisRecord,
) -> UnitVector3D:
    camera_direction = (direction.x, direction.y, direction.z)
    scene_direction = (
        _dot(camera_direction, _vector_tuple(axis_basis.right_camera)),
        _dot(camera_direction, _vector_tuple(axis_basis.up_camera)),
        _dot(camera_direction, _vector_tuple(axis_basis.back_camera)),
    )
    normalized = _normalize(scene_direction)
    if normalized is None:
        raise ValueError("camera direction cannot be transformed to scene basis")
    return UnitVector3D(
        space=CoordinateFrame3D.SCENE_PSEUDO_M,
        x=normalized[0],
        y=normalized[1],
        z=normalized[2],
    )


def _eye_pair_right_camera(
    left_eye: SceneEyeRecord,
    right_eye: SceneEyeRecord,
) -> UnitVector3D | None:
    if (
        not left_eye.valid
        or not right_eye.valid
        or left_eye.camera_point_m is None
        or right_eye.camera_point_m is None
    ):
        return None
    right_vector = (
        right_eye.camera_point_m.x - left_eye.camera_point_m.x,
        right_eye.camera_point_m.y - left_eye.camera_point_m.y,
        right_eye.camera_point_m.z - left_eye.camera_point_m.z,
    )
    normalized = _normalize(right_vector)
    if normalized is None:
        return None
    return UnitVector3D(
        space=CoordinateFrame3D.CAMERA_OPENCV_PSEUDO_M,
        x=normalized[0],
        y=normalized[1],
        z=normalized[2],
    )


def _load_json_model[T](path: Path, model_type: type[T]) -> T:
    return cast(
        T,
        model_type.model_validate_json(path.read_text(encoding="utf-8")),  # type: ignore[attr-defined]
    )


def _load_source_frames(path: Path) -> list[FrameRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        FrameRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_contiguous_frame_indices(frames: list[FrameRecord]) -> None:
    for expected_index, frame in enumerate(frames):
        if frame.frame_index != expected_index:
            raise ValueError(
                "source frame indices must be contiguous from zero; "
                f"expected {expected_index}, got {frame.frame_index}"
            )


def _write_json(path: Path, model: object) -> None:
    payload = model.model_dump(mode="json", by_alias=True)  # type: ignore[attr-defined]
    atomic_write_bytes(path, _json_bytes(payload))


def _write_jsonl(path: Path, records: list[SceneFrameRecord]) -> None:
    lines = [
        json.dumps(
            record.model_dump(mode="json", by_alias=True),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    atomic_write_bytes(path, ("\n".join(lines) + ("\n" if lines else "")).encode())


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _enum_value(value: object) -> str:
    enum_value = getattr(value, "value", value)
    if enum_value is None:
        return ""
    return str(enum_value)


def _vector_tuple(vector: UnitVector3D | Vector3D) -> tuple[float, float, float]:
    return (vector.x, vector.y, vector.z)


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return (left[0] * right[0]) + (left[1] * right[1]) + (left[2] * right[2])


def _normalize(
    vector: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    if not all(math.isfinite(value) for value in vector):
        return None
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 0.0:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)
