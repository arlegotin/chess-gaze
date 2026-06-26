# 3D Scene Artifact Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `chess-gaze analyze` so each completed run writes strict 3D scene artifacts and a local 3D viewer that reconstructs the streamer's head, eyes, UniGaze ray, inferred monitor plane, and every valid gaze hit point from the analyzed video.

**Architecture:** Keep the existing per-frame analysis pipeline stable and add a deep scene-artifact layer after `records/frames.jsonl` is complete and before `qa_summary.json` is built. New scene modules own strict schemas, adult-male and desktop-monitor assumptions, robust pseudo-metric geometry, scene artifact writing, and static viewer generation. Historical implementation used local vendored Three.js assets; ADR-0003 supersedes that detail with pinned remote Three.js module loading.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, mypy, NumPy, Pydantic v2, existing PyAV/MediaPipe/UniGaze pipeline, local static HTML/CSS/JavaScript, Three.js `0.185.0` vendored from the npm tarball recorded in the approved spec.

> Historical note, 2026-06-26: ADR-0003 supersedes this plan's local-vendored
> Three.js constraint. Current generated viewers load Three.js `0.185.0` from
> pinned jsDelivr npm module URLs and no longer copy `viewer/vendor/`.
>
> Historical note, 2026-06-26: the scene-axis and monitor-normal guidance in
> this plan was superseded by
> `docs/superpowers/plans/2026-06-26-anatomical-scene-coordinate-repair.md`.
> Current scene axes are anatomical frontal-webcam right/up/back columns
> `[-1,0,0]`, `[0,-1,0]`, `[0,0,1]`; the scene ray flips UniGaze canonical Z
> into a physical eye-to-monitor ray, and robust dominant UniGaze direction
> places the monitor center only.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Use installed Superpowers skills for implementation flow; this plan is written for subagent-driven development.
- Approved spec: `docs/superpowers/specs/2026-06-26-3d-scene-artifact-viewer-design.md`.
- Preserve the existing `FrameRecord` schema unless a necessary scene diagnostic cannot be represented in the new scene layer. If `FrameRecord` changes, update every fake record and schema test in the same task.
- Build scene artifacts only from validated `records/frames.jsonl`, `run_manifest.json`, and `video_manifest.json`. Do not infer scene records from processed JPEG overlays.
- Write scene artifacts after the frame loop closes `frames.jsonl` and `errors.jsonl`, then generate viewer files, then build `qa_summary.json`. QA byte counts are invalid if viewer files are written after QA.
- Every decoded frame must produce exactly one `records/scene_frames.jsonl` line with the same `frame_id` and `frame_index` identity as `records/frames.jsonl`.
- Do not drop, sample, smooth, average across time, deduplicate, merge, clamp, cluster, or heatmap-substitute monitor hit points. A valid forward ray-plane hit produces exactly one persisted point.
- Use `appearance_gaze` as the UniGaze source for scene rays. Do not substitute `recommended_gaze`.
- Superseded by `docs/superpowers/plans/2026-06-26-anatomical-scene-coordinate-repair.md`: scene rays preserve `pitch_yaw_to_unit_vector()` X/Y overlay semantics, negate vector Y when entering `camera_opencv_pseudo_m`, and negate canonical Z so the scene ray is a physical frontal-webcam eye-to-monitor direction.
- No scene artifact may contain NaN, Infinity, silently coerced unknown enum strings, or unknown JSON fields.
- Keep all new modules deep. Do not create pass-through packages or generic `core`, `services`, `engine`, or `domain` layers.
- Constants from the spec must be centralized, tested for exact values, and persisted into `scene/scene_manifest.json` with unit, source, and uncertainty metadata.
- Use `nakamura_1.mp4` during development, not only at closeout. Current verified facts: `artifacts/input/nakamura_1.mp4`, 1920x1080, 1973 decoded frames, sha256 `eca8b3c81c2bd33a639dbe4926924b9462b6f90fd8fd14bda3bae97b956a1a45`.
- Existing real-video tests still reference `artifacts/input/test_1.mp4` and `artifacts/input/test_2.mp4`. Do not weaken those tests globally. Add `nakamura_1.mp4` checkpoints for this feature and report absent legacy media separately when full gates fail.
- Real model runs may need unsandboxed execution on this machine because README documents MediaPipe native macOS GL/Metal failures inside the managed sandbox. If blocked, capture the exact command and error.
- Superseded by ADR-0003 for Three.js runtime modules: do not vendor Three.js
  locally. Load only pinned jsDelivr URLs for `three@0.185.0`. Remote telemetry,
  uploaded frames, uploaded model data, and frontend build services remain
  disallowed.
- Do not add `package.json`, `node_modules`, Playwright, or a frontend build unless an ADR or spec update compares that dependency against the task requirements. The initial viewer is a Python-packaged static asset generator.
- Axis convention correction, superseded by the anatomical scene coordinate
  repair: implementation must persist a right-handed transform basis with
  anatomical frontal-webcam columns `[scene_right_camera, scene_up_camera,
  scene_back_camera] = [[-1,0,0], [0,-1,0], [0,0,1]]`. The robust dominant
  UniGaze direction remains semantic monitor-center evidence, not a scene-axis
  input. Tests must prove determinant near `+1`, image-right/his-left maps to
  negative scene X, and monitor-directed gaze maps to negative scene Z.

---

## File Structure

- Create `src/chess_gaze/scene_calibration.py` for scene constants and assumption records.
- Create `src/chess_gaze/scene_records.py` for strict scene manifest, scene frame, viewer-data, and summary schemas.
- Create `src/chess_gaze/scene_geometry.py` for pseudo-metric back-projection, robust estimators, scene axes, monitor plane construction, coordinate transforms, and ray-plane intersection.
- Create `src/chess_gaze/scene_artifacts.py` for reading run/frame artifacts and writing scene JSON artifacts.
- Create `src/chess_gaze/scene_viewer.py` for viewer data generation, static asset copying, and localhost static-server helpers.
- Create `src/chess_gaze/viewer_assets/` for packaged `index.html`,
  `scene_viewer.js`, `styles.css`, and viewer dependency metadata. Historical
  local-vendor instructions below are superseded by ADR-0003.
- Modify `src/chess_gaze/artifact_runs.py` to add `scene_dir` and `viewer_dir` to `RunLayout`.
- Modify `src/chess_gaze/pipeline.py` to call scene artifact and viewer generation before QA summary validation and return scene/viewer paths in `AnalyzeResult`.
- Modify `src/chess_gaze/qa_summary.py` to validate scene artifacts, count scene frames, include scene/viewer bytes, and keep `QASummary.source_artifacts` consistent with `ArtifactValidationResult.source_artifacts`.
- Modify `src/chess_gaze/cli.py` to print viewer path after `analyze` and add `chess-gaze view <run-dir>`.
- Modify `pyproject.toml` only for packaged viewer assets if resource tests prove Hatch does not include them automatically.
- Modify `README.md` for artifact layout and viewer usage.
- Modify `docs/development/architecture/source-layout.md` because its current package map is stale.
- Create implementation closeout `docs/superpowers/closeouts/2026-06-26-3d-scene-artifact-viewer.md`.
- Add tests under `tests/chess_gaze/` and package metadata tests under `tests/` as named in each task.

---

## Shared Type Contracts

Implementers may add private helpers, but later tasks rely on these public names
and field meanings.

`src/chess_gaze/scene_calibration.py` must expose:

```python
from __future__ import annotations

from typing import Literal

from chess_gaze.geometry import StrictSchemaModel

DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M = 0.063
DEFAULT_MONITOR_DISTANCE_FROM_EYES_M = 0.700
DEFAULT_MONITOR_WIDTH_M = 0.600
DEFAULT_MONITOR_HEIGHT_M = 0.340
DEFAULT_EXTENDED_PLANE_SCALE = 3.0
DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M = 0.090
DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M = 0.120
DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M = 0.100
DEFAULT_EYE_SPHERE_RADIUS_M = 0.012
DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M = (0.0, 0.035, 0.020)
RAY_PLANE_PARALLEL_EPSILON = 1e-6
DEFAULT_SCENE_CENTER_CAMERA_M = (0.0, 0.0, 0.650)
SCENE_CENTER_MIN_AXIS_TOLERANCE_M = 0.015
MIN_SCENE_CENTER_INLIER_FRAMES = 5
MIN_MAIN_DIRECTION_INLIER_FRAMES = 5
DIRECTION_INLIER_ANGLE_RADIANS = 0.35


class SceneAssumptionRecord(StrictSchemaModel):
    name: str
    value: float | int | tuple[float, float, float]
    unit: str
    source: str
    uncertainty: Literal["low", "medium", "high"]


class SceneAssumptions(StrictSchemaModel):
    adult_male_interpupillary_distance_m: float
    monitor_distance_from_eyes_m: float
    monitor_width_m: float
    monitor_height_m: float
    extended_plane_scale: float
    head_ellipsoid_radius_m: tuple[float, float, float]
    eye_sphere_radius_m: float
    head_center_from_eye_midpoint_m: tuple[float, float, float]
    ray_plane_parallel_epsilon: float
    default_scene_center_camera_m: tuple[float, float, float]
    scene_center_min_axis_tolerance_m: float
    min_scene_center_inlier_frames: int
    min_main_direction_inlier_frames: int
    direction_inlier_angle_radians: float
    records: list[SceneAssumptionRecord]


def default_scene_assumptions() -> SceneAssumptions:
    records = [
        SceneAssumptionRecord(
            name="DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M",
            value=DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M,
            unit="m",
            source="adult_male_default",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_MONITOR_DISTANCE_FROM_EYES_M",
            value=DEFAULT_MONITOR_DISTANCE_FROM_EYES_M,
            unit="m",
            source="desktop_monitor_default",
            uncertainty="high",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_MONITOR_WIDTH_M",
            value=DEFAULT_MONITOR_WIDTH_M,
            unit="m",
            source="desktop_monitor_default",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_MONITOR_HEIGHT_M",
            value=DEFAULT_MONITOR_HEIGHT_M,
            unit="m",
            source="desktop_monitor_default",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_EXTENDED_PLANE_SCALE",
            value=DEFAULT_EXTENDED_PLANE_SCALE,
            unit="multiplier",
            source="viewer_default",
            uncertainty="low",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_HEAD_ELLIPSOID_RADIUS_M",
            value=(
                DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M,
                DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M,
                DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M,
            ),
            unit="m",
            source="adult_male_default",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_EYE_SPHERE_RADIUS_M",
            value=DEFAULT_EYE_SPHERE_RADIUS_M,
            unit="m",
            source="adult_male_default",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M",
            value=DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M,
            unit="m_in_head_local_axes",
            source="adult_male_default",
            uncertainty="high",
        ),
        SceneAssumptionRecord(
            name="RAY_PLANE_PARALLEL_EPSILON",
            value=RAY_PLANE_PARALLEL_EPSILON,
            unit="unitless",
            source="algorithm_constant",
            uncertainty="low",
        ),
        SceneAssumptionRecord(
            name="DEFAULT_SCENE_CENTER_CAMERA_M",
            value=DEFAULT_SCENE_CENTER_CAMERA_M,
            unit="camera_opencv_pseudo_m",
            source="fallback_default",
            uncertainty="high",
        ),
        SceneAssumptionRecord(
            name="SCENE_CENTER_MIN_AXIS_TOLERANCE_M",
            value=SCENE_CENTER_MIN_AXIS_TOLERANCE_M,
            unit="camera_opencv_pseudo_m",
            source="algorithm_constant",
            uncertainty="medium",
        ),
        SceneAssumptionRecord(
            name="MIN_SCENE_CENTER_INLIER_FRAMES",
            value=MIN_SCENE_CENTER_INLIER_FRAMES,
            unit="frames",
            source="algorithm_constant",
            uncertainty="low",
        ),
        SceneAssumptionRecord(
            name="MIN_MAIN_DIRECTION_INLIER_FRAMES",
            value=MIN_MAIN_DIRECTION_INLIER_FRAMES,
            unit="frames",
            source="algorithm_constant",
            uncertainty="low",
        ),
        SceneAssumptionRecord(
            name="DIRECTION_INLIER_ANGLE_RADIANS",
            value=DIRECTION_INLIER_ANGLE_RADIANS,
            unit="radians",
            source="algorithm_constant",
            uncertainty="medium",
        ),
    ]
    return SceneAssumptions(
        adult_male_interpupillary_distance_m=(
            DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M
        ),
        monitor_distance_from_eyes_m=DEFAULT_MONITOR_DISTANCE_FROM_EYES_M,
        monitor_width_m=DEFAULT_MONITOR_WIDTH_M,
        monitor_height_m=DEFAULT_MONITOR_HEIGHT_M,
        extended_plane_scale=DEFAULT_EXTENDED_PLANE_SCALE,
        head_ellipsoid_radius_m=(
            DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M,
            DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M,
            DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M,
        ),
        eye_sphere_radius_m=DEFAULT_EYE_SPHERE_RADIUS_M,
        head_center_from_eye_midpoint_m=(
            DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M
        ),
        ray_plane_parallel_epsilon=RAY_PLANE_PARALLEL_EPSILON,
        default_scene_center_camera_m=DEFAULT_SCENE_CENTER_CAMERA_M,
        scene_center_min_axis_tolerance_m=SCENE_CENTER_MIN_AXIS_TOLERANCE_M,
        min_scene_center_inlier_frames=MIN_SCENE_CENTER_INLIER_FRAMES,
        min_main_direction_inlier_frames=MIN_MAIN_DIRECTION_INLIER_FRAMES,
        direction_inlier_angle_radians=DIRECTION_INLIER_ANGLE_RADIANS,
        records=records,
    )
```

The final implementation may add private validators, but it must preserve these
public fields and persisted assumption names.

`src/chess_gaze/scene_records.py` must expose:

```python
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from chess_gaze.geometry import Point2D, StrictSchemaModel
from chess_gaze.scene_calibration import SceneAssumptionRecord


class CoordinateFrame3D(StrEnum):
    IMAGE_PX = "image_px"
    CAMERA_OPENCV_PSEUDO_M = "camera_opencv_pseudo_m"
    SCENE_PSEUDO_M = "scene_pseudo_m"
    MONITOR_PLANE_PSEUDO_M = "monitor_plane_pseudo_m"
    THREE_VIEW = "three_view"


class SceneInvalidReason(StrEnum):
    LEFT_EYE_INVALID = "LEFT_EYE_INVALID"
    RIGHT_EYE_INVALID = "RIGHT_EYE_INVALID"
    EYE_MIDPOINT_INVALID = "EYE_MIDPOINT_INVALID"
    UNIGAZE_INVALID = "UNIGAZE_INVALID"
    RAY_PARALLEL_TO_MONITOR = "RAY_PARALLEL_TO_MONITOR"
    RAY_COPLANAR_WITH_MONITOR = "RAY_COPLANAR_WITH_MONITOR"
    RAY_INTERSECTION_NON_FINITE = "RAY_INTERSECTION_NON_FINITE"
    RAY_INTERSECTION_BEHIND_ORIGIN = "RAY_INTERSECTION_BEHIND_ORIGIN"
    SCENE_CENTER_INSUFFICIENT_INLIERS = "SCENE_CENTER_INSUFFICIENT_INLIERS"
    MAIN_DIRECTION_INSUFFICIENT_INLIERS = "MAIN_DIRECTION_INSUFFICIENT_INLIERS"
    SCENE_AXIS_DEGENERATE = "SCENE_AXIS_DEGENERATE"
    MONITOR_PLANE_DEGENERATE = "MONITOR_PLANE_DEGENERATE"
    NON_FINITE_INPUT = "NON_FINITE_INPUT"


class Vector3D(StrictSchemaModel):
    space: CoordinateFrame3D
    x: float
    y: float
    z: float


class UnitVector3D(Vector3D):
    pass


class SceneCameraModel(StrictSchemaModel):
    frame_width_px: int
    frame_height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    model: Literal["estimated_pinhole_from_frame_size"]


class SceneEyeRecord(StrictSchemaModel):
    valid: bool
    image_px: Point2D | None
    camera_point_m: Vector3D | None
    scene_point_m: Vector3D | None
    source_reason_invalid: str | None
    reason_invalid: SceneInvalidReason | None


class SceneEyeMidpointRecord(StrictSchemaModel):
    valid: bool
    camera_point_m: Vector3D | None
    scene_point_m: Vector3D | None
    pupil_distance_px: float | None
    estimated_depth_m: float | None
    reason_invalid: SceneInvalidReason | None


class SceneHeadRecord(StrictSchemaModel):
    valid: bool
    ellipsoid_center_scene_m: Vector3D | None
    radii_m: tuple[float, float, float]
    reason_invalid: SceneInvalidReason | None


class SceneUniGazeRayRecord(StrictSchemaModel):
    valid: bool
    source: Literal["appearance_gaze"]
    origin_camera_m: Vector3D | None
    origin_scene_m: Vector3D | None
    direction_camera: UnitVector3D | None
    direction_scene: UnitVector3D | None
    pitch_radians: float | None
    yaw_radians: float | None
    reason_invalid: SceneInvalidReason | None


class SceneMonitorHitRecord(StrictSchemaModel):
    valid: bool
    point_camera_m: Vector3D | None
    point_scene_m: Vector3D | None
    u_m: float | None
    v_m: float | None
    t: float | None
    denominator: float | None
    signed_distance_m: float | None
    within_physical_monitor: bool | None
    within_extended_plane: bool | None
    reason_invalid: SceneInvalidReason | None


class SceneAxisBasisRecord(StrictSchemaModel):
    right_camera: UnitVector3D
    up_camera: UnitVector3D
    back_camera: UnitVector3D
    forward_camera: UnitVector3D
    determinant_right_up_back: float
    convention: Literal["right_up_back_columns_right_handed"]
    fallbacks: list[str]


class SceneMonitorPlaneRecord(StrictSchemaModel):
    center_camera_m: Vector3D
    center_scene_m: Vector3D
    normal_camera: UnitVector3D
    right_camera: UnitVector3D
    up_camera: UnitVector3D
    width_m: float
    height_m: float
    extended_width_m: float
    extended_height_m: float
    distance_from_scene_center_m: float


class SceneFrameRecord(StrictSchemaModel):
    schema_version: Literal["gaze-scene-frame-v1"] = "gaze-scene-frame-v1"
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    left_eye: SceneEyeRecord
    right_eye: SceneEyeRecord
    eye_midpoint: SceneEyeMidpointRecord
    head: SceneHeadRecord
    unigaze_ray: SceneUniGazeRayRecord
    main_monitor_hit: SceneMonitorHitRecord
    diagnostics: dict[str, str | int | float | bool | None]


class SceneManifest(StrictSchemaModel):
    schema_version: Literal["gaze-scene-manifest-v1"] = "gaze-scene-manifest-v1"
    run_id: str
    source_video_path: str
    source_video_sha256: str
    camera_model: SceneCameraModel
    assumptions: list[SceneAssumptionRecord]
    scene_center_camera_m: Vector3D
    axis_basis: SceneAxisBasisRecord
    monitor_plane: SceneMonitorPlaneRecord
    robust_estimators: dict[str, object]
    viewer_dependency: dict[str, object]


class SceneSummary(StrictSchemaModel):
    schema_version: Literal["gaze-scene-summary-v1"] = "gaze-scene-summary-v1"
    run_id: str
    decoded_frames: int
    scene_frame_records: int
    valid_eye_midpoint_frames: int
    valid_unigaze_ray_frames: int
    valid_monitor_hit_frames: int
    invalid_reason_counts: dict[str, int]
    representative_invalid_frame_ids: list[str]
    count_validation_passed: bool


class ViewerHitPoint(StrictSchemaModel):
    frame_id: str
    frame_index: int
    point_scene_m: Vector3D
    u_m: float
    v_m: float
    within_physical_monitor: bool
    within_extended_plane: bool


class ViewerSceneData(StrictSchemaModel):
    schema_version: Literal["gaze-scene-viewer-data-v1"] = (
        "gaze-scene-viewer-data-v1"
    )
    run_id: str
    source_video_stem: str
    frame_count: int
    frames: list[SceneFrameRecord]
    valid_hit_points: list[ViewerHitPoint]
    monitor_plane: SceneMonitorPlaneRecord
    axis_basis: SceneAxisBasisRecord
    assumptions: list[SceneAssumptionRecord]
    summary: SceneSummary
```

`src/chess_gaze/scene_geometry.py` must expose these dataclasses:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chess_gaze.frame_records import FrameRecord
from chess_gaze.scene_calibration import SceneAssumptions
from chess_gaze.scene_records import (
    SceneAxisBasisRecord,
    SceneCameraModel,
    SceneEyeMidpointRecord,
    SceneFrameRecord,
    SceneMonitorHitRecord,
    SceneMonitorPlaneRecord,
    SceneUniGazeRayRecord,
    UnitVector3D,
    Vector3D,
)


@dataclass(frozen=True)
class SceneEyePairProjection:
    left_eye_valid: bool
    right_eye_valid: bool
    midpoint: SceneEyeMidpointRecord
    diagnostics: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class RobustPointEstimate:
    point_camera_m: Vector3D
    candidate_count: int
    finite_candidate_count: int
    inlier_count: int
    mad_m: tuple[float, float, float]
    thresholds_m: tuple[float, float, float]
    iteration_count: int
    fallback_used: bool
    uncertainty: str


@dataclass(frozen=True)
class RobustDirectionEstimate:
    direction_camera: UnitVector3D
    candidate_count: int
    finite_candidate_count: int
    inlier_count: int
    angle_threshold_radians: float
    median_angular_residual_radians: float | None
    fallback_used: bool
    uncertainty: str
```

`src/chess_gaze/scene_geometry.py` must expose these public signatures:

- `estimated_camera_model(frame_width: int, frame_height: int) -> SceneCameraModel`
- `back_project_eye_points(frame_record: FrameRecord, camera: SceneCameraModel, assumptions: SceneAssumptions) -> SceneEyePairProjection`
- `robust_scene_center(points: Sequence[Vector3D], assumptions: SceneAssumptions) -> RobustPointEstimate`
- `unigaze_ray_from_frame(frame_record: FrameRecord, midpoint: SceneEyeMidpointRecord) -> SceneUniGazeRayRecord`
- `robust_main_direction(rays: Sequence[SceneUniGazeRayRecord], assumptions: SceneAssumptions) -> RobustDirectionEstimate`
- `build_scene_axis_basis(direction: RobustDirectionEstimate, eye_pair_right_vectors: Sequence[UnitVector3D], assumptions: SceneAssumptions) -> SceneAxisBasisRecord`
- `build_monitor_plane(center: RobustPointEstimate, direction: RobustDirectionEstimate, axes: SceneAxisBasisRecord, assumptions: SceneAssumptions) -> SceneMonitorPlaneRecord`
- `camera_point_to_scene(point: Vector3D, center: Vector3D, axes: SceneAxisBasisRecord) -> Vector3D`
- `intersect_ray_with_monitor(ray: SceneUniGazeRayRecord, monitor: SceneMonitorPlaneRecord, assumptions: SceneAssumptions) -> SceneMonitorHitRecord`

Tasks 2 through 4 define the required behavior for each signature.

`src/chess_gaze/scene_artifacts.py` and `src/chess_gaze/scene_viewer.py` must expose these dataclasses:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.scene_records import SceneFrameRecord, ViewerSceneData


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


@dataclass(frozen=True)
class ViewerBuildResult:
    index_path: Path
    scene_data_path: Path
    vendor_dir: Path
```

They must expose these public signatures:

- `build_scene_artifacts(run_layout: RunLayout) -> SceneArtifactResult`
- `load_scene_frames(path: Path) -> list[SceneFrameRecord]`
- `build_scene_viewer(run_layout: RunLayout, scene_result: SceneArtifactResult) -> ViewerBuildResult`
- `copy_viewer_assets(viewer_dir: Path) -> None`
- `write_viewer_scene_data(viewer_dir: Path, data: ViewerSceneData) -> Path`

Tasks 5 through 8 define the required behavior for those signatures.

---

## Task 1: Scene Constants And Strict Schemas

**Files:**
- Create: `src/chess_gaze/scene_calibration.py`
- Create: `src/chess_gaze/scene_records.py`
- Create: `tests/chess_gaze/test_scene_calibration.py`
- Create: `tests/chess_gaze/test_scene_records.py`

**Interfaces:**
- `SceneAssumptionRecord`
- `SceneAssumptions`
- `default_scene_assumptions() -> SceneAssumptions`
- `CoordinateFrame3D`
- `SceneInvalidReason`
- `Vector3D`
- `UnitVector3D`
- `SceneCameraModel`
- `SceneEyeRecord`
- `SceneEyeMidpointRecord`
- `SceneHeadRecord`
- `SceneUniGazeRayRecord`
- `SceneMonitorHitRecord`
- `SceneMonitorPlaneRecord`
- `SceneAxisBasisRecord`
- `SceneFrameRecord`
- `SceneManifest`
- `SceneSummary`
- `ViewerSceneData`

- [ ] **Step 1: Write failing scene constant tests**

Create `tests/chess_gaze/test_scene_calibration.py` with tests that assert exact default values:

- `DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M == 0.063`
- `DEFAULT_MONITOR_DISTANCE_FROM_EYES_M == 0.700`
- `DEFAULT_MONITOR_WIDTH_M == 0.600`
- `DEFAULT_MONITOR_HEIGHT_M == 0.340`
- `DEFAULT_EXTENDED_PLANE_SCALE == 3.0`
- `DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M == 0.090`
- `DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M == 0.120`
- `DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M == 0.100`
- `DEFAULT_EYE_SPHERE_RADIUS_M == 0.012`
- `DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M == (0.0, 0.035, 0.020)`
- `RAY_PLANE_PARALLEL_EPSILON == 1e-6`
- `DEFAULT_SCENE_CENTER_CAMERA_M == (0.0, 0.0, 0.650)`
- `SCENE_CENTER_MIN_AXIS_TOLERANCE_M == 0.015`
- `MIN_SCENE_CENTER_INLIER_FRAMES == 5`
- `MIN_MAIN_DIRECTION_INLIER_FRAMES == 5`
- `DIRECTION_INLIER_ANGLE_RADIANS == 0.35`

Also assert that `default_scene_assumptions()` is strict and frozen, and that every persisted assumption has `name`, `value`, `unit`, `source`, and `uncertainty`.

- [ ] **Step 2: Write failing strict schema tests**

Create `tests/chess_gaze/test_scene_records.py` with tests for:

- `Vector3D` rejects NaN, Infinity, and unknown fields.
- `UnitVector3D` rejects norms outside `[0.999, 1.001]`.
- unknown string values for `CoordinateFrame3D` and `SceneInvalidReason` are rejected, not silently accepted.
- valid `SceneEyeRecord` requires both `image_px` and `camera_point_m`.
- invalid `SceneEyeRecord` requires `reason_invalid`.
- valid `SceneEyeMidpointRecord` requires both valid eyes and `camera_point_m`.
- valid `SceneUniGazeRayRecord` requires a valid origin, unit direction, and source `appearance_gaze`.
- valid `SceneMonitorHitRecord` requires a valid ray, finite point, finite UV, finite denominator, finite signed distance, and `t >= 0`.
- invalid nested records retain explicit non-null scene invalid reasons.
- `SceneFrameRecord` serializes with `schema_version == "gaze-scene-frame-v1"`.
- `SceneManifest` serializes with `schema_version == "gaze-scene-manifest-v1"`.
- `SceneSummary` serializes with `schema_version == "gaze-scene-summary-v1"`.
- `ViewerSceneData` serializes with `schema_version == "gaze-scene-viewer-data-v1"` and includes all frames plus valid hit identities.

- [ ] **Step 3: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py -q
```

Expected before implementation: import or assertion failures proving the tests exercise missing scene modules.

- [ ] **Step 4: Implement constants and schemas**

Implement `scene_calibration.py` and `scene_records.py`.

Implementation requirements:

- Use `StrictSchemaModel` from `src/chess_gaze/geometry.py`.
- Add explicit finite validation for vector/list fields because `StrictSchemaModel` only guards direct float fields.
- Use `StrEnum` for scene enum values.
- Include scene invalid reasons at minimum:
  - `LEFT_EYE_INVALID`
  - `RIGHT_EYE_INVALID`
  - `EYE_MIDPOINT_INVALID`
  - `UNIGAZE_INVALID`
  - `RAY_PARALLEL_TO_MONITOR`
  - `RAY_COPLANAR_WITH_MONITOR`
  - `RAY_INTERSECTION_NON_FINITE`
  - `RAY_INTERSECTION_BEHIND_ORIGIN`
  - `SCENE_CENTER_INSUFFICIENT_INLIERS`
  - `MAIN_DIRECTION_INSUFFICIENT_INLIERS`
  - `SCENE_AXIS_DEGENERATE`
  - `MONITOR_PLANE_DEGENERATE`
  - `NON_FINITE_INPUT`
- Persist original frame-level invalid reasons as diagnostic strings instead of mixing `ErrorCode` into the scene invalid enum.
- Keep `FrameRecord` unchanged in this task.

- [ ] **Step 5: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py -q
```

Expected after implementation: all tests in both files pass.

---

## Task 2: Camera Model And Eye Back-Projection

**Files:**
- Modify: `src/chess_gaze/scene_geometry.py`
- Modify: `tests/chess_gaze/test_scene_geometry.py`

**Interfaces:**
- `estimated_camera_model(frame_width: int, frame_height: int) -> SceneCameraModel`
- `back_project_eye_points(frame_record: FrameRecord, camera: SceneCameraModel, assumptions: SceneAssumptions) -> SceneEyePairProjection`
- `SceneEyePairProjection`

- [ ] **Step 1: Write failing camera and eye projection tests**

Create or extend `tests/chess_gaze/test_scene_geometry.py` with tests for:

- landscape intrinsics: `estimated_camera_model(1920, 1080)` gives `fx=1920`, `fy=1920`, `cx=960`, `cy=540`.
- portrait intrinsics: `estimated_camera_model(1080, 1920)` gives `fx=1920`, `fy=1920`, `cx=540`, `cy=960`.
- normal back-projection: with frame size 1920x1080 and pupil centers `(900, 540)` and `(1020, 540)`, depth is `0.063 * 1920 / 120 == 1.008`, eyes are `x=-0.0315` and `x=0.0315`, midpoint is `[0.0, 0.0, 1.008]`.
- zero, negative, or non-finite pupil distance returns invalid eyes or midpoint with scene invalid reasons.
- one missing eye returns invalid midpoint with `EYE_MIDPOINT_INVALID`.
- image coordinates outside the frame are persisted as diagnostics, not clamped into the image.

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Expected before implementation: failures for missing geometry functions.

- [ ] **Step 3: Implement camera and back-projection math**

Implement `scene_geometry.py` camera and eye projection behavior.

Implementation requirements:

- Use OpenCV-style pseudo-metric camera frame: `+X` image-right, `+Y` image-down, `+Z` camera-forward.
- Compute `fx = fy = max(frame_width, frame_height)`, `cx = frame_width / 2`, `cy = frame_height / 2`.
- Use `z = DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M * fx / pupil_distance_px`.
- Store source constants and pupil-distance diagnostics in the returned projection.
- Never claim calibrated metric scale. Name units `camera_opencv_pseudo_m`.
- Do not clamp or hide invalid inputs. Return invalid scene records with diagnostics.

- [ ] **Step 4: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Expected after implementation: camera and eye projection tests pass.

---

## Task 3: Robust Scene Center, Main Direction, And Axis Basis

**Files:**
- Modify: `src/chess_gaze/scene_geometry.py`
- Modify: `tests/chess_gaze/test_scene_geometry.py`

**Interfaces:**
- `robust_scene_center(points: Sequence[Vector3D], assumptions: SceneAssumptions) -> RobustPointEstimate`
- `unigaze_ray_from_frame(frame_record: FrameRecord, midpoint: SceneEyeMidpointRecord) -> SceneUniGazeRayRecord`
- `robust_main_direction(rays: Sequence[SceneUniGazeRayRecord], assumptions: SceneAssumptions) -> RobustDirectionEstimate`
- `build_scene_axis_basis(direction: RobustDirectionEstimate, eye_pair_right_vectors: Sequence[UnitVector3D], assumptions: SceneAssumptions) -> SceneAxisBasisRecord`

- [ ] **Step 1: Write failing robust estimator tests**

Extend `tests/chess_gaze/test_scene_geometry.py` with tests for:

- geometric median after median/MAD screening keeps inliers and rejects injected outliers.
- fewer than `MIN_SCENE_CENTER_INLIER_FRAMES` inliers uses `DEFAULT_SCENE_CENTER_CAMERA_M` and records fallback.
- zero MAD still accepts natural small motion using `SCENE_CENTER_MIN_AXIS_TOLERANCE_M`.
- non-finite center candidates are dropped and counted.
- `unigaze_ray_from_frame()` uses `appearance_gaze`, never `recommended_gaze`.
- UniGaze ray conversion preserves `pitch_yaw_to_unit_vector()` X/Y overlay semantics, negates vector Y for OpenCV camera space so positive frame-record pitch maps to camera up, and negates vector Z so the scene ray points from eyes toward the monitor/camera side.
- angular RANSAC selects a dominant direction with outlier rays present.
- angular RANSAC tie-breaks by inlier count, then lower median angular residual, then lower seed frame index.
- fewer than `MIN_MAIN_DIRECTION_INLIER_FRAMES` valid rays falls back to `[0.0, 0.0, -1.0]`.
- opposite-direction rays are outliers, not equivalent inliers.
- scene axes are finite, unit length, mutually orthogonal, and determinant `+1` for columns `[right, up, back]`.
- scene axes are anatomical frontal-webcam axes; do not add right-vs-forward
  or up-vs-normal projection fallback behavior.

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Expected before implementation: robust estimator and axis tests fail.

- [ ] **Step 3: Implement robust estimators**

Implement the algorithms from the approved spec with the axis correction from this plan:

- `robust_scene_center()`:
  - drop non-finite points;
  - compute component medians and MAD values;
  - keep candidates within `max(3.5 * MAD, SCENE_CENTER_MIN_AXIS_TOLERANCE_M)` on each axis;
  - run Weiszfeld geometric median on survivors;
  - persist candidate count, dropped count, inlier count, MAD values, iteration count, convergence tolerance, fallback state, and uncertainty.
- `unigaze_ray_from_frame()`:
  - require valid `appearance_gaze`;
  - convert pitch/yaw with the scene-specific OpenCV boundary: reuse `pitch_yaw_to_unit_vector()` for x/y overlay signs, negate its y component for `camera_opencv_pseudo_m`, and negate its z component for physical eye-to-monitor direction;
  - use valid eye midpoint as origin;
  - mark invalid with `EYE_MIDPOINT_INVALID` or `UNIGAZE_INVALID` when required.
- `robust_main_direction()`:
  - normalize valid finite candidate directions;
  - seed deterministic angular RANSAC with fixed quantiles of frame order plus coordinate-wise median direction;
  - count inliers within `DIRECTION_INLIER_ANGLE_RADIANS`;
  - tie-break by median angular residual and seed frame index;
  - output normalized mean of inliers or explicit fallback.
- `build_scene_axis_basis()`:
  - build a right-handed transform basis with anatomical frontal-webcam columns
    `[scene_right_camera, scene_up_camera, scene_back_camera]`;
  - set `scene_right_camera = [-1.0, 0.0, 0.0]`;
  - set `scene_up_camera = [0.0, -1.0, 0.0]`;
  - set `scene_back_camera = [0.0, 0.0, 1.0]`;
  - set `scene_forward_camera = [0.0, 0.0, -1.0]`;
  - do not rotate scene axes from dominant UniGaze or eye-pair evidence.

- [ ] **Step 4: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Expected after implementation: all robust estimator and axis tests pass.

---

## Task 4: Monitor Plane And Ray-Plane Intersection

**Files:**
- Modify: `src/chess_gaze/scene_geometry.py`
- Modify: `tests/chess_gaze/test_scene_geometry.py`

**Interfaces:**
- `build_monitor_plane(center: RobustPointEstimate, direction: RobustDirectionEstimate, axes: SceneAxisBasisRecord, assumptions: SceneAssumptions) -> SceneMonitorPlaneRecord`
- `intersect_ray_with_monitor(ray: SceneUniGazeRayRecord, monitor: SceneMonitorPlaneRecord, assumptions: SceneAssumptions) -> SceneMonitorHitRecord`
- `camera_point_to_scene(point: Vector3D, center: Vector3D, axes: SceneAxisBasisRecord) -> Vector3D`

- [ ] **Step 1: Write failing monitor and intersection tests**

Extend `tests/chess_gaze/test_scene_geometry.py` with tests for:

- monitor center is `scene_center_camera + dominant_unigaze_direction_camera * DEFAULT_MONITOR_DISTANCE_FROM_EYES_M`.
- monitor normal is anatomical `scene_back_camera`, not the opposite of the
  dominant UniGaze direction.
- monitor basis is finite, unit length, and equal to the scene right/up/back
  axes.
- physical width and height are `0.600` and `0.340`.
- extended plane width and height are physical dimensions multiplied by `3.0`.
- valid ray-plane hit persists denominator, signed distance, `t`, camera point, scene point, monitor `u/v`, and bounds booleans.
- parallel ray with nonzero signed distance is invalid with `RAY_PARALLEL_TO_MONITOR`.
- coplanar ray is invalid with `RAY_COPLANAR_WITH_MONITOR`.
- behind-origin intersection is invalid with `RAY_INTERSECTION_BEHIND_ORIGIN`.
- non-finite `t` is invalid with `RAY_INTERSECTION_NON_FINITE`.
- physical out-of-bounds but extended in-bounds hit remains a valid unclamped point.
- extended out-of-bounds hit remains a valid unclamped point with `within_extended_plane == False`.
- denominator boundary uses `abs(denom) < 1e-6` as invalid and `abs(denom) == 1e-6` as computable.

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Expected before implementation: monitor and ray-plane tests fail.

- [ ] **Step 3: Implement monitor plane and intersections**

Implementation requirements:

- Use `DEFAULT_MONITOR_DISTANCE_FROM_EYES_M` unless a future calibration artifact exists. This feature has no measured monitor distance.
- Persist the assumed physical monitor rectangle and the extended plane rectangle.
- Persist both camera and scene coordinates for valid hits.
- Persist ray-plane denominator, signed distance, and `t` even for invalid cases when finite.
- Never clamp `u/v` or hit points to physical monitor bounds.

- [ ] **Step 4: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_geometry.py -q
```

Expected after implementation: all geometry tests pass.

---

## Task 5: Scene Artifact Writer And Summary

**Files:**
- Create: `src/chess_gaze/scene_artifacts.py`
- Create: `tests/chess_gaze/test_scene_artifacts.py`
- Create: `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`

**Interfaces:**
- `SceneArtifactPaths`
- `SceneArtifactResult`
- `build_scene_artifacts(run_layout: RunLayout) -> SceneArtifactResult`
- `load_scene_frames(path: Path) -> list[SceneFrameRecord]`
- `build_viewer_scene_data(result: SceneArtifactResult) -> ViewerSceneData`

- [ ] **Step 1: Write failing synthetic scene artifact tests**

Create `tests/chess_gaze/test_scene_artifacts.py` with tests that build a minimal run directory containing `run_manifest.json`, `video_manifest.json`, and `records/frames.jsonl` from strict fake records.

Assert:

- `build_scene_artifacts()` writes `scene/scene_manifest.json`.
- `build_scene_artifacts()` writes `scene/scene_summary.json`.
- `build_scene_artifacts()` writes `records/scene_frames.jsonl`.
- scene manifest includes source video path, source sha256, run ID, camera model, assumptions, robust estimator diagnostics, monitor plane, and axis convention metadata.
- scene summary includes decoded frame count, scene frame count, valid eye midpoint count, valid UniGaze ray count, valid monitor hit count, invalid reason counts, and representative frame IDs.
- `records/scene_frames.jsonl` contains exactly one record per source `FrameRecord`.
- scene frame indices are contiguous from zero.
- valid forward intersections produce exactly one monitor hit per valid frame.
- invalid frames retain explicit invalid reasons.
- no hit point is deduplicated when two frames have identical `u/v`.
- `ViewerSceneData` includes all frames, all valid hit identities, monitor plane, axis basis, run ID, source video stem, and summary counts.

- [ ] **Step 2: Write failing `nakamura_1.mp4` contract test**

Create `tests/chess_gaze/test_scene_artifacts_real_video_contract.py` with a model-free deterministic test that:

- requires `artifacts/input/nakamura_1.mp4`;
- runs `analyze_video()` with fake observers that emit deterministic valid eye centers and valid `appearance_gaze` for every decoded frame;
- writes to a temporary output root;
- asserts decoded frame count is `1973`;
- asserts scene frame record count is `1973`;
- asserts viewer data preserves one valid hit per frame when fake gaze is valid for every frame;
- asserts `scene_summary.json` count validation passes.

Do not use default MediaPipe or UniGaze models in this model-free test.

- [ ] **Step 3: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected before implementation: scene artifact tests fail because writer integration is missing.

- [ ] **Step 4: Implement scene artifact writer**

Implement `scene_artifacts.py`.

Implementation requirements:

- Read only run manifests and `records/frames.jsonl`.
- Validate input JSON through existing `RunManifest`, `VideoManifest`, and `FrameRecord` models.
- Estimate camera model once from `video_manifest`.
- First pass: build per-frame eye projections and UniGaze rays, collecting center and direction candidates.
- Compute robust scene center, main direction, axes, and monitor plane.
- Second pass: compute per-frame scene coordinates and monitor hits using the final scene model.
- Write JSON atomically where current project helpers support it.
- Write JSONL with one strict `SceneFrameRecord.model_dump_json()` per line.
- Write scene manifest and summary with strict models.
- Return all paths and counts in `SceneArtifactResult`.

- [ ] **Step 5: Verify synthetic GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts.py -q
```

Expected after implementation: synthetic scene artifact tests pass.

- [ ] **Step 6: Run early real-video checkpoint**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected: test passes, including `1973` decoded frames and `1973` scene frames. If the local video is missing, record the exact missing path and stop this task.

---

## Task 6: Run Layout, Pipeline, And QA Summary Integration

**Files:**
- Modify: `src/chess_gaze/artifact_runs.py`
- Modify: `src/chess_gaze/pipeline.py`
- Modify: `src/chess_gaze/qa_summary.py`
- Modify: `tests/chess_gaze/test_artifact_runs.py`
- Modify: `tests/chess_gaze/test_frame_observation.py`
- Modify: `tests/chess_gaze/test_pipeline_contract.py`
- Modify: `tests/chess_gaze/test_qa_summary.py`

**Interfaces:**
- `RunLayout.scene_dir: Path`
- `RunLayout.viewer_dir: Path`
- `AnalyzeResult.scene_manifest_path: Path`
- `AnalyzeResult.scene_summary_path: Path`
- `AnalyzeResult.scene_frames_jsonl_path: Path`
- `AnalyzeResult.viewer_index_path: Path`
- `AnalyzeResult.viewer_scene_data_path: Path`
- `AnalyzeResult.valid_scene_frame_count: int`
- `AnalyzeResult.valid_monitor_hit_count: int`

- [ ] **Step 1: Write failing run layout tests**

Extend `tests/chess_gaze/test_artifact_runs.py` to assert `create_run_layout()` creates:

- `scene/`
- `viewer/`
- existing raw, processed, crop, and records directories unchanged.

Update direct `RunLayout` constructor fixtures in `tests/chess_gaze/test_frame_observation.py` and `tests/chess_gaze/test_qa_summary.py`.

- [ ] **Step 2: Write failing pipeline and QA tests**

Extend tests with:

- `tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_writes_scene_artifacts_and_viewer_files`
- `tests/chess_gaze/test_pipeline_contract.py::test_analyze_video_fails_when_scene_artifact_validation_fails`
- `tests/chess_gaze/test_qa_summary.py::test_qa_summary_validates_scene_artifacts_and_counts_scene_bytes`
- `tests/chess_gaze/test_qa_summary.py::test_qa_summary_reports_missing_or_malformed_scene_artifacts`

Assert:

- pipeline order writes scene and viewer files before QA summary;
- `AnalyzeResult` includes all scene and viewer paths;
- `validated_record_count` remains the count of original frame records;
- scene-specific counts are separate;
- QA `SOURCE_ARTIFACTS` includes `scene_manifest`, `scene_summary`, `scene_frames_jsonl`, `viewer_index`, and `viewer_scene_data`;
- `ArtifactCounts.scene_frame_records == decoded_frames`;
- scene frame indices are contiguous from zero;
- `ByteCounts` includes scene JSONL bytes, scene directory bytes, viewer bytes, and total run bytes after viewer generation;
- `QASummary.source_artifacts` and `ArtifactValidationResult.source_artifacts` are identical.

- [ ] **Step 3: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py -q
```

Expected before implementation: failures for missing layout fields, missing scene artifacts, and QA source artifact mismatches.

- [ ] **Step 4: Integrate run layout, pipeline, and QA**

Implementation requirements:

- Add `scene_dir` and `viewer_dir` to `RunLayout`.
- Create both directories in `create_run_layout()`.
- In `pipeline.analyze_video()`:
  - write frame artifacts as today;
  - close observers;
  - call `build_scene_artifacts(layout)`;
  - call viewer generation in Task 8 once available, or write a strict minimal `viewer/scene-data.json` in this task if Task 8 has not been implemented by the same worker;
  - build QA summary after scene and viewer files exist;
  - return scene and viewer paths in `AnalyzeResult`.
- In `qa_summary.py`:
  - load and validate scene manifest, scene summary, and scene frames JSONL through strict scene models;
  - validate `scene_frame_records == decoded_frames`;
  - validate scene frame indices are contiguous;
  - count scene/viewer bytes without double-counting raw/processed/crop bytes;
  - include malformed scene artifacts in schema validation failures.

- [ ] **Step 5: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py -q
```

Expected after implementation: focused layout, pipeline, and QA tests pass.

- [ ] **Step 6: Repeat real-video model-free checkpoint**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected: the model-free `nakamura_1.mp4` contract still passes through integrated pipeline and QA.

---

## Task 7: Vendored Viewer Assets And Package Resources

> Superseded by ADR-0003 and
> `docs/superpowers/plans/2026-06-26-remote-three-viewer-assets.md`.
> Do not create or restore `src/chess_gaze/viewer_assets/vendor/` or
> `vendor_manifest.json`; current viewer package resources use
> `viewer_dependency_manifest.json` and pinned remote Three.js module URLs.

**Files:**
- Create: `src/chess_gaze/viewer_assets/index.html`
- Create: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Create: `src/chess_gaze/viewer_assets/styles.css`
- Create: `src/chess_gaze/viewer_assets/viewer_dependency_manifest.json`
- Modify: `pyproject.toml` if resource packaging test requires explicit force include.
- Create: `tests/test_package_metadata.py`

**Interfaces:**
- Packaged resources are loaded with `importlib.resources.files("chess_gaze").joinpath("viewer_assets")`.
- `viewer_dependency_manifest.json` records Three.js version, license,
  repository, tarball URL, npm integrity, CDN provider, and pinned module URLs.

- [ ] **Step 1: Write failing package resource test**

Create `tests/test_package_metadata.py::test_viewer_assets_are_packaged`.

Assert package resources expose:

- `viewer_assets/index.html`
- `viewer_assets/scene_viewer.js`
- `viewer_assets/styles.css`
- `viewer_assets/viewer_dependency_manifest.json`

Also assert `viewer_dependency_manifest.json` includes:

- `package_name == "three"`
- `version == "0.185.0"`
- `license == "MIT"`
- `tarball == "https://registry.npmjs.org/three/-/three-0.185.0.tgz"`
- `integrity == "sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEAuj25bNAj7k1QQdf+srZywVK6w=="`

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_package_metadata.py -q
```

Expected before implementation: resource files are missing.

- [ ] **Step 3: Record pinned remote Three.js assets**

Use Three.js `0.185.0`, selected in the approved spec and superseded by
ADR-0003 for runtime loading. Record the pinned jsDelivr module URLs and npm
integrity in `viewer_dependency_manifest.json`. Do not commit `node_modules` or
vendored Three.js source files.

- [ ] **Step 4: Add static asset shells**

Create `index.html`, `scene_viewer.js`, and `styles.css` as packaged templates. The full rendering implementation lands in Task 8, but these files must already:

- use a first-screen app layout, not a landing page;
- include stable selectors:
  - `data-testid="scene-canvas"`
  - `data-testid="frame-slider"`
  - `data-testid="mode-instant"`
  - `data-testid="mode-accumulated"`
  - `data-testid="play-pause"`
  - `data-testid="step-prev"`
  - `data-testid="step-next"`
  - `data-testid="hit-count"`
  - `data-testid="frame-status"`
- import `three` and `three/addons/controls/OrbitControls.js` through the
  generated import map;
- fetch local `./scene-data.json`;
- render visible fallback status text if `scene-data.json` cannot be loaded.

If Hatch does not package non-Python files automatically, add:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/chess_gaze/viewer_assets" = "chess_gaze/viewer_assets"
```

- [ ] **Step 5: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_package_metadata.py -q
```

Expected after implementation: resource and vendor metadata tests pass.

---

## Task 8: Viewer Data Generation And Static Viewer

**Files:**
- Create or modify: `src/chess_gaze/scene_viewer.py`
- Modify: `src/chess_gaze/viewer_assets/index.html`
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Modify: `src/chess_gaze/viewer_assets/styles.css`
- Modify: `src/chess_gaze/pipeline.py`
- Create: `tests/chess_gaze/test_scene_viewer.py`

**Interfaces:**
- `ViewerBuildResult`
- `build_scene_viewer(run_layout: RunLayout, scene_result: SceneArtifactResult) -> ViewerBuildResult`
- `copy_viewer_assets(viewer_dir: Path) -> None`
- `write_viewer_scene_data(viewer_dir: Path, data: ViewerSceneData) -> Path`

- [ ] **Step 1: Write failing viewer generation tests**

Create `tests/chess_gaze/test_scene_viewer.py` with tests for:

- `build_scene_viewer()` writes `viewer/index.html`.
- `build_scene_viewer()` writes `viewer/scene-data.json`.
- `build_scene_viewer()` does not write local vendor assets under
  `viewer/vendor/` and removes stale old vendor directories.
- `scene-data.json` is schema-versioned and strict.
- `scene-data.json` includes all scene frames, not only valid frames.
- `scene-data.json` contains exactly one hit identity per valid monitor hit frame and preserves duplicate `u/v` coordinates when different frames hit the same point.
- generated HTML includes required selectors.
- generated HTML references only approved pinned remote Three.js module URLs and
  local app assets.
- generated CSS uses a light theme and the required semantic color roles.
- generated JavaScript contains mode names `Instant` and `Accumulated`.

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py -q
```

Expected before implementation: viewer tests fail because generator is missing.

- [ ] **Step 3: Implement viewer generator**

Implement `scene_viewer.py`.

Implementation requirements:

- Copy packaged `viewer_assets` into each run's `viewer/` directory.
- Write `viewer/scene-data.json` using `ViewerSceneData.model_dump(mode="json")`.
- Keep `viewer/scene-data.json` finite-only and strict.
- Include all frame records in slider order.
- Include a separate valid-hit identity list with `frame_id`, `frame_index`, `point_scene_m`, `point_monitor_uv_m`, and bounds booleans.
- Include run ID, source video stem, monitor plane, axis basis, assumptions summary, viewer dependency metadata, and summary counts.
- Call `build_scene_viewer()` from `pipeline.analyze_video()` after scene artifacts and before QA summary.

- [ ] **Step 4: Implement viewer UI behavior**

Implement `scene_viewer.js` and `styles.css`.

Required 3D behavior:

- Create a Three.js scene with local OrbitControls.
- Render a full usable 3D viewport on first screen.
- Draw transparent head ellipsoid for current frame when valid.
- Draw left and right eye spheres for current frame when valid.
- Draw UniGaze line from eye midpoint to monitor hit when valid.
- If ray is valid but hit is invalid, draw a warning ray segment and show reason.
- Draw current monitor hit point when valid.
- Draw all accumulated hit points with `frame_index <= slider` only in `Accumulated` mode.
- Draw physical monitor rectangle and extended monitor plane.
- Draw axes with labels or an accessible legend.
- Support orbit, pan, and zoom.

Required controls:

- frame slider;
- exact frame label and numeric input;
- `Instant` and `Accumulated` mode switch;
- play/pause;
- step previous and next;
- toggles for head, eyes, UniGaze ray, monitor plane, physical monitor rectangle, extended plane, axes, and hit points;
- status panel with valid/invalid reason, hit counts, and frame identity.

Visual requirements:

- light background;
- head translucent slate;
- left eye calm blue;
- right eye warm coral;
- UniGaze ray deep teal;
- current hit dark violet;
- accumulated hits muted amber;
- monitor plane soft neutral gray;
- warning muted red/orange;
- no neon palette, dark-only palette, gradient-orb decoration, or overlapping UI text.

- [ ] **Step 5: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py -q
```

Expected after implementation: viewer, pipeline, and QA focused tests pass with generated viewer files included before QA.

- [ ] **Step 6: Run real-video model-free checkpoint after viewer**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected: generated viewer exists for the `nakamura_1.mp4` model-free run and `scene-data.json` contains 1973 frames.

---

## Task 9: Localhost Viewer Command And Browser Smoke

**Files:**
- Modify: `src/chess_gaze/cli.py`
- Modify: `src/chess_gaze/scene_viewer.py`
- Modify: `tests/chess_gaze/test_cli.py`
- Modify: `tests/chess_gaze/test_scene_viewer.py`

**Interfaces:**
- `serve_viewer(run_dir: Path, host: str = "127.0.0.1", port: int = 0) -> ViewerServer`
- `ViewerServer.url: str`
- CLI command: `uv run chess-gaze view <run-dir> --host 127.0.0.1 --port 0`

- [ ] **Step 1: Write failing CLI and static-server tests**

Extend tests with:

- `tests/chess_gaze/test_cli.py::test_analyze_prints_run_dir_and_viewer_path`
- `tests/chess_gaze/test_cli.py::test_view_prints_localhost_url_for_run_viewer`
- `tests/chess_gaze/test_scene_viewer.py::test_static_server_serves_viewer_files`
- `tests/chess_gaze/test_scene_viewer.py::test_static_server_does_not_escape_viewer_root`

Assert:

- `analyze` still prints the run directory.
- `analyze` also prints `viewer: <run-dir>/viewer/index.html`.
- `view` rejects missing run directory, missing `viewer/index.html`, and missing `viewer/scene-data.json`.
- `view` serves only from `<run-dir>/viewer`.
- path traversal attempts return 404 or 403.
- host defaults to `127.0.0.1`.

- [ ] **Step 2: Verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py tests/chess_gaze/test_scene_viewer.py -q
```

Expected before implementation: CLI view tests fail.

- [ ] **Step 3: Implement CLI view command**

Implementation requirements:

- Add `view` subcommand to `build_parser()`.
- Use Python standard library HTTP serving; do not add a web framework.
- Bind to localhost by default.
- Print a single local URL that can be opened in Chrome.
- Keep server root locked to `viewer/`.
- Do not upload, transmit, or log frame data outside localhost.
- Ensure command can be interrupted cleanly.

- [ ] **Step 4: Verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_cli.py tests/chess_gaze/test_scene_viewer.py -q
```

Expected after implementation: CLI and static-server tests pass.

- [ ] **Step 5: Browser smoke with generated viewer**

Use the latest `nakamura_1.mp4` model-free or model-backed run. Start the viewer:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view <run-dir> --host 127.0.0.1 --port 0
```

Using available browser automation for this coding environment, open the printed URL and verify:

- canvas is nonblank by reading screenshot or canvas pixels;
- slider changes rendered head/eyes/ray state;
- `Instant` mode shows at most one current hit;
- `Accumulated` mode's displayed hit count equals the number of valid hits with `frame_index <= slider`;
- orbit/pan/zoom controls change the camera or rendered image;
- no visible text overlaps at desktop and mobile-width viewports.

Do not add Playwright for this task unless a separate dependency decision is written first.

---

## Task 10: Default Model Real-Video Verification, Docs, And Closeout

**Files:**
- Modify: `README.md`
- Modify: `docs/development/architecture/source-layout.md`
- Create: `docs/superpowers/closeouts/2026-06-26-3d-scene-artifact-viewer.md`
- Modify tests only if validation reveals a root-cause defect.

- [ ] **Step 1: Run focused test suite**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py tests/chess_gaze/test_artifact_runs.py tests/chess_gaze/test_pipeline_contract.py tests/chess_gaze/test_qa_summary.py tests/chess_gaze/test_scene_viewer.py tests/chess_gaze/test_cli.py tests/test_package_metadata.py -q
```

Expected: all focused tests pass. If any fail, use `superpowers:systematic-debugging` before changing implementation.

- [ ] **Step 2: Run default model analysis on `nakamura_1.mp4`**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_1.mp4 --output-root artifacts/output --models-root models
```

Expected:

- command exits `0`, or reports an exact environment/model blocker;
- run directory is printed;
- viewer path is printed;
- `video_manifest.json` reports `frame_count_decoded == 1973`;
- `records/scene_frames.jsonl` has 1973 lines;
- `scene/scene_summary.json` count validation passes;
- `viewer/index.html` and `viewer/scene-data.json` exist;
- valid/invalid scene reason counts are inspectable.

If the command fails from MediaPipe sandboxed native initialization, rerun unsandboxed according to repository permissions and record both attempts.

- [ ] **Step 3: Inspect produced scene artifacts**

For the run from Step 2, inspect:

- `scene/scene_manifest.json`;
- `scene/scene_summary.json`;
- `records/scene_frames.jsonl` line count;
- `viewer/scene-data.json` frame and hit counts;
- `qa_summary.json` scene validation fields.

Verify:

- assumptions are persisted with values, units, sources, and uncertainty;
- robust scene center diagnostics include candidate count, inlier count, MAD thresholds, fallback state, iterations, and convergence tolerance;
- main direction diagnostics include candidate count, inlier count, angular threshold, residual percentiles, and fallback state;
- axis basis determinant is near `+1`;
- ray-plane diagnostics include denominator, signed distance, `t`, bounds booleans, and invalid reason;
- no valid hit point is clamped to the physical monitor rectangle.

- [ ] **Step 4: Run browser smoke on the default-model run**

Start:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view <run-dir> --host 127.0.0.1 --port 0
```

Open the URL with browser automation and record:

- desktop viewport screenshot or pixel evidence;
- mobile-width viewport screenshot or pixel evidence;
- nonblank canvas assertion;
- slider interaction assertion;
- `Instant` hit-count assertion;
- `Accumulated` hit-count assertion;
- orbit/pan/zoom assertion.

- [ ] **Step 5: Update docs**

Update `README.md`:

- artifact layout now includes `scene/`, `records/scene_frames.jsonl`, and `viewer/`;
- `analyze` prints viewer path;
- `chess-gaze view <run-dir>` serves the viewer locally;
- explain that scene units are pseudo-metric unless future calibration supplies measured scale;
- list the persisted adult-male and monitor assumptions at a concise level;
- state that every valid gaze hit point is preserved without merging.

Update `docs/development/architecture/source-layout.md`:

- current package map includes existing runtime modules and new scene modules;
- viewer assets are package resources under `src/chess_gaze/viewer_assets/`;
- scene modules are named by owned concepts and are not pass-through layers.

- [ ] **Step 6: Run full local gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
```

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check .
```

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
```

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run mypy
```

Expected: all gates pass. If full `pytest` fails only because local media files other than `nakamura_1.mp4` are absent, record exact missing-file failures and rerun the broadest meaningful subset that excludes those absent-media tests.

- [ ] **Step 7: Write implementation closeout**

Create `docs/superpowers/closeouts/2026-06-26-3d-scene-artifact-viewer.md`.

Include:

- implemented files and behavior summary;
- `nakamura_1.mp4` run directory;
- decoded frame count, scene frame count, valid midpoint count, valid ray count, valid monitor hit count;
- axis convention correction and determinant evidence;
- browser smoke evidence;
- focused tests and full gate results with command output summaries;
- exact blockers if any model or environment verification could not run;
- residual uncertainty in pseudo-metric scale and monitor distance.

---

## Implementation Notes For Subagent Splitting

- Worker A can own Tasks 1 through 4: scene constants, schemas, and geometry tests. Write scope: `scene_calibration.py`, `scene_records.py`, `scene_geometry.py`, and the scene schema/geometry tests.
- Worker B can own Tasks 5 and 6 after Worker A's schemas are available. Write scope: `scene_artifacts.py`, `artifact_runs.py`, `pipeline.py`, `qa_summary.py`, and artifact/pipeline/QA tests.
- Worker C can own Tasks 7 through 9 after Worker A's viewer data schema exists. Write scope: `scene_viewer.py`, `viewer_assets/`, CLI view behavior, package metadata test, and viewer tests.
- The main agent must integrate branches, resolve schema names once, run `nakamura_1.mp4` checkpoints after Tasks 5, 6, 8, and 10, and perform final docs and closeout.
- Workers are not alone in the codebase. They must not revert edits outside their ownership scope and must adjust to already-integrated schema names rather than creating duplicate model types.
