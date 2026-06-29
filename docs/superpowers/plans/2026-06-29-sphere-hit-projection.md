# Sphere Hit Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace canonical monitor-plane gaze hits with head-centered gaze sphere hits and render hit points and hit areas on that sphere.

**Architecture:** Move ray-surface projection into a new focused `sphere_projection.py` domain module and remove monitor-plane projection from active scene artifacts. Strict scene records become sphere-first: per-frame artifacts persist `sphere_hit`, manifests persist `gaze_sphere`, summaries persist sphere counts and angular bounds, and the generated Three.js viewer draws the sphere plus sphere-projected hit points and angular hit-area patches. The sphere center is the existing robust scene center, which is the scene-coordinate origin after camera-to-scene normalization.

**Tech Stack:** Python 3.12, Pydantic v2 strict schemas, pytest, generated static HTML/CSS/JS, Three.js `0.185.0` through the existing ADR-0003 import map.

## Global Constraints

- Work in the current branch.
- Use installed Superpowers skills for implementation discipline.
- Use subagents for implementation tasks unless the user explicitly chooses inline execution after this plan.
- Old run compatibility is not required; old ignored runs may be wiped.
- New completed runs project gaze hits onto a sphere, not onto a screen or plane.
- The sphere center is the robust scene center in scene coordinates.
- The default sphere radius is `0.700m`.
- Persist the radius as a gaze-sphere radius, not as monitor width, monitor height, physical screen bounds, extended-plane bounds, or plane UV.
- The viewer must expose a live `Sphere Radius` slider with min `0.35m`, max `1.20m`, and step `0.01m`.
- `records/scene_frames.jsonl` must contain `sphere_hit` and must not contain `main_monitor_hit`, `monitor_hit`, or `plane_uv_m`.
- Scene summaries, QA summaries, run equivalence, README, viewer controls, and viewer status labels must use sphere terminology.
- Hit areas must be sampled angular cones intersected with the sphere.
- Valid rays may produce rear-hemisphere sphere hits; this is intentional direction-surface policy.
- `artifacts/input/nakamura_short.mp4` must be used for real verification.
- Preserve strict artifact validation: no NaN, no Infinity, and no unknown fields.
- Do not add a new runtime dependency.
- Keep module boundaries meaningful; avoid pass-through modules.
- Every implementation task ends with focused tests and a meaningful commit.

---

## File Structure

- Create `src/chess_gaze/sphere_projection.py`
  - Owns gaze-sphere surface construction, ray-sphere intersections, spherical angles, and hemisphere classification.
  - Depends on `scene_calibration.py` and `scene_records.py`.
  - Does not read or write artifacts.
- Create `tests/chess_gaze/test_sphere_projection.py`
  - Covers origin-inside, rear-hemisphere, outside-nearest-root, tangent, miss, behind-origin, invalid-radius, and missing-ray cases.
- Modify `src/chess_gaze/scene_calibration.py`
  - Replace active monitor-plane projection assumptions with `DEFAULT_GAZE_SPHERE_RADIUS_M` and `gaze_sphere_radius_m`.
  - Remove unused monitor width, monitor height, extended plane scale, and ray-plane epsilon from active assumptions.
- Modify `tests/chess_gaze/test_scene_calibration.py`
  - Pin sphere-radius assumptions and strict/frozen behavior.
- Modify `src/chess_gaze/scene_records.py`
  - Replace monitor-plane projection records with sphere projection records.
  - Bump scene manifest, scene frame, summary, and viewer-data schema versions from `v1` to `v2`.
- Modify `tests/chess_gaze/test_scene_records.py`
  - Replace monitor-hit schema tests with sphere-hit schema tests.
- Modify `src/chess_gaze/scene_geometry.py`
  - Remove active monitor-plane projection imports and functions after artifact integration no longer calls them.
- Modify `tests/chess_gaze/test_scene_geometry.py`
  - Remove monitor-plane tests and keep camera projection, robust center, axis basis, and gaze-ray tests.
- Modify `src/chess_gaze/scene_artifacts.py`
  - Build one `GazeSphereSurface` per run.
  - Populate per-frame `sphere_hit`.
  - Build sphere summaries and viewer hit points from sphere hits.
- Modify `tests/chess_gaze/test_scene_artifacts.py`
  - Assert output JSON uses sphere fields and has no monitor-hit fields.
- Modify `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`
  - Keep the mandatory Nakamura short model-free run and assert sphere fields.
- Modify `src/chess_gaze/qa_summary.py`
  - Validate new strict schema versions and sphere summary fields.
- Modify `src/chess_gaze/run_equivalence.py`
  - Compare sphere hit angles and ray-space validity instead of monitor UV.
- Modify run-equivalence tests under `tests/chess_gaze/`
  - Pin angle tolerance and remove monitor-UV expectations.
- Modify `src/chess_gaze/viewer_assets/index.html`
  - Replace monitor-plane controls with a gaze-sphere toggle and radius slider.
- Modify `src/chess_gaze/viewer_assets/scene_viewer.js`
  - Render sphere surface, sphere hits, sphere hit areas, and live radius updates.
- Modify `src/chess_gaze/viewer_assets/styles.css`
  - Replace monitor-plane color variables and control layout with sphere-specific styling.
- Modify `tests/chess_gaze/test_scene_viewer.py`
  - Pin new viewer controls, source math, absence of monitor-plane labels, cached rendering, and summary preservation.
- Modify `README.md`
  - Replace monitor-plane hit description with sphere projection contract.
- Modify `docs/development/architecture/source-layout.md`
  - Add `sphere_projection.py` to the package map and document why it owns projection math.
- Add `docs/superpowers/closeouts/2026-06-29-sphere-hit-projection.md`
  - Record implementation, verification evidence, Nakamura run path, residual risks, and commit list.

## Subagent Boundaries

- Task 1 subagent: calibration and projection math.
- Task 2 subagent: strict schemas and geometry cleanup.
- Task 3 subagent: artifact generation and real-video contract tests.
- Task 4 subagent: QA summary and run equivalence.
- Task 5 subagent: viewer UI, Three.js rendering, and viewer tests.
- Task 6 subagent: documentation, closeout, full verification, and final review.

The main agent reviews every subagent result before committing. If a subagent edits files outside its task, the main agent discards those unrelated edits by patching the intended files, not by resetting the worktree.

---

### Task 1: Calibration And Sphere Projection Math

**Files:**
- Create: `src/chess_gaze/sphere_projection.py`
- Create: `tests/chess_gaze/test_sphere_projection.py`
- Modify: `src/chess_gaze/scene_calibration.py`
- Modify: `tests/chess_gaze/test_scene_calibration.py`
- Modify: `src/chess_gaze/scene_records.py`

**Interfaces:**
- Consumes:
  - `SceneAssumptions.gaze_sphere_radius_m: float`
  - `Vector3D`
  - `UnitVector3D`
  - `CoordinateFrame3D.SCENE_PSEUDO_M`
  - `SceneInvalidReason`
- Produces:
  - `DEFAULT_GAZE_SPHERE_RADIUS_M = 0.700`
  - `GazeSphereSurface(center_scene_m: Vector3D, radius_m: float, radius_source: Literal["DEFAULT_GAZE_SPHERE_RADIUS_M"], center_source: Literal["robust_scene_center"])`
  - `SphereHitResult(valid: bool, point_scene_m: Vector3D | None, ray_t_m: float | None, radius_m: float | None, theta_radians: float | None, phi_radians: float | None, hemisphere: Literal["front", "rear", "equator"] | None, source_reason_invalid: str | None, reason_invalid: SceneInvalidReason | None)`
  - `build_gaze_sphere(assumptions: SceneAssumptions) -> GazeSphereSurface`
  - `intersect_ray_with_sphere(origin_scene_m: Vector3D | None, direction_scene: UnitVector3D | None, sphere: GazeSphereSurface, source_reason_invalid: str | None = None, invalid_reason: SceneInvalidReason = SceneInvalidReason.UNIGAZE_INVALID) -> SphereHitResult`

- [ ] **Step 1: Write failing calibration tests**

Replace monitor-projection assertions in `tests/chess_gaze/test_scene_calibration.py` with sphere-radius assertions:

```python
from chess_gaze.scene_calibration import (
    DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M,
    DEFAULT_EYE_SPHERE_RADIUS_M,
    DEFAULT_GAZE_SPHERE_RADIUS_M,
    DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M,
    DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M,
    DEFAULT_SCENE_CENTER_CAMERA_M,
    DIRECTION_INLIER_ANGLE_RADIANS,
    MIN_MAIN_DIRECTION_INLIER_FRAMES,
    MIN_SCENE_CENTER_INLIER_FRAMES,
    SCENE_CENTER_MIN_AXIS_TOLERANCE_M,
    SceneAssumptions,
    default_scene_assumptions,
)
```

Update the constant and strict/frozen assertions to these exact checks:

```python
def test_scene_constant_values_are_exact() -> None:
    assert DEFAULT_ADULT_MALE_INTERPUPILLARY_DISTANCE_M == 0.063
    assert DEFAULT_GAZE_SPHERE_RADIUS_M == 0.700
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_X_M == 0.090
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_Y_M == 0.120
    assert DEFAULT_HEAD_ELLIPSOID_RADIUS_Z_M == 0.100
    assert DEFAULT_EYE_SPHERE_RADIUS_M == 0.012
    assert DEFAULT_HEAD_CENTER_FROM_EYE_MIDPOINT_M == (0.0, 0.035, 0.020)
    assert DEFAULT_SCENE_CENTER_CAMERA_M == (0.0, 0.0, 0.650)
    assert SCENE_CENTER_MIN_AXIS_TOLERANCE_M == 0.015
    assert MIN_SCENE_CENTER_INLIER_FRAMES == 5
    assert MIN_MAIN_DIRECTION_INLIER_FRAMES == 5
    assert DIRECTION_INLIER_ANGLE_RADIANS == 0.35
```

```python
def test_default_scene_assumptions_is_strict_and_frozen() -> None:
    assumptions = default_scene_assumptions()

    assert SceneAssumptions.model_config["strict"] is True
    assert SceneAssumptions.model_config["frozen"] is True

    with pytest.raises(ValidationError):
        SceneAssumptions.model_validate(
            {
                **assumptions.model_dump(),
                "gaze_sphere_radius_m": "0.7",
            }
        )

    with pytest.raises(ValidationError):
        assumptions.gaze_sphere_radius_m = 0.5
```

Add this assertion to `test_default_scene_assumptions_persists_metadata_for_every_record`:

```python
    names = {record.name for record in assumptions.records}
    assert "DEFAULT_GAZE_SPHERE_RADIUS_M" in names
    assert "DEFAULT_MONITOR_DISTANCE_FROM_EYES_M" not in names
    assert "DEFAULT_MONITOR_WIDTH_M" not in names
    assert "DEFAULT_MONITOR_HEIGHT_M" not in names
    assert "DEFAULT_EXTENDED_PLANE_SCALE" not in names
    assert "RAY_PLANE_PARALLEL_EPSILON" not in names
```

- [ ] **Step 2: Write failing sphere projection tests**

Create `tests/chess_gaze/test_sphere_projection.py`:

```python
from __future__ import annotations

import math

import pytest

from chess_gaze.scene_calibration import DEFAULT_GAZE_SPHERE_RADIUS_M, default_scene_assumptions
from chess_gaze.scene_records import CoordinateFrame3D, SceneInvalidReason, UnitVector3D, Vector3D
from chess_gaze.sphere_projection import build_gaze_sphere, intersect_ray_with_sphere


def _scene_point(x: float, y: float, z: float) -> Vector3D:
    return Vector3D(space=CoordinateFrame3D.SCENE_PSEUDO_M, x=x, y=y, z=z)


def _scene_unit(x: float, y: float, z: float) -> UnitVector3D:
    norm = math.sqrt((x * x) + (y * y) + (z * z))
    return UnitVector3D(space=CoordinateFrame3D.SCENE_PSEUDO_M, x=x / norm, y=y / norm, z=z / norm)


def test_build_gaze_sphere_uses_scene_origin_and_default_radius() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    assert sphere.center_scene_m == _scene_point(0.0, 0.0, 0.0)
    assert sphere.radius_m == pytest.approx(DEFAULT_GAZE_SPHERE_RADIUS_M)
    assert sphere.radius_source == "DEFAULT_GAZE_SPHERE_RADIUS_M"
    assert sphere.center_source == "robust_scene_center"


def test_origin_inside_sphere_hits_front_surface() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, 0.0),
        direction_scene=_scene_unit(0.0, 0.0, -1.0),
        sphere=sphere,
    )

    assert hit.valid is True
    assert hit.reason_invalid is None
    assert hit.point_scene_m == _scene_point(0.0, 0.0, -sphere.radius_m)
    assert hit.ray_t_m == pytest.approx(sphere.radius_m)
    assert hit.radius_m == pytest.approx(sphere.radius_m)
    assert hit.theta_radians == pytest.approx(0.0)
    assert hit.phi_radians == pytest.approx(0.0)
    assert hit.hemisphere == "front"


def test_rear_hemisphere_hit_is_valid_direction_evidence() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, 0.0),
        direction_scene=_scene_unit(0.0, 0.0, 1.0),
        sphere=sphere,
    )

    assert hit.valid is True
    assert hit.point_scene_m == _scene_point(0.0, 0.0, sphere.radius_m)
    assert hit.hemisphere == "rear"
    assert hit.theta_radians == pytest.approx(math.pi)


def test_origin_outside_sphere_selects_nearest_forward_root() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, -1.0),
        direction_scene=_scene_unit(0.0, 0.0, 1.0),
        sphere=sphere,
    )

    assert hit.valid is True
    assert hit.point_scene_m == _scene_point(0.0, 0.0, -sphere.radius_m)
    assert hit.ray_t_m == pytest.approx(1.0 - sphere.radius_m)


def test_tangent_ray_has_one_forward_intersection() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(-1.0, sphere.radius_m, 0.0),
        direction_scene=_scene_unit(1.0, 0.0, 0.0),
        sphere=sphere,
    )

    assert hit.valid is True
    assert hit.point_scene_m == _scene_point(0.0, sphere.radius_m, 0.0)
    assert hit.hemisphere == "equator"
    assert hit.phi_radians == pytest.approx(math.pi / 2.0)


def test_sphere_miss_and_behind_origin_are_invalid_with_reasons() -> None:
    sphere = build_gaze_sphere(default_scene_assumptions())

    miss = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(-1.0, sphere.radius_m + 0.1, 0.0),
        direction_scene=_scene_unit(1.0, 0.0, 0.0),
        sphere=sphere,
    )
    behind = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, -1.0),
        direction_scene=_scene_unit(0.0, 0.0, -1.0),
        sphere=sphere,
    )

    assert miss.valid is False
    assert miss.reason_invalid == SceneInvalidReason.RAY_SPHERE_DISCRIMINANT_NEGATIVE
    assert behind.valid is False
    assert behind.reason_invalid == SceneInvalidReason.RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN


def test_invalid_radius_and_missing_ray_are_invalid() -> None:
    invalid_sphere = build_gaze_sphere(default_scene_assumptions()).model_copy(update={"radius_m": 0.0})

    radius_hit = intersect_ray_with_sphere(
        origin_scene_m=_scene_point(0.0, 0.0, 0.0),
        direction_scene=_scene_unit(0.0, 0.0, -1.0),
        sphere=invalid_sphere,
    )
    missing_ray_hit = intersect_ray_with_sphere(
        origin_scene_m=None,
        direction_scene=None,
        sphere=build_gaze_sphere(default_scene_assumptions()),
        source_reason_invalid="appearance gaze unavailable",
    )

    assert radius_hit.valid is False
    assert radius_hit.reason_invalid == SceneInvalidReason.SPHERE_RADIUS_INVALID
    assert missing_ray_hit.valid is False
    assert missing_ray_hit.reason_invalid == SceneInvalidReason.UNIGAZE_INVALID
    assert missing_ray_hit.source_reason_invalid == "appearance gaze unavailable"
```

- [ ] **Step 3: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py -q
```

Expected before implementation: tests fail because `DEFAULT_GAZE_SPHERE_RADIUS_M`, `gaze_sphere_radius_m`, and `chess_gaze.sphere_projection` are missing.

- [ ] **Step 4: Implement calibration and invalid-reason names**

In `src/chess_gaze/scene_calibration.py`, replace active monitor-projection constants with:

```python
DEFAULT_GAZE_SPHERE_RADIUS_M = 0.700
```

In `SceneAssumptions`, replace the monitor-projection fields with:

```python
    gaze_sphere_radius_m: float
```

In `default_scene_assumptions()`, replace monitor-projection records with:

```python
        SceneAssumptionRecord(
            name="DEFAULT_GAZE_SPHERE_RADIUS_M",
            value=DEFAULT_GAZE_SPHERE_RADIUS_M,
            unit="m",
            source="hypothetical_gaze_sphere_default",
            uncertainty="high",
        ),
```

In the returned `SceneAssumptions`, set:

```python
        gaze_sphere_radius_m=DEFAULT_GAZE_SPHERE_RADIUS_M,
```

Remove these active constants, records, and model fields from `scene_calibration.py`:

```python
DEFAULT_MONITOR_DISTANCE_FROM_EYES_M
DEFAULT_MONITOR_WIDTH_M
DEFAULT_MONITOR_HEIGHT_M
DEFAULT_EXTENDED_PLANE_SCALE
RAY_PLANE_PARALLEL_EPSILON
monitor_distance_from_eyes_m
monitor_width_m
monitor_height_m
extended_plane_scale
ray_plane_parallel_epsilon
```

In `src/chess_gaze/scene_records.py`, add these enum values to `SceneInvalidReason`:

```python
    SPHERE_RADIUS_INVALID = "SPHERE_RADIUS_INVALID"
    RAY_SPHERE_DISCRIMINANT_NEGATIVE = "RAY_SPHERE_DISCRIMINANT_NEGATIVE"
    RAY_SPHERE_INTERSECTION_NON_FINITE = "RAY_SPHERE_INTERSECTION_NON_FINITE"
    RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN = "RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN"
```

- [ ] **Step 5: Implement `sphere_projection.py`**

Create `src/chess_gaze/sphere_projection.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict

from chess_gaze.geometry import StrictSchemaModel
from chess_gaze.scene_calibration import SceneAssumptions
from chess_gaze.scene_records import CoordinateFrame3D, SceneInvalidReason, UnitVector3D, Vector3D

_INTERSECTION_EPSILON = 1e-9
_EQUATOR_EPSILON = 1e-9


class GazeSphereSurface(StrictSchemaModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    center_scene_m: Vector3D
    radius_m: float
    radius_source: Literal["DEFAULT_GAZE_SPHERE_RADIUS_M"]
    center_source: Literal["robust_scene_center"]


class SphereHitResult(StrictSchemaModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    valid: bool
    point_scene_m: Vector3D | None = None
    ray_t_m: float | None = None
    radius_m: float | None = None
    theta_radians: float | None = None
    phi_radians: float | None = None
    hemisphere: Literal["front", "rear", "equator"] | None = None
    source_reason_invalid: str | None = None
    reason_invalid: SceneInvalidReason | None = None


@dataclass(frozen=True)
class _QuadraticRoots:
    near: float
    far: float


def build_gaze_sphere(assumptions: SceneAssumptions) -> GazeSphereSurface:
    return GazeSphereSurface(
        center_scene_m=_scene_vector(0.0, 0.0, 0.0),
        radius_m=assumptions.gaze_sphere_radius_m,
        radius_source="DEFAULT_GAZE_SPHERE_RADIUS_M",
        center_source="robust_scene_center",
    )


def intersect_ray_with_sphere(
    *,
    origin_scene_m: Vector3D | None,
    direction_scene: UnitVector3D | None,
    sphere: GazeSphereSurface,
    source_reason_invalid: str | None = None,
    invalid_reason: SceneInvalidReason = SceneInvalidReason.UNIGAZE_INVALID,
) -> SphereHitResult:
    if not _valid_radius(sphere.radius_m):
        return _invalid_hit(
            SceneInvalidReason.SPHERE_RADIUS_INVALID,
            "sphere radius must be finite and > 0",
        )
    if origin_scene_m is None or direction_scene is None:
        return _invalid_hit(
            invalid_reason,
            source_reason_invalid or "ray origin or direction unavailable",
        )
    if origin_scene_m.space != CoordinateFrame3D.SCENE_PSEUDO_M or direction_scene.space != CoordinateFrame3D.SCENE_PSEUDO_M:
        return _invalid_hit(
            SceneInvalidReason.NON_FINITE_INPUT,
            "ray origin and direction must use scene_pseudo_m",
        )

    origin = _tuple(origin_scene_m)
    direction = _normalize(_tuple(direction_scene))
    center = _tuple(sphere.center_scene_m)
    if direction is None or not _finite_tuple(origin) or not _finite_tuple(center):
        return _invalid_hit(
            SceneInvalidReason.NON_FINITE_INPUT,
            "ray-sphere input contains non-finite values",
        )

    roots = _ray_sphere_roots(origin=origin, direction=direction, center=center, radius=sphere.radius_m)
    if roots is None:
        return _invalid_hit(
            SceneInvalidReason.RAY_SPHERE_DISCRIMINANT_NEGATIVE,
            "ray does not intersect gaze sphere",
        )

    candidates = [root for root in (roots.near, roots.far) if math.isfinite(root) and root >= -_INTERSECTION_EPSILON]
    if not candidates:
        return _invalid_hit(
            SceneInvalidReason.RAY_SPHERE_INTERSECTION_BEHIND_ORIGIN,
            "ray-sphere intersections are behind ray origin",
        )
    ray_t = max(0.0, min(candidates))
    point_tuple = (
        origin[0] + direction[0] * ray_t,
        origin[1] + direction[1] * ray_t,
        origin[2] + direction[2] * ray_t,
    )
    if not _finite_tuple(point_tuple):
        return _invalid_hit(
            SceneInvalidReason.RAY_SPHERE_INTERSECTION_NON_FINITE,
            "ray-sphere intersection point is non-finite",
        )

    point = _scene_vector(point_tuple[0], point_tuple[1], point_tuple[2])
    relative = (
        point_tuple[0] - center[0],
        point_tuple[1] - center[1],
        point_tuple[2] - center[2],
    )
    theta = math.atan2(relative[0], -relative[2])
    phi = math.asin(max(-1.0, min(1.0, relative[1] / sphere.radius_m)))
    return SphereHitResult(
        valid=True,
        point_scene_m=point,
        ray_t_m=ray_t,
        radius_m=sphere.radius_m,
        theta_radians=theta,
        phi_radians=phi,
        hemisphere=_hemisphere(relative[2]),
        source_reason_invalid=None,
        reason_invalid=None,
    )


def _ray_sphere_roots(
    *,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    center: tuple[float, float, float],
    radius: float,
) -> _QuadraticRoots | None:
    oc = (origin[0] - center[0], origin[1] - center[1], origin[2] - center[2])
    a = _dot(direction, direction)
    b = 2.0 * _dot(oc, direction)
    c = _dot(oc, oc) - (radius * radius)
    discriminant = (b * b) - (4.0 * a * c)
    if not math.isfinite(discriminant):
        return None
    if discriminant < -_INTERSECTION_EPSILON:
        return None
    sqrt_discriminant = math.sqrt(max(0.0, discriminant))
    denominator = 2.0 * a
    if abs(denominator) <= _INTERSECTION_EPSILON:
        return None
    near = (-b - sqrt_discriminant) / denominator
    far = (-b + sqrt_discriminant) / denominator
    return _QuadraticRoots(near=min(near, far), far=max(near, far))


def _valid_radius(radius_m: float) -> bool:
    return math.isfinite(radius_m) and radius_m > 0.0


def _invalid_hit(reason: SceneInvalidReason, source_reason: str) -> SphereHitResult:
    return SphereHitResult(
        valid=False,
        source_reason_invalid=source_reason,
        reason_invalid=reason,
    )


def _hemisphere(z: float) -> Literal["front", "rear", "equator"]:
    if z < -_EQUATOR_EPSILON:
        return "front"
    if z > _EQUATOR_EPSILON:
        return "rear"
    return "equator"


def _scene_vector(x: float, y: float, z: float) -> Vector3D:
    return Vector3D(space=CoordinateFrame3D.SCENE_PSEUDO_M, x=x, y=y, z=z)


def _tuple(vector: Vector3D | UnitVector3D) -> tuple[float, float, float]:
    return (vector.x, vector.y, vector.z)


def _finite_tuple(values: tuple[float, float, float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _normalize(values: tuple[float, float, float]) -> tuple[float, float, float] | None:
    if not _finite_tuple(values):
        return None
    norm = math.sqrt(_dot(values, values))
    if norm <= _INTERSECTION_EPSILON:
        return None
    return (values[0] / norm, values[1] / norm, values[2] / norm)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return (left[0] * right[0]) + (left[1] * right[1]) + (left[2] * right[2])
```

- [ ] **Step 6: Run GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py -q
```

Expected after implementation: all selected tests pass.

- [ ] **Step 7: Commit Task 1**

Run:

```sh
git add src/chess_gaze/scene_calibration.py src/chess_gaze/scene_records.py src/chess_gaze/sphere_projection.py tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py
git commit -m "feat: add gaze sphere projection math"
```

---

### Task 2: Strict Sphere Scene Schemas And Geometry Cleanup

**Files:**
- Modify: `src/chess_gaze/scene_records.py`
- Modify: `tests/chess_gaze/test_scene_records.py`
- Modify: `src/chess_gaze/scene_geometry.py`
- Modify: `tests/chess_gaze/test_scene_geometry.py`

**Interfaces:**
- Consumes:
  - `SphereHitResult`
  - `GazeSphereSurface`
  - `SceneInvalidReason` sphere values from Task 1
- Produces:
  - `CoordinateFrame3D.GAZE_SPHERE_PSEUDO_M`
  - `SceneSphereHitRecord`
  - `SceneGazeSphereRecord`
  - `SceneSphereHitAngleBoundsRecord`
  - `SceneFrameRecord.schema_version == "gaze-scene-frame-v2"`
  - `SceneFrameRecord.valid_for_sphere_projection`
  - `SceneFrameRecord.sphere_hit`
  - `SceneCoordinateFramesRecord.projection_frame`
  - `SceneManifest.schema_version == "gaze-scene-manifest-v2"`
  - `SceneManifest.gaze_sphere`
  - `SceneSummary.schema_version == "gaze-scene-summary-v2"`
  - `SceneSummary.valid_sphere_hit_frames`
  - `SceneSummary.invalid_sphere_hit_reasons`
  - `SceneSummary.sphere_hit_angle_bounds`
  - `ViewerHitPoint.theta_radians`
  - `ViewerHitPoint.phi_radians`
  - `ViewerHitPoint.hemisphere`
  - `ViewerSceneData.schema_version == "gaze-scene-viewer-data-v2"`
  - `ViewerSceneData.gaze_sphere`

- [ ] **Step 1: Write failing schema tests**

In `tests/chess_gaze/test_scene_records.py`, replace monitor-hit helper payloads with these helpers:

```python
def _sphere_hit_payload() -> dict[str, Any]:
    return {
        "valid": True,
        "point_scene_m": _scene_vector(0.0, 0.0, -0.7),
        "ray_t_m": 0.7,
        "radius_m": 0.7,
        "theta_radians": 0.0,
        "phi_radians": 0.0,
        "hemisphere": "front",
        "source_reason_invalid": None,
        "reason_invalid": None,
    }


def _gaze_sphere_payload() -> dict[str, Any]:
    return {
        "center_scene_m": _scene_vector(0.0, 0.0, 0.0),
        "radius_m": 0.7,
        "radius_source": "DEFAULT_GAZE_SPHERE_RADIUS_M",
        "center_source": "robust_scene_center",
    }
```

Add these tests:

```python
def test_valid_sphere_hit_requires_point_angles_radius_and_forward_t() -> None:
    hit = SceneSphereHitRecord.model_validate(_sphere_hit_payload())

    assert hit.valid is True
    assert hit.point_scene_m is not None
    assert hit.ray_t_m == 0.7
    assert hit.radius_m == 0.7
    assert hit.theta_radians == 0.0
    assert hit.phi_radians == 0.0
    assert hit.hemisphere == "front"

    with pytest.raises(ValidationError):
        SceneSphereHitRecord.model_validate({**_sphere_hit_payload(), "ray_t_m": -0.001})

    with pytest.raises(ValidationError):
        SceneSphereHitRecord.model_validate({**_sphere_hit_payload(), "radius_m": 0.0})


def test_invalid_sphere_hit_requires_explicit_reason() -> None:
    hit = SceneSphereHitRecord.model_validate(
        {
            "valid": False,
            "point_scene_m": None,
            "ray_t_m": None,
            "radius_m": None,
            "theta_radians": None,
            "phi_radians": None,
            "hemisphere": None,
            "source_reason_invalid": "appearance gaze unavailable",
            "reason_invalid": "UNIGAZE_INVALID",
        }
    )

    assert hit.valid is False
    assert hit.reason_invalid == SceneInvalidReason.UNIGAZE_INVALID

    with pytest.raises(ValidationError):
        SceneSphereHitRecord.model_validate({"valid": False, "reason_invalid": None})


def test_frame_record_uses_sphere_hit_and_rejects_monitor_hit() -> None:
    frame = SceneFrameRecord.model_validate(_scene_frame_payload())

    assert frame.schema_version == "gaze-scene-frame-v2"
    assert frame.sphere_hit.valid is True
    assert frame.valid_for_sphere_projection is True
    assert "main_monitor_hit" not in frame.model_dump()

    with pytest.raises(ValidationError):
        SceneFrameRecord.model_validate(
            {**_scene_frame_payload(), "main_monitor_hit": _sphere_hit_payload()}
        )


def test_manifest_and_viewer_data_use_gaze_sphere() -> None:
    manifest = SceneManifest.model_validate(_manifest_payload())
    viewer_data = ViewerSceneData.model_validate(_viewer_scene_data_payload())

    assert manifest.schema_version == "gaze-scene-manifest-v2"
    assert manifest.coordinate_frames.projection_frame == CoordinateFrame3D.GAZE_SPHERE_PSEUDO_M
    assert manifest.gaze_sphere.radius_m == 0.7
    assert viewer_data.schema_version == "gaze-scene-viewer-data-v2"
    assert viewer_data.gaze_sphere == manifest.gaze_sphere
    assert "monitor_plane" not in viewer_data.model_dump()


def test_summary_reports_sphere_hit_bounds_and_reasons() -> None:
    summary = SceneSummary.model_validate(_summary_payload())
    payload = summary.model_dump()

    assert summary.schema_version == "gaze-scene-summary-v2"
    assert payload["valid_sphere_hit_frames"] == 6
    assert payload["invalid_sphere_hit_reasons"] == {"UNIGAZE_INVALID": 2}
    assert payload["sphere_hit_angle_bounds"]["theta_min_radians"] == -0.42
    assert payload["sphere_hit_angle_bounds"]["theta_max_radians"] == 0.42
    assert payload["sphere_hit_angle_bounds"]["phi_min_radians"] == -0.2
    assert payload["sphere_hit_angle_bounds"]["phi_max_radians"] == 0.2
```

- [ ] **Step 2: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_records.py -q
```

Expected before implementation: tests fail because sphere schema classes and fields are missing.

- [ ] **Step 3: Implement schema records**

In `src/chess_gaze/scene_records.py`:

1. Add `GAZE_SPHERE_PSEUDO_M` and remove active `MONITOR_PLANE_PSEUDO_M` usage:

```python
class CoordinateFrame3D(StrEnum):
    IMAGE_PX = "image_px"
    CAMERA_OPENCV_PSEUDO_M = "camera_opencv_pseudo_m"
    SCENE_PSEUDO_M = "scene_pseudo_m"
    GAZE_SPHERE_PSEUDO_M = "gaze_sphere_pseudo_m"
    THREE_VIEW = "three_view"
```

2. Replace `SceneMonitorHitRecord` with `SceneSphereHitRecord`:

```python
class SceneSphereHitRecord(SceneSchemaModel):
    valid: bool
    point_scene_m: Vector3D | None = None
    ray_t_m: float | None = None
    radius_m: float | None = None
    theta_radians: float | None = None
    phi_radians: float | None = None
    hemisphere: Literal["front", "rear", "equator"] | None = None
    source_reason_invalid: str | None = None
    reason_invalid: SceneInvalidReason | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        _coerce_enum_field(
            coerced,
            field_name="reason_invalid",
            enum_type=SceneInvalidReason,
        )
        return coerced

    @model_validator(mode="after")
    def validate_hit(self) -> SceneSphereHitRecord:
        _require_vector_space(
            self.point_scene_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="point_scene_m",
        )
        if self.valid:
            required_values = (
                self.point_scene_m,
                self.ray_t_m,
                self.radius_m,
                self.theta_radians,
                self.phi_radians,
                self.hemisphere,
            )
            if any(value is None for value in required_values):
                raise ValueError(
                    "valid sphere hit requires point, t, radius, angles, and hemisphere"
                )
            if self.ray_t_m is not None and self.ray_t_m < 0:
                raise ValueError("valid sphere hit requires ray_t_m >= 0")
            if self.radius_m is not None and self.radius_m <= 0:
                raise ValueError("valid sphere hit requires radius_m > 0")
            if self.reason_invalid is not None:
                raise ValueError("valid sphere hit cannot have reason_invalid")
            return self
        if self.reason_invalid is None:
            raise ValueError("invalid sphere hit requires reason_invalid")
        return self
```

3. Add `SceneGazeSphereRecord`:

```python
class SceneGazeSphereRecord(SceneSchemaModel):
    center_scene_m: Vector3D
    radius_m: float
    radius_source: Literal["DEFAULT_GAZE_SPHERE_RADIUS_M"]
    center_source: Literal["robust_scene_center"]

    @model_validator(mode="after")
    def validate_sphere(self) -> SceneGazeSphereRecord:
        _require_vector_space(
            self.center_scene_m,
            expected=CoordinateFrame3D.SCENE_PSEUDO_M,
            field_name="center_scene_m",
        )
        if self.radius_m <= 0:
            raise ValueError("gaze sphere radius_m must be > 0")
        return self
```

4. Replace frame, coordinate-frame, manifest, summary, viewer-hit, and viewer-data fields:

```python
class SceneFrameRecord(SceneSchemaModel):
    schema_version: Literal["gaze-scene-frame-v2"] = "gaze-scene-frame-v2"
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    source_frame_status: FrameStatus
    valid_for_scene_center: bool
    valid_for_sphere_projection: bool
    camera: SceneFrameCameraRecord
    left_eye: SceneEyeRecord
    right_eye: SceneEyeRecord
    eye_midpoint: SceneEyeMidpointRecord
    head: SceneHeadRecord
    unigaze_ray: SceneUniGazeRayRecord
    sphere_hit: SceneSphereHitRecord
    diagnostics: SceneFrameDiagnosticsRecord
```

```python
class SceneCoordinateFramesRecord(SceneSchemaModel):
    math_frame: CoordinateFrame3D
    scene_frame: CoordinateFrame3D
    projection_frame: CoordinateFrame3D
    viewer_frame: CoordinateFrame3D
```

```python
class SceneSphereHitAngleBoundsRecord(SceneSchemaModel):
    theta_min_radians: float
    theta_max_radians: float
    phi_min_radians: float
    phi_max_radians: float
    front_hemisphere_frames: int
    rear_hemisphere_frames: int
    equator_frames: int
```

```python
class SceneSummary(SceneSchemaModel):
    schema_version: Literal["gaze-scene-summary-v2"] = "gaze-scene-summary-v2"
    run_id: str
    decoded_frames: int
    scene_frame_records: int
    valid_eye_midpoint_frames: int
    valid_unigaze_ray_frames: int
    valid_sphere_hit_frames: int
    invalid_sphere_hit_reasons: dict[str, int]
    sphere_hit_angle_bounds: SceneSphereHitAngleBoundsRecord
    representative_scene_warning_frame_ids: list[str]
    artifact_validation: SceneArtifactValidationRecord
```

```python
class ViewerHitPoint(SceneSchemaModel):
    frame_id: str
    frame_index: int
    point_scene_m: Vector3D
    radius_m: float
    theta_radians: float
    phi_radians: float
    hemisphere: Literal["front", "rear", "equator"]
```

```python
class ViewerSceneData(SceneSchemaModel):
    schema_version: Literal["gaze-scene-viewer-data-v2"] = "gaze-scene-viewer-data-v2"
    run_id: str
    source_video_stem: str
    frame_count: int
    frames: list[SceneFrameRecord]
    valid_hit_points: list[ViewerHitPoint]
    gaze_sphere: SceneGazeSphereRecord
    axis_basis: SceneAxisBasisRecord
    assumptions: list[SceneAssumptionRecord]
    summary: SceneSummary
```

5. Update validators so `valid_for_sphere_projection` requires a valid `unigaze_ray`, a valid `sphere_hit` requires a valid `unigaze_ray`, and `projection_frame` must equal `CoordinateFrame3D.GAZE_SPHERE_PSEUDO_M`.

- [ ] **Step 4: Remove monitor-plane active geometry**

In `src/chess_gaze/scene_geometry.py`, remove imports of `SceneMonitorHitRecord` and `SceneMonitorPlaneRecord`, and delete these functions:

```python
build_monitor_plane
intersect_ray_with_monitor
_invalid_monitor_hit
_monitor_scene_center_camera
_camera_point_to_scene_from_monitor
```

In `tests/chess_gaze/test_scene_geometry.py`, delete tests whose names start with:

```python
test_build_monitor_plane_
test_intersect_ray_with_monitor_
```

Keep tests for `estimated_camera_model`, `back_project_eye_points`, `robust_scene_center`, `robust_main_direction`, `build_scene_axis_basis`, `camera_point_to_scene`, and `unigaze_ray_from_frame`.

- [ ] **Step 5: Run GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_geometry.py -q
```

Expected after implementation: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```sh
git add src/chess_gaze/scene_records.py src/chess_gaze/scene_geometry.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_geometry.py
git commit -m "feat: define sphere scene schemas"
```

---

### Task 3: Scene Artifact Sphere Integration

**Files:**
- Modify: `src/chess_gaze/scene_artifacts.py`
- Modify: `tests/chess_gaze/test_scene_artifacts.py`
- Modify: `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`

**Interfaces:**
- Consumes:
  - `build_gaze_sphere(assumptions)`
  - `intersect_ray_with_sphere(origin_scene_m=ray.origin_scene_m, direction_scene=ray.direction_scene, sphere=gaze_sphere, source_reason_invalid=ray.source_reason_invalid, invalid_reason=ray.reason_invalid or SceneInvalidReason.UNIGAZE_INVALID)`
  - `SceneGazeSphereRecord`
  - `SceneSphereHitRecord`
- Produces:
  - `SceneArtifactResult.valid_sphere_hit_count: int`
  - `manifest.gaze_sphere`
  - `frame.sphere_hit`
  - `summary.valid_sphere_hit_frames`
  - `viewer_data.gaze_sphere`
  - `viewer_data.valid_hit_points` derived from `sphere_hit`

- [ ] **Step 1: Write failing artifact tests**

In `tests/chess_gaze/test_scene_artifacts.py`, replace monitor assertions with sphere assertions:

```python
def test_scene_artifacts_manifest_summary_and_frames_use_sphere_projection(tmp_path: Path) -> None:
    result = _build_scene_artifacts_for_test(tmp_path)

    manifest = SceneManifest.model_validate_json(
        result.paths.scene_manifest_path.read_text(encoding="utf-8")
    )
    summary = SceneSummary.model_validate_json(
        result.paths.scene_summary_path.read_text(encoding="utf-8")
    )
    records = load_scene_frames(result.paths.scene_frames_jsonl_path)
    raw_jsonl = result.paths.scene_frames_jsonl_path.read_text(encoding="utf-8")

    assert manifest.gaze_sphere.radius_m == pytest.approx(0.7)
    assert manifest.coordinate_frames.projection_frame == CoordinateFrame3D.GAZE_SPHERE_PSEUDO_M
    assert summary.valid_sphere_hit_frames == result.valid_sphere_hit_count
    assert all(record.sphere_hit.valid for record in records[:-1])
    assert records[-1].sphere_hit.valid is False
    assert "main_monitor_hit" not in raw_jsonl
    assert "monitor_hit" not in raw_jsonl
    assert "plane_uv_m" not in raw_jsonl
```

Add viewer-data assertions:

```python
    viewer_data = result.viewer_data
    assert viewer_data.gaze_sphere == manifest.gaze_sphere
    assert len(viewer_data.valid_hit_points) == result.valid_sphere_hit_count
    assert [point.frame_index for point in viewer_data.valid_hit_points] == [
        frame.frame_index for frame in records if frame.sphere_hit.valid
    ]
    assert all(point.radius_m == pytest.approx(0.7) for point in viewer_data.valid_hit_points)
```

In `tests/chess_gaze/test_scene_artifacts_real_video_contract.py`, replace monitor assertions with:

```python
    assert generated_summary.valid_sphere_hit_frames == expected_frame_count
    assert len(generated_viewer_data.valid_hit_points) == expected_frame_count
    assert generated_viewer_data.gaze_sphere.radius_m == pytest.approx(0.7)
    assert all(frame.sphere_hit.valid for frame in generated_viewer_data.frames)
    assert "main_monitor_hit" not in viewer_scene_data_path.read_text(encoding="utf-8")
    assert "monitor_plane" not in viewer_scene_data_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected before implementation: tests fail because `SceneArtifactResult.valid_sphere_hit_count`, `manifest.gaze_sphere`, and per-frame `sphere_hit` are not populated.

- [ ] **Step 3: Implement artifact integration**

In `src/chess_gaze/scene_artifacts.py`:

1. Replace imports:

```python
from chess_gaze.sphere_projection import GazeSphereSurface, SphereHitResult, build_gaze_sphere, intersect_ray_with_sphere
```

2. Change `SceneArtifactResult`:

```python
@dataclass(frozen=True)
class SceneArtifactResult:
    paths: SceneArtifactPaths
    scene_frame_count: int
    valid_sphere_hit_count: int
    viewer_data: ViewerSceneData
    manifest: SceneManifest
    summary: SceneSummary
    frames: list[SceneFrameRecord]
```

3. In `build_scene_artifacts()`, replace the existing call that assigns `monitor_plane` from `build_monitor_plane` with:

```python
    gaze_sphere = build_gaze_sphere(assumptions)
```

Pass `gaze_sphere=gaze_sphere` into `_build_scene_frame`, `_build_manifest`, and `_viewer_scene_data_from_parts`.

4. Add mapper helpers:

```python
def _sphere_record(surface: GazeSphereSurface) -> SceneGazeSphereRecord:
    return SceneGazeSphereRecord(
        center_scene_m=surface.center_scene_m,
        radius_m=surface.radius_m,
        radius_source=surface.radius_source,
        center_source=surface.center_source,
    )


def _sphere_hit_record(result: SphereHitResult) -> SceneSphereHitRecord:
    return SceneSphereHitRecord(
        valid=result.valid,
        point_scene_m=result.point_scene_m,
        ray_t_m=result.ray_t_m,
        radius_m=result.radius_m,
        theta_radians=result.theta_radians,
        phi_radians=result.phi_radians,
        hemisphere=result.hemisphere,
        source_reason_invalid=result.source_reason_invalid,
        reason_invalid=result.reason_invalid,
    )
```

5. In `_build_scene_frame`, compute:

```python
    sphere_hit_result = intersect_ray_with_sphere(
        origin_scene_m=ray.origin_scene_m,
        direction_scene=ray.direction_scene,
        sphere=gaze_sphere,
        source_reason_invalid=ray.source_reason_invalid,
        invalid_reason=ray.reason_invalid or SceneInvalidReason.UNIGAZE_INVALID,
    )
    sphere_hit = _sphere_hit_record(sphere_hit_result)
```

Set:

```python
        valid_for_sphere_projection=ray.valid,
        sphere_hit=sphere_hit,
```

6. Replace warning creation with:

```python
def _scene_warnings(
    midpoint: SceneEyeMidpointRecord,
    ray: SceneUniGazeRayRecord,
    sphere_hit: SceneSphereHitRecord,
) -> list[str]:
    warnings: list[str] = []
    if not midpoint.valid and midpoint.reason_invalid is not None:
        warnings.append(f"eye_midpoint:{_enum_value(midpoint.reason_invalid)}")
    if not ray.valid and ray.reason_invalid is not None:
        warnings.append(f"unigaze_ray:{_enum_value(ray.reason_invalid)}")
    if not sphere_hit.valid and sphere_hit.reason_invalid is not None:
        warnings.append(f"sphere_hit:{_enum_value(sphere_hit.reason_invalid)}")
    return warnings
```

7. Replace `_monitor_hit_bounds` with:

```python
def _sphere_hit_angle_bounds(frames: list[SceneFrameRecord]) -> SceneSphereHitAngleBoundsRecord:
    valid_hits = [
        frame.sphere_hit
        for frame in frames
        if frame.sphere_hit.valid
        and frame.sphere_hit.theta_radians is not None
        and frame.sphere_hit.phi_radians is not None
        and frame.sphere_hit.hemisphere is not None
    ]
    if not valid_hits:
        return SceneSphereHitAngleBoundsRecord(
            theta_min_radians=0.0,
            theta_max_radians=0.0,
            phi_min_radians=0.0,
            phi_max_radians=0.0,
            front_hemisphere_frames=0,
            rear_hemisphere_frames=0,
            equator_frames=0,
        )
    theta_values = [hit.theta_radians for hit in valid_hits if hit.theta_radians is not None]
    phi_values = [hit.phi_radians for hit in valid_hits if hit.phi_radians is not None]
    hemispheres = [hit.hemisphere for hit in valid_hits]
    return SceneSphereHitAngleBoundsRecord(
        theta_min_radians=min(theta_values),
        theta_max_radians=max(theta_values),
        phi_min_radians=min(phi_values),
        phi_max_radians=max(phi_values),
        front_hemisphere_frames=sum(1 for hemisphere in hemispheres if hemisphere == "front"),
        rear_hemisphere_frames=sum(1 for hemisphere in hemispheres if hemisphere == "rear"),
        equator_frames=sum(1 for hemisphere in hemispheres if hemisphere == "equator"),
    )
```

8. Replace `_valid_hit_points` with:

```python
def _valid_hit_points(frames: list[SceneFrameRecord]) -> list[ViewerHitPoint]:
    hit_points: list[ViewerHitPoint] = []
    for frame in frames:
        hit = frame.sphere_hit
        if not hit.valid:
            continue
        if (
            hit.point_scene_m is None
            or hit.radius_m is None
            or hit.theta_radians is None
            or hit.phi_radians is None
            or hit.hemisphere is None
        ):
            raise ValueError("valid sphere hit is missing persisted viewer fields")
        hit_points.append(
            ViewerHitPoint(
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                point_scene_m=hit.point_scene_m,
                radius_m=hit.radius_m,
                theta_radians=hit.theta_radians,
                phi_radians=hit.phi_radians,
                hemisphere=hit.hemisphere,
            )
        )
    return hit_points
```

- [ ] **Step 4: Run GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py -q
```

Expected after implementation: selected artifact tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```sh
git add src/chess_gaze/scene_artifacts.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py
git commit -m "feat: write sphere scene artifacts"
```

---

### Task 4: QA Summary And Run Equivalence

**Files:**
- Modify: `src/chess_gaze/qa_summary.py`
- Modify: `src/chess_gaze/run_equivalence.py`
- Modify: run-equivalence and QA tests under `tests/chess_gaze/`

**Interfaces:**
- Consumes:
  - strict `SceneManifest`, `SceneSummary`, `SceneFrameRecord`, and `ViewerSceneData` v2 schemas
  - per-frame `sphere_hit.theta_radians` and `sphere_hit.phi_radians`
- Produces:
  - run equivalence tolerance `sphere_hit_angle_radians: float = 1e-6`
  - report field `max_sphere_hit_angle_delta_radians`
  - diff kind `"sphere_hit_angle"`

- [ ] **Step 1: Write failing QA and equivalence tests**

In run-equivalence tests, replace monitor-UV assertions with:

```python
def test_run_equivalence_compares_sphere_hit_angles() -> None:
    baseline = _run_payload_with_sphere_hit(theta=0.1, phi=-0.2)
    candidate = _run_payload_with_sphere_hit(theta=0.1 + 5e-7, phi=-0.2)

    report = compare_run_payloads(baseline, candidate)

    assert report.equivalent is True
    assert report.max_sphere_hit_angle_delta_radians == pytest.approx(5e-7)
    assert report.max_delta_kind == "sphere_hit_angle"
```

Add a failing mismatch case:

```python
def test_run_equivalence_rejects_sphere_hit_angle_delta_above_tolerance() -> None:
    baseline = _run_payload_with_sphere_hit(theta=0.1, phi=-0.2)
    candidate = _run_payload_with_sphere_hit(theta=0.1 + 2e-6, phi=-0.2)

    report = compare_run_payloads(baseline, candidate)

    assert report.equivalent is False
    assert any("sphere_hit.theta_radians" in difference.path for difference in report.differences)
```

In QA summary tests, add assertions that `SceneSummary.model_validate_json(summary_json)` succeeds for v2 and that generated QA text contains `valid_sphere_hit_frames` and does not contain `valid_monitor_hit_frames`.

- [ ] **Step 2: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze -q -k "qa_summary or run_equivalence"
```

Expected before implementation: tests fail because equivalence and QA code still names monitor UV and monitor hit summaries.

- [ ] **Step 3: Implement run equivalence**

In `src/chess_gaze/run_equivalence.py`:

1. Replace tolerance and report fields:

```python
@dataclass(frozen=True)
class EquivalenceTolerances:
    appearance_radians: float = 1e-6
    scene_ray_component: float = 1e-6
    sphere_hit_angle_radians: float = 1e-6
```

```python
    max_sphere_hit_angle_delta_radians: float
```

2. Replace monitor-hit comparison helpers with:

```python
def _compare_sphere_hit(
    comparison: _ComparisonBuilder,
    prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    baseline_hit = _sphere_hit_record(baseline)
    candidate_hit = _sphere_hit_record(candidate)
    hit_path = f"{prefix}.sphere_hit"
    _compare_bool(
        comparison,
        f"{hit_path}.valid",
        baseline_hit.get("valid"),
        candidate_hit.get("valid"),
    )
    if not baseline_hit.get("valid") or not candidate_hit.get("valid"):
        _compare_exact(
            comparison,
            f"{hit_path}.reason_invalid",
            baseline_hit.get("reason_invalid"),
            candidate_hit.get("reason_invalid"),
        )
        return
    for component in ("theta_radians", "phi_radians"):
        _compare_float(
            comparison,
            f"{hit_path}.{component}",
            baseline_hit.get(component),
            candidate_hit.get(component),
            tolerance=comparison.tolerances.sphere_hit_angle_radians,
            max_delta_kind="sphere_hit_angle",
        )
    _compare_exact(
        comparison,
        f"{hit_path}.hemisphere",
        baseline_hit.get("hemisphere"),
        candidate_hit.get("hemisphere"),
    )


def _sphere_hit_record(frame: dict[str, Any]) -> dict[str, Any]:
    sphere_hit = frame.get("sphere_hit")
    if isinstance(sphere_hit, dict):
        return sphere_hit
    return {}
```

3. Delete `_monitor_hit_record`, `_compare_monitor_hit`, `_compare_monitor_hit_field_presence`, `_monitor_hit_path`, and `_monitor_uv_component`.

- [ ] **Step 4: Implement QA summary naming**

In `src/chess_gaze/qa_summary.py`, update labels and any diagnostic strings so generated summaries name:

```python
valid_sphere_hit_frames
invalid_sphere_hit_reasons
sphere_hit_angle_bounds
```

Remove references to:

```python
valid_monitor_hit_frames
invalid_monitor_hit_reasons
monitor_hit_bounds
```

- [ ] **Step 5: Run GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze -q -k "qa_summary or run_equivalence"
```

Expected after implementation: selected QA and equivalence tests pass.

- [ ] **Step 6: Commit Task 4**

Run:

```sh
git add src/chess_gaze/qa_summary.py src/chess_gaze/run_equivalence.py tests/chess_gaze
git commit -m "feat: validate sphere hit summaries"
```

---

### Task 5: Sphere Viewer Rendering

**Files:**
- Modify: `src/chess_gaze/viewer_assets/index.html`
- Modify: `src/chess_gaze/viewer_assets/scene_viewer.js`
- Modify: `src/chess_gaze/viewer_assets/styles.css`
- Modify: `tests/chess_gaze/test_scene_viewer.py`
- Modify: `tests/test_package_metadata.py` only if packaged asset expectations need new file names

**Interfaces:**
- Consumes:
  - `sceneData.gaze_sphere.radius_m`
  - `frame.sphere_hit`
  - `frame.unigaze_ray.origin_scene_m`
  - `frame.unigaze_ray.direction_scene`
  - `state.sceneData.valid_hit_points`
- Produces:
  - `data-testid="toggle-gaze-sphere"`
  - `data-testid="sphere-radius-m"`
  - `data-testid="sphere-radius-label"`
  - live sphere radius redraw for static sphere, current hit, accumulated hits, and hit-area patches
  - no viewer text or selectors named monitor plane, monitor rectangle, extended plane, or plane UV

- [ ] **Step 1: Write failing viewer tests**

In `tests/chess_gaze/test_scene_viewer.py`, replace monitor-control assertions with:

```python
def test_viewer_exposes_gaze_sphere_controls() -> None:
    html = VIEWER_INDEX.read_text(encoding="utf-8")

    assert 'data-testid="toggle-gaze-sphere"' in html
    assert 'data-testid="sphere-radius-m"' in html
    assert 'data-testid="sphere-radius-label"' in html
    assert "Sphere Radius" in html
    assert 'data-testid="toggle-monitor-plane"' not in html
    assert 'data-testid="toggle-monitor-rectangle"' not in html
    assert 'data-testid="toggle-extended-plane"' not in html
```

Add JavaScript source assertions:

```python
def test_scene_viewer_projects_hits_and_hit_areas_to_sphere() -> None:
    js = VIEWER_JS.read_text(encoding="utf-8")

    assert "DEFAULT_SPHERE_RADIUS_M = 0.7" in js
    assert "SPHERE_MIN_RADIUS_M = 0.35" in js
    assert "SPHERE_MAX_RADIUS_M = 1.2" in js
    assert "function intersectRayWithSphere" in js
    assert "function sphereHitForFrame" in js
    assert "function writeSphereHitAreaPatchPositions" in js
    assert "new THREE.SphereGeometry(1, 48, 24)" in js
    assert "frame?.sphere_hit" in js
    assert "main_monitor_hit" not in js
    assert "monitor_plane" not in js
    assert "plane_uv_m" not in js
```

Add CSS assertion:

```python
def test_viewer_styles_use_gaze_sphere_color_variable() -> None:
    css = VIEWER_CSS.read_text(encoding="utf-8")

    assert "--color-gaze-sphere:" in css
    assert "--color-monitor-plane:" not in css
```

- [ ] **Step 2: Run RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/test_package_metadata.py -q
```

Expected before implementation: tests fail because monitor controls and monitor viewer math still exist.

- [ ] **Step 3: Update viewer HTML**

In `src/chess_gaze/viewer_assets/index.html`, replace projection layer controls with:

```html
<label><input data-testid="toggle-gaze-sphere" type="checkbox" checked> Gaze Sphere</label>
<label class="range-control" for="sphere-radius-m">
  <span>Sphere Radius</span>
  <input
    id="sphere-radius-m"
    data-testid="sphere-radius-m"
    type="range"
    min="0.35"
    max="1.20"
    step="0.01"
    value="0.70"
  >
  <span data-testid="sphere-radius-label">0.70 m</span>
</label>
```

Remove these labels and inputs:

```html
Monitor Plane
Monitor Rectangle
Extended Plane
```

- [ ] **Step 4: Update viewer JavaScript constants and element bindings**

In `src/chess_gaze/viewer_assets/scene_viewer.js`, add constants:

```javascript
const DEFAULT_SPHERE_RADIUS_M = 0.7;
const SPHERE_MIN_RADIUS_M = 0.35;
const SPHERE_MAX_RADIUS_M = 1.2;
const SPHERE_RADIUS_STEP_M = 0.01;
const SPHERE_SURFACE_OFFSET_M = 0.002;
```

Change element bindings:

```javascript
gazeSphere: document.querySelector('[data-testid="toggle-gaze-sphere"]'),
sphereRadius: document.querySelector('[data-testid="sphere-radius-m"]'),
sphereRadiusLabel: document.querySelector('[data-testid="sphere-radius-label"]'),
```

Delete bindings for:

```javascript
monitorPlane
monitorRectangle
extendedPlane
```

- [ ] **Step 5: Add viewer sphere math**

Add these functions to `scene_viewer.js`:

```javascript
function sphereRadiusMeters() {
  const parsed = Number.parseFloat(elements.controls.sphereRadius?.value);
  if (!Number.isFinite(parsed)) {
    return state.sceneData?.gaze_sphere?.radius_m || DEFAULT_SPHERE_RADIUS_M;
  }
  return Math.min(SPHERE_MAX_RADIUS_M, Math.max(SPHERE_MIN_RADIUS_M, parsed));
}

function updateSphereRadiusLabel() {
  const radius = sphereRadiusMeters();
  if (elements.controls.sphereRadiusLabel) {
    elements.controls.sphereRadiusLabel.textContent = `${radius.toFixed(2)} m`;
  }
}

function intersectRayWithSphere(origin, direction, radius) {
  if (!origin || !direction || !Number.isFinite(radius) || radius <= 0) {
    return null;
  }
  const normalizedDirection = direction.clone().normalize();
  const a = normalizedDirection.dot(normalizedDirection);
  const b = 2 * origin.dot(normalizedDirection);
  const c = origin.dot(origin) - radius * radius;
  const discriminant = b * b - 4 * a * c;
  if (!Number.isFinite(discriminant) || discriminant < -1e-9) {
    return null;
  }
  const sqrtDiscriminant = Math.sqrt(Math.max(0, discriminant));
  const denominator = 2 * a;
  if (Math.abs(denominator) <= 1e-9) {
    return null;
  }
  const roots = [(-b - sqrtDiscriminant) / denominator, (-b + sqrtDiscriminant) / denominator]
    .filter((root) => Number.isFinite(root) && root >= -1e-9);
  if (!roots.length) {
    return null;
  }
  const nearestRoot = roots.reduce((best, root) => Math.min(best, root), Number.POSITIVE_INFINITY);
  const rayT = Math.max(0, nearestRoot);
  return origin.clone().add(normalizedDirection.multiplyScalar(rayT));
}

function sphereHitForFrame(frame) {
  const persisted = vector(frame?.sphere_hit?.point_scene_m);
  const radius = sphereRadiusMeters();
  const origin = vector(frame?.unigaze_ray?.origin_scene_m || frame?.unigaze_ray?.scene_m);
  const direction = vector(frame?.unigaze_ray?.direction_scene);
  const projected = intersectRayWithSphere(origin, direction, radius);
  return projected || persisted;
}

function surfaceOffsetPoint(point) {
  if (!point) {
    return null;
  }
  const normal = point.clone().normalize();
  return point.clone().add(normal.multiplyScalar(SPHERE_SURFACE_OFFSET_M));
}
```

Add hit-area patch functions that sample cone boundary rays and intersect each boundary ray with the current sphere radius. Use `BufferGeometry.setDrawRange(0, vertexCount)` so the geometry does not resize while the slider moves.

- [ ] **Step 6: Replace static projection layer rendering**

Replace monitor plane mesh and rectangles with a translucent sphere:

```javascript
function buildGazeSphere() {
  const geometry = new THREE.SphereGeometry(1, 48, 24);
  const material = materials.gazeSphere;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.scale.setScalar(sphereRadiusMeters());
  mesh.userData.layer = "gazeSphere";
  mesh.visible = elements.toggles.gazeSphere.checked;
  return mesh;
}
```

In `materials`, replace monitor materials with:

```javascript
gazeSphere: new THREE.MeshBasicMaterial({
  color: COLORS.gazeSphere,
  transparent: true,
  opacity: 0.14,
  depthWrite: false,
  side: THREE.DoubleSide,
}),
```

- [ ] **Step 7: Rebuild current and accumulated hits from sphere projection**

Update current-frame hit rendering so it calls:

```javascript
const hitPoint = surfaceOffsetPoint(sphereHitForFrame(frame));
```

Update accumulated hit buffers so they iterate:

```javascript
for (const frame of state.sceneData?.frames || []) {
  const hitPoint = surfaceOffsetPoint(sphereHitForFrame(frame));
  if (!hitPoint) {
    continue;
  }
  positions[index] = hitPoint.x;
  positions[index + 1] = hitPoint.y;
  positions[index + 2] = hitPoint.z;
  index += 3;
}
```

Keep cached typed arrays and update draw ranges instead of creating new `BufferGeometry` objects on every radius change.

- [ ] **Step 8: Wire radius slider updates**

Add event handling:

```javascript
elements.controls.sphereRadius?.addEventListener("input", () => {
  updateSphereRadiusLabel();
  rebuildStaticProjectionSurface();
  rebuildCurrentFrame();
  updateAccumulatedHitPoints();
  updateHitAreaPatches();
  render();
});
```

On scene-data load, set the slider value from persisted data:

```javascript
const persistedRadius = sceneData.gaze_sphere?.radius_m || DEFAULT_SPHERE_RADIUS_M;
elements.controls.sphereRadius.value = String(
  Math.min(SPHERE_MAX_RADIUS_M, Math.max(SPHERE_MIN_RADIUS_M, persistedRadius)).toFixed(2)
);
updateSphereRadiusLabel();
```

- [ ] **Step 9: Update status copy**

Change viewer status text to use:

```javascript
elements.hitStatus.textContent = frame?.sphere_hit?.valid
  ? "valid sphere hit"
  : frame?.sphere_hit?.reason_invalid || "invalid";
```

Change hit count to:

```javascript
elements.hitCount.textContent = String(
  state.sceneData?.frames?.filter((frame) => frame?.sphere_hit?.valid).length || 0
);
```

- [ ] **Step 10: Run GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_viewer.py tests/test_package_metadata.py -q
```

Expected after implementation: selected viewer and packaging tests pass.

- [ ] **Step 11: Commit Task 5**

Run:

```sh
git add src/chess_gaze/viewer_assets/index.html src/chess_gaze/viewer_assets/scene_viewer.js src/chess_gaze/viewer_assets/styles.css tests/chess_gaze/test_scene_viewer.py tests/test_package_metadata.py
git commit -m "feat: render gaze hits on sphere"
```

---

### Task 6: Documentation, Real Verification, And Closeout

**Files:**
- Modify: `README.md`
- Modify: `docs/development/architecture/source-layout.md`
- Add: `docs/superpowers/closeouts/2026-06-29-sphere-hit-projection.md`

**Interfaces:**
- Consumes:
  - completed Tasks 1 through 5
  - `artifacts/input/nakamura_short.mp4`
- Produces:
  - documented sphere projection contract
  - source-layout entry for `sphere_projection.py`
  - closeout with exact commands and pass/fail evidence

- [ ] **Step 1: Update README**

In `README.md`, replace the viewer and scene-coordinate descriptions so they state:

```markdown
The scene artifacts project every valid gaze ray onto a head-centered gaze sphere.
The sphere center is the robust scene center derived from valid eye midpoints, and
the default sphere radius is `0.700 m`. This radius represents the plausible
distance to a hypothetical screen-like surface that the gaze lands on.
```

Replace hit-area text with:

```markdown
The viewer includes a `Hit Area` layer. It samples the angular uncertainty cone
around each gaze ray and intersects those boundary rays with the current gaze
sphere radius, so hit areas stay on the sphere when the radius slider changes.
```

Remove README claims about:

```markdown
monitor plane
monitor rectangle
extended plane
physical monitor bounds
plane UV
```

- [ ] **Step 2: Update source-layout documentation**

In `docs/development/architecture/source-layout.md`, add `src/chess_gaze/sphere_projection.py` to the scene package map with this responsibility:

```markdown
- `sphere_projection.py`: focused domain module for gaze-sphere construction,
  ray-sphere intersection, spherical angles, and hemisphere classification.
  It is separate from `scene_geometry.py` because projection-surface math has a
  strict artifact contract and independent numerical edge cases.
```

- [ ] **Step 3: Write closeout skeleton with concrete evidence fields**

Create `docs/superpowers/closeouts/2026-06-29-sphere-hit-projection.md`:

```markdown
# Sphere Hit Projection Closeout

## Result

- Replaced monitor-plane hit artifacts with gaze-sphere hit artifacts.
- Rendered hit points and hit areas on a live-radius gaze sphere in the Three.js viewer.
- Preserved rear-hemisphere hits as valid direction evidence.

## Verification Evidence

- Focused Python tests:
- Viewer tests:
- Full pytest:
- Ruff check:
- Ruff format check:
- Mypy:
- Nakamura analyze command:
- Nakamura run directory:
- Nakamura scene frame count:
- Nakamura valid sphere hit count:
- Nakamura viewer smoke:
- Browser visual review:

## Residual Risk

- Model-backed verification depends on the local UniGaze and MediaPipe runtime installed on the machine running the command.
- The sphere radius slider is a visualization control; persisted artifacts keep the default radius for reproducibility.

## Commits

- docs: specify sphere hit projection
- docs: plan sphere hit projection
- feat: add gaze sphere projection math
- feat: define sphere scene schemas
- feat: write sphere scene artifacts
- feat: validate sphere hit summaries
- feat: render gaze hits on sphere
- docs: document sphere hit projection
```

During final verification, fill the evidence fields with exact command outputs, not estimates.

- [ ] **Step 4: Run focused gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/chess_gaze/test_scene_calibration.py tests/chess_gaze/test_sphere_projection.py tests/chess_gaze/test_scene_records.py tests/chess_gaze/test_scene_geometry.py tests/chess_gaze/test_scene_artifacts.py tests/chess_gaze/test_scene_artifacts_real_video_contract.py tests/chess_gaze/test_scene_viewer.py tests/test_package_metadata.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run full local gates**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Expected: all commands pass. If a command fails, use `superpowers:systematic-debugging`, reproduce the exact failure, add or repair a regression at the correct seam, rerun focused and broad gates, and record the failure plus repair in the closeout.

- [ ] **Step 6: Run real Nakamura verification**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze analyze artifacts/input/nakamura_short.mp4 --output-root artifacts/output --no-resume
```

Expected: command exits `0`, prints a run directory under `artifacts/output/nakamura_short/runs/`, and writes:

```text
records/scene_frames.jsonl
scene/scene_manifest.json
scene/scene_summary.json
viewer/index.html
viewer/scene-data.json
```

Then run:

```sh
find artifacts/output/nakamura_short/runs -mindepth 1 -maxdepth 1 -type d -print
```

Expected: at least one run directory is printed. Use the newest printed run directory for the next checks.

Run these artifact assertions with the run directory printed by `find`:

```sh
UV_CACHE_DIR=.uv-cache uv run python -c 'import json, pathlib; run=max(pathlib.Path("artifacts/output/nakamura_short/runs").iterdir(), key=lambda path: path.stat().st_mtime); frames=(run/"records/scene_frames.jsonl").read_text().splitlines(); summary=json.loads((run/"scene/scene_summary.json").read_text()); viewer=json.loads((run/"viewer/scene-data.json").read_text()); assert len(frames) == 180; assert summary["valid_sphere_hit_frames"] == 180; assert len(viewer["valid_hit_points"]) == 180; assert "main_monitor_hit" not in "\n".join(frames); assert "monitor_plane" not in (run/"viewer/scene-data.json").read_text(); print(run); print(summary["valid_sphere_hit_frames"])'
```

Expected output includes the run directory path and `180`.

- [ ] **Step 7: Run viewer smoke**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run chess-gaze view "$(python -c 'import pathlib; print(max(pathlib.Path("artifacts/output/nakamura_short/runs").iterdir(), key=lambda path: path.stat().st_mtime))')" --host 127.0.0.1 --port 0
```

Expected: command prints a loopback URL and keeps serving until interrupted. Open the printed URL with browser tooling, verify the scene is nonblank, verify a translucent sphere is visible, move the sphere-radius slider to `0.35`, `0.70`, and `1.20`, verify hit points and hit-area patches stay on the sphere surface, and capture at least one screenshot path or browser observation in the closeout.

- [ ] **Step 8: Search for stale active monitor projection terms**

Run:

```sh
rg -n "main_monitor_hit|monitor_hit|monitor_plane|plane_uv_m|valid_monitor_hit|invalid_monitor_hit|monitor_hit_bounds|toggle-monitor|extended-plane|Monitor Plane|Monitor Rectangle|Extended Plane" src tests README.md docs/development/architecture docs/superpowers/specs/2026-06-29-sphere-hit-projection-design.md
```

Expected: no hits in active source, tests, README, or current docs except historical notes in the approved spec that explicitly describe the replaced behavior.

- [ ] **Step 9: Commit Task 6**

Run:

```sh
git add README.md docs/development/architecture/source-layout.md docs/superpowers/closeouts/2026-06-29-sphere-hit-projection.md
git commit -m "docs: document sphere hit projection"
```

---

### Task 7: Final Whole-Branch Review

**Files:**
- No planned code edits.
- Use review output to decide whether a targeted repair commit is required.

**Interfaces:**
- Consumes:
  - all commits from Tasks 1 through 6
  - final verification output
- Produces:
  - final reviewer findings or explicit no-findings statement
  - final clean working tree or documented unrelated dirty files

- [ ] **Step 1: Use code-review skill**

Use `superpowers:requesting-code-review` before claiming the branch is complete.

The review prompt must include these binding requirements:

```text
Review the current branch for the sphere hit projection task. Required behavior:
1. No active artifact schema writes main_monitor_hit, monitor_hit, monitor_plane, or plane_uv_m.
2. Every valid gaze ray projects to sphere_hit on a sphere centered at scene origin.
3. Rear-hemisphere sphere hits are valid.
4. Viewer hit points and hit-area patches stay on the current slider radius.
5. Sphere radius slider range is 0.35m to 1.20m with default 0.70m.
6. Nakamura short verification used artifacts/input/nakamura_short.mp4 and recorded evidence.
```

- [ ] **Step 2: Repair review findings**

If the review finds a defect, write or update the smallest failing test that proves it, fix the durable surface, rerun focused tests, rerun full gates affected by the fix, update the closeout, then run `git status --short` and commit the concrete repaired files with message `fix: address sphere projection review findings`.

- [ ] **Step 3: Final verification before completion**

Use `superpowers:verification-before-completion`. Confirm:

```sh
git status --short
UV_CACHE_DIR=.uv-cache uv run pytest -q
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
UV_CACHE_DIR=.uv-cache uv run mypy
```

Expected: only ignored/generated artifacts may remain outside commits, and all gates pass. If full model-backed verification was not rerun after the last repair, rerun Task 6 Step 6 and Task 6 Step 7 before final response.
