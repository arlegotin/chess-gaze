from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from chess_gaze.artifact_runs import RunLayout
from chess_gaze.image_io import atomic_write_bytes
from chess_gaze.scene_artifacts import (
    SceneArtifactResult,
    scene_result_with_viewer_exists,
)
from chess_gaze.scene_records import SceneSummary, ViewerSceneData


@dataclass(frozen=True)
class ViewerBuildResult:
    viewer_dir: Path
    index_path: Path
    scene_data_path: Path


def build_scene_viewer(
    run_layout: RunLayout, scene_result: SceneArtifactResult
) -> ViewerBuildResult:
    viewer_dir = run_layout.viewer_dir
    updated_scene_result = scene_result_with_viewer_exists(
        scene_result,
        viewer_exists=True,
    )
    copy_viewer_assets(viewer_dir)
    _write_scene_summary(
        updated_scene_result.paths.scene_summary_path,
        updated_scene_result.summary,
    )
    scene_data_path = write_viewer_scene_data(
        viewer_dir,
        updated_scene_result.viewer_data,
    )
    return ViewerBuildResult(
        viewer_dir=viewer_dir,
        index_path=viewer_dir / "index.html",
        scene_data_path=scene_data_path,
    )


def copy_viewer_assets(viewer_dir: Path) -> None:
    viewer_assets = resources.files("chess_gaze").joinpath("viewer_assets")
    _copy_resource_tree(viewer_assets, viewer_dir)


def write_viewer_scene_data(viewer_dir: Path, data: ViewerSceneData) -> Path:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    path = viewer_dir / "scene-data.json"
    _write_json(path, data.model_dump(mode="json", by_alias=True))
    return path


def _write_scene_summary(path: Path, summary: SceneSummary) -> None:
    _write_json(path, summary.model_dump(mode="json", by_alias=True))


def _write_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(path, encoded)


def _copy_resource_tree(source: Traversable, destination: Path) -> None:
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_resource_tree(child, destination / child.name)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, source.read_bytes())
