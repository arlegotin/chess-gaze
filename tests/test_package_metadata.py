import json
from importlib import resources
from importlib.metadata import version

import chess_gaze
from chess_gaze.viewer_dependencies import viewer_dependency_manifest

THREE_VERSION = "0.185.0"
THREE_NPM_INTEGRITY = (
    "sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEA"
    "uj25bNAj7k1QQdf+srZywVK6w=="
)
THREE_MODULE_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js"
)
THREE_CORE_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.core.js"
)
THREE_ADDONS_URL = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/"
ORBIT_CONTROLS_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/controls/"
    "OrbitControls.js"
)
EXPECTED_MODULE_URLS = {
    "three": THREE_MODULE_URL,
    "three/core": THREE_CORE_URL,
    "three/addons/": THREE_ADDONS_URL,
    "three/addons/controls/OrbitControls.js": ORBIT_CONTROLS_URL,
}


def test_package_version_matches_installed_distribution() -> None:
    assert chess_gaze.__version__ == version("chess-gaze")


def test_public_api_is_metadata_only() -> None:
    assert chess_gaze.__all__ == ("__version__",)


def test_viewer_assets_are_packaged_for_remote_three_loading() -> None:
    expected_resources = [
        "index.html",
        "scene_viewer.js",
        "styles.css",
        "viewer_dependency_manifest.json",
    ]

    viewer_assets = resources.files("chess_gaze").joinpath("viewer_assets")
    for resource_path in expected_resources:
        assert viewer_assets.joinpath(resource_path).is_file()

    assert not viewer_assets.joinpath("vendor").is_dir()

    manifest = json.loads(
        viewer_assets.joinpath("viewer_dependency_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest == viewer_dependency_manifest()
    assert manifest["package_name"] == "three"
    assert manifest["version"] == THREE_VERSION
    assert manifest["source"] == "npm:three"
    assert manifest["license"] == "MIT"
    assert manifest["repository"] == "https://github.com/mrdoob/three.js"
    assert manifest["tarball"] == (
        f"https://registry.npmjs.org/three/-/three-{THREE_VERSION}.tgz"
    )
    assert manifest["integrity"] == THREE_NPM_INTEGRITY
    assert manifest["cdn_provider"] == "cdn.jsdelivr.net"
    assert manifest["module_urls"] == EXPECTED_MODULE_URLS
    assert "copied_files" not in manifest
    assert "local_patches" not in manifest


def test_packaged_scene_viewer_uses_import_map_specifiers() -> None:
    viewer_assets = resources.files("chess_gaze").joinpath("viewer_assets")
    scene_viewer_js = viewer_assets.joinpath("scene_viewer.js").read_text(
        encoding="utf-8"
    )

    assert 'from "three"' in scene_viewer_js
    assert 'from "three/addons/controls/OrbitControls.js"' in scene_viewer_js
    assert "./vendor/" not in scene_viewer_js
    assert "https://" not in scene_viewer_js
    assert "http://" not in scene_viewer_js
