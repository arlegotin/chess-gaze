import hashlib
import json
from importlib import resources
from importlib.metadata import version

import chess_gaze

THREE_NPM_INTEGRITY = (
    "sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEA"
    "uj25bNAj7k1QQdf+srZywVK6w=="
)


def test_package_version_matches_installed_distribution() -> None:
    assert chess_gaze.__version__ == version("chess-gaze")


def test_public_api_is_metadata_only() -> None:
    assert chess_gaze.__all__ == ("__version__",)


def test_viewer_assets_are_packaged() -> None:
    expected_resources = [
        "index.html",
        "scene_viewer.js",
        "styles.css",
        "vendor/three.module.js",
        "vendor/OrbitControls.js",
        "vendor/THREE_LICENSE.txt",
        "vendor/vendor_manifest.json",
    ]

    viewer_assets = resources.files("chess_gaze").joinpath("viewer_assets")
    for resource_path in expected_resources:
        assert viewer_assets.joinpath(resource_path).is_file()

    manifest = json.loads(
        viewer_assets.joinpath("vendor/vendor_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["package_name"] == "three"
    assert manifest["version"] == "0.185.0"
    assert manifest["license"] == "MIT"
    assert manifest["repository"] == "https://github.com/mrdoob/three.js"
    assert manifest["tarball"] == "https://registry.npmjs.org/three/-/three-0.185.0.tgz"
    assert manifest["integrity"] == THREE_NPM_INTEGRITY

    copied_files = manifest["copied_files"]
    assert {
        copied_file["packaged_path"] for copied_file in copied_files
    } == {
        "vendor/three.module.js",
        "vendor/OrbitControls.js",
        "vendor/THREE_LICENSE.txt",
    }
    for copied_file in copied_files:
        resource = viewer_assets.joinpath(copied_file["packaged_path"])
        actual_sha256 = hashlib.sha256(resource.read_bytes()).hexdigest()
        assert copied_file["sha256"] == actual_sha256
