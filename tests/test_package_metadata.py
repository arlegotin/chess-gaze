import hashlib
import json
import shutil
import subprocess
from importlib import resources
from importlib.metadata import version
from pathlib import Path

import pytest

import chess_gaze

THREE_NPM_INTEGRITY = (
    "sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEA"
    "uj25bNAj7k1QQdf+srZywVK6w=="
)
ORBIT_CONTROLS_LOCAL_IMPORT_PATCH = (
    "Replaced bare 'three' import with local './three.module.js' import."
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
        "vendor/three.core.js",
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
    assert manifest["local_patches"] == [
        {
            "packaged_path": "vendor/OrbitControls.js",
            "description": ORBIT_CONTROLS_LOCAL_IMPORT_PATCH,
        }
    ]

    copied_files = manifest["copied_files"]
    assert {copied_file["packaged_path"] for copied_file in copied_files} == {
        "vendor/three.module.js",
        "vendor/three.core.js",
        "vendor/OrbitControls.js",
        "vendor/THREE_LICENSE.txt",
    }
    for copied_file in copied_files:
        resource = viewer_assets.joinpath(copied_file["packaged_path"])
        actual_sha256 = hashlib.sha256(resource.read_bytes()).hexdigest()
        assert copied_file["sha256"] == actual_sha256

    orbit_controls = viewer_assets.joinpath("vendor/OrbitControls.js").read_text(
        encoding="utf-8"
    )
    assert "from 'three'" not in orbit_controls
    assert 'from "three"' not in orbit_controls
    assert "from './three.module.js'" in orbit_controls


def test_viewer_vendor_modules_are_esm_importable() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; skipping vendored ESM import check.")

    vendor_assets = Path(
        str(resources.files("chess_gaze").joinpath("viewer_assets/vendor"))
    )
    three_module_url = vendor_assets.joinpath("three.module.js").as_uri()
    orbit_controls_url = vendor_assets.joinpath("OrbitControls.js").as_uri()
    import_script = """
const modules = process.argv.slice(1);
const results = await Promise.allSettled(modules.map((moduleUrl) => import(moduleUrl)));
const failures = results
  .map((result, index) => ({ result, moduleUrl: modules[index] }))
  .filter(({ result }) => result.status === "rejected");
if (failures.length > 0) {
  for (const { result, moduleUrl } of failures) {
    console.error(`${moduleUrl}: ${result.reason.message}`);
  }
  process.exit(1);
}
"""

    result = subprocess.run(
        [
            node,
            "--experimental-default-type=module",
            "--input-type=module",
            "-e",
            import_script,
            three_module_url,
            orbit_controls_url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
