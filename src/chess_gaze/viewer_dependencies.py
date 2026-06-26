from __future__ import annotations

from typing import Final

THREE_PACKAGE_NAME: Final = "three"
THREE_VERSION: Final = "0.185.0"
THREE_LICENSE: Final = "MIT"
THREE_SOURCE: Final = "npm:three"
THREE_REPOSITORY_URL: Final = "https://github.com/mrdoob/three.js"
THREE_TARBALL_URL: Final = (
    f"https://registry.npmjs.org/three/-/three-{THREE_VERSION}.tgz"
)
THREE_NPM_INTEGRITY: Final = (
    "sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP"
    "91SEAuj25bNAj7k1QQdf+srZywVK6w=="
)
THREE_CDN_PROVIDER: Final = "cdn.jsdelivr.net"
THREE_MODULE_URL: Final = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js"
)
THREE_CORE_URL: Final = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.core.js"
)
THREE_ADDONS_URL: Final = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/"
)
ORBIT_CONTROLS_URL: Final = f"{THREE_ADDONS_URL}controls/OrbitControls.js"

THREE_MODULE_URLS: Final[dict[str, str]] = {
    "three": THREE_MODULE_URL,
    "three/core": THREE_CORE_URL,
    "three/addons/": THREE_ADDONS_URL,
    "three/addons/controls/OrbitControls.js": ORBIT_CONTROLS_URL,
}


def three_import_map() -> dict[str, dict[str, str]]:
    return {
        "imports": {
            "three": THREE_MODULE_URL,
            "three/addons/": THREE_ADDONS_URL,
        }
    }


def viewer_dependency_manifest() -> dict[str, object]:
    return {
        "package_name": THREE_PACKAGE_NAME,
        "version": THREE_VERSION,
        "source": THREE_SOURCE,
        "license": THREE_LICENSE,
        "repository": THREE_REPOSITORY_URL,
        "tarball": THREE_TARBALL_URL,
        "integrity": THREE_NPM_INTEGRITY,
        "cdn_provider": THREE_CDN_PROVIDER,
        "module_urls": THREE_MODULE_URLS,
    }
