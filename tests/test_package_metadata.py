from importlib.metadata import version

import chess_gaze


def test_package_version_matches_installed_distribution() -> None:
    assert chess_gaze.__version__ == version("chess-gaze")


def test_public_api_is_metadata_only() -> None:
    assert chess_gaze.__all__ == ("__version__",)
