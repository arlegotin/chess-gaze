from pathlib import Path

from chess_gaze.cli import main


def test_analyze_requires_video_path(capsys) -> None:
    exit_code = main(["analyze"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "video_path" in captured.err


def test_unknown_command_returns_usage(capsys) -> None:
    exit_code = main(["unknown"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "usage:" in captured.err


def test_missing_input_returns_stable_error_without_output_dir(
    tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "missing.mp4"
    output_root = tmp_path / "output"

    exit_code = main(
        [
            "analyze",
            str(missing),
            "--output-root",
            str(output_root),
            "--models-root",
            str(tmp_path / "models"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 10
    assert "INPUT_NOT_FOUND" in captured.err
    assert not output_root.exists()
