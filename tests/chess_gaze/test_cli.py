from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
from pytest import CaptureFixture, MonkeyPatch

import chess_gaze.cli as cli
from chess_gaze.cli import main
from chess_gaze.frame_records import FrameRecord
from chess_gaze.pipeline import AnalyzeRequest, ObserverBundle, ObserverFrame
from chess_gaze.pipeline import analyze_video as real_analyze_video


def make_tiny_video(path: Path) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=3)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_analyze_requires_video_path(capsys: CaptureFixture[str]) -> None:
    exit_code = main(["analyze"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "video_path" in captured.err


def test_unknown_command_returns_usage(capsys: CaptureFixture[str]) -> None:
    exit_code = main(["unknown"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "usage:" in captured.err


def test_missing_input_returns_stable_error_without_output_dir(
    tmp_path: Path, capsys: CaptureFixture[str]
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


def test_analyze_missing_models_returns_stable_error_without_output_dir(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path)

    exit_code = main(
        [
            "analyze",
            str(video_path),
            "--output-root",
            str(output_root),
            "--models-root",
            str(tmp_path / "models"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 12
    assert "MODEL_ASSET_MISSING" in captured.err
    assert not output_root.exists()


def test_analyze_prints_run_dir_and_viewer_path(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    viewer_index_path = run_dir / "viewer" / "index.html"
    make_tiny_video(video_path)

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=viewer_index_path,
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    exit_code = main(["analyze", str(video_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines() == [
        str(run_dir),
        f"viewer: {viewer_index_path}",
    ]


def test_analyze_passes_unigaze_cli_overrides(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    viewer_index_path = run_dir / "viewer" / "index.html"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=viewer_index_path,
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    exit_code = main(
        [
            "analyze",
            str(video_path),
            "--unigaze-device",
            "mps",
            "--unigaze-batch-size",
            "7",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().err == ""
    [request] = captured_requests
    assert request.unigaze_device == "mps"
    assert request.unigaze_batch_size == 7


def test_analyze_rejects_invalid_unigaze_batch_size_override(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    output_root = tmp_path / "output"
    make_tiny_video(video_path)

    def fail_if_observer_runs(frame: ObserverFrame) -> FrameRecord:
        del frame
        raise AssertionError("invalid overrides must fail before observer execution")

    def analyze_with_fake_observer(request: AnalyzeRequest) -> object:
        return real_analyze_video(
            request,
            observers=ObserverBundle(frame_observer=fail_if_observer_runs),
        )

    monkeypatch.setattr(cli, "analyze_video", analyze_with_fake_observer)

    exit_code = main(
        [
            "analyze",
            str(video_path),
            "--output-root",
            str(output_root),
            "--unigaze-batch-size",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "USAGE" in captured.err
    assert "unigaze_batch_size" in captured.err
    assert not output_root.exists()


def test_view_prints_localhost_url_for_run_viewer(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class FakeViewerServer:
        url = "http://127.0.0.1:54321/"
        closed = False

        def serve_forever(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    server = FakeViewerServer()

    def fake_serve_viewer(
        received_run_dir: Path, host: str = "127.0.0.1", port: int = 0
    ) -> FakeViewerServer:
        assert received_run_dir == run_dir
        assert host == "127.0.0.1"
        assert port == 0
        return server

    monkeypatch.setattr(cli, "serve_viewer", fake_serve_viewer, raising=False)

    exit_code = main(["view", str(run_dir)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "http://127.0.0.1:54321/\n"
    assert server.closed is True


def test_view_rejects_missing_run_directory_or_viewer_files(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    missing_run_dir = tmp_path / "missing"

    assert main(["view", str(missing_run_dir)]) == 2
    assert "usage:" in capsys.readouterr().err

    run_dir = tmp_path / "run"
    viewer_dir = run_dir / "viewer"
    viewer_dir.mkdir(parents=True)

    assert main(["view", str(run_dir)]) == 2
    assert "usage:" in capsys.readouterr().err

    (viewer_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")

    assert main(["view", str(run_dir)]) == 2
    assert "usage:" in capsys.readouterr().err


def test_view_rejects_non_loopback_host(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    viewer_dir = run_dir / "viewer"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (viewer_dir / "scene-data.json").write_text("{}", encoding="utf-8")

    exit_code = main(["view", str(run_dir), "--host", "0.0.0.0"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "loopback" in captured.err
