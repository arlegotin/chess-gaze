from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
from pytest import CaptureFixture, MonkeyPatch

import chess_gaze.cli as cli
from chess_gaze.cli import main
from chess_gaze.frame_records import FrameRecord
from chess_gaze.pipeline import (
    AnalysisProgressEvent,
    AnalyzeRequest,
    ObserverBundle,
    ObserverFrame,
)
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


def test_missing_input_does_not_load_analysis_pipeline(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    missing = tmp_path / "missing.mp4"

    def fail_if_pipeline_loads() -> object:
        raise AssertionError("missing input must fail before loading pipeline")

    monkeypatch.setattr(cli, "_pipeline_dependencies", fail_if_pipeline_loads)

    assert main(["analyze", str(missing)]) == 10


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
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
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
    [request] = captured_requests
    assert request.unigaze_device is None
    assert request.unigaze_batch_size is None
    assert request.save_frame_images is None
    assert request.save_crop_images is None


def test_analyze_enables_resume_by_default(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=run_dir / "viewer" / "index.html",
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path)]) == 0

    [request] = captured_requests
    assert request.resume is True


def test_analyze_no_resume_forces_fresh_run_request(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=run_dir / "viewer" / "index.html",
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path), "--no-resume"]) == 0

    [request] = captured_requests
    assert request.resume is False


def test_analyze_save_frames_flag_requests_frame_image_retention(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=run_dir / "viewer" / "index.html",
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path), "--save-frames"]) == 0

    [request] = captured_requests
    assert request.save_frame_images is True
    assert request.save_crop_images is None


def test_analyze_save_crops_flag_requests_crop_image_retention(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=run_dir / "viewer" / "index.html",
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path), "--save-crops"]) == 0

    [request] = captured_requests
    assert request.save_frame_images is None
    assert request.save_crop_images is True


def test_analyze_progress_off_disables_progress_callback(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=run_dir / "viewer" / "index.html",
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path), "--progress", "off"]) == 0

    [request] = captured_requests
    assert request.progress_callback is None


def test_analyze_progress_on_requests_progress_callback(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    make_tiny_video(video_path)
    captured_requests: list[AnalyzeRequest] = []

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=run_dir / "viewer" / "index.html",
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)

    assert main(["analyze", str(video_path), "--progress", "on"]) == 0

    [request] = captured_requests
    assert request.progress_callback is not None


def test_analyze_uses_native_log_filter_for_progress_stream(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.mp4"
    run_dir = tmp_path / "runs" / "run-1"
    make_tiny_video(video_path)
    entered = False
    exited = False

    class FakeNativeLogFilter:
        stderr = SimpleNamespace(
            isatty=lambda: True,
            write=lambda _text: None,
            flush=lambda: None,
        )

        def __enter__(self) -> "FakeNativeLogFilter":
            nonlocal entered
            entered = True
            return self

        def __exit__(self, *args: object) -> None:
            nonlocal exited
            exited = True

    def fake_analyze_video(request: AnalyzeRequest) -> object:
        if request.progress_callback is not None:
            request.progress_callback(
                AnalysisProgressEvent(
                    run_dir=run_dir,
                    completed_frames=0,
                    total_frames=1,
                )
            )
            request.progress_callback(
                AnalysisProgressEvent(
                    run_dir=run_dir,
                    completed_frames=1,
                    total_frames=1,
                )
            )
        return SimpleNamespace(
            layout=SimpleNamespace(run_dir=run_dir),
            viewer_index_path=run_dir / "viewer" / "index.html",
        )

    monkeypatch.setattr(cli, "analyze_video", fake_analyze_video)
    monkeypatch.setattr(
        cli,
        "suppress_known_native_analysis_logs",
        lambda: FakeNativeLogFilter(),
    )

    assert main(["analyze", str(video_path), "--progress", "on"]) == 0
    assert entered is True
    assert exited is True


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
