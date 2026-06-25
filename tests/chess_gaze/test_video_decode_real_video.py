from __future__ import annotations

from pathlib import Path

import pytest

from chess_gaze.video_decode import inspect_video, iter_decoded_frames


@pytest.mark.parametrize(
    ("path", "expected_count"),
    [
        (Path("artifacts/input/test_1.mp4"), 3613),
        (Path("artifacts/input/test_2.mp4"), 1973),
    ],
)
def test_real_video_decode_matches_local_evidence(
    path: Path, expected_count: int
) -> None:
    assert path.is_file(), f"missing mandatory real-data video: {path}"

    inspection = inspect_video(path)
    decoded_count = sum(1 for _ in iter_decoded_frames(path))
    print(
        f"{path}: expected={expected_count} "
        f"inspected={inspection.frame_count_decoded} decoded={decoded_count}"
    )

    assert inspection.frame_count_decoded == decoded_count
    assert decoded_count == expected_count, (
        f"{path} decoded {decoded_count} frames; expected {expected_count}"
    )
