from chess_gaze.native_log_filter import _NativeStderrLineFilter


def test_native_log_filter_suppresses_known_dependency_startup_lines() -> None:
    line_filter = _NativeStderrLineFilter()

    assert line_filter.should_suppress(
        "objc[1]: Class AVFFrameReceiver is implemented in both a and b."
    )
    assert line_filter.should_suppress(
        "objc[1]: Class AVFAudioReceiver is implemented in both a and b."
    )
    assert line_filter.should_suppress(
        "I0000 00:00:1 init-domain.cc:128] Fiber init: default domain = pthread"
    )
    assert line_filter.should_suppress(
        "W0000 00:00:1 face_landmarker_graph.cc:180] "
        "Sets FaceBlendshapesGraph acceleration to xnnpack by default."
    )
    assert line_filter.should_suppress(
        "I0000 00:00:1 gl_context.cc:407] GL version: 2.1"
    )
    assert line_filter.should_suppress(
        "INFO: Created TensorFlow Lite XNNPACK delegate for CPU."
    )
    assert line_filter.should_suppress(
        "W0000 00:00:1 inference_feedback_manager.cc:121] "
        "Feedback manager requires a model with a single signature inference."
    )


def test_native_log_filter_suppresses_clearcut_block_trace() -> None:
    line_filter = _NativeStderrLineFilter()

    assert line_filter.should_suppress(
        "E0000 00:00:1 portable_clearcut_uploader.cc:90] "
        "Failed to send to clearcut: FAILED_PRECONDITION"
    )
    assert line_filter.should_suppress("=== Source Location Trace: ===")
    assert line_filter.should_suppress(
        "wireless/android/play/playlog/cplusplus/portable_clearcut_uploader.cc:180"
    )


def test_native_log_filter_passes_unknown_stderr() -> None:
    line_filter = _NativeStderrLineFilter()

    assert not line_filter.should_suppress("SCHEMA_VALIDATION_FAILED: bad frame")
    assert not line_filter.should_suppress("E0000 unknown_native_file.cc:90] bad")
