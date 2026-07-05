from __future__ import annotations

import math

import numpy as np
import pytest

from chess_gaze.gaze_calibration import (
    GazeCalibrationSample,
    evaluate_affine_gaze_calibrator,
    fit_affine_gaze_calibrator,
)


def _make_sample(
    *,
    yaw: float,
    pitch: float,
    head_yaw: float,
    head_pitch: float,
    coefficients_x: tuple[float, float, float, float, float],
    coefficients_y: tuple[float, float, float, float, float],
) -> GazeCalibrationSample:
    features = (1.0, yaw, pitch, head_yaw, head_pitch)
    return GazeCalibrationSample(
        yaw_radians=yaw,
        pitch_radians=pitch,
        head_yaw_radians=head_yaw,
        head_pitch_radians=head_pitch,
        target_x=sum(weight * value for weight, value in zip(coefficients_x, features)),
        target_y=sum(weight * value for weight, value in zip(coefficients_y, features)),
    )


def test_fit_affine_gaze_calibrator_recovers_exact_affine_mapping() -> None:
    coefficients_x = (0.25, 1.5, -0.75, 0.5, 0.25)
    coefficients_y = (-0.10, 0.2, 1.1, -0.4, 0.9)
    training_samples = [
        _make_sample(
            yaw=yaw,
            pitch=pitch,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            coefficients_x=coefficients_x,
            coefficients_y=coefficients_y,
        )
        for yaw, pitch, head_yaw, head_pitch in (
            (-0.4, -0.2, -0.1, 0.0),
            (-0.2, 0.3, 0.0, -0.4),
            (0.0, -0.1, 0.2, 0.5),
            (0.3, 0.1, -0.3, 0.2),
            (0.5, -0.4, 0.4, -0.2),
            (0.7, 0.6, 0.1, 0.3),
        )
    ]
    held_out_samples = [
        _make_sample(
            yaw=yaw,
            pitch=pitch,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            coefficients_x=coefficients_x,
            coefficients_y=coefficients_y,
        )
        for yaw, pitch, head_yaw, head_pitch in (
            (-0.6, 0.4, 0.2, -0.1),
            (0.2, -0.5, -0.4, 0.6),
        )
    ]

    model = fit_affine_gaze_calibrator(training_samples, ridge_lambda=0.0)

    np.testing.assert_allclose(
        model.coefficients,
        np.asarray((coefficients_x, coefficients_y), dtype=np.float64),
        atol=1e-12,
    )
    assert model.training_sample_count == len(training_samples)
    assert model.training_mean_absolute_error == pytest.approx(0.0, abs=1e-12)
    assert model.training_root_mean_squared_error == pytest.approx(0.0, abs=1e-12)

    held_out_metrics = evaluate_affine_gaze_calibrator(model, held_out_samples)

    assert held_out_metrics.sample_count == len(held_out_samples)
    assert held_out_metrics.mean_absolute_error == pytest.approx(0.0, abs=1e-12)
    assert held_out_metrics.root_mean_squared_error == pytest.approx(0.0, abs=1e-12)


def test_evaluate_affine_gaze_calibrator_reports_held_out_error_separately() -> None:
    training_relation_x = (0.0, 1.0, 0.5, -0.25, 0.75)
    training_relation_y = (0.1, -0.3, 0.8, 0.2, -0.6)
    held_out_relation_x = (0.4, -0.2, 0.6, 0.5, 0.1)
    held_out_relation_y = (-0.2, 0.9, -0.4, 0.3, 0.7)
    training_samples = [
        _make_sample(
            yaw=yaw,
            pitch=pitch,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            coefficients_x=training_relation_x,
            coefficients_y=training_relation_y,
        )
        for yaw, pitch, head_yaw, head_pitch in (
            (-0.5, -0.4, -0.2, 0.3),
            (-0.1, 0.2, 0.5, -0.3),
            (0.0, -0.1, 0.1, 0.4),
            (0.4, 0.5, -0.4, 0.2),
            (0.8, -0.2, 0.3, -0.5),
        )
    ]
    held_out_samples = [
        _make_sample(
            yaw=yaw,
            pitch=pitch,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            coefficients_x=held_out_relation_x,
            coefficients_y=held_out_relation_y,
        )
        for yaw, pitch, head_yaw, head_pitch in (
            (-0.6, 0.1, 0.4, -0.2),
            (0.3, -0.5, -0.3, 0.7),
        )
    ]

    model = fit_affine_gaze_calibrator(training_samples, ridge_lambda=0.0)
    held_out_metrics = evaluate_affine_gaze_calibrator(model, held_out_samples)

    assert model.training_mean_absolute_error == pytest.approx(0.0, abs=1e-12)
    assert model.training_root_mean_squared_error == pytest.approx(0.0, abs=1e-12)
    assert held_out_metrics.mean_absolute_error > 0.1
    assert held_out_metrics.root_mean_squared_error > 0.1


def test_fit_affine_gaze_calibrator_rejects_insufficient_samples() -> None:
    samples = [
        GazeCalibrationSample(
            yaw_radians=float(index),
            pitch_radians=float(index) / 10.0,
            head_yaw_radians=float(index) / 20.0,
            head_pitch_radians=float(index) / 30.0,
            target_x=float(index) / 40.0,
            target_y=float(index) / 50.0,
        )
        for index in range(4)
    ]

    with pytest.raises(ValueError, match="at least 5 samples"):
        fit_affine_gaze_calibrator(samples)


def test_fit_affine_gaze_calibrator_rejects_non_finite_inputs() -> None:
    samples = [
        GazeCalibrationSample(
            yaw_radians=0.1,
            pitch_radians=0.2,
            head_yaw_radians=0.3,
            head_pitch_radians=0.4,
            target_x=0.5,
            target_y=0.6,
        ),
        GazeCalibrationSample(
            yaw_radians=math.nan,
            pitch_radians=0.0,
            head_yaw_radians=0.0,
            head_pitch_radians=0.0,
            target_x=0.0,
            target_y=0.0,
        ),
        GazeCalibrationSample(
            yaw_radians=0.7,
            pitch_radians=0.8,
            head_yaw_radians=0.9,
            head_pitch_radians=1.0,
            target_x=1.1,
            target_y=1.2,
        ),
        GazeCalibrationSample(
            yaw_radians=1.3,
            pitch_radians=1.4,
            head_yaw_radians=1.5,
            head_pitch_radians=1.6,
            target_x=1.7,
            target_y=1.8,
        ),
        GazeCalibrationSample(
            yaw_radians=1.9,
            pitch_radians=2.0,
            head_yaw_radians=2.1,
            head_pitch_radians=2.2,
            target_x=2.3,
            target_y=2.4,
        ),
    ]

    with pytest.raises(ValueError, match="non-finite"):
        fit_affine_gaze_calibrator(samples)
