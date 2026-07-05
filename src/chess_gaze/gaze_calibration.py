from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

_AFFINE_FEATURE_COUNT = 5


@dataclass(frozen=True)
class GazeCalibrationSample:
    yaw_radians: float
    pitch_radians: float
    head_yaw_radians: float
    head_pitch_radians: float
    target_x: float
    target_y: float


@dataclass(frozen=True)
class AffineGazeCalibrationEvaluation:
    sample_count: int
    mean_absolute_error: float
    root_mean_squared_error: float


@dataclass(frozen=True)
class AffineGazeCalibrator:
    coefficients: npt.NDArray[np.float64]
    ridge_lambda: float
    training_sample_count: int
    training_mean_absolute_error: float
    training_root_mean_squared_error: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.shape != (2, _AFFINE_FEATURE_COUNT):
            raise ValueError("coefficients must have shape (2, 5)")
        if not np.isfinite(coefficients).all():
            raise ValueError("coefficients must be finite")
        object.__setattr__(self, "coefficients", coefficients)

    def predict(self, sample: GazeCalibrationSample) -> tuple[float, float]:
        feature_vector = _feature_vector(sample)
        prediction = self.coefficients @ feature_vector
        return float(prediction[0]), float(prediction[1])


def fit_affine_gaze_calibrator(
    samples: Sequence[GazeCalibrationSample],
    ridge_lambda: float = 0.0,
) -> AffineGazeCalibrator:
    _require_non_negative_finite(ridge_lambda, "ridge_lambda")
    if len(samples) < _AFFINE_FEATURE_COUNT:
        raise ValueError("fit_affine_gaze_calibrator requires at least 5 samples")

    design_matrix, targets = _design_matrix_and_targets(samples)
    regularizer = np.eye(_AFFINE_FEATURE_COUNT, dtype=np.float64) * ridge_lambda
    coefficients_t = np.linalg.solve(
        design_matrix.T @ design_matrix + regularizer,
        design_matrix.T @ targets,
    )
    coefficients = coefficients_t.T
    training_metrics = _evaluate_coefficients(coefficients, design_matrix, targets)
    return AffineGazeCalibrator(
        coefficients=coefficients,
        ridge_lambda=ridge_lambda,
        training_sample_count=len(samples),
        training_mean_absolute_error=training_metrics.mean_absolute_error,
        training_root_mean_squared_error=training_metrics.root_mean_squared_error,
    )


def evaluate_affine_gaze_calibrator(
    model: AffineGazeCalibrator,
    samples: Sequence[GazeCalibrationSample],
) -> AffineGazeCalibrationEvaluation:
    if not samples:
        raise ValueError("evaluate_affine_gaze_calibrator requires at least 1 sample")
    design_matrix, targets = _design_matrix_and_targets(samples)
    return _evaluate_coefficients(model.coefficients, design_matrix, targets)


def _design_matrix_and_targets(
    samples: Sequence[GazeCalibrationSample],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    design_rows = [_feature_vector(sample) for sample in samples]
    target_rows = [_target_vector(sample) for sample in samples]
    return (
        np.asarray(design_rows, dtype=np.float64),
        np.asarray(target_rows, dtype=np.float64),
    )


def _feature_vector(sample: GazeCalibrationSample) -> npt.NDArray[np.float64]:
    values = np.asarray(
        (
            1.0,
            sample.yaw_radians,
            sample.pitch_radians,
            sample.head_yaw_radians,
            sample.head_pitch_radians,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(
            "gaze calibration samples must not contain non-finite features"
        )
    return values


def _target_vector(sample: GazeCalibrationSample) -> npt.NDArray[np.float64]:
    values = np.asarray((sample.target_x, sample.target_y), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            "gaze calibration samples must not contain non-finite targets"
        )
    return values


def _evaluate_coefficients(
    coefficients: npt.NDArray[np.float64],
    design_matrix: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
) -> AffineGazeCalibrationEvaluation:
    residuals = (design_matrix @ coefficients.T) - targets
    distances = np.linalg.norm(residuals, axis=1)
    mean_absolute_error = float(np.mean(distances))
    root_mean_squared_error = float(np.sqrt(np.mean(np.square(distances))))
    return AffineGazeCalibrationEvaluation(
        sample_count=int(targets.shape[0]),
        mean_absolute_error=mean_absolute_error,
        root_mean_squared_error=root_mean_squared_error,
    )


def _require_non_negative_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
