"""Evaluation module for iceberg drift prediction."""

from .metrics import (
    compute_all_metrics,
    position_error_km,
    velocity_error,
    along_cross_track_error,
    drift_distance_error,
    direction_error,
)
from .trainer import DriftTrainer, train_model, TrainingConfig
from .validator import DriftValidator, validate_model, ValidationConfig

__all__ = [
    "compute_all_metrics",
    "position_error_km",
    "velocity_error",
    "along_cross_track_error",
    "drift_distance_error",
    "direction_error",
    "DriftTrainer",
    "train_model",
    "TrainingConfig",
    "DriftValidator",
    "validate_model",
    "ValidationConfig",
]