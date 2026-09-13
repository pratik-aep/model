"""Tests for evaluation metrics."""

import pytest
import numpy as np
import pandas as pd

from iceberg_drift.evaluation.metrics import (
    position_error_km,
    velocity_error,
    along_cross_track_error,
    drift_distance_error,
    direction_error,
    compute_all_metrics,
    bootstrap_confidence_interval,
    compare_models_bootstrap,
    regression_metrics,
)


class TestMetrics:
    """Test metric functions."""

    def test_position_error_km(self):
        # Same position
        err = position_error_km(0, 0, 0, 0)
        assert err == 0

        # 1 degree latitude ≈ 111 km
        err = position_error_km(0, 0, 1, 0)
        assert abs(err - 111.0) < 2

        # 1 degree longitude at equator ≈ 111 km
        err = position_error_km(0, 0, 0, 1)
        assert abs(err - 111.0) < 2

        # 1 degree longitude at 60° latitude ≈ 55.5 km
        err = position_error_km(60, 0, 60, 1)
        assert abs(err - 55.5) < 2

    def test_velocity_error(self):
        pred_u, pred_v = np.array([1.0]), np.array([0.0])
        true_u, true_v = np.array([1.0]), np.array([0.0])
        speed_err, dir_err = velocity_error(pred_u, pred_v, true_u, true_v)
        assert speed_err[0] == 0
        assert dir_err[0] == 0

        # Different speed
        pred_u, pred_v = np.array([2.0]), np.array([0.0])
        true_u, true_v = np.array([1.0]), np.array([0.0])
        speed_err, dir_err = velocity_error(pred_u, pred_v, true_u, true_v)
        assert speed_err[0] == 1.0

        # 90 degree direction difference
        pred_u, pred_v = np.array([0.0]), np.array([1.0])
        true_u, true_v = np.array([1.0]), np.array([0.0])
        speed_err, dir_err = velocity_error(pred_u, pred_v, true_u, true_v)
        assert abs(dir_err[0] - 90) < 1

    def test_direction_error(self):
        # Same direction
        err = direction_error(np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
        assert err[0] == 0

        # Opposite direction
        err = direction_error(np.array([1.0]), np.array([0.0]), np.array([-1.0]), np.array([0.0]))
        assert err[0] == 180

        # 90 degrees
        err = direction_error(np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([0.0]))
        assert err[0] == 90

    def test_along_cross_track_error(self):
        # True track: North (lat increasing)
        true_lat = np.array([-65.0, -64.0])
        true_lon = np.array([0.0, 0.0])
        ref_lat = np.array([-65.0])
        ref_lon = np.array([0.0])

        # Prediction: 100km East of true track
        pred_lat = np.array([-64.0])
        pred_lon = np.array([1.0])  # ~111km at equator, less at -64

        along, cross = along_cross_track_error(
            pred_lat, pred_lon, true_lat[-1:], true_lon[-1:], ref_lat, ref_lon
        )
        # Should be mostly cross-track error
        assert abs(cross[0]) > abs(along[0])

    def test_drift_distance_error(self):
        init_lat, init_lon = -65.0, 0.0
        true_lat, true_lon = -64.0, 0.0  # ~111km North
        pred_lat, pred_lon = -64.0, 1.0  # ~111km North + ~111km East

        dist_err, dir_err = drift_distance_error(
            np.array([pred_lat]), np.array([pred_lon]),
            np.array([true_lat]), np.array([true_lon]),
            np.array([init_lat]), np.array([init_lon]),
        )
        # Distance error should be ~111km (extra eastward drift)
        assert abs(dist_err[0] - 111) < 20
        # Direction error should be ~45 degrees
        assert abs(dir_err[0] - 45) < 10

    def test_compute_all_metrics(self):
        n = 50
        h = 12
        pred_u = np.random.normal(0.1, 0.05, (n, h))
        pred_v = np.random.normal(0.05, 0.05, (n, h))
        true_u = pred_u + np.random.normal(0, 0.02, (n, h))
        true_v = pred_v + np.random.normal(0, 0.02, (n, h))

        metadata = [{'init_lat': -65, 'init_lon': 0, 'prev_lat': -65.1, 'prev_lon': 0.1}] * n

        metrics = compute_all_metrics(
            {'u': pred_u, 'v': pred_v},
            {'u': true_u, 'v': true_v},
            metadata,
            horizon_hours=[6, 12, 24],
        )

        assert metrics.rmse_position_km > 0
        assert metrics.mean_position_error_km > 0
        assert metrics.mean_direction_error_deg >= 0
        assert -1 <= metrics.skill_score_vs_persistence <= 1

    def test_bootstrap_confidence_interval(self):
        preds = np.random.normal(0, 1, 100)
        targets = np.random.normal(0, 1, 100)

        point, lower, upper = bootstrap_confidence_interval(
            lambda p, t: np.mean((p - t)**2),  # MSE
            preds, targets,
            n_bootstrap=100,
            confidence=0.95,
        )

        assert lower <= point <= upper

    def test_compare_models_bootstrap(self):
        preds1 = np.random.normal(0, 1, 100)
        preds2 = np.random.normal(0.5, 1, 100)  # Worse model
        targets = np.random.normal(0, 1, 100)

        result = compare_models_bootstrap(
            preds1, preds2, targets,
            lambda p, t: np.sqrt(np.mean((p - t)**2)),  # RMSE
            n_bootstrap=100,
        )

        assert 'mean_difference' in result
        assert 'p_value' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result

    def test_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        metrics = regression_metrics(y_pred, y_true)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['r2'] > 0.9  # Good fit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])