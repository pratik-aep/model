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
        # Distance error should be ~9.5km (difference between diagonal distance and straight north)
        assert abs(dist_err[0] - 9.5) < 2
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
        if not np.isnan(metrics.skill_score_vs_persistence):
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

    def test_compute_skill_score_velocity_mode(self):
        """Test BUG-013: skill scores in velocity mode are real numbers, not NaN."""
        from iceberg_drift.evaluation.metrics import compute_all_metrics
        import numpy as np
        
        n = 10
        h = 6
        pred_u = np.random.normal(0.1, 0.05, (n, h))
        pred_v = np.random.normal(0.05, 0.05, (n, h))
        true_u = pred_u + np.random.normal(0, 0.02, (n, h))
        true_v = pred_v + np.random.normal(0, 0.02, (n, h))
        metadata = [{'init_lat': -65, 'init_lon': 0, 'current_uo': 0, 'current_vo': 0, 'wind_u10': 0, 'wind_v10': 0}] * n

        metrics = compute_all_metrics(
            {'u': pred_u, 'v': pred_v},
            {'u': true_u, 'v': true_v},
            metadata,
            horizon_hours=[6],
        )

        assert not np.isnan(metrics.skill_score_vs_persistence)
        assert not np.isnan(metrics.skill_score_vs_physics)


def test_predict_output_arrays_same_length():
    """Test that predict() returns arrays with matching lengths (BUG-012 fix)."""
    # Create a simple mock model and predictor to test array length consistency
    import torch
    import torch.nn as nn
    from iceberg_drift.utils.inference import DriftPredictor, InferenceConfig

    # Create a minimal mock model that returns predictable outputs
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)  # 10 input features, 2 outputs (u,v)

        def forward(self, x, targets=None, teacher_forcing_ratio=0.0, metadata=None):
            # Return fixed predictions: batch_size=1, seq_len=5, output_dim=2
            batch_size = x.shape[0]
            seq_len = 5  # Fixed horizon
            # Predict constant velocity: 0.1 m/s east, 0.05 m/s north
            predictions = torch.full((batch_size, seq_len, 2), 0.1)  # u component
            predictions[:, :, 1] = 0.05  # v component
            return predictions

    from unittest.mock import patch

    # Create mock predictor with our mock model
    config = InferenceConfig(
        model_path="dummy_path.pt",  # Won't be used since we're mocking _load_model
        model_type="lstm",
        prediction_horizon_hours=30,  # 5 timesteps * 6 hours = 30 hours
        time_step_hours=6,
        sequence_length_hours=24,
    )
    
    with patch('iceberg_drift.utils.inference.DriftPredictor._load_model') as mock_load_model:
        mock_load_model.return_value = MockModel()
        predictor = DriftPredictor(config)
        predictor.feature_cols = None
        predictor.scaler = None
        predictor.feature_idx_map = None

    # Create dummy input data matching expected format
    n_hist = 4  # 4 timesteps * 6 hours = 24 hours history
    timestamps = pd.date_range('2015-01-01', periods=n_hist, freq='6h')
    latitudes = np.full(n_hist, -65.0)
    longitudes = np.full(n_hist, 0.0)
    current_u = np.full(n_hist, 0.1)
    current_v = np.full(n_hist, 0.05)
    wind_u = np.full(n_hist, 5.0)
    wind_v = np.full(n_hist, 3.0)

    # Run prediction
    result = predictor.predict(
        timestamps=timestamps,
        latitudes=latitudes,
        longitudes=longitudes,
        current_u=current_u,
        current_v=current_v,
        wind_u=wind_u,
        wind_v=wind_v,
    )

    # Check that all array-valued entries in result have the same length
    lengths = []
    for key, value in result.items():
        if hasattr(value, '__len__') and not isinstance(value, str):
            try:
                lengths.append(len(value))
            except TypeError:
                # Skip items that don't have a sensible length
                pass

    # All lengths should be equal (5 predictions for 5 timesteps horizon)
    assert len(set(lengths)) == 1, f"Mismatched array lengths: {dict(zip([k for k in result.keys() if hasattr(result[k], '__len__') and not isinstance(result[k], str)], lengths))}"
    # Should be 5 (prediction horizon in timesteps: 30 hours / 6 hours per timestep = 5)
    assert lengths[0] == 5, f"Expected length 5, got {lengths[0]}"


@pytest.mark.parametrize("checkpoint_format", ["nested", "top_level"])
def test_predictor_loads_feature_metadata(checkpoint_format):
    """Test that DriftPredictor correctly loads feature_idx_map and scaler from checkpoint (BUG-011 fix)."""
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from iceberg_drift.utils.inference import load_model_for_inference
    from iceberg_drift.models.pinn import PINNDriftModel as PINNDriftModelClass

    # Create a temporary directory for our test checkpoint
    import tempfile
    import os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "test_model.pt"

        # Create a simple PINN model with known architecture
        input_dim = 5
        hidden_dim = 16
        model = PINNDriftModelClass(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            prediction_horizon=4,
            feature_idx_map={'feature_0': 0, 'feature_1': 1, 'feature_2': 2, 'feature_3': 3, 'feature_4': 4}
        )

        # Create a scaler and fit it to some dummy data
        scaler = StandardScaler()
        dummy_data = np.random.randn(100, input_dim)
        scaler.fit(dummy_data)

        # Create feature columns list
        feature_cols = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']

        # Save checkpoint in either nested or top-level format
        if checkpoint_format == "nested":
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'feature_idx_map': {'feature_0': 0, 'feature_1': 1, 'feature_2': 2, 'feature_3': 3, 'feature_4': 4},
                    'scaler': scaler,
                    'feature_cols': feature_cols
                }
            }
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'feature_idx_map': {'feature_0': 0, 'feature_1': 1, 'feature_2': 2, 'feature_3': 3, 'feature_4': 4},
                'scaler': scaler,
                'feature_cols': feature_cols
            }
        
        torch.save(checkpoint, checkpoint_path)

        # Load the model using our inference utility
        predictor = load_model_for_inference(str(checkpoint_path))

        # Assert that the feature metadata was loaded correctly
        assert predictor.feature_idx_map is not None, "feature_idx_map should not be None after loading checkpoint"
        assert predictor.scaler is not None, "scaler should not be None after loading checkpoint"
        assert predictor.feature_cols is not None, "feature_cols should not be None after loading checkpoint"

        # Check that the loaded values match what we saved
        assert predictor.feature_idx_map == {'feature_0': 0, 'feature_1': 1, 'feature_2': 2, 'feature_3': 3, 'feature_4': 4}
        assert predictor.feature_cols == ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']

        # Verify the scaler is functional by testing transform
        test_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        scaled_data = predictor.scaler.transform(test_data)
        assert scaled_data.shape == (1, input_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])