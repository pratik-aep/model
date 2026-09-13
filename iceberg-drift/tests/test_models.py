"""Tests for iceberg drift models."""

import pytest
import torch
import numpy as np

from iceberg_drift.models import (
    PhysicsDriftModel,
    IcebergParameters,
    LSTMDriftModel,
    GRUDriftModel,
    MLPDriftModel,
    TransformerDriftModel,
    PINNDriftModel,
    HybridDriftModel,
    create_model,
    # GBM models
    XGBoostRegressor,
    LightGBMRegressor,
    MultiTargetGBM,
    PhysicsInformedGBM,
    EnsembleGBM,
    GBMConfig,
    create_gbm_config,
)


class TestPhysicsModel:
    """Test physics-based drift model."""

    def test_coriolis_parameter(self):
        from iceberg_drift.models.physics_model import compute_coriolis_parameter
        # At equator
        assert abs(compute_coriolis_parameter(0)) < 1e-10
        # At poles
        f_pole = compute_coriolis_parameter(90)
        assert abs(f_pole - 2 * 7.2921150e-5) < 1e-10
        # Southern hemisphere (negative)
        f_south = compute_coriolis_parameter(-65)
        assert f_south < 0

    def test_wind_drag_factor(self):
        from iceberg_drift.models.physics_model import compute_wind_drag_factor
        iceberg = IcebergParameters(length_m=2000, width_m=1000, height_m=300, draft_m=270)
        factor = compute_wind_drag_factor(iceberg)
        # Should be around 1-3%
        assert 0.005 <= factor < 0.05

    def test_physics_drift(self):
        from iceberg_drift.models.physics_model import compute_physics_drift, EnvironmentalForcing
        iceberg = IcebergParameters()
        forcing = EnvironmentalForcing(
            current_u=0.1, current_v=0.05,
            wind_u=10.0, wind_v=5.0,
            latitude=-65.0,
        )
        u, v = compute_physics_drift(iceberg, forcing)
        # Drift should be dominated by current + small wind component
        assert abs(u - 0.1) < 0.5  # Wind adds ~0.2 m/s
        assert abs(v - 0.05) < 0.5

    def test_trajectory_integration(self):
        model = PhysicsDriftModel()
        n_steps = 10
        current_u = np.full(n_steps, 0.1)
        current_v = np.full(n_steps, 0.05)
        wind_u = np.full(n_steps, 10.0)
        wind_v = np.full(n_steps, 5.0)

        lats, lons = model.predict_trajectory(
            init_lat=-65.0, init_lon=0.0,
            current_u_series=current_u,
            current_v_series=current_v,
            wind_u_series=wind_u,
            wind_v_series=wind_v,
            dt=3600.0,
        )
        assert len(lats) == n_steps + 1
        assert len(lons) == n_steps + 1
        # Should drift northeast-ish (current + wind)
        assert lats[-1] > -65.0  # Moved north
        assert lons[-1] > 0.0    # Moved east


class TestMLModels:
    """Test ML model architectures."""

    @pytest.fixture
    def sample_input(self):
        batch_size = 4
        seq_len = 24
        input_dim = 50
        return torch.randn(batch_size, seq_len, input_dim)

    @pytest.fixture
    def sample_targets(self):
        batch_size = 4
        horizon = 12
        output_dim = 2
        return torch.randn(batch_size, horizon, output_dim)

    def test_lstm_model(self, sample_input, sample_targets):
        model = LSTMDriftModel(input_dim=50, output_dim=2, prediction_horizon=12)
        # Without teacher forcing
        out = model(sample_input)
        assert out.shape == (4, 12, 2)
        # With teacher forcing
        out = model(sample_input, targets=sample_targets, teacher_forcing_ratio=1.0)
        assert out.shape == (4, 12, 2)

    def test_gru_model(self, sample_input, sample_targets):
        model = GRUDriftModel(input_dim=50, output_dim=2, prediction_horizon=12)
        out = model(sample_input)
        assert out.shape == (4, 12, 2)

    def test_mlp_model(self, sample_input):
        model = MLPDriftModel(input_dim=50, output_dim=2, prediction_horizon=12)
        out = model(sample_input)
        assert out.shape == (4, 12, 2)

    def test_transformer_model(self, sample_input, sample_targets):
        model = TransformerDriftModel(input_dim=50, output_dim=2, prediction_horizon=12, max_seq_len=50)
        out = model(sample_input)
        assert out.shape == (4, 12, 2)
        # With teacher forcing
        out = model(sample_input, targets=sample_targets, teacher_forcing_ratio=1.0)
        assert out.shape == (4, 12, 2)


class TestPINNModels:
    """Test physics-informed models."""

    @pytest.fixture
    def sample_input(self):
        batch_size = 2
        seq_len = 24
        input_dim = 50
        # Ensure last 4 features are current_u, current_v, wind_u, wind_v
        x = torch.randn(batch_size, seq_len, input_dim)
        x[:, -1, -4] = 0.1   # current_u
        x[:, -1, -3] = 0.05  # current_v
        x[:, -1, -2] = 10.0  # wind_u
        x[:, -1, -1] = 5.0   # wind_v
        return x

    @pytest.fixture
    def sample_targets(self):
        return torch.randn(2, 12, 2)

    @pytest.fixture
    def sample_metadata(self):
        return [{'init_lat': -65.0, 'init_lon': 0.0}] * 2

    def test_pinn_model(self, sample_input, sample_targets, sample_metadata):
        model = PINNDriftModel(input_dim=50, output_dim=2, prediction_horizon=12)
        out = model(sample_input, metadata=sample_metadata)
        assert 'predictions' in out
        assert 'physics_pred' in out
        assert 'ml_correction' in out
        assert out['predictions'].shape == (2, 12, 2)

        # Test loss computation
        loss_dict = model.compute_loss(out, sample_targets, sample_input, sample_metadata)
        assert 'total_loss' in loss_dict
        assert 'data_loss' in loss_dict
        assert 'physics_loss' in loss_dict

    def test_hybrid_model(self, sample_input, sample_targets, sample_metadata):
        model = HybridDriftModel(input_dim=50, output_dim=2, prediction_horizon=12)
        out = model(sample_input, metadata=sample_metadata)
        assert 'predictions' in out
        assert 'wind_factor' in out
        assert 'coriolis_alpha' in out
        # Parameters should be in reasonable ranges
        assert 0.001 < out['wind_factor'].item() < 0.1
        assert 0.001 < out['coriolis_alpha'].item() < 0.05


class TestModelFactory:
    """Test model factory function."""

    def test_create_lstm(self):
        model = create_model("lstm", input_dim=50, output_dim=2, prediction_horizon=12)
        assert isinstance(model, LSTMDriftModel)

    def test_create_gru(self):
        model = create_model("gru", input_dim=50, output_dim=2, prediction_horizon=12)
        assert isinstance(model, GRUDriftModel)

    def test_create_transformer(self):
        model = create_model("transformer", input_dim=50, output_dim=2, prediction_horizon=12)
        assert isinstance(model, TransformerDriftModel)

    def test_create_mlp(self):
        model = create_model("mlp", input_dim=50, output_dim=2, prediction_horizon=12)
        assert isinstance(model, MLPDriftModel)

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            create_model("invalid", input_dim=50, output_dim=2)


class TestEnsemble:
    """Test ensemble model."""

    def test_mean_ensemble(self):
        from iceberg_drift.models import EnsembleDriftModel, LSTMDriftModel
        models = [
            LSTMDriftModel(input_dim=10, output_dim=2, prediction_horizon=6)
            for _ in range(3)
        ]
        ensemble = EnsembleDriftModel(models, ensemble_method="mean")
        x = torch.randn(2, 10, 10)
        out = ensemble(x)
        assert 'predictions' in out
        assert 'uncertainty' in out
        assert out['predictions'].shape == (2, 6, 2)

    def test_weighted_ensemble(self):
        from iceberg_drift.models import EnsembleDriftModel, LSTMDriftModel
        models = [
            LSTMDriftModel(input_dim=10, output_dim=2, prediction_horizon=6)
            for _ in range(3)
        ]
        ensemble = EnsembleDriftModel(models, ensemble_method="weighted")
        x = torch.randn(2, 10, 10)
        out = ensemble(x)
        assert out['predictions'].shape == (2, 6, 2)


@pytest.mark.skip(reason="GBM tests segfault on macOS in this environment")
class TestGBMModels:
    """Test Gradient Boosted Machine models."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randn(n_samples, 2).astype(np.float32)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        target_names = ["target_u", "target_v"]
        return X, y, feature_names, target_names

    @pytest.mark.skip(reason="XGBoost segfaults on macOS with early stopping/callbacks in this environment")
    def test_xgboost_regressor(self, sample_data):
        X, y, feature_names, target_names = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        config = GBMConfig(model_type="xgboost")
        config.xgb_params.update({
            "n_estimators": 50,
            "max_depth": 3,
            "early_stopping_rounds": 10,
        })

        model = XGBoostRegressor(config=config, target_name="target_u")
        model.fit(X_train, y_train[:, 0], X_val, y_val[:, 0], feature_names=feature_names)

        assert model._is_fitted
        pred = model.predict(X_val)
        assert pred.shape == (20,)

        # Check feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(feature_names)

    @pytest.mark.skipif(
        not pytest.importorskip("lightgbm", reason="LightGBM not installed"),
        reason="LightGBM not available"
    )
    def test_lightgbm_regressor(self, sample_data):
        X, y, feature_names, target_names = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        config = GBMConfig(model_type="lightgbm")
        config.lgb_params.update({
            "n_estimators": 50,
            "max_depth": 3,
            "early_stopping_rounds": 10,
        })

        model = LightGBMRegressor(config=config, target_name="target_u")
        model.fit(X_train, y_train[:, 0], X_val, y_val[:, 0], feature_names=feature_names)

        assert model._is_fitted
        pred = model.predict(X_val)
        assert pred.shape == (20,)

    def test_multitarget_gbm(self, sample_data):
        X, y, feature_names, target_names = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        config = GBMConfig(model_type="xgboost")
        config.xgb_params.update({
            "n_estimators": 50,
            "max_depth": 3,
            "early_stopping_rounds": 10,
        })

        model = MultiTargetGBM(config=config, target_names=target_names)
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        assert len(model.models) == 2
        pred = model.predict(X_val)
        assert pred.shape == (20, 2)

        # Test feature importance
        importance = model.get_feature_importance_summary()
        assert "feature" in importance.columns
        assert "mean_importance" in importance.columns

    def test_physics_informed_gbm(self, sample_data):
        X, y, feature_names, target_names = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        config = GBMConfig(model_type="xgboost")
        config.xgb_params.update({
            "n_estimators": 50,
            "max_depth": 3,
            "early_stopping_rounds": 10,
        })

        physics_model = PhysicsDriftModel()
        model = PhysicsInformedGBM(
            gbm_config=config,
            physics_model=physics_model,
            target_names=target_names,
        )
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        pred = model.predict(X_val)
        assert pred.shape == (20, 2)

    def test_ensemble_gbm(self, sample_data):
        X, y, feature_names, target_names = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        config = GBMConfig(model_type="xgboost")
        config.xgb_params.update({
            "n_estimators": 30,
            "max_depth": 3,
            "early_stopping_rounds": 10,
        })

        ensemble = EnsembleGBM(
            n_models=3,
            model_type="xgboost",
            config=config,
            target_names=target_names,
            aggregation="mean",
        )
        ensemble.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        pred = ensemble.predict(X_val)
        assert pred.shape == (20, 2)

        # Test with uncertainty
        pred, unc = ensemble.predict_with_uncertainty(X_val)
        assert pred.shape == (20, 2)
        assert unc.shape == (20, 2)

    def test_gbm_save_load(self, sample_data, tmp_path):
        X, y, feature_names, target_names = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        config = GBMConfig(model_type="xgboost")
        config.xgb_params.update({
            "n_estimators": 30,
            "max_depth": 3,
            "early_stopping_rounds": 10,
        })

        model = MultiTargetGBM(config=config, target_names=target_names)
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        # Save
        save_path = tmp_path / "gbm_test_model"
        model.save(str(save_path))

        # Load
        loaded = MultiTargetGBM.load(str(save_path))
        pred_orig = model.predict(X_val)
        pred_loaded = loaded.predict(X_val)

        np.testing.assert_array_almost_equal(pred_orig, pred_loaded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])