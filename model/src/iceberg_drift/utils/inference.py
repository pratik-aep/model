"""Inference utilities for iceberg drift prediction."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..models.pinn import PINNDriftModel, HybridDriftModel
from ..models.physics_model import (
    IcebergParameters,
    EnvironmentalForcing,
    compute_physics_drift_batch,
    integrate_trajectory_batch,
)
from ..data_processing.preprocessing import engineer_features, compute_coriolis
from ..config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for drift prediction inference."""
    # Model
    model_path: str
    model_type: str = "auto"  # "auto", "pinn", "hybrid", "lstm", "gru", "transformer", "mlp"

    # Prediction
    prediction_horizon_hours: int = 72
    time_step_hours: int = 6
    sequence_length_hours: int = 24

    # Iceberg parameters (used if not in input data)
    default_length_m: float = 2000.0
    default_width_m: float = 1000.0
    default_draft_m: float = 270.0

    # Physics parameters
    wind_factor: float = 0.02
    coriolis_alpha: float = 0.015

    # Output
    output_format: str = "dict"  # "dict", "dataframe", "geojson", "netcdf"
    include_uncertainty: bool = True
    n_ensemble_samples: int = 50

    # Device
    device: str = "auto"


# =============================================================================
# Drift Predictor
# =============================================================================

class DriftPredictor:
    """
    High-level interface for iceberg drift prediction.

    Usage:
        predictor = DriftPredictor("path/to/model.pt")
        trajectory = predictor.predict(
            init_lat=-65.0, init_lon=0.0,
            current_u=..., current_v=..., wind_u=..., wind_v=...
        )
    """

    def __init__(self, config: InferenceConfig):
        self.config = config

        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Iceberg parameters
        self.iceberg_params = IcebergParameters(
            length_m=config.default_length_m,
            width_m=config.default_width_m,
            draft_m=config.default_draft_m,
        )

        logger.info(f"DriftPredictor initialized on {self.device}")
        logger.info(f"Prediction horizon: {config.prediction_horizon_hours}h")

    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=False)

        # Determine model type
        if self.config.model_type == "auto":
            # Try to infer from checkpoint state_dict
            if "model_state_dict" in checkpoint:
                keys = list(checkpoint["model_state_dict"].keys())
                if any(k.startswith("ml_model.") for k in keys):
                    model_type = "pinn"
                elif any("physics" in k for k in keys):
                    model_type = "pinn"
                else:
                    model_type = "lstm"
            else:
                model_type = "lstm"
        else:
            model_type = self.config.model_type

        # Determine dimensions from state dict
        input_dim = 50
        hidden_dim = 128
        if "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
            weight_key = "ml_model.encoder.weight_ih_l0" if "ml_model.encoder.weight_ih_l0" in sd else "encoder.weight_ih_l0"
            if weight_key in sd:
                shape = sd[weight_key].shape
                hidden_dim = shape[0] // 4
                input_dim = shape[1]

        # Create model architecture
        if model_type == "pinn":
            from ..models.pinn import PINNDriftModel
            model = PINNDriftModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=2,
                prediction_horizon=self.config.prediction_horizon_hours // self.config.time_step_hours,
            )
        elif model_type == "hybrid":
            from ..models.pinn import HybridDriftModel
            model = HybridDriftModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=2,
                prediction_horizon=self.config.prediction_horizon_hours // self.config.time_step_hours,
            )
        else:
            from ..models.ml_models import create_model
            model = create_model(
                model_type,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=2,
                prediction_horizon=self.config.prediction_horizon_hours // self.config.time_step_hours,
            )

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        logger.info(f"Loaded {model_type} model from {self.config.model_path}")
        return model

    def prepare_input_sequence(
        self,
        timestamps: pd.DatetimeIndex,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        current_u: np.ndarray,
        current_v: np.ndarray,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        iceberg_length: Optional[float] = None,
        iceberg_width: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Prepare input sequence for model prediction.

        Args:
            timestamps: Array of timestamps (length = sequence_length)
            latitudes: Historical latitudes
            longitudes: Historical longitudes
            current_u, current_v: Ocean current velocities (m/s)
            wind_u, wind_v: Wind velocities (m/s)
            iceberg_length, iceberg_width: Iceberg dimensions (optional)

        Returns:
            (input_tensor, metadata_list)
        """
        n_steps = len(timestamps)
        seq_len = self.config.sequence_length_hours // self.config.time_step_hours

        if n_steps < seq_len:
            raise ValueError(f"Need at least {seq_len} historical steps, got {n_steps}")

        # Use last seq_len steps
        start_idx = n_steps - seq_len

        # Build feature matrix (matching training features)
        features = []

        for i in range(start_idx, n_steps):
            lat = latitudes[i]
            lon = longitudes[i]
            t = timestamps[i]

            # Base features
            feat = {
                'iceberg_id': 'test_iceberg',
                'lat': lat,
                'lon': lon,
                'current_uo': current_u[i],
                'current_vo': current_v[i],
                'wind_u10': wind_u[i],
                'wind_v10': wind_v[i],
                'datetime': t,
            }

            # Iceberg size
            feat['length_m'] = iceberg_length or self.config.default_length_m
            feat['width_m'] = iceberg_width or self.config.default_width_m

            features.append(feat)

        # Convert to DataFrame for feature engineering
        df = pd.DataFrame(features)

        # Apply feature engineering
        df = engineer_features(df, add_cyclical_time=True, add_physics_features=True, add_lag_features=True)

        # Get feature columns (exclude metadata)
        feature_cols = [c for c in df.columns if c not in
                       ['datetime', 'lat', 'lon', 'length_m', 'width_m', 'iceberg_area', 'iceberg_aspect_ratio', 'iceberg_draft_estimate']]

        # Ensure all expected features exist (fill missing with 0)
        from iceberg_drift.data_processing.preprocessing import get_feature_columns
        expected_features = get_feature_columns(df)
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0.0

        # Select and order features
        feature_matrix = df[expected_features].values.astype(np.float32)

        # Create tensor
        x = torch.tensor(feature_matrix[None, :, :], device=self.device)  # (1, seq_len, n_features)

        # Metadata
        metadata = [{
            'init_lat': float(latitudes[-1]),
            'init_lon': float(longitudes[-1]),
            'init_time': timestamps[-1],
            'length_m': float(iceberg_length or self.config.default_length_m),
            'width_m': float(iceberg_width or self.config.default_width_m),
        }]

        return x, metadata

    def _get_expected_features(self) -> List[str]:
        """Get expected feature columns (should match training)."""
        # This should ideally be saved with the model
        # For now, return a comprehensive list
        base_features = [
            'lat', 'lon',
            'current_uo', 'current_vo', 'current_speed', 'current_dir',
            'wind_u10', 'wind_v10', 'wind_speed', 'wind_dir',
            'wind_current_alignment', 'wind_current_cross', 'wind_current_rel_dir',
            'coriolis_f', 'physics_u', 'physics_v', 'physics_speed', 'physics_dir',
            'iceberg_area', 'iceberg_aspect_ratio', 'iceberg_draft_estimate',
            'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
        ]

        # Add lag features
        lag_features = []
        for lag_h in [6, 12, 24, 48]:
            for base in ['lat', 'lon', 'wind_u10', 'wind_v10', 'wind_speed', 'wind_dir',
                        'current_uo', 'current_vo', 'current_speed', 'current_dir',
                        'physics_u', 'physics_v', 'physics_speed', 'physics_dir']:
                lag_features.append(f'{base}_lag{lag_h}h')

        return base_features + lag_features

    def predict(
        self,
        timestamps: pd.DatetimeIndex,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        current_u: np.ndarray,
        current_v: np.ndarray,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        iceberg_length: Optional[float] = None,
        iceberg_width: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict iceberg drift trajectory.

        Args:
            timestamps: Historical timestamps (at least sequence_length hours)
            latitudes: Historical latitudes
            longitudes: Historical longitudes
            current_u, current_v: Ocean current velocities (m/s)
            wind_u, wind_v: Wind velocities (m/s)
            iceberg_length, iceberg_width: Iceberg dimensions (optional)

        Returns:
            Dict with predicted trajectory and metadata
        """
        # Prepare input
        x, metadata = self.prepare_input_sequence(
            timestamps, latitudes, longitudes,
            current_u, current_v, wind_u, wind_v,
            iceberg_length, iceberg_width,
        )

        # Predict
        with torch.no_grad():
            if isinstance(self.model, (PINNDriftModel, HybridDriftModel)):
                output = self.model(x, targets=None, teacher_forcing_ratio=0.0, metadata=metadata)
                pred_velocities = output["predictions"].cpu().numpy()[0]  # (horizon, 2)
                physics_pred = output.get("physics_pred", None)
                if physics_pred is not None:
                    physics_pred = physics_pred.cpu().numpy()[0]
                ml_correction = output.get("ml_correction", None)
                if ml_correction is not None:
                    ml_correction = ml_correction.cpu().numpy()[0]
            else:
                pred_velocities = self.model(x, targets=None, teacher_forcing_ratio=0.0).cpu().numpy()[0]
                physics_pred = None
                ml_correction = None

        # Extract predicted u, v
        pred_u = pred_velocities[:, 0]
        pred_v = pred_velocities[:, 1]

        # Integrate to get trajectory
        init_lat = metadata[0]['init_lat']
        init_lon = metadata[0]['init_lon']

        pred_lats, pred_lons = integrate_trajectory_batch(
            init_lats=np.array([init_lat]),
            init_lons=np.array([init_lon]),
            u_drift=pred_u[None, :],
            v_drift=pred_v[None, :],
            dt=self.config.time_step_hours * 3600,
        )

        pred_lats = pred_lats[0]
        pred_lons = pred_lons[0]

        # Build result
        n_steps = len(pred_lats) - 1  # Exclude initial position
        pred_times = pd.date_range(
            timestamps[-1],
            periods=n_steps + 1,
            freq=f"{self.config.time_step_hours}h",
        )

        result = {
            "timestamps": pred_times,
            "latitudes": pred_lats,
            "longitudes": pred_lons,
            "velocities_u": pred_u,
            "velocities_v": pred_v,
            "init_lat": init_lat,
            "init_lon": init_lon,
            "init_time": timestamps[-1],
            "prediction_horizon_hours": self.config.prediction_horizon_hours,
        }

        if physics_pred is not None:
            phys_lats, phys_lons = integrate_trajectory_batch(
                init_lats=np.array([init_lat]),
                init_lons=np.array([init_lon]),
                u_drift=physics_pred[:, 0][None, :],
                v_drift=physics_pred[:, 1][None, :],
                dt=self.config.time_step_hours * 3600,
            )
            result["physics_latitudes"] = phys_lats[0]
            result["physics_longitudes"] = phys_lons[0]
            result["physics_velocities_u"] = physics_pred[:, 0]
            result["physics_velocities_v"] = physics_pred[:, 1]

        if ml_correction is not None:
            result["ml_correction_u"] = ml_correction[:, 0]
            result["ml_correction_v"] = ml_correction[:, 1]

        # Uncertainty estimation (if enabled)
        if self.config.include_uncertainty and hasattr(self.model, 'ml_model'):
            uncertainty = self._estimate_uncertainty(x, metadata)
            result["uncertainty_km"] = uncertainty

        return result

    def _estimate_uncertainty(
        self,
        x: torch.Tensor,
        metadata: List[Dict],
        n_samples: int = None,
    ) -> np.ndarray:
        """Estimate prediction uncertainty using Monte Carlo dropout."""
        n_samples = n_samples or self.config.n_ensemble_samples

        self.model.train()  # Enable dropout
        samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                if isinstance(self.model, (PINNDriftModel, HybridDriftModel)):
                    output = self.model(x, targets=None, teacher_forcing_ratio=0.0, metadata=metadata)
                    pred = output["predictions"].cpu().numpy()[0]
                else:
                    pred = self.model(x, targets=None, teacher_forcing_ratio=0.0).cpu().numpy()[0]

                # Convert to position error (approximate)
                pred_u, pred_v = pred[:, 0], pred[:, 1]
                lats, lons = integrate_trajectory_batch(
                    init_lats=np.array([metadata[0]['init_lat']]),
                    init_lons=np.array([metadata[0]['init_lon']]),
                    u_drift=pred_u[None, :],
                    v_drift=pred_v[None, :],
                    dt=self.config.time_step_hours * 3600,
                )
                samples.append(np.stack([lats[0], lons[0]], axis=1))

        samples = np.stack(samples, axis=0)  # (n_samples, n_steps+1, 2)

        # Compute position uncertainty (std dev in km)
        mean_pos = samples.mean(axis=0)
        std_pos = samples.std(axis=0)

        # Convert to km
        lat_km = std_pos[:, 0] * 111.0
        lon_km = std_pos[:, 1] * 111.0 * np.cos(np.radians(mean_pos[:, 0]))

        uncertainty_km = np.sqrt(lat_km**2 + lon_km**2)

        return uncertainty_km


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model_for_inference(
    model_path: str,
    prediction_horizon_hours: int = 72,
    device: str = "auto",
    **kwargs,
) -> DriftPredictor:
    """Load model and create predictor."""
    config = InferenceConfig(
        model_path=model_path,
        prediction_horizon_hours=prediction_horizon_hours,
        device=device,
        **kwargs,
    )
    return DriftPredictor(config)


def predict_trajectory(
    model_path: str,
    init_lat: float,
    init_lon: float,
    current_forecast: Dict[str, np.ndarray],
    wind_forecast: Dict[str, np.ndarray],
    prediction_horizon_hours: int = 72,
    iceberg_length: float = 2000.0,
    iceberg_width: float = 1000.0,
) -> Dict[str, Any]:
    """
    Simple one-call trajectory prediction.

    Args:
        model_path: Path to model checkpoint
        init_lat, init_lon: Initial position
        current_forecast: Dict with 'u', 'v', 'time' arrays
        wind_forecast: Dict with 'u', 'v', 'time' arrays
        prediction_horizon_hours: How far to predict
        iceberg_length, iceberg_width: Iceberg dimensions

    Returns:
        Predicted trajectory
    """
    predictor = load_model_for_inference(model_path, prediction_horizon_hours)

    # Build historical sequence (use forecast as history for simplicity)
    # In practice, you'd have actual historical data
    n_hist = 24 // 6  # 24 hours history at 6-hourly
    timestamps = current_forecast['time'][-n_hist:]
    lats = np.full(n_hist, init_lat)
    lons = np.full(n_hist, init_lon)

    return predictor.predict(
        timestamps=timestamps,
        latitudes=lats,
        longitudes=lons,
        current_u=current_forecast['u'][-n_hist:],
        current_v=current_forecast['v'][-n_hist:],
        wind_u=wind_forecast['u'][-n_hist:],
        wind_v=wind_forecast['v'][-n_hist:],
        iceberg_length=iceberg_length,
        iceberg_width=iceberg_width,
    )