"""Physics-Informed Neural Network (PINN) for iceberg drift prediction.

Combines physics-based drift model with ML correction:
- Physics model provides baseline drift (current + wind + Coriolis)
- ML model learns residual/correction from data
- Physics-informed loss ensures predictions obey physical constraints
"""

import logging
from typing import Tuple, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .physics_model import (
    IcebergParameters,
    EnvironmentalForcing,
    compute_physics_drift,
    compute_physics_drift_batch,
    integrate_trajectory_batch,
    EARTH_ROTATION_RATE,
)
from .ml_models import BaseDriftModel, create_model

logger = logging.getLogger(__name__)


# =============================================================================
# Physics-Informed Loss Functions
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Composite loss combining data fidelity and physics constraints.

    Loss = w_data * L_data + w_physics * L_physics + w_boundary * L_boundary
    """

    def __init__(
        self,
        data_weight: float = 1.0,
        physics_weight: float = 0.5,
        boundary_weight: float = 0.1,
        wind_factor: float = 0.02,
        coriolis_alpha: float = 0.015,
        feature_idx_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.boundary_weight = boundary_weight
        self.wind_factor = wind_factor
        self.coriolis_alpha = coriolis_alpha
        self.feature_idx_map = feature_idx_map

        # Data loss (MSE on velocity/position)
        self.data_loss_fn = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,      # (batch, horizon, 2) - predicted u,v
        targets: torch.Tensor,          # (batch, horizon, 2) - true u,v
        inputs: torch.Tensor,           # (batch, seq_len, input_dim) - features
        metadata: List[Dict],           # List of dicts with lat, lon, etc.
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.

        Args:
            predictions: Model predictions (u, v) for each horizon step
            targets: Ground truth (u, v)
            inputs: Input features containing current_u, current_v, wind_u, wind_v
            metadata: List of dicts with 'init_lat', 'init_lon', etc.

        Returns:
            Dict with total_loss and component losses
        """
        batch_size, horizon, _ = predictions.shape
        device = predictions.device

        # --- Data Loss ---
        data_loss = self.data_loss_fn(predictions, targets)

        # --- Physics Loss ---
        physics_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # 1. Physics loss: Does the model respect the basic physics of iceberg drift?
        if self.physics_weight > 0:
            physics_loss = self._compute_physics_loss(
                predictions, inputs, metadata, device
            )

        boundary_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if self.boundary_weight > 0:
            boundary_loss = self._compute_boundary_loss(
                predictions, metadata, device
            )

        # Total loss
        total_loss = (
            self.data_weight * data_loss +
            self.physics_weight * physics_loss +
            self.boundary_weight * boundary_loss
        )

        return {
            "total_loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "boundary_loss": boundary_loss,
        }

    def _compute_physics_loss(
        self,
        predictions: torch.Tensor,
        inputs: torch.Tensor,
        metadata: List[Dict],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Physics loss: predictions should be close to physics-based drift.

        Physics drift = current + wind_factor * wind + Coriolis_correction
        """
        batch_size, horizon, _ = predictions.shape

        # Extract environmental forcings from inputs
        # Use feature indices if provided, else fallback to last 4 columns
        if self.feature_idx_map is not None:
            current_u_idx = self.feature_idx_map['current_uo']
            current_v_idx = self.feature_idx_map['current_vo']
            wind_u_idx = self.feature_idx_map['wind_u10']
            wind_v_idx = self.feature_idx_map['wind_v10']
            current_u = inputs[:, -1, current_u_idx]
            current_v = inputs[:, -1, current_v_idx]
            wind_u = inputs[:, -1, wind_u_idx]
            wind_v = inputs[:, -1, wind_v_idx]
        else:
            # Fallback to original indexing (last 4 columns)
            current_u = inputs[:, -1, -4]
            current_v = inputs[:, -1, -3]
            wind_u = inputs[:, -1, -2]
            wind_v = inputs[:, -1, -1]

        # Get latitudes from metadata
        lats = torch.tensor([m.get('init_lat', -65.0) for m in metadata], device=device, dtype=torch.float32)

        # Compute physics-based drift
        # Use the loss function's wind factor and coriolis alpha
        wind_factor = self.wind_factor
        coriolis_alpha = self.coriolis_alpha

        # Vectorized physics drift
        hem_sign = torch.where(lats < 0, -1.0, 1.0)

        physics_u = (current_u +
                     wind_factor * wind_u +
                     hem_sign * coriolis_alpha * wind_v)

        physics_v = (current_v +
                     wind_factor * wind_v -
                     hem_sign * coriolis_alpha * wind_u)

        # Expand to horizon
        physics_u = physics_u.unsqueeze(1).expand(-1, horizon)
        physics_v = physics_v.unsqueeze(1).expand(-1, horizon)

        # Physics loss: MSE between predictions and physics baseline
        physics_loss = F.mse_loss(predictions[:, :, 0], physics_u) + \
                       F.mse_loss(predictions[:, :, 1], physics_v)

        return physics_loss

    def _compute_boundary_loss(
        self,
        predictions: torch.Tensor,
        metadata: List[Dict],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Boundary loss: enforce physical constraints.

        Constraints:
        - Drift speed should be reasonable (< 2 m/s typically)
        - In Southern Hemisphere, wind-driven component should deflect left
        """
        # Speed constraint
        pred_speed = torch.sqrt(predictions[:, :, 0]**2 + predictions[:, :, 1]**2)
        max_speed = 2.0  # m/s - reasonable max for icebergs
        speed_violation = F.relu(pred_speed - max_speed)
        speed_loss = speed_violation.mean()

        # Hemisphere deflection constraint (Southern: wind from North -> drift West)
        # This is a soft constraint
        lats = torch.tensor([m.get('init_lat', -65.0) for m in metadata], device=device, dtype=torch.float32)
        southern_mask = (lats < 0).float()

        # For wind from North (negative v), expect negative u (Westward) in South
        # This is complex to enforce generally, so we use speed constraint primarily
        boundary_loss = speed_loss

        return boundary_loss


# =============================================================================
# PINN Drift Model
# =============================================================================

class PINNDriftModel(nn.Module):
    """
    Physics-Informed Neural Network for iceberg drift.

    Architecture:
    1. Physics model computes baseline drift (current + 2% wind + Coriolis)
    2. ML model learns correction/residual from data
    3. Combined prediction = physics + ML_correction
    4. Physics-informed loss regularizes ML predictions
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,  # u, v correction
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 24,
        ml_model_type: str = "lstm",
        # Physics parameters
        wind_factor: float = 0.02,
        coriolis_alpha: float = 0.015,
        # Loss weights
        loss_weights: Dict[str, float] = None,
        # Iceberg parameters
        iceberg_params: Optional[IcebergParameters] = None,
        # Feature indices for environmental variables
        feature_idx_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        self.wind_factor = wind_factor
        self.coriolis_alpha = coriolis_alpha
        self.feature_idx_map = feature_idx_map

        # Physics model
        self.physics_model = PhysicsDriftModel(iceberg_params or IcebergParameters())
        # Override wind factor if provided
        self.physics_model.wind_factor = wind_factor

        # ML correction model
        self.ml_model = create_model(
            ml_model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=prediction_horizon,
        )

        # Loss function
        loss_weights = loss_weights or {}
        self.loss_fn = PhysicsInformedLoss(
            data_weight=loss_weights.get("data", 1.0),
            physics_weight=loss_weights.get("physics", 0.5),
            boundary_weight=loss_weights.get("boundary", 0.1),
            wind_factor=wind_factor,
            coriolis_alpha=coriolis_alpha,
            feature_idx_map=feature_idx_map,
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        metadata: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics + ML correction.

        Args:
            x: Input features (batch, seq_len, input_dim)
            targets: Target velocities for teacher forcing (batch, horizon, 2)
            teacher_forcing_ratio: For autoregressive decoding
            metadata: List of dicts with 'init_lat', 'init_lon', etc.

        Returns:
            Dict with:
                - predictions: Combined physics + ML prediction (batch, horizon, 2)
                - physics_pred: Physics-only prediction (batch, horizon, 2)
                - ml_correction: ML correction (batch, horizon, 2)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Default metadata
        if metadata is None:
            metadata = [{'init_lat': -65.0, 'init_lon': 0.0}] * batch_size

        # Extract current and wind from last timestep of input
        # Use feature indices if provided, else fallback to last 4 columns (for backward compatibility)
        if self.feature_idx_map is not None:
            current_u_idx = self.feature_idx_map['current_uo']
            current_v_idx = self.feature_idx_map['current_vo']
            wind_u_idx = self.feature_idx_map['wind_u10']
            wind_v_idx = self.feature_idx_map['wind_v10']
            current_u = x[:, -1, current_u_idx]
            current_v = x[:, -1, current_v_idx]
            wind_u = x[:, -1, wind_u_idx]
            wind_v = x[:, -1, wind_v_idx]
        else:
            # Fallback to original indexing (last 4 columns) - but this is likely wrong after feature engineering
            current_u = x[:, -1, -4]
            current_v = x[:, -1, -3]
            wind_u = x[:, -1, -2]
            wind_v = x[:, -1, -1]

        lats = torch.tensor([m.get('init_lat', -65.0) for m in metadata], device=device, dtype=torch.float32)
        lons = torch.tensor([m.get('init_lon', 0.0) for m in metadata], device=device, dtype=torch.float32)

        # --- Physics prediction ---
        physics_u, physics_v = compute_physics_drift_batch(
            latitudes=lats.cpu().numpy(),
            longitudes=lons.cpu().numpy(),
            current_u=current_u.cpu().numpy(),
            current_v=current_v.cpu().numpy(),
            wind_u=wind_u.cpu().numpy(),
            wind_v=wind_v.cpu().numpy(),
            wind_factor=self.wind_factor,
            coriolis_alpha=self.coriolis_alpha,
        )

        physics_u = torch.tensor(physics_u, device=device, dtype=torch.float32)
        physics_v = torch.tensor(physics_v, device=device, dtype=torch.float32)

        # Expand to prediction horizon (assume constant forcing for now)
        physics_u = physics_u.unsqueeze(1).expand(-1, self.prediction_horizon)
        physics_v = physics_v.unsqueeze(1).expand(-1, self.prediction_horizon)
        physics_pred = torch.stack([physics_u, physics_v], dim=-1)  # (batch, horizon, 2)

        # --- ML correction ---
        # ML model predicts correction to physics
        ml_correction = self.ml_model(x, targets, teacher_forcing_ratio)

        # --- Combined prediction ---
        predictions = physics_pred + ml_correction

        return {
            "predictions": predictions,
            "physics_pred": physics_pred,
            "ml_correction": ml_correction,
        }

    def compute_loss(
        self,
        predictions_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        inputs: torch.Tensor,
        metadata: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-informed loss."""
        return self.loss_fn(
            predictions=predictions_dict["predictions"],
            targets=targets,
            inputs=inputs,
            metadata=metadata,
        )

    def predict_trajectory(
        self,
        init_lat: float,
        init_lon: float,
        forcing_sequence: Dict[str, np.ndarray],
        feature_sequence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict full trajectory for inference.

        Args:
            init_lat, init_lon: Initial position
            forcing_sequence: Dict with 'current_u', 'current_v', 'wind_u', 'wind_v' arrays (n_steps,)
            feature_sequence: Full feature sequence for ML model (seq_len, input_dim)

        Returns:
            (lats, lons) trajectory arrays
        """
        self.eval()
        device = next(self.parameters()).device

        n_steps = len(forcing_sequence['current_u'])

        # Prepare input tensor
        x = torch.tensor(feature_sequence[None, :, :], dtype=torch.float32, device=device)
        # Expand to prediction horizon steps
        metadata = [{'init_lat': init_lat, 'init_lon': init_lon}]

        with torch.no_grad():
            result = self.forward(x, metadata=metadata)
            pred_velocities = result["predictions"].cpu().numpy()[0]  # (horizon, 2)

        # Integrate trajectory
        u_drift = pred_velocities[:, 0]
        v_drift = pred_velocities[:, 1]

        lats, lons = integrate_trajectory_batch(
            init_lats=np.array([init_lat]),
            init_lons=np.array([init_lon]),
            u_drift=u_drift[None, :],
            v_drift=v_drift[None, :],
            dt=3600.0,
        )

        return lats[0], lons[0]


# =============================================================================
# Hybrid Model (Physics + ML with learnable physics parameters)
# =============================================================================

class HybridDriftModel(nn.Module):
    """
    Hybrid model where physics parameters are learnable.

    - wind_factor and coriolis_alpha are learned parameters
    - ML model learns residual correction
    - Can be trained end-to-end
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 24,
        ml_model_type: str = "lstm",
        init_wind_factor: float = 0.02,
        init_coriolis_alpha: float = 0.015,
        feature_idx_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        self.prediction_horizon = prediction_horizon
        self.feature_idx_map = feature_idx_map

        # Learnable physics parameters (constrained to reasonable ranges)
        self.log_wind_factor = nn.Parameter(torch.log(torch.tensor(init_wind_factor)))
        self.log_coriolis_alpha = nn.Parameter(torch.log(torch.tensor(init_coriolis_alpha)))

        # ML correction model
        self.ml_model = create_model(
            ml_model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=prediction_horizon,
        )

    @property
    def wind_factor(self) -> torch.Tensor:
        """Wind factor constrained to (0.001, 0.1)."""
        return torch.sigmoid(self.log_wind_factor) * 0.099 + 0.001

    @property
    def coriolis_alpha(self) -> torch.Tensor:
        """Coriolis alpha constrained to (0.001, 0.05)."""
        return torch.sigmoid(self.log_coriolis_alpha) * 0.049 + 0.001

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        metadata: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        if metadata is None:
            metadata = [{'init_lat': -65.0, 'init_lon': 0.0}] * batch_size

        # Extract forcings
        if self.feature_idx_map is not None:
            current_u_idx = self.feature_idx_map['current_uo']
            current_v_idx = self.feature_idx_map['current_vo']
            wind_u_idx = self.feature_idx_map['wind_u10']
            wind_v_idx = self.feature_idx_map['wind_v10']
            current_u = x[:, -1, current_u_idx]
            current_v = x[:, -1, current_v_idx]
            wind_u = x[:, -1, wind_u_idx]
            wind_v = x[:, -1, wind_v_idx]
        else:
            # Fallback to original indexing (last 4 columns)
            current_u = x[:, -1, -4]
            current_v = x[:, -1, -3]
            wind_u = x[:, -1, -2]
            wind_v = x[:, -1, -1]

        lats = torch.tensor([m.get('init_lat', -65.0) for m in metadata], device=device, dtype=torch.float32)
        lons = torch.tensor([m.get('init_lon', 0.0) for m in metadata], device=device, dtype=torch.float32)

        # Physics prediction with learnable parameters
        wf = self.wind_factor
        ca = self.coriolis_alpha

        hem_sign = torch.where(lats < 0, -1.0, 1.0)

        physics_u = (current_u +
                     wf * wind_u +
                     hem_sign * ca * wind_v)

        physics_v = (current_v +
                     wf * wind_v -
                     hem_sign * ca * wind_u)

        physics_u = physics_u.unsqueeze(1).expand(-1, self.prediction_horizon)
        physics_v = physics_v.unsqueeze(1).expand(-1, self.prediction_horizon)
        physics_pred = torch.stack([physics_u, physics_v], dim=-1)

        # ML correction
        ml_correction = self.ml_model(x, targets, teacher_forcing_ratio)

        # Combined
        predictions = physics_pred + ml_correction

        return {
            "predictions": predictions,
            "physics_pred": physics_pred,
            "ml_correction": ml_correction,
            "wind_factor": wf,
            "coriolis_alpha": ca,
        }


# =============================================================================
# Import PhysicsDriftModel (defined in physics_model.py)
# =============================================================================

from .physics_model import PhysicsDriftModel