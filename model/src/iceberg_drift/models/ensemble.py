"""Ensemble methods for iceberg drift prediction."""

import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnsembleDriftModel(nn.Module):
    """
    Ensemble of multiple drift models for improved predictions and uncertainty quantification.

    Methods:
    - Mean ensemble: Average predictions
    - Weighted ensemble: Learn weights for each model
    - Quantile regression: Predict prediction intervals
    """

    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "mean",  # "mean", "weighted", "quantile"
        output_dim: int = 2,
        prediction_horizon: int = 24,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        self.n_models = len(models)

        if ensemble_method == "weighted":
            # Learnable weights for each model
            self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        elif ensemble_method == "quantile":
            # For quantile regression, each model predicts different quantiles
            pass

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        metadata: Optional[List[Dict]] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all ensemble members.

        Returns:
            Dict with ensemble prediction and optionally individual predictions
        """
        all_predictions = []
        all_physics_preds = []
        all_ml_corrections = []

        for model in self.models:
            if hasattr(model, 'forward') and callable(model.forward):
                # Check if it's a PINN/hybrid model
                if hasattr(model, 'physics_model') or hasattr(model, 'wind_factor'):
                    out = model(x, targets, teacher_forcing_ratio, metadata)
                    all_predictions.append(out["predictions"])
                    if "physics_pred" in out:
                        all_physics_preds.append(out["physics_pred"])
                    if "ml_correction" in out:
                        all_ml_corrections.append(out["ml_correction"])
                else:
                    # Standard ML model
                    pred = model(x, targets, teacher_forcing_ratio)
                    all_predictions.append(pred)

        # Stack: (n_models, batch, horizon, output_dim)
        all_predictions = torch.stack(all_predictions, dim=0)

        # Ensemble combination
        if self.ensemble_method == "mean":
            ensemble_pred = all_predictions.mean(dim=0)
            ensemble_std = all_predictions.std(dim=0)

        elif self.ensemble_method == "weighted":
            weights = torch.softmax(self.weights, dim=0)
            # weights: (n_models,) -> (n_models, 1, 1, 1)
            weights = weights.view(-1, 1, 1, 1)
            ensemble_pred = (all_predictions * weights).sum(dim=0)
            # Weighted std
            ensemble_std = torch.sqrt(((all_predictions - ensemble_pred.unsqueeze(0))**2 * weights).sum(dim=0))

        elif self.ensemble_method == "quantile":
            # Assume models predict different quantiles
            # Model 0: 0.1 quantile, Model 1: 0.5 (median), Model 2: 0.9 quantile
            ensemble_pred = all_predictions[1]  # Median
            ensemble_std = (all_predictions[2] - all_predictions[0]) / 2.56  # Approx 90% interval

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        result = {
            "predictions": ensemble_pred,
            "uncertainty": ensemble_std,
        }

        if return_all:
            result["individual_predictions"] = all_predictions
            if all_physics_preds:
                result["physics_predictions"] = torch.stack(all_physics_preds, dim=0)
            if all_ml_corrections:
                result["ml_corrections"] = torch.stack(all_ml_corrections, dim=0)

        return result

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo dropout for uncertainty estimation.

        Returns:
            mean_pred, std_pred, samples (n_samples, batch, horizon, output_dim)
        """
        self.train()  # Enable dropout
        samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x, metadata=metadata)
                samples.append(out["predictions"].cpu().numpy())

        samples = np.stack(samples, axis=0)  # (n_samples, batch, horizon, output_dim)
        mean_pred = samples.mean(axis=0)
        std_pred = samples.std(axis=0)

        return mean_pred, std_pred, samples


def create_ensemble(
    model_configs: List[Dict[str, Any]],
    input_dim: int,
    output_dim: int,
    prediction_horizon: int,
    ensemble_method: str = "mean",
) -> EnsembleDriftModel:
    """
    Create ensemble from list of model configurations.

    Args:
        model_configs: List of dicts with keys: 'type', 'hidden_dim', 'num_layers', etc.
        input_dim: Input feature dimension
        output_dim: Output dimension
        prediction_horizon: Prediction horizon
        ensemble_method: "mean", "weighted", or "quantile"

    Returns:
        EnsembleDriftModel
    """
    from .ml_models import create_model
    from .pinn import PINNDriftModel, HybridDriftModel

    models = []

    for config in model_configs:
        model_type = config.pop("type", "lstm")

        if model_type == "pinn":
            model = PINNDriftModel(
                input_dim=input_dim,
                output_dim=output_dim,
                prediction_horizon=prediction_horizon,
                **config,
            )
        elif model_type == "hybrid":
            model = HybridDriftModel(
                input_dim=input_dim,
                output_dim=output_dim,
                prediction_horizon=prediction_horizon,
                **config,
            )
        else:
            model = create_model(
                model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                prediction_horizon=prediction_horizon,
                **config,
            )

        models.append(model)

    return EnsembleDriftModel(
        models=models,
        ensemble_method=ensemble_method,
        output_dim=output_dim,
        prediction_horizon=prediction_horizon,
    )


# =============================================================================
# Simple Ensemble (non-neural) for traditional ML models
# =============================================================================

class SklearnEnsemble:
    """
    Ensemble wrapper for scikit-learn compatible models (XGBoost, LightGBM, etc.).
    """

    def __init__(
        self,
        models: List[Any],
        weights: Optional[List[float]] = None,
    ):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with ensemble."""
        predictions = []
        for model, w in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * w)

        return np.sum(predictions, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)."""
        probas = []
        for model, w in zip(self.models, self.weights):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probas.append(proba * w)
            else:
                # For regression, return point prediction
                pred = model.predict(X)
                probas.append(pred * w)

        return np.sum(probas, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "SklearnEnsemble":
        """Fit all models."""
        for model in self.models:
            model.fit(X, y, **kwargs)
        return self