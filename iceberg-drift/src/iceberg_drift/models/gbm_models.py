"""Gradient Boosted Machine models for iceberg drift prediction.

Implements XGBoost and LightGBM regressors with scikit-learn compatible interface
for predicting iceberg drift velocity components (u, v).
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class GBMConfig:
    """Configuration for GBM models."""
    # Model type
    model_type: str = "xgboost"  # "xgboost", "lightgbm", "both"

    # Target configuration
    target_names: List[str] = field(default_factory=lambda: ["target_u", "target_v"])

    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "gamma": 0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    })

    # LightGBM parameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "early_stopping_rounds": 50,
    })

    # Training
    use_gpu: bool = False
    val_split: float = 0.15
    random_state: int = 42


@dataclass
class GBMModelResult:
    """Result container for trained GBM model."""
    model: Any
    feature_names: List[str]
    target_name: str
    best_iteration: int
    feature_importance: pd.DataFrame
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]


# =============================================================================
# Base GBM Wrapper
# =============================================================================

class BaseGBMRegressor(BaseEstimator, RegressorMixin):
    """Base class for GBM regressors with scikit-learn compatibility."""

    def __init__(self, config: GBMConfig = None, target_name: str = "target_u"):
        self.config = config or GBMConfig()
        self.target_name = target_name
        self.model = None
        self.feature_names = None
        self.best_iteration = None
        self.feature_importance_ = None
        self._is_fitted = False

    def _get_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _create_model(self, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def _fit_with_early_stopping(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "BaseGBMRegressor":
        """Fit the model with optional validation set for early stopping."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # If no validation set provided, create one from training data
        if X_val is None or y_val is None:
            n_val = int(len(X) * self.config.val_split)
            if n_val > 0:
                idx = np.random.RandomState(self.config.random_state).permutation(len(X))
                val_idx = idx[:n_val]
                train_idx = idx[n_val:]
                X_val, y_val = X[val_idx], y[val_idx]
                X, y = X[train_idx], y[train_idx]
            else:
                X_val, y_val = None, None

        self._fit_with_early_stopping(X, y, X_val, y_val)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._predict(X)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        return self.feature_importance_

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "best_iteration": self.best_iteration,
            "feature_importance": self.feature_importance_,
            "config": self.config,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseGBMRegressor":
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(config=data["config"], target_name=data["target_name"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.best_iteration = data["best_iteration"]
        instance.feature_importance_ = data["feature_importance"]
        instance._is_fitted = True
        return instance


# =============================================================================
# XGBoost Regressor
# =============================================================================

class XGBoostRegressor(BaseGBMRegressor):
    """XGBoost regressor for iceberg drift prediction."""

    def _get_params(self) -> Dict[str, Any]:
        params = self.config.xgb_params.copy()
        if self.config.use_gpu:
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = 0
        return params

    def _create_model(self, params: Dict[str, Any]) -> Any:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        return xgb.XGBRegressor(**params)

    def _fit_with_early_stopping(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        params = self._get_params()
        early_stopping = params.pop("early_stopping_rounds", 50)

        # Pass early_stopping_rounds via constructor (sklearn API), not fit()'s callback
        # path, to avoid the XGBoost 3.x callback segfault on macOS noted previously.
        # Only enable it when we actually have a validation set to watch.
        if X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = early_stopping

        self.model = self._create_model(params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(X, y, eval_set=eval_set, verbose=False)

        self.best_iteration = getattr(self.model, 'best_iteration', None)
        self._extract_feature_importance()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _extract_feature_importance(self) -> None:
        """Extract feature importance from XGBoost model."""
        if self.model is not None and self.feature_names is not None:
            importance = self.model.feature_importances_
            self.feature_importance_ = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False).reset_index(drop=True)


# =============================================================================
# LightGBM Regressor
# =============================================================================

class LightGBMRegressor(BaseGBMRegressor):
    """LightGBM regressor for iceberg drift prediction."""

    def _get_params(self) -> Dict[str, Any]:
        params = self.config.lgb_params.copy()
        if self.config.use_gpu:
            params["device"] = "gpu"
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0
        return params

    def _create_model(self, params: Dict[str, Any]) -> Any:
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        return lgb.LGBMRegressor(**params)

    def _fit_with_early_stopping(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        params = self._get_params()
        early_stopping = params.pop("early_stopping_rounds", 50)

        self.model = self._create_model(params)

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [lgb.early_stopping(early_stopping), lgb.log_evaluation(0)]
        else:
            eval_set = None
            callbacks = [lgb.log_evaluation(0)]

        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        self.best_iteration = getattr(self.model, 'best_iteration_', None)
        self._extract_feature_importance()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _extract_feature_importance(self) -> None:
        """Extract feature importance from LightGBM model."""
        if self.model is not None and self.feature_names is not None:
            importance = self.model.feature_importances_
            self.feature_importance_ = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False).reset_index(drop=True)


# =============================================================================
# Multi-Target GBM Wrapper
# =============================================================================

class MultiTargetGBM:
    """
    Wrapper to train separate GBM models for each target (velocity_u, velocity_v).

    This follows the checklist requirement: "one for velocity_u, one for velocity_v"
    """

    def __init__(
        self,
        config: GBMConfig = None,
        target_names: List[str] = None,
    ):
        self.config = config or GBMConfig()
        self.target_names = target_names or ["target_u", "target_v"]
        self.models: Dict[str, BaseGBMRegressor] = {}
        self.feature_names: List[str] = []
        self.training_results: Dict[str, GBMModelResult] = {}

    def _create_model(self, target_name: str) -> BaseGBMRegressor:
        """Create appropriate model based on config."""
        if self.config.model_type == "xgboost":
            return XGBoostRegressor(config=self.config, target_name=target_name)
        elif self.config.model_type == "lightgbm":
            return LightGBMRegressor(config=self.config, target_name=target_name)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,  # shape: (n_samples, n_targets)
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "MultiTargetGBM":
        """
        Fit separate models for each target.

        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples, n_targets)
            X_val, y_val: Validation data
            feature_names: Names of features
        """
        n_targets = y.shape[1] if len(y.shape) > 1 else 1
        target_names = self.target_names[:n_targets]

        logger.info(f"Training GBM models for targets: {target_names}")

        for i, target_name in enumerate(target_names):
            logger.info(f"Training model for {target_name} ({i+1}/{len(target_names)})")

            y_i = y[:, i] if n_targets > 1 else y
            y_val_i = y_val[:, i] if y_val is not None and n_targets > 1 else y_val

            model = self._create_model(target_name)
            model.fit(X, y_i, X_val, y_val_i, feature_names)

            # Store model and results
            self.models[target_name] = model
            self.feature_names = model.feature_names

            # Compute training metrics
            train_pred = model.predict(X)
            train_metrics = self._compute_metrics(y_i, train_pred, "train")

            val_metrics = {}
            if X_val is not None:
                val_pred = model.predict(X_val)
                val_metrics = self._compute_metrics(y_val_i, val_pred, "val")

            self.training_results[target_name] = GBMModelResult(
                model=model.model,
                feature_names=model.feature_names,
                target_name=target_name,
                best_iteration=model.best_iteration,
                feature_importance=model.get_feature_importance(),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            logger.info(f"  {target_name}: train_RMSE={train_metrics['rmse']:.4f}, "
                       f"val_RMSE={val_metrics.get('rmse', 'N/A'):.4f}, "
                       f"best_iter={model.best_iteration}")

        return self

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
        """Compute regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        return {
            f"{prefix}_mse": float(mse),
            f"{prefix}_rmse": float(np.sqrt(mse)),
            f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict all targets.

        Returns:
            Array of shape (n_samples, n_targets)
        """
        if not self.models:
            raise ValueError("Models not fitted. Call fit() first.")

        predictions = []
        for target_name in self.target_names:
            if target_name in self.models:
                pred = self.models[target_name].predict(X)
                predictions.append(pred)
            else:
                raise ValueError(f"Model for {target_name} not found")

        return np.column_stack(predictions)

    def predict_single(self, X: np.ndarray, target_name: str) -> np.ndarray:
        """Predict a single target."""
        if target_name not in self.models:
            raise ValueError(f"Model for {target_name} not found")
        return self.models[target_name].predict(X)

    def get_feature_importance(self, target_name: str = None) -> pd.DataFrame:
        """Get feature importance for all targets or a specific one."""
        if target_name:
            if target_name in self.models:
                return self.models[target_name].get_feature_importance()
            raise ValueError(f"Model for {target_name} not found")

        # Combine importance across all targets
        dfs = []
        for target_name, model in self.models.items():
            imp = model.get_feature_importance()
            if not imp.empty:
                imp = imp.copy()
                imp["target"] = target_name
                dfs.append(imp)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get aggregated feature importance across all targets."""
        all_imp = self.get_feature_importance()
        if all_imp.empty:
            return pd.DataFrame()

        # Average importance across targets
        summary = all_imp.groupby("feature")["importance"].agg(["mean", "std", "min", "max"])
        summary = summary.sort_values("mean", ascending=False).reset_index()
        summary.columns = ["feature", "mean_importance", "std_importance", "min_importance", "max_importance"]
        return summary

    def save(self, path: str) -> None:
        """Save all models to directory."""
        Path(path).mkdir(parents=True, exist_ok=True)
        for target_name, model in self.models.items():
            model.save(Path(path) / f"{target_name}.pkl")

        # Save metadata
        metadata = {
            "config": self.config,
            "target_names": self.target_names,
            "feature_names": self.feature_names,
        }
        joblib.dump(metadata, Path(path) / "metadata.pkl")
        logger.info(f"Multi-target GBM saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MultiTargetGBM":
        """Load all models from directory."""
        metadata = joblib.load(Path(path) / "metadata.pkl")
        instance = cls(config=metadata["config"], target_names=metadata["target_names"])
        instance.feature_names = metadata["feature_names"]

        for target_name in instance.target_names:
            model_path = Path(path) / f"{target_name}.pkl"
            if model_path.exists():
                if instance.config.model_type == "xgboost":
                    instance.models[target_name] = XGBoostRegressor.load(str(model_path))
                elif instance.config.model_type == "lightgbm":
                    instance.models[target_name] = LightGBMRegressor.load(str(model_path))

        logger.info(f"Multi-target GBM loaded from {path}")
        return instance


# =============================================================================
# Physics-Informed GBM (Residual Learning)
# =============================================================================

class PhysicsInformedGBM:
    """
    Physics-informed GBM that learns residuals from physics baseline.

    Prediction = Physics_Baseline + GBM_Residual
    """

    def __init__(
        self,
        gbm_config: GBMConfig = None,
        physics_model: Any = None,
        target_names: List[str] = None,
    ):
        self.gbm_config = gbm_config or GBMConfig()
        self.physics_model = physics_model
        self.target_names = target_names or ["target_u", "target_v"]
        self.gbm = MultiTargetGBM(config=self.gbm_config, target_names=self.target_names)
        self.feature_names = []

    def _compute_physics_predictions(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        """Compute physics baseline predictions from features."""
        if self.physics_model is None:
            return np.zeros((X.shape[0], len(self.target_names)))

        # Extract current_u, current_v, wind_u, wind_v, lat from features
        # Assuming feature names match preprocessing output
        fnames = feature_names or self.feature_names

        # Find indices
        idx_map = {name: i for i, name in enumerate(fnames)}

        physics_preds = []
        for target in self.target_names:
            pred = np.zeros(X.shape[0])
            if target == "target_u":
                # u_drift = current_u + wind_factor * wind_u + coriolis * wind_v
                if all(k in idx_map for k in ["current_uo", "wind_u10", "wind_v10", "lat"]):
                    current_u = X[:, idx_map["current_uo"]]
                    wind_u = X[:, idx_map["wind_u10"]]
                    wind_v = X[:, idx_map["wind_v10"]]
                    lat = X[:, idx_map["lat"]]

                    # Simple physics: current + 2% wind + Coriolis deflection
                    f = 2 * 7.2921150e-5 * np.sin(np.radians(lat))
                    hem_sign = np.where(lat < 0, -1.0, 1.0)
                    pred = current_u + 0.02 * wind_u + hem_sign * 0.015 * wind_v
            elif target == "target_v":
                # v_drift = current_v + wind_factor * wind_v - coriolis * wind_u
                if all(k in idx_map for k in ["current_vo", "wind_u10", "wind_v10", "lat"]):
                    current_v = X[:, idx_map["current_vo"]]
                    wind_u = X[:, idx_map["wind_u10"]]
                    wind_v = X[:, idx_map["wind_v10"]]
                    lat = X[:, idx_map["lat"]]

                    f = 2 * 7.2921150e-5 * np.sin(np.radians(lat))
                    hem_sign = np.where(lat < 0, -1.0, 1.0)
                    pred = current_v + 0.02 * wind_v - hem_sign * 0.015 * wind_u

            physics_preds.append(pred)

        return np.column_stack(physics_preds)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "PhysicsInformedGBM":
        """
        Fit GBM on residuals: y - physics_baseline.
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Compute physics predictions
        physics_train = self._compute_physics_predictions(X, self.feature_names)

        # Target residuals
        residuals = y - physics_train

        if X_val is not None:
            physics_val = self._compute_physics_predictions(X_val, self.feature_names)
            residuals_val = y_val - physics_val
        else:
            residuals_val = None

        logger.info("Training Physics-Informed GBM on residuals...")
        logger.info(f"Physics baseline RMSE: {np.sqrt(np.mean(physics_train**2)):.4f}")
        logger.info(f"Residual std: {np.std(residuals):.4f}")

        # Train GBM on residuals
        self.gbm.fit(X, residuals, X_val, residuals_val, self.feature_names)

        # Log improvement
        gbm_pred = self.gbm.predict(X)
        combined_pred = physics_train + gbm_pred
        combined_rmse = np.sqrt(np.mean((combined_pred - y)**2))
        physics_rmse = np.sqrt(np.mean((physics_train - y)**2))
        logger.info(f"Physics RMSE: {physics_rmse:.4f}")
        logger.info(f"Combined (Physics+GBM) RMSE: {combined_rmse:.4f}")
        logger.info(f"Improvement: {physics_rmse - combined_rmse:.4f} ({100*(physics_rmse-combined_rmse)/physics_rmse:.1f}%)")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict: physics_baseline + GBM_residual."""
        physics_pred = self._compute_physics_predictions(X, self.feature_names)
        gbm_pred = self.gbm.predict(X)
        return physics_pred + gbm_pred

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the residual GBM."""
        return self.gbm.get_feature_importance_summary()

    def save(self, path: str) -> None:
        """Save model."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.gbm.save(Path(path) / "gbm")
        # Save physics model reference if needed
        joblib.dump({
            "gbm_config": self.gbm_config,
            "target_names": self.target_names,
            "feature_names": self.feature_names,
        }, Path(path) / "physics_informed_meta.pkl")

    @classmethod
    def load(cls, path: str, physics_model: Any = None) -> "PhysicsInformedGBM":
        """Load model."""
        meta = joblib.load(Path(path) / "physics_informed_meta.pkl")
        instance = cls(
            gbm_config=meta["gbm_config"],
            physics_model=physics_model,
            target_names=meta["target_names"],
        )
        instance.feature_names = meta["feature_names"]
        instance.gbm = MultiTargetGBM.load(str(Path(path) / "gbm"))
        return instance


# =============================================================================
# Factory Functions
# =============================================================================

def create_gbm_model(
    model_type: str,
    config: GBMConfig = None,
    target_names: List[str] = None,
) -> Union[MultiTargetGBM, PhysicsInformedGBM]:
    """Factory to create GBM models."""
    if model_type in ["xgboost", "lightgbm", "both"]:
        return MultiTargetGBM(config=config, target_names=target_names)
    elif model_type == "physics_informed":
        return PhysicsInformedGBM(gbm_config=config, target_names=target_names)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_gbm_config(
    model_type: str = "xgboost",
    n_estimators: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    use_gpu: bool = False,
) -> GBMConfig:
    """Create GBM configuration with sensible defaults."""
    config = GBMConfig(model_type=model_type)
    config.xgb_params.update({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    })
    config.lgb_params.update({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    })
    config.use_gpu = use_gpu
    return config


# =============================================================================
# Ensemble GBM (Bagging/Boosting combinations)
# =============================================================================

class EnsembleGBM:
    """
    Ensemble of multiple GBM models for improved predictions and uncertainty.
    """

    def __init__(
        self,
        n_models: int = 5,
        model_type: str = "xgboost",
        config: GBMConfig = None,
        target_names: List[str] = None,
        aggregation: str = "mean",  # "mean", "median", "weighted"
    ):
        self.n_models = n_models
        self.model_type = model_type
        self.config = config or GBMConfig(model_type=model_type)
        self.target_names = target_names or ["target_u", "target_v"]
        self.aggregation = aggregation
        self.models: List[MultiTargetGBM] = []
        self.weights = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        bootstrap: bool = True,
    ) -> "EnsembleGBM":
        """Train ensemble with bootstrap sampling."""
        n_samples = X.shape[0]

        for i in range(self.n_models):
            logger.info(f"Training ensemble model {i+1}/{self.n_models}")

            # Bootstrap sample
            if bootstrap:
                idx = np.random.choice(n_samples, n_samples, replace=True)
                X_boot, y_boot = X[idx], y[idx]
            else:
                X_boot, y_boot = X, y

            model = MultiTargetGBM(config=self.config, target_names=self.target_names)
            model.fit(X_boot, y_boot, X_val, y_val, feature_names)
            self.models.append(model)

        # Compute ensemble weights based on validation performance
        if self.aggregation == "weighted" and X_val is not None:
            self._compute_weights(X_val, y_val)

        return self

    def _compute_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Compute weights based on validation RMSE."""
        errors = []
        for model in self.models:
            pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((pred - y_val)**2))
            errors.append(rmse)

        errors = np.array(errors)
        # Inverse error weighting (lower error = higher weight)
        self.weights = 1.0 / (errors + 1e-8)
        self.weights = self.weights / self.weights.sum()
        logger.info(f"Ensemble weights: {self.weights}")

    def predict(self, X: np.ndarray, return_all: bool = False) -> np.ndarray:
        """Predict with ensemble."""
        if not self.models:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        all_preds = np.stack([model.predict(X) for model in self.models], axis=0)

        if self.aggregation == "mean":
            ensemble_pred = all_preds.mean(axis=0)
        elif self.aggregation == "median":
            ensemble_pred = np.median(all_preds, axis=0)
        elif self.aggregation == "weighted":
            if self.weights is not None:
                ensemble_pred = np.average(all_preds, axis=0, weights=self.weights)
            else:
                ensemble_pred = all_preds.mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        if return_all:
            return ensemble_pred, all_preds
        return ensemble_pred

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty (std across ensemble members)."""
        ensemble_pred, all_preds = self.predict(X, return_all=True)
        uncertainty = all_preds.std(axis=0)
        return ensemble_pred, uncertainty

    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get aggregated feature importance across ensemble."""
        all_importances = []
        for model in self.models:
            imp = model.get_feature_importance_summary()
            if not imp.empty:
                all_importances.append(imp[["feature", "mean_importance"]].rename(
                    columns={"mean_importance": "importance"}
                ))

        if all_importances:
            combined = pd.concat(all_importances)
            summary = combined.groupby("feature")["importance"].agg(
                ["mean", "std", "min", "max"]
            ).reset_index()
            summary.columns = ["feature", "mean_importance", "std_importance", "min_importance", "max_importance"]
            return summary.sort_values("mean_importance", ascending=False)
        return pd.DataFrame()


# =============================================================================
# Update models/__init__.py exports
# =============================================================================

__all__ = [
    "GBMConfig",
    "GBMModelResult",
    "BaseGBMRegressor",
    "XGBoostRegressor",
    "LightGBMRegressor",
    "MultiTargetGBM",
    "PhysicsInformedGBM",
    "EnsembleGBM",
    "create_gbm_model",
    "create_gbm_config",
]