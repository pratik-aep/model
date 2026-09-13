"""Validation and evaluation utilities for iceberg drift prediction."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import (
    compute_all_metrics,
    bootstrap_confidence_interval,
    compare_models_bootstrap,
    DriftMetrics,
)
from ..models import PhysicsDriftModel
from ..models.pinn import PINNDriftModel, HybridDriftModel

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    # Metrics
    metric_horizons: List[int] = None
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # Comparison
    compare_with_persistence: bool = True
    compare_with_physics: bool = True

    # Output
    save_predictions: bool = True
    save_metrics: bool = True
    output_dir: str = "output/validation"

    # Visualization
    plot_trajectories: bool = True
    plot_error_distribution: bool = True
    plot_skill_scores: bool = True
    max_trajectories_to_plot: int = 20


@dataclass
class ValidationResult:
    """Results from model validation."""
    metrics: DriftMetrics
    predictions: Dict[str, np.ndarray]
    targets: Dict[str, np.ndarray]
    metadata: List[Dict]
    bootstrap_ci: Dict[str, Tuple[float, float, float]]
    comparison_results: Dict[str, Any]


# =============================================================================
# Validator Class
# =============================================================================

class DriftValidator:
    """
    Comprehensive model validation for iceberg drift prediction.

    Features:
    - Multi-horizon evaluation
    - Bootstrap confidence intervals
    - Baseline comparisons (persistence, physics)
    - Trajectory visualization
    - Error analysis by region, season, iceberg size
    """

    def __init__(
        self,
        model: nn.Module,
        config: ValidationConfig,
        device: str = "auto",
    ):
        self.model = model
        self.config = config
        if config.metric_horizons is None:
            self.config.metric_horizons = [6, 12, 24, 48, 72]

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Validator initialized on {self.device}")

    def validate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = True,
    ) -> ValidationResult:
        """
        Run full validation on test set.

        Args:
            test_loader: DataLoader for test data
            return_predictions: Whether to store all predictions

        Returns:
            ValidationResult with metrics and analysis
        """
        logger.info("Running validation...")

        all_predictions = []
        all_targets = []
        all_metadata = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                metadata = batch["meta"]

                # Model-specific forward
                if isinstance(self.model, (PINNDriftModel, HybridDriftModel)):
                    output = self.model(x, targets=None, teacher_forcing_ratio=0.0, metadata=metadata)
                    predictions = output["predictions"]
                else:
                    predictions = self.model(x, targets=None, teacher_forcing_ratio=0.0)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_metadata.extend(metadata)

        # Concatenate
        predictions_np = np.concatenate(all_predictions, axis=0)  # (N, H, 2)
        targets_np = np.concatenate(all_targets, axis=0)

        # Format for metrics
        pred_dict = {"u": predictions_np[:, :, 0], "v": predictions_np[:, :, 1]}
        true_dict = {"u": targets_np[:, :, 0], "v": targets_np[:, :, 1]}

        # Compute metrics
        metrics = compute_all_metrics(
            pred_dict, true_dict, all_metadata,
            horizon_hours=self.config.metric_horizons,
        )

        # Bootstrap confidence intervals
        bootstrap_ci = {}
        for metric_name in ["mean_position_error_km", "rmse_position_km", "mean_direction_error_deg"]:
            metric_func = self._get_metric_func(metric_name)
            point, lower, upper = bootstrap_confidence_interval(
                metric_func,
                predictions_np,
                targets_np,
                n_bootstrap=self.config.bootstrap_samples,
                confidence=self.config.confidence_level,
            )
            bootstrap_ci[metric_name] = (point, lower, upper)

        # Baseline comparisons
        comparison_results = {}
        if self.config.compare_with_persistence:
            comparison_results["persistence"] = self._compare_with_persistence(
                predictions_np, targets_np, all_metadata
            )
        if self.config.compare_with_physics:
            comparison_results["physics"] = self._compare_with_physics(
                predictions_np, targets_np, all_metadata
            )

        result = ValidationResult(
            metrics=metrics,
            predictions=pred_dict,
            targets=true_dict,
            metadata=all_metadata,
            bootstrap_ci=bootstrap_ci,
            comparison_results=comparison_results,
        )

        # Save results
        if self.config.save_metrics:
            self._save_metrics(result)
        if self.config.save_predictions:
            self._save_predictions(result)

        # Generate plots
        if self.config.plot_trajectories:
            self._plot_trajectories(result)
        if self.config.plot_error_distribution:
            self._plot_error_distribution(result)
        if self.config.plot_skill_scores:
            self._plot_skill_scores(result)

        return result

    def _get_metric_func(self, metric_name: str) -> Callable:
        """Get metric function for bootstrap."""
        if metric_name == "mean_position_error_km":
            return lambda pred, true: np.mean(
                np.sqrt((pred[:, :, 0] - true[:, :, 0])**2 + (pred[:, :, 1] - true[:, :, 1])**2)
            )
        elif metric_name == "rmse_position_km":
            return lambda pred, true: np.sqrt(np.mean(
                (pred[:, :, 0] - true[:, :, 0])**2 + (pred[:, :, 1] - true[:, :, 1])**2
            ))
        elif metric_name == "mean_direction_error_deg":
            def _dir_error(pred, true):
                pred_dir = np.degrees(np.arctan2(pred[:, :, 1], pred[:, :, 0])) % 360
                true_dir = np.degrees(np.arctan2(true[:, :, 1], true[:, :, 0])) % 360
                diff = np.abs(pred_dir - true_dir)
                diff = np.minimum(diff, 360 - diff)
                return np.mean(diff)
            return _dir_error
        else:
            return lambda pred, true: np.mean((pred - true)**2)

    def _compare_with_persistence(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: List[Dict],
    ) -> Dict[str, Any]:
        """Compare with persistence baseline (no movement)."""
        # Persistence: predict initial position for all horizons
        n_samples, horizon, output_dim = predictions.shape

        # Get initial positions from metadata
        init_lats = np.array([m.get('init_lat', 0) for m in metadata])
        init_lons = np.array([m.get('init_lon', 0) for m in metadata])

        # Expand to match
        if len(init_lats) != n_samples:
            n_per = n_samples // len(init_lats)
            init_lats = np.repeat(init_lats, n_per)
            init_lons = np.repeat(init_lons, n_per)

        # Persistence prediction (zero velocity)
        persist_pred = np.zeros_like(predictions)

        # Reshape for regression metrics: (samples, horizon * output_dim)
        predictions_flat = predictions.reshape(n_samples, -1)
        targets_flat = targets.reshape(n_samples, -1)
        persist_pred_flat = persist_pred.reshape(n_samples, -1)

        # Compare
        from .metrics import regression_metrics
        model_metrics = regression_metrics(predictions_flat, targets_flat)
        persist_metrics = regression_metrics(persist_pred_flat, targets_flat)

        return {
            "model_rmse": model_metrics["rmse"],
            "persistence_rmse": persist_metrics["rmse"],
            "improvement": persist_metrics["rmse"] - model_metrics["rmse"],
            "relative_improvement_pct": 100 * (persist_metrics["rmse"] - model_metrics["rmse"]) / persist_metrics["rmse"],
        }

    def _compare_with_physics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: List[Dict],
    ) -> Dict[str, Any]:
        """Compare with physics baseline (current + wind drift + Coriolis)."""
        try:
            from iceberg_drift.models import PhysicsDriftModel
            from iceberg_drift.data_processing.preprocessing import compute_coriolis

            n_samples, horizon, _ = predictions.shape

            # Extract environmental forcings from metadata
            # We need: current_u, current_v, wind_u, wind_v, lat for each sample
            current_u = np.array([m.get('current_uo', 0) for m in metadata])
            current_v = np.array([m.get('current_vo', 0) for m in metadata])
            wind_u = np.array([m.get('wind_u10', 0) for m in metadata])
            wind_v = np.array([m.get('wind_v10', 0) for m in metadata])
            lats = np.array([m.get('init_lat', -65) for m in metadata])

            # Expand to match prediction shape if needed
            if len(current_u) != n_samples:
                n_per = n_samples // len(current_u)
                current_u = np.repeat(current_u, n_per)
                current_v = np.repeat(current_v, n_per)
                wind_u = np.repeat(wind_u, n_per)
                wind_v = np.repeat(wind_v, n_per)
                lats = np.repeat(lats, n_per)

            # Physics model with default parameters
            physics_model = PhysicsDriftModel()
            wind_factor = physics_model.wind_factor  # ~0.02
            coriolis_alpha = 0.015  # Coriolis deflection coefficient

            # Compute physics baseline for each timestep
            # For multi-horizon, we assume constant forcing (persistence of environmental conditions)
            hem_sign = np.where(lats < 0, -1.0, 1.0)

            # Physics prediction for each horizon step
            physics_u = np.zeros((n_samples, horizon))
            physics_v = np.zeros((n_samples, horizon))

            for h in range(horizon):
                physics_u[:, h] = (current_u +
                                   wind_factor * wind_u +
                                   hem_sign * coriolis_alpha * wind_v)
                physics_v[:, h] = (current_v +
                                   wind_factor * wind_v -
                                   hem_sign * coriolis_alpha * wind_u)

            physics_pred = np.stack([physics_u, physics_v], axis=-1)

            # Compare model vs physics
            from .metrics import regression_metrics, position_error_km

            model_metrics = regression_metrics(predictions, targets)
            physics_metrics = regression_metrics(physics_pred, targets)

            # Also compute position-level errors
            # Need to integrate velocities to positions for fair comparison
            # Get initial positions
            init_lats = np.array([m.get('init_lat', -65) for m in metadata])
            init_lons = np.array([m.get('init_lon', 0) for m in metadata])

            if len(init_lats) != n_samples:
                n_per = n_samples // len(init_lats)
                init_lats = np.repeat(init_lats, n_per)
                init_lons = np.repeat(init_lons, n_per)

            # Integrate predictions and physics to get positions
            from iceberg_drift.models.physics_model import integrate_trajectory_batch
            dt = 3600.0  # 1 hour (assuming hourly timesteps for velocity)

            # For position error, we need the true positions
            # These would need to be in metadata or reconstructed from true velocities
            # For now, use velocity-level comparison

            return {
                "model_rmse": model_metrics["rmse"],
                "physics_rmse": physics_metrics["rmse"],
                "model_mae": model_metrics["mae"],
                "physics_mae": physics_metrics["mae"],
                "improvement_rmse": physics_metrics["rmse"] - model_metrics["rmse"],
                "relative_improvement_pct": 100 * (physics_metrics["rmse"] - model_metrics["rmse"]) / physics_metrics["rmse"],
                "physics_wind_factor": wind_factor,
                "physics_coriolis_alpha": coriolis_alpha,
            }

        except Exception as e:
            logger.warning(f"Physics baseline comparison failed: {e}")
            return {
                "note": f"Physics comparison failed: {str(e)}",
                "model_rmse": float(np.sqrt(np.mean((predictions - targets)**2))),
            }

    def _save_metrics(self, result: ValidationResult):
        """Save metrics to file."""
        import json

        metrics_dict = {
            "mean_position_error_km": result.metrics.mean_position_error_km,
            "median_position_error_km": result.metrics.median_position_error_km,
            "rmse_position_km": result.metrics.rmse_position_km,
            "p90_position_error_km": result.metrics.p90_position_error_km,
            "p95_position_error_km": result.metrics.p95_position_error_km,
            "mean_speed_error": result.metrics.mean_speed_error,
            "rmse_speed": result.metrics.rmse_speed,
            "mean_direction_error_deg": result.metrics.mean_direction_error_deg,
            "median_direction_error_deg": result.metrics.median_direction_error_deg,
            "mean_drift_distance_error_km": result.metrics.mean_drift_distance_error_km,
            "mean_drift_direction_error_deg": result.metrics.mean_drift_direction_error_deg,
            "skill_score_vs_persistence": result.metrics.skill_score_vs_persistence,
            "skill_score_vs_physics": result.metrics.skill_score_vs_physics,
            "horizon_metrics": result.metrics.horizon_metrics,
            "bootstrap_ci": {
                k: {"point": v[0], "lower": v[1], "upper": v[2]}
                for k, v in result.bootstrap_ci.items()
            },
            "comparison_results": result.comparison_results,
        }

        output_file = self.output_dir / "metrics.json"
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        logger.info(f"Metrics saved to {output_file}")

    def _save_predictions(self, result: ValidationResult):
        """Save predictions to file."""
        output_file = self.output_dir / "predictions.npz"
        np.savez_compressed(
            output_file,
            pred_u=result.predictions["u"],
            pred_v=result.predictions["v"],
            true_u=result.targets["u"],
            true_v=result.targets["v"],
        )
        logger.info(f"Predictions saved to {output_file}")

    def _plot_trajectories(self, result: ValidationResult):
        """Plot predicted vs true trajectories."""
        try:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            pred_u = result.predictions["u"]
            pred_v = result.predictions["v"]
            true_u = result.targets["u"]
            true_v = result.targets["v"]
            metadata = result.metadata

            # Limit number of trajectories
            n_plot = min(self.config.max_trajectories_to_plot, len(metadata))
            indices = np.random.choice(len(metadata), n_plot, replace=False)

            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
            ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.COASTLINE)
            ax.gridlines(draw_labels=True)

            for idx in indices:
                meta = metadata[idx]
                init_lat = meta.get('init_lat', -65)
                init_lon = meta.get('init_lon', 0)

                # Integrate predicted trajectory
                pred_lats, pred_lons = self._integrate_velocities(
                    init_lat, init_lon, pred_u[idx], pred_v[idx]
                )
                true_lats, true_lons = self._integrate_velocities(
                    init_lat, init_lon, true_u[idx], true_v[idx]
                )

                # Plot
                ax.plot(pred_lons, pred_lats, 'r-', alpha=0.5, linewidth=1,
                       transform=ccrs.PlateCarree(), label='Predicted' if idx == indices[0] else "")
                ax.plot(true_lons, true_lats, 'b-', alpha=0.5, linewidth=1,
                       transform=ccrs.PlateCarree(), label='True' if idx == indices[0] else "")
                ax.plot(init_lon, init_lat, 'go', markersize=5,
                       transform=ccrs.PlateCarree())

            ax.legend()
            ax.set_title("Iceberg Drift Trajectories: Predicted vs True")
            plt.savefig(self.output_dir / "trajectories.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Trajectory plot saved")

        except ImportError:
            logger.warning("Cartopy not available, skipping trajectory plot")
        except Exception as e:
            logger.warning(f"Failed to save trajectory plot due to: {e}")

    def _integrate_velocities(
        self,
        init_lat: float,
        init_lon: float,
        u: np.ndarray,
        v: np.ndarray,
        dt_hours: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate velocity to get trajectory."""
        lats = [init_lat]
        lons = [init_lon]
        lat, lon = init_lat, init_lon

        for uu, vv in zip(u, v):
            dt_seconds = dt_hours * 3600
            lat += (vv * dt_seconds) / 111000.0
            lon += (uu * dt_seconds) / (111000.0 * np.cos(np.radians(lat)))
            lats.append(lat)
            lons.append(lon)

        return np.array(lats), np.array(lons)

    def _plot_error_distribution(self, result: ValidationResult):
        """Plot error distributions."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            pred_u = result.predictions["u"]
            pred_v = result.predictions["v"]
            true_u = result.targets["u"]
            true_v = result.targets["v"]

            # Position errors
            pos_errors = np.sqrt(
                (pred_u - true_u)**2 + (pred_v - true_v)**2
            ).flatten() * 111  # Convert to km (approximate)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Position error histogram
            axes[0, 0].hist(pos_errors, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(np.mean(pos_errors), color='red', linestyle='--',
                              label=f'Mean: {np.mean(pos_errors):.1f} km')
            axes[0, 0].axvline(np.median(pos_errors), color='blue', linestyle='--',
                              label=f'Median: {np.median(pos_errors):.1f} km')
            axes[0, 0].set_xlabel('Position Error (km)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Position Error Distribution')
            axes[0, 0].legend()

            # Direction error
            pred_dir = np.degrees(np.arctan2(pred_v, pred_u)) % 360
            true_dir = np.degrees(np.arctan2(true_v, true_u)) % 360
            dir_errors = np.abs(pred_dir - true_dir)
            dir_errors = np.minimum(dir_errors, 360 - dir_errors).flatten()

            axes[0, 1].hist(dir_errors, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(np.mean(dir_errors), color='red', linestyle='--',
                              label=f'Mean: {np.mean(dir_errors):.1f}°')
            axes[0, 1].set_xlabel('Direction Error (degrees)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Direction Error Distribution')
            axes[0, 1].legend()

            # Error vs horizon
            if pred_u.ndim == 3:
                n_samples, horizon, _ = pred_u.shape
                horizon_errors = []
                for h in range(horizon):
                    h_err = np.sqrt(
                        (pred_u[:, h, 0] - true_u[:, h, 0])**2 +
                        (pred_v[:, h, 0] - true_v[:, h, 0])**2
                    ) * 111
                    horizon_errors.append(h_err)

                axes[1, 0].boxplot(horizon_errors)
                axes[1, 0].set_xlabel('Prediction Horizon (steps)')
                axes[1, 0].set_ylabel('Position Error (km)')
                axes[1, 0].set_title('Error by Prediction Horizon')
                axes[1, 0].set_xticklabels([f'{h*6}h' for h in range(horizon)])

            # Speed error
            pred_speed = np.sqrt(pred_u**2 + pred_v**2)
            true_speed = np.sqrt(true_u**2 + true_v**2)
            speed_errors = (pred_speed - true_speed).flatten()

            axes[1, 1].hist(speed_errors, bins=50, edgecolor='black', alpha=0.7)
            axes[1, 1].axvline(np.mean(speed_errors), color='red', linestyle='--',
                              label=f'Mean: {np.mean(speed_errors):.3f} m/s')
            axes[1, 1].set_xlabel('Speed Error (m/s)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Speed Error Distribution')
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / "error_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Error distribution plot saved")

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping error distribution plot")

    def _plot_skill_scores(self, result: ValidationResult):
        """Plot skill scores by horizon."""
        try:
            import matplotlib.pyplot as plt

            horizon_metrics = result.metrics.horizon_metrics
            if not horizon_metrics:
                return

            horizons = sorted(horizon_metrics.keys())
            skill_persist = [horizon_metrics[h].get('skill_vs_persistence', np.nan) for h in horizons]
            skill_physics = [horizon_metrics[h].get('skill_vs_physics', np.nan) for h in horizons]

            plt.figure(figsize=(10, 6))
            plt.plot(horizons, skill_persist, 'o-', label='vs Persistence', linewidth=2)
            plt.plot(horizons, skill_physics, 's-', label='vs Physics', linewidth=2)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel('Prediction Horizon (hours)')
            plt.ylabel('Skill Score')
            plt.title('Model Skill Score by Prediction Horizon')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "skill_scores.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Skill scores plot saved")

        except ImportError:
            logger.warning("Matplotlib not available, skipping skill scores plot")


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_errors_by_region(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: List[Dict],
    region_bounds: Dict[str, Tuple[float, float, float, float]] = None,
) -> Dict[str, DriftMetrics]:
    """
    Analyze errors by geographic region.

    Args:
        predictions: Predicted velocities (N, H, 2)
        targets: True velocities (N, H, 2)
        metadata: List of dicts with 'init_lat', 'init_lon'
        region_bounds: Dict of region_name -> (min_lat, max_lat, min_lon, max_lon)

    Returns:
        Dict of region_name -> DriftMetrics
    """
    if region_bounds is None:
        # Default Antarctic sectors
        region_bounds = {
            "Weddell_Sea": (-80, -65, -60, 20),
            "Ross_Sea": (-85, -70, 160, -160),
            "Amundsen_Sea": (-80, -70, -130, -100),
            "Bellingshausen_Sea": (-75, -65, -100, -60),
            "Indian_Ocean_Sector": (-70, -55, 20, 90),
            "Pacific_Ocean_Sector": (-70, -55, 90, 160),
            "Atlantic_Ocean_Sector": (-70, -55, -60, 20),
        }

    results = {}
    init_lats = np.array([m.get('init_lat', -65) for m in metadata])
    init_lons = np.array([m.get('init_lon', 0) for m in metadata])

    for region_name, (min_lat, max_lat, min_lon, max_lon) in region_bounds.items():
        # Handle longitude wrap-around
        if min_lon > max_lon:  # Crosses 180/-180
            mask = ((init_lat >= min_lat) & (init_lat <= max_lat) &
                    ((init_lon >= min_lon) | (init_lon <= max_lon)))
        else:
            mask = ((init_lats >= min_lat) & (init_lats <= max_lat) &
                    (init_lons >= min_lon) & (init_lons <= max_lon))

        if np.sum(mask) > 10:  # Minimum samples
            region_pred = predictions[mask]
            region_true = targets[mask]
            region_meta = [metadata[i] for i in np.where(mask)[0]]

            pred_dict = {"u": region_pred[:, :, 0], "v": region_pred[:, :, 1]}
            true_dict = {"u": region_true[:, :, 0], "v": region_true[:, :, 1]}

            results[region_name] = compute_all_metrics(pred_dict, true_dict, region_meta)

    return results


def analyze_errors_by_season(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: List[Dict],
) -> Dict[str, DriftMetrics]:
    """Analyze errors by season."""
    # Extract month from metadata
    months = []
    for m in metadata:
        if 'init_time' in m:
            if hasattr(m['init_time'], 'month'):
                months.append(m['init_time'].month)
            else:
                months.append(1)  # Default
        else:
            months.append(1)

    months = np.array(months)

    # Define seasons (Southern Hemisphere)
    season_map = {
        "Summer": [12, 1, 2],   # DJF
        "Autumn": [3, 4, 5],    # MAM
        "Winter": [6, 7, 8],    # JJA
        "Spring": [9, 10, 11],  # SON
    }

    results = {}
    for season_name, season_months in season_map.items():
        mask = np.isin(months, season_months)
        if np.sum(mask) > 10:
            region_pred = predictions[mask]
            region_true = targets[mask]
            region_meta = [metadata[i] for i in np.where(mask)[0]]

            pred_dict = {"u": region_pred[:, :, 0], "v": region_pred[:, :, 1]}
            true_dict = {"u": region_true[:, :, 0], "v": region_true[:, :, 1]}

            results[season_name] = compute_all_metrics(pred_dict, true_dict, region_meta)

    return results


def analyze_errors_by_size(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: List[Dict],
    size_bins: List[Tuple[float, float]] = None,
) -> Dict[str, DriftMetrics]:
    """Analyze errors by iceberg size."""
    if size_bins is None:
        size_bins = [(0, 1000), (1000, 5000), (5000, 20000), (20000, np.inf)]

    results = {}

    for i, (min_size, max_size) in enumerate(size_bins):
        mask = []
        for m in metadata:
            length = m.get('length_m', 0)
            if min_size <= length < max_size:
                mask.append(True)
            else:
                mask.append(False)

        mask = np.array(mask)
        if np.sum(mask) > 10:
            region_pred = predictions[mask]
            region_true = targets[mask]
            region_meta = [metadata[j] for j in np.where(mask)[0]]

            pred_dict = {"u": region_pred[:, :, 0], "v": region_pred[:, :, 1]}
            true_dict = {"u": region_true[:, :, 0], "v": region_true[:, :, 1]}

            label = f"{int(min_size/1000)}km-{int(max_size/1000)}km" if max_size < np.inf else f">{int(min_size/1000)}km"
            results[label] = compute_all_metrics(pred_dict, true_dict, region_meta)

    return results


# =============================================================================
# Convenience Function
# =============================================================================

def validate_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: Optional[ValidationConfig] = None,
    device: str = "auto",
) -> ValidationResult:
    """
    Convenience function to validate a model.

    Args:
        model: Trained model
        test_loader: Test DataLoader
        config: ValidationConfig (uses defaults if None)
        device: Device to run on

    Returns:
        ValidationResult
    """
    if config is None:
        config = ValidationConfig()

    validator = DriftValidator(model, config, device)
    return validator.validate(test_loader)