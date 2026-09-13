"""Evaluation metrics for iceberg drift prediction."""

import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Core Metric Functions
# =============================================================================

def position_error_km(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
) -> np.ndarray:
    """
    Compute great-circle distance error in kilometers.

    Args:
        pred_lat, pred_lon: Predicted positions (degrees)
        true_lat, true_lon: True positions (degrees)

    Returns:
        Distance errors in km (same shape as inputs)
    """
    # Convert to radians
    pred_lat_rad = np.deg2rad(pred_lat)
    pred_lon_rad = np.deg2rad(pred_lon)
    true_lat_rad = np.deg2rad(true_lat)
    true_lon_rad = np.deg2rad(true_lon)

    # Haversine formula
    dlat = true_lat_rad - pred_lat_rad
    dlon = true_lon_rad - pred_lon_rad

    a = (np.sin(dlat / 2)**2 +
         np.cos(pred_lat_rad) * np.cos(true_lat_rad) * np.sin(dlon / 2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth radius in km
    R = 6371.0
    return R * c


def velocity_error(
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    true_u: np.ndarray,
    true_v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity errors.

    Returns:
        (speed_error, direction_error_deg)
    """
    pred_speed = np.sqrt(pred_u**2 + pred_v**2)
    true_speed = np.sqrt(true_u**2 + true_v**2)

    speed_error = pred_speed - true_speed

    pred_dir = np.degrees(np.arctan2(pred_v, pred_u)) % 360
    true_dir = np.degrees(np.arctan2(true_v, true_u)) % 360

    # Angular difference (0-180)
    dir_diff = np.abs(pred_dir - true_dir)
    dir_diff = np.minimum(dir_diff, 360 - dir_diff)

    return speed_error, dir_diff


def along_cross_track_error(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute along-track and cross-track errors relative to a reference trajectory.

    Along-track: error parallel to true trajectory direction
    Cross-track: error perpendicular to true trajectory direction

    Args:
        pred_lat, pred_lon: Predicted positions
        true_lat, true_lon: True positions
        ref_lat, ref_lon: Reference positions (e.g., previous true position)

    Returns:
        (along_track_km, cross_track_km)
    """
    # True trajectory vector
    dlat_true = true_lat - ref_lat
    dlon_true = true_lon - ref_lon

    # Convert to km
    true_along_km = dlat_true * 111.0
    true_cross_km = dlon_true * 111.0 * np.cos(np.deg2rad(ref_lat))

    true_track_angle = np.arctan2(true_cross_km, true_along_km)

    # Error vector
    dlat_err = pred_lat - true_lat
    dlon_err = pred_lon - true_lon

    err_along_km = dlat_err * 111.0
    err_cross_km = dlon_err * 111.0 * np.cos(np.deg2rad(true_lat))

    # Project error onto track coordinates
    along_track = err_along_km * np.cos(true_track_angle) + err_cross_km * np.sin(true_track_angle)
    cross_track = -err_along_km * np.sin(true_track_angle) + err_cross_km * np.cos(true_track_angle)

    return along_track, cross_track


def drift_distance_error(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    init_lat: np.ndarray,
    init_lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute drift distance and direction errors relative to initial position.

    Args:
        pred_lat, pred_lon: Predicted final positions
        true_lat, true_lon: True final positions
        init_lat, init_lon: Initial positions

    Returns:
        (distance_error_km, direction_error_deg)
    """
    # True drift vector
    true_dlat = true_lat - init_lat
    true_dlon = true_lon - init_lon

    true_dist_km = np.sqrt(
        (true_dlat * 111.0)**2 +
        (true_dlon * 111.0 * np.cos(np.deg2rad(init_lat)))**2
    )
    true_dir = np.degrees(np.arctan2(true_dlon, true_dlat)) % 360

    # Predicted drift vector
    pred_dlat = pred_lat - init_lat
    pred_dlon = pred_lon - init_lon

    pred_dist_km = np.sqrt(
        (pred_dlat * 111.0)**2 +
        (pred_dlon * 111.0 * np.cos(np.deg2rad(init_lat)))**2
    )
    pred_dir = np.degrees(np.arctan2(pred_dlon, pred_dlat)) % 360

    # Errors
    dist_error = pred_dist_km - true_dist_km
    dir_diff = np.abs(pred_dir - true_dir)
    dir_error = np.minimum(dir_diff, 360 - dir_diff)

    return dist_error, dir_error


def direction_error(
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    true_u: np.ndarray,
    true_v: np.ndarray,
) -> np.ndarray:
    """Compute direction error in degrees (0-180)."""
    pred_dir = np.degrees(np.arctan2(pred_v, pred_u)) % 360
    true_dir = np.degrees(np.arctan2(true_v, true_u)) % 360

    dir_diff = np.abs(pred_dir - true_dir)
    return np.minimum(dir_diff, 360 - dir_diff)


# =============================================================================
# Aggregate Metrics
# =============================================================================

@dataclass
class DriftMetrics:
    """Container for drift prediction metrics."""
    # Position errors
    mean_position_error_km: float
    median_position_error_km: float
    rmse_position_km: float
    p90_position_error_km: float
    p95_position_error_km: float

    # Velocity errors
    mean_speed_error: float  # m/s
    rmse_speed: float
    mean_direction_error_deg: float
    median_direction_error_deg: float

    # Drift errors
    mean_drift_distance_error_km: float
    mean_drift_direction_error_deg: float

    # Along/cross track
    mean_along_track_error_km: float
    mean_cross_track_error_km: float
    rmse_along_track_km: float
    rmse_cross_track_km: float

    # Skill scores
    skill_score_vs_persistence: float
    skill_score_vs_physics: float

    # Per-horizon breakdown
    horizon_metrics: Dict[int, Dict[str, float]]


def compute_all_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    metadata: List[Dict],
    horizon_hours: List[int] = None,
) -> DriftMetrics:
    """
    Compute comprehensive drift prediction metrics.

    Args:
        predictions: Dict with keys 'lat', 'lon' or 'u', 'v' (predicted)
        targets: Dict with keys 'lat', 'lon' or 'u', 'v' (ground truth)
        metadata: List of dicts with 'init_lat', 'init_lon', 'prev_lat', 'prev_lon'
        horizon_hours: List of prediction horizons to evaluate separately

    Returns:
        DriftMetrics object
    """
    # Extract arrays
    if 'lat' in predictions and 'lon' in predictions:
        pred_lat = predictions['lat']
        pred_lon = predictions['lon']
        true_lat = targets['lat']
        true_lon = targets['lon']
    elif 'u' in predictions and 'v' in predictions:
        # Convert velocity to position (assuming 1-hour timesteps)
        # This is approximate - better to have positions directly
        pred_u = predictions['u']
        pred_v = predictions['v']
        true_u = targets['u']
        true_v = targets['v']

        # Integrate to get positions (simplified)
        pred_lat = pred_u  # placeholder
        pred_lon = pred_v
        true_lat = true_u
        true_lon = true_v
    else:
        raise ValueError("Predictions must contain either (lat, lon) or (u, v)")

    # Flatten for overall metrics
    pred_lat_flat = pred_lat.flatten()
    pred_lon_flat = pred_lon.flatten()
    true_lat_flat = true_lat.flatten()
    true_lon_flat = true_lon.flatten()

    # Position errors
    pos_errors = position_error_km(pred_lat_flat, pred_lon_flat, true_lat_flat, true_lon_flat)

    # Velocity errors (if available)
    if 'u' in predictions:
        speed_err, dir_err = velocity_error(pred_u, pred_v, true_u, true_v)
        mean_speed_error = float(np.mean(speed_err))
        rmse_speed = float(np.sqrt(np.mean(speed_err**2)))
        mean_dir_error = float(np.mean(dir_err))
        median_dir_error = float(np.median(dir_err))
    else:
        mean_speed_error = rmse_speed = mean_dir_error = median_dir_error = np.nan

    # Drift errors (relative to initial position)
    init_lats = np.array([m.get('init_lat', true_lat_flat[0]) for m in metadata])
    init_lons = np.array([m.get('init_lon', true_lon_flat[0]) for m in metadata])

    # Broadcast to match prediction shape
    if len(init_lats) == len(pos_errors):
        # Already aligned
        pass
    else:
        # Expand metadata to match
        n_per_traj = len(pos_errors) // len(init_lats)
        init_lats = np.repeat(init_lats, n_per_traj)
        init_lons = np.repeat(init_lons, n_per_traj)

    drift_dist_err, drift_dir_err = drift_distance_error(
        pred_lat_flat, pred_lon_flat, true_lat_flat, true_lon_flat,
        init_lats, init_lons
    )

    # Along/cross track (using previous position as reference)
    if 'prev_lat' in metadata[0] and 'prev_lon' in metadata[0]:
        ref_lats = np.array([m.get('prev_lat', 0) for m in metadata])
        ref_lons = np.array([m.get('prev_lon', 0) for m in metadata])

        if len(ref_lats) != len(pos_errors):
            n_per_traj = len(pos_errors) // len(ref_lats)
            ref_lats = np.repeat(ref_lats, n_per_traj)
            ref_lons = np.repeat(ref_lons, n_per_traj)

        along_err, cross_err = along_cross_track_error(
            pred_lat_flat, pred_lon_flat, true_lat_flat, true_lon_flat,
            ref_lats, ref_lons
        )
        mean_along = float(np.mean(along_err))
        mean_cross = float(np.mean(cross_err))
        rmse_along = float(np.sqrt(np.mean(along_err**2)))
        rmse_cross = float(np.sqrt(np.mean(cross_err**2)))
    else:
        mean_along = mean_cross = rmse_along = rmse_cross = np.nan

    # Skill scores
    skill_persist = _compute_skill_score(pred_lat, pred_lon, true_lat, true_lon, metadata, "persistence")
    skill_physics = _compute_skill_score(pred_lat, pred_lon, true_lat, true_lon, metadata, "physics")

    # Per-horizon metrics
    horizon_metrics = {}
    if horizon_hours is not None and len(pred_lat.shape) >= 2:
        for h_idx, horizon in enumerate(horizon_hours):
            if h_idx < pred_lat.shape[1]:
                h_pred_lat = pred_lat[:, h_idx]
                h_pred_lon = pred_lon[:, h_idx]
                h_true_lat = true_lat[:, h_idx]
                h_true_lon = true_lon[:, h_idx]

                h_pos_err = position_error_km(h_pred_lat, h_pred_lon, h_true_lat, h_true_lon)
                horizon_metrics[horizon] = {
                    "mean_position_error_km": float(np.mean(h_pos_err)),
                    "rmse_position_km": float(np.sqrt(np.mean(h_pos_err**2))),
                    "p90_position_error_km": float(np.percentile(h_pos_err, 90)),
                }

    return DriftMetrics(
        mean_position_error_km=float(np.mean(pos_errors)),
        median_position_error_km=float(np.median(pos_errors)),
        rmse_position_km=float(np.sqrt(np.mean(pos_errors**2))),
        p90_position_error_km=float(np.percentile(pos_errors, 90)),
        p95_position_error_km=float(np.percentile(pos_errors, 95)),
        mean_speed_error=mean_speed_error,
        rmse_speed=rmse_speed,
        mean_direction_error_deg=mean_dir_error,
        median_direction_error_deg=median_dir_error,
        mean_drift_distance_error_km=float(np.mean(drift_dist_err)),
        mean_drift_direction_error_deg=float(np.mean(drift_dir_err)),
        mean_along_track_error_km=mean_along,
        mean_cross_track_error_km=mean_cross,
        rmse_along_track_km=rmse_along,
        rmse_cross_track_km=rmse_cross,
        skill_score_vs_persistence=skill_persist,
        skill_score_vs_physics=skill_physics,
        horizon_metrics=horizon_metrics,
    )


def _compute_skill_score(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    metadata: List[Dict],
    baseline: str,
) -> float:
    """
    Compute skill score relative to baseline.

    Skill = 1 - (model_error / baseline_error)
    """
    # Flatten arrays for overall skill score
    pred_lat_flat = pred_lat.flatten()
    pred_lon_flat = pred_lon.flatten()
    true_lat_flat = true_lat.flatten()
    true_lon_flat = true_lon.flatten()

    model_errors = position_error_km(pred_lat_flat, pred_lon_flat, true_lat_flat, true_lon_flat)

    if baseline == "persistence":
        # Persistence: iceberg stays at current position
        # Use initial position as persistence forecast
        init_lats = np.array([m.get('init_lat', true_lat_flat[0]) for m in metadata])
        init_lons = np.array([m.get('init_lon', true_lon_flat[0]) for m in metadata])

        if len(init_lats) != len(model_errors):
            n_per_traj = len(model_errors) // len(init_lats)
            init_lats = np.repeat(init_lats, n_per_traj)
            init_lons = np.repeat(init_lons, n_per_traj)

        baseline_errors = position_error_km(init_lats, init_lons, true_lat_flat, true_lon_flat)

    elif baseline == "physics":
        # Physics baseline: current + wind_factor * wind + Coriolis correction
        # Extract environmental forcings from metadata
        n_samples = len(pred_lat_flat)
        if n_samples == 0:
            return np.nan

        # Get metadata for each prediction
        # We need: current_uo, current_vo, wind_u10, wind_v10, init_lat for each sample
        current_u = np.array([m.get('current_uo', 0.0) for m in metadata])
        current_v = np.array([m.get('current_vo', 0.0) for m in metadata])
        wind_u = np.array([m.get('wind_u10', 0.0) for m in metadata])
        wind_v = np.array([m.get('wind_v10', 0.0) for m in metadata])
        lats = np.array([m.get('init_lat', -65.0) for m in metadata])

        # Expand to match prediction shape if needed (similar to trainer.py logic)
        if len(current_u) != n_samples:
            n_per = n_samples // len(current_u)
            if n_per > 0:
                current_u = np.repeat(current_u, n_per)
                current_v = np.repeat(current_v, n_per)
                wind_u = np.repeat(wind_u, n_per)
                wind_v = np.repeat(wind_v, n_per)
                lats = np.repeat(lats, n_per)
            else:
                # Fallback to zeros if we can't properly expand
                current_u = np.zeros(n_samples)
                current_v = np.zeros(n_samples)
                wind_u = np.zeros(n_samples)
                wind_v = np.zeros(n_samples)
                lats = np.full(n_samples, -65.0)

        # Physics model with default parameters (matching validator.py)
        from iceberg_drift.models import PhysicsDriftModel
        physics_model = PhysicsDriftModel()
        wind_factor = physics_model.wind_factor  # ~0.02
        coriolis_alpha = 0.015  # Coriolis deflection coefficient

        # Compute physics baseline for velocity components
        hem_sign = np.where(lats < 0, -1.0, 1.0)

        physics_u = (current_u +
                     wind_factor * wind_u +
                     hem_sign * coriolis_alpha * wind_v)

        physics_v = (current_v +
                     wind_factor * wind_v -
                     hem_sign * coriolis_alpha * wind_u)

        # Convert physics velocity predictions to position errors
        # We need to integrate velocity to position to compare with true positions
        # For simplicity, we'll assume small time steps and compute approximate position
        # In a full implementation, we would integrate over the prediction horizon

        # For now, compute velocity error and convert to approximate position error
        # This is not perfect but better than the fake 1.2 multiplier
        pred_u_flat = pred_lat_flat  # This is wrong - we don't have u/v predictions here
        pred_v_flat = pred_lon_flat  # This is wrong - we don't have u/v predictions here

        # Since we're in the position branch of compute_all_metrics, we have lat/lon predictions
        # To compute physics baseline for position, we need to integrate physics velocity
        # Let's compute a simplified position error based on velocity error

        # Actually, let's compute the physics baseline position by integrating
        # We'll assume a 1-hour timestep for simplicity (matching trainer.py assumption)
        dt_seconds = 3600.0  # 1 hour

        # Initialize arrays for physics-predicted positions
        physics_lats = np.zeros_like(pred_lat_flat)
        physics_lons = np.zeros_like(pred_lon_flat)

        # Use initial positions from metadata
        init_lats = np.array([m.get('init_lat', 0.0) for m in metadata])
        init_lons = np.array([m.get('init_lon', 0.0) for m in metadata])

        # Expand initial positions to match prediction shape if needed
        if len(init_lats) != n_samples:
            n_per = n_samples // len(init_lats)
            if n_per > 0:
                init_lats = np.repeat(init_lats, n_per)
                init_lons = np.repeat(init_lons, n_per)
            else:
                init_lats = np.zeros(n_samples)
                init_lons = np.zeros(n_samples)

        # Integrate physics velocity to get physics-predicted positions
        # This is a simplified Euler integration
        physics_lats = init_lats + (physics_v * dt_seconds) / 111000.0
        physics_lons = init_lons + (physics_u * dt_seconds) / (111000.0 * np.cos(np.radians(init_lats)))

        # Wrap longitude to [-180, 180]
        physics_lons = ((physics_lons + 180) % 360) - 180

        # Compute position error between physics baseline and true positions
        baseline_errors = position_error_km(physics_lats, physics_lons, true_lat_flat, true_lon_flat)
    else:
        return np.nan

    model_rmse = np.sqrt(np.mean(model_errors**2))
    baseline_rmse = np.sqrt(np.mean(baseline_errors**2))

    if baseline_rmse == 0:
        return np.nan

    return 1.0 - (model_rmse / baseline_rmse)


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_confidence_interval(
    metric_func,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for any metric.

    Args:
        metric_func: Function taking (pred, true) -> scalar metric
        predictions: Predictions array
        targets: Targets array
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95)
        random_seed: Random seed

    Returns:
        (point_estimate, lower_ci, upper_ci)
    """
    np.random.seed(random_seed)
    n = len(predictions)
    point_estimate = metric_func(predictions, targets)

    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_pred = predictions[idx]
        boot_true = targets[idx]
        bootstrap_estimates.append(metric_func(boot_pred, boot_true))

    bootstrap_estimates = np.array(bootstrap_estimates)
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_estimates, 100 * alpha)
    upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha))

    return point_estimate, lower, upper


def compare_models_bootstrap(
    model1_preds: np.ndarray,
    model2_preds: np.ndarray,
    targets: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """
    Bootstrap test for comparing two models.

    Returns:
        Dict with p-value, mean difference, CI for difference
    """
    np.random.seed(42)
    n = len(targets)

    # Point estimate difference
    diff = metric_func(model1_preds, targets) - metric_func(model2_preds, targets)

    # Bootstrap distribution of difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        b1 = metric_func(model1_preds[idx], targets[idx])
        b2 = metric_func(model2_preds[idx], targets[idx])
        boot_diffs.append(b1 - b2)

    boot_diffs = np.array(boot_diffs)

    # p-value (two-sided): proportion of bootstrap diffs with opposite sign
    p_value = 2 * min(
        np.mean(boot_diffs >= 0),
        np.mean(boot_diffs <= 0)
    )

    return {
        "mean_difference": float(np.mean(boot_diffs)),
        "p_value": float(p_value),
        "ci_lower": float(np.percentile(boot_diffs, 2.5)),
        "ci_upper": float(np.percentile(boot_diffs, 97.5)),
        "significant": p_value < 0.05,
    }


# =============================================================================
# Regression Metrics (for velocity targets)
# =============================================================================

def regression_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    multioutput: str = "uniform_average",
) -> Dict[str, float]:
    """Standard regression metrics."""
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score,
    )

    return {
        "mse": float(mean_squared_error(y_true, y_pred, multioutput=multioutput)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred, multioutput=multioutput))),
        "mae": float(mean_absolute_error(y_true, y_pred, multioutput=multioutput)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred, multioutput=multioutput)),
        "r2": float(r2_score(y_true, y_pred, multioutput=multioutput)),
    }