"""Preprocessing and feature engineering for iceberg drift prediction."""

import logging
from typing import List, Tuple, Dict, Optional, Union
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Environmental Data Matching
# =============================================================================

def match_environmental_data(
    iceberg_df: pd.DataFrame,
    wind_ds: xr.Dataset,
    current_ds: xr.Dataset,
    wind_vars: List[str] = None,
    current_vars: List[str] = None,
    interp_method: str = "linear",
) -> pd.DataFrame:
    """
    Match environmental data (wind, currents) to iceberg positions at each timestamp.

    Args:
        iceberg_df: DataFrame with columns [iceberg_id, datetime, lat, lon, ...]
        wind_ds: xarray Dataset with wind data (time, lat, lon)
        current_ds: xarray Dataset with current data (time, depth, lat, lon)
        wind_vars: Variables to extract from wind_ds (default: ['u10', 'v10'])
        current_vars: Variables to extract from current_ds (default: ['uo', 'vo'])
        interp_method: Interpolation method ('linear', 'nearest')

    Returns:
        DataFrame with environmental variables added as columns
    """
    if wind_vars is None:
        wind_vars = ["u10", "v10"]
    if current_vars is None:
        current_vars = ["uo", "vo"]

    logger.info(f"Matching environmental data to {len(iceberg_df)} iceberg positions")

    # Create interpolators
    wind_interpolators = _create_interpolators(wind_ds, wind_vars, "wind")
    current_interpolators = _create_interpolators(current_ds, current_vars, "current", has_depth=True)

    # Process each row
    matched_data = []

    for _, row in tqdm(iceberg_df.iterrows(), total=len(iceberg_df), desc="Matching env data"):
        # Cast timestamp to float (nanoseconds since epoch) to match the float coordinates
        t = float(pd.Timestamp(row["datetime"]).value)
        lat = row["lat"]
        lon = row["lon"]

        # Match wind (surface)
        wind_values = {}
        for var, interp in wind_interpolators.items():
            try:
                val = interp((t, lat, lon))
                wind_values[f"wind_{var}"] = float(val) if not np.isnan(val) else np.nan
            except Exception as e:
                logger.error(f"Failed to interp wind {var}: {e}")
                wind_values[f"wind_{var}"] = np.nan

        # Match currents (surface - depth=0)
        current_values = {}
        surface_depth = current_ds["depth"].values[0] if "depth" in current_ds.coords else 0.5
        for var, interp in current_interpolators.items():
            try:
                val = interp((t, surface_depth, lat, lon))  # surface
                

                current_values[f"current_{var}"] = float(val) if not np.isnan(val) else np.nan
            except Exception as e:
                logger.error(f"Failed to interp current {var}: {e}")
                current_values[f"current_{var}"] = np.nan

        # Combine
        combined = {**row.to_dict(), **wind_values, **current_values}
        matched_data.append(combined)

    result = pd.DataFrame(matched_data)
    logger.info(f"Matched data shape: {result.shape}")
    return result


def _create_interpolators(
    ds: xr.Dataset,
    variables: List[str],
    prefix: str,
    has_depth: bool = False,
) -> Dict[str, RegularGridInterpolator]:
    """Create scipy interpolators for each variable."""
    interpolators = {}

    # Sort coordinates to ensure they are strictly ascending for RegularGridInterpolator
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    if "time" in ds.coords:
        ds = ds.sortby("time")
    if has_depth and "depth" in ds.coords:
        ds = ds.sortby("depth")

    for var in variables:
        if var not in ds:
            logger.warning(f"Variable {var} not found in {prefix} dataset")
            continue

        data = ds[var].values

        if has_depth:
            # Dimensions: time, depth, lat, lon
            coords = (
                ds["time"].values.astype("datetime64[ns]").astype(float),
                ds["depth"].values,
                ds["latitude"].values,
                ds["longitude"].values,
            )
        else:
            # Dimensions: time, lat, lon
            coords = (
                ds["time"].values.astype("datetime64[ns]").astype(float),
                ds["latitude"].values,
                ds["longitude"].values,
            )

        interpolators[var] = RegularGridInterpolator(
            coords, data, method="linear", bounds_error=False, fill_value=np.nan
        )

    return interpolators


# =============================================================================
# Coriolis and Physics Features
# =============================================================================

def compute_coriolis(lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute Coriolis parameter f = 2 * Ω * sin(lat).

    Args:
        lat: Latitude in degrees (negative for Southern Hemisphere)

    Returns:
        Coriolis parameter in s^-1
    """
    omega = 7.2921150e-5  # Earth's rotation rate (rad/s)
    lat_rad = np.radians(lat)
    return 2 * omega * np.sin(lat_rad)


def compute_coriolis_terms(
    lat: Union[float, np.ndarray],
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Coriolis acceleration terms.

    In Southern Hemisphere (lat < 0), f < 0:
    - Coriolis deflects moving objects to the LEFT
    - u_coriolis = -f * v
    - v_coriolis = f * u

    Args:
        lat: Latitude in degrees
        u: Zonal velocity (eastward, m/s)
        v: Meridional velocity (northward, m/s)

    Returns:
        (u_coriolis, v_coriolis) acceleration terms in m/s^2
    """
    f = compute_coriolis(lat)
    u_coriolis = -f * v
    v_coriolis = f * u
    return u_coriolis, v_coriolis


# =============================================================================
# Feature Engineering
# =============================================================================

def engineer_features(
    df: pd.DataFrame,
    add_cyclical_time: bool = True,
    add_physics_features: bool = True,
    add_lag_features: bool = True,
    lag_hours: List[int] = None,
) -> pd.DataFrame:
    """
    Engineer features for iceberg drift prediction.

    Base features (from matched data):
    - wind_u, wind_v: Wind velocity components (m/s)
    - current_u, current_v: Ocean current velocity components (m/s)
    - lat, lon: Position
    - length_m, width_m: Iceberg dimensions (if available)

    Engineered features:
    - Wind speed, direction
    - Current speed, direction
    - Wind-current alignment
    - Coriolis parameter
    - Cyclical time encoding (sin/cos of hour, day, month)
    - Lag features (previous positions/velocities)
    """
    if lag_hours is None:
        lag_hours = [6, 12, 24, 48]

    df = df.copy()
    df = df.sort_values(["iceberg_id", "datetime"]).reset_index(drop=True)

    logger.info(f"Engineering features for {len(df)} records")

    # --- Wind features ---
    if "wind_u10" in df.columns and "wind_v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["wind_u10"]**2 + df["wind_v10"]**2)
        df["wind_dir"] = np.degrees(np.arctan2(df["wind_v10"], df["wind_u10"])) % 360
        df["wind_stress_magnitude"] = 0.0012 * df["wind_speed"]**2  # Approximate

    # --- Current features ---
    if "current_uo" in df.columns and "current_vo" in df.columns:
        df["current_speed"] = np.sqrt(df["current_uo"]**2 + df["current_vo"]**2)
        df["current_dir"] = np.degrees(np.arctan2(df["current_vo"], df["current_uo"])) % 360

    # --- Wind-current interaction ---
    if all(c in df.columns for c in ["wind_u10", "wind_v10", "current_uo", "current_vo"]):
        # Dot product (alignment)
        df["wind_current_alignment"] = (
            df["wind_u10"] * df["current_uo"] + df["wind_v10"] * df["current_vo"]
        ) / (df["wind_speed"] * df["current_speed"] + 1e-6)

        # Cross product (relative angle)
        df["wind_current_cross"] = (
            df["wind_u10"] * df["current_vo"] - df["wind_v10"] * df["current_uo"]
        ) / (df["wind_speed"] * df["current_speed"] + 1e-6)

        # Relative direction
        df["wind_current_rel_dir"] = (df["wind_dir"] - df["current_dir"]) % 360

    # --- Physics features ---
    if add_physics_features:
        # Coriolis parameter
        df["coriolis_f"] = compute_coriolis(df["lat"])

        # Theoretical drift from current + wind (2% rule)
        if all(c in df.columns for c in ["current_uo", "current_vo", "wind_u10", "wind_v10"]):
            df["physics_u"] = df["current_uo"] + 0.02 * df["wind_u10"]
            df["physics_v"] = df["current_vo"] + 0.02 * df["wind_v10"]
            df["physics_speed"] = np.sqrt(df["physics_u"]**2 + df["physics_v"]**2)
            df["physics_dir"] = np.degrees(np.arctan2(df["physics_v"], df["physics_u"])) % 360

        # Iceberg size features
        if "length_m" in df.columns and "width_m" in df.columns:
            df["iceberg_area"] = df["length_m"] * df["width_m"] / 1e6  # km^2
            df["iceberg_aspect_ratio"] = df["length_m"] / (df["width_m"] + 1e-6)
            df["iceberg_draft_estimate"] = df["length_m"] * 0.9  # Rough estimate

    # --- Cyclical time encoding ---
    if add_cyclical_time:
        df["hour"] = df["datetime"].dt.hour
        df["day_of_year"] = df["datetime"].dt.dayofyear
        df["month"] = df["datetime"].dt.month

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Lag features (per iceberg) ---
    if add_lag_features:
        df = _add_lag_features(df, lag_hours)

    logger.info(f"Feature engineering complete. Columns: {list(df.columns)}")
    # Log feature statistics for debugging (BUG-009)
    feature_cols = [c for c in df.columns if c not in
                   ['datetime', 'lat', 'lon', 'length_m', 'width_m', 'iceberg_area', 'iceberg_aspect_ratio', 'iceberg_draft_estimate']]
    if feature_cols:
        logger.info(f"Feature stats:\n{df[feature_cols].describe().to_string()}")
    return df


def _add_lag_features(
    df: pd.DataFrame,
    lag_hours: List[int],
) -> pd.DataFrame:
    """Add lagged features for each iceberg trajectory."""
    lag_cols = [
        "lat", "lon",
        "wind_u10", "wind_v10", "wind_speed", "wind_dir",
        "current_uo", "current_vo", "current_speed", "current_dir",
        "physics_u", "physics_v", "physics_speed", "physics_dir",
    ]

    # Only use columns that exist
    lag_cols = [c for c in lag_cols if c in df.columns]

    for iceberg_id in df["iceberg_id"].unique():
        mask = df["iceberg_id"] == iceberg_id
        traj = df.loc[mask].copy()

        for lag_h in lag_hours:
            lag_steps = lag_h // 6  # Assuming 6-hourly data
            for col in lag_cols:
                if col in traj.columns:
                    lag_col = f"{col}_lag{lag_h}h"
                    df.loc[mask, lag_col] = traj[col].shift(lag_steps).values

    return df


# =============================================================================
# Target Creation
# =============================================================================

def create_targets(
    df: pd.DataFrame,
    prediction_horizon_hours: int = 24,
    target_type: str = "velocity",
) -> pd.DataFrame:
    """
    Create prediction targets for iceberg drift model.

    Args:
        df: DataFrame with iceberg positions and features (sorted by iceberg_id, datetime)
        prediction_horizon_hours: How many hours ahead to predict
        target_type: "velocity" (delta per hour), "position_delta" (total delta), "position" (absolute)

    Returns:
        DataFrame with target columns added
    """
    df = df.copy()
    df = df.sort_values(["iceberg_id", "datetime"]).reset_index(drop=True)

    horizon_steps = prediction_horizon_hours // 6  # Assuming 6-hourly data

    logger.info(f"Creating targets: horizon={prediction_horizon_hours}h, type={target_type}")

    targets_list = []

    for iceberg_id in df["iceberg_id"].unique():
        mask = df["iceberg_id"] == iceberg_id
        traj = df.loc[mask].copy()

        if len(traj) <= horizon_steps:
            continue

        # Future position
        future_lat = traj["lat"].shift(-horizon_steps)
        future_lon = traj["lon"].shift(-horizon_steps)

        # Current position
        curr_lat = traj["lat"]
        curr_lon = traj["lon"]

        if target_type == "velocity":
            # Velocity in m/s (delta position / time)
            # 1 deg lat ≈ 111 km, 1 deg lon ≈ 111 km * cos(lat)
            lat_dist_m = (future_lat - curr_lat) * 111000
            lon_dist_m = (future_lon - curr_lon) * 111000 * np.cos(np.radians(curr_lat))
            dt_seconds = prediction_horizon_hours * 3600

            traj["target_u"] = lon_dist_m / dt_seconds  # Eastward velocity (m/s)
            traj["target_v"] = lat_dist_m / dt_seconds  # Northward velocity (m/s)

        elif target_type == "position_delta":
            # Total position change in degrees
            traj["target_dlat"] = future_lat - curr_lat
            traj["target_dlon"] = future_lon - curr_lon

        elif target_type == "position":
            # Absolute future position
            traj["target_lat"] = future_lat
            traj["target_lon"] = future_lon

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        # Distance and bearing targets (useful for evaluation)
        traj["target_distance_km"] = np.sqrt(
            ((future_lat - curr_lat) * 111)**2 +
            ((future_lon - curr_lon) * 111 * np.cos(np.radians(curr_lat)))**2
        )
        traj["target_bearing"] = np.degrees(
            np.arctan2(future_lon - curr_lon, future_lat - curr_lat)
        ) % 360

        targets_list.append(traj)

    if not targets_list:
        logger.warning("No valid targets created - trajectories too short")
        return df

    result = pd.concat(targets_list, ignore_index=True)

    # Log target statistics for debugging (BUG-009)
    target_cols = [c for c in result.columns if c.startswith("target_")]
    if target_cols:
        logger.info(f"Target stats:\n{result[target_cols].describe().to_string()}")

    # Filter out near-constant velocity/target sequences (BUG-006)
    # Check for zero-variance in target velocity/position columns
    if target_cols:
        target_std = result.groupby("iceberg_id")[target_cols].transform("std")
        # For velocity targets, check both u and v components
        # For position delta targets, check dlat and dlon
        # For position targets, check lat and lon (but these are less meaningful for variance)
        low_variance_mask = pd.Series(False, index=result.index)

        # Check velocity components if they exist
        if "target_u" in target_std.columns and "target_v" in target_std.columns:
            low_variance_mask = (target_std["target_u"] < 1e-5) & (target_std["target_v"] < 1e-5)
        # Check position delta components if they exist
        elif "target_dlat" in target_std.columns and "target_dlon" in target_std.columns:
            low_variance_mask = (target_std["target_dlat"] < 1e-5) & (target_std["target_dlon"] < 1e-5)
        # Check position components if they exist (less meaningful but still check)
        elif "target_lat" in target_std.columns and "target_lon" in target_std.columns:
            low_variance_mask = (target_std["target_lat"] < 1e-5) & (target_std["target_lon"] < 1e-5)

        if low_variance_mask.any():
            logger.info(f"Dropping {low_variance_mask.sum()} near-constant target rows")
            result = result[~low_variance_mask].reset_index(drop=True)

    # Drop rows with NaN targets (last horizon_steps of each trajectory)
    initial_len = len(result)

    # Also drop rows where any lag feature columns are NaN (to prevent NaN feature leakage)
    lag_feature_cols = [c for c in result.columns if '_lag' in c and c not in target_cols]
    if lag_feature_cols:
        nan_counts = result[lag_feature_cols].isna().sum()
        logger.info(f"NaN counts in lag features: \n{nan_counts[nan_counts > 0]}")
        # Drop rows with NaN in either targets or lag features
        result = result.dropna(subset=target_cols + lag_feature_cols).reset_index(drop=True)
        logger.info(f"Dropped {initial_len - len(result)} rows with NaN targets or lag features")
    else:
        result = result.dropna(subset=target_cols).reset_index(drop=True)
        logger.info(f"Dropped {initial_len - len(result)} rows with NaN targets")

    return result


# =============================================================================
# Data Splitting (Trajectory-aware)
# =============================================================================

def split_trajectories(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_seed: int = 42,
    strategy: str = "trajectory",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by entire trajectories (not random rows) to avoid leakage.

    Args:
        df: DataFrame with iceberg_id column
        train_frac, val_frac, test_frac: Split fractions
        random_seed: For reproducibility
        strategy: "trajectory" (split by iceberg_id) or "time" (split by date)

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    np.random.seed(random_seed)

    if strategy == "trajectory":
        # Split by unique iceberg IDs
        iceberg_ids = df["iceberg_id"].unique()
        np.random.shuffle(iceberg_ids)

        n_train = int(len(iceberg_ids) * train_frac)
        n_val = int(len(iceberg_ids) * val_frac)

        train_ids = set(iceberg_ids[:n_train])
        val_ids = set(iceberg_ids[n_train:n_train + n_val])
        test_ids = set(iceberg_ids[n_train + n_val:])

        train_df = df[df["iceberg_id"].isin(train_ids)].copy()
        val_df = df[df["iceberg_id"].isin(val_ids)].copy()
        test_df = df[df["iceberg_id"].isin(test_ids)].copy()

    elif strategy == "time":
        # Split by time (chronological)
        df = df.sort_values("datetime").reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    logger.info(f"Train icebergs: {train_df['iceberg_id'].nunique()}")
    logger.info(f"Val icebergs: {val_df['iceberg_id'].nunique()}")
    logger.info(f"Test icebergs: {test_df['iceberg_id'].nunique()}")

    return train_df, val_df, test_df


# =============================================================================
# Feature Selection and Scaling
# =============================================================================

def get_feature_columns(df: pd.DataFrame, exclude_prefixes: List[str] = None) -> List[str]:
    """Get list of feature columns (non-target, non-metadata)."""
    if exclude_prefixes is None:
        exclude_prefixes = ["target_", "iceberg_id", "datetime", "lat", "lon"]

    feature_cols = []
    for col in df.columns:
        if not any(col.startswith(p) for p in exclude_prefixes):
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                feature_cols.append(col)
    return feature_cols


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Scale features using training data statistics only.

    Returns scaled DataFrames and the fitted scaler.
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")

    # Fit on training data only
    scaler.fit(train_df[feature_cols])

    # Transform all splits
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if not df.empty:
            df[feature_cols] = scaler.transform(df[feature_cols])

    return train_df, val_df, test_df, scaler