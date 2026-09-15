"""Data quality checks and filters for iceberg drift prediction."""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# =============================================================================
# Iceberg Data Quality Checks
# =============================================================================

def check_iceberg_data_quality(
    df: pd.DataFrame,
    min_observations: int = 10,
    max_gap_hours: int = 48,
    max_speed_kmh: float = 50.0,
    coord_bounds: Tuple[float, float, float, float] = (-180, -80, 180, -50),
) -> Dict[str, Any]:
    """
    Comprehensive quality check for iceberg position data.

    Args:
        df: DataFrame with iceberg positions [iceberg_id, datetime, lat, lon, ...]
        min_observations: Minimum observations per iceberg to keep
        max_gap_hours: Maximum allowed gap between observations (hours)
        max_speed_kmh: Maximum plausible iceberg speed (km/h)
        coord_bounds: (min_lon, min_lat, max_lon, max_lat) valid bounds

    Returns:
        Dict with quality report and filtered DataFrame
    """
    report = {
        "initial_records": len(df),
        "initial_icebergs": df["iceberg_id"].nunique(),
        "issues_found": [],
        "removed_icebergs": [],
        "removed_records": 0,
    }

    df = df.copy()
    df = df.sort_values(["iceberg_id", "datetime"]).reset_index(drop=True)

    min_lon, min_lat, max_lon, max_lat = coord_bounds

    # 1. Check coordinate bounds
    lat_oob = (df["lat"] < min_lat) | (df["lat"] > max_lat)
    lon_oob = (df["lon"] < min_lon) | (df["lon"] > max_lon)
    oob_count = (lat_oob | lon_oob).sum()
    if oob_count > 0:
        report["issues_found"].append(f"Out of bounds coordinates: {oob_count} records")
        df = df[~(lat_oob | lon_oob)].copy()

    # 2. Check for duplicate timestamps per iceberg
    dup_mask = df.duplicated(subset=["iceberg_id", "datetime"], keep=False)
    dup_count = dup_mask.sum()
    if dup_count > 0:
        report["issues_found"].append(f"Duplicate timestamps: {dup_count} records")
        # Keep first occurrence
        df = df.drop_duplicates(subset=["iceberg_id", "datetime"], keep="first")

    # 3. Check for reversed timestamps (datetime not monotonic per iceberg)
    reversed_count = 0
    for iceberg_id in df["iceberg_id"].unique():
        mask = df["iceberg_id"] == iceberg_id
        traj = df.loc[mask]
        if not traj["datetime"].is_monotonic_increasing:
            reversed_count += 1
    if reversed_count > 0:
        report["issues_found"].append(f"Non-monotonic timestamps: {reversed_count} icebergs")
        # Re-sort
        df = df.sort_values(["iceberg_id", "datetime"]).reset_index(drop=True)

    # 4. Check for large gaps in observations
    gap_icebergs = []
    for iceberg_id in df["iceberg_id"].unique():
        mask = df["iceberg_id"] == iceberg_id
        traj = df.loc[mask]
        time_diffs = traj["datetime"].diff().dt.total_seconds() / 3600  # hours
        max_gap = time_diffs.max()
        if max_gap > max_gap_hours:
            gap_icebergs.append(iceberg_id)
    if gap_icebergs:
        report["issues_found"].append(f"Large time gaps (>{max_gap_hours}h): {len(gap_icebergs)} icebergs")
        report["removed_icebergs"].extend(gap_icebergs)

    # 5. Check for implausible speeds (iceberg teleportation)
    speed_icebergs = []
    for iceberg_id in df["iceberg_id"].unique():
        mask = df["iceberg_id"] == iceberg_id
        traj = df.loc[mask]
        if len(traj) > 1:
            dlat = traj["lat"].diff() * 111  # km
            dlon = traj["lon"].diff() * 111 * np.cos(np.radians(traj["lat"]))  # km
            dt_hours = traj["datetime"].diff().dt.total_seconds() / 3600
            speeds = np.sqrt(dlat**2 + dlon**2) / dt_hours  # km/h
            max_speed = speeds.max()
            if max_speed > max_speed_kmh:
                speed_icebergs.append(iceberg_id)
    if speed_icebergs:
        report["issues_found"].append(f"Implausible speeds (>{max_speed_kmh} km/h): {len(speed_icebergs)} icebergs")
        report["removed_icebergs"].extend(speed_icebergs)

    # 6. Filter icebergs with too few observations
    obs_counts = df.groupby("iceberg_id").size()
    valid_icebergs = obs_counts[obs_counts >= min_observations].index.tolist()
    removed_short = [ib for ib in df["iceberg_id"].unique() if ib not in valid_icebergs]
    if removed_short:
        report["issues_found"].append(f"Too few observations (<{min_observations}): {len(removed_short)} icebergs")
        report["removed_icebergs"].extend(removed_short)

    # Remove flagged icebergs
    all_removed = set(report["removed_icebergs"])
    df = df[~df["iceberg_id"].isin(all_removed)].copy()

    report["final_records"] = len(df)
    report["final_icebergs"] = df["iceberg_id"].nunique()
    report["removed_records"] = report["initial_records"] - report["final_records"]
    report["removed_icebergs"] = list(set(report["removed_icebergs"]))

    logger.info(f"Quality check: {report['initial_icebergs']} -> {report['final_icebergs']} icebergs "
               f"({report['removed_records']} records removed)")

    return {"data": df, "report": report}


def resample_iceberg_trajectory(
    df: pd.DataFrame,
    freq: str = "6h",
    interpolation: str = "linear",
    max_gap: str = "24h",
) -> pd.DataFrame:
    """
    Resample irregular iceberg observations to regular time grid.

    Args:
        df: DataFrame with iceberg positions [iceberg_id, datetime, lat, lon, ...]
        freq: Target frequency (e.g., "6h", "1d")
        interpolation: "linear", "time", "nearest"
        max_gap: Maximum gap to interpolate (larger gaps become NaN)

    Returns:
        Resampled DataFrame with regular time steps
    """
    if df.empty or df["iceberg_id"].unique().size == 0:
        import logging
        logging.getLogger(__name__).warning("No icebergs survived quality filtering for this period - returning empty result")
        return pd.DataFrame(columns=df.columns)
        
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    resampled_list = []

    for iceberg_id in df["iceberg_id"].unique():
        mask = df["iceberg_id"] == iceberg_id
        traj = df.loc[mask].copy().set_index("datetime")

        # Select numeric columns to resample
        numeric_cols = traj.select_dtypes(include=[np.number]).columns.tolist()

        # Resample
        traj_resampled = traj[numeric_cols].resample(freq).interpolate(method=interpolation)

        # Limit interpolation to max_gap
        if max_gap:
            traj_resampled = traj_resampled.interpolate(method=interpolation, limit=int(pd.Timedelta(max_gap) / pd.Timedelta(freq)))

        # Add iceberg_id back
        traj_resampled["iceberg_id"] = iceberg_id
        resampled_list.append(traj_resampled.reset_index())

    result = pd.concat(resampled_list, ignore_index=True)
    result = result.sort_values(["iceberg_id", "datetime"]).reset_index(drop=True)

    logger.info(f"Resampled to {freq}: {len(result)} records from {len(df)}")
    return result


def filter_coastal_nan(
    df: pd.DataFrame,
    bathymetry_ds: xr.Dataset,
    depth_threshold: float = 100.0,
    buffer_km: float = 50.0,
) -> pd.DataFrame:
    """
    Filter or flag iceberg observations in shallow/coastal waters where
    current data may be unreliable.

    Args:
        df: Iceberg DataFrame with lat, lon
        bathymetry_ds: xarray Dataset with bathymetry (depth > 0 for ocean)
        depth_threshold: Minimum depth (m) to consider valid
        buffer_km: Buffer distance from coast (km)

    Returns:
        DataFrame with 'coastal_flag' column added
    """
    df = df.copy()

    # Interpolate bathymetry at iceberg positions
    if "bathymetry" not in bathymetry_ds:
        logger.warning("No bathymetry variable in dataset")
        df["coastal_flag"] = False
        return df

    bath = bathymetry_ds["bathymetry"]

    lat_arr = bath.latitude.values if hasattr(bath, "latitude") else bath.lat.values
    lon_arr = bath.longitude.values if hasattr(bath, "longitude") else bath.lon.values

    # Create interpolator
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (lat_arr, lon_arr),
        bath.values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Query depths
    points = np.column_stack([df["lat"].values, df["lon"].values])
    depths = interp(points)

    # Flag shallow/coastal
    df["bathymetry_depth"] = depths
    df["coastal_flag"] = (df["bathymetry_depth"] < depth_threshold) | df["bathymetry_depth"].isna()

    n_coastal = df["coastal_flag"].sum()
    logger.info(f"Coastal/shallow flag: {n_coastal}/{len(df)} records")

    return df


def fill_coastal_current_nan(
    current_ds: xr.Dataset,
    method: str = "nearest_valid",
    max_distance_km: float = 100.0,
    max_depth_m: float = 20.0,
) -> xr.Dataset:
    """
    Fill NaN values in coastal current data using nearest valid ocean point.

    Args:
        current_ds: xarray Dataset with uo, vo variables (time, depth, lat, lon)
        method: "nearest_valid" or "spatial_interpolation"
        max_distance_km: Maximum distance to search for valid point
        max_depth_m: Maximum depth level to consider (meters) - only fill this depth range

    Returns:
        Dataset with filled NaN values
    """
    import logging
    logger = logging.getLogger(__name__)

    ds = current_ds

    # Restrict to only the depth levels actually needed (BUG-013 fix)
    # Check what depth coordinate name this dataset uses
    depth_coord = None
    for coord_name in ['depth', 'z_l', 'z']:
        if coord_name in ds.coords:
            depth_coord = coord_name
            break

    if depth_coord is not None:
        # Subset to only the depth range we need BEFORE copying/processing
        depth_values = ds[depth_coord].values
        # Only keep depths <= max_depth_m (assuming positive down convention)
        depth_mask = depth_values <= max_depth_m
        if depth_mask.any():
            ds = ds.sel({depth_coord: slice(0, max_depth_m)})
            logger.info(f"Subset to depth range 0-{max_depth_m}m using coord '{depth_coord}'")
        else:
            logger.warning(f"No depth levels <= {max_depth_m}m found in {depth_coord}, using all levels")

    ds = ds.copy()

    for var in ["uo", "vo"]:
        if var not in ds:
            continue

        data = ds[var].values  # (time, depth, lat, lon)
        nans = np.isnan(data)

        if not nans.any():
            continue

        # BUG-013 fix: Add hard size guard
        total_cells = data.size
        if total_cells > 5_000_000:  # threshold for unusually large grid
            logger.warning(
                f"Current grid has {total_cells:,} cells — this is unusually large "
                f"for a regional/surface dataset and may indicate the cached file "
                f"wasn't properly cropped. Fill will be slow; consider re-cropping "
                f"the source file to a smaller bbox/depth range."
            )

        logger.info(f"Filling {nans.sum()} NaN values in {var}")

        if method == "nearest_valid":
            # For each time/depth slice, fill NaN with nearest valid spatial point
            from scipy.spatial import cKDTree
            
            # Assume land mask is static across time/depth for ocean models
            ref_slice = data[0, 0, :, :]
            valid_mask = ~np.isnan(ref_slice)
            invalid_mask = np.isnan(ref_slice)
            
            if invalid_mask.any() and valid_mask.any():
                lat_idx, lon_idx = np.meshgrid(
                    np.arange(data.shape[2]),
                    np.arange(data.shape[3]),
                    indexing="ij",
                )
                
                valid_coords = np.column_stack([lat_idx[valid_mask], lon_idx[valid_mask]])
                invalid_coords = np.column_stack([lat_idx[invalid_mask], lon_idx[invalid_mask]])
                
                # Build tree ONCE
                tree = cKDTree(valid_coords)
                _, indices = tree.query(invalid_coords)
                
                # Apply to all time and depth slices
                for t in range(data.shape[0]):
                    if t % 5 == 0:
                        logger.info(f"  Coastal fill progress: time step {t}/{data.shape[0]}")
                    for d in range(data.shape[1]):
                        slice_data = data[t, d, :, :]
                        # valid_vals might contain NaNs if the mask is time-varying, but we assume static land mask
                        valid_vals = slice_data[valid_mask]
                        slice_data[invalid_mask] = valid_vals[indices]
                        data[t, d, :, :] = slice_data

        elif method == "spatial_interpolation":
            # Use xarray's interpolate_na (requires scipy)
            ds[var] = ds[var].interpolate_na(dim="latitude", method="linear")
            ds[var] = ds[var].interpolate_na(dim="longitude", method="linear")

        else:
            raise ValueError(f"Unknown fill method: {method}")

        ds[var].values = data

    return ds


# =============================================================================
# Environmental Data Quality
# =============================================================================

class DataQualityError(Exception):
    """Exception raised when environmental data quality checks fail."""
    pass


def check_environmental_data_quality(
    wind_ds: xr.Dataset,
    current_ds: xr.Dataset,
    check_dims: Tuple[str, ...] = ("time", "latitude", "longitude"),
) -> Dict[str, Any]:
    """
    Check quality of wind and current datasets.

    Args:
        wind_ds: Wind dataset
        current_ds: Current dataset
        check_dims: Dimensions to check for consistency

    Returns:
        Quality report

    Raises:
        DataQualityError: If data quality checks fail badly enough to be untrustworthy
    """
    report = {"wind": {}, "current": {}}

    for name, ds in [("wind", wind_ds), ("current", current_ds)]:
        report[name]["variables"] = list(ds.data_vars)
        report[name]["dims"] = dict(ds.dims)
        report[name]["time_range"] = (
            str(ds.time.min().values),
            str(ds.time.max().values),
        )
        report[name]["spatial_bounds"] = {
            "lat": [float(ds.latitude.min()), float(ds.latitude.max())],
            "lon": [float(ds.longitude.min()), float(ds.longitude.max())],
        }

        # Check for NaN.
        # ds[var].values materialises the whole array: a full-year circumpolar
        # GLORYS file is ~2.1 GB per variable (569M cells), so counting NaN this
        # way across uo/vo/thetao/so needs ~8.5 GB and gets the process OOM-killed.
        # Accumulate over slices of the leading (time) dimension instead.
        for var in ds.data_vars:
            da = ds[var]
            total = da.size
            if total == 0:
                report[name][f"{var}_nan_pct"] = 0.0
                continue
            lead = da.dims[0] if da.ndim else None
            n = da.sizes[lead] if lead else 0
            if lead is None or n == 0:
                nan_count = int(np.isnan(da.values).sum())
            else:
                # cap each slab at roughly 128 MB of float32
                per = max(1, int(128 * 1024**2 / max(1, (total // n) * 4)))
                nan_count = 0
                for i in range(0, n, per):
                    blk = da.isel({lead: slice(i, i + per)}).values
                    nan_count += int(np.isnan(blk).sum())
                    del blk
            report[name][f"{var}_nan_pct"] = 100 * nan_count / total

    # Check temporal alignment (overlap of ranges)
    wind_times = wind_ds.time.values
    current_times = current_ds.time.values
    
    if len(wind_times) > 0 and len(current_times) > 0:
        wind_min, wind_max = wind_times.min(), wind_times.max()
        curr_min, curr_max = current_times.min(), current_times.max()
        
        overlap_min = max(wind_min, curr_min)
        overlap_max = min(wind_max, curr_max)
        
        if overlap_max >= overlap_min:
            wind_dur = float((wind_max - wind_min) / np.timedelta64(1, 's'))
            curr_dur = float((curr_max - curr_min) / np.timedelta64(1, 's'))
            overlap_dur = float((overlap_max - overlap_min) / np.timedelta64(1, 's'))
            max_dur = max(wind_dur, curr_dur)
            overlap_pct = 100 * overlap_dur / max_dur if max_dur > 0 else 100.0
        else:
            overlap_pct = 0.0
    else:
        overlap_pct = 0.0

    report["temporal_overlap"] = {
        "wind_times": len(wind_times),
        "current_times": len(current_times),
        "overlap_pct": overlap_pct,
    }

    # BUG-007 fix: Add range and variance checks
    # Define physical ranges for key variables
    variable_ranges = {
        "uo": (-5, 5),      # ocean current zonal velocity (m/s)
        "vo": (-5, 5),      # ocean current meridional velocity (m/s)
        "u10": (-60, 60),   # 10m wind zonal velocity (m/s)
        "v10": (-60, 60),   # 10m wind meridional velocity (m/s)
    }

    # Check each dataset for range violations and zero variance
    for name, ds in [("wind", wind_ds), ("current", current_ds)]:
        for var, (lo, hi) in variable_ranges.items():
            if var in ds.data_vars:
                vals = ds[var].values
                # Check for values outside physical range
                out_of_range_mask = (vals < lo) | (vals > hi)
                out_of_range_pct = out_of_range_mask.mean() * 100
                if out_of_range_pct > 5.0:  # More than 5% out of range is suspicious
                    raise DataQualityError(
                        f"{name}.{var}: {out_of_range_pct:.1f}% of values outside physical range [{lo}, {hi}]"
                    )
                # Check for zero variance (data may be corrupted or all-default)
                if np.nanstd(vals) < 1e-6:
                    raise DataQualityError(
                        f"{name}.{var}: zero variance — data may be corrupted or all-default"
                    )

    return report


# =============================================================================
# Data Validation Pipeline
# =============================================================================

class DataQualityPipeline:
    """Complete data quality pipeline for iceberg drift data."""

    def __init__(
        self,
        min_observations: int = 10,
        max_gap_hours: int = 48,
        max_speed_kmh: float = 50.0,
        resample_freq: str = "6h",
        coastal_depth_threshold: float = 100.0,
        fill_coastal_nan: bool = True,
    ):
        self.min_observations = min_observations
        self.max_gap_hours = max_gap_hours
        self.max_speed_kmh = max_speed_kmh
        self.resample_freq = resample_freq
        self.coastal_depth_threshold = coastal_depth_threshold
        self.fill_coastal_nan = fill_coastal_nan

        self.reports = {}

    def run(
        self,
        iceberg_df: pd.DataFrame,
        wind_ds: Optional[xr.Dataset] = None,
        current_ds: Optional[xr.Dataset] = None,
        bathymetry_ds: Optional[xr.Dataset] = None,
    ) -> Tuple[pd.DataFrame, Optional[xr.Dataset], Optional[xr.Dataset]]:
        """
        Run complete quality pipeline.

        Returns:
            (cleaned_iceberg_df, cleaned_wind_ds, cleaned_current_ds)
        """
        logger.info("=" * 60)
        logger.info("DATA QUALITY PIPELINE")
        logger.info("=" * 60)

        # 1. Iceberg data quality
        logger.info("Step 1: Iceberg data quality checks")
        result = check_iceberg_data_quality(
            iceberg_df,
            min_observations=self.min_observations,
            max_gap_hours=self.max_gap_hours,
            max_speed_kmh=self.max_speed_kmh,
        )
        self.reports["iceberg_quality"] = result["report"]
        iceberg_df = result["data"]

        # 2. Resample to regular grid
        logger.info(f"Step 2: Resampling to {self.resample_freq}")
        iceberg_df = resample_iceberg_trajectory(
            iceberg_df,
            freq=self.resample_freq,
            interpolation="time",
            max_gap=f"{self.max_gap_hours}h",
        )

        # 3. Coastal flagging
        if bathymetry_ds is not None:
            logger.info("Step 3: Coastal/shallow water flagging")
            iceberg_df = filter_coastal_nan(
                iceberg_df,
                bathymetry_ds,
                depth_threshold=self.coastal_depth_threshold,
            )

        # 4. Environmental data quality
        if wind_ds is not None and current_ds is not None:
            logger.info("Step 4: Environmental data quality")
            env_report = check_environmental_data_quality(wind_ds, current_ds)
            self.reports["environmental_quality"] = env_report

            # Fill coastal NaN in currents
            if self.fill_coastal_nan:
                logger.info("Step 5: Filling coastal NaN in currents")
                current_ds = fill_coastal_current_nan(current_ds)

        logger.info("Data quality pipeline complete")
        return iceberg_df, wind_ds, current_ds

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all quality reports."""
        return self.reports


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_and_clean_iceberg_data(
    df: pd.DataFrame,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function for iceberg data validation."""
    result = check_iceberg_data_quality(df, **kwargs)
    return result["data"], result["report"]


def create_quality_report(
    iceberg_df: pd.DataFrame,
    wind_ds: Optional[xr.Dataset] = None,
    current_ds: Optional[xr.Dataset] = None,
) -> Dict[str, Any]:
    """Generate comprehensive quality report."""
    pipeline = DataQualityPipeline()
    pipeline.run(iceberg_df, wind_ds, current_ds)
    return pipeline.get_summary()