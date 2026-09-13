"""Data download utilities for iceberg drift prediction."""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import xarray as xr
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_chunked_bboxes(iceberg_df: pd.DataFrame, buffer: float = 5.0, max_span: float = 90.0, band_width: float = 30.0) -> List[Tuple[float, float, float, float]]:
    min_lon = iceberg_df['lon'].min()
    max_lon = iceberg_df['lon'].max()
    if max_lon - min_lon <= max_span:
        return [(
            max(-180.0, min_lon - buffer),
            max(-90.0, iceberg_df['lat'].min() - buffer),
            min(180.0, max_lon + buffer),
            min(90.0, iceberg_df['lat'].max() + buffer)
        )]
    bands_info = []
    for band_start in np.arange(-180.0, 180.0, band_width):
        band_end = band_start + band_width
        mask = (iceberg_df['lon'] >= band_start) & (iceberg_df['lon'] < band_end)
        if not mask.any():
            continue
        chunk_df = iceberg_df[mask]
        bands_info.append({
            'band_start': band_start,
            'band_end': band_end,
            'lon_min': chunk_df['lon'].min(),
            'lon_max': chunk_df['lon'].max(),
            'lat_min': chunk_df['lat'].min(),
            'lat_max': chunk_df['lat'].max()
        })
        
    bboxes = []
    for i, b in enumerate(bands_info):
        c_min_lon = max(-180.0, b['lon_min'] - buffer) if i == 0 else b['band_start']
        c_max_lon = min(180.0, b['lon_max'] + buffer) if i == len(bands_info) - 1 else b['band_end']
        
        c_min_lat = max(-90.0, b['lat_min'] - buffer)
        c_max_lat = min(90.0, b['lat_max'] + buffer)
        bboxes.append((float(c_min_lon), float(c_min_lat), float(c_max_lon), float(c_max_lat)))
    return bboxes

def _validate_date_range(ds: xr.Dataset, start_date: str, end_date: str):
    time_coord = ds["time"] if "time" in ds.coords else ds.get("valid_time")
    if time_coord is None:
        return
    ds_min = pd.Timestamp(time_coord.values.min())
    ds_max = pd.Timestamp(time_coord.values.max())
    req_start = pd.Timestamp(start_date)
    req_end = pd.Timestamp(end_date)
    if req_start < ds_min - pd.Timedelta(days=1) or req_end > ds_max + pd.Timedelta(days=1):
        raise ValueError(f"Cached data date range ({ds_min.date()} to {ds_max.date()}) does not cover requested range ({req_start.date()} to {req_end.date()}).")


# =============================================================================
# Iceberg Position Data (US National Ice Center / BYU)
# =============================================================================


def download_iceberg_positions(
    output_dir: str = "data/raw/icebergs",
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    min_length_m: int = 100,
    source: str = "NIC",
    local_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download iceberg position data from NIC or BYU database.

    NIC: National Ice Center weekly iceberg analysis (operational)
    BYU: Brigham Young University long-term iceberg database (research)

    Returns DataFrame with columns:
    - iceberg_id: unique identifier
    - datetime: timestamp
    - lat: latitude (degrees, negative for South)
    - lon: longitude (degrees, -180 to 180)
    - length_m: estimated length (if available)
    - width_m: estimated width (if available)
    - area_km2: estimated area (if available)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_file = output_path / f"icebergs_{source}_{start_date}_{end_date}.parquet"

    if cache_file.exists():
        logger.info(f"Loading cached iceberg data from {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"Downloading {source} iceberg data from {start_date} to {end_date}")

    if source == "NIC":
        df = _download_nic_icebergs(start_date, end_date, min_length_m)
    elif source == "BYU":
        df = _download_byu_icebergs(start_date, end_date, min_length_m)
    elif source == "BYU_LOCAL":
        if not local_path:
            raise ValueError("local_path must be provided for BYU_LOCAL source")
        df = _load_byu_local_icebergs(local_path, start_date, end_date, min_length_m)
    elif source == "SYNTHETIC":
        df = _generate_synthetic_icebergs(start_date, end_date, min_length_m, source="SYNTHETIC")
    elif source == "local":
        df = _load_local_icebergs(output_dir, start_date, end_date, min_length_m)
    else:
        raise ValueError(f"Unknown source: {source}")

    # Filter by minimum length
    if "length_m" in df.columns:
        df = df[df["length_m"] >= min_length_m].copy()

    # Save cache
    df.to_parquet(cache_file, index=False)
    logger.info(f"Saved {len(df)} records to {cache_file}")

    return df


def _load_local_icebergs(
    data_dir: str,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    min_length_m: int = 100,
) -> pd.DataFrame:
    """
    Load iceberg position data from local CSV files.

    Expects CSV files with columns: lat, lon, date, and optional auxiliary columns.
    Combines all CSV files in the given directory.

    Parameters
    ----------
    data_dir : str
        Directory containing CSV files.
    start_date : str
        Start date for filtering (inclusive).
    end_date : str
        End date for filtering (inclusive).
    min_length_m : int
        Minimum iceberg length to keep (if length_m column exists).

    Returns
    -------
    pd.DataFrame
        Combined and filtered DataFrame with standardized columns.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Local data directory not found: {data_dir}")

    # Find all CSV files
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info(f"Loading {len(csv_files)} CSV files from {data_dir}")

    # List to hold dataframes
    df_list = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Ensure required columns exist
            if not {'lat', 'lon', 'date'}.issubset(df.columns):
                logger.warning(f"Skipping {csv_file.name}: missing required columns")
                continue

            # Convert date to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])

            # Rename 'date' column to 'datetime' for consistency with the rest of the pipeline
            df = df.rename(columns={'date': 'datetime'})

            # Filter by date range
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            df = df[(df['datetime'] >= start) & (df['datetime'] <= end)].copy()

            # Add placeholder iceberg_id if not present (using filename as proxy)
            if 'iceberg_id' not in df.columns:
                # Use filename without extension as iceberg_id, but note: this is not ideal
                # Ideally, the data would have a proper iceberg_id column.
                # For now, we'll use the filename as a temporary identifier.
                df['iceberg_id'] = csv_file.stem

            # Ensure length_m column exists for filtering (if not, assume all pass)
            if 'length_m' not in df.columns:
                # If we don't have length data, we cannot filter by min_length_m.
                # We'll log a warning and keep all rows.
                logger.warning(f"No length_m column in {csv_file.name}; skipping length filter")
            else:
                df = df[df['length_m'] >= min_length_m].copy()

            df_list.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")

    if not df_list:
        raise ValueError(f"No valid data loaded from {data_dir}")

    # Combine all dataframes
    combined = pd.concat(df_list, ignore_index=True)

    # Sort by iceberg_id and datetime for consistency
    if 'iceberg_id' in combined.columns and 'datetime' in combined.columns:
        combined = combined.sort_values(['iceberg_id', 'datetime']).reset_index(drop=True)

    logger.info(f"Loaded {len(combined)} total records from local files")
    return combined



def _load_byu_local_icebergs(
    data_dir: str,
    start_date: str,
    end_date: str,
    min_length_m: int,
) -> pd.DataFrame:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Local BYU data directory not found: {data_dir}")

    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info(f"Parsing {len(csv_files)} BYU CSV files from {data_dir}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    records = []

    for csv_file in tqdm(csv_files, desc="Parsing BYU Data"):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            continue
            
        for _, row in df.iterrows():
            try:
                date_str = str(int(row['date']))
                year = int(date_str[:4])
                day = int(date_str[4:])
                dt = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day-1)
            except:
                continue
                
            if dt < start or dt > end:
                continue
                
            length_m, width_m = np.nan, np.nan
            if 'size_1' in row and not pd.isna(row['size_1']) and float(row['size_1']) > 0:
                length_m = float(row['size_1']) * 1852.0
            if 'size_2' in row and not pd.isna(row['size_2']) and float(row['size_2']) > 0:
                width_m = float(row['size_2']) * 1852.0
                
            lat, lon = np.nan, np.nan
            for col in df.columns:
                if col.endswith('_1') and col != 'size_1':
                    prefix = col[:-2]
                    col2 = f"{prefix}_2"
                    if col2 in df.columns:
                        lat_val = float(row[col])
                        lon_val = float(row[col2])
                        if lat_val != 0.0 and lon_val != 0.0 and not np.isnan(lat_val):
                            lat, lon = lat_val, lon_val
                            break
                            
            if not np.isnan(lat) and (np.isnan(length_m) or length_m >= min_length_m):
                records.append({
                    'iceberg_id': csv_file.stem,
                    'datetime': dt,
                    'lat': lat,
                    'lon': lon,
                    'length_m': length_m,
                    'width_m': width_m
                })

    if not records:
        raise ValueError(f"No valid data loaded from {data_dir} for given dates.")

    combined = pd.DataFrame(records)
    combined = combined.sort_values(['iceberg_id', 'datetime']).reset_index(drop=True)

    logger.info(f"Loaded {len(combined)} valid BYU records")
    return combined

def _download_nic_icebergs(
    start_date: str,
    end_date: str,
    min_length_m: int,
) -> pd.DataFrame:
    """
    Download from US National Ice Center (NIC) weekly iceberg reports.

    NIC publishes weekly iceberg position reports at:
    https://www.natice.noaa.gov/pub/icebergs/

    Format: CSV with columns for iceberg ID, lat, lon, length, width, date
    """
    base_url = "https://www.natice.noaa.gov/pub/icebergs/"

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # NIC provides weekly files
    # For full implementation, would need to:
    # 1. List available weekly files
    # 2. Download and parse each CSV
    # 3. Combine into single DataFrame

    logger.warning("NIC download uses synthetic data - implement full NIC parser for production")
    logger.info("To implement: parse weekly CSV files from https://www.natice.noaa.gov/pub/icebergs/")

    return _generate_synthetic_icebergs(start_date, end_date, min_length_m, source="NIC")


def _download_byu_icebergs(
    start_date: str,
    end_date: str,
    min_length_m: int,
) -> pd.DataFrame:
    """
    Download from BYU Antarctic Iceberg Database.

    BYU database: https://www.scp.byu.edu/data/iceberg/
    Provides long-term tracking of large icebergs (>10km)
    Data available as NetCDF or CSV
    """
    logger.warning("BYU download uses synthetic data - implement full BYU parser for production")
    logger.info("To implement: download NetCDF from https://www.scp.byu.edu/data/iceberg/")

    return _generate_synthetic_icebergs(start_date, end_date, min_length_m, source="BYU")


# =============================================================================
# Real Data Downloaders (ERA5, Copernicus)
# =============================================================================


def download_era5_wind(
    output_dir: str = "data/raw/era5",
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    bbox: Tuple[float, float, float, float] = None,
    bboxes: List[Tuple[float, float, float, float]] = None,
    variables: List[str] = None,
    pressure_level: Optional[int] = None,
    allow_synthetic_fallback: bool = True,
) -> xr.Dataset:
    if variables is None:
        variables = ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]
        
    if bboxes is None:
        bboxes = [bbox] if bbox else [(-180, -80, 180, -50)]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_file = output_path / f"era5_wind_{start_date}_{end_date}.nc"

    if cache_file.exists():
        logger.info(f"Loading cached ERA5 data from {cache_file}")
        ds = xr.open_dataset(cache_file)
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        _validate_date_range(ds, start_date, end_date)
        union_min_lon = min(b[0] for b in bboxes)
        union_min_lat = min(b[1] for b in bboxes)
        union_max_lon = max(b[2] for b in bboxes)
        union_max_lat = max(b[3] for b in bboxes)
        lat_slice = slice(union_max_lat, union_min_lat) if ds.latitude.values[0] > ds.latitude.values[-1] else slice(union_min_lat, union_max_lat)
        return ds.sel(latitude=lat_slice, longitude=slice(union_min_lon, union_max_lon))

    logger.info(f"Downloading ERA5 data from {start_date} to {end_date}")

    try:
        import cdsapi
        client = cdsapi.Client()

        chunks = []
        for i, box in enumerate(bboxes):
            chunk_file = output_path / f"era5_wind_{start_date}_{end_date}_chunk{i}.nc"
            request = {
                "product_type": "reanalysis",
                "variable": variables,
                "date": f"{start_date}/{end_date}",
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": [box[3], box[0], box[1], box[2]],  # N, W, S, E
                "data_format": "netcdf",
            }

            if pressure_level:
                request["pressure_level"] = str(pressure_level)
                dataset = "reanalysis-era5-pressure-levels"
            else:
                dataset = "reanalysis-era5-single-levels"

            if not chunk_file.exists():
                logger.info(f"Requesting ERA5 data chunk {i} from CDS...")
                client.retrieve(dataset, request, str(chunk_file))

            ds = xr.open_dataset(chunk_file)
            if "valid_time" in ds.coords:
                ds = ds.rename({"valid_time": "time"})
            chunks.append(ds)
            
        return xr.combine_by_coords(chunks, join="outer") if len(chunks) > 1 else chunks[0]

    except ImportError:
        logger.warning("cdsapi not installed. Install with: pip install cdsapi")
        if not allow_synthetic_fallback: raise
        return _generate_synthetic_era5(start_date, end_date, bboxes[0], variables)
    except Exception as e:
        if not allow_synthetic_fallback:
            raise RuntimeError(f"ERA5 download failed: {e}") from e
        logger.warning(f"ERA5 download failed: {e}. Generating synthetic data.")
        return _generate_synthetic_era5(start_date, end_date, bboxes[0], variables)


def download_copernicus_currents(
    output_dir: str = "data/raw/copernicus",
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    bbox: Tuple[float, float, float, float] = None,
    bboxes: List[Tuple[float, float, float, float]] = None,
    depth_levels: List[float] = None,
    variables: List[str] = None,
    product: str = "GLORYS12",
    allow_synthetic_fallback: bool = True,
) -> xr.Dataset:
    """
    Download ocean current data from Copernicus Marine Service (CMEMS).
    """
    if depth_levels is None:
        depth_levels = [0, 10, 50, 100, 200]
    if variables is None:
        variables = ["uo", "vo", "thetao", "so"]

    if bboxes is None:
        bboxes = [bbox] if bbox else [(-180, -80, 180, -50)]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_file = output_path / f"currents_{product}_{start_date}_{end_date}.nc"

    if cache_file.exists():
        logger.info(f"Loading cached currents data from {cache_file}")
        ds = xr.open_dataset(cache_file)
        _validate_date_range(ds, start_date, end_date)
        union_min_lon = min(b[0] for b in bboxes)
        union_min_lat = min(b[1] for b in bboxes)
        union_max_lon = max(b[2] for b in bboxes)
        union_max_lat = max(b[3] for b in bboxes)
        lat_slice = slice(union_max_lat, union_min_lat) if ds.latitude.values[0] > ds.latitude.values[-1] else slice(union_min_lat, union_max_lat)
        ds = ds.sel(latitude=lat_slice, longitude=slice(union_min_lon, union_max_lon))
        if "depth" in ds.coords:
            ds = ds.sel(depth=slice(0, max(depth_levels)))
        return ds

    logger.info(f"Downloading {product} currents from {start_date} to {end_date}")

    try:
        import copernicusmarine as cm

        product_map = {
            "GLORYS12": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            "GLORYS": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            "HYCOM": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            "GLOBAL_ANALYSIS_FORECAST": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        }
        dataset_id = product_map.get(product, product_map["GLORYS12"])

        chunks = []
        for i, box in enumerate(bboxes):
            chunk_name = f"currents_{product}_{start_date}_{end_date}_chunk{i}.nc"
            chunk_file = output_path / chunk_name
            if not chunk_file.exists():
                cm.subset(
                    dataset_id=dataset_id,
                    variables=variables,
                    start_datetime=start_date,
                    end_datetime=end_date,
                    minimum_longitude=box[0],
                    maximum_longitude=box[2],
                    minimum_latitude=box[1],
                    maximum_latitude=box[3],
                    minimum_depth=min(depth_levels),
                    maximum_depth=max(depth_levels),
                    output_filename=chunk_name,
                    output_directory=str(output_path),
                )
            ds = xr.open_dataset(chunk_file)
            chunks.append(ds)
            
        return xr.combine_by_coords(chunks, join="outer") if len(chunks) > 1 else chunks[0]

    except ImportError as e:
        logger.warning("copernicusmarine not installed.")
        if not allow_synthetic_fallback:
            raise RuntimeError(f"Copernicus download failed: {e}") from e
        return _generate_synthetic_currents(start_date, end_date, bboxes[0], depth_levels, variables)
    except Exception as e:
        if not allow_synthetic_fallback:
            raise RuntimeError(f"Copernicus download failed: {e}") from e
        logger.warning(f"Copernicus download failed: {e}. Generating synthetic data.")
        return _generate_synthetic_currents(start_date, end_date, bboxes[0], depth_levels, variables)
def download_bathymetry(
    output_dir: str = "data/raw/bathymetry",
    bbox: Tuple[float, float, float, float] = (-180, -80, 180, -50),
    resolution_m: int = 500,
    source: str = "IBCSO",
) -> xr.Dataset:
    """
    Download bathymetry data.

    Sources:
    - IBCSO v2: International Bathymetric Chart of the Southern Ocean (500m)
    - GEBCO: General Bathymetric Chart of the Oceans (global, 15 arc-sec)
    - REMA: Reference Elevation Model of Antarctica (ice surface)

    IBCSO: https://www.ibcso.org/
    GEBCO: https://www.gebco.net/
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_file = output_path / f"{source.lower()}_bathymetry_{resolution_m}m.nc"

    if cache_file.exists():
        logger.info(f"Loading cached bathymetry from {cache_file}")
        return xr.open_dataset(cache_file)

    logger.info(f"Downloading {source} bathymetry at {resolution_m}m resolution")

    try:
        if source == "IBCSO":
            # IBCSO v2 available via PANGAEA or direct download
            # For full implementation, would download from PANGAEA
            # DOI: 10.1594/PANGAEA.927442
            logger.warning("IBCSO download not fully implemented - using synthetic data")
            return _generate_synthetic_bathymetry(bbox, resolution_m)

        elif source == "GEBCO":
            # GEBCO available via WMS/WFS or direct NetCDF download
            logger.warning("GEBCO download not fully implemented - using synthetic data")
            return _generate_synthetic_bathymetry(bbox, resolution_m)

        else:
            raise ValueError(f"Unknown bathymetry source: {source}")

    except Exception as e:
        logger.warning(f"Bathymetry download failed: {e}. Generating synthetic data.")
        return _generate_synthetic_bathymetry(bbox, resolution_m)


# =============================================================================
# Utility Functions for Real Data
# =============================================================================


def check_cds_credentials() -> bool:
    """Check if CDS API credentials are configured."""
    cdsapirc = Path.home() / ".cdsapirc"
    if cdsapirc.exists():
        logger.info("CDS API credentials found")
        return True
    logger.warning("CDS API credentials not found at ~/.cdsapirc")
    logger.info("Create ~/.cdsapirc with:")
    logger.info("url: https://cds.climate.copernicus.eu/api/v2")
    logger.info("key: YOUR_UID:YOUR_API_KEY")
    return False


def check_cmems_credentials() -> bool:
    """Check if CMEMS credentials are configured."""
    # CMEMS uses username/password, via environment variables or credentials file
    username = os.environ.get("CMEMS_USERNAME")
    password = os.environ.get("CMEMS_PASSWORD")
    if username and password:
        logger.info("CMEMS credentials found in environment")
        return True
    
    cred_file = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"
    if cred_file.exists():
        logger.info("CMEMS credentials found in ~/.copernicusmarine")
        return True
        
    logger.warning("CMEMS credentials not found in environment variables or ~/.copernicusmarine")
    logger.info("Run `copernicusmarine login` to generate the credentials file")
    return False


def list_available_era5_variables() -> List[str]:
    """List available ERA5 single-level variables."""
    return [
        "u10", "v10", "u100", "v100",  # Wind
        "msl", "sp",  # Pressure
        "t2m", "d2m",  # Temperature
        "tp", "cp", "lsp",  # Precipitation
        "sshf", "slhf",  # Heat fluxes
        "ssr", "str", "ssrd", "strd",  # Radiation
        "sst", "sst_kd",  # SST
        "wave_height", "wave_period",  # Waves
    ]


def list_available_copernicus_products() -> Dict[str, str]:
    """List available Copernicus Marine products for Antarctic region."""
    return {
        "GLORYS12": "Global 1/12° reanalysis (1993-present, daily)",
        "GLORYS": "Global 1/12° reanalysis (alias for GLORYS12)",
        "GLOBAL_ANALYSIS_FORECAST": "Global 1/12° analysis+forecast (daily, 7-day forecast)",
        "HYCOM": "HYCOM 1/12° reanalysis (via CMEMS)",
        "ANTARCTIC_1KM": "Antarctic coastal 1km (limited area)",
    }


def _generate_synthetic_icebergs(
    start_date: str,
    end_date: str,
    min_length_m: int,
    source: str = "SYNTHETIC",
) -> pd.DataFrame:
    """Generate realistic synthetic iceberg trajectories for development/testing."""
    np.random.seed(42)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_range = pd.date_range(start, end, freq="6h")

    # Number of icebergs to simulate
    n_icebergs = 50

    records = []

    for i in range(n_icebergs):
        iceberg_id = f"{source}_{i:04d}"

        # Initial position (Antarctic coastal region)
        lat = np.random.uniform(-75, -60)
        lon = np.random.uniform(-180, 180)

        # Iceberg size (log-normal distribution)
        length_m = np.random.lognormal(mean=np.log(2000), sigma=0.5)
        length_m = max(length_m, min_length_m)
        width_m = length_m * np.random.uniform(0.3, 0.8)

        # Simulate trajectory with physics-based drift
        trajectory = _simulate_iceberg_trajectory(
            lat, lon, length_m, width_m, date_range
        )

        for t_idx, (t, t_lat, t_lon) in enumerate(trajectory):
            records.append({
                "iceberg_id": iceberg_id,
                "datetime": t,
                "lat": t_lat,
                "lon": t_lon,
                "length_m": length_m,
                "width_m": width_m,
                "area_km2": (length_m * width_m) / 1e6,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["iceberg_id", "datetime"]).reset_index(drop=True)

    return df


def _simulate_iceberg_trajectory(
    init_lat: float,
    init_lon: float,
    length_m: float,
    width_m: float,
    timestamps: pd.DatetimeIndex,
) -> List[Tuple[pd.Timestamp, float, float]]:
    """Simple physics-based trajectory simulation for synthetic data."""
    # Simplified drift: ~2% wind + ~100% current + Coriolis
    trajectory = []
    lat, lon = init_lat, init_lon

    for t in timestamps:
        trajectory.append((t, lat, lon))

        # Simplified drift model
        # In reality, this would use actual wind/current data
        dt_hours = 6

        # Simulated environmental forcing
        wind_u = np.random.normal(0, 5)  # m/s
        wind_v = np.random.normal(0, 5)
        curr_u = np.random.normal(0, 0.2)
        curr_v = np.random.normal(0, 0.2)

        # Drift velocity (m/s)
        # Water drag dominates: ~100% current + ~2% wind
        u_drift = curr_u + 0.02 * wind_u
        v_drift = curr_v + 0.02 * wind_v

        # Coriolis effect (Southern Hemisphere: deflects left)
        f = 2 * 7.2921e-5 * np.sin(np.radians(lat))  # Coriolis parameter
        u_coriolis = -f * v_drift * dt_hours * 3600
        v_coriolis = f * u_drift * dt_hours * 3600

        # Convert to lat/lon change
        # 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
        lat += (v_drift + v_coriolis) * dt_hours * 3600 / 111000
        lon += (u_drift + u_coriolis) * dt_hours * 3600 / (111000 * np.cos(np.radians(lat)))

        # Keep in reasonable bounds
        lat = np.clip(lat, -80, -50)

    return trajectory


# =============================================================================
# ERA5 Wind Data
# =============================================================================


def _generate_synthetic_era5(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    variables: List[str],
) -> xr.Dataset:
    """Generate synthetic ERA5-like data for development."""
    np.random.seed(42)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    times = pd.date_range(start, end, freq="6h")

    lats = np.arange(bbox[1], bbox[3], 0.25)  # 0.25 deg resolution
    lons = np.arange(bbox[0], bbox[2], 0.25)

    data_vars = {}
    for var in variables:
        if var in ["u10", "v10"]:
            # Wind: typical Southern Ocean patterns
            data = np.random.normal(0, 8, size=(len(times), len(lats), len(lons)))
        elif var == "msl":
            # Mean sea level pressure
            data = np.random.normal(1013, 10, size=(len(times), len(lats), len(lons)))
        else:
            data = np.random.normal(0, 1, size=(len(times), len(lats), len(lons)))

        data_vars[var] = (["time", "latitude", "longitude"], data)

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.attrs["source"] = "synthetic"

    return ds


# =============================================================================
# Copernicus Marine Service Ocean Currents
# =============================================================================


def _generate_synthetic_currents(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    depth_levels: List[float],
    variables: List[str],
) -> xr.Dataset:
    """Generate synthetic ocean current data for development."""
    np.random.seed(123)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    times = pd.date_range(start, end, freq="D")  # Daily

    lats = np.arange(bbox[1], bbox[3], 0.08)  # ~8km resolution
    lons = np.arange(bbox[0], bbox[2], 0.08)

    data_vars = {}
    for var in variables:
        if var in ["uo", "vo"]:
            # Currents: typically 0.1-0.3 m/s in Southern Ocean
            data = np.random.normal(0, 0.15, size=(len(times), len(depth_levels), len(lats), len(lons)))
        elif var == "temperature":
            data = np.random.normal(-1, 2, size=(len(times), len(depth_levels), len(lats), len(lons)))
        elif var == "salinity":
            data = np.random.normal(34.5, 0.2, size=(len(times), len(depth_levels), len(lats), len(lons)))
        else:
            data = np.random.normal(0, 1, size=(len(times), len(depth_levels), len(lats), len(lons)))

        data_vars[var] = (["time", "depth", "latitude", "longitude"], data)

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "depth": depth_levels,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.attrs["source"] = "synthetic"

    return ds


# =============================================================================
# Bathymetry (IBCSO)
# =============================================================================


def _generate_synthetic_bathymetry(
    bbox: Tuple[float, float, float, float],
    resolution_m: int,
) -> xr.Dataset:
    """Generate synthetic bathymetry for development."""
    np.random.seed(456)

    # Resolution in degrees
    res_deg = resolution_m / 111000

    lats = np.arange(bbox[1], bbox[3], res_deg)
    lons = np.arange(bbox[0], bbox[2], res_deg)

    # Synthetic bathymetry: deep ocean (~4000m) with continental shelf near coast
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Distance from Antarctic coast (approximate)
    dist_from_pole = 90 + lat_grid  # degrees from South Pole
    dist_km = dist_from_pole * 111

    # Depth increases with distance from coast
    depth = -np.where(
        dist_km < 500,
        500 + dist_km * 5,  # Continental shelf/slope
        3000 + np.random.normal(0, 500, size=lat_grid.shape)  # Abyssal plain
    )

    ds = xr.Dataset(
        {
            "bathymetry": (["latitude", "longitude"], depth),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.attrs["source"] = "synthetic"
    ds["bathymetry"].attrs["units"] = "meters"
    ds["bathymetry"].attrs["positive"] = "down"

    return ds


# =============================================================================
# Update models/__init__.py exports
# =============================================================================

__all__ = [
    "download_iceberg_positions",
    "download_era5_wind",
    "download_copernicus_currents",
    "download_bathymetry",
    "check_cds_credentials",
    "check_cmems_credentials",
    "list_available_era5_variables",
    "list_available_copernicus_products",
    "_generate_synthetic_icebergs",
    "_generate_synthetic_era5",
    "_generate_synthetic_currents",
    "_generate_synthetic_bathymetry",
]