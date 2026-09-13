"""Data processing module for iceberg drift prediction."""

from .download import (
    download_iceberg_positions,
    download_era5_wind,
    download_copernicus_currents,
    download_bathymetry,
)
from .preprocessing import (
    match_environmental_data,
    compute_coriolis,
    engineer_features,
    create_targets,
    split_trajectories,
    get_feature_columns,
    scale_features,
)
from .dataset import IcebergDriftDataset, create_dataloaders
from .quality import (
    check_iceberg_data_quality,
    resample_iceberg_trajectory,
    filter_coastal_nan,
    fill_coastal_current_nan,
    check_environmental_data_quality,
    DataQualityPipeline,
    validate_and_clean_iceberg_data,
    create_quality_report,
)

__all__ = [
    "download_iceberg_positions",
    "download_era5_wind",
    "download_copernicus_currents",
    "download_bathymetry",
    "match_environmental_data",
    "compute_coriolis",
    "engineer_features",
    "create_targets",
    "split_trajectories",
    "get_feature_columns",
    "scale_features",
    "IcebergDriftDataset",
    "create_dataloaders",
    # Quality
    "check_iceberg_data_quality",
    "resample_iceberg_trajectory",
    "filter_coastal_nan",
    "fill_coastal_current_nan",
    "check_environmental_data_quality",
    "DataQualityPipeline",
    "validate_and_clean_iceberg_data",
    "create_quality_report",
]