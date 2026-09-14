#!/usr/bin/env python

import sys
import os
from pathlib import Path
import logging
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceberg_drift.config import load_config
from iceberg_drift.data_processing.quality import DataQualityPipeline
from iceberg_drift.evaluation.validator import DriftValidator, ValidationConfig
from iceberg_drift.utils import load_model_for_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    import numpy as np
    import pandas as pd
    logger.info("Starting NIC dataset evaluation...")
    config = load_config("config/settings.yaml")
    
    # 2. Load the NIC iceberg data
    import pandas as pd
    nic_file = "/Users/pratiksmac/Downloads/data/processed/icebergs_NIC_2024-01-01_2026-09-14.parquet"
    logger.info(f"Loading NIC validation set: {nic_file}")
    df_nic = pd.read_parquet(nic_file)
    
    # Map raw NIC columns to standard pipeline schema
    df_nic = df_nic.rename(columns={
        'iceberg': 'iceberg_id',
        'latitude': 'lat',
        'longitude': 'lon',
        'last_update': 'datetime'
    })
    df_nic['datetime'] = pd.to_datetime(df_nic['datetime'], utc=True)
    if 'length_nm' in df_nic.columns:
        df_nic['length_m'] = df_nic['length_nm'].astype(float) * 1852.0
        df_nic['width_m'] = df_nic['width_nm'].astype(float) * 1852.0
    
    # Filter to 2024 to match our environmental data
    df_nic = df_nic[(df_nic['datetime'] >= '2024-01-01') & (df_nic['datetime'] <= '2024-12-31')]
    
    # Interpolate to 1D just like the training data to ensure we pass max_gap_hours=48
    resampled_dfs = []
    for iceberg_id, group in df_nic.groupby('iceberg_id'):
        group = group.set_index('datetime').sort_index()
        group = group[~group.index.duplicated(keep='first')]
        if len(group) > 1:
            numeric_cols = group.select_dtypes(include='number').columns
            group_num = group[numeric_cols].resample('1D').interpolate(method='time')
            non_num_cols = group.select_dtypes(exclude='number').columns
            group_non_num = group[non_num_cols].resample('1D').ffill()
            group = pd.concat([group_num, group_non_num], axis=1)
            group['iceberg_id'] = iceberg_id
        resampled_dfs.append(group.reset_index())
    df_nic = pd.concat(resampled_dfs, ignore_index=True)
    df_nic = df_nic.dropna(subset=['lat', 'lon'])
    
    logger.info(f"Filtered NIC data to 2024 (interpolated): {len(df_nic)} records remaining")
    
    # 3. Load environmental data
    logger.info("Loading environmental data...")
    import xarray as xr
    wind_ds = xr.open_dataset("/Users/pratiksmac/Downloads/data/era5_wind_2024-01-01_2024-12-31.nc")
    current_ds = xr.open_dataset("/Users/pratiksmac/Downloads/data/currents_GLORYS12_2024-01-01_2024-12-31.nc")
    
    # 4. Run DataQualityPipeline
    quality_pipeline = DataQualityPipeline(
        min_observations=10,
        max_gap_hours=48,
        max_speed_kmh=50.0,
        resample_freq="6h",
        coastal_depth_threshold=100.0,
        fill_coastal_nan=True,
    )
    
    bathy_cfg = config.data_sources.get('bathymetry', {})
    bathy_path = bathy_cfg.get('local_path')
    bathymetry_ds = None
    if bathy_path and os.path.exists(bathy_path):
        import xarray as xr
        bathymetry_ds = xr.open_dataset(bathy_path)
        if 'z' in bathymetry_ds:
            bathymetry_ds = bathymetry_ds.rename({'z': 'bathymetry'})
        logger.info(f"Loaded bathymetry from {bathy_path}")
        
    df_clean, wind_ds, current_ds = quality_pipeline.run(
        df_nic, wind_ds, current_ds, bathymetry_ds=bathymetry_ds
    )
    logger.info(f"Quality pipeline complete. Records remaining: {len(df_clean)}")
    if len(df_clean) == 0:
        logger.error("No icebergs survived the quality pipeline!")
        return
        
    from iceberg_drift.data_processing.preprocessing import (
        match_environmental_data, 
        engineer_features, 
        create_targets,
        get_feature_columns
    )
    # 5. Match and engineer features
    logger.info("Matching environmental data...")
    df_matched = match_environmental_data(df_clean, wind_ds, current_ds)
    logger.info(f"Matched shape: {df_matched.shape}")
    
    logger.info("Engineering features...")
    df_features = engineer_features(df_matched, add_cyclical_time=True, add_physics_features=True, add_lag_features=True)
    logger.info(f"Features shape: {df_features.shape}")
    
    logger.info("Creating targets...")
    prediction_horizon_hours = config.model.ml_correction.prediction_horizon * 6
    # Rebuild targets using the same parameters as training
    targets_df = create_targets(
        df_features,
        prediction_horizon_hours=config.model.ml_correction.prediction_horizon * 6,
        target_type="velocity",
    )
    
    # Get feature map and model from checkpoint
    model_path = "output/checkpoints/pinn_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    feature_idx_map = checkpoint.get("feature_idx_map") or checkpoint.get("model_config", {}).get("feature_idx_map")
    scaler = checkpoint.get("scaler") or checkpoint.get("model_config", {}).get("scaler")
    
    feature_cols = checkpoint.get("feature_cols") or checkpoint.get("model_config", {}).get("feature_cols")
    if not feature_cols:
        feature_cols = get_feature_columns(df_features)
        
    for col in feature_cols:
        if col not in df_features.columns:
            df_features[col] = 0.0
            
    hidden_dim = checkpoint["model_state_dict"]["ml_model.encoder.weight_ih_l0"].shape[0] // 4
    from iceberg_drift.models.pinn import PINNDriftModel
    model = PINNDriftModel(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        output_dim=2,
        prediction_horizon=config.model.ml_correction.prediction_horizon,
        feature_idx_map=feature_idx_map,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Create dataset and loader
    from iceberg_drift.data_processing.dataset import IcebergDriftDataset, iceberg_drift_collate_fn
    from torch.utils.data import DataLoader
    
    df_eval = targets_df.copy()
    
    # Ensure missing columns exist in df_eval (e.g. from checkpoint feature_cols)
    for col in feature_cols:
        if col not in df_eval.columns:
            df_eval[col] = 0.0
    
    nan_counts = df_eval[feature_cols].isna().sum().sum()
    if nan_counts > 0:
        logger.info(f"Filling {nan_counts} NaN values in feature columns with 0")
        df_eval[feature_cols] = df_eval[feature_cols].fillna(0)
        
    df_eval[feature_cols] = scaler.transform(df_eval[feature_cols])
    
    test_dataset = IcebergDriftDataset(
        df_eval,
        feature_cols,
        target_cols=['target_u', 'target_v'],
        sequence_length=config.model.ml_correction.sequence_length,
        prediction_horizon=config.model.ml_correction.prediction_horizon,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False,
        collate_fn=iceberg_drift_collate_fn
    )
    
    validator = DriftValidator(model, ValidationConfig(), device)
    metrics = validator.validate(test_loader)
    
    logger.info("============================================================")
    logger.info("NIC DATASET EVALUATION COMPLETE")
    logger.info("============================================================")
    logger.info(f"  Position RMSE: {metrics.metrics.rmse_position_km:.2f} km")
    logger.info(f"  Mean Position Error: {metrics.metrics.mean_position_error_km:.2f} km")
    logger.info(f"  Median Position Error: {metrics.metrics.median_position_error_km:.2f} km")
    logger.info("============================================================")
    
if __name__ == "__main__":
    main()
