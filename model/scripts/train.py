#!/usr/bin/env python
"""
Training script for iceberg drift prediction model.

Usage:
    python scripts/train.py --config config/settings.yaml --model-type xgboost
    python scripts/train.py --config config/settings.yaml --model-type pinn --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceberg_drift.config import load_config
from iceberg_drift.data_processing.download import (
    download_iceberg_positions,
    download_era5_wind,
    download_copernicus_currents,
    _generate_synthetic_icebergs,
    _generate_synthetic_era5,
    _generate_synthetic_currents,
)
from iceberg_drift.data_processing import (

    match_environmental_data,
    engineer_features,
    create_targets,
    split_trajectories,
    get_feature_columns,
    scale_features,
    DataQualityPipeline,
)
from iceberg_drift.models import (
    PhysicsDriftModel,
    PINNDriftModel,
    HybridDriftModel,
    create_model,
    MultiTargetGBM,
    PhysicsInformedGBM,
    create_gbm_config,
)
from iceberg_drift.evaluation import TrainingConfig, train_model, validate_model, ValidationConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("output/training.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Train iceberg drift prediction model")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file path")
    parser.add_argument("--model-type", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm", "pinn", "hybrid", "lstm", "gru", "transformer", "mlp", "physics_informed"])
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (NN only)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (NN only)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (NN only)")
    parser.add_argument("--n-estimators", type=int, default=500, help="Number of trees (GBM only)")
    parser.add_argument("--max-depth", type=int, default=5, help="Max tree depth (GBM only)")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="GBM learning rate")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for GBM")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic data if credentials missing (for testing only)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # =============================================================================
    # CREDENTIAL CHECK
    # =============================================================================
    logger.info("=" * 60)
    logger.info("CREDENTIAL CHECK")
    logger.info("=" * 60)

    # Import credential check functions
    from iceberg_drift.data_processing.download import check_cds_credentials, check_cmems_credentials

    # Check credentials
    cds_ok = check_cds_credentials()
    cmems_ok = check_cmems_credentials()

    # Determine if we can proceed
    missing_creds = []
    if not cds_ok:
        missing_creds.append("ERA5 (CDS API): ~/.cdsapirc not found")
    if not cmems_ok:
        missing_creds.append("Copernicus Marine: not logged in")

    # Print results
    if cds_ok:
        logger.info("[OK]      ERA5 (CDS API):        ~/.cdsapirc found")
    else:
        logger.info("[MISSING] ERA5 (CDS API):        ~/.cdsapirc not found")

    if cmems_ok:
        logger.info("[OK]      Copernicus Marine:     logged in")
    else:
        logger.info("[MISSING] Copernicus Marine:     not logged in")

    logger.info("=" * 60)

    # Handle missing credentials
    if missing_creds and not args.allow_synthetic:
        logger.error("ERROR: Missing required credentials. Fix with:")
        if not cds_ok:
            logger.error("  Create ~/.cdsapirc — see https://cds.climate.copernicus.eu/how-to-api")
        if not cmems_ok:
            logger.error("  Run: copernicusmarine login")
        logger.error("")
        logger.error("Then re-run: python3 scripts/train.py")
        logger.error("(Or pass --allow-synthetic to run on placeholder data for testing only — ")
        logger.error(" NOT suitable for a real trained model.)")
        sys.exit(1)
    elif missing_creds and args.allow_synthetic:
        logger.warning("=" * 60)
        logger.warning("WARNING: RUNNING ON SYNTHETIC DATA ONLY")
        logger.warning("WARNING: This is NOT suitable for a real trained model!")
        logger.warning("=" * 60)

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Determine if we are using synthetic data
    synthetic_data_used = False
    if args.allow_synthetic:
        if not cds_ok or not cmems_ok:
            synthetic_data_used = True
            logger.warning("=" * 60)
            logger.warning("WARNING: RUNNING ON SYNTHETIC DATA ONLY")
            logger.warning("WARNING: This is NOT suitable for a real trained model!")
            logger.warning("=" * 60)

    model_suffix = "_SYNTHETIC" if synthetic_data_used else ""

    # =============================================================================
    # Step 1: Download/Generate Data
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preparation")
    logger.info("=" * 60)

    # Get iceberg position configuration
    iceberg_cfg = config.data_sources.iceberg_positions
    # For local source, use the specified local_path; otherwise use raw_data for caching
    data_dir = config.paths.raw_data
    if iceberg_cfg.source == "local":
        data_dir = iceberg_cfg.local_path

    logger.info(f"Loading iceberg data from source: {iceberg_cfg.source}")
    if synthetic_data_used:
        iceberg_df = _generate_synthetic_icebergs(
            start_date="2015-01-01",
            end_date="2015-01-31",
            min_length_m=iceberg_cfg.min_length_m,
            source="SYNTHETIC",
        )
        logger.info("Generating synthetic ERA5 wind data...")
        wind_ds = _generate_synthetic_era5(
            start_date="2015-01-01",
            end_date="2015-01-31",
            bbox=(-65, -65, -55, -60),
            variables=["u10", "v10", "msl"]
        )
        logger.info("Generating synthetic Copernicus current data...")
        current_ds = _generate_synthetic_currents(
            start_date="2015-01-01",
            end_date="2015-01-31",
            bbox=(-65, -65, -55, -60),
            depth_levels=[0, 10],
            variables=["uo", "vo", "temperature", "salinity"]
        )
    else:
        iceberg_df = download_iceberg_positions(
            output_dir=data_dir,
            start_date=iceberg_cfg.start_date,
            end_date=iceberg_cfg.end_date,
            min_length_m=iceberg_cfg.min_length_m,
            source=iceberg_cfg.source,
            local_path=iceberg_cfg.get("local_path", None),
        )
        logger.info(f"Loaded {len(iceberg_df)} iceberg position records")
    
        # Download environmental data (use small area and short time for testing to avoid CDS limits)
        logger.info("Downloading ERA5 wind data (1 month for testing)...")
        wind_ds = download_era5_wind(
            output_dir=config.paths.raw_data,
            start_date="2015-01-01",
            end_date="2015-01-31",
            bbox=(-180, -90, 180, -50),  # min_lon, min_lat, max_lon, max_lat
        )
        logger.info("Downloading Copernicus current data (1 month for testing)...")
        current_ds = download_copernicus_currents(
            output_dir=config.paths.raw_data,
            start_date="2015-01-01",
            end_date="2015-01-31",
            bbox=(-180, -90, 180, -50),
        )

    # =============================================================================
    # Step 2: Data Quality Pipeline
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Data Quality Pipeline")
    logger.info("=" * 60)

    quality_pipeline = DataQualityPipeline(
        min_observations=10,
        max_gap_hours=48,
        max_speed_kmh=50.0,
        resample_freq="6h",
        coastal_depth_threshold=100.0,
        fill_coastal_nan=True,
    )

    iceberg_df, wind_ds, current_ds = quality_pipeline.run(
        iceberg_df, wind_ds, current_ds
    )

    quality_summary = quality_pipeline.get_summary()
    logger.info(f"Quality report: {quality_summary}")

    # =============================================================================
    # Step 3: Match Environmental Data
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Environmental Data Matching")
    logger.info("=" * 60)

    matched_df = match_environmental_data(
        iceberg_df,
        wind_ds,
        current_ds,
    )
    logger.info(f"Matched data shape: {matched_df.shape}")

    # =============================================================================
    # Step 4: Feature Engineering
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Feature Engineering")
    logger.info("=" * 60)

    features_df = engineer_features(
        matched_df,
        add_cyclical_time=True,
        add_physics_features=True,
        add_lag_features=True,
    )
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Feature columns: {list(features_df.columns)}")

    # =============================================================================
    # Step 5: Create Targets
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Target Creation")
    logger.info("=" * 60)

    prediction_horizon = config.model.ml_correction.prediction_horizon  # This is 6 (timesteps)
    prediction_horizon_hours = prediction_horizon * 6  # Convert to hours (6 timesteps * 6 hours/timestep = 36 hours)
    targets_df = create_targets(
        features_df,
        prediction_horizon_hours=prediction_horizon_hours,
        target_type="position",
    )
    logger.info(f"Targets shape: {targets_df.shape}")

    # Define target columns for model output dimension
    target_cols = ['target_lat', 'target_lon']

    # =============================================================================
    # Step 6: Train/Val/Test Split
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 6: Data Splitting")
    logger.info("=" * 60)

    train_df, val_df, test_df = split_trajectories(
        targets_df,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        random_seed=42,
    )
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Validation shape: {val_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    # =============================================================================
    # Step 7: Feature Scaling
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 7: Feature Scaling")
    logger.info("=" * 60)

    # Get feature columns (exclude non-feature columns)
    feature_cols = get_feature_columns(train_df)

    # Check for NaN in feature columns and fill with 0 to prevent NaN propagation
    nan_counts = train_df[feature_cols].isna().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        logger.info(f"Filling {total_nan} NaN values in feature columns with 0")
        train_df[feature_cols] = train_df[feature_cols].fillna(0)
        if len(val_df) > 0:
            val_df[feature_cols] = val_df[feature_cols].fillna(0)
        if len(test_df) > 0:
            test_df[feature_cols] = test_df[feature_cols].fillna(0)

    logger.info(f"Number of features: {len(feature_cols)}")

    # Scale features - fit scaler on training data and transform all splits
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    if len(val_df) > 0:
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    if len(test_df) > 0:
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # =============================================================================
    # Step 8: Create DataLoaders
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 8: Creating DataLoaders")
    logger.info("=" * 60)

    from iceberg_drift.data_processing.dataset import IcebergDriftDataset, create_dataloaders

    train_dataset = IcebergDriftDataset(
        train_df,
        feature_cols,
        target_cols=['target_lat', 'target_lon'],
        sequence_length=config.model.ml_correction.prediction_horizon,
    )

    val_dataset = IcebergDriftDataset(
        val_df,
        feature_cols,
        target_cols=['target_lat', 'target_lon'],
        sequence_length=config.model.ml_correction.prediction_horizon,
    ) if len(val_df) > 0 else None

    test_dataset = IcebergDriftDataset(
        test_df,
        feature_cols,
        target_cols=['target_lat', 'target_lon'],
        sequence_length=config.model.ml_correction.prediction_horizon,
    ) if len(test_df) > 0 else None

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df,
        val_df,
        test_df,
        feature_cols,
        target_cols=['target_lat', 'target_lon'],
        sequence_length=config.model.ml_correction.prediction_horizon,
        prediction_horizon=config.model.ml_correction.prediction_horizon,
        time_step_hours=6,
        batch_size=config.training.batch_size,
    )

    # =============================================================================
    # Step 9: Model Creation
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 9: Model Creation")
    logger.info("=" * 60)

    # Get feature idx map for environmental features
    feature_idx_map = {}
    for i, col in enumerate(feature_cols):
        feature_idx_map[col] = i

    # Add indices for wind and current if they exist in features
    env_feature_mapping = {
        'wind_u10': -2,
        'wind_v10': -1,
        'current_uo': -4,
        'current_vo': -3,
    }
    for feat, default_idx in env_feature_mapping.items():
        if feat in feature_cols:
            feature_idx_map[feat] = feature_cols.index(feat)
        else:
            feature_idx_map[feat] = default_idx

    # Create model
    output_dim = len(target_cols)  # Number of target variables (e.g., 2 for lat/lon or u/v)
    model = create_model(
        model_type=config.model.type,
        input_dim=len(feature_cols),
        output_dim=output_dim,
        hidden_dim=config.model.ml_correction.hidden_dim,
        num_layers=config.model.ml_correction.num_layers,
        prediction_horizon=config.model.ml_correction.prediction_horizon,
        dropout=config.model.ml_correction.dropout,
        feature_idx_map=feature_idx_map,
    )

    logger.info(f"Created {config.model.type} model with {sum(p.numel() for p in model.parameters())} parameters")

    # =============================================================================
    # Step 10: Model Training
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 10: Model Training")
    logger.info("=" * 60)

    # Training configuration
    train_config = TrainingConfig(
        epochs=args.epochs if args.epochs != 100 else config.training.epochs,  # Use CLI arg if provided
        learning_rate=args.lr if args.lr != 1e-3 else config.training.learning_rate,
        weight_decay=float(config.training.weight_decay),
        optimizer=config.training.optimizer,
        scheduler=config.training.scheduler,
        scheduler_params=config.training.scheduler_params,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_metric=config.training.early_stopping_metric,
        gradient_clip=config.training.gradient_clip,
        use_amp=config.training.use_amp,
        initial_teacher_forcing=config.training.teacher_forcing.initial,
        final_teacher_forcing=config.training.teacher_forcing.final,
        teacher_forcing_decay=config.training.teacher_forcing.decay,
    )

    # Train model
    trained_model, training_history = train_model(
        model,
        train_loader,
        val_loader,
        train_config,
    )

    # =============================================================================
    # Step 11: Save Model
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 11: Saving Model")
    logger.info("=" * 60)

    # Save model checkpoint
    output_dir = Path(config.paths.models)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"{config.model.type}{model_suffix}_model.pt"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'model_type': config.model.type,
            'input_dim': len(feature_cols),
            'hidden_dim': config.model.ml_correction.hidden_dim,
            'num_layers': config.model.ml_correction.num_layers,
            'prediction_horizon': config.model.ml_correction.prediction_horizon,
            'dropout': config.model.ml_correction.dropout,
            'feature_idx_map': feature_idx_map,
            'synthetic_data_used': synthetic_data_used,
        },
        'scaler': scaler,
        'feature_cols': feature_cols,
        'training_history': training_history,
    }, checkpoint_path)

    logger.info(f"Model saved to {checkpoint_path}")
    if synthetic_data_used:
        logger.warning("=" * 60)
        logger.warning("WARNING: TRAINED ON SYNTHETIC DATA - NOT SUITABLE FOR PRODUCTION USE")
        logger.warning("=" * 60)

    # =============================================================================
    # Step 12: Final Validation
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 12: Final Validation")
    logger.info("=" * 60)

    if test_loader is not None:
        test_metrics = validate_model(trained_model, test_loader)
        logger.info(f"Test metrics: {test_metrics}")
    else:
        logger.info("No test set available for validation")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()