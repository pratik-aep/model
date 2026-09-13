#!/usr/bin/env python
"""
Training script for iceberg drift prediction model using REAL CSV data.

This script loads the user's CSV datasets (AAD, NPI) and combines them
with synthetic data for training.

Usage:
    python scripts/train_with_real_data.py --data-dir "/path/to/csv/files" --model-type xgboost --n-estimators 500
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceberg_drift.config import load_config
from iceberg_drift.data_processing import (
    download_era5_wind,
    download_copernicus_currents,
    match_environmental_data,
    engineer_features,
    create_targets,
    split_trajectories,
    get_feature_columns,
    scale_features,
)
from iceberg_drift.models import (
    PhysicsDriftModel,
    MultiTargetGBM,
    PhysicsInformedGBM,
    create_gbm_config,
)
from iceberg_drift.evaluation import TrainingConfig, train_model, validate_model, ValidationConfig


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("output/training_real.log"),
        ],
    )


def load_real_iceberg_data(data_dir: str) -> pd.DataFrame:
    """
    Load and merge all real iceberg CSV datasets.
    """
    logger = logging.getLogger(__name__)
    data_path = Path(data_dir)

    all_dfs = []

    # AAD 1984-2011
    aad_84_11 = data_path / "AAD_Iceberg_database_1984-2011_version_2022-06-28.csv"
    if aad_84_11.exists():
        logger.info(f"Loading {aad_84_11}")
        df = pd.read_csv(aad_84_11)
        df['source'] = 'AAD_1984_2011'
        all_dfs.append(df)

    # AAD 1979-1984
    aad_79_84 = data_path / "AAD_Iceberg_database_1979-1984_version_2022-06-28.csv"
    if aad_79_84.exists():
        logger.info(f"Loading {aad_79_84}")
        df = pd.read_csv(aad_79_84)
        df['source'] = 'AAD_1979_1984'
        all_dfs.append(df)

    # NPI 1977-2010 (semicolon separated, may need latin-1 encoding)
    npi = data_path / "NPI_Iceberg_database_1977-2010_version_2022-05-20.csv"
    if npi.exists():
        logger.info(f"Loading {npi}")
        try:
            df = pd.read_csv(npi, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 failed, trying latin-1 encoding for NPI data")
            df = pd.read_csv(npi, sep=';', encoding='latin-1')
        df['source'] = 'NPI_1977_2010'
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No CSV files found in {data_dir}")

    # Combine all
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined {len(all_dfs)} datasets: {len(combined)} total records")

    # Standardize column names
    combined = standardize_iceberg_columns(combined)

    return combined


def standardize_iceberg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across datasets."""
    logger = logging.getLogger(__name__)

    # Column mapping for AAD datasets (Obs_Lat/Obs_Lon are correct for most)
    aad_mapping = {
        'ID': 'orig_iceberg_id',
        'Cruise-ID': 'cruise_id',
        'Vessel': 'vessel',
        'Obs_Date_ISO': 'datetime',
        'Obs_Lat': 'lat',
        'Obs_Lon': 'lon',
        'Ice_conc': 'ice_concentration',
        'RV': 'rv',
        'Total': 'total_count',
        'size1': 'size_cat_1',
        'size2': 'size_cat_2',
        'size3': 'size_cat_3',
        'size4': 'size_cat_4',
        'size5': 'size_cat_5',
        'size6': 'size_cat_6',
        'size7': 'size_cat_7',
    }

    # Column mapping for AAD 84-11 (lat/lon are SWAPPED in this dataset!)
    aad_84_11_mapping = {
        'ID': 'orig_iceberg_id',
        'Cruise-ID': 'cruise_id',
        'Vessel': 'vessel',
        'Obs_Date_ISO': 'datetime',
        'Obs_Lat': 'lon',   # SWAPPED!
        'Obs_Lon': 'lat',   # SWAPPED!
        'Ice_conc': 'ice_concentration',
        'RV': 'rv',
        'Total': 'total_count',
        'size1': 'size_cat_1',
        'size2': 'size_cat_2',
        'size3': 'size_cat_3',
        'size4': 'size_cat_4',
        'size5': 'size_cat_5',
        'size6': 'size_cat_6',
        'size7': 'size_cat_7',
    }

    # Column mapping for NPI dataset
    npi_mapping = {
        'ID': 'orig_iceberg_id',
        'Cruise-ID': 'cruise_id',
        'Vessel': 'vessel',
        'Obs_Date': 'datetime',
        'Obs_Date.1': 'datetime_nz',
        'Obs_Lat': 'lat',
        'Obs_Lon': 'lon',
        'Ice_conc': 'ice_concentration',
        'RV': 'rv',
        'Total': 'total_count',
        'no10-50': 'size_cat_1',
        'no50-200': 'size_cat_2',
        'no200-500': 'size_cat_3',
        'no500-1000': 'size_cat_4',
        'gt1000': 'size_cat_5',
        'Freeboard': 'freeboard',
        'Length': 'length_m',
        'Width': 'width_m',
        'Dim_as_average': 'dim_avg',
        'View_radius': 'view_radius',
        'comments': 'comments',
    }

    # Apply mapping based on source - SKIP AAD_1984_2011 (corrupted: all zeros in longitude)
    for src in ['AAD_1979_1984']:
        mask = df['source'] == src
        if mask.any():
            logger.info(f"Applying AAD mapping for {src}, rows: {mask.sum()}")
            df.loc[mask] = df.loc[mask].rename(columns=aad_mapping)

    # AAD 1984-2011 has corrupted coordinates (longitude all zeros) - SKIP
    mask = df['source'] == 'AAD_1984_2011'
    if mask.any():
        logger.warning(f"SKIPPING AAD_1984_2011 (corrupted: longitude all zeros), rows: {mask.sum()}")
        df = df[~mask].copy()

    mask = df['source'] == 'NPI_1977_2010'
    if mask.any():
        logger.info(f"Applying NPI mapping for {mask.sum()} rows")
        df.loc[mask] = df.loc[mask].rename(columns=npi_mapping)

    # Check if orig_iceberg_id exists after renaming
    if 'orig_iceberg_id' not in df.columns:
        # Try to find the ID column
        id_cols = [c for c in df.columns if c.lower() in ['id', 'orig_iceberg_id']]
        if id_cols:
            df['orig_iceberg_id'] = df[id_cols[0]]
            logger.warning(f"Using {id_cols[0]} as orig_iceberg_id")
        else:
            raise ValueError(f"orig_iceberg_id not found after renaming. Columns: {df.columns.tolist()}")

    # Handle datetime column - check which column exists
    datetime_cols = ['Obs_Date_ISO', 'Obs_Date', 'datetime']
    for col in datetime_cols:
        if col in df.columns:
            # Handle backslash format in NPI data (e.g., 1977-01-05\T21:00\Z)
            datetime_series = df[col].astype(str).str.replace(r'\\\\', '', regex=True)
            df['datetime'] = pd.to_datetime(datetime_series, errors='coerce', format='ISO8601')
            break
    else:
        raise ValueError("No datetime column found in data")

    # Ensure lat/lon columns exist
    lat_cols = ['Obs_Lat', 'lat']
    for col in lat_cols:
        if col in df.columns:
            df['lat'] = pd.to_numeric(df[col], errors='coerce')
            break

    lon_cols = ['Obs_Lon', 'lon']
    for col in lon_cols:
        if col in df.columns:
            df['lon'] = pd.to_numeric(df[col], errors='coerce')
            break

    # Ensure numeric types
    numeric_cols = ['lat', 'lon', 'length_m', 'width_m', 'freeboard']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create unique iceberg_id combining source and original ID
    df['iceberg_id'] = df['source'] + '_' + df['orig_iceberg_id'].astype(str)

    # Sort by iceberg_id and datetime
    df = df.sort_values(['iceberg_id', 'datetime']).reset_index(drop=True)

    # Filter valid coordinates
    df = df.dropna(subset=['lat', 'lon', 'datetime'])
    df = df[(df['lat'] >= -90) & (df['lat'] <= -50)]  # Antarctic region
    df = df[(df['lon'] >= -180) & (df['lon'] <= 180)]

    logger = logging.getLogger(__name__)
    logger.info(f"After cleaning: {len(df)} records, {df['iceberg_id'].nunique()} icebergs")

    return df


def create_iceberg_trajectories(df: pd.DataFrame, min_obs: int = 1) -> pd.DataFrame:
    """Filter icebergs with enough observations for trajectory modeling.

    Note: Real data has 1 observation per iceberg (snapshot sightings), not trajectories.
    We use min_obs=1 to allow single-observation icebergs for training.
    """
    # Count observations per iceberg
    obs_counts = df.groupby('iceberg_id').size()
    valid_icebergs = obs_counts[obs_counts >= min_obs].index

    df_filtered = df[df['iceberg_id'].isin(valid_icebergs)].copy()

    logger = logging.getLogger(__name__)
    logger.info(f"Icebergs with >={min_obs} obs: {len(valid_icebergs)} out of {len(obs_counts)}")
    logger.info(f"Records: {len(df)} -> {len(df_filtered)}")

    return df_filtered


def main():
    parser = argparse.ArgumentParser(description="Train iceberg drift model with real CSV data")
    parser.add_argument("--data-dir", type=str, default="/Users/pratiksmac/Downloads/icccy/data set", help="Directory with CSV files")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm", "physics_informed"])
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    config = load_config("src/iceberg_drift/config/settings.yaml")
    logger.info(f"Loaded config")

    # =============================================================================
    # Step 1: Load Real Iceberg Data
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Real Iceberg Data")
    logger.info("=" * 60)

    iceberg_df = load_real_iceberg_data(args.data_dir)

    # Create trajectories with minimum observations
    iceberg_df = create_iceberg_trajectories(iceberg_df, min_obs=5)

    # =============================================================================
    # Step 2: Download Environmental Data (synthetic for dev, real for prod)
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Environmental Data")
    logger.info("=" * 60)

    wind_ds = download_era5_wind(
        output_dir=config.paths.raw_data,
        start_date="2010-01-01",
        end_date="2024-12-31",
    )
    current_ds = download_copernicus_currents(
        output_dir=config.paths.raw_data,
        start_date="2010-01-01",
        end_date="2024-12-31",
    )

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

    # =============================================================================
    # Step 5: Create Targets
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Target Creation")
    logger.info("=" * 60)

    prediction_horizon = config.model.ml_correction.prediction_horizon
    targets_df = create_targets(
        features_df,
        prediction_horizon_hours=prediction_horizon,
        target_type="velocity",
    )
    logger.info(f"Targets shape: {targets_df.shape}")

    # =============================================================================
    # Step 6: Train/Val/Test Split (use time-based for real data)
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 6: Data Splitting")
    logger.info("=" * 60)

    train_df, val_df, test_df = split_trajectories(
        targets_df,
        train_frac=config.training.train_split,
        val_frac=config.training.val_split,
        test_frac=config.training.test_split,
        random_seed=config.training.random_seed,
        strategy="time",
    )

    # =============================================================================
    # Step 7: Feature Selection & Scaling
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 7: Feature Selection & Scaling")
    logger.info("=" * 60)

    feature_cols = [c for c in train_df.columns if not c.startswith(('target_', 'iceberg_id', 'datetime', 'lat', 'lon', 'source', 'cruise_id', 'vessel', 'rv', 'ice_concentration', 'total_count'))]
    target_cols = [c for c in train_df.columns if c.startswith('target_')]

    logger.info(f"Selected {len(feature_cols)} features, {len(target_cols)} targets")

    train_df, val_df, test_df, scaler = scale_features(
        train_df, val_df, test_df, feature_cols, scaler_type="standard"
    )

    # Save scaler
    import joblib
    joblib.dump(scaler, Path(config.paths.models) / "feature_scaler.pkl")

    # =============================================================================
    # Step 8: Prepare Training Data
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 8: Prepare Training Data")
    logger.info("=" * 60)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_cols].values.astype(np.float32)

    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # =============================================================================
    # Step 9: Model Training
    # =============================================================================
    logger.info("=" * 60)
    logger.info("STEP 9: Model Training")
    logger.info("=" * 60)

    if args.model_type in ["xgboost", "lightgbm", "physics_informed"]:
        logger.info(f"Training {args.model_type.upper()} model...")

        gbm_config = create_gbm_config(
            model_type=args.model_type if args.model_type != "physics_informed" else "xgboost",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            use_gpu=False,
        )

        if args.model_type == "physics_informed":
            from iceberg_drift.models import PhysicsDriftModel, PhysicsInformedGBM
            physics_model = PhysicsDriftModel()
            model = PhysicsInformedGBM(
                gbm_config=gbm_config,
                physics_model=physics_model,
                target_names=target_cols,
            )
            model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
            model.save(Path(config.paths.models) / "physics_informed_gbm")
        else:
            model = MultiTargetGBM(config=gbm_config, target_names=target_cols)
            model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
            model.save(Path(config.paths.models) / f"{args.model_type}_model")

        # Evaluate on test
        test_pred = model.predict(X_test)
        from iceberg_drift.evaluation.metrics import regression_metrics
        test_metrics = regression_metrics(test_pred, y_test)
        logger.info(f"Test Metrics: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}")

        # Feature importance
        importance = model.get_feature_importance_summary()
        logger.info(f"Top 15 features:\n{importance.head(15)}")
        importance.to_csv(Path(config.paths.output) / f"{args.model_type}_feature_importance.csv", index=False)

    else:
        # Neural Networks
        logger.info(f"Training {args.model_type.upper()} neural network...")

        from iceberg_drift.data_processing import create_dataloaders

        sequence_length = config.model.ml_correction.sequence_length
        pred_horizon = config.model.ml_correction.prediction_horizon
        time_step = 6

        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            sequence_length=sequence_length,
            prediction_horizon=pred_horizon,
            time_step_hours=time_step,
            batch_size=args.batch_size,
            num_workers=4,
            mode="sequence",
        )

        logger.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

        input_dim = len(feature_cols)
        output_dim = len(target_cols)

        # Build feature index map for PINN/Hybrid models
        feature_idx_map = {}
        for idx, col in enumerate(feature_cols):
            if col in ['current_uo', 'current_vo', 'wind_u10', 'wind_v10']:
                feature_idx_map[col] = idx

        required_cols = ['current_uo', 'current_vo', 'wind_u10', 'wind_v10']
        missing_cols = [col for col in required_cols if col not in feature_idx_map]
        if missing_cols:
            logger.warning(f"Missing required columns for PINN/Hybrid: {missing_cols}. "
                           f"Available columns: {feature_cols}")
            feature_idx_map = None

        if args.model_type == "pinn":
            from iceberg_drift.models import PINNDriftModel
            model = PINNDriftModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.model.ml_correction.hidden_dim,
                num_layers=config.model.ml_correction.num_layers,
                dropout=config.model.ml_correction.dropout,
                prediction_horizon=pred_horizon,
                ml_model_type="lstm",
                loss_weights={
                    "data": 1.0,
                    "physics": config.model.ml_correction.loss_weights.physics,
                    "boundary": config.model.ml_correction.loss_weights.boundary,
                },
                feature_idx_map=feature_idx_map,
            )
        elif args.model_type == "hybrid":
            from iceberg_drift.models import HybridDriftModel
            model = HybridDriftModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.model.ml_correction.hidden_dim,
                num_layers=config.model.ml_correction.num_layers,
                dropout=config.model.ml_correction.dropout,
                prediction_horizon=pred_horizon,
                ml_model_type="lstm",
                feature_idx_map=feature_idx_map,
            )
        else:
            from iceberg_drift.models import create_model
            model = create_model(
                args.model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.model.ml_correction.hidden_dim,
                num_layers=config.model.ml_correction.num_layers,
                dropout=config.model.ml_correction.dropout,
                prediction_horizon=pred_horizon,
            )

        logger.info(f"Created {args.model_type} model with {model.get_num_params():,} parameters")

        train_config = TrainingConfig(
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=15,
            output_dir=config.paths.models,
            metric_horizons=[6, 12, 24, 48, 72],
        )

        trained_model, state = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=args.device,
        )

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best epoch: {state.best_epoch + 1}")
        logger.info(f"Best validation loss: {state.best_metric:.4f}")

        # Final Evaluation
        logger.info("=" * 60)
        logger.info("STEP 10: Final Test Evaluation")
        logger.info("=" * 60)

        val_config = ValidationConfig(
            metric_horizons=[6, 12, 24, 48, 72],
            output_dir="output/validation",
        )

        val_result = validate_model(
            trained_model,
            test_loader,
            config=val_config,
            device=args.device,
        )

        logger.info("Final Test Metrics:")
        logger.info(f"  Position RMSE: {val_result.metrics.rmse_position_km:.2f} km")
        logger.info(f"  Position MAE: {val_result.metrics.mean_position_error_km:.2f} km")
        logger.info(f"  Direction Error: {val_result.metrics.mean_direction_error_deg:.1f}°")
        logger.info(f"  Skill vs Persistence: {val_result.metrics.skill_score_vs_persistence:.3f}")
        logger.info(f"  Skill vs Physics: {val_result.metrics.skill_score_vs_physics:.3f}")


if __name__ == "__main__":
    main()