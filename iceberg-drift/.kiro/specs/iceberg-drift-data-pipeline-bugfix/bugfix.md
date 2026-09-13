# Bugfix Requirements Document

## Introduction

This document specifies requirements for fixing critical bugs in the Iceberg Drift Prediction Model that prevent the model from learning meaningful patterns and achieving acceptable prediction accuracy. The current system produces position errors exceeding 8000 km due to zero environmental data, incorrect feature indexing, and inadequate training configuration. These fixes will restore the model to its intended operational capability with position RMSE < 100 km and positive skill scores versus baseline methods.

## Bug Analysis

### Current Behavior (Defect)

#### 1. Zero Environmental Data (BUG-001)

1.1 WHEN synthetic data is generated or loaded for training THEN the system produces training samples where current_uo=0.0, current_vo=0.0, wind_u10=0.0, wind_v10=0.0 for all samples

1.2 WHEN the model trains on this zero-valued environmental data THEN the system produces prediction errors exceeding 8000 km due to inability to learn meaningful environmental forcing patterns

#### 2. Missing Feature Index Mapping (BUG-002)

1.3 WHEN PINN models are initialized during training THEN the system uses fallback indexing (last 4 columns) instead of explicit feature_idx_map

1.4 WHEN the model accesses environmental forcings using fallback indices THEN the system reads incorrect columns because feature engineering has added many additional columns beyond the expected last 4

1.5 WHEN physics constraints use wrong feature columns THEN the system applies incorrect physics-informed loss calculations

#### 3. Scheduler Step Crash (BUG-003)

1.6 WHEN the training loop accesses val_metrics[self.config.early_stopping_metric] without checking key existence THEN the system crashes with KeyError if the early_stopping_metric is not present in validation metrics

#### 4. Python Version Inconsistency (BUG-004)

1.7 WHEN pyproject.toml specifies requires-python = ">=3.11" but tool.mypy specifies python_version = "3.10" THEN the system has configuration mismatches that cause type checking errors

#### 5. Physics Comparison Dimension Error (BUG-005)

1.8 WHEN validator compares predictions with physics baseline using sklearn metrics THEN the system crashes with "Found array with dim 3, while dim <= 2 is required" due to incorrect array shapes

#### 6. Constant Target Values (BUG-006)

1.9 WHEN trajectory segments contain identical v-velocity values repeated many times (e.g., 49.9952 repeated 20 times) THEN the system has no useful gradient signal for learning

1.10 WHEN stationary icebergs or low-quality trajectory segments are included in training THEN the system cannot improve model performance due to lack of variation in targets

#### 7. Missing Data Quality Checks (BUG-007)

1.11 WHEN environmental data is loaded before training THEN the system does not validate data ranges, NaN values, or zero variance

1.12 WHEN invalid data silently enters the training pipeline THEN the system produces garbage predictions without warning

#### 8. Inadequate Training Configuration (BUG-008)

1.13 WHEN default config specifies epochs=3 and batch_size=2 THEN the system cannot converge or learn meaningful patterns

1.14 WHEN testing configuration is used for production training THEN the system produces undertrained models with poor accuracy

#### 9. No Logging of Data Statistics (BUG-009)

1.15 WHEN data is loaded and preprocessed THEN the system does not log feature ranges, target statistics, or missing value counts

1.16 WHEN data pipeline issues occur THEN the system is difficult to debug due to lack of diagnostic information

#### 10. Unrealistic Synthetic Data (BUG-010)

1.17 WHEN synthetic data generation uses simplistic random sampling THEN the system produces training data without realistic spatiotemporal correlations

1.18 WHEN the model trains on unrealistic synthetic data THEN the system learns patterns that do not transfer to real-world iceberg drift

### Expected Behavior (Correct)

#### 1. Non-Zero Environmental Data (BUG-001 Fix)

2.1 WHEN synthetic data is generated or real data is loaded for training THEN the system SHALL populate environmental features with realistic non-zero values (ocean currents: 0.1-0.5 m/s, wind: 5-15 m/s)

2.2 WHEN the model trains on realistic environmental data THEN the system SHALL achieve position RMSE < 100 km through learning meaningful forcing patterns

#### 2. Explicit Feature Index Mapping (BUG-002 Fix)

2.3 WHEN PINN models are initialized during training THEN the system SHALL create and pass an explicit feature_idx_map dictionary with keys {'current_uo': idx, 'current_vo': idx, 'wind_u10': idx, 'wind_v10': idx}

2.4 WHEN the model accesses environmental forcings THEN the system SHALL use the feature_idx_map to read correct columns regardless of feature engineering changes

2.5 WHEN physics constraints are computed THEN the system SHALL apply correct physics-informed loss using properly indexed features

#### 3. Safe Scheduler Step Access (BUG-003 Fix)

2.6 WHEN the training loop accesses early stopping metrics THEN the system SHALL use val_metrics.get(self.config.early_stopping_metric, val_loss) with fallback to avoid KeyError

#### 4. Consistent Python Version (BUG-004 Fix)

2.7 WHEN Python version requirements are specified THEN the system SHALL align pyproject.toml and tool.mypy configurations to requires-python = ">=3.11" and python_version = "3.11"

#### 5. Correct Physics Comparison Dimensions (BUG-005 Fix)

2.8 WHEN validator compares predictions with physics baseline THEN the system SHALL reshape arrays to 2D before passing to sklearn metrics to avoid dimension errors

#### 6. Filtered Target Values (BUG-006 Fix)

2.9 WHEN trajectory segments are processed for training THEN the system SHALL filter out or separately handle segments with constant velocity values (variance below threshold)

2.10 WHEN training data is created THEN the system SHALL include only trajectory segments with sufficient variation in velocity targets

#### 7. Data Quality Validation (BUG-007 Fix)

2.11 WHEN environmental data is loaded before training THEN the system SHALL validate that features have no NaN values, non-zero variance, and values within expected physical ranges

2.12 WHEN data quality checks fail THEN the system SHALL raise informative errors identifying which features are invalid and why

#### 8. Appropriate Training Configuration (BUG-008 Fix)

2.13 WHEN production training is performed THEN the system SHALL use configuration with epochs >= 50, batch_size >= 16, and appropriate learning rate schedules

2.14 WHEN test/development training is performed THEN the system SHALL use separate configuration files (test_settings.yaml) to avoid confusion

#### 9. Comprehensive Data Logging (BUG-009 Fix)

2.15 WHEN data is loaded and preprocessed THEN the system SHALL log min/max/mean/std for each feature, target statistics, sample counts, and missing value counts

2.16 WHEN training begins THEN the system SHALL output data statistics summary for diagnostic purposes

#### 10. Realistic Synthetic Data (BUG-010 Fix)

2.17 WHEN synthetic data is generated THEN the system SHALL create spatiotemporal patterns with physically plausible correlations (wind-current alignment, Coriolis effects, realistic gradients)

2.18 WHEN synthetic trajectories are created THEN the system SHALL ensure drift velocities are consistent with environmental forcings and physics-based drift equations

### Unchanged Behavior (Regression Prevention)

#### Model Architecture Preservation

3.1 WHEN fixes are applied to PINN models THEN the system SHALL CONTINUE TO support all existing model types (LSTM, GRU, Transformer, MLP, PINN, Hybrid)

3.2 WHEN feature indexing is corrected THEN the system SHALL CONTINUE TO maintain backward compatibility with previously trained models through version detection

#### Configuration Interface Preservation

3.3 WHEN training configuration is updated THEN the system SHALL CONTINUE TO support command-line argument overrides (--model-type, --epochs, --batch-size)

3.4 WHEN config files are separated THEN the system SHALL CONTINUE TO load settings.yaml by default

#### Data Pipeline Interface Preservation

3.5 WHEN data quality checks are added THEN the system SHALL CONTINUE TO support both synthetic and real data sources without changing the data loading API

3.6 WHEN data preprocessing is enhanced THEN the system SHALL CONTINUE TO produce PyTorch DataLoader objects with the same interface

#### Export Functionality Preservation

3.7 WHEN prediction pipeline is fixed THEN the system SHALL CONTINUE TO support all export formats (GeoJSON, NetCDF, CSV, GPX, KML)

3.8 WHEN model accuracy improves THEN the system SHALL CONTINUE TO generate trajectory visualizations and error heatmaps

#### Evaluation Metrics Preservation

3.9 WHEN metrics computation is corrected THEN the system SHALL CONTINUE TO compute all existing metrics (position RMSE/MAE, direction error, along-track/cross-track error, skill scores)

3.10 WHEN validation is enhanced THEN the system SHALL CONTINUE TO support bootstrap confidence intervals and multi-horizon evaluation

#### Physics Model Preservation

3.11 WHEN PINN physics constraints are fixed THEN the system SHALL CONTINUE TO use the same physics equations (u_drift = u_current + α·u_wind + sign(lat)·β·v_wind)

3.12 WHEN environmental data is corrected THEN the system SHALL CONTINUE TO apply Coriolis deflection and wind drift factors as specified in configuration

#### API and Inference Preservation

3.13 WHEN models are retrained with fixes THEN the system SHALL CONTINUE TO support the DriftPredictor.predict() API for inference

3.14 WHEN model checkpoints are saved THEN the system SHALL CONTINUE TO be loadable via load_model_for_inference() function

#### Logging and Monitoring Preservation

3.15 WHEN additional logging is added THEN the system SHALL CONTINUE TO write training.log files and support TensorBoard/W&B integration

3.16 WHEN error messages are improved THEN the system SHALL CONTINUE TO provide stack traces and debug information at appropriate verbosity levels
