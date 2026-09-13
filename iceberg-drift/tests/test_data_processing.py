"""Tests for data processing pipeline."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from iceberg_drift.data_processing import (
    compute_coriolis,
    engineer_features,
    create_targets,
    split_trajectories,
    get_feature_columns,
    scale_features,
)
from iceberg_drift.data_processing.download import _generate_synthetic_icebergs, _generate_synthetic_era5, _generate_synthetic_currents


class TestDownload:
    """Test data download (synthetic generation)."""

    def test_synthetic_icebergs(self):
        df = _generate_synthetic_icebergs("2020-01-01", "2020-12-31", 100)
        assert len(df) > 0
        assert 'iceberg_id' in df.columns
        assert 'datetime' in df.columns
        assert 'lat' in df.columns
        assert 'lon' in df.columns
        assert 'length_m' in df.columns
        assert df['length_m'].min() >= 100

    def test_synthetic_era5(self):
        ds = _generate_synthetic_era5("2020-01-01", "2020-01-31", (-180, -80, 180, -50), ["u10", "v10"])
        assert 'u10' in ds
        assert 'v10' in ds
        assert len(ds.time) > 0

    def test_synthetic_currents(self):
        ds = _generate_synthetic_currents("2020-01-01", "2020-01-31", (-180, -80, 180, -50), [0, 10], ["uo", "vo"])
        assert 'uo' in ds
        assert 'vo' in ds
        assert len(ds.time) > 0
        assert len(ds.depth) == 2



@pytest.fixture
def sample_iceberg_df():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="6h")
    iceberg_ids = ['ICE_001'] * 50 + ['ICE_002'] * 50
    lats1 = -65 + np.cumsum(np.random.normal(0, 0.01, 50))
    lons1 = 0 + np.cumsum(np.random.normal(0, 0.01, 50))
    lats2 = -65 + np.cumsum(np.random.normal(0, 0.01, 50))
    lons2 = 0 + np.cumsum(np.random.normal(0, 0.01, 50))
    df = pd.DataFrame({
        'iceberg_id': ['ICE_001'] * 50 + ['ICE_002'] * 50,
        'datetime': np.concatenate([dates[:50], dates[:50]]),
        'lat': np.concatenate([lats1, lats2]),
        'lon': np.concatenate([lons1, lons2]),
        'length_m': 2000,
        'width_m': 1000,
    })
    return df

@pytest.fixture
def sample_wind_ds():
    times = pd.date_range("2020-01-01", periods=100, freq="6h")
    lats = np.linspace(-70, -60, 11)
    lons = np.linspace(-10, 10, 21)
    u10 = np.random.normal(0, 8, (len(times), len(lats), len(lons)))
    v10 = np.random.normal(0, 8, (len(times), len(lats), len(lons)))
    ds = xr.Dataset(
        {'u10': (['time', 'latitude', 'longitude'], u10),
         'v10': (['time', 'latitude', 'longitude'], v10)},
        coords={'time': times, 'latitude': lats, 'longitude': lons},
    )
    return ds

@pytest.fixture
def sample_current_ds():
    times = pd.date_range("2020-01-01", periods=100, freq="D")
    lats = np.linspace(-70, -60, 11)
    lons = np.linspace(-10, 10, 21)
    depths = [0, 10, 50]
    uo = np.random.normal(0, 0.1, (len(times), len(depths), len(lats), len(lons)))
    vo = np.random.normal(0, 0.1, (len(times), len(depths), len(lats), len(lons)))
    ds = xr.Dataset(
        {'uo': (['time', 'depth', 'latitude', 'longitude'], uo),
         'vo': (['time', 'depth', 'latitude', 'longitude'], vo)},
        coords={'time': times, 'depth': depths, 'latitude': lats, 'longitude': lons},
    )
    return ds

class TestPreprocessing:
    """Test preprocessing functions."""

    def test_coriolis(self):
        # Equator
        assert abs(compute_coriolis(0)) < 1e-10
        # North pole
        f_north = compute_coriolis(90)
        assert f_north > 0
        # South pole
        f_south = compute_coriolis(-90)
        assert f_south < 0
        # Southern hemisphere negative
        assert compute_coriolis(-65) < 0

    def test_engineer_features(self, sample_iceberg_df, sample_wind_ds, sample_current_ds):
        from iceberg_drift.data_processing import match_environmental_data

        # Match environmental data
        matched = match_environmental_data(sample_iceberg_df, sample_wind_ds, sample_current_ds)

        # Engineer features
        featured = engineer_features(matched, add_cyclical_time=True, add_physics_features=True)

        # Check new columns
        assert 'wind_speed' in featured.columns
        assert 'wind_dir' in featured.columns
        assert 'current_speed' in featured.columns
        assert 'current_dir' in featured.columns
        assert 'coriolis_f' in featured.columns
        assert 'physics_u' in featured.columns
        assert 'physics_v' in featured.columns
        assert 'hour_sin' in featured.columns
        assert 'doy_cos' in featured.columns

    def test_create_targets(self, sample_iceberg_df, sample_wind_ds, sample_current_ds):
        from iceberg_drift.data_processing import match_environmental_data, engineer_features

        matched = match_environmental_data(sample_iceberg_df, sample_wind_ds, sample_current_ds)
        featured = engineer_features(matched)
        targets = create_targets(featured, prediction_horizon_hours=24, target_type="velocity")

        # Check target columns
        assert 'target_u' in targets.columns
        assert 'target_v' in targets.columns
        assert 'target_distance_km' in targets.columns
        assert 'target_bearing' in targets.columns

        # Check NaN rows dropped
        assert targets['target_u'].notna().all()

    def test_split_trajectories(self, sample_iceberg_df, sample_wind_ds, sample_current_ds):
        from iceberg_drift.data_processing import match_environmental_data, engineer_features, create_targets

        matched = match_environmental_data(sample_iceberg_df, sample_wind_ds, sample_current_ds)
        featured = engineer_features(matched)
        targets = create_targets(featured)

        train, val, test = split_trajectories(
            targets, train_frac=0.7, val_frac=0.15, test_frac=0.15,
            random_seed=42, strategy="trajectory"
        )

        # Check splits
        assert len(train) + len(val) + len(test) == len(targets)
        # Check no iceberg in multiple splits
        train_ids = set(train['iceberg_id'].unique())
        val_ids = set(val['iceberg_id'].unique())
        test_ids = set(test['iceberg_id'].unique())
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_feature_selection(self, sample_iceberg_df, sample_wind_ds, sample_current_ds):
        from iceberg_drift.data_processing import match_environmental_data, engineer_features, create_targets

        matched = match_environmental_data(sample_iceberg_df, sample_wind_ds, sample_current_ds)
        featured = engineer_features(matched)
        targets = create_targets(featured)

        features = get_feature_columns(targets)
        assert len(features) > 0
        # Should not include target or metadata columns
        for f in features:
            assert not f.startswith('target_')
            assert f not in ['iceberg_id', 'datetime', 'lat', 'lon']

    def test_scale_features(self, sample_iceberg_df, sample_wind_ds, sample_current_ds):
        from iceberg_drift.data_processing import match_environmental_data, engineer_features, create_targets, split_trajectories

        matched = match_environmental_data(sample_iceberg_df, sample_wind_ds, sample_current_ds)
        featured = engineer_features(matched)
        targets = create_targets(featured)

        train, val, test = split_trajectories(targets, random_seed=42)
        features = get_feature_columns(train)

        train_s, val_s, test_s, scaler = scale_features(train, val, test, features)

        # Check scaling applied
        assert abs(train_s[features].mean().mean()) < 0.1  # Near zero mean
        assert abs(train_s[features].std().mean() - 1.0) < 0.1  # Near unit variance


class TestDataset:
    """Test PyTorch dataset."""

    def test_dataset_creation(self, sample_iceberg_df, sample_wind_ds, sample_current_ds):
        from iceberg_drift.data_processing import (
            match_environmental_data, engineer_features, create_targets,
            split_trajectories, get_feature_columns, IcebergDriftDataset
        )

        matched = match_environmental_data(sample_iceberg_df, sample_wind_ds, sample_current_ds)
        featured = engineer_features(matched)
        targets = create_targets(featured)
        train, val, test = split_trajectories(targets, random_seed=42)

        features = get_feature_columns(train)
        target_cols = [c for c in train.columns if c.startswith('target_')]

        dataset = IcebergDriftDataset(
            train, features, target_cols,
            sequence_length=4, prediction_horizon=4, time_step_hours=6,
            mode="sequence",
        )

        assert len(dataset) > 0
        sample = dataset[0]
        assert 'x' in sample
        assert 'y' in sample
        assert 'meta' in sample
        assert sample['x'].shape[0] == 4  # sequence_length
        assert sample['y'].shape[0] == 4  # prediction_horizon


if __name__ == "__main__":
    pytest.main([__file__, "-v"])