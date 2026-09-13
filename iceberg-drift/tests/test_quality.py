"""Tests for data quality module."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from iceberg_drift.data_processing.quality import (
    check_iceberg_data_quality,
    resample_iceberg_trajectory,
    filter_coastal_nan,
    fill_coastal_current_nan,
    check_environmental_data_quality,
    DataQualityPipeline,
    validate_and_clean_iceberg_data,
    create_quality_report,
)


class TestIcebergQuality:
    """Test iceberg data quality checks."""

    @pytest.fixture
    def sample_iceberg_df(self):
        """Create sample iceberg DataFrame with various issues."""
        np.random.seed(42)
        n = 100

        # Create a base trajectory
        dates = pd.date_range("2020-01-01", periods=n, freq="6h")
        lats = -65 + np.cumsum(np.random.normal(0, 0.01, n))
        lons = 0 + np.cumsum(np.random.normal(0, 0.01, n))

        df = pd.DataFrame({
            'iceberg_id': ['ICE_001'] * n,
            'datetime': dates,
            'lat': lats,
            'lon': lons,
            'length_m': 2000,
            'width_m': 1000,
        })

        # Add some problematic records
        # 1. Out of bounds
        df.loc[0, 'lat'] = -30  # Too far north
        df.loc[1, 'lon'] = 200  # Invalid longitude

        # 2. Duplicate timestamps
        dup_row = df.iloc[2].copy()
        dup_row['datetime'] = df.iloc[2]['datetime']  # Same timestamp
        df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)

        # 3. Too few observations iceberg
        short_dates = pd.date_range("2020-01-01", periods=3, freq="6h")
        short_df = pd.DataFrame({
            'iceberg_id': ['ICE_SHORT'] * 3,
            'datetime': short_dates,
            'lat': [-65, -64.9, -64.8],
            'lon': [0, 0.1, 0.2],
            'length_m': 500,
            'width_m': 200,
        })
        df = pd.concat([df, short_df], ignore_index=True)

        return df

    def test_check_iceberg_data_quality(self, sample_iceberg_df):
        """Test quality check function."""
        result = check_iceberg_data_quality(
            sample_iceberg_df,
            min_observations=10,
            max_gap_hours=48,
            max_speed_kmh=50.0,
        )

        assert 'report' in result
        assert 'data' in result

        report = result['report']
        assert report['initial_records'] > report['final_records']
        assert report['initial_icebergs'] > report['final_icebergs']
        assert len(report['issues_found']) > 0
        assert 'Out of bounds coordinates' in str(report['issues_found'])
        assert 'Duplicate timestamps' in str(report['issues_found'])
        assert 'Too few observations' in str(report['issues_found'])

        # Check cleaned data
        clean_df = result['data']
        assert len(clean_df) < len(sample_iceberg_df)
        assert 'ICE_SHORT' not in clean_df['iceberg_id'].values

    def test_resample_iceberg_trajectory(self):
        """Test trajectory resampling to regular grid."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="3h")  # Irregular 3-hour
        lats = -65 + np.cumsum(np.random.normal(0, 0.01, n))
        lons = 0 + np.cumsum(np.random.normal(0, 0.01, n))

        df = pd.DataFrame({
            'iceberg_id': ['ICE_001'] * n,
            'datetime': dates,
            'lat': lats,
            'lon': lons,
            'length_m': 2000,
            'width_m': 1000,
        })

        # Resample to 6H
        resampled = resample_iceberg_trajectory(df, freq="6h", interpolation="time")

        # Should have fewer points at 6H frequency
        expected_n = len(pd.date_range(dates[0], dates[-1], freq="6h"))
        assert len(resampled) == expected_n
        assert resampled['iceberg_id'].iloc[0] == 'ICE_001'

        # Check regular spacing
        time_diffs = resampled['datetime'].diff().dt.total_seconds() / 3600
        time_diffs = time_diffs.dropna()
        assert all(abs(d - 6) < 0.1 for d in time_diffs)


class TestCoastalFiltering:
    """Test coastal/shallow water filtering."""

    @pytest.fixture
    def sample_bathymetry(self):
        """Create synthetic bathymetry dataset."""
        lats = np.linspace(-70, -50, 21)
        lons = np.linspace(-30, 30, 61)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Depth increases away from coast (simplified)
        dist_from_coast = np.abs(lat_grid + 60) * 111  # km from -60 lat
        depth = np.where(dist_from_coast < 100, 50, 1000 + dist_from_coast)

        ds = xr.Dataset(
            {'bathymetry': (['latitude', 'longitude'], depth)},
            coords={'latitude': lats, 'longitude': lons},
        )
        return ds

    @pytest.fixture
    def sample_positions(self, sample_bathymetry):
        """Create positions at various depths."""
        df = pd.DataFrame({
            'iceberg_id': ['ICE_001'] * 5,
            'datetime': pd.date_range("2020-01-01", periods=5, freq="6h"),
            'lat': [-60.0, -60.5, -62.0, -65.0, -70.0],  # From shallow to deep
            'lon': [0.0, 0.0, 0.0, 0.0, 0.0],
            'length_m': 2000,
            'width_m': 1000,
        })
        return df

    def test_filter_coastal_nan(self, sample_positions, sample_bathymetry):
        """Test coastal filtering."""
        flagged = filter_coastal_nan(
            sample_positions,
            sample_bathymetry,
            depth_threshold=100.0,
        )

        assert 'coastal_flag' in flagged.columns
        assert 'bathymetry_depth' in flagged.columns

        # First few positions should be flagged (shallow or interpolated to < 100)
        # Note: at -60.5 the linear interpolation gives 580.5 which is > 100.
        assert flagged['coastal_flag'].iloc[0] == True
        assert flagged['coastal_flag'].iloc[1] == False

        # Deep positions should not be flagged
        assert flagged['coastal_flag'].iloc[-1] == False


class TestCurrentFilling:
    """Test coastal NaN filling in current data."""

    def test_fill_coastal_current_nan(self):
        """Test filling NaN in coastal current data."""
        # Create synthetic current data with coastal NaN
        times = pd.date_range("2020-01-01", periods=5, freq="D")
        depths = [0, 10, 50]
        lats = np.linspace(-65, -55, 11)
        lons = np.linspace(-20, 20, 41)

        # Create data with NaN near "coast" (high latitudes)
        uo = np.random.normal(0, 0.1, (len(times), len(depths), len(lats), len(lons)))
        vo = np.random.normal(0, 0.1, (len(times), len(depths), len(lats), len(lons)))

        # Set NaN in top 3 latitude rows (simulating coastal)
        uo[:, :, :3, :] = np.nan
        vo[:, :, :3, :] = np.nan

        ds = xr.Dataset(
            {'uo': (['time', 'depth', 'latitude', 'longitude'], uo),
             'vo': (['time', 'depth', 'latitude', 'longitude'], vo)},
            coords={'time': times, 'depth': depths, 'latitude': lats, 'longitude': lons},
        )

        filled = fill_coastal_current_nan(ds, method="nearest_valid")

        # Check that NaN are filled
        assert not np.isnan(filled['uo'].values).any()
        assert not np.isnan(filled['vo'].values).any()

        # Original non-NaN values should be preserved (approximately)
        # Check a deep point that wasn't NaN
        orig_deep = uo[0, 0, -1, 20]
        filled_deep = filled['uo'].values[0, 0, -1, 20]
        assert abs(orig_deep - filled_deep) < 1e-6


class TestEnvironmentalQuality:
    """Test environmental data quality checks."""

    def test_check_environmental_data_quality(self):
        """Test environmental data quality report."""
        # Create synthetic wind and current datasets
        times = pd.date_range("2020-01-01", periods=10, freq="D")
        lats = np.linspace(-65, -55, 11)
        lons = np.linspace(-20, 20, 41)

        wind_ds = xr.Dataset(
            {'u10': (['time', 'latitude', 'longitude'], np.random.normal(0, 8, (10, 11, 41))),
             'v10': (['time', 'latitude', 'longitude'], np.random.normal(0, 8, (10, 11, 41)))},
            coords={'time': times, 'latitude': lats, 'longitude': lons},
        )

        current_ds = xr.Dataset(
            {'uo': (['time', 'depth', 'latitude', 'longitude'], np.random.normal(0, 0.1, (10, 3, 11, 41))),
             'vo': (['time', 'depth', 'latitude', 'longitude'], np.random.normal(0, 0.1, (10, 3, 11, 41)))},
            coords={'time': times, 'depth': [0, 10, 50], 'latitude': lats, 'longitude': lons},
        )

        report = check_environmental_data_quality(wind_ds, current_ds)

        assert 'wind' in report
        assert 'current' in report
        assert 'temporal_overlap' in report
        assert report['wind']['variables'] == ['u10', 'v10']
        assert report['current']['variables'] == ['uo', 'vo']
        assert 'overlap_pct' in report['temporal_overlap']


class TestDataQualityPipeline:
    """Test complete data quality pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for pipeline."""
        np.random.seed(42)

        # Iceberg data
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="3h")  # Irregular
        lats = -65 + np.cumsum(np.random.normal(0, 0.01, n))
        lons = 0 + np.cumsum(np.random.normal(0, 0.01, n))

        iceberg_df = pd.DataFrame({
            'iceberg_id': ['ICE_001'] * n,
            'datetime': dates,
            'lat': lats,
            'lon': lons,
            'length_m': 2000,
            'width_m': 1000,
        })

        # Add a short trajectory iceberg
        short_dates = pd.date_range("2020-01-01", periods=3, freq="6h")
        short_df = pd.DataFrame({
            'iceberg_id': ['ICE_SHORT'] * 3,
            'datetime': short_dates,
            'lat': [-65, -64.9, -64.8],
            'lon': [0, 0.1, 0.2],
            'length_m': 500,
            'width_m': 200,
        })
        iceberg_df = pd.concat([iceberg_df, short_df], ignore_index=True)

        # Wind data
        times = pd.date_range("2020-01-01", periods=20, freq="D")
        lats_w = np.linspace(-70, -50, 11)
        lons_w = np.linspace(-30, 30, 61)
        wind_ds = xr.Dataset(
            {'u10': (['time', 'latitude', 'longitude'], np.random.normal(0, 8, (20, 11, 61))),
             'v10': (['time', 'latitude', 'longitude'], np.random.normal(0, 8, (20, 11, 61)))},
            coords={'time': times, 'latitude': lats_w, 'longitude': lons_w},
        )

        # Current data
        current_ds = xr.Dataset(
            {'uo': (['time', 'depth', 'latitude', 'longitude'], np.random.normal(0, 0.1, (20, 3, 11, 61))),
             'vo': (['time', 'depth', 'latitude', 'longitude'], np.random.normal(0, 0.1, (20, 3, 11, 61)))},
            coords={'time': times, 'depth': [0, 10, 50], 'latitude': lats_w, 'longitude': lons_w},
        )

        # Bathymetry
        lats_b = np.linspace(-70, -50, 21)
        lons_b = np.linspace(-30, 30, 61)
        lon_g, lat_g = np.meshgrid(lons_b, lats_b)
        depth = np.where(lat_g < -60, 1000, 50)  # Shallow near coast
        bathymetry_ds = xr.Dataset(
            {'bathymetry': (['latitude', 'longitude'], depth)},
            coords={'latitude': lats_b, 'longitude': lons_b},
        )

        return iceberg_df, wind_ds, current_ds, bathymetry_ds

    def test_pipeline_run(self, sample_data):
        """Test running the complete pipeline."""
        iceberg_df, wind_ds, current_ds, bathymetry_ds = sample_data

        pipeline = DataQualityPipeline(
            min_observations=10,
            max_gap_hours=48,
            max_speed_kmh=50.0,
            resample_freq="6h",
            coastal_depth_threshold=100.0,
            fill_coastal_nan=True,
        )

        clean_iceberg, clean_wind, clean_current = pipeline.run(
            iceberg_df, wind_ds, current_ds, bathymetry_ds
        )

        # Check results
        assert len(clean_iceberg) < len(iceberg_df)  # Some removed
        assert 'ICE_SHORT' not in clean_iceberg['iceberg_id'].values
        assert 'coastal_flag' in clean_iceberg.columns

        # Check resampling
        time_diffs = clean_iceberg['datetime'].diff().dt.total_seconds() / 3600
        time_diffs = time_diffs.dropna()
        assert all(abs(d - 6) < 0.1 for d in time_diffs)

        # Check reports
        summary = pipeline.get_summary()
        assert 'iceberg_quality' in summary
        assert 'environmental_quality' in summary

    def test_validate_and_clean_convenience(self, sample_data):
        """Test convenience function."""
        iceberg_df, _, _, _ = sample_data
        clean_df, report = validate_and_clean_iceberg_data(
            iceberg_df,
            min_observations=10,
            max_gap_hours=48,
        )

        assert 'report' in dir(report) or isinstance(report, dict)
        assert len(clean_df) < len(iceberg_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])