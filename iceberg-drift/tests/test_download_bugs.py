import pytest
import pandas as pd
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock

from iceberg_drift.data_processing.download import (
    get_chunked_bboxes,
    download_era5_wind,
    _validate_date_range,
)

def test_bug1_chunked_bboxes():
    # Synthetic iceberg dataframe spanning the globe
    df = pd.DataFrame({
        "lon": [-170, -50, 60, 170],
        "lat": [-60, -65, -70, -60]
    })
    bboxes = get_chunked_bboxes(df, buffer=5.0, band_width=30.0)
    # Expected: points at -170, -50, 60, 170
    assert len(bboxes) == 4
    # Check that they don't cover the whole globe and NO overlap in longitude
    bboxes_sorted = sorted(bboxes, key=lambda x: x[0])
    for i in range(len(bboxes_sorted) - 1):
        assert bboxes_sorted[i][2] <= bboxes_sorted[i+1][0]
    
    for box in bboxes:
        span = box[2] - box[0]
        assert span <= 30.0 + 10.0  # bandwidth + 2*buffer

def test_bug1_chunked_combine_by_coords():
    df = pd.DataFrame({
        "lon": [-170, -50, 60, 170],
        "lat": [-60, -65, -70, -60]
    })
    bboxes = get_chunked_bboxes(df, buffer=5.0, band_width=30.0)
    
    chunks = []
    for i, box in enumerate(bboxes):
        # Create a small synthetic dataset for this exact bbox
        lons = np.linspace(box[0], box[2], 5)
        lats = np.linspace(box[1], box[3], 5)
        ds = xr.Dataset(
            data_vars={"test_var": (("latitude", "longitude"), np.random.rand(len(lats), len(lons)))},
            coords={"latitude": lats, "longitude": lons}
        )
        chunks.append(ds)
    
    # This should succeed without ValueError about monotonic indexes
    combined = xr.combine_by_coords(chunks, join="outer")
    assert combined is not None

@patch("cdsapi.Client")
def test_bug2_synthetic_fallback(mock_client_class, tmp_path):
    mock_client = MagicMock()
    mock_client.retrieve.side_effect = Exception("Simulated 403 Forbidden")
    mock_client_class.return_value = mock_client
    
    # allow_synthetic_fallback=False should raise
    with pytest.raises(RuntimeError, match="ERA5 download failed: Simulated 403 Forbidden"):
        download_era5_wind(
            output_dir=str(tmp_path),
            start_date="2020-01-01",
            end_date="2020-01-02",
            bbox=(-10, -10, 10, 10),
            allow_synthetic_fallback=False
        )

    # allow_synthetic_fallback=True should not raise and return a synthetic dataset
    ds = download_era5_wind(
        output_dir=str(tmp_path),
        start_date="2020-01-01",
        end_date="2020-01-02",
        bbox=(-10, -10, 10, 10),
        allow_synthetic_fallback=True
    )
    assert "10m_u_component_of_wind" in ds.data_vars

def test_bug3_date_range_validation():
    # Mock cached dataset covering Jan 2015
    times = pd.date_range("2015-01-01", "2015-01-31", freq="D")
    ds = xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), np.zeros((len(times), 1, 1)))},
        coords={"time": times, "latitude": [0], "longitude": [0]}
    )
    
    # Requested range within cache -> no error
    _validate_date_range(ds, "2015-01-05", "2015-01-10")
    
    # Requested range outside cache -> raises ValueError
    with pytest.raises(ValueError, match="does not cover requested range"):
        _validate_date_range(ds, "2018-01-01", "2020-12-31")

def test_synthetic_fallback_multi_chunk():
    """Verify that synthetic fallback generates data covering ALL bboxes, not just the first one."""
    df = pd.DataFrame({
        "lon": [-170, -50, 60, 170],
        "lat": [-60, -65, -70, -60]
    })
    bboxes = get_chunked_bboxes(df, buffer=5.0, band_width=30.0)
    assert len(bboxes) == 4
    
    from iceberg_drift.data_processing.download import _generate_synthetic_era5
    ds = _generate_synthetic_era5("2020-01-01", "2020-01-02", bboxes, ["u10"])
    
    min_lon = ds.longitude.values.min()
    max_lon = ds.longitude.values.max()
    
    assert min_lon <= -170
    assert max_lon >= 170

