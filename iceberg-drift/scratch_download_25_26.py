import sys
import logging
from pathlib import Path
import traceback

sys.path.insert(0, str(Path("/Users/pratiksmac/Downloads/icccy/iceberg-drift/src")))
from iceberg_drift.data_processing.download import download_era5_wind, download_copernicus_currents

logging.basicConfig(level=logging.INFO)

# Define directories
era5_dir = "/Users/pratiksmac/Downloads/data/era5"
cop_dir = "/Users/pratiksmac/Downloads/data/copernicus"
bboxes = [(-180, -90, 180, -50)]

# We will try 2025 full year, and 2026 Jan-Aug
periods = [
    ("2025-01-01", "2025-12-31"),
    ("2026-01-01", "2026-08-31")
]

print("Starting background download for 2025 and 2026 data...")

for start, end in periods:
    print(f"\n--- Downloading ERA5 Wind for {start} to {end} ---")
    try:
        download_era5_wind(
            output_dir=era5_dir,
            start_date=start,
            end_date=end,
            bboxes=bboxes,
            allow_synthetic_fallback=False
        )
        print(f"Successfully finished ERA5 for {start} to {end}")
    except Exception as e:
        print(f"Failed ERA5 {start}-{end}: {e}")
        
    print(f"\n--- Downloading Copernicus Currents for {start} to {end} ---")
    try:
        download_copernicus_currents(
            output_dir=cop_dir,
            start_date=start,
            end_date=end,
            bboxes=bboxes,
            allow_synthetic_fallback=False
        )
        print(f"Successfully finished Copernicus for {start} to {end}")
    except Exception as e:
        print(f"Failed Copernicus {start}-{end}: {e}")

print("\nAll download requests processed!")
