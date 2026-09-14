import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/pratiksmac/Downloads/icccy/iceberg-drift/src")))

from iceberg_drift.data_processing.download import download_era5_wind
import logging

logging.basicConfig(level=logging.INFO)

# We want to download the second half of 2024
# According to earlier logs, we have Jan-Jun 2024 in era5_2024.
# Let's download Jul-Dec 2024.
# We will save it in /Users/pratiksmac/Downloads/data/era5_2024
output_dir = "/Users/pratiksmac/Downloads/data/era5_2024"
start_date = "2024-07-01"
end_date = "2024-12-31"

# General bounding box for Southern Ocean
bboxes = [(-180, -90, 180, -50)]

print(f"Downloading ERA5 wind from {start_date} to {end_date}...")
try:
    ds = download_era5_wind(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        bboxes=bboxes,
        allow_synthetic_fallback=False
    )
    print("Download successful!")
except Exception as e:
    print(f"Failed: {e}")
