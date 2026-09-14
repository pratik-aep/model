import os
import shutil
from pathlib import Path

data_dir = Path('/Users/pratiksmac/Downloads/data')
raw_dir = data_dir / 'raw'
proc_dir = data_dir / 'processed'

# Define standard dirs
era5_dir = raw_dir / 'era5'
cop_dir = raw_dir / 'copernicus'
bathy_dir = raw_dir / 'bathymetry'
ice_dir = raw_dir / 'iceberg_positions'
misc_dir = raw_dir / 'misc'

# Create them
for d in [era5_dir, cop_dir, bathy_dir, ice_dir, misc_dir, proc_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Move ERA5
for f in data_dir.glob('era5_wind*.nc'):
    shutil.move(str(f), str(era5_dir / f.name))
if (data_dir / 'era5').exists():
    for f in (data_dir / 'era5').glob('*.nc'):
        shutil.move(str(f), str(era5_dir / f.name))
if (data_dir / 'era5_2024').exists():
    for f in (data_dir / 'era5_2024').glob('*.nc'):
        shutil.move(str(f), str(era5_dir / f.name))

# Move Copernicus
for f in data_dir.glob('currents_*.nc'):
    shutil.move(str(f), str(cop_dir / f.name))
if (data_dir / 'copernicus').exists():
    for f in (data_dir / 'copernicus').iterdir():
        shutil.move(str(f), str(cop_dir / f.name))
if (data_dir / 'copernicus_2024').exists():
    for f in (data_dir / 'copernicus_2024').glob('*.nc'):
        shutil.move(str(f), str(cop_dir / f.name))

# Move Bathymetry
for f in data_dir.glob('IBCSO*.nc'):
    shutil.move(str(f), str(bathy_dir / f.name))

# Move Icebergs (Directories)
for d_name in ['updated7_consol', 'iceberg_positions', 'archive', 'archive_2024_2026']:
    src = data_dir / d_name
    if src.exists() and src.is_dir():
        dst = ice_dir / d_name
        if not dst.exists():
            shutil.move(str(src), str(dst))

# Move Parquets
for f in data_dir.glob('*.parquet'):
    shutil.move(str(f), str(proc_dir / f.name))

# Move Misc
for f in data_dir.glob('1940.grib*'):
    shutil.move(str(f), str(misc_dir / f.name))

# Clean up empty dirs
for d_name in ['era5_2024', 'copernicus_2024', 'era5', 'copernicus']:
    d = data_dir / d_name
    if d.exists() and not any(d.iterdir()):
        d.rmdir()

print("Data folder successfully reorganized!")
