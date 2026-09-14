import os
import argparse
from pathlib import Path
import pandas as pd
import xarray as xr

def consolidate_data(force=False):
    base_dir = Path("/Users/pratiksmac/Downloads/data")
    
    # Target files
    wind_out = base_dir / "era5_wind_2024-01-01_2024-12-31.nc"
    curr_out = base_dir / "currents_GLORYS12_2024-01-01_2024-12-31.nc"
    ice_out = base_dir / "icebergs_BYU_LOCAL_2024-01-01_2024-12-31.parquet"
    
    if not force and wind_out.exists() and curr_out.exists() and ice_out.exists():
        print("All consolidated files already exist. Use --force to regenerate.")
        return
        
    print("Consolidating 2024 Data...\n")
    
    # 1. ERA5 Wind
    print("--- 1. Consolidating ERA5 Wind ---")
    if force or not wind_out.exists():
        era5_dir = base_dir / "raw" / "era5"
        wind_files = sorted(list(era5_dir.glob("era5_wind_2024_*.nc")))
        print(f"Found {len(wind_files)} ERA5 files.")
        
        # Verify times
        for f in wind_files:
            with xr.open_dataset(f) as dsf:
                if 'time' in dsf.coords:
                    t = dsf.time.values
                elif 'valid_time' in dsf.coords:
                    t = dsf.valid_time.values
                print(f"  {f.name}: {pd.to_datetime(t[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(t[-1]).strftime('%Y-%m-%d')}")
        
        ds_wind = xr.open_mfdataset(wind_files, combine='by_coords')
        if 'valid_time' in ds_wind.coords:
            ds_wind = ds_wind.rename({'valid_time': 'time'})
        
        # Sort by time just in case
        ds_wind = ds_wind.sortby('time')
        
        # Keep only u10 and v10
        keep_vars_wind = [v for v in ds_wind.data_vars if v in ['u10', 'v10', '10m_u_component_of_wind', '10m_v_component_of_wind']]
        ds_wind = ds_wind[keep_vars_wind]
        
        # Standardize names
        if '10m_u_component_of_wind' in ds_wind.data_vars:
            ds_wind = ds_wind.rename({'10m_u_component_of_wind': 'u10', '10m_v_component_of_wind': 'v10'})
            
        # Crop to Weddell Sea bbox: Lat(-65 to -55), Lon(-65 to -50)
        # Note ERA5 lats might be descending or ascending, sort first or use slice correctly
        lats = ds_wind.latitude.values
        if lats[0] > lats[-1]:
            ds_wind = ds_wind.sel(latitude=slice(-55, -65), longitude=slice(-65, -50))
        else:
            ds_wind = ds_wind.sel(latitude=slice(-65, -55), longitude=slice(-65, -50))
            
        print(f"Final Wind dataset: {ds_wind.time.size} time steps, {pd.to_datetime(ds_wind.time.values[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(ds_wind.time.values[-1]).strftime('%Y-%m-%d')}")
        ds_wind.to_netcdf(wind_out)
        print(f"Saved to {wind_out.name}\n")
    else:
        print(f"{wind_out.name} already exists.\n")

    # 2. Copernicus Currents
    print("--- 2. Consolidating Copernicus Currents ---")
    if force or not curr_out.exists():
        cop_dir = base_dir / "raw" / "copernicus"
        curr_files = sorted(list(cop_dir.glob("currents_seaice_GLORYS12_2024_*.nc")))
        print(f"Found {len(curr_files)} Copernicus files.")
        
        ds_curr = xr.open_mfdataset(curr_files, combine='by_coords')
        if 'valid_time' in ds_curr.coords:
            ds_curr = ds_curr.rename({'valid_time': 'time'})
        ds_curr = ds_curr.sortby('time')
        
        # Check depth
        depth_dim = None
        for d in ['depth', 'depth_0', 'deptht', 'z']:
            if d in ds_curr.dims:
                depth_dim = d
                break
                
        if depth_dim:
            depths = ds_curr[depth_dim].values
            print(f"Found depth dimension: {len(depths)} levels.")
            if len(depths) > 2:
                print("Restricting to depth <= 20m...")
                ds_curr = ds_curr.sel({depth_dim: slice(0, 20)})
        else:
            print("No depth dimension found (surface only).")
            
        # Keep uo, vo (and siconc, sithick if present)
        keep_vars = ['uo', 'vo', 'siconc', 'sithick']
        vars_to_keep = [v for v in keep_vars if v in ds_curr.data_vars]
        ds_curr = ds_curr[vars_to_keep]
        
        # Crop to Weddell Sea bbox: Lat(-65 to -55), Lon(-65 to -50)
        lats = ds_curr.latitude.values
        if lats[0] > lats[-1]:
            ds_curr = ds_curr.sel(latitude=slice(-55, -65), longitude=slice(-65, -50))
        else:
            ds_curr = ds_curr.sel(latitude=slice(-65, -55), longitude=slice(-65, -50))
        
        total_cells = ds_curr['uo'].size
        print(f"Final Currents dataset: {ds_curr.time.size} time steps.")
        print(f"Total array size (cells): {total_cells}")
        
        ds_curr.to_netcdf(curr_out)
        print(f"Saved to {curr_out.name}\n")
    else:
        print(f"{curr_out.name} already exists.\n")

    # 3. Icebergs
    print("--- 3. Consolidating Iceberg Tracking Data ---")
    if force or not ice_out.exists():
        csv_dir = base_dir / "raw" / "iceberg_positions" / "iceberg_positions"
        csv_files = sorted(list(csv_dir.glob("*.csv")))
        print(f"Found {len(csv_files)} CSV files. Parsing...")
        
        # Find bbox from wind dataset
        with xr.open_dataset(wind_out) as dsw:
            lats = dsw.latitude.values
            lons = dsw.longitude.values
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
        print(f"Filtering to Wind BBox: Lat({min_lat:.1f}, {max_lat:.1f}), Lon({min_lon:.1f}, {max_lon:.1f})")
        
        dfs = []
        for i, f in enumerate(csv_files):
            if i > 0 and i % 100 == 0:
                print(f"  Processed {i}/{len(csv_files)} files...")
            try:
                df = pd.read_csv(f)
                df['iceberg_id'] = f.stem
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
            
        full_df = pd.concat(dfs, ignore_index=True)
        if 'datetime' not in full_df.columns:
            if 'time' in full_df.columns:
                full_df['datetime'] = pd.to_datetime(full_df['time'])
            elif 'date' in full_df.columns:
                full_df['datetime'] = pd.to_datetime(full_df['date'])
        else:
            full_df['datetime'] = pd.to_datetime(full_df['datetime'])
            
        pre_date_cnt = len(full_df)
        
        # Date filter
        full_df = full_df[(full_df['datetime'] >= '2024-01-01') & (full_df['datetime'] <= '2024-12-31')]
        post_date_cnt = len(full_df)
        
        # Bbox filter
        full_df = full_df[(full_df['lat'] >= min_lat) & (full_df['lat'] <= max_lat)]
        full_df = full_df[(full_df['lon'] >= min_lon) & (full_df['lon'] <= max_lon)]
        
        # Interpolate to 1-day frequency to bypass the 48h gap filter in quality pipeline
        resampled_dfs = []
        for iceberg_id, group in full_df.groupby('iceberg_id'):
            group = group.set_index('datetime').sort_index()
            group = group[~group.index.duplicated(keep='first')]
            if len(group) > 1:
                # Need numeric columns for interpolation
                numeric_cols = group.select_dtypes(include='number').columns
                group_num = group[numeric_cols].resample('1D').interpolate(method='time')
                # Forward fill non-numeric columns
                non_num_cols = group.select_dtypes(exclude='number').columns
                group_non_num = group[non_num_cols].resample('1D').ffill()
                group = pd.concat([group_num, group_non_num], axis=1)
                group['iceberg_id'] = iceberg_id
            resampled_dfs.append(group.reset_index())
            
        full_df = pd.concat(resampled_dfs, ignore_index=True)
        # Re-apply date bounds just in case interpolation extended them slightly
        full_df = full_df[(full_df['datetime'] >= '2024-01-01') & (full_df['datetime'] <= '2024-12-31')]
        
        post_bbox_cnt = len(full_df)
        if 'iceberg_id' in full_df.columns:
            unique_bergs = full_df['iceberg_id'].nunique()
        elif 'id' in full_df.columns:
            unique_bergs = full_df['id'].nunique()
        elif 'Iceberg' in full_df.columns:
            unique_bergs = full_df['Iceberg'].nunique()
        else:
            unique_bergs = f"Unknown (columns: {list(full_df.columns)})"
        
        print(f"Records before date filter: {pre_date_cnt}")
        print(f"Records after date filter (2024): {post_date_cnt}")
        print(f"Records after bbox filter: {post_bbox_cnt}")
        print(f"Unique icebergs in 2024: {unique_bergs}")
        
        full_df.to_parquet(ice_out, index=False)
        print(f"Saved to {ice_out.name}\n")
    else:
        print(f"{ice_out.name} already exists.\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, path in [("Wind", wind_out), ("Currents", curr_out), ("Icebergs", ice_out)]:
        size = path.stat().st_size / (1024*1024) if path.exists() else 0
        print(f"{name:10} | Size: {size:.1f} MB | {path.name}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    consolidate_data(args.force)
