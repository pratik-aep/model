#!/usr/bin/env python
"""
Transform script for BYU updated7_consol iceberg position data.
Converts the sensor-pair format to standard latitude/longitude/date format.
Handles various file structures found in the dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re

def detect_file_structure(df):
    """
    Detect the structure of the CSV file to determine how to extract position data.

    Returns:
        dict: Information about the file structure including:
            - date_col: name of date column
            - lat_col: name of latitude column (or None)
            - lon_col: name of longitude column (or None)
            - flag_col: name of flag column (or None)
            - sensor_prefix: prefix used for sensor columns (nic, sass, ers, etc.)
            - has_triplet: whether data is in (lat, lon, flag) triplets
    """
    cols = list(df.columns)

    # Find date column (case insensitive)
    date_col = None
    for col in cols:
        if col.lower() == 'date':
            date_col = col
            break

    if date_col is None:
        # Try to find first column that looks like a date
        for col in cols:
            # Check if first few values look like YYYYDDD or similar numeric date
            sample_vals = df[col].dropna().head(3)
            if len(sample_vals) > 0:
                # Convert to string and check pattern
                str_vals = [str(int(v)) if isinstance(v, (int, float)) and not pd.isna(v) else str(v)
                           for v in sample_vals]
                # Check if they look like 5-7 digit numbers (YYYYDDD or similar)
                if all(re.match(r'^\d{5,7}$', v) for v in str_vals):
                    date_col = col
                    break

    # Check for standard sensor triplets (nic, sass, ascat, oscat, qscat, ers)
    sensor_prefixes = ['nic', 'sass', 'ascat', 'oscat', 'qscat', 'ers']

    for prefix in sensor_prefixes:
        lat_col = f"{prefix}_1"
        lon_col = f"{prefix}_2"
        flag_col = f"{prefix}_3"

        if lat_col in df.columns and lon_col in df.columns and flag_col in df.columns:
            # Check if this looks like a valid triplet by examining some values
            lat_sample = df[lat_col].dropna().head(5)
            lon_sample = df[lon_col].dropna().head(5)
            flag_sample = df[flag_col].dropna().head(5)

            # If we have non-zero values in lat/lon and flags are 0/1, this is likely our structure
            if (len(lat_sample) > 0 and len(lon_sample) > 0 and len(flag_sample) > 0 and
                ((flag_sample.isin([0, 1]).all()) or (flag_sample.abs().max() <= 1))):
                return {
                    'date_col': date_col,
                    'lat_col': lat_col,
                    'lon_col': lon_col,
                    'flag_col': flag_col,
                    'sensor_prefix': prefix,
                    'has_triplet': True
                }

    # Fallback: look for any column pairs that might be lat/lon
    # Look for columns with names suggesting latitude/longitude
    lat_candidates = [c for c in cols if 'lat' in c.lower()]
    lon_candidates = [c for c in cols if 'lon' in c.lower()]

    if lat_candidates and lon_candidates:
        return {
            'date_col': date_col,
            'lat_col': lat_candidates[0],
            'lon_col': lon_candidates[0],
            'flag_col': None,
            'sensor_prefix': 'unknown',
            'has_triplet': False
        }

    # If we still haven't found anything, return basic info
    return {
        'date_col': date_col,
        'lat_col': None,
        'lon_col': None,
        'flag_col': None,
        'sensor_prefix': 'unknown',
        'has_triplet': False
    }

def extract_position_from_triplet(row, lat_col, lon_col, flag_col):
    """Extract position from (lat, lon, flag) triplet if flag indicates valid data."""
    try:
        lat_val = row[lat_col]
        lon_val = row[lon_col]
        flag_val = row[flag_col]

        # Check if flag indicates valid data (typically 1 = valid, 0 = invalid/missing)
        if pd.notna(flag_val) and flag_val == 1:
            if pd.notna(lat_val) and pd.notna(lon_val):
                # Additional sanity checks for coordinates
                lat_float = float(lat_val)
                lon_float = float(lon_val)
                if -90 <= lat_float <= 90 and -180 <= lon_float <= 180:
                    # Avoid obvious fill values like 0,0 or 999,999
                    if abs(lat_float) > 0.01 or abs(lon_float) > 0.01:
                        return pd.Series({'lat': lat_float, 'lon': lon_float})
    except (ValueError, TypeError):
        pass

    return pd.Series({'lat': np.nan, 'lon': np.nan})

def extract_position_from_pair(row, lat_col, lon_col):
    """Extract position from lat/lon pair, doing basic validation."""
    try:
        lat_val = row[lat_col]
        lon_val = row[lon_col]

        if pd.notna(lat_val) and pd.notna(lon_val):
            lat_float = float(lat_val)
            lon_float = float(lon_val)
            if -90 <= lat_float <= 90 and -180 <= lon_float <= 180:
                if abs(lat_float) > 0.01 or abs(lon_float) > 0.01:
                    return pd.Series({'lat': lat_float, 'lon': lon_float})
    except (ValueError, TypeError):
        pass

    return pd.Series({'lat': np.nan, 'lon': np.nan})

def transform_file(input_path, output_path=None):
    """
    Transform a single updated7_consol CSV file.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (if None, overwrites input)

    Returns:
        Path to transformed file
    """
    if output_path is None:
        output_path = input_path

    print(f"Processing {input_path.name}...")

    # Read the CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"  ❌ Failed to read CSV: {e}")
        return None

    if len(df) == 0:
        print(f"  ⚠️  Empty file")
        return None

    # Detect file structure
    struct_info = detect_file_structure(df)

    if struct_info['date_col'] is None:
        print(f"  ❌ Could not find date column")
        return None

    date_col = struct_info['date_col']

    # Extract position data
    if struct_info['has_triplet'] and struct_info['lat_col'] and struct_info['lon_col'] and struct_info['flag_col']:
        # Use triplet extraction (lat, lon, flag)
        df[['lat', 'lon']] = df.apply(
            lambda row: extract_position_from_triplet(
                row,
                struct_info['lat_col'],
                struct_info['lon_col'],
                struct_info['flag_col']
            ),
            axis=1
        )
        print(f"  ✅ Using {struct_info['sensor_prefix']}_triplet format")
    elif struct_info['lat_col'] and struct_info['lon_col']:
        # Use pair extraction (lat, lon only)
        df[['lat', 'lon']] = df.apply(
            lambda row: extract_position_from_pair(
                row,
                struct_info['lat_col'],
                struct_info['lon_col']
            ),
            axis=1
        )
        print(f"  ✅ Using {struct_info['lat_col']}/{struct_info['lon_col']} pair format")
    else:
        print(f"  ❌ Could not determine position columns")
        print(f"     Available columns: {list(df.columns)}")
        return None

    # Convert date from YYYYDDD or similar format to datetime
    try:
        # First, ensure date column is string and clean it
        date_series = df[date_col].astype(str).str.strip()

        # Try to parse as YYYYDDD first
        df['date'] = pd.to_datetime(date_series, format='%Y%j', errors='coerce')

        # If that failed for many rows, try inferring the format
        null_count = df['date'].isna().sum()
        if null_count > len(df) * 0.5:  # More than half failed
            print(f"  ⚠️  YYYYDDD format failed for {null_count} rows, trying mixed format...")
            df['date'] = pd.to_datetime(date_series, format='mixed', errors='coerce')

    except Exception as e:
        print(f"  ❌ Date conversion failed: {e}")
        return None

    # Drop rows without valid position or date
    initial_count = len(df)
    df = df.dropna(subset=['lat', 'lon', 'date'])
    final_count = len(df)
    print(f"  Dropped {initial_count - final_count} rows with missing position/date")

    if final_count == 0:
        print(f"  ⚠️  No valid data remaining after cleaning")
        # Still create the file with headers for consistency
        df = pd.DataFrame(columns=['lat', 'lon', 'date'])

    # Select and reorder columns for iceberg drift model
    # Keep useful auxiliary columns
    size_cols = [col for col in df.columns if col.startswith('size_')]
    # Keep sensor measurement columns (excluding flag columns)
    sensor_cols = []
    for prefix in ['nic', 'sass', 'ascat', 'oscat', 'qscat', 'ers']:
        sensor_cols.extend([col for col in df.columns
                           if col.startswith(prefix) and not col.endswith('_3')])

    # Base columns we always need
    base_cols = ['lat', 'lon', 'date']

    # Optional columns to keep (auxiliary data that might be useful)
    optional_cols = size_cols + sensor_cols

    # Final column order
    final_cols = base_cols + optional_cols

    # Filter to only columns that exist
    final_cols = [col for col in final_cols if col in df.columns]

    df_transformed = df[final_cols].copy()

    # Save transformed data
    df_transformed.to_csv(output_path, index=False)
    print(f"  💾 Saved {len(df_transformed)} rows to {output_path.name}")

    return output_path

def transform_directory(input_dir, output_dir=None, file_pattern="*.csv"):
    """
    Transform all CSV files in a directory.

    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory for output CSV files (if None, overwrites input)
        file_pattern: Glob pattern for files to process
    """
    input_path = Path(input_dir)
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    # Find all matching files
    files = list(input_path.glob(file_pattern))
    print(f"Found {len(files)} files to process")

    successful = 0
    failed = 0

    for file_path in files:
        if output_dir:
            out_file = output_path / file_path.name
        else:
            out_file = file_path

        result = transform_file(file_path, out_file)
        if result is not None:
            successful += 1
        else:
            failed += 1
        print()  # Empty line for readability

    print(f"Transformation complete!")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    return successful, failed

def main():
    parser = argparse.ArgumentParser(description='Transform BYU updated7_consol iceberg data')
    parser.add_argument('input', help='Input CSV file or directory')
    parser.add_argument('--output', '-o', help='Output file or directory (default: overwrite input)')
    parser.add_argument('--pattern', default='*.csv', help='File pattern for directory processing (default: *.csv)')

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: {args.input} does not exist")
        return 1

    if input_path.is_file():
        # Single file
        result = transform_file(str(input_path), args.output)
        if result is None:
            return 1
    elif input_path.is_dir():
        # Directory of files
        successful, failed = transform_directory(str(input_path), args.output, args.pattern)
        if failed > 0:
            return 1
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1

    print("\n🎉 Transformation completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())