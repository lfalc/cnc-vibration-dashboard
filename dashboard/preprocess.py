#!/usr/bin/env python3
"""
Preprocessing script for vibration dashboard.

Reads compressed CSV files once and generates compact derived cache files
for the Streamlit dashboard. The cache files contain:
- Daily median, 10th percentile, 90th percentile, and sample count for all features
- Time-binned (1-minute) spectral band summaries for each machine, axis, and band
- Time-binned (1-minute) traces for all machine features
"""

import pandas as pd
from pathlib import Path
import json
from vibration_dashboard.config import ALL_FEATURES, AXES, BANDS, DATA_DIR


def read_machine_file(filepath: Path) -> pd.DataFrame:
    """Read a single machine CSV.gz file."""
    print(f"Reading {filepath}...")
    # Read with gzip decompression
    df = pd.read_csv(filepath, compression='gzip', parse_dates=['time'])
    print(f"  Loaded {len(df)} rows, date range: {df['time'].min()} to {df['time'].max()}")
    return df


def compute_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily statistics (median, p10, p90, count) for all features."""
    # Set time as index for resampling
    df = df.set_index('time')
    
    # Resample to daily frequency and compute statistics
    daily_data = {}
    
    # For each feature column (excluding time which is now index)
    feature_cols = [c for c in df.columns if c != 'time']
    
    for col in feature_cols:
        # Compute daily median, p10, p90, count
        daily_median = df[col].resample('D').median()
        daily_p10 = df[col].resample('D').quantile(0.10)
        daily_p90 = df[col].resample('D').quantile(0.90)
        daily_count = df[col].resample('D').count()
        
        daily_data[f'{col}_median'] = daily_median
        daily_data[f'{col}_p10'] = daily_p10
        daily_data[f'{col}_p90'] = daily_p90
        daily_data[f'{col}_count'] = daily_count
    
    daily_df = pd.DataFrame(daily_data)
    daily_df.index.name = 'date'
    daily_df = daily_df.reset_index()
    
    return daily_df


def compute_spectral_bins(df: pd.DataFrame, bin_minutes: int = 1) -> pd.DataFrame:
    """Compute time-binned (1-min) median spectral band values for heatmap."""
    df = df.set_index('time')
    
    # Only compute for spectral bands
    band_cols = []
    for axis in AXES:
        for band in BANDS:
            col = f'{axis}_{band}'
            if col in df.columns:
                band_cols.append(col)
    
    binned_data = {}
    
    for col in band_cols:
        binned = df[col].resample(f'{bin_minutes}min').median()
        binned_data[col] = binned
    
    binned_df = pd.DataFrame(binned_data)
    binned_df.index.name = 'time_bin'
    binned_df = binned_df.reset_index()
    
    return binned_df


def compute_feature_bins(df: pd.DataFrame, bin_minutes: int = 1) -> pd.DataFrame:
    """Compute 1-minute median traces for every recorded feature."""
    indexed = df.set_index('time')
    feature_cols = [c for c in indexed.columns if c != 'time']
    binned_df = indexed[feature_cols].resample(f'{bin_minutes}min').median()
    binned_df.index.name = 'time_bin'
    return binned_df.reset_index()


def process_machine(machine_name: str, input_path: Path, cache_dir: Path):
    """Process a single machine file and write cache files."""
    print(f"\nProcessing {machine_name}...")
    
    df = read_machine_file(input_path)
    
    # Compute daily statistics
    print("  Computing daily statistics...")
    daily_stats = compute_daily_stats(df)
    daily_path = cache_dir / f'{machine_name}_daily.parquet'
    daily_stats.to_parquet(daily_path, index=False)
    print(f"  Saved daily stats to {daily_path} ({len(daily_stats)} days)")
    
    # Compute spectral heatmap bins
    print("  Computing spectral heatmap bins (1-min)...")
    spectral_bins = compute_spectral_bins(df, bin_minutes=1)
    spectral_path = cache_dir / f'{machine_name}_spectral_1min.parquet'
    spectral_bins.to_parquet(spectral_path, index=False)
    print(f"  Saved spectral bins to {spectral_path} ({len(spectral_bins)} bins)")

    print("  Computing feature traces (1-min)...")
    feature_bins = compute_feature_bins(df, bin_minutes=1)
    feature_path = cache_dir / f'{machine_name}_features_1min.parquet'
    feature_bins.to_parquet(feature_path, index=False)
    print(f"  Saved feature traces to {feature_path} ({len(feature_bins)} bins)")

    # Save metadata
    metadata = {
        'machine': machine_name,
        'source_file': str(input_path),
        'row_count': len(df),
        'date_range': {
            'start': df['time'].min().isoformat(),
            'end': df['time'].max().isoformat()
        },
        'daily_stats_rows': len(daily_stats),
        'spectral_bins_rows': len(spectral_bins),
        'feature_bins_rows': len(feature_bins),
        'bin_minutes': 1,
        'features': ALL_FEATURES,
        'axes': AXES,
        'bands': BANDS,
        'note': 'Dates are shifted. No raw waveforms, machine-state, fault, or anomaly labels in dataset. Band values are relative dB.'
    }
    metadata_path = cache_dir / f'{machine_name}_metadata.json'
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata to {metadata_path}")
    except (OSError, TypeError, ValueError) as e:
        raise RuntimeError(f"Failed to write metadata for {machine_name}: {e}") from e


def main():
    data_dir = DATA_DIR
    cache_dir = Path(__file__).resolve().parent / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    machine_files = sorted(data_dir.glob('machine_*.csv.gz'))
    
    print(f"Found {len(machine_files)} machine files")
    print(f"Cache directory: {cache_dir}")
    
    for machine_file in machine_files:
        machine_name = machine_file.stem.replace('.csv', '')  # machine_01
        process_machine(machine_name, machine_file, cache_dir)
    
    print("\nPreprocessing complete!")
    
    # Print summary
    print("\n=== Cache Summary ===")
    for machine_file in machine_files:
        machine_name = machine_file.stem.replace('.csv', '')
        meta_path = cache_dir / f'{machine_name}_metadata.json'
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                print(f"  {machine_name}: {meta['row_count']:,} rows, "
                      f"{meta['daily_stats_rows']} daily stats, "
                      f"{meta['spectral_bins_rows']} spectral bins")
            except (OSError, json.JSONDecodeError) as e:
                print(f"  {machine_name}: failed to read metadata: {e}")


if __name__ == '__main__':
    main()
