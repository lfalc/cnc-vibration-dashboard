"""Extract chunked accelerometer features for the prepared window manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def accelerometer_file(root: Path, dataset_id: str) -> Path:
    dataset_dir = root / dataset_id
    matches = sorted(dataset_dir.glob("*_acc.csv"))
    if not matches:
        raise FileNotFoundError(f"no accelerometer CSV for {dataset_id}")
    return matches[0]


def _aggregate(path: Path, starts: np.ndarray, window_seconds: float, chunk_size: int) -> pd.DataFrame:
    n_windows = len(starts)
    columns = ["acc0", "acc1"]
    count = np.zeros((n_windows, len(columns)), dtype=np.int64)
    total = np.zeros((n_windows, len(columns)), dtype=np.float64)
    square = np.zeros_like(total)
    minimum = np.full_like(total, np.inf)
    maximum = np.full_like(total, -np.inf)

    for chunk in pd.read_csv(path, usecols=["time", *columns], chunksize=chunk_size):
        times = chunk["time"].to_numpy(dtype=float)
        values = chunk[columns].to_numpy(dtype=float)
        base = np.floor(times / (starts[1] - starts[0] if len(starts) > 1 else window_seconds)).astype(int)
        for offset in (0, -1):
            ids = base + offset
            valid = (ids >= 0) & (ids < n_windows)
            valid &= (times >= starts[np.clip(ids, 0, n_windows - 1)])
            valid &= times < starts[np.clip(ids, 0, n_windows - 1)] + window_seconds
            ids = ids[valid]
            samples = values[valid]
            if len(ids) == 0:
                continue
            np.add.at(count, (ids[:, None], np.arange(len(columns))[None, :]), 1)
            np.add.at(total, (ids[:, None], np.arange(len(columns))[None, :]), samples)
            np.add.at(square, (ids[:, None], np.arange(len(columns))[None, :]), samples * samples)
            for channel in range(len(columns)):
                np.minimum.at(minimum[:, channel], ids, samples[:, channel])
                np.maximum.at(maximum[:, channel], ids, samples[:, channel])

    mean = total / count
    variance = np.maximum(square / count - mean * mean, 0.0)
    rms = np.sqrt(square / count)
    peak = np.maximum(np.abs(minimum), np.abs(maximum))
    output = {}
    for channel, name in enumerate(columns):
        output[f"{name}_sample_count"] = count[:, channel]
        output[f"{name}_mean"] = mean[:, channel]
        output[f"{name}_std"] = np.sqrt(variance[:, channel])
        output[f"{name}_rms"] = rms[:, channel]
        output[f"{name}_peak_to_peak"] = maximum[:, channel] - minimum[:, channel]
        output[f"{name}_crest_factor"] = peak[:, channel] / np.maximum(rms[:, channel], 1e-12)
    for name in columns:
        missing = output[f"{name}_sample_count"] == 0
        for feature in ("mean", "std", "rms", "peak_to_peak", "crest_factor"):
            output[f"{name}_{feature}"] = np.where(
                missing, np.nan, output[f"{name}_{feature}"]
            )
    output["accelerometer_available"] = (count > 0).all(axis=1)
    return pd.DataFrame(output)


def extract_features(
    windows: pd.DataFrame,
    root: Path,
    *,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    required = {"dataset_id", "window_start", "window_end"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"window manifest missing columns: {sorted(missing)}")
    results = []
    for dataset_id, group in windows.groupby("dataset_id", sort=False):
        group = group.sort_values("window_start").reset_index(drop=True)
        starts = group["window_start"].to_numpy(dtype=float)
        window_seconds = float(group["window_end"].iloc[0] - starts[0])
        try:
            path = accelerometer_file(root, dataset_id)
        except FileNotFoundError:
            features = pd.DataFrame(
                {
                    f"{channel}_{feature}": np.nan
                    for channel in ("acc0", "acc1")
                    for feature in ("sample_count", "mean", "std", "rms", "peak_to_peak", "crest_factor")
                },
                index=group.index,
            )
            features["accelerometer_available"] = False
        else:
            features = _aggregate(path, starts, window_seconds, chunk_size)
        results.append(pd.concat([group, features], axis=1))
    return pd.concat(results, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("archive"))
    parser.add_argument("--windows", type=Path, default=Path("research_outputs/window_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/accelerometer_features.csv"))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()
    features = extract_features(pd.read_csv(args.windows), args.root, chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Extracted {len(features):,} feature rows")
    print(f"Unavailable accelerometer windows: {(~features['accelerometer_available']).sum():,}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
