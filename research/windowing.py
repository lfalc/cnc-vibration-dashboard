"""Create a leakage-safe window manifest from interval annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _state_for_window(start: float, end: float, intervals: pd.DataFrame) -> tuple[str, int | None]:
    overlaps = intervals[(intervals["start"] < end) & (intervals["end"] > start)]
    cutting_seconds = float(
        sum(min(end, row.end) - max(start, row.start) for row in overlaps.itertuples())
    )
    if cutting_seconds == 0:
        return "non_cutting", None
    if cutting_seconds == end - start:
        labels = set(int(value) for value in overlaps["label"])
        return "cutting", next(iter(labels)) if len(labels) == 1 else None
    return "transition", None


def make_windows(
    dataset_manifest: pd.DataFrame,
    interval_manifest: pd.DataFrame,
    *,
    window_seconds: float = 2.0,
    stride_seconds: float = 1.0,
) -> pd.DataFrame:
    """Generate window labels without loading the raw sensor files."""
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window and stride must be positive")
    required = {"dataset_id", "timeline_seconds"}
    missing = required - set(dataset_manifest.columns)
    if missing:
        raise ValueError(f"dataset manifest missing columns: {sorted(missing)}")
    if not {"dataset_id", "start", "end", "label"}.issubset(interval_manifest.columns):
        raise ValueError("interval manifest must contain dataset_id, start, end, and label")

    rows: list[dict[str, object]] = []
    for dataset in dataset_manifest.itertuples(index=False):
        intervals = interval_manifest[interval_manifest["dataset_id"] == dataset.dataset_id]
        max_start = max(0.0, float(dataset.timeline_seconds) - window_seconds)
        start = 0.0
        while start <= max_start + 1e-9:
            end = start + window_seconds
            state, source_label = _state_for_window(start, end, intervals)
            row = {
                "dataset_id": dataset.dataset_id,
                "window_start": round(start, 6),
                "window_end": round(end, 6),
                "window_seconds": window_seconds,
                "state": state,
                "source_label": source_label,
                "primary_binary": state in {"cutting", "non_cutting"},
            }
            if hasattr(dataset, "split"):
                row["split"] = dataset.split
            rows.append(row)
            start += stride_seconds
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, default=Path("research_outputs/partition_manifest.csv"))
    parser.add_argument("--interval-manifest", type=Path, default=Path("research_outputs/interval_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/window_manifest.csv"))
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    args = parser.parse_args()
    windows = make_windows(
        pd.read_csv(args.dataset_manifest),
        pd.read_csv(args.interval_manifest),
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output, index=False)
    print(f"Generated {len(windows):,} windows")
    print(windows.groupby("state", sort=False).size().to_string())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
