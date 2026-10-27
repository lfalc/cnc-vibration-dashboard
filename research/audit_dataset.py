#!/usr/bin/env python3
"""Audit MSM label intervals and create a derived dataset manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from research.labeling import load_intervals, union_duration


def timeline_file(dataset_dir: Path) -> Path | None:
    for pattern in ("*_mtc.csv", "*_ct.csv", "*_acc.csv"):
        matches = sorted(dataset_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def timeline_end(path: Path) -> float:
    """Read the final sorted timestamp without loading sensor values."""
    last = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "time" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no time column")
        for row in reader:
            last = float(row["time"])
    if last is None:
        raise ValueError(f"{path} is empty")
    return last


def audit_dataset(root: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    interval_rows = []

    for label_path in sorted(root.glob("*/dataset*/label.csv")):
        dataset_dir = label_path.parent
        dataset_id = f"{dataset_dir.parent.name}/{dataset_dir.name}"
        intervals = load_intervals(label_path)
        timeline = timeline_file(dataset_dir)
        if timeline is None:
            raise FileNotFoundError(f"No timeline CSV found for {dataset_id}")
        duration = timeline_end(timeline)
        out_of_bounds = int((intervals["end"] > duration).sum())
        cutting_seconds = union_duration(intervals)
        label_seconds = intervals.assign(seconds=intervals["end"] - intervals["start"]).groupby("label")["seconds"].sum()
        interval_rows.append(intervals.assign(dataset_id=dataset_id))
        manifest_rows.append({
            "dataset_id": dataset_id,
            "source_family": dataset_dir.parent.name,
            "timeline_file": str(timeline),
            "timeline_seconds": duration,
            "interval_count": len(intervals),
            "cutting_seconds": cutting_seconds,
            "non_cutting_complement_seconds": max(0.0, duration - cutting_seconds),
            "cutting_fraction": cutting_seconds / duration if duration else 0.0,
            "invalid_end_count": out_of_bounds,
            "label_0_seconds": float(label_seconds.get(0, 0.0)),
            "label_1_seconds": float(label_seconds.get(1, 0.0)),
            "label_2_seconds": float(label_seconds.get(2, 0.0)),
        })

    manifest = pd.DataFrame(manifest_rows).sort_values("dataset_id")
    intervals = pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame()
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    intervals.to_csv(output_dir / "interval_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("archive"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs"))
    args = parser.parse_args()
    manifest = audit_dataset(args.root, args.output)
    print(f"Audited {len(manifest)} dataset groups")
    print(f"Timeline seconds: {manifest['timeline_seconds'].sum():,.3f}")
    print(f"Cutting seconds: {manifest['cutting_seconds'].sum():,.3f}")
    print(f"Cutting fraction: {manifest['cutting_seconds'].sum() / manifest['timeline_seconds'].sum():.3%}")
    print(f"Invalid interval ends: {manifest['invalid_end_count'].sum()}")
    print(f"Wrote {args.output / 'dataset_manifest.csv'}")
    print(f"Wrote {args.output / 'interval_manifest.csv'}")


if __name__ == "__main__":
    main()
