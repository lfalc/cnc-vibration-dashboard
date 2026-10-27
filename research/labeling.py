"""Convert MSM cutting-path intervals into auditable binary state labels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_LABEL_COLUMNS = {"start", "end", "label"}
VALID_ANOMALY_LABELS = {0, 1, 2}


def load_intervals(path: Path) -> pd.DataFrame:
    """Load and validate one MSM ``label.csv`` file.

    Intervals use left-closed/right-open boundaries: ``start <= t < end``.
    Every original annotation column is retained.
    """
    frame = pd.read_csv(path)
    missing = REQUIRED_LABEL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["start"] = pd.to_numeric(frame["start"], errors="raise")
    frame["end"] = pd.to_numeric(frame["end"], errors="raise")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if (frame["start"] < 0).any() or (frame["end"] <= frame["start"]).any():
        raise ValueError(f"{path} contains invalid interval boundaries")
    unknown = set(frame["label"]) - VALID_ANOMALY_LABELS
    if unknown:
        raise ValueError(f"{path} contains unsupported labels: {sorted(unknown)}")

    frame = frame.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    if len(frame) > 1 and (frame["start"].iloc[1:].to_numpy() < frame["end"].iloc[:-1].to_numpy()).any():
        raise ValueError(f"{path} contains overlapping intervals")
    frame["binary_state"] = "cutting"
    return frame


def label_times(times: Iterable[float], intervals: pd.DataFrame) -> pd.DataFrame:
    """Assign binary state and source anomaly label to arbitrary timestamps."""
    values = np.asarray(list(times), dtype=float)
    starts = intervals["start"].to_numpy(dtype=float)
    ends = intervals["end"].to_numpy(dtype=float)
    labels = intervals["label"].to_numpy(dtype=int)
    positions = np.searchsorted(starts, values, side="right") - 1
    inside = positions >= 0
    safe_positions = np.maximum(positions, 0)
    inside &= values < ends[safe_positions]
    source_label = np.full(values.shape, pd.NA, dtype=object)
    source_label[inside] = labels[safe_positions[inside]]
    return pd.DataFrame({
        "time": values,
        "binary_state": np.where(inside, "cutting", "non_cutting"),
        "anomaly_label": source_label,
    })


def union_duration(intervals: pd.DataFrame) -> float:
    """Return seconds covered by the union of validated intervals."""
    if intervals.empty:
        return 0.0
    total = 0.0
    current_start = float(intervals.iloc[0]["start"])
    current_end = float(intervals.iloc[0]["end"])
    for row in intervals.iloc[1:].itertuples():
        start, end = float(row.start), float(row.end)
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start
