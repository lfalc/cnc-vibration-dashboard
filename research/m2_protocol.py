"""Freeze M2 manifests, visible-label budgets, and evaluation boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0)


def window_id(row: pd.Series) -> str:
    return f"{row.dataset_id}:{row.window_start:.6f}:{row.window_end:.6f}"


def validate_manifests(partition: pd.DataFrame, windows: pd.DataFrame, features: pd.DataFrame) -> dict[str, int]:
    required_partition = {"dataset_id", "split"}
    required_windows = {"dataset_id", "window_start", "window_end", "state", "split"}
    required_features = required_windows | {"accelerometer_available"}
    for name, frame, required in (
        ("partition", partition, required_partition),
        ("windows", windows, required_windows),
        ("features", features, required_features),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} manifest missing columns: {sorted(missing)}")
    if partition.dataset_id.duplicated().any():
        raise ValueError("partition manifest contains duplicate dataset groups")
    if set(partition.split) != {"train", "validation", "test"}:
        raise ValueError("partition manifest must contain train, validation, and test")
    split_map = partition.set_index("dataset_id")["split"]
    for name, frame in (("windows", windows), ("features", features)):
        if frame.dataset_id.isna().any() or not frame.dataset_id.isin(split_map.index).all():
            raise ValueError(f"{name} contains unknown dataset groups")
        if not (frame["split"] == frame.dataset_id.map(split_map)).all():
            raise ValueError(f"{name} split assignments disagree with partition manifest")
    keys = windows.assign(_window_id=windows.apply(window_id, axis=1))["_window_id"]
    if keys.duplicated().any():
        raise ValueError("window manifest contains duplicate windows")
    feature_keys = features.assign(_window_id=features.apply(window_id, axis=1))["_window_id"]
    if set(keys) != set(feature_keys):
        raise ValueError("feature and window manifests do not contain the same windows")
    if not (features.window_end > features.window_start).all():
        raise ValueError("window ends must be after starts")
    if not features.window_seconds.dropna().between(2.0, 2.0).all():
        raise ValueError("M2 requires 2-second windows")
    if not set(windows.state).issubset({"cutting", "non_cutting", "transition"}):
        raise ValueError("unexpected window state")
    return {
        "dataset_groups": int(len(partition)),
        "windows": int(len(windows)),
        "feature_rows": int(len(features)),
        "accelerometer_available_windows": int(features.accelerometer_available.sum()),
        "transition_windows": int((windows.state == "transition").sum()),
    }


def _stable_rank(window: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{window}".encode()).hexdigest()


def build_protocol(features: pd.DataFrame, output_dir: Path, *, seed: int = 856) -> dict[str, int]:
    primary = features[
        features.state.isin(["cutting", "non_cutting"])
        & features.accelerometer_available.astype(bool)
    ].copy()
    primary["window_id"] = primary.apply(window_id, axis=1)
    primary["reference_label"] = (primary.state == "cutting").astype(int)

    reference_columns = [
        "window_id", "dataset_id", "window_start", "window_end", "split",
        "reference_label", "source_label",
    ]
    reference = primary[reference_columns].sort_values("window_id")
    public_columns = [
        column for column in primary.columns
        if column not in {"state", "source_label", "primary_binary", "reference_label", "window_id"}
    ]
    public = primary[["window_id", *public_columns]].sort_values("window_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir = output_dir / "evaluator_only"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    reference.to_csv(evaluator_dir / "reference_labels.csv", index=False)
    public.to_csv(output_dir / "public_features.csv", index=False)

    budget_rows = []
    train = primary[primary.split == "train"].copy()
    for budget in BUDGETS:
        selected = []
        for label, group in train.groupby("reference_label", sort=True):
            ranked = group.assign(
                _rank=group.window_id.map(lambda value: _stable_rank(value, seed))
            ).sort_values("_rank")
            selected.append(ranked.head(max(1, round(len(group) * budget))))
        visible = pd.concat(selected).sort_values("window_id")
        name = f"visible_labels_{budget:g}.csv"
        visible[["window_id", "reference_label"]].rename(
            columns={"reference_label": "visible_label"}
        ).to_csv(output_dir / name, index=False)
        budget_rows.append({
            "budget": budget,
            "visible_rows": len(visible),
            "visible_cutting": int((visible.reference_label == 1).sum()),
            "visible_non_cutting": int((visible.reference_label == 0).sum()),
        })
    pd.DataFrame(budget_rows).to_csv(output_dir / "label_budget_manifest.csv", index=False)
    return {
        "primary_windows": int(len(primary)),
        "primary_train_windows": int((primary.split == "train").sum()),
        "primary_validation_windows": int((primary.split == "validation").sum()),
        "primary_test_windows": int((primary.split == "test").sum()),
        "budget_count": len(BUDGETS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, default=Path("research_outputs/partition_manifest.csv"))
    parser.add_argument("--windows", type=Path, default=Path("research_outputs/window_manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path("research_outputs/accelerometer_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/m2"))
    parser.add_argument("--seed", type=int, default=856)
    args = parser.parse_args()
    partition = pd.read_csv(args.partition)
    windows = pd.read_csv(args.windows)
    features = pd.read_csv(args.features)
    checks = validate_manifests(partition, windows, features)
    protocol = build_protocol(features, args.output, seed=args.seed)
    summary = {"seed": args.seed, **checks, **protocol}
    (args.output / "protocol_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
