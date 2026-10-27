"""Create deterministic group-aware train/validation/test partitions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


SPLITS = ("train", "validation", "test")


def _rank(dataset_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{dataset_id}".encode()).hexdigest()


def make_partition(
    manifest: pd.DataFrame,
    *,
    seed: int = 856,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    """Assign each dataset group to exactly one split.

    Groups are sorted by a seeded stable hash and assigned within each source
    family using fixed 70/15/15 proportions. This preserves representation
    from every family in every split while keeping the split independent of
    row/window order.
    """
    required = {"dataset_id", "source_family"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave test data")

    frame = manifest.copy()
    if frame["dataset_id"].duplicated().any():
        raise ValueError("dataset_id values must be unique")

    rows = []
    for family, family_frame in frame.groupby("source_family", sort=True):
        ordered = family_frame.assign(
            _rank=family_frame["dataset_id"].map(lambda value: _rank(str(value), seed))
        ).sort_values("_rank")
        count = len(ordered)
        train_count = max(1, round(count * train_fraction))
        validation_count = max(1, round(count * validation_fraction))
        if train_count + validation_count >= count:
            validation_count = max(1, count - train_count - 1)
        split_values = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * (count - train_count - validation_count)
        )
        rows.extend(
            {"dataset_id": dataset_id, "split": split, "source_family": family}
            for dataset_id, split in zip(ordered["dataset_id"], split_values)
        )

    assignments = pd.DataFrame(rows)
    result = frame.merge(assignments, on=["dataset_id", "source_family"], how="left", validate="one_to_one")
    if result["split"].isna().any() or len(result) != len(frame):
        raise AssertionError("partition did not assign every dataset group exactly once")
    return result.sort_values("dataset_id").reset_index(drop=True)


def write_partition(manifest_path: Path, output_path: Path, *, seed: int = 856) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    partitioned = make_partition(manifest, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partitioned.to_csv(output_path, index=False)
    return partitioned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("research_outputs/dataset_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/partition_manifest.csv"))
    parser.add_argument("--seed", type=int, default=856)
    args = parser.parse_args()
    partitioned = write_partition(args.manifest, args.output, seed=args.seed)
    print(f"Partitioned {len(partitioned)} dataset groups")
    print(partitioned.groupby("split", sort=False).size().to_string())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
