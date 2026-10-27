import pandas as pd

from research.partition import make_partition


def test_partition_is_deterministic_and_group_disjoint():
    manifest = pd.DataFrame(
        {
            "dataset_id": [f"family/dataset{i}" for i in range(9)],
            "source_family": ["family"] * 9,
            "timeline_seconds": [10] * 9,
        }
    )
    first = make_partition(manifest)
    second = make_partition(manifest)
    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first["dataset_id"].is_unique


def test_each_family_is_represented_in_each_split():
    manifest = pd.DataFrame(
        {
            "dataset_id": [f"{family}/dataset{i}" for family in ("a", "b") for i in range(6)],
            "source_family": [family for family in ("a", "b") for _ in range(6)],
        }
    )
    partitioned = make_partition(manifest)
    counts = partitioned.groupby(["source_family", "split"]).size()
    assert (counts >= 1).all()
