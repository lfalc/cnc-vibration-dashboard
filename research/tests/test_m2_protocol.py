import pandas as pd
import pytest

from research.evaluate import evaluate_predictions
from research.m2_protocol import build_protocol, validate_manifests


def _frames():
    partition = pd.DataFrame({"dataset_id": ["m/train", "m/val", "m/test"], "split": ["train", "validation", "test"]})
    rows = []
    for dataset, split in zip(partition.dataset_id, partition.split):
        for i, state in enumerate(["non_cutting", "cutting"]):
            rows.append({"dataset_id": dataset, "window_start": float(i * 2), "window_end": float(i * 2 + 2), "window_seconds": 2.0, "state": state, "source_label": 0, "primary_binary": True, "split": split, "accelerometer_available": True, "acc0_rms": 1.0})
    return partition, pd.DataFrame(rows)


def test_m2_validation_rejects_split_mismatch():
    partition, windows = _frames()
    with pytest.raises(ValueError, match="split assignments"):
        validate_manifests(partition, windows, windows.assign(split="train"))


def test_protocol_hides_reference_labels_and_evaluator_scores(tmp_path):
    partition, features = _frames()
    assert validate_manifests(partition, features, features)["windows"] == 6
    build_protocol(features, tmp_path)
    public = pd.read_csv(tmp_path / "public_features.csv")
    assert "state" not in public.columns
    assert "reference_label" not in public.columns
    reference = pd.read_csv(tmp_path / "evaluator_only/reference_labels.csv")
    predictions = reference[["window_id"]].assign(predicted_label=reference.reference_label)
    result = evaluate_predictions(predictions, reference, split="test")
    assert result["macro_f1"] == 1.0
