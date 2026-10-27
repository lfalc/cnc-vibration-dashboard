"""Evaluate frozen predictions against evaluator-only reference labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def evaluate_predictions(predictions: pd.DataFrame, reference: pd.DataFrame, *, split: str | None = None) -> dict[str, float | int]:
    if not {"window_id", "predicted_label"}.issubset(predictions.columns):
        raise ValueError("predictions require window_id and predicted_label")
    if not {"window_id", "reference_label", "split"}.issubset(reference.columns):
        raise ValueError("reference requires window_id, reference_label, and split")
    if predictions.window_id.duplicated().any():
        raise ValueError("predictions contain duplicate window IDs")
    scored = reference if split is None else reference[reference.split == split]
    scored = scored.merge(predictions[["window_id", "predicted_label"]], on="window_id", how="left", validate="one_to_one")
    if scored.predicted_label.isna().any():
        raise ValueError("predictions do not cover the selected reference set")
    if not scored.predicted_label.isin([0, 1]).all():
        raise ValueError("predicted_label must be 0 or 1")
    y = scored.reference_label.astype(int)
    p = scored.predicted_label.astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    non_cut_precision = tn / (tn + fn) if tn + fn else 0.0
    non_cut_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "rows": len(scored), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "cutting_precision": precision, "cutting_recall": recall,
        "macro_f1": ((2 * precision * recall / (precision + recall) if precision + recall else 0.0)
                     + (2 * non_cut_precision * non_cut_recall / (non_cut_precision + non_cut_recall)
                        if non_cut_precision + non_cut_recall else 0.0)) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("research_outputs/m2/evaluator_only/reference_labels.csv"))
    parser.add_argument("--split", choices=["train", "validation", "test"])
    args = parser.parse_args()
    print(evaluate_predictions(pd.read_csv(args.predictions), pd.read_csv(args.reference), split=args.split))


if __name__ == "__main__":
    main()
