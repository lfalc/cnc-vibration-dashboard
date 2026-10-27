"""M4 non-agent confidence-filtered self-training benchmark.

M4 deliberately implements a conventional pseudo-labeling baseline before any
agent behavior.  The validation command freezes the protocol and writes all
validation decisions; the test command requires that frozen configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.m3_baseline import (
    BUDGETS,
    FEATURES,
    MODEL,
    SEEDS,
    _metric_row,
    _validate_input,
    fit_predict,
    select_visible_labels,
)


PSEUDO_CONFIDENCE = 0.80


def _quality_row(actual: pd.Series, predicted: np.ndarray, *, budget: float, seed: int, candidates: int, accepted: int) -> dict:
    y = actual.to_numpy(dtype=int)
    p = np.asarray(predicted, dtype=int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = (tp + tn) / len(y) if len(y) else 0.0
    cp = tp / (tp + fp) if tp + fp else 0.0
    cr = tp / (tp + fn) if tp + fn else 0.0
    np_ = tn / (tn + fn) if tn + fn else 0.0
    nr = tn / (tn + fp) if tn + fp else 0.0
    f1c = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
    f1n = 2 * np_ * nr / (np_ + nr) if np_ + nr else 0.0
    return {
        "budget": budget, "seed": seed, "candidate_rows": candidates,
        "accepted_rows": accepted, "coverage": accepted / candidates if candidates else 0.0,
        "pseudo_accuracy": precision, "pseudo_macro_f1": (f1c + f1n) / 2,
        "pseudo_cutting_precision": cp, "pseudo_cutting_recall": cr,
        "pseudo_non_cutting_precision": np_, "pseudo_non_cutting_recall": nr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _visible_and_candidates(train: pd.DataFrame, budget: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    visible = select_visible_labels(
        train[["window_id", "label", "split"]].rename(columns={"label": "reference_label"}), budget, seed
    )
    fit = train.drop(columns=["label"]).merge(visible, on="window_id", validate="one_to_one")
    fit = fit.rename(columns={"visible_label": "label"})
    visible_ids = set(visible.window_id)
    candidates = train[~train.window_id.isin(visible_ids)].copy()
    return fit, candidates


def generate_pseudo_labels(fit: pd.DataFrame, candidates: pd.DataFrame, *, confidence: float = PSEUDO_CONFIDENCE) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate accepted pseudo-labels without consulting candidate labels."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    predicted, probability = fit_predict(fit, candidates)
    certainty = np.maximum(probability, 1 - probability)
    accepted = candidates.loc[certainty >= confidence].copy()
    accepted["label"] = predicted[certainty >= confidence]
    accepted["pseudo_probability"] = probability[certainty >= confidence]
    accepted["pseudo_confidence"] = certainty[certainty >= confidence]
    return accepted, probability


def _run_split(frame: pd.DataFrame, split: str, confidence: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame.split == "train"].copy()
    scored = frame[frame.split == split].copy()
    metric_rows, pseudo_rows, prediction_rows = [], [], []
    for budget in BUDGETS:
        for seed in SEEDS:
            fit, candidates = _visible_and_candidates(train, budget, seed)
            accepted, _ = generate_pseudo_labels(fit, candidates, confidence=confidence)
            augmented = pd.concat([fit, accepted[fit.columns]], ignore_index=True)

            base_pred, base_probability = fit_predict(fit, scored)
            augmented_pred, augmented_probability = fit_predict(augmented, scored)
            common = dict(budget=budget, seed=seed, split=split, train_rows=len(fit), train_cutting=int((fit.label == 1).sum()), train_non_cutting=int((fit.label == 0).sum()), feature_available=len(FEATURES))
            metric_rows.append(_metric_row(scored.label, base_pred, model="limited_label", **common))
            metric_rows.append(_metric_row(scored.label, augmented_pred, model="self_training", train_rows=len(augmented), train_cutting=int((augmented.label == 1).sum()), train_non_cutting=int((augmented.label == 0).sum()), split=split, budget=budget, seed=seed, feature_available=len(FEATURES)))
            prediction_rows.extend([
                pd.DataFrame({"window_id": scored.window_id, "budget": budget, "seed": seed, "split": split, "method": "limited_label", "predicted_label": base_pred, "predicted_probability": base_probability}),
                pd.DataFrame({"window_id": scored.window_id, "budget": budget, "seed": seed, "split": split, "method": "self_training", "predicted_label": augmented_pred, "predicted_probability": augmented_probability}),
            ])

            quality = _quality_row(candidates.loc[accepted.index, "label"], accepted.label.to_numpy(), budget=budget, seed=seed, candidates=len(candidates), accepted=len(accepted)) if len(accepted) else _quality_row(pd.Series(dtype=int), np.array([], dtype=int), budget=budget, seed=seed, candidates=len(candidates), accepted=0)
            pseudo_rows.append(quality)
    return pd.DataFrame(metric_rows), pd.DataFrame(pseudo_rows), pd.concat(prediction_rows, ignore_index=True)


def _config() -> dict:
    return {
        "protocol": "M4",
        "method": "confidence_filtered_self_training",
        "confidence_threshold": PSEUDO_CONFIDENCE,
        "accepted_rule": "max(p, 1-p) >= 0.80; pseudo label is argmax",
        "budgets": list(BUDGETS), "seeds": list(SEEDS), "features": list(FEATURES),
        "downstream_model": MODEL,
        "pseudo_labels_are_not_verified": True,
        "test_evaluation": "allowed only after this configuration is frozen",
    }


def freeze_validation(m3_dir: Path, output_dir: Path) -> dict:
    m3_config = json.loads((m3_dir / "experiment_config.json").read_text())
    if not m3_config.get("frozen"):
        raise RuntimeError("M4 requires the frozen M3 configuration")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config() | {"m3_configuration_sha256": m3_config["configuration_sha256"], "train_rows": int((frame.split == "train").sum()), "validation_rows": int((frame.split == "validation").sum()), "test_rows": int((frame.split == "test").sum()), "frozen": True}
    config_text = json.dumps(config, sort_keys=True, indent=2) + "\n"
    config["configuration_sha256"] = hashlib.sha256(config_text.encode()).hexdigest()
    (output_dir / "experiment_config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    metrics, quality, predictions = _run_split(frame, "validation", PSEUDO_CONFIDENCE)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    quality.to_csv(output_dir / "validation_pseudo_label_quality.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    aggregate = metrics.groupby(["model", "budget"], as_index=False)[["macro_f1", "cutting_precision", "cutting_recall", "non_cutting_precision", "non_cutting_recall"]].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in aggregate.columns]
    aggregate.to_csv(output_dir / "validation_aggregate_metrics.csv", index=False)
    return {"validation_metrics": len(metrics), "pseudo_label_quality_rows": len(quality), "validation_predictions": len(predictions)}


def evaluate_frozen_test(m3_dir: Path, output_dir: Path) -> dict:
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists() or not json.loads(config_path.read_text()).get("frozen"):
        raise RuntimeError("freeze M4 validation configuration before evaluating final test")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    metrics, quality, predictions = _run_split(frame, "test", PSEUDO_CONFIDENCE)
    metrics.to_csv(output_dir / "frozen_test_metrics.csv", index=False)
    quality.to_csv(output_dir / "frozen_test_pseudo_label_quality.csv", index=False)
    predictions.to_csv(output_dir / "frozen_test_predictions.csv", index=False)
    return {"test_metrics": len(metrics), "test_predictions": len(predictions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-dir", type=Path, default=Path("research_outputs/m3"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/m4"))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.evaluate_test:
        parser.error("choose exactly one of --freeze or --evaluate-test")
    result = freeze_validation(args.m3_dir, args.output) if args.freeze else evaluate_frozen_test(args.m3_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
