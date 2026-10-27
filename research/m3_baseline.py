"""Run the frozen M3 supervised limited-label baseline.

The validation phase is intentionally separate from the final-test phase.  A
configuration file written by ``--freeze`` is required before test scoring is
allowed, making accidental test-driven protocol changes difficult.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0)
SEEDS = (856, 857, 858, 859, 860)
FEATURES = tuple(
    f"{axis}_{stat}"
    for axis in ("acc0", "acc1")
    for stat in ("mean", "std", "rms", "peak_to_peak", "crest_factor")
)
MODEL = {
    "classifier": "LogisticRegression",
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "liblinear",
    "max_iter": 2000,
    "threshold": 0.5,
    "preprocessing": "StandardScaler",
}


def _rank(window_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{window_id}".encode()).hexdigest()


def select_visible_labels(train: pd.DataFrame, budget: float, seed: int) -> pd.DataFrame:
    """Select a deterministic, stratified visible-label subset from train only."""
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    required = {"window_id", "reference_label", "split"}
    if not required.issubset(train.columns):
        raise ValueError(f"training frame missing columns: {sorted(required - set(train.columns))}")
    if (train.split != "train").any():
        raise ValueError("visible labels may only be selected from the training split")
    pieces = []
    for label, group in train.groupby("reference_label", sort=True):
        ranked = group.assign(_rank=group.window_id.map(lambda value: _rank(value, seed)))
        pieces.append(ranked.sort_values("_rank").head(max(1, round(len(group) * budget))))
    selected = pd.concat(pieces, ignore_index=True).sort_values("window_id")
    return selected[["window_id", "reference_label"]].rename(columns={"reference_label": "visible_label"})


def _metric_row(y: pd.Series, predicted: np.ndarray, *, budget, seed, split, model,
                train_rows, train_cutting, train_non_cutting, feature_available) -> dict:
    actual = y.to_numpy(dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    cp = tp / (tp + fp) if tp + fp else 0.0
    cr = tp / (tp + fn) if tp + fn else 0.0
    np_ = tn / (tn + fn) if tn + fn else 0.0
    nr = tn / (tn + fp) if tn + fp else 0.0
    f1c = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
    f1n = 2 * np_ * nr / (np_ + nr) if np_ + nr else 0.0
    return {
        "model": model, "budget": budget, "seed": seed, "split": split,
        "rows": len(actual), "train_rows": train_rows,
        "train_cutting": train_cutting, "train_non_cutting": train_non_cutting,
        "feature_available": feature_available, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "macro_f1": (f1c + f1n) / 2, "cutting_precision": cp, "cutting_recall": cr,
        "non_cutting_precision": np_, "non_cutting_recall": nr,
    }


def fit_predict(train: pd.DataFrame, scored: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaler and classifier on *train* only, then return labels and probabilities."""
    if set(train.window_id) & set(scored.window_id):
        raise ValueError("fit and scored rows overlap")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[list(FEATURES)])
    x_scored = scaler.transform(scored[list(FEATURES)])
    classifier = LogisticRegression(**{k: v for k, v in MODEL.items() if k in {"C", "class_weight", "solver", "max_iter"}})
    classifier.fit(x_train, train.label.astype(int))
    probabilities = classifier.predict_proba(x_scored)[:, 1]
    return (probabilities >= MODEL["threshold"]).astype(int), probabilities


def _validate_input(public: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(public.columns)
    if missing:
        raise ValueError(f"public features missing requested numeric features: {sorted(missing)}")
    joined = public.merge(reference[["window_id", "reference_label", "split"]], on=["window_id", "split"], validate="one_to_one")
    if joined[list(FEATURES)].isna().any().any():
        raise ValueError("M3 feature columns contain missing values")
    return joined.rename(columns={"reference_label": "label"})


def _run_split(frame: pd.DataFrame, split: str, output_dir: Path, *, test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame.split == "train"].copy()
    scored = frame[frame.split == split].copy()
    metric_rows, prediction_rows = [], []
    for budget in BUDGETS:
        for seed in SEEDS:
            visible = select_visible_labels(train[["window_id", "reference_label", "split"]].rename(columns={"label": "reference_label"}) if "reference_label" in train else train[["window_id", "label", "split"]].rename(columns={"label": "reference_label"}), budget, seed)
            fit = train.drop(columns=["label"]).merge(visible, on="window_id", validate="one_to_one")
            fit = fit.rename(columns={"visible_label": "label"})
            predicted, probability = fit_predict(fit, scored)
            metric_rows.append(_metric_row(scored.label, predicted, budget=budget, seed=seed, split=split, model="logistic_regression", train_rows=len(fit), train_cutting=int((fit.label == 1).sum()), train_non_cutting=int((fit.label == 0).sum()), feature_available=len(FEATURES)))
            prediction_rows.append(pd.DataFrame({"window_id": scored.window_id, "budget": budget, "seed": seed, "split": split, "predicted_label": predicted, "predicted_probability": probability}))

    # The all-visible model is the full-supervision reference; it is also part
    # of the repeated budget table, with identical selections for every seed.
    full = train.copy()
    predicted, probability = fit_predict(full, scored)
    metric_rows.append(_metric_row(scored.label, predicted, budget=1.0, seed="full_reference", split=split, model="full_label_reference", train_rows=len(full), train_cutting=int((full.label == 1).sum()), train_non_cutting=int((full.label == 0).sum()), feature_available=len(FEATURES)))
    prediction_rows.append(pd.DataFrame({"window_id": scored.window_id, "budget": 1.0, "seed": "full_reference", "split": split, "predicted_label": predicted, "predicted_probability": probability}))

    majority = int(train.label.mean() >= 0.5)
    baseline_pred = np.full(len(scored), majority, dtype=int)
    metric_rows.append(_metric_row(scored.label, baseline_pred, budget="majority_baseline", seed=0, split=split, model="majority_class", train_rows=len(train), train_cutting=int((train.label == 1).sum()), train_non_cutting=int((train.label == 0).sum()), feature_available=len(FEATURES)))
    prediction_rows.append(pd.DataFrame({"window_id": scored.window_id, "budget": "majority_baseline", "seed": 0, "split": split, "predicted_label": baseline_pred, "predicted_probability": float(majority),}))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return metrics, predictions


def _config() -> dict:
    return {"protocol": "M3", "budgets": list(BUDGETS), "seeds": list(SEEDS), "features": list(FEATURES), "model": MODEL, "target": {"1": "cutting", "0": "non_cutting"}, "test_evaluation": "allowed only after this configuration is frozen"}


def freeze_validation(m2_dir: Path, output_dir: Path) -> dict:
    public = pd.read_csv(m2_dir / "public_features.csv")
    reference = pd.read_csv(m2_dir / "evaluator_only/reference_labels.csv")
    frame = _validate_input(public, reference)
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("expected train, validation, and test partitions")
    train = frame[frame.split == "train"]
    for budget in BUDGETS:
        for seed in SEEDS:
            selected = select_visible_labels(train[["window_id", "label", "split"]].rename(columns={"label": "reference_label"}), budget, seed)
            if set(selected.visible_label) != {0, 1}:
                raise ValueError(f"budget {budget} seed {seed} does not contain both classes")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config()
    config["validation_rows"] = int((frame.split == "validation").sum())
    config["test_rows"] = int((frame.split == "test").sum())
    config["train_rows"] = int((frame.split == "train").sum())
    config["frozen"] = True
    config_text = json.dumps(config, sort_keys=True, indent=2) + "\n"
    config["configuration_sha256"] = hashlib.sha256(config_text.encode()).hexdigest()
    (output_dir / "experiment_config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")

    metrics, predictions = _run_split(frame, "validation", output_dir)
    metrics.to_csv(output_dir / "per_run_metrics.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    aggregate = (metrics[metrics.model == "logistic_regression"].groupby(["budget", "split"], as_index=False)[["macro_f1", "cutting_precision", "cutting_recall", "non_cutting_precision", "non_cutting_recall"]].agg(["mean", "std"]).reset_index())
    aggregate.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in aggregate.columns]
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    metrics[metrics.model == "full_label_reference"].to_csv(output_dir / "full_label_reference_results.csv", index=False)
    return {"validation_metrics": len(metrics), "validation_predictions": len(predictions), "config": str(output_dir / "experiment_config.json")}


def evaluate_frozen_test(m2_dir: Path, output_dir: Path) -> dict:
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists() or not json.loads(config_path.read_text()).get("frozen"):
        raise RuntimeError("freeze validation configuration before evaluating final test")
    frame = _validate_input(pd.read_csv(m2_dir / "public_features.csv"), pd.read_csv(m2_dir / "evaluator_only/reference_labels.csv"))
    metrics, predictions = _run_split(frame, "test", output_dir, test=True)
    metrics.to_csv(output_dir / "frozen_test_metrics.csv", index=False)
    predictions.to_csv(output_dir / "frozen_test_predictions.csv", index=False)
    return {"test_metrics": len(metrics), "test_predictions": len(predictions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-dir", type=Path, default=Path("research_outputs/m2"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/m3"))
    parser.add_argument("--freeze", action="store_true", help="run validation and freeze configuration")
    parser.add_argument("--evaluate-test", action="store_true", help="evaluate only an already frozen configuration")
    args = parser.parse_args()
    if args.freeze == args.evaluate_test:
        parser.error("choose exactly one of --freeze or --evaluate-test")
    result = freeze_validation(args.m2_dir, args.output) if args.freeze else evaluate_frozen_test(args.m2_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
