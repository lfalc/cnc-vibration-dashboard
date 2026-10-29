"""M6 matched comparison of verified, conventional, and agent augmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.m3_baseline import BUDGETS, FEATURES, MODEL, SEEDS, _metric_row, _validate_input, fit_predict, select_visible_labels
from research.m4_pseudolabel import PSEUDO_CONFIDENCE, generate_pseudo_labels
from research.m5_agent import MARGIN_THRESHOLD, agent_label


def _check_frozen(path: Path, name: str) -> dict:
    config = json.loads((path / "experiment_config.json").read_text())
    if not config.get("frozen"):
        raise RuntimeError(f"{name} configuration is not frozen")
    return config


def _pool_rows(method: str, budget: float, seed: int, fit: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
    visible = fit[["window_id", "label"]].copy()
    visible["source"] = "verified_visible"
    rows = [visible]
    if len(pseudo):
        accepted = pseudo[["window_id", "label"]].copy()
        accepted["source"] = "pseudo_label"
        rows.append(accepted)
    result = pd.concat(rows, ignore_index=True)
    result.insert(0, "method", method)
    result.insert(1, "budget", budget)
    result.insert(2, "seed", seed)
    return result.rename(columns={"label": "assigned_label"})


def _quality(method, budget, seed, actual, pseudo, candidates):
    if not len(pseudo):
        return {"method": method, "budget": budget, "seed": seed, "candidate_rows": candidates, "accepted_rows": 0, "coverage": 0.0, "pseudo_accuracy": 0.0, "pseudo_macro_f1": 0.0, "pseudo_cutting_precision": 0.0, "pseudo_cutting_recall": 0.0, "pseudo_non_cutting_precision": 0.0, "pseudo_non_cutting_recall": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    y = actual.to_numpy(dtype=int)
    p = pseudo.label.to_numpy(dtype=int)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    cp = tp / (tp + fp) if tp + fp else 0.0; cr = tp / (tp + fn) if tp + fn else 0.0; np_ = tn / (tn + fn) if tn + fn else 0.0; nr = tn / (tn + fp) if tn + fp else 0.0
    f1c = 2 * cp * cr / (cp + cr) if cp + cr else 0.0; f1n = 2 * np_ * nr / (np_ + nr) if np_ + nr else 0.0
    return {"method": method, "budget": budget, "seed": seed, "candidate_rows": candidates, "accepted_rows": len(pseudo), "coverage": len(pseudo) / candidates if candidates else 0.0, "pseudo_accuracy": (tp + tn) / len(pseudo), "pseudo_macro_f1": (f1c + f1n) / 2, "pseudo_cutting_precision": cp, "pseudo_cutting_recall": cr, "pseudo_non_cutting_precision": np_, "pseudo_non_cutting_recall": nr, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _run_split(frame: pd.DataFrame, split: str, *, write_pool: bool = True):
    train = frame[frame.split == "train"].copy()
    scored = frame[frame.split == split].copy()
    metrics, quality, pools, predictions = [], [], [], []
    for budget in BUDGETS:
        for seed in SEEDS:
            visible = select_visible_labels(train[["window_id", "label", "split"]].rename(columns={"label": "reference_label"}), budget, seed)
            fit = train.drop(columns=["label"]).merge(visible, on="window_id", validate="one_to_one").rename(columns={"visible_label": "label"})
            candidates = train[~train.window_id.isin(set(visible.window_id))].copy()
            base_pred, base_probability = fit_predict(fit, scored)
            common = dict(budget=budget, seed=seed, split=split, train_rows=len(fit), train_cutting=int((fit.label == 1).sum()), train_non_cutting=int((fit.label == 0).sum()), feature_available=len(FEATURES))
            metrics.append(_metric_row(scored.label, base_pred, model="limited_label", **common))
            predictions.append(pd.DataFrame({"window_id": scored.window_id, "budget": budget, "seed": seed, "split": split, "method": "limited_label", "predicted_label": base_pred, "predicted_probability": base_probability}))

            self_training, _ = generate_pseudo_labels(fit, candidates, confidence=PSEUDO_CONFIDENCE)
            self_training = self_training[fit.columns] if len(self_training) else candidates.iloc[0:0].copy()
            agent_decisions = agent_label(fit, candidates, margin_threshold=MARGIN_THRESHOLD)
            agent_pseudo = candidates.drop(columns=["label"]).merge(agent_decisions[agent_decisions.accepted][["window_id", "predicted_label"]], on="window_id", validate="one_to_one") if agent_decisions.accepted.any() else candidates.iloc[0:0].copy()
            agent_pseudo = agent_pseudo.rename(columns={"predicted_label": "label"})[fit.columns] if len(agent_pseudo) else agent_pseudo
            for method, pseudo in (("self_training", self_training), ("prototype_agent", agent_pseudo)):
                augmented = pd.concat([fit, pseudo], ignore_index=True)
                pred, probability = fit_predict(augmented, scored)
                metrics.append(_metric_row(scored.label, pred, model=method, budget=budget, seed=seed, split=split, train_rows=len(augmented), train_cutting=int((augmented.label == 1).sum()), train_non_cutting=int((augmented.label == 0).sum()), feature_available=len(FEATURES)))
                predictions.append(pd.DataFrame({"window_id": scored.window_id, "budget": budget, "seed": seed, "split": split, "method": method, "predicted_label": pred, "predicted_probability": probability}))
                if split == "validation":
                    actual = candidates.set_index("window_id").loc[pseudo.window_id, "label"] if len(pseudo) else pd.Series(dtype=int)
                    quality.append(_quality(method, budget, seed, actual, pseudo, len(candidates)))
                    if write_pool:
                        pools.append(_pool_rows(method, budget, seed, fit, pseudo))

    full = train.copy()
    pred, probability = fit_predict(full, scored)
    metrics.append(_metric_row(scored.label, pred, model="full_verified", budget=1.0, seed="full_reference", split=split, train_rows=len(full), train_cutting=int((full.label == 1).sum()), train_non_cutting=int((full.label == 0).sum()), feature_available=len(FEATURES)))
    predictions.append(pd.DataFrame({"window_id": scored.window_id, "budget": 1.0, "seed": "full_reference", "split": split, "method": "full_verified", "predicted_label": pred, "predicted_probability": probability}))
    pool_frame = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    return pd.DataFrame(metrics), pd.DataFrame(quality), pool_frame, pd.concat(predictions, ignore_index=True)


def _aggregate(metrics):
    result = metrics.groupby(["model", "budget"], as_index=False)[["macro_f1", "cutting_precision", "cutting_recall", "non_cutting_precision", "non_cutting_recall"]].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in result.columns]
    return result


def _deltas(metrics):
    base = metrics[metrics.model == "limited_label"][["budget", "seed", "split", "macro_f1"]].rename(columns={"macro_f1": "limited_macro_f1"})
    other = metrics[metrics.model.isin(["self_training", "prototype_agent"])].merge(base, on=["budget", "seed", "split"], validate="many_to_one")
    other["macro_f1_delta"] = other.macro_f1 - other.limited_macro_f1
    return other[["model", "budget", "seed", "split", "macro_f1", "limited_macro_f1", "macro_f1_delta"]]


def _config(m4, m5):
    return {"protocol": "M6", "methods": ["limited_label", "self_training", "prototype_agent", "full_verified"], "m4_configuration_sha256": m4["configuration_sha256"], "m5_configuration_sha256": m5["configuration_sha256"], "budgets": list(BUDGETS), "seeds": list(SEEDS), "features": list(FEATURES), "downstream_model": MODEL, "test_evaluation": "allowed only after this configuration is frozen"}


def freeze_validation(m3_dir: Path, m4_dir: Path, m5_dir: Path, output_dir: Path):
    m3 = _check_frozen(m3_dir, "M3"); m4 = _check_frozen(m4_dir, "M4"); m5 = _check_frozen(m5_dir, "M5")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config(m4, m5) | {"train_rows": int((frame.split == "train").sum()), "validation_rows": int((frame.split == "validation").sum()), "test_rows": int((frame.split == "test").sum()), "frozen": True}
    config_text = json.dumps(config, sort_keys=True, indent=2) + "\n"; config["configuration_sha256"] = hashlib.sha256(config_text.encode()).hexdigest()
    (output_dir / "experiment_config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    metrics, quality, pools, predictions = _run_split(frame, "validation")
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False); quality.to_csv(output_dir / "validation_pseudo_label_quality.csv", index=False); pools.to_csv(output_dir / "validation_augmented_pool.csv", index=False); predictions.to_csv(output_dir / "validation_predictions.csv", index=False); _aggregate(metrics).to_csv(output_dir / "validation_aggregate_metrics.csv", index=False); _deltas(metrics).to_csv(output_dir / "validation_deltas.csv", index=False)
    return {"validation_metrics": len(metrics), "quality_rows": len(quality), "augmented_pool_rows": len(pools), "validation_predictions": len(predictions)}


def evaluate_frozen_test(m3_dir: Path, output_dir: Path):
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists() or not json.loads(config_path.read_text()).get("frozen"): raise RuntimeError("freeze M6 validation configuration before evaluating final test")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    metrics, _, _, predictions = _run_split(frame, "test", write_pool=False)
    metrics.to_csv(output_dir / "frozen_test_metrics.csv", index=False); predictions.to_csv(output_dir / "frozen_test_predictions.csv", index=False); _aggregate(metrics).to_csv(output_dir / "frozen_test_aggregate_metrics.csv", index=False); _deltas(metrics).to_csv(output_dir / "frozen_test_deltas.csv", index=False)
    return {"test_metrics": len(metrics), "test_predictions": len(predictions)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--m3-dir", type=Path, default=Path("research_outputs/m3")); parser.add_argument("--m4-dir", type=Path, default=Path("research_outputs/m4")); parser.add_argument("--m5-dir", type=Path, default=Path("research_outputs/m5")); parser.add_argument("--output", type=Path, default=Path("research_outputs/m6")); parser.add_argument("--freeze", action="store_true"); parser.add_argument("--evaluate-test", action="store_true"); args = parser.parse_args()
    if args.freeze == args.evaluate_test: parser.error("choose exactly one of --freeze or --evaluate-test")
    result = freeze_validation(args.m3_dir, args.m4_dir, args.m5_dir, args.output) if args.freeze else evaluate_frozen_test(args.m3_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
