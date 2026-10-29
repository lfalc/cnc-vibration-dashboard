"""M7 reliability and attribution analysis for the frozen M6 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from research.m3_baseline import BUDGETS, FEATURES, MODEL, SEEDS, _metric_row, _validate_input, fit_predict, select_visible_labels
from research.m5_agent import MARGIN_THRESHOLD, agent_label


def _frozen(path: Path, name: str) -> dict:
    config = json.loads((path / "experiment_config.json").read_text())
    if not config.get("frozen"):
        raise RuntimeError(f"{name} configuration is not frozen")
    return config


def _correlations(quality: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    if "method" in quality.columns:
        quality = quality.rename(columns={"method": "model"})
    joined = quality.merge(deltas[["model", "budget", "seed", "macro_f1_delta"]], on=["model", "budget", "seed"], validate="one_to_one")
    rows = []
    def corr(left, right):
        if len(left) < 2 or left.std() == 0 or right.std() == 0:
            return np.nan
        return float(np.corrcoef(left, right)[0, 1])
    for (method, budget), group in joined.groupby(["model", "budget"], sort=True):
        rows.append({"method": method, "budget": budget, "seeds": len(group), "coverage_noise_pearson": corr(group.coverage.to_numpy(), (1 - group.pseudo_macro_f1).to_numpy()), "coverage_downstream_delta_pearson": corr(group.coverage.to_numpy(), group.macro_f1_delta.to_numpy())})
    return pd.DataFrame(rows)


def _group_metrics(predictions: pd.DataFrame, frame: pd.DataFrame, split: str) -> pd.DataFrame:
    refs = frame[frame.split == split][["window_id", "dataset_id", "label"]]
    scored = predictions.merge(refs, on="window_id", validate="many_to_one")
    rows = []
    for (method, budget, seed, dataset_id), group in scored.groupby(["method", "budget", "seed", "dataset_id"], sort=True):
        rows.append({"method": method, "budget": budget, "seed": seed, "split": split, "dataset_id": dataset_id, "rows": len(group), "macro_f1": f1_score(group.label, group.predicted_label, average="macro", zero_division=0), "cutting_recall": f1_score(group.label, group.predicted_label, pos_label=1, average="binary", zero_division=0)})
    return pd.DataFrame(rows)


def _ablation(frame: pd.DataFrame, split: str, *, no_abstention: bool) -> pd.DataFrame:
    train = frame[frame.split == "train"].copy(); scored = frame[frame.split == split].copy(); rows = []
    method = "prototype_no_abstention" if no_abstention else "prototype_agent_recomputed"
    for budget in BUDGETS:
        for seed in SEEDS:
            visible = select_visible_labels(train[["window_id", "label", "split"]].rename(columns={"label": "reference_label"}), budget, seed)
            fit = train.drop(columns=["label"]).merge(visible, on="window_id", validate="one_to_one").rename(columns={"visible_label": "label"})
            candidates = train[~train.window_id.isin(set(visible.window_id))].copy()
            decisions = agent_label(fit, candidates, margin_threshold=MARGIN_THRESHOLD)
            if no_abstention:
                chosen = decisions.copy()
            else:
                chosen = decisions[decisions.accepted].copy()
            pseudo = candidates.drop(columns=["label"]).merge(chosen[["window_id", "predicted_label"]], on="window_id", validate="one_to_one")
            if len(pseudo):
                pseudo = pseudo.rename(columns={"predicted_label": "label"})[fit.columns]
            augmented = pd.concat([fit, pseudo], ignore_index=True)
            predicted, _ = fit_predict(augmented, scored)
            rows.append(_metric_row(scored.label, predicted, model=method, budget=budget, seed=seed, split=split, train_rows=len(augmented), train_cutting=int((augmented.label == 1).sum()), train_non_cutting=int((augmented.label == 0).sum()), feature_available=len(FEATURES)))
    return pd.DataFrame(rows)


def _aggregate(metrics):
    result = metrics.groupby(["model", "budget"], as_index=False)[["macro_f1", "cutting_precision", "cutting_recall", "non_cutting_precision", "non_cutting_recall"]].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in result.columns]
    return result


def _config(m6):
    return {"protocol": "M7", "analysis": ["seed_variability", "group_dependence", "coverage_noise_attribution", "prototype_no_abstention_ablation"], "m6_configuration_sha256": m6["configuration_sha256"], "budgets": list(BUDGETS), "seeds": list(SEEDS), "features": list(FEATURES), "downstream_model": MODEL, "test_evaluation": "allowed only after this configuration is frozen"}


def freeze_validation(m3_dir: Path, m6_dir: Path, output_dir: Path):
    m6 = _frozen(m6_dir, "M6")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config(m6) | {"train_rows": int((frame.split == "train").sum()), "validation_rows": int((frame.split == "validation").sum()), "test_rows": int((frame.split == "test").sum()), "frozen": True}
    config_text = json.dumps(config, sort_keys=True, indent=2) + "\n"; config["configuration_sha256"] = hashlib.sha256(config_text.encode()).hexdigest()
    (output_dir / "experiment_config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    m6_metrics = pd.read_csv(m6_dir / "validation_metrics.csv"); quality = pd.read_csv(m6_dir / "validation_pseudo_label_quality.csv"); deltas = pd.read_csv(m6_dir / "validation_deltas.csv"); predictions = pd.read_csv(m6_dir / "validation_predictions.csv")
    _aggregate(m6_metrics).to_csv(output_dir / "validation_method_summary.csv", index=False)
    m6_metrics.groupby(["model", "budget"], as_index=False)[["macro_f1", "cutting_recall"]].std().rename(columns={"macro_f1": "macro_f1_std", "cutting_recall": "cutting_recall_std"}).to_csv(output_dir / "seed_variability.csv", index=False)
    _correlations(quality, deltas).to_csv(output_dir / "coverage_noise_correlations.csv", index=False)
    _group_metrics(predictions, frame, "validation").to_csv(output_dir / "validation_group_metrics.csv", index=False)
    ablation = _ablation(frame, "validation", no_abstention=True); ablation.to_csv(output_dir / "validation_ablation_metrics.csv", index=False); _aggregate(pd.concat([m6_metrics[m6_metrics.model.isin(["limited_label", "prototype_agent"])], ablation])).to_csv(output_dir / "validation_ablation_summary.csv", index=False)
    return {"method_rows": len(m6_metrics), "group_rows": len(_group_metrics(predictions, frame, "validation")), "ablation_rows": len(ablation)}


def evaluate_frozen_test(m3_dir: Path, m6_dir: Path, output_dir: Path):
    if not (output_dir / "experiment_config.json").exists() or not json.loads((output_dir / "experiment_config.json").read_text()).get("frozen"):
        raise RuntimeError("freeze M7 validation analysis before final-test analysis")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    predictions = pd.read_csv(m6_dir / "frozen_test_predictions.csv"); metrics = pd.read_csv(m6_dir / "frozen_test_metrics.csv")
    _group_metrics(predictions, frame, "test").to_csv(output_dir / "frozen_test_group_metrics.csv", index=False)
    _ablation(frame, "test", no_abstention=True).to_csv(output_dir / "frozen_test_ablation_metrics.csv", index=False)
    _aggregate(metrics).to_csv(output_dir / "frozen_test_method_summary.csv", index=False)
    return {"test_group_rows": len(_group_metrics(predictions, frame, "test")), "test_ablation_rows": 35}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--m3-dir", type=Path, default=Path("research_outputs/m3")); parser.add_argument("--m6-dir", type=Path, default=Path("research_outputs/m6")); parser.add_argument("--output", type=Path, default=Path("research_outputs/m7")); parser.add_argument("--freeze", action="store_true"); parser.add_argument("--evaluate-test", action="store_true"); args = parser.parse_args()
    if args.freeze == args.evaluate_test: parser.error("choose exactly one of --freeze or --evaluate-test")
    result = freeze_validation(args.m3_dir, args.m6_dir, args.output) if args.freeze else evaluate_frozen_test(args.m3_dir, args.m6_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
