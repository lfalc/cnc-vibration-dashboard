"""M5 auditable prototype-evidence labeling agent.

This is a fixed labeling procedure, not an LLM wrapper.  It analyzes visible
class prototypes, emits per-example evidence, and abstains when the nearest
prototype is not sufficiently separated from the other prototype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.m3_baseline import (
    BUDGETS, FEATURES, MODEL, SEEDS, _metric_row, _validate_input,
    fit_predict, select_visible_labels,
)
from research.m4_pseudolabel import _quality_row


AGENT_VERSION = "prototype-evidence-agent-v1"
MARGIN_THRESHOLD = 0.30


def _fit_prototypes(fit: pd.DataFrame) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Fit a training-only standardizer and one centroid per class."""
    values = fit[list(FEATURES)].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (values - mean) / scale
    prototypes = {
        label: standardized[fit.label.to_numpy() == label].mean(axis=0)
        for label in (0, 1)
    }
    return standardized, prototypes, np.column_stack([mean, scale])


def agent_label(fit: pd.DataFrame, candidates: pd.DataFrame, *, margin_threshold: float = MARGIN_THRESHOLD) -> pd.DataFrame:
    """Return auditable candidate decisions, including abstentions.

    The agent observes only feature values and visible labels in ``fit``.
    Confidence is a geometric separation score, not a calibrated probability.
    """
    if not 0 < margin_threshold < 1:
        raise ValueError("margin threshold must be between 0 and 1")
    if set(fit.window_id) & set(candidates.window_id):
        raise ValueError("fit and candidate rows overlap")
    _, prototypes, scaling = _fit_prototypes(fit)
    x = candidates[list(FEATURES)].to_numpy(dtype=float)
    standardized = (x - scaling[:, 0]) / scaling[:, 1]
    distances = np.column_stack([
        np.linalg.norm(standardized - prototypes[0], axis=1),
        np.linalg.norm(standardized - prototypes[1], axis=1),
    ])
    predicted = distances.argmin(axis=1).astype(int)
    nearest = distances[np.arange(len(distances)), predicted]
    other = distances[np.arange(len(distances)), 1 - predicted]
    margin = (other - nearest) / (other + nearest + 1e-12)
    accepted = margin >= margin_threshold
    result = candidates[["window_id", *FEATURES]].copy()
    result["agent_version"] = AGENT_VERSION
    result["predicted_label"] = predicted
    result["nearest_distance"] = nearest
    result["other_distance"] = other
    result["evidence_margin"] = margin
    result["agent_confidence"] = 0.5 + 0.5 * np.clip(margin, -1, 1)
    result["decision"] = np.where(accepted, "accept", "abstain")
    result["accepted"] = accepted
    return result


def _run_split(frame: pd.DataFrame, split: str, margin_threshold: float):
    train = frame[frame.split == "train"].copy()
    scored = frame[frame.split == split].copy()
    metrics, quality, decisions, predictions = [], [], [], []
    for budget in BUDGETS:
        for seed in SEEDS:
            visible = select_visible_labels(train[["window_id", "label", "split"]].rename(columns={"label": "reference_label"}), budget, seed)
            fit = train.drop(columns=["label"]).merge(visible, on="window_id", validate="one_to_one").rename(columns={"visible_label": "label"})
            candidates = train[~train.window_id.isin(set(visible.window_id))].copy()
            agent = agent_label(fit, candidates, margin_threshold=margin_threshold)
            accepted = agent[agent.accepted].copy()
            pseudo = candidates.drop(columns=["label"]).merge(accepted[["window_id", "predicted_label"]], on="window_id", validate="one_to_one") if len(accepted) else candidates.iloc[0:0].copy()
            if len(pseudo):
                pseudo = pseudo.rename(columns={"predicted_label": "label"})
                augmented = pd.concat([fit, pseudo[fit.columns]], ignore_index=True)
            else:
                augmented = fit
            base_pred, base_probability = fit_predict(fit, scored)
            agent_pred, agent_probability = fit_predict(augmented, scored)
            metrics.append(_metric_row(scored.label, base_pred, model="limited_label", budget=budget, seed=seed, split=split, train_rows=len(fit), train_cutting=int((fit.label == 1).sum()), train_non_cutting=int((fit.label == 0).sum()), feature_available=len(FEATURES)))
            metrics.append(_metric_row(scored.label, agent_pred, model="prototype_agent", budget=budget, seed=seed, split=split, train_rows=len(augmented), train_cutting=int((augmented.label == 1).sum()), train_non_cutting=int((augmented.label == 0).sum()), feature_available=len(FEATURES)))
            predictions.extend([
                pd.DataFrame({"window_id": scored.window_id, "budget": budget, "seed": seed, "split": split, "method": "limited_label", "predicted_label": base_pred, "predicted_probability": base_probability}),
                pd.DataFrame({"window_id": scored.window_id, "budget": budget, "seed": seed, "split": split, "method": "prototype_agent", "predicted_label": agent_pred, "predicted_probability": agent_probability}),
            ])
            if len(accepted):
                actual = candidates.loc[accepted.index, "label"]
                quality.append(_quality_row(actual, accepted.predicted_label.to_numpy(), budget=budget, seed=seed, candidates=len(candidates), accepted=len(accepted)))
            else:
                quality.append(_quality_row(pd.Series(dtype=int), np.array([], dtype=int), budget=budget, seed=seed, candidates=len(candidates), accepted=0))
            decisions.append(agent.assign(budget=budget, seed=seed))
    return pd.DataFrame(metrics), pd.DataFrame(quality), pd.concat(decisions, ignore_index=True), pd.concat(predictions, ignore_index=True)


def _config() -> dict:
    return {
        "protocol": "M5",
        "agent_version": AGENT_VERSION,
        "agent_observes": ["visible training labels", "ten numeric accelerometer features"],
        "agent_does_not_observe": ["hidden training labels", "validation labels", "test labels"],
        "procedure": "standardize visible rows; compute class centroids; nearest-centroid label; emit evidence distances and margin; abstain below margin",
        "margin_threshold": MARGIN_THRESHOLD,
        "confidence_note": "agent_confidence is a geometric score, not calibrated probability",
        "budgets": list(BUDGETS), "seeds": list(SEEDS), "features": list(FEATURES),
        "downstream_model": MODEL,
        "test_evaluation": "allowed only after this configuration is frozen",
    }


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.groupby(["model", "budget"], as_index=False)[["macro_f1", "cutting_precision", "cutting_recall", "non_cutting_precision", "non_cutting_recall"]].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in result.columns]
    return result


def freeze_validation(m3_dir: Path, output_dir: Path) -> dict:
    m3_config = json.loads((m3_dir / "experiment_config.json").read_text())
    if not m3_config.get("frozen"):
        raise RuntimeError("M5 requires the frozen M3 configuration")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config() | {"m3_configuration_sha256": m3_config["configuration_sha256"], "train_rows": int((frame.split == "train").sum()), "validation_rows": int((frame.split == "validation").sum()), "test_rows": int((frame.split == "test").sum()), "frozen": True}
    config_text = json.dumps(config, sort_keys=True, indent=2) + "\n"
    config["configuration_sha256"] = hashlib.sha256(config_text.encode()).hexdigest()
    (output_dir / "experiment_config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    metrics, quality, decisions, predictions = _run_split(frame, "validation", MARGIN_THRESHOLD)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    quality.to_csv(output_dir / "validation_pseudo_label_quality.csv", index=False)
    decisions.to_csv(output_dir / "validation_agent_decisions.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    _aggregate(metrics).to_csv(output_dir / "validation_aggregate_metrics.csv", index=False)
    return {"validation_metrics": len(metrics), "quality_rows": len(quality), "agent_decisions": len(decisions), "validation_predictions": len(predictions)}


def evaluate_frozen_test(m3_dir: Path, output_dir: Path) -> dict:
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists() or not json.loads(config_path.read_text()).get("frozen"):
        raise RuntimeError("freeze M5 validation configuration before evaluating final test")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    metrics, quality, decisions, predictions = _run_split(frame, "test", MARGIN_THRESHOLD)
    metrics.to_csv(output_dir / "frozen_test_metrics.csv", index=False)
    quality.to_csv(output_dir / "frozen_test_pseudo_label_quality.csv", index=False)
    decisions.to_csv(output_dir / "frozen_test_agent_decisions.csv", index=False)
    predictions.to_csv(output_dir / "frozen_test_predictions.csv", index=False)
    return {"test_metrics": len(metrics), "test_predictions": len(predictions), "test_agent_decisions": len(decisions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-dir", type=Path, default=Path("research_outputs/m3"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/m5"))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.evaluate_test:
        parser.error("choose exactly one of --freeze or --evaluate-test")
    result = freeze_validation(args.m3_dir, args.output) if args.freeze else evaluate_frozen_test(args.m3_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
