"""Validation runner for the provider-backed M5-AI labeling method.

This runner intentionally stops at validation unless a frozen configuration is
written. It never sends reference labels to the provider: labels are retained
only in the evaluator-side frame and looked up after model decisions return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from research.ai_agent import AbstentionPolicy, OpenRouterClient, apply_policy, build_prompt, label_candidates
from research.m3_baseline import BUDGETS, FEATURES, SEEDS, _metric_row, _validate_input, fit_predict, select_visible_labels
from research.m5_agent import agent_label


def _quality(method, budget, seed, actual, pseudo, candidates):
    if not len(pseudo):
        return {"method": method, "budget": budget, "seed": seed, "candidate_rows": candidates, "accepted_rows": 0, "coverage": 0.0, "pseudo_accuracy": 0.0, "pseudo_macro_f1": 0.0, "pseudo_cutting_precision": 0.0, "pseudo_cutting_recall": 0.0, "pseudo_non_cutting_precision": 0.0, "pseudo_non_cutting_recall": 0.0}
    predicted = pseudo["predicted_label"] if "predicted_label" in pseudo else pseudo["label"]
    y, p = actual.to_numpy(dtype=int), predicted.to_numpy(dtype=int)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    cp = tp / (tp + fp) if tp + fp else 0.0; cr = tp / (tp + fn) if tp + fn else 0.0; np_ = tn / (tn + fn) if tn + fn else 0.0; nr = tn / (tn + fp) if tn + fp else 0.0
    f1c = 2 * cp * cr / (cp + cr) if cp + cr else 0.0; f1n = 2 * np_ * nr / (np_ + nr) if np_ + nr else 0.0
    return {"method": method, "budget": budget, "seed": seed, "candidate_rows": candidates,
            "accepted_rows": len(pseudo), "coverage": len(pseudo) / candidates if candidates else 0.0,
            "pseudo_accuracy": float((y == p).mean()), "pseudo_macro_f1": (f1c + f1n) / 2,
            "pseudo_cutting_precision": cp, "pseudo_cutting_recall": cr,
            "pseudo_non_cutting_precision": np_, "pseudo_non_cutting_recall": nr}


def _batches(frame: pd.DataFrame, batch_size: int):
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start:start + batch_size].copy()


def _ai_decisions(fit, candidates, provider, policy, batch_size):
    pieces = []
    effective_batch_size = batch_size if len(fit) <= 32 else min(batch_size, 16)
    for batch in _batches(candidates, effective_batch_size):
        pieces.append(apply_policy(label_candidates(fit, batch, provider), policy))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _checkpoint_path(output_dir: Path, budget: float, seed: int) -> Path:
    name = f"budget_{budget:g}_seed_{seed}.csv"
    return output_dir / "checkpoints" / re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def run_validation(m3_dir: Path, output_dir: Path, *, model: str, cache_dir: Path,
                   trace_path: Path, batch_size: int, mode: str, dry_run: bool,
                   policy: AbstentionPolicy, scored_split: str = "validation",
                   frozen: bool = True) -> dict:
    config = {"protocol": "M5-AI", "agent_version": "ai-evidence-agent-v1", "model": model,
              "mode": mode, "batch_size": batch_size, "policy": policy.__dict__,
              "budgets": list(BUDGETS), "seeds": list(SEEDS), "features": list(FEATURES),
              "dry_run": dry_run, "scored_split": scored_split, "frozen": frozen}
    output_dir.mkdir(parents=True, exist_ok=True)
    config["configuration_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if scored_split == "validation":
        (output_dir / "experiment_config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    frame = _validate_input(pd.read_csv(m3_dir / "../m2/public_features.csv"), pd.read_csv(m3_dir / "../m2/evaluator_only/reference_labels.csv"))
    train, scored = frame[frame.split == "train"].copy(), frame[frame.split == scored_split].copy()
    provider = OpenRouterClient(model=model, cache_dir=cache_dir, trace_path=trace_path, dry_run=dry_run)
    metrics, quality, decisions, predictions, estimates = [], [], [], [], []
    for budget in BUDGETS:
        for seed in SEEDS:
            visible = select_visible_labels(train[["window_id", "label", "split"]].rename(columns={"label": "reference_label"}), budget, seed)
            fit = train.drop(columns=["label"]).merge(visible, on="window_id", validate="one_to_one").rename(columns={"visible_label": "label"})
            candidates = train[~train.window_id.isin(set(visible.window_id))].copy()
            if dry_run:
                for batch in _batches(candidates, batch_size):
                    estimate = provider.estimate("", build_prompt(fit, batch))
                    estimates.append({"budget": budget, "seed": seed, "candidate_rows": len(batch), **estimate.__dict__})
                continue
            checkpoint = _checkpoint_path(output_dir, budget, seed)
            if candidates.empty:
                ai = pd.DataFrame(columns=["window_id", "predicted_label", "accepted", "accepted_by_policy"])
                pseudo = candidates.drop(columns=["label"])
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                ai.to_csv(checkpoint, index=False)
            else:
                if checkpoint.exists() and checkpoint.stat().st_size > 1:
                    ai = pd.read_csv(checkpoint)
                elif mode == "direct":
                    ai = _ai_decisions(fit, candidates, provider, policy, batch_size)
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    ai.to_csv(checkpoint, index=False)
                elif mode == "hybrid":
                    prototype = agent_label(fit, candidates)
                    accepted = prototype[prototype.accepted]
                    uncertain = candidates[~candidates.window_id.isin(set(accepted.window_id))]
                    ai = _ai_decisions(fit, uncertain, provider, policy, batch_size)
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    ai.to_csv(checkpoint, index=False)
                else:
                    raise ValueError("mode must be direct or hybrid")
                if mode == "direct":
                    pseudo = candidates.drop(columns=["label"]).merge(ai[ai.accepted_by_policy][["window_id", "predicted_label"]], on="window_id", validate="one_to_one")
                    pseudo = pseudo.rename(columns={"predicted_label": "label"})
                else:
                    prototype = agent_label(fit, candidates)
                    accepted = prototype[prototype.accepted]
                    uncertain = candidates[~candidates.window_id.isin(set(accepted.window_id))]
                    ai_pseudo = uncertain.drop(columns=["label"]).merge(ai[ai.accepted_by_policy][["window_id", "predicted_label"]], on="window_id", validate="one_to_one")
                    pseudo = pd.concat([accepted[["window_id", "predicted_label"]], ai_pseudo[["window_id", "predicted_label"]]], ignore_index=True)
                    pseudo = candidates.drop(columns=["label"]).merge(pseudo, on="window_id", validate="one_to_one").rename(columns={"predicted_label": "label"})
            augmented = pd.concat([fit, pseudo[fit.columns]], ignore_index=True) if len(pseudo) else fit
            predicted, probability = fit_predict(augmented, scored)
            metrics.append(_metric_row(scored.label, predicted, model=f"ai_{mode}", budget=budget, seed=seed, split=scored_split, train_rows=len(augmented), train_cutting=int((augmented.label == 1).sum()), train_non_cutting=int((augmented.label == 0).sum()), feature_available=len(FEATURES)))
            predictions.append(pd.DataFrame({"window_id": scored.window_id, "budget": budget, "seed": seed, "split": scored_split, "method": f"ai_{mode}", "predicted_label": predicted, "predicted_probability": probability}))
            actual = candidates.set_index("window_id").loc[pseudo.window_id, "label"] if len(pseudo) else pd.Series(dtype=int)
            quality.append(_quality(f"ai_{mode}", budget, seed, actual, pseudo, len(candidates)))
            decisions.append(ai.assign(budget=budget, seed=seed))
    if dry_run:
        pd.DataFrame(estimates).to_csv(output_dir / "dry_run_cost_estimates.csv", index=False)
        return {"dry_run": True, "batches": len(estimates), "estimated_usd": sum(row["estimated_usd"] for row in estimates)}
    prefix = "validation" if scored_split == "validation" else "frozen_test"
    pd.DataFrame(metrics).to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame(quality).to_csv(output_dir / f"{prefix}_pseudo_label_quality.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(output_dir / f"{prefix}_ai_decisions.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    return {"dry_run": False, "metrics": len(metrics), "quality_rows": len(quality), "decision_rows": sum(len(row) for row in decisions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-dir", type=Path, default=Path("research_outputs/m3"))
    parser.add_argument("--output", type=Path, default=Path("research_outputs/m5_ai"))
    parser.add_argument("--model", default="baseten/joey-1")
    parser.add_argument("--cache-dir", type=Path, default=Path("research_outputs/ai_cache"))
    parser.add_argument("--trace", type=Path, default=Path("research_outputs/ai_traces.jsonl"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=["direct", "hybrid"], default="direct")
    parser.add_argument("--top-probability", type=float, default=0.85)
    parser.add_argument("--minimum-margin", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.evaluate_test:
        parser.error("choose exactly one of --freeze or --evaluate-test")
    scored_split = "validation"
    if args.evaluate_test:
        saved = json.loads((args.output / "experiment_config.json").read_text())
        if not saved.get("frozen"):
            parser.error("freeze AI validation configuration before test evaluation")
        args.model = saved["model"]
        args.mode = saved["mode"]
        args.batch_size = saved["batch_size"]
        args.top_probability = saved["policy"]["top_probability"]
        args.minimum_margin = saved["policy"]["minimum_margin"]
        scored_split = "test"
    policy = AbstentionPolicy(top_probability=args.top_probability, minimum_margin=args.minimum_margin)
    print(json.dumps(run_validation(args.m3_dir, args.output, model=args.model, cache_dir=args.cache_dir, trace_path=args.trace, batch_size=args.batch_size, mode=args.mode, dry_run=args.dry_run, policy=policy, scored_split=scored_split, frozen=args.freeze), indent=2))


if __name__ == "__main__":
    main()
