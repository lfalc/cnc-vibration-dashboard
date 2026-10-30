"""Merge frozen AI validation/test outputs into the existing M6 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _frozen(path: Path, name: str) -> dict:
    config = json.loads((path / "experiment_config.json").read_text())
    if not config.get("frozen"):
        raise RuntimeError(f"{name} configuration is not frozen")
    return config


def _deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics.model == "limited_label"][['budget', 'seed', 'split', 'macro_f1']].rename(columns={'macro_f1': 'limited_macro_f1'})
    other = metrics[metrics.model != "limited_label"].merge(base, on=['budget', 'seed', 'split'], validate='many_to_one')
    other['macro_f1_delta'] = other.macro_f1 - other.limited_macro_f1
    return other[['model', 'budget', 'seed', 'split', 'macro_f1', 'limited_macro_f1', 'macro_f1_delta']]


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.groupby(['model', 'budget'], as_index=False)[['macro_f1', 'cutting_precision', 'cutting_recall', 'non_cutting_precision', 'non_cutting_recall']].agg(['mean', 'std']).reset_index()
    result.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in result.columns]
    return result


def _merge(m6_dir: Path, ai_dirs: list[Path], prefix: str, output_dir: Path) -> dict:
    metrics = [pd.read_csv(m6_dir / f'{prefix}_metrics.csv')]
    predictions = [pd.read_csv(m6_dir / f'{prefix}_predictions.csv')]
    quality = [pd.read_csv(m6_dir / f'{prefix}_pseudo_label_quality.csv')] if (m6_dir / f'{prefix}_pseudo_label_quality.csv').exists() else []
    for ai_dir in ai_dirs:
        metrics.append(pd.read_csv(ai_dir / f'{prefix}_metrics.csv'))
        predictions.append(pd.read_csv(ai_dir / f'{prefix}_predictions.csv'))
        quality.append(pd.read_csv(ai_dir / f'{prefix}_pseudo_label_quality.csv'))
    merged_metrics, merged_predictions = pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True)
    merged_quality = pd.concat(quality, ignore_index=True) if quality else pd.DataFrame()
    merged_metrics.to_csv(output_dir / f'{prefix}_metrics.csv', index=False)
    merged_predictions.to_csv(output_dir / f'{prefix}_predictions.csv', index=False)
    if len(merged_quality):
        merged_quality.to_csv(output_dir / f'{prefix}_pseudo_label_quality.csv', index=False)
    _aggregate(merged_metrics).to_csv(output_dir / f'{prefix}_aggregate_metrics.csv', index=False)
    _deltas(merged_metrics).to_csv(output_dir / f'{prefix}_deltas.csv', index=False)
    return {'metrics': len(merged_metrics), 'predictions': len(merged_predictions), 'quality': len(merged_quality)}


def freeze_validation(m6_dir: Path, ai_dirs: list[Path], output_dir: Path) -> dict:
    m6 = _frozen(m6_dir, 'M6')
    for index, path in enumerate(ai_dirs):
        _frozen(path, f'AI method {index}')
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {'protocol': 'M6-AI', 'base_m6_configuration_sha256': m6['configuration_sha256'],
              'ai_configurations': [json.loads((path / 'experiment_config.json').read_text())['configuration_sha256'] for path in ai_dirs],
              'methods': ['limited_label', 'self_training', 'prototype_agent', 'full_verified'] + [f'ai_{json.loads((path / "experiment_config.json").read_text())["mode"]}' for path in ai_dirs],
              'frozen': True}
    config['configuration_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (output_dir / 'experiment_config.json').write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
    result = _merge(m6_dir, ai_dirs, 'validation', output_dir)
    return result


def evaluate_frozen_test(m6_dir: Path, ai_dirs: list[Path], output_dir: Path) -> dict:
    _frozen(output_dir, 'M6-AI')
    return _merge(m6_dir, ai_dirs, 'frozen_test', output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--m6-dir', type=Path, default=Path('research_outputs/m6'))
    parser.add_argument('--ai-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, default=Path('research_outputs/m6_ai'))
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    if args.freeze == args.evaluate_test:
        parser.error('choose exactly one of --freeze or --evaluate-test')
    result = freeze_validation(args.m6_dir, args.ai_dir, args.output) if args.freeze else evaluate_frozen_test(args.m6_dir, args.ai_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
