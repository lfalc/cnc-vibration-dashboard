# M3 evidence: supervised reference points

**Status:** Complete for the frozen accelerometer-primary experiment.

## Protocol

M3 uses the M2 binary target (`1 = cutting`, `0 = non-cutting`), excludes
transition and unavailable-accelerometer windows, and keeps the M2
group-level train/validation/test partition. The only model inputs are these
ten numeric columns:

`acc0_mean`, `acc0_std`, `acc0_rms`, `acc0_peak_to_peak`,
`acc0_crest_factor`, `acc1_mean`, `acc1_std`, `acc1_rms`,
`acc1_peak_to_peak`, and `acc1_crest_factor`.

Each run fits `StandardScaler` and balanced liblinear logistic regression on
the visible training labels only. Predictions use a fixed 0.5 probability
threshold. Budgets are 0.5%, 1%, 2%, 5%, 10%, 25%, and 100%; each is repeated
with seeds 856–860 using deterministic, class-stratified stable-hash
selection. The test partition was evaluated only after the configuration was
written as frozen.

## Reproducibility checks

- Synthetic separable-data test: passed.
- Same seed produces identical selected IDs, fitted preprocessing outcome,
  predictions, and metrics: passed.
- Different seeds are recorded as separate selections: passed.
- Selection rejects non-training rows; fitting rejects train/scored overlap:
  passed.
- Feature contract excludes identifiers, metadata, sample counts, and labels:
  passed.
- Every budget and seed contains both classes: passed.
- Full existing research suite: 13 tests passed.

Commands used:

```bash
PYTHONPATH=. pytest -q research/tests
PYTHONPATH=. python3 -m research.m3_baseline --freeze
PYTHONPATH=. python3 -m research.m3_baseline --evaluate-test
```

## Validation results

The validation partition contains 892 windows and the training partition 3,694
windows. Mean ± standard deviation of macro F1 across the five selections:

| Visible budget | Macro F1 | Cutting precision | Cutting recall |
|---:|---:|---:|---:|
| 0.5% | 0.708 ± 0.027 | 0.803 ± 0.050 | 0.588 ± 0.103 |
| 1% | 0.697 ± 0.026 | 0.795 ± 0.043 | 0.566 ± 0.087 |
| 2% | 0.698 ± 0.018 | 0.785 ± 0.051 | 0.577 ± 0.059 |
| 5% | 0.715 ± 0.015 | 0.803 ± 0.027 | 0.593 ± 0.043 |
| 10% | 0.746 ± 0.023 | 0.839 ± 0.026 | 0.627 ± 0.051 |
| 25% | 0.761 ± 0.014 | 0.845 ± 0.021 | 0.655 ± 0.025 |
| 100% | 0.779 ± 0.000 | 0.860 ± 0.000 | 0.679 ± 0.000 |

The full-label validation reference is macro F1 0.779. The majority-class
sanity baseline is included in the per-run output.

## Frozen test results

The test partition contains 1,092 windows. The frozen full-label reference
has macro F1 **0.594**, cutting precision **0.813**, cutting recall **0.205**,
non-cutting precision **0.767**, and non-cutting recall **0.982**. The five
limited-label test results, all confusion counts, probabilities, and
predictions are in the ignored M3 outputs; no test result was used to choose
features, model settings, budgets, seeds, or threshold.

## Outputs

Ignored reproducibility artifacts are under `research_outputs/m3/`:

- `experiment_config.json`: frozen protocol and configuration hash;
- `per_run_metrics.csv`: 35 validation runs plus full-reference and majority
  baseline rows;
- `aggregate_metrics.csv`: validation mean and standard deviation by budget;
- `validation_predictions.csv` and `frozen_test_predictions.csv`;
- `frozen_test_metrics.csv` and `full_label_reference_results.csv`.

## Limitations and M4 entry decision

This is a supervised reference-point benchmark, not an agent or pseudo-label
experiment. The non-cutting class is the verified interval complement, and
the accelerometer features are window summaries rather than raw waveforms.
The large validation-to-test difference, especially for cutting recall, is a
result to carry into error analysis rather than a reason to tune the test
partition.

M3 permits entry to M4 only for a pre-registered limited-label augmentation
comparison. M4 must preserve the frozen M3 partitions, feature contract,
classifier reference, and test lock, and must add pseudo-label provenance and
quality checks before claiming an agent benefit.
