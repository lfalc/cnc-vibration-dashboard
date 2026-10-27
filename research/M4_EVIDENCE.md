# M4 evidence: non-agent pseudo-labeling benchmark

**Status:** Complete for the frozen confidence-filtered self-training baseline.

## Decision

M4 implements the essential non-agent alternative required by the roadmap:
conventional confidence-filtered self-training. For every M3 visible-label
budget and seed, a balanced liblinear logistic-regression classifier is fit on
visible verified labels. It predicts only the remaining training-pool rows;
rows with `max(p, 1-p) >= 0.80` are accepted as pseudo-labels, using the
argmax class. Accepted pseudo-labels are concatenated with the visible
verified labels and used to train the identical M3 downstream classifier.

The 0.80 threshold, feature contract, classifier, budgets, seeds, and
acceptance rule were frozen before test evaluation. No validation or test rows
enter the pseudo-labeled training pool.

## Evaluation and provenance

M4 preserves the M2 partitions and M3 configuration. Hidden reference labels
are used only by the evaluator after pseudo-label decisions are fixed. For
each budget and seed, outputs record candidate count, accepted count,
coverage, pseudo-label precision/recall/F1, confusion counts, downstream
predictions, and downstream metrics. The limited-label comparator uses the
same visible labels and classifier but no pseudo-labels.

## Reproducibility checks

- Confidence filtering emits only labels meeting the frozen threshold: passed.
- Overlapping fit/scored rows are rejected: passed.
- Synthetic pseudo-label test and full research suite: passed.
- Every budget/seed has a separately recorded validation run and pseudo-label
  quality row.
- Final-test evaluation requires the frozen M4 configuration.

Commands:

```bash
PYTHONPATH=. pytest -q research/tests
PYTHONPATH=. python3 -m research.m4_pseudolabel --freeze
PYTHONPATH=. python3 -m research.m4_pseudolabel --evaluate-test
```

## Outputs

Ignored artifacts are under `research_outputs/m4/`:

- `experiment_config.json`: frozen M4 protocol and configuration hash;
- `validation_metrics.csv` and `validation_aggregate_metrics.csv`;
- `validation_pseudo_label_quality.csv`;
- `validation_predictions.csv`;
- `frozen_test_metrics.csv`, `frozen_test_pseudo_label_quality.csv`, and
  `frozen_test_predictions.csv`.

## Interpretation and M5 entry

M4 is a non-agent control, not evidence for an agent contribution. Any
improvement must be compared against the limited-label comparator at the same
budget and seed, while pseudo-label quality and coverage are reported
separately. M4 permits M5 development of an auditable agent labeling system;
M5 must use the same evaluator and matched information boundaries, and must
freeze agent behavior before its final test evaluation.
