# M7 evidence: reliability and attribution analysis

**Status:** Complete for the frozen M6 comparison.

## Analyses

M7 evaluates seed variability, dataset-group dependence, pseudo-label coverage
versus noise and downstream delta, and a no-abstention prototype-agent
ablation. It consumes the frozen M6 outputs and preserves the M2 partitions,
M3 downstream classifier, M4 self-training rule, and M5 agent rule.

The ablation forcibly accepts every prototype decision instead of applying the
frozen 0.30 evidence-margin abstention rule. It is an attribution diagnostic,
not a replacement for the frozen agent result. All ablation choices were
frozen before the test analysis.

## Findings

- Prototype-agent validation macro-F1 seed standard deviation was 0.066 at
  0.5%, 0.044 at 1%, 0.022 at 2%, 0.027 at 5%, 0.018 at 10%, and 0.014 at
  25%; limited-label variability was generally lower.
- The frozen test prototype-agent mean macro-F1 advantage over limited-label
  training was approximately 0.034, 0.144, 0.199, 0.153, 0.101, and 0.057
  at the six limited budgets, respectively; it was zero at 100%.
- No-abstention validation macro-F1 was 0.647 at 0.5%, below the frozen
  prototype-agent value 0.736, demonstrating that abstention materially
  affects the result. Its test ablation is preserved separately.
- Coverage/noise correlations vary by budget and method, with only five seeds
  per cell. They are descriptive diagnostics, not causal estimates.
- Group-level metrics are recorded for every dataset group and method on both
  validation and frozen test partitions; adjacent windows are not treated as
  independent replicates.

## Reproducibility and safety

- M6 configuration hash is recorded in `research_outputs/m7/experiment_config.json`.
- Validation produces 106 method rows, 636 group rows, and 35 ablation rows.
- Frozen test produces 530 group rows and 35 ablation rows.
- Full research suite: 19 tests passed.
- Final-test analysis requires the frozen M7 configuration.

## Outputs and M8 entry

Ignored outputs under `research_outputs/m7/` include seed variability,
coverage/noise correlations, group metrics, ablation metrics, method summaries,
and frozen-test analyses. M7 permits M8 final evidence packaging: freeze the
protocol, record environment and artifact hashes, and provide a concise
research handoff without expanding claims beyond the measured benchmark.
