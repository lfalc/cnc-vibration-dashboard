# Research data preparation

This directory contains the first reproducible preparation step for the MSM
candidate dataset. It does not copy or modify the raw archive.

Run from the repository root:

```bash
python3 -m research.audit_dataset --root archive --output research_outputs
python3 -m research.partition \
  --manifest research_outputs/dataset_manifest.csv \
  --output research_outputs/partition_manifest.csv \
  --seed 856
python3 -m research.windowing
python3 -m research.features
python3 -m research.m2_protocol
PYTHONPATH=. python3 -m research.m3_baseline --freeze
PYTHONPATH=. python3 -m research.m3_baseline --evaluate-test
PYTHONPATH=. python3 -m research.m4_pseudolabel --freeze
PYTHONPATH=. python3 -m research.m4_pseudolabel --evaluate-test
PYTHONPATH=. python3 -m research.m5_agent --freeze
PYTHONPATH=. python3 -m research.m5_agent --evaluate-test
PYTHONPATH=. python3 -m research.m6_comparison --freeze
PYTHONPATH=. python3 -m research.m6_comparison --evaluate-test
PYTHONPATH=. python3 -m research.m7_reliability --freeze
PYTHONPATH=. python3 -m research.m7_reliability --evaluate-test
PYTHONPATH=. python3 -m research.m8_finalize
```

The audit writes ignored derived files:

- `dataset_manifest.csv`: one row per dataset group with timeline duration,
  labeled cutting duration, complement duration, class-duration totals, and
  boundary checks;
- `interval_manifest.csv`: normalized source intervals with original annotation
  columns and `binary_state=cutting`.
- `partition_manifest.csv`: deterministic train/validation/test assignments,
  made at the whole-dataset-group level before window extraction. The fixed
  seed is recorded in the command; each source family is represented in all
  three splits.
- `window_manifest.csv`: 2-second windows with 1-second stride. Windows are
  labeled `cutting`, `non_cutting`, or `transition`; only the first two are
  included in `primary_binary`.
- `accelerometer_features.csv`: chunked per-window accelerometer statistics
  for `acc0` and `acc1`, joined to the window labels and partition assignments.
- `m2/`: frozen M2 protocol outputs. `public_features.csv` contains no
  reference labels; `evaluator_only/reference_labels.csv` is consumed only by
  the evaluator. `visible_labels_*.csv` contains deterministic training-pool
  label budgets.

The M2 decision record is [`M2_EVIDENCE.md`](M2_EVIDENCE.md). It records the
scope, assumptions, validation results, and remaining limitations without
modifying the thesis documents.

The M3 supervised reference-point decision record is [`M3_EVIDENCE.md`](M3_EVIDENCE.md).
M3 writes ignored validation and frozen-test artifacts under
`research_outputs/m3/`; final-test evaluation requires the explicit frozen
configuration produced by the first command.

The M4 non-agent pseudo-labeling decision record is [`M4_EVIDENCE.md`](M4_EVIDENCE.md).
M4 writes ignored self-training and pseudo-label-quality artifacts under
`research_outputs/m4/` and requires its frozen configuration before test scoring.

The M5 auditable prototype-agent decision record is [`M5_EVIDENCE.md`](M5_EVIDENCE.md).
M5 writes agent decisions, evidence margins, pseudo-label quality, and matched
downstream results under `research_outputs/m5/`.

The M6 matched augmentation decision record is [`M6_EVIDENCE.md`](M6_EVIDENCE.md).
M6 writes augmented-pool manifests, matched metrics, deltas, and frozen-test
comparisons under `research_outputs/m6/`.

The M7 reliability and attribution decision record is [`M7_EVIDENCE.md`](M7_EVIDENCE.md).
M7 writes seed, group-dependence, coverage/noise, and ablation analyses under
`research_outputs/m7/`.

The M8 final reproducibility handoff is [`M8_EVIDENCE.md`](M8_EVIDENCE.md).
M8 verifies milestone artifacts, records configuration and artifact hashes,
and writes the ignored final manifest under `research_outputs/m8/`.

The binary state is derived from interval membership. All source anomaly labels
(`0`, `1`, and `2`) remain available in `label`; they are not collapsed into an
anomaly target by this tool.

Windows crossing an annotation boundary are retained for auditability but are
excluded from the primary binary experiment. A cutting window whose source
intervals contain multiple anomaly labels has a missing `source_label`, rather
than an invented anomaly class.

The current audit produced 69 groups: 48 train, 11 validation, and 10 test.
There are no duplicate or unassigned groups, and all three source families
occur in every split. Derived outputs are ignored by Git; the raw archive is
not copied or modified.
