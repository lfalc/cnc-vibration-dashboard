# M2 evidence: reproducible data and evaluation protocol

**Status:** Complete for the accelerometer-primary experiment.

## Decision

- Use whole dataset groups as the partition unit; split assignment is fixed by
  seed `856` and recorded in `partition_manifest.csv`.
- Use 2-second windows with a 1-second stride.
- Exclude `transition` windows and windows without complete accelerometer
  coverage from the primary accelerometer experiment.
- Define binary reference labels as `1=cutting` and `0=non_cutting`, where
  non-cutting is the complement of the verified interval annotations.
- Preserve source anomaly labels separately; they are not the binary target.
- Reveal labels only from the training pool at 0.5%, 1%, 2%, 5%, 10%, 25%,
  and 100% budgets using deterministic class-stratified sampling.

## Evidence and validation

- `research_outputs/dataset_manifest.csv`: 69 audited dataset groups.
- `research_outputs/partition_manifest.csv`: 48 train, 11 validation, and 10
  test groups; every source family occurs in every split.
- `research_outputs/window_manifest.csv`: 29,437 windows, including 9,530
  transition windows.
- `research_outputs/accelerometer_features.csv`: 29,437 feature rows; 11,739
  windows have accelerometer coverage and 17,698 are explicitly unavailable.
- `research_outputs/m2/`: sanitized public features, evaluator-only reference
  labels, visible-label budgets, and protocol summary.
- Automated checks verify unique groups, split consistency, unique windows,
  fixed window duration, feature/window alignment, and prediction coverage.

## Risks and limitations

- The binary non-cutting class is a derived complement, not a separately
  annotated class; setup, gaps, and unobserved states require caution.
- Accelerometer coverage is incomplete across the candidate archive, so the
  primary results will not represent every dataset group.
- The evaluator reference file is protected by directory/interface convention
  in this local repository; experiments must not expose it to labeling code.

## Next step

Proceed to M3 with a fixed classifier and the visible-label budgets. Audio and
MTConnect experiments remain separate modality-specific extensions.
