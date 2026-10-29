# M6 evidence: matched augmentation comparison

**Status:** Complete for the frozen limited-label, conventional, and prototype-agent comparison.

## Protocol

M6 uses the frozen M2 partitions, M3 features/classifier, M4 confidence rule,
and M5 prototype-agent rule. For every approved budget and seed, it builds a
training pool containing visible verified labels plus either M4 or M5 accepted
pseudo-labels. The same balanced logistic-regression downstream classifier is
fit for limited-label, conventional self-training, prototype-agent, and full
verified conditions.

Only training rows appear in augmented pools. Validation and test rows remain
held out. Hidden reference labels are used by the evaluator for pseudo-label
quality and downstream scoring only after decisions are fixed.

## Reproducibility and safety checks

- M3, M4, and M5 frozen configuration hashes are recorded in the M6 config.
- Augmented pool manifests contain only window ID, assigned label, method,
  budget, seed, and source; they contain no hidden reference label.
- All methods are scored on the same held-out rows for each budget and seed.
- Delta tables are matched by budget, seed, and split.
- Full research suite: 18 tests passed.
- Validation reruns are deterministic.
- Final-test evaluation requires the frozen M6 configuration.

## Results

Validation macro-F1 means for the prototype agent were 0.736, 0.749, 0.777,
0.758, 0.747, 0.755, and 0.779 across 0.5%, 1%, 2%, 5%, 10%, 25%, and 100%
budgets. The conventional M4 baseline was 0.698, 0.691, 0.693, 0.709,
0.739, 0.759, and 0.779. The limited-label reference was 0.708, 0.697,
0.698, 0.715, 0.746, 0.761, and 0.779.

Frozen-test prototype-agent macro-F1 means were 0.619, 0.688, 0.741, 0.690,
0.635, 0.613, and 0.594. Conventional self-training means were 0.535,
0.529, 0.525, 0.528, 0.528, 0.555, and 0.594; limited-label means were
0.584, 0.543, 0.542, 0.536, 0.534, 0.556, and 0.594.

These are frozen descriptive results. M6 does not claim the agent is better
at every budget: pseudo-label noise, coverage, and method regressions are
preserved in the output tables.

## M7 entry decision

M6 permits M7 reliability and attribution analysis. M7 should examine the
large seed variability, group dependence, pseudo-label noise versus coverage,
and whether the prototype-agent result is attributable to the labeling rule or
simply to adding many examples.
