# M8 evidence: final reproducibility handoff

**Status:** Complete for the implemented M2–M7 benchmark sequence.

## Frozen scope

The completed sequence is:

1. M2: group-aware partitions, binary reference labels, transition/missingness
   exclusions, and the evaluator boundary.
2. M3: supervised logistic-regression reference points at seven budgets and
   five deterministic seeds.
3. M4: conventional confidence-filtered self-training.
4. M5: auditable prototype-evidence labeling agent with abstention.
5. M6: matched limited-label, conventional, prototype-agent, and full-label
   downstream comparison with augmented-pool manifests.
6. M7: seed variability, group dependence, coverage/noise diagnostics, and
   no-abstention attribution ablation.

M8 does not introduce another model, threshold, dataset split, or test-tuned
result. It freezes the evidence already generated and records hashes,
configuration lineage, environment information, and the complete test result.

## Reproducibility handoff

Run from the repository root:

```bash
PYTHONPATH=. python3 -m research.m8_finalize
```

This verifies the required M2–M7 artifacts and frozen configurations, runs
`PYTHONPATH=. pytest -q research/tests`, and writes the ignored manifest to
`research_outputs/m8/final_manifest.json`. The manifest records source commit,
Python/platform details, configuration hashes, and SHA-256 hashes for the
principal artifacts.

Tracked evidence records are `M2_EVIDENCE.md`, `M3_EVIDENCE.md`,
`M4_EVIDENCE.md`, `M5_EVIDENCE.md`, `M6_EVIDENCE.md`, `M7_EVIDENCE.md`, and
this document. Derived CSVs, predictions, configurations, and manifests remain
ignored under `research_outputs/`.

## Final scientific boundary

The results establish a controlled benchmark on the accepted M2 dataset and
show that the prototype-agent rule can outperform matched limited-label
training on some frozen test budgets. They do not establish transfer to the
original unlabeled industrial data, general agent superiority, calibrated
confidence, or causal benefit independent of adding pseudo-labeled examples.
The dataset-label limitations and group dependence remain part of the final
claim boundary.

## Completion decision

M8 is complete when the final manifest exists, all milestone configurations
are frozen, artifact hashes are recorded, the full research suite passes, and
the final handoff is committed. Further work belongs to thesis synthesis or a
new explicitly scoped experiment, not an unrecorded change to these results.
