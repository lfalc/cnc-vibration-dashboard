# M5 evidence: auditable prototype-evidence labeling agent

**Status:** Complete for the frozen prototype-evidence agent.

## Agent specification

M5 uses `prototype-evidence-agent-v1`, a fixed, auditable labeling procedure.
For each visible-label budget and seed, it observes only the visible training
labels and the ten M3 accelerometer features. It standardizes visible rows,
computes one centroid per class, assigns each candidate to the nearest
centroid, and emits both distances and a geometric separation margin. It
accepts a pseudo-label only when the margin is at least 0.30; otherwise it
abstains. The reported confidence is explicitly a geometric score, not a
calibrated probability.

The procedure is distinct from M4 self-training because it does not wrap a
logistic classifier or use its probability threshold. Its evidence is a
class-prototype comparison with a visible abstention rule. It is intentionally
simple and reproducible; this milestone does not claim that a centroid rule is
equivalent to a general-purpose AI agent.

## Information boundary and safety checks

- Hidden training labels are used only after decisions are written, by the
  evaluator for pseudo-label quality.
- Validation and final-test rows are never added to the pseudo-labeled
  training pool.
- Fit/candidate overlap is rejected.
- Every decision records agent version, predicted label, nearest and other
  prototype distances, margin, confidence score, and accept/abstain status.
- The fixed label vocabulary is `{0, 1}` and the downstream classifier is the
  frozen M3 classifier.
- Final-test evaluation requires the frozen M5 configuration.

## Reproducibility checks

- Synthetic prototype decisions and abstention behavior: passed.
- Invalid threshold and overlapping-row guards: passed.
- Full research test suite: passed.
- Repeated runs with the same configuration are deterministic.
- Seven budgets and five seeds are recorded for both limited-label and agent
  downstream results, plus pseudo-label quality and decision provenance.

Commands:

```bash
PYTHONPATH=. pytest -q research/tests
PYTHONPATH=. python3 -m research.m5_agent --freeze
PYTHONPATH=. python3 -m research.m5_agent --evaluate-test
```

## Outputs

Ignored artifacts are under `research_outputs/m5/`:

- `experiment_config.json`: frozen agent procedure and configuration hash;
- `validation_agent_decisions.csv`: accepted and abstained training-pool
  decisions with evidence;
- validation and frozen-test metrics, predictions, and pseudo-label quality;
- aggregate matched downstream results.

## Results

On validation, prototype-agent downstream macro F1 by budget was 0.736 at
0.5%, 0.749 at 1%, 0.777 at 2%, 0.758 at 5%, 0.747 at 10%, 0.755 at 25%,
and 0.779 at 100%. The limited-label reference values for the same budgets
were 0.708, 0.697, 0.698, 0.715, 0.746, 0.761, and 0.779. Mean accepted
pseudo-label coverage ranged from 0.388 to 0.446 for the non-full budgets;
mean pseudo-label macro F1 ranged from 0.469 to 0.605, demonstrating why
coverage and quality must be reported separately.

The frozen test macro-F1 means were 0.619, 0.688, 0.741, 0.690, 0.635,
0.613, and 0.594 for the prototype agent, versus 0.584, 0.543, 0.542,
0.536, 0.534, 0.556, and 0.594 for limited-label training. These test values
are descriptive frozen results, not threshold-selection evidence.

## M6 entry decision

M5 permits M6: direct comparison of limited-label, M4 conventional
self-training, and M5 prototype-agent augmentation. M6 must report gains and
regressions, pseudo-label coverage and noise, and matched downstream results;
it must not treat pseudo-labels as verified labels or retune against the final
test partition.
