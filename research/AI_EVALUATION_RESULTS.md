# AI-agent evaluation results

**Status:** Complete for the frozen Joey-1 direct and prototype-hybrid runs.

## Protocol

Both methods used the frozen M2 partitions, seven visible-label budgets, five
seeds, the M3 ten-feature contract, the existing M3 downstream classifier, and
the M6 evaluator. The direct method sent all eligible candidates to Joey-1.
The hybrid method retained accepted M5 prototype decisions and sent only
prototype-uncertain candidates to Joey-1. The local acceptance gate remained
`top_probability >= 0.85` and probability margin `>= 0.10`; it was not retuned
on the final test partition.

## Pseudo-label quality

| Method | Coverage | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Joey direct | 0.000 | 0.000 | 0.000 |
| Joey prototype-hybrid | 0.364 | 0.751 | 0.454 |

The direct model selected a binary class for 47.3% of decision rows, but none
passed the frozen local gate. Its downstream model therefore reproduced the
limited-label baseline. The hybrid method inherited the prototype labels and
accepted AI labels for the remaining uncertain pool.

## Downstream macro-F1

| Budget | Direct validation | Hybrid validation | Direct test | Hybrid test |
| ---: | ---: | ---: | ---: | ---: |
| 0.5% | 0.708 | 0.736 | 0.584 | 0.619 |
| 1% | 0.697 | 0.750 | 0.543 | 0.688 |
| 2% | 0.698 | 0.777 | 0.542 | 0.741 |
| 5% | 0.715 | 0.758 | 0.536 | 0.690 |
| 10% | 0.746 | 0.747 | 0.534 | 0.635 |
| 25% | 0.761 | 0.755 | 0.556 | 0.613 |
| 100% | 0.779 | 0.779 | 0.594 | 0.594 |

These are frozen descriptive comparisons, not evidence of universal agent
superiority. The hybrid method matches the prototype-agent downstream values
in this run because the AI gate accepted no additional labels beyond the
prototype path at the evaluated budgets.

## Cost and latency

| Mode | Provider requests | Estimated input cost | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: |
| Direct | 5,865 | $0.305 | 0.560 s | 0.730 s |
| Hybrid | 3,371 | $0.175 | 0.516 s | 0.600 s |

Costs are estimates from trace token counts and configured Joey-1 pricing;
provider billing should be reconciled against the account records before
publication. Local cache replays are excluded from provider-request counts.

## Artifacts and limitations

- M6-AI validation/test outputs: `research_outputs/m6_ai/`.
- M7-AI validation/test outputs: `research_outputs/m7_ai/`.
- Raw decisions and traces: `research_outputs/m5_ai_joey_direct/`,
  `research_outputs/m5_ai_joey_hybrid/`, and the corresponding ignored cache and
  trace paths.
- Elephant-1 comparison and planning-model policy generation were not run.
- No threshold retuning, external label source, or hidden test label was used.
