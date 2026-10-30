# AI evaluation runbook

The direct Joey-1 validation run is complete. Its observed validation result
was zero accepted pseudo-label coverage under the frozen `0.85` top-score and
`0.10` margin gate, so its downstream result matched limited-label training.
The raw model selected a binary class for 47.3% of decision rows, but those
scores were below the local acceptance gate. This is an observed result, not a
threshold recommendation.

Direct trace accounting recorded 5,865 provider-cache misses, approximately
30.46 million estimated input tokens, and approximately `$0.305` estimated
input cost at the configured Joey-1 rate. All trace rows, including local
cache replays, are retained under `research_outputs/`.

When direct and hybrid validation outputs exist, run:

```bash
PYTHONPATH=. python3 -m research.m6_ai_comparison --freeze \
  --m6-dir research_outputs/m6 \
  --ai-dir research_outputs/m5_ai_joey_direct \
  --ai-dir research_outputs/m5_ai_joey_hybrid \
  --output research_outputs/m6_ai

PYTHONPATH=. python3 -m research.m7_reliability --freeze \
  --m3-dir research_outputs/m3 \
  --m6-dir research_outputs/m6_ai \
  --output research_outputs/m7_ai
```

Only after validation policies and both configurations are frozen:

```bash
PYTHONPATH=. python3 -m research.m5_ai --evaluate-test \
  --output research_outputs/m5_ai_joey_direct
PYTHONPATH=. python3 -m research.m5_ai --evaluate-test \
  --output research_outputs/m5_ai_joey_hybrid
PYTHONPATH=. python3 -m research.m6_ai_comparison --evaluate-test \
  --m6-dir research_outputs/m6 \
  --ai-dir research_outputs/m5_ai_joey_direct \
  --ai-dir research_outputs/m5_ai_joey_hybrid \
  --output research_outputs/m6_ai
PYTHONPATH=. python3 -m research.m7_reliability --evaluate-test \
  --m3-dir research_outputs/m3 \
  --m6-dir research_outputs/m6_ai \
  --output research_outputs/m7_ai
```

Do not interpret the current direct result as evidence that the model is
accurate or inaccurate until pseudo-label quality and the hybrid comparison
are evaluated independently.
