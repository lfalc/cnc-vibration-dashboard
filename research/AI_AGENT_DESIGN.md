# AI-agent extension design

## Role in the research project

The existing M4 and M5 procedures remain controls. This branch adds an AI
provider-backed M5 variant rather than replacing them:

| Stage | Role |
| --- | --- |
| M3 | Fixed low-label supervised reference |
| M4 | Conventional confidence-filtered self-training control |
| M5 | Existing prototype-evidence agent control |
| M5-AI | AI assistant proposes labels and abstentions from visible evidence |
| M6 | Same downstream classifier and budgets compare all methods |
| M7 | Reliability, cost, coverage/noise, and agent-capability ablations |
| M8 | Freeze prompts, provider/model metadata, responses, hashes, and results |

The scientific claim remains whether AI-generated labels improve independently
measured pseudo-label quality or downstream performance at the same verified
label budget. A larger model, more labels, or access to hidden labels cannot be
counted as an agent advantage.

## Implemented contract

`research/ai_agent.py` defines a provider interface and an OpenRouter HTTP
provider. The provider receives visible examples with labels and candidate
feature rows without labels. It must return one JSON decision per candidate:
`accept` with label `0` or `1`, or `abstain` with a null label. The adapter
rejects unknown/duplicate/missing IDs, invalid labels, overlap between fit and
candidate rows, and malformed responses.

The returned rationale is trace evidence, not ground truth or calibrated
confidence. Live calls require `OPENROUTER_API_KEY`; the provider interface also
supports deterministic replay providers for tests and frozen experiments.

## Implemented vertical slice

`research/m5_ai.py` runs direct and hybrid validation modes using the existing
M2/M3 budgets, group splits, and logistic downstream classifier. The hybrid
mode accepts high-margin prototype-agent decisions first and sends only the
remaining candidates to the AI provider. Both modes use the same local
probability/margin/transition acceptance gate.

`OpenRouterClient` provides request hashing, disk caching, bounded retries for
408/429/5xx/network failures, dry-run token/cost estimates, and JSONL request
traces. `AbstentionPolicy` and policy export/import keep acceptance decisions
in deterministic application code.

## M6/M7 integration

`research/m6_ai_comparison.py` merges frozen direct and hybrid AI outputs with
the existing M6 metrics, predictions, quality tables, aggregates, and deltas.
The resulting `research_outputs/m6_ai/` directory is compatible with the M7
runner for group metrics and reliability analysis. It does not overwrite the
original M6 artifacts.

## Remaining live experiment steps

Before any final-test claim, run the M5-AI runner with a deliberately supplied
key, inspect the exact model/provider metadata, prompt/response traces, costs,
accepted labels, abstentions, and configuration hash. Run validation for direct
and hybrid modes, freeze both configurations, evaluate the locked test split,
merge with M6, and run M7. A planning-model policy generator and a
no-rationale/no-tool ablation remain separate planned comparisons; they must
not be represented as completed results until implemented and executed.

No live provider call or research result is claimed by this branch yet.
