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

`research/ai_agent.py` defines a provider interface and an OpenAI-compatible
HTTP provider. The provider receives visible examples with labels and candidate
feature rows without labels. It must return one JSON decision per candidate:
`accept` with label `0` or `1`, or `abstain` with a null label. The adapter
rejects unknown/duplicate/missing IDs, invalid labels, overlap between fit and
candidate rows, and malformed responses.

The returned rationale is trace evidence, not ground truth or calibrated
confidence. Live calls require `OPENAI_API_KEY`; the provider interface also
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

## Required next integration

Before any final-test claim, add an M5-AI runner that records the exact model,
provider endpoint, system prompt hash, user-prompt hashes, raw responses,
request failures, token/cost metadata, accepted labels, abstentions, and
configuration hash. Run it on validation only, freeze the configuration and
response policy, then add `ai_agent` as a fourth M6 method using the existing
matched budgets/seeds and evaluator. M7 must include a no-rationale/no-tool
ablation and report provider cost and retry behavior.

No live provider call or research result is claimed by this branch yet.
