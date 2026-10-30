# AI-Agent Implementation Brief: Cost-Constrained CNC Vibration Labeling

## Objective

Extend the existing CNC vibration pseudo-labeling project with a real LLM-based labeling agent while preserving the current deterministic prototype agent as a baseline.

The agent must label vibration windows as:

- `cutting`
- `non_cutting`
- `uncertain`

The main design constraint is cost. Do not use flagship models for bulk labeling. The preferred model family is Chinese or Qwen-derived, and the system must remain reproducible and suitable for a thesis experiment.

## Current baseline

The project already contains a deterministic prototype/centroid labeling agent with abstention. Treat it as an experimental baseline; do not delete or silently change it.

The benchmark uses:

- group-aware train/validation/test splits;
- fixed label budgets;
- two-second windows with one-second stride;
- a frozen final test set;
- downstream macro-F1 as the primary task metric.

All new experiments must use the same splits, budgets, and test protocol.

## Target architecture

Implement a two-level system:

1. A small decision model performs high-volume window classification.
2. A small generative model is used sparingly to propose labeling policies and inspect ambiguous cases.

The generative model must never be called once per window by default.

```text
raw vibration data
        |
deterministic feature extraction
        |
decision model: cutting / non_cutting / uncertain probabilities
        |
probability + temporal-consistency gate
        |
pseudo-label or abstain

verified examples + feature schema
        |
small planning model
        |
versioned labeling rules and abstention policy
```

## Candidate OpenRouter models

Use model IDs as configuration values, not hard-coded strings in the labeling logic.

Primary candidates:

- `baseten/joey-1`: first candidate. OpenRouter describes it as a 2B decision model built on Qwen3.5-2B. Current listed price is approximately `$0.01/M` input tokens and `$0/M` output tokens.
- `baseten/elephant-1`: second candidate. OpenRouter describes it as a 9B decision model built on Qwen3.5-9B. Current listed price is approximately `$0.035/M` input tokens and `$0/M` output tokens.
- `perplexity/pplx-decider-flash-v1.2`: cheap non-Chinese comparison model. Current listed price is approximately `$0.01/M` input tokens and `$0/M` output tokens.
- `nace-ai/drex-v1.5`: optional 9B decision-model comparison. Current listed price is approximately `$0.04/M` input tokens and `$0/M` output tokens.
- `deepseek/deepseek-v4.1-flash`: optional Chinese planning/review model. Use only for rule generation or ambiguous cases, not bulk classification. Current listed price is approximately `$0.022/M` input and `$0.192/M` output tokens.

Do not use DeepSeek Pro, DeepSeek V3.1, or other flagship-tier models in the main experiment.

Prices and availability change. Record the model ID, provider, date, and observed pricing in every experiment manifest.

## Decision-model interface

The preferred interface is OpenRouter's typed Decisions API when available. Do not parse free-form prose for the primary label.

Represent each input window as compact JSON:

```json
{
  "window_id": "group_03_window_0182",
  "features": {
    "rms_x": 0.18,
    "rms_y": 0.21,
    "rms_z": 0.09,
    "peak_to_peak_x": 0.72,
    "kurtosis_x": 4.1,
    "crest_factor_x": 3.4,
    "spectral_entropy_x": 0.61,
    "band_power_low": 0.22,
    "band_power_mid": 0.48,
    "band_power_high": 0.30
  },
  "context": {
    "previous_window_label": null,
    "next_window_label": null,
    "distance_to_cutting_prototype": 0.31,
    "distance_to_non_cutting_prototype": 0.45,
    "near_transition": false
  }
}
```

Ask one typed choice question with three options:

```text
cutting, non_cutting, uncertain
```

The application must retain the complete returned probability distribution, selected option, model ID, request hash, timestamp, and schema version.

## Abstention policy

Do not treat the model's most likely class as automatically correct. Start with a configurable policy:

- accept only when the top probability is at least `0.85`;
- require a minimum probability margin over the second class;
- require temporal agreement with neighboring windows when context is available;
- abstain near transitions unless confidence is unusually high;
- cap the number of pseudo-labels per group to prevent group imbalance.

All thresholds must be configuration parameters and must be selected on validation data only.

## Planning-agent interface

The planning model receives:

- the feature-schema description;
- a small verified labeled subset;
- aggregate class statistics;
- representative easy and ambiguous examples;
- the current deterministic-agent performance;
- explicit instructions that test labels are unavailable.

It may propose:

- feature combinations;
- threshold-based labeling rules;
- confidence and abstention thresholds;
- temporal-consistency rules;
- conflict-resolution rules.

It must return strict JSON conforming to a local schema. Reject and log malformed responses. Never execute arbitrary code returned by the model. Convert proposed rules into a safe internal representation and evaluate them with deterministic application code.

Each generated policy must have:

- a stable policy ID;
- model ID and model settings;
- prompt/schema version;
- input example hashes;
- generated rules;
- validation metrics;
- creation timestamp.

## Required implementation pieces

1. Add an OpenRouter client using environment variable `OPENROUTER_API_KEY`.
2. Add configurable model IDs for:
   - bulk decision labeling;
   - planning/review;
   - optional fallback model.
3. Add request caching keyed by model ID, request payload, schema version, and relevant generation settings.
4. Add retries with bounded exponential backoff for transient failures only.
5. Add strict response validation.
6. Add a dry-run mode that estimates token usage and cost without sending requests.
7. Add batch processing with resumability and no duplicate requests.
8. Add structured JSONL traces for every labeled window.
9. Add policy export/import so an experiment can be rerun without calling the model again.
10. Preserve the current deterministic agent and make it selectable from the same experiment runner.

## Required experiment

Run the same frozen benchmark for:

1. existing prototype agent;
2. Joey 1 direct decision labeling;
3. Elephant 1 direct decision labeling;
4. hybrid routing: deterministic high-confidence labels, decision model for ambiguous windows;
5. hybrid routing plus planning-model-generated rules.

Use identical label budgets and group splits. The final test set must remain untouched until all thresholds and policies are frozen.

Report:

- downstream macro-F1;
- pseudo-label precision, recall, and coverage;
- abstention rate;
- class distribution;
- group-level macro-F1;
- validation-to-test performance gap;
- number of API calls;
- input/output tokens;
- estimated dollar cost;
- latency and failure rate.

## Scientific safeguards

- Never send hidden test labels to OpenRouter.
- Never tune thresholds on the final test set.
- Pin exact model IDs and settings; do not use a moving `latest` alias.
- Use low or zero temperature where supported.
- Cache all requests and retain raw responses.
- Record model, prompt, schema, feature version, and code commit for every run.
- Clearly distinguish the operational label definition from physical machine state. The current target is cutting interval versus the outside-interval complement; do not call the latter “idle” without evidence.
- Treat returned probabilities as scores requiring validation, not guaranteed calibrated probabilities.

## Acceptance criteria

The implementation is complete when:

- a clean checkout can run the existing baseline unchanged;
- a dry-run can estimate cost for a complete budget;
- a cached run can be resumed without duplicate model calls;
- all model outputs validate against schemas;
- every pseudo-label has a traceable reason and model response;
- the final benchmark table compares all required systems on the same frozen test set;
- the system can run entirely from a saved policy and cached responses;
- tests cover payload construction, caching, schema validation, abstention, retries, and deterministic rule execution.

## Implementation priority

Start with the smallest useful vertical slice:

1. OpenRouter client abstraction;
2. Joey 1 decision-model adapter;
3. one-window typed classification;
4. cache and JSONL trace;
5. abstention gate;
6. benchmark integration;
7. Elephant 1 comparison;
8. planning-model rule generation.

Do not build a general autonomous agent framework. The research contribution is a reproducible, cost-aware labeling agent for structured vibration evidence.
