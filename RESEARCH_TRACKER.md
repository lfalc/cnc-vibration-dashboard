# Research Tracker and Thesis Evidence Register

**Project:** Agent-assisted pseudo-label generation for industrial vibration classification  
**Status:** Planning / implementation not yet started  
**Related documents:** [`METHODOLOGY.md`](METHODOLOGY.md) (scientific protocol) · [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) (milestones and acceptance gates)

> **Instructions for the coding agent:** Maintain this file as the project's durable research record. It is not a place to invent results or fill gaps with plausible text. Record what was done, why, what evidence supports it, and where the evidence can be found. Keep the document useful to a thesis author who did not watch the experiments being run.

## 1. Purpose and authority

This register connects **research questions → sources and data → decisions → experiments → evidence → defensible thesis claims**. It should allow someone to reconstruct the work, assess its limitations, and draft the thesis with accurate citations.

**Precedence:** `METHODOLOGY.md` defines the scientific constraints; `IMPLEMENTATION_ROADMAP.md` defines milestone gates; this file documents execution and evidence. If a discrepancy arises, record it here and flag it rather than silently rewriting the research objective.

**Research context (motivation, not experimental ground truth):** Earlier enclosure-mounted CNC IMU recordings consist of unlabeled feature vectors approximately every five seconds, derived from 2.56-second vibration windows sampled at 800 Hz. They motivate limited-label methods, but **do not serve as verified training or test labels**. The intended controlled experiments use independently labeled public datasets, with most training labels hidden from the labeling methods.

## 2. Agent maintenance protocol

Update this document **at each meaningful project event**, particularly after identifying a source, acquiring a dataset, making a methodological choice, completing an experiment, obtaining a result, discovering a problem, or finishing a roadmap milestone. Do not wait until the project is complete.

At the beginning of a work session, review the current state, open questions, and the next milestone. At the end, update the current-state panel, append a brief dated activity entry, and link new artifacts. When a milestone is declared complete, cite the exact evidence that satisfies its acceptance gate in `IMPLEMENTATION_ROADMAP.md`.

Rules:

- Use dates in `YYYY-MM-DD` and times in UTC when timestamps matter. Do not backdate undocumented work.
- Use stable identifiers: `SRC-###` for literature; `DATA-###` for datasets; `DEC-###` for decisions; `EXP-###` for experiments; `CLAIM-###` for thesis-relevant claims; `ISSUE-###` for problems.
- Distinguish **planned**, **in progress**, **completed**, **failed**, **blocked**, and **superseded**. Never describe a planned experiment as a completed one.
- Separate **observed evidence**, **author/source claims**, **interpretation**, and **hypothesis**. Quote or cite precisely when attributing a finding to a paper.
- Every reported number must name a source artifact (metrics file, figure, log, table, raw output), data version, experiment ID, and the evaluation partition it came from.
- Cite primary papers, dataset releases, repositories, and documentation via stable URLs/DOIs where possible. Mark anything not independently verified as **UNVERIFIED**. Never fabricate bibliographic metadata.
- Record **negative, inconclusive, and failed** experiments, not only favorable outcomes. Document deviations from the original plan.
- Keep the final-test set untouched during development; do not expose hidden labels to the labeling agent. An evaluator may use withheld training-pool labels for post-hoc auditing only after predictions are frozen.
- If this document becomes unwieldy, move detailed logs to versioned files and maintain indexed links here. Do not delete history or silently overwrite previous results.
- Do not include private identifiers, credentials, proprietary raw data, or sensitive machine information in a tracked public repository. Record provenance and access restrictions without leaking contents.

## 3. Current project state

| Field | Current value |
| --- | --- |
| Current milestone | M0 — Establish research contract |
| Milestone status | Not yet verified |
| Primary question | Can an agent expand a small verified label set into useful pseudo-labels for CNC cutting/non-cutting classification? |
| Benchmark dataset | Not selected |
| Experimental protocol | Proposed in `METHODOLOGY.md`; not frozen |
| Baselines | Planned, not executed |
| Agent approach | Open; implementation choices deferred |
| Experiments completed | None recorded; dashboard prototype exists separately |
| Major blocker | Verify whether `DATA-001` has independently defensible cutting vs. non-cutting labels and suitable recordings |
| Next action | Complete `DATASET_CARD.md`, register the original source, and evaluate `DATA-001` against the M1 suitability gate |
| Last updated | 2026-10-27 (initial document creation) |

## 4. Milestone progress and evidence

Update **status** and **proof/remaining gate** as work progresses; do not mark complete based on code existing alone.

| ID | Outcome to reach | Status | Evidence / missing gate |
| --- | --- | --- | --- |
| M0 | Research contract and constraints fixed | Not verified | Confirm methodology, tasks, baselines, and evaluation safeguards |
| M1 | Public benchmark suitability established | Not started | Dataset provenance, verified labels, class definitions, licensing, and usable recording groups |
| M2 | Reproducible, leakage-free protocol | Not started | Frozen partitions, preprocessing, label-budget protocol, provenance |
| M3 | Full-label and low-label supervised baselines | Not started | Repeated metrics and run artifacts |
| M4 | Conventional non-agent pseudo-labeling baselines | Not started | Matched-budget comparisons |
| M5 | Auditable labeling agent | Not started | Agent inputs, tools, label outputs, abstentions, traceability |
| M6 | Downstream learning with generated labels | Not started | Test-set comparison using a controlled downstream model |
| M7 | Robustness and ablation analysis | Not started | Multiple seeds/groups, uncertainty, cost, failure analysis |
| M8 | Frozen thesis evidence package | Not started | Reproducible result tables, figures, citation register, limitations |

## 5. Literature and technical sources

Add one row for each source **actually consulted**. Separate bibliographic details from how a source informs the project. A candidate mentioned in a conversation is **not** automatically a verified citation.

| ID | Citation / exact title | Year | DOI or stable URL | Source type | Verified? | Relevant claim, method, or limitation | Used in | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | No sources independently registered yet | — | — | — | — | — | — | — |

For every source used substantively, capture: bibliographic metadata; the exact section/page/table/figure supporting a claim; scope and assumptions; whether results are directly applicable to vibration-based CNC states; and potential limitations. If a literature claim informs a method, link its `SRC-###` from the corresponding decision or experiment.

**Recommended literature topics:** time-series/spectral classification; machine-state recognition from vibration; self-training and semi-supervised learning; weak supervision and programmatic labeling; agent-generated labeling functions; calibration and abstention; split leakage and domain shift; evaluation of correlated time-series data.

## 6. Dataset provenance and suitability

Register **candidates and rejected datasets**, not just the chosen benchmark.

| ID | Dataset / release | Primary source and access link | License / restrictions | Signals / sample rate | Ground-truth origin and exact label semantics | Recording groups | Decision & reason | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DATA-001 | Candidate archive in `archive/` | Original source not yet registered | Local copy; license not yet verified | CSV/WAV/NetCDF/metadata; exact rates pending | `label.csv` contains interval labels with values 0, 1, and 2; semantics unverified | 69 dataset groups across three source families | Promising candidate; suitability audit required | Candidate |

For each acquired or used dataset, record in a linked data card or in this section:

- Download date, original publisher, version/release, persistent identifier, download/checksum and storage location (do not commit restricted raw data).
- Which information establishes **cutting versus non-cutting**, independently of the sensor features being classified. Distinguish anomaly/quality labels from cutting-state labels.
- Class definitions, ambiguous/transitional windows, missingness, sampling characteristics, number of machines/sessions, and any grouping identifiers.
- Whether raw vibration can be transformed to the intended feature representation; differences in sensor placement and domain.
- Inclusion/exclusion decision and reasons. If reference labels cannot be established, **do not promote predictions to ground truth**.
- License/citation requirements and any limitations on redistribution or publication.

### Dataset compatibility decision

**Status:** Pending. The repository's original unlabeled enclosure-mounted dataset is a motivating example, not the performance benchmark. Public benchmark suitability must be demonstrated under milestone M1 before proceeding to classification claims.

## 7. Methodological decision log

**Never silently rewrite a consequential decision.** Add a new decision and cross-reference the superseded one.

| ID | Date | Decision and alternatives | Rationale / evidence | Consequences and risks | Status | Related IDs |
| --- | --- | --- | --- | --- | --- | --- |
| DEC-001 | 2026-10-27 | Evaluate limited-label augmentation primarily on independently labeled public data, while using legacy unlabeled CNC data as research motivation | Independent labels are required to measure pseudo-label correctness and downstream generalization | Benchmark-to-industrial domain shift limits external claims | Proposed in methodology | M0–M2 |
| DEC-002 | 2026-10-27 | Keep feature extractor, model families, and agent internals open until implementation | Research plan specifies verification targets rather than architecture | Choices must be justified and versioned when made | Open | M2–M5 |

For major decisions, note who/what approved the change, whether it affects comparability, and whether previously reported results need to be rerun.

## 8. Experiment registry

Register **before running** an experiment whenever practical, then complete the result fields. The registry should make post-hoc selection of favorable experiments visible.

| ID | Date / status | Question or hypothesis | Data/split and label budget | Method vs. comparator | Seeds / grouping | Config, code revision and output path | Metrics / outcome | Interpretation / limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | No experiments registered | — | — | — | — | — | — | — |

### Required experiment record

For each experiment, capture enough information to reproduce and interpret it:

1. **Purpose:** question, hypothesis, and metric(s) decided in advance.
2. **Inputs:** `DATA-###`, dataset version, selection criteria, train/validation/test group IDs or frozen manifests, preprocessing and feature schema.
3. **Label visibility:** fraction/absolute count of verified training labels; how seed subsets were selected; which labels remained hidden; access controls for the agent and evaluator.
4. **Methods:** comparator and agent variant, prompts/tool specifications if applicable, model and library versions, thresholds, training budget and hyperparameters.
5. **Reproducibility:** code commit, configuration, random seeds, hardware/runtime when relevant, command or entry point, artifact paths.
6. **Results:** pseudo-label correctness, class-specific precision/recall/F1, macro F1 and balanced accuracy, coverage/abstention, downstream held-out metrics, uncertainty and annotation/inference cost as applicable.
7. **Analysis:** errors, class imbalance, per-machine/session behavior, negative outcomes, deviations, and whether the result supports or contradicts the hypothesis.
8. **Evidence:** immutable or versioned logs, prediction files, metrics, plots, and table-ready summaries; mark the split used for each result.

**Critical distinction:** agreement with pseudo-labels is not accuracy against independent ground truth. Keep label-generation quality and downstream test performance in separate reports.

## 9. Thesis claim–evidence matrix

Only promote findings supported by traced evidence to statements in the thesis. A planned method or plausible explanation is not an established result.

| Claim ID | Draft claim / finding | Status (hypothesis / supported / contradicted / qualified) | Direct evidence (EXP, DATA, artifact) | Supporting literature (SRC) | Caveats / thesis section |
| --- | --- | --- | --- | --- | --- |
| CLAIM-001 | Agent-assisted pseudo-labeling can improve downstream classification over a small-label supervised baseline | Hypothesis | Not tested | Not registered | Results / Discussion |
| CLAIM-002 | A proposed agent provides benefit beyond conventional self-training at matched label budgets | Hypothesis | Not tested | Not registered | Requires direct non-agent comparison |
| CLAIM-003 | Performance generalizes across CNC machines or recording sessions | Hypothesis | Not tested | Not registered | Depends on benchmark groups and independent test design |

As experiments finish, add exact numerical findings with uncertainty, conditions, and references. Separate source-derived background statements from novel findings demonstrated by this project.

## 10. Risks, issues, and limitations

| ID | Discovered | Issue / threat to validity | Impact | Proposed mitigation or follow-up | Status |
| --- | --- | --- | --- | --- | --- |
| ISSUE-001 | 2026-10-27 | Candidate public datasets may label anomalies rather than cutting vs. non-cutting | Could invalidate primary binary benchmark | Audit original annotations and raw supporting records before M1 approval | Open |
| ISSUE-002 | 2026-10-27 | Public sensor placement may differ from enclosure-mounted IMU in motivating application | Limits claims about applicability to original machines | State external validity limits; optional cross-machine evaluation | Open |
| ISSUE-003 | 2026-10-27 | Adjacent/overlapping windows can leak across random train/test splits | Inflated metrics | Split by original runs/sessions/machines before windowing | Open |
| ISSUE-004 | 2026-10-27 | Labeling agent might indirectly access hidden labels or final-test statistics | Circular evaluation and optimistic bias | Enforce label-access boundaries and record agent-accessible inputs | Open |

Record limitations that cannot be resolved as limitations, not as solved issues. Escalate stop conditions described in the implementation roadmap.

## 11. Results and figure index

| Artifact ID / path | Produced by | Figure/table purpose | Data split | Status | Thesis chapter or section |
| --- | --- | --- | --- | --- | --- |
| — | No results yet | — | — | — | — |

Priority thesis figures to produce *once justified by data*: dataset/class distribution; evaluation split diagram; pseudo-label quality versus annotation budget; downstream macro-F1 versus annotation budget; coverage–accuracy trade-off; supervised/non-agent/agent comparative table; per-run or per-machine results; agent ablation and error analysis. Do not create fabricated plots or placeholder numbers that could be confused with observations.

## 12. Activity log (append-only)

Use short entries of the following form. Do not replace older entries with a polished summary; summaries belong in the current-state panel.

### 2026-10-27 — Initial research-tracker setup

- **Milestone:** M0 (not yet verified).
- **Completed:** Created this record to connect source provenance, methodological choices, experiments, limitations, and thesis evidence.
- **Evidence:** `METHODOLOGY.md`, `IMPLEMENTATION_ROADMAP.md`, `RESEARCH_TRACKER.md`.
- **Unresolved:** Select and verify a public benchmark with independent cutting/non-cutting labels.
- **Next:** Register candidate literature and datasets, and evaluate dataset suitability before approving M1.

### Template for future entries

### 2026-10-30 — Start AI-provider-backed agent branch
- Milestone: M5 extension / M6 integration design
- Action / decision: Created branch `ai-agent`. Preserve M4 self-training and the existing M5 prototype agent as controls; add an AI-provider-backed labeling method as a separate comparator.
- Evidence and related IDs: `research/ai_agent.py`, `research/AI_AGENT_DESIGN.md`, `research/tests/test_ai_agent.py`
- Outcome (including failures or inconclusive findings): Provider interface, strict JSON decision schema, candidate-label boundary, and replay-test path implemented. No live provider run or research result yet.
- Risks or deviations: Prompt/provider metadata, response replay, cost accounting, and M6/M7 integration remain required before final-test evaluation.
- Next action: Implement the frozen validation runner and matched M6 method with a selected provider/model, without exposing hidden labels.

### 2026-10-30 — Add AI benchmark merge and offline validation
- Milestone: M5-AI / M6 integration
- Action / decision: Added `m6_ai_comparison.py` to merge frozen direct and hybrid AI outputs with the existing M6 benchmark without overwriting the deterministic results.
- Evidence and related IDs: `research/m5_ai.py`, `research/m6_ai_comparison.py`, `research/tests/test_m6_ai_comparison.py`
- Outcome (including failures or inconclusive findings): Full offline suite passed with 30 tests; direct and hybrid dry-runs each estimate 3,225 validation batches. No live labels were generated because `OPENROUTER_API_KEY` is not present.
- Risks or deviations: Live provider execution, planning-model policy generation, and final M7/M8 AI artifacts remain incomplete; no metric is reported for the AI method.
- Next action: Supply a rotatable OpenRouter key through the environment, run validation direct/hybrid, freeze configurations, then evaluate test and run M7.

```text
### YYYY-MM-DD — Brief activity title
- Milestone:
- Action / decision:
- Evidence and related IDs:
- Outcome (including failures or inconclusive findings):
- Risks or deviations:
- Next action:
```

## 13. Handoff checklist for every work session

Before ending a coding-agent session, verify:

- [ ] Current milestone, status, next action, and blockers are accurate.
- [ ] New sources and datasets have provenance and verification status.
- [ ] Scientific decisions and changes to protocol are documented.
- [ ] Experiments and negative results have identifiers, configs, and evidence paths.
- [ ] Any reported metric names the dataset partition and code/config version.
- [ ] Thesis claims reference supporting evidence and explicitly state limitations.
- [ ] No hidden or final-test labels were used to drive the labeling agent or tuning.
- [ ] The activity log contains a dated, specific entry for the session.

**Completion criterion for this tracker:** A thesis author can trace each important statement, metric, plot, methodological choice, and dataset citation to a verifiable source or experiment without reconstructing the entire development history from chat logs.
