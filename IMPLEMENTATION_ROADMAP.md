# Implementation Roadmap: Agent-Assisted Pseudo-Labeling

**Status:** Milestone and verification guide — not an implementation specification  
**Companion document:** [`METHODOLOGY.md`](METHODOLOGY.md)  
**Owner of implementation decisions:** Coding agent  
**Research objective:** Test whether an agent given a small amount of independently verified vibration data can produce useful pseudo-labels for training a downstream CNC milling-versus-non-cutting classifier.

## Purpose and operating principles

This roadmap defines **what must be demonstrated, in what order, and what evidence constitutes completion**. It intentionally does not prescribe programming languages, libraries, model architectures, orchestration frameworks, file layouts, or prompts. The coding agent should choose and justify those implementation details.

The scientific protocol in `METHODOLOGY.md` takes precedence if any detail here is ambiguous. Do not change the scientific task or evaluation rules merely to make a model appear successful.

- **Verified labels** come from an independent source; model predictions are **pseudo-labels**, never ground truth.
- The previously collected enclosure-mounted CNC IMU data is **motivation, not required training or evaluation data**. It contains 2.56-second summaries of 800 Hz vibration sampled at approximately five-second intervals, but lacks verified state labels.
- Use a public dataset with defensible cutting/non-cutting annotations for the primary controlled experiments. If none is suitable, stop and document alternatives instead of inventing labels or silently replacing the task.
- Keep **visible verified labels**, **hidden training-pool labels**, **validation labels**, and **final-test labels** logically separate. Hidden training labels may be revealed only to the evaluator for diagnostic scoring after pseudo-label decisions are frozen.
- The final test set must not guide development or tuning.
- Compare the proposed agent with simple, credible non-agent methods under matched data and annotation budgets.
- A negative result, an agent that abstains frequently, or a finding that simple methods outperform the agent can all be valid results.

## Milestone map

| ID | Milestone | Completion evidence | Depends on |
| --- | --- | --- | --- |
| M0 | Research contract and feasibility criteria | Written, reviewable protocol and open questions | None |
| M1 | Suitable benchmark and verified labels | Dataset suitability report with go/no-go decision | M0 |
| M2 | Reproducible data and evaluation partitions | Dataset audit, partition manifest, leakage checks | M1 |
| M3 | Baseline classification and limited-label benchmark | Reproducible baseline results on approved partitions | M2 |
| M4 | Non-agent pseudo-labeling benchmark | Label-quality and downstream comparison | M3 |
| M5 | Agent labeling system | Fixed, auditable agent behavior and generated labels | M3–M4 |
| M6 | Agent-augmented classifier evaluation | Apples-to-apples downstream results | M5 |
| M7 | Robustness, ablations, and uncertainty | Repeated analyses and uncertainty estimates | M6 |
| M8 | Final synthesis and research handoff | Reproducibility package and evidence-backed conclusions | M7 |

**Progression rule:** Do not mark a milestone complete merely because a script runs. Each milestone needs the specified verification evidence. If a gate fails, report the failure and the smallest defensible change in scope. Do not silently proceed.

## M0 — Establish the research contract

**Goal:** Translate the methodology into unambiguous experimental commitments before analyzing results.

**Deliverables**
- A short protocol stating target classes, unit of prediction, definition of independent recording groups, and ground-truth requirements.
- A list of candidate datasets and feasibility risks.
- Proposed primary metric, secondary metrics, label budgets, repetitions, and permitted agent information.
- A clear boundary between exploratory development and confirmatory evaluation.

**Verification / exit gate**
- A reviewer can answer: *What is being classified? Where do reference labels come from? What information does the agent see? What will count as success or failure?*
- Predeclared metrics include macro F1, milling precision/recall, pseudo-label quality, and pseudo-label coverage.
- Explicit test-set access rules are recorded.

**Do not proceed if:** Cutting/non-cutting is still conflated with anomaly/no-anomaly, program running/not running, or spindle on/off.

## M1 — Select and verify a public benchmark

**Goal:** Find a dataset that genuinely supports the intended binary state-classification task.

**Deliverables**
- Candidate dataset comparison and a selected primary benchmark, or a documented no-go finding.
- Evidence for how cutting and non-cutting intervals are established independently (controller information, experimental protocol, synchronized records, verified annotations, etc.).
- Dataset provenance, access conditions, label semantics, sensor modalities, grouping identifiers, class counts, and known gaps.
- A defensible rule for ambiguous windows, transitions, and unavailable labels.

**Verification / exit gate**
- Inspect actual dataset files and annotation definitions; do not rely solely on descriptions in a paper.
- Demonstrate a small sample of reference-labeled intervals with traceable evidence.
- Confirm the dataset supports multiple independent groups suitable for separation into development and evaluation partitions.
- Confirm both classes are present at useful scale and that non-cutting is not defined exclusively as a stopped spindle unless that limitation is explicit.

**Decision:** **GO** only if verified binary state labels and defensible evaluation groups can be built. Otherwise **NO-GO**: document why, assess other public sources, or propose an explicitly approved change in task. Do not treat classifier output as reference truth.

## M2 — Establish a reproducible data and evaluation protocol

**Goal:** Make all subsequent comparisons fair and prevent temporal or group leakage.

**Deliverables**
- Documented observation and feature definitions; information about raw recordings and any derived features.
- Reproducible group-aware train/validation/final-test partitions.
- A scheme for revealing a small, prespecified portion of the **training pool** labels while hiding the rest from labeling methods.
- A controlled evaluation interface that can score hidden labels without exposing them to the agent.
- Integrity/audit results for timestamps, missing data, duplicates, class balance, and ambiguous states.

**Verification / exit gate**
- No recording group appears across incompatible partitions; overlapping or neighboring windows cannot leak across train and test.
- Normalization, feature selection, and learned preprocessing do not use validation/test information in ways prohibited by the protocol.
- Repeating the split and masking process with a recorded seed produces identical partitions and visible-label selections.
- Automated checks demonstrate that training/agent components cannot read hidden training labels or final-test labels.
- Any inability to support group-wise validation is prominently documented and restricts later claims.

**Do not proceed if:** A random window-level split is the only available split and its dependence cannot be addressed.

## M3 — Establish supervised reference points

**Goal:** Determine the task's difficulty and the cost of missing labels before introducing an agent.

**Deliverables**
- One consistent, justified downstream classification/evaluation setup.
- **Limited-label baselines:** models trained with only the visible verified labels at several annotation budgets.
- **Full-label reference:** the same downstream setup trained with all verified training labels.
- Results and failure analysis on validation data; a documented fixed plan for the final test.

**Verification / exit gate**
- Each budget is well-defined, feasible for the class distribution, and repeated over documented label selections where practical.
- Full- and limited-label models use comparable features, training/evaluation boundaries, and metrics.
- Confusion matrices and class-specific results show more than aggregate accuracy.
- Results are reproducible from the recorded experimental configuration.

**Interpretation:** The full-label baseline is a performance reference, **not** an automatic ceiling; limited-label performance measures what augmentation needs to improve.

## M4 — Implement and validate essential non-agent alternatives

**Goal:** Establish whether pseudo-label generation is useful without an AI agent.

**Deliverables**
- At minimum, a conventional confidence-filtered self-training/pseudo-labeling baseline.
- At least one simple rule-based or weak-supervision comparison if feasible and justified by available features.
- Pseudo-labels produced for the hidden-label training pool, with abstentions and confidence/selection policies recorded where applicable.
- Downstream results using the same classifier and visible-label budgets as M3.

**Verification / exit gate**
- Pseudo-label decisions are fixed before the hidden training-pool labels are opened for evaluation.
- Report precision, recall, macro F1, **coverage**, and error types for emitted pseudo-labels.
- Evaluate downstream classifiers only on legitimately held-out reference-labeled data.
- Record compute requirements and any human intervention.

**Do not proceed if:** The apparent benefit depends on access to hidden reference labels or on changing the downstream evaluation protocol for one method.

## M5 — Develop an auditable AI labeling agent

**Goal:** Build an agent that uses only allowed evidence to create labels or reusable labeling rules, and show what makes its behavior distinct from simple pseudo-labeling.

**Permitted conceptual capabilities** (the coding agent chooses a defensible subset): examine visible examples; analyze distributions and spectral/statistical features; compare prototypes; invoke analytical tools; propose labeling functions; reconcile multiple sources of evidence; abstain on ambiguous cases.

**Deliverables**
- A concise specification of what the agent can observe, decide, and invoke.
- An explicit description of where the agent contributes beyond wrapping a conventional classifier.
- A record of its generated labeling procedure or individual predictions, accepted labels, abstentions, and supporting rationale/evidence.
- An agent version/configuration that can be frozen and rerun for matched comparisons.
- Failure and safety checks for invalid rule generation, inconsistent outputs, information leakage, and label overconfidence.

**Verification / exit gate**
- Demonstrate the agent on allowed development examples without revealing hidden labels.
- Verify that repeated runs are traceable, differences are explainable, and outputs conform to the declared label vocabulary.
- Evaluate pseudo-label quality and coverage through the same evaluator used for M4.
- Compare at identical visible-label budgets and sufficiently comparable computational/resource constraints; disclose any mismatch.
- Do not report verbal certainty as calibrated confidence without calibration evidence.

**Key question:** Does an agent capability measurably help, or is the behavior equivalent to a simpler classifier or rule engine?

## M6 — Test whether agent-generated labels improve downstream learning

**Goal:** Answer the central thesis question rather than only measuring agreement with reference labels.

**Deliverables**
- A pseudo-labeled augmented training pool built using the agent's accepted labels and the original visible verified labels.
- The **same downstream classifier** trained under matched conditions on (a) visible labels only, (b) conventional pseudo-labels, (c) agent pseudo-labels, and (d) full verified labels.
- Comparisons across multiple annotation budgets, including cases where label generation hurts performance.
- Results relating accepted-label coverage and pseudo-label noise to final model performance.

**Verification / exit gate**
- Pseudo-labeled training points are never treated as verified test labels.
- No examples from validation or final test are silently added to the training pool.
- Performance is compared on the same held-out examples and predefined metrics.
- Report improvements **and regressions** relative to limited-label and non-agent alternatives, with computation/annotation costs.
- Distinguish any performance effect of *more training examples* from effects attributable to *agent-generated labels*.

**Success is not defined as beating every baseline.** Completion means a controlled answer to whether the proposed method helps, by how much, and under which conditions.

## M7 — Assess reliability and what actually drives performance

**Goal:** Determine whether observed differences are robust, attributable, and scientifically credible.

**Deliverables**
- Repeated runs/label-budget experiments and uncertainty estimates that respect recording/run/machine dependence.
- Agent ablations that remove important capabilities one at a time, where feasible.
- Error analysis for milling versus non-cutting, class imbalance, uncertain windows, and any spindle/axis-motion confounds.
- Optional cross-machine or cross-dataset test **only when independently labeled compatible groups exist**.
- A threats-to-validity register separating methodological findings from unsupported transfer claims.

**Verification / exit gate**
- Compare methods on identical group splits and visible-label selections.
- Confidence intervals or variability estimates use independent groups where possible, not an assumption that adjacent windows are independent.
- Ablations support or challenge the claim that the agent architecture offers value.
- Findings explicitly state where evidence is insufficient to assert applicability to the original unlabeled industrial data.

## M8 — Freeze experiments and deliver research evidence

**Goal:** Make the results inspectable, repeatable, and ready for thesis writing.

**Deliverables**
- Final protocol, dataset/license/provenance report, data-selection rules, partition/masking manifests, and method descriptions.
- Reproducible experiments with configuration records, seeds, environment/dependency information, and preserved outputs.
- Tables/plots of pseudo-label quality and coverage; downstream performance versus annotation budget; baseline comparisons; ablations and uncertainty.
- A structured answer to each research question, including negative results and limitations.
- A short research handoff explaining precisely what was established and what remains unverified.

**Verification / exit gate**
- A second researcher could reconstruct the partitions and reproduce the principal tables from the permitted dataset and documented procedure.
- Every final result traces to a fixed method, known label budget, independent evaluation partition, and recorded experiment.
- Claims do not exceed the evidence (especially regarding ground truth, agent-specific improvement, and transfer to enclosure-mounted industrial recordings).

## Cross-milestone evidence requirements

For each completed milestone, maintain a compact record containing:

| Field | Required content |
| --- | --- |
| Status | Not started / in progress / blocked / complete |
| Decision | What was chosen and why; note alternatives or rejected options |
| Evidence | Artifacts, summaries, checks, or plots supporting completion |
| Validation | Tests performed, observed results, and explicit pass/fail against exit gate |
| Risks | Remaining uncertainties, limitations, or threats to validity |
| Next step | The next milestone or a clearly scoped resolution of a blocker |

**Do not use arbitrary target scores as gates.** A method that performs poorly but is correctly evaluated can complete a milestone. The gates concern validity, completeness, and reproducibility rather than producing a favorable result.

## Stop-and-escalate conditions

Pause the affected line of work and document a decision request if any of the following occur:

- No public benchmark has defensible independent labels for the proposed target task.
- No independent groups support a credible evaluation; leakage cannot be mitigated.
- Labeling methods inadvertently gain access to hidden labels or final-test outcomes.
- The dataset cannot support the minimum visible-label budgets or both target classes.
- Agent behavior cannot be recorded or evaluated reproducibly enough for a scientific comparison.
- Results require changing the target labels, split, primary metric, or test boundary after observing final-test outcomes.

When blocked, present **the issue, its scientific consequence, feasible alternatives, and a recommended decision**. Do not silently revise the core research question.

## Definition of overall completion

The project is ready for thesis reporting when it can answer, with independently evaluated evidence:

1. How accurately does the agent pseudo-label previously unlabeled vibration observations at each verified-label budget?
2. Do those pseudo-labels improve an otherwise fixed downstream classifier versus limited-label training?
3. Is any gain larger than that obtained from conventional self-training or weak supervision under fair comparison?
4. How does performance trade off with coverage, labeling resources, and the amount of verified data?
5. Under what conditions do the results fail or stop generalizing?

**Guiding instruction for the coding agent:** Own the implementation, but treat dataset legitimacy, data separation, baseline fairness, and independently verifiable evidence as non-negotiable research requirements. Build only what is needed to answer the research questions reliably.
