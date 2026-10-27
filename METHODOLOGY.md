# Research Methodology: Agent-Assisted Pseudo-Labeling of Industrial Vibration Data

**Status:** Research methodology proposal — implementation intentionally deferred  
**Primary application:** Binary classification of CNC machine states: **cutting/milling** vs. **non-cutting/idling**  
**Primary contribution:** Evaluation of whether an AI agent can transform a small set of verified labels into useful pseudo-labels for a much larger unlabeled dataset.

## 1. Motivation and scope

Industrial vibration monitoring often produces large amounts of sensor-derived data without corresponding machine-state annotations. A motivating dataset consists of enclosure-mounted IMU measurements from CNC machines, stored as feature vectors approximately every 5 seconds. Each vector summarizes a 2.56-second vibration window acquired from an accelerometer at 800 Hz (nominal Nyquist frequency: 400 Hz). Recordings span weeks, contain offline gaps, and have no verified machine-state or machining-run labels; timestamps have been shifted for anonymization.

**These proprietary/unlabeled recordings motivate the problem; they are not assumed to provide either training labels or ground truth for the primary experiments.** They may be discussed as a realistic use case but are not required to demonstrate performance.

The empirical study will instead use one or more **public datasets containing independently established reference labels**, ideally CNC cutting versus non-cutting states. Most reference labels will be intentionally hidden during training to simulate limited annotation availability. Their availability to the evaluator enables a scientifically defensible assessment of pseudo-label quality.

### In scope

- Limited-label learning for vibration or other related industrial time-series features.
- AI-agent-assisted creation of pseudo-labels, potentially by generating and evaluating reusable labeling functions.
- Comparison with conventional supervised, semi-supervised, and weak-supervision baselines.
- Evaluation of whether pseudo-labeled examples improve a fixed downstream classifier.
- Analysis of annotation efficiency, uncertainty, robustness, and generalization.

### Out of scope for the initial study

- New CNC measurements or physical machine access.
- Treating inferred operating states in the motivating dataset as verified ground truth.
- Real-time deployment, hardware integration, or optimization for embedded execution.
- Designing a novel downstream classifier architecture as the main research contribution.
- Choosing frameworks, packages, prompts, APIs, or repository architecture at this stage.

## 2. Research objective and questions

**Central question:** Can an AI agent use a small verified labeled subset to generate pseudo-labels that improve downstream industrial vibration classification relative to conventional limited-label methods?

| ID | Research question |
| --- | --- |
| RQ1 | How accurate are agent-generated pseudo-labels relative to hidden reference labels? |
| RQ2 | Does training on agent-augmented data improve held-out classification performance over training on the same small verified subset alone? |
| RQ3 | Does the agent improve over conventional self-training and programmatic/rule-based labeling under matched label budgets? |
| RQ4 | How does performance vary with the amount of verified data, pseudo-label coverage, and label confidence? |
| RQ5 | How robust are the findings across recordings, machines, operating conditions, and (if feasible) datasets? |

### Testable hypotheses

- **H1:** Agent-assisted pseudo-label augmentation improves held-out performance over a limited-label-only classifier at one or more prespecified annotation budgets.
- **H2:** Agent-assisted pseudo-labeling improves at least one prespecified quality/efficiency trade-off relative to conventional pseudo-labeling at the same verified-label budget.
- **H3 (extension):** Useful pseudo-labeling behavior transfers to held-out machines, sessions, or a second compatible dataset, with a measurable generalization gap.

These are hypotheses to test, **not expected results**. Null or negative findings remain valid research outcomes.

## 3. Terminology and roles of labels

- **Verified/reference label:** A state established independently of the method being evaluated, e.g., synchronized controller events corroborated by cutting intervals, an experimental protocol, or expert-reviewed annotation.
- **Visible label:** A verified training label revealed to the labeling methods.
- **Hidden label:** A verified label withheld from the labeling methods and used only for scoring or a designated evaluation step.
- **Pseudo-label:** A predicted label created by a model, agent, rule, or cluster interpretation. It is **not** ground truth.
- **Abstention:** No label emitted when evidence is insufficient.

A fully supervised classifier trained on verified labels is a **reference benchmark**, not a source of ground truth. Its predicted labels must never be silently promoted to verified annotations.

## 4. Dataset selection and suitability gate

Before experiments, audit candidate public datasets against these conditions:

1. **Target definition:** Can **material removal/cutting** be distinguished from **non-cutting**, including spindle rotation without engagement and air movements where possible? Do not conflate *normal vs. anomalous cutting* with *cutting vs. idling*.
2. **Label provenance:** Are annotations independently documented, and are start/end times or events traceable to reliable evidence? If labels are model-generated, treat them as weak labels rather than ground truth.
3. **Signal access:** Are raw accelerometer signals or sufficiently documented feature vectors available? Can features be computed consistently?
4. **Grouping:** Are machine IDs, recording/session IDs, machining runs, or other defensible independence groups available for splitting?
5. **Coverage:** Are both target classes sufficiently represented under multiple operating conditions?
6. **Access and reproducibility:** Can data, licenses, labeling definitions, and preprocessing assumptions be documented?

**Go/no-go rule:** Do not begin the central CNC binary-classification study unless a dataset supports defensible independent cutting/non-cutting labels and group-wise evaluation. If no candidate meets this threshold, identify a different *labeled* vibration/time-series benchmark and explicitly reframe the task; do not fabricate CNC labels.

A published dataset may be a candidate without being suitable. Suitability must be verified from its files and annotation protocol, not assumed from the paper title.

## 5. Experimental data protocol

### 5.1 Unit of prediction

Define a fixed observation window and target label at that window. Document how windows overlapping state transitions are handled (e.g., exclude ambiguous windows or apply a declared minimum cutting-occupancy rule). Fix this definition **before** examining test results.

Where raw vibration is available, extract a consistent, documented set of time-domain and spectral features. The motivating measurement setup (2.56-second window, 800 Hz raw sampling, ~5-second feature cadence) is a useful reference, but should be adopted only when technically appropriate and supported by the public dataset.

The 0.2 Hz feature-vector cadence is **not** the raw vibration sampling rate. Precomputed summaries cannot be assumed to contain reconstructable waveforms or full spectrograms.

### 5.2 Independent partitions

**Split by whole machines, sessions, runs, or other defensible groups *before* window extraction or label masking. Never randomly split adjacent or overlapping windows across sets.**

- **Training pool:** Used to select visible labeled examples and pseudo-label the remaining examples.
- **Validation:** Used for model selection, thresholds, confidence policies, and agent development; never treated as part of the unlabeled training pool without an explicit redesigned protocol.
- **Final test:** Completely held out until all methods, settings, and comparisons are frozen.

No model, agent, human-in-the-loop process, prompt, labeling-function search, normalization procedure, or threshold may consult hidden test labels. Data-driven preprocessing parameters are estimated on the permitted training data only.

Use group-aware repeated splits or group-aware cross-validation where feasible. If groups or independent test coverage are inadequate, reduce the strength of generalization claims.

### 5.3 Simulated annotation scarcity

Within the training pool, reveal a prespecified fraction of verified labels; hide the rest from all labeling methods. Proposed budgets: **0.5%, 1%, 2%, 5%, 10%, 25%**, plus **100% verified** as a full-supervision reference. Adapt budgets if the smallest subsets cannot contain enough examples of each class or independent runs.

At each budget, sample visible labeled examples using a documented strategy, repeat across multiple seeds/groups, and make the same visible examples available to all competing methods. Keep class availability and leakage risks explicit; class-stratified sampling is acceptable only if it matches the stated annotation protocol.

Hidden training-pool labels may be accessed **only by the evaluator after predictions are fixed** to score pseudo-label quality, not by the agent, labeling functions, or downstream training procedure.

## 6. Methods to compare

Use **the same downstream classifier, features, validation protocol, and verified-label budget** where possible. This isolates the effect of how labels are generated.

| Method | Description | Role |
| --- | --- | --- |
| Limited-label supervised | Train only on revealed verified labels | Minimum baseline |
| Full-label supervised | Train on all verified labels in the training pool | Reference benchmark |
| Conventional self-training | Train a classifier on visible labels; add confidence-filtered predicted labels | Essential pseudo-label baseline |
| Programmatic weak supervision | Hand-authored or data-driven simple labeling functions, with a documented combination policy | Non-agent labeling baseline |
| Agent-assisted labeling | Tool-using agent analyzes visible examples and training features to propose labels or reusable labeling functions | Primary method |
| Agent ablation(s) | Remove one capability at a time (e.g., tool use, temporal context, rule generation) | Identify sources of benefit |

Unsupervised clustering may be included as an exploratory or auxiliary labeling method, but cluster IDs have no inherent semantic meaning. Any cluster-to-class mapping must be learned using only permitted visible labels.

### Agent boundary

The proposed agent is a **labeling system**, not the final evaluated classifier. It may inspect allowable training examples, derive statistics, compare prototypes, invoke analytical tools, and develop labeling functions. It may abstain and provide evidence/confidence for a decision. It must not access hidden labels, final-test observations for iterative tuning, or external information that reveals test outcomes.

An agent-generated Python rule or workflow must be versioned conceptually as a fixed labeling procedure before it is applied and scored. Calling a conventional classifier through an agent wrapper is not by itself evidence of a novel agent contribution.

## 7. Core experiment sequence

1. **Verify dataset and label provenance.** Establish target classes, group IDs, transition policy, and reliable reference labels.
2. **Freeze partitions and evaluation protocol.** Reserve an independent final test; define label budgets, metrics, and repetitions.
3. **Establish supervised bounds.** Train the limited-label baseline at each budget and the full-label reference benchmark.
4. **Run conventional pseudo-labeling baselines.** Generate candidate labels using the same visible subset as the agent.
5. **Develop the agent on training/validation data only.** Allow tool-assisted analysis and candidate labeling-function generation; freeze its procedure before final testing.
6. **Score pseudo-label quality.** Compare predicted labels against hidden reference labels in the *training pool*, with metrics and coverage reported separately. This diagnostic scoring must not feed hidden labels back to the labeling methods.
7. **Train identical downstream classifiers.** Combine visible verified labels with accepted pseudo-labeled training examples under a prespecified weighting/selection policy.
8. **Evaluate downstream performance.** Score all resulting classifiers on the untouched test partition.
9. **Repeat across label budgets and independent seeds/groups.** Estimate variability and investigate failure conditions.
10. **Conduct ablations and synthesis.** Determine whether any benefit is attributable to agent capabilities rather than added data, a stronger classifier, or privileged information.

## 8. Evaluation criteria

### 8.1 Pseudo-label quality

Report, at minimum:

- Precision, recall, macro F1, and confusion matrix against hidden reference labels.
- **Coverage:** Fraction of eligible unlabeled training examples assigned a pseudo-label.
- **Selective performance:** Performance among accepted labels, plotted against coverage.
- Class-specific error rates, especially false *milling* labels and missed cutting intervals.
- Calibration or confidence–accuracy relationships **only if** the method emits a defined numeric confidence score. Free-text certainty is not calibrated probability.

### 8.2 Downstream classification

Report on independently held-out, verified test recordings:

- Macro F1 and milling-class precision/recall as primary metrics.
- Balanced accuracy and confusion matrix as supporting metrics.
- Performance versus number of verified training labels (annotation-efficiency curves).
- Gain or loss relative to limited-label supervision, full-label supervision, and conventional self-training.
- Optional event-level detection delay and false transitions if the dataset supplies valid temporal event boundaries.

Class imbalance and the cost of false-positive/false-negative predictions must be documented. Avoid interpreting overall accuracy alone as sufficient evidence.

### 8.3 Statistical reliability

- Repeat experiments over multiple prespecified visible-label samples and, when possible, independent groups.
- Report variability and confidence intervals computed at the **recording/run/machine level**, not by treating adjacent windows as independent samples.
- Compare paired methods on identical data partitions and label budgets.
- Clearly distinguish exploratory observations from confirmatory evaluation.
- Report negative results, failed label generation, and cases where the agent abstains.

### 8.4 Cost and efficiency

Measure human-verified labels required, pseudo-label coverage, labeling computation/LLM usage, and (if applicable) human review effort. Report performance relative to both annotation budget and computational cost. An agent is not considered better solely because it produces more labels.

## 9. Confounds and validity threats

| Threat | Required mitigation or qualification |
| --- | --- |
| Time-series leakage | Split by independent groups before windowing; prevent overlap across partitions. |
| Circular validation | Only externally verified annotations count as ground truth; pseudo-labels never validate themselves. |
| Wrong task labels | Verify that annotation classes encode actual material removal, not just program execution or cutting quality. |
| Spindle/axis confounding | Include non-cutting spindle rotation and air cutting where available; report if absent. |
| Machine/domain shift | Separate within-machine and cross-machine claims; document sensor-placement differences. |
| Label noise and transitions | Document ambiguous intervals and uncertainty policies; do not force questionable labels. |
| Agent information advantage | Match visible labels, permitted data, and tuning budgets across methods. |
| Validation overfitting | Freeze protocols before final test; reserve test results for final reporting. |
| Confirmation by physical intuition | Treat plausible spectral explanations as hypotheses unless independently verified. |
| Overstated novelty | Compare against standard self-training and weak-supervision methods; evaluate agent-specific ablations. |

## 10. Expected research outputs

The thesis should produce:

1. A documented and reproducible dataset/annotation protocol with independent group-wise splits.
2. A limited-label baseline and a fully supervised reference benchmark.
3. A clearly specified agent-assisted pseudo-label generation methodology.
4. Direct pseudo-label-quality comparisons with non-agent alternatives.
5. Downstream classification results across verified-label budgets.
6. Ablations, uncertainty/coverage trade-offs, and generalization/validity analysis.
7. A discussion of how the methodology could apply to long-running, unlabeled enclosure-mounted CNC IMU recordings **without claiming their unknown states have been validated**.

## 11. Decision gates before implementation

Resolve these research decisions in order:

- **Dataset validity:** Is there a public dataset with trustworthy cutting/non-cutting labels and independent groups?
- **Representation:** Are raw vibration measurements available, and which feature representation is justified?
- **Annotation protocol:** What exactly is the scarce-label selection policy, and what constitutes an ambiguous window?
- **Agent contribution:** Is the agent proposing labels, generating reusable labeling functions, combining evidence, or a prespecified combination?
- **Comparison fairness:** Which conventional baselines and resource budgets are mandatory?
- **Evaluation freeze:** What primary metric, label budgets, seeds/groups, and final-test boundary will be fixed before results are examined?

**Do not begin implementation until the dataset suitability gate and independent validation protocol are settled.**

## 12. Reference research areas

Foundational background to investigate during the literature review (not an implementation mandate):

- Semi-supervised learning and conventional confidence-based pseudo-labeling (e.g., *FixMatch*).
- Weak supervision and aggregation of imperfect labeling functions (e.g., *Snorkel*).
- Agentic tool use and automatic labeling-function induction for time-series data.
- Time-series classification on industrial vibration features.
- Group-aware evaluation, dataset shift, and leakage prevention in industrial sensing.

**Guiding principle:** The scientific claim is not that an agent can assign labels; it is whether **agent-generated pseudo-labels improve independently measured classification performance and annotation efficiency compared with strong non-agent baselines**.
