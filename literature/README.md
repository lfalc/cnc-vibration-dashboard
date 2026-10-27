# Literature index

These papers support the data-preparation and labeling decisions in the
roadmap. The local PDFs are retained as research inputs; the thesis should
cite the published versions directly.

## Local papers

| File | Topic | Use in this project |
| --- | --- | --- |
| [`s41597-026-07255-7.pdf`](s41597-026-07255-7.pdf) | MSM multi-sensor milling dataset | Dataset provenance, modalities, annotation protocol, and label semantics. The labels describe normal cutting, abnormal cutting, and tool defects; they are not automatically binary cutting/idling labels. |
| [`778_2019_Article_552.pdf`](778_2019_Article_552.pdf) | Snorkel weak supervision | Reference for labeling functions, abstention, label-model aggregation, and the non-agent weak-supervision baseline. |
| [`ssrn-7601837.pdf`](ssrn-7601837.pdf) | Temporal leakage in windowed time-series classification | Supports splitting by complete recording groups before extracting overlapping windows. |
| `inventions-03-00025.pdf` | Unknown | Empty file; not usable until replaced with a valid copy. |

## Additional recommended reading

- [A novel method for signal labeling and precise location in a variable parameter milling process](https://www.sciencedirect.com/science/article/pii/S1474034622003081) — sequence labeling across static, idle, acceleration, deceleration, and cutting states; useful for transition-window policy.
- [On-line chatter detection in milling with hybrid machine learning and physics-based model](https://doi.org/10.1016/j.cirpj.2021.05.006) — moving short-time frequency spectra and multiple machining states; useful for spectrogram feature design.
- [Learning Time Series Detection Models from Temporally Imprecise Labels](https://proceedings.mlr.press/v54/adams17a.html) — methods for labels whose temporal boundaries are uncertain.
- [Learning Under Temporal Label Noise](https://openreview.net/pdf?id=5o0phqAhsP) — recent treatment of time-varying label noise in sequential classification.

## Current implementation implications

1. Keep train, validation, and test groups disjoint before window extraction.
2. Exclude or explicitly mark windows crossing annotation boundaries; do not
   silently force them into a binary class.
3. Use labeling functions with abstention as the first non-agent baseline.
4. Evaluate pseudo-label quality separately from downstream classifier quality.
5. Preserve the distinction between cutting-state labels and anomaly labels.
