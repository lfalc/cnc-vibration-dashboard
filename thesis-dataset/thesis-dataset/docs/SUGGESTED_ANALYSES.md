# Suggested starting points (ideas only)

These are neutral ideas, not a prescribed method and not a description of any existing product algorithm.

1. **Exploration.** Per machine: plot `x_rms`, `y_rms`, `z_rms` over time; histograms of rms (log scale); daily and weekly profiles using UTC hour and weekday; spectrogram-style heatmap of the 16 band columns over time.
2. **State discovery (unsupervised).** Cluster minutes or windows into operating regimes (idle / spindle running / heavy cut) using gaussian mixtures, k-means or HMMs on rms and band energies. Report stability across machines.
3. **Activity detection baselines.** Thresholding on band energy relative to a per-machine noise floor, change-point detection (PELT, BOCPD), and compare with the clusters from (2).
4. **Representation learning.** Autoencoder or contrastive embeddings of the 84-dimensional vectors; examine whether regimes separate; test transfer between machines.
5. **Anomaly detection.** Isolation forest, one-class SVM, reconstruction error on the "normal" distribution of each machine; evaluate with synthetic anomalies or manual inspection since no labels exist.
6. **Sequence models.** Next-step prediction or masked-feature modelling over sliding windows; evaluate with blocked time splits.
7. **Spindle-speed inference.** The strongest low-frequency peak and its harmonics (peak_hz columns) relate to rotation frequency; try estimating and tracking it. Note the 0.39 Hz frequency grid.
8. **Cross-machine transfer.** Train on machines 1-3, test on 4-6, with per-machine normalisation. Document how performance drops without it.

Keep a clear record of every derived label and threshold, so results are reproducible.
