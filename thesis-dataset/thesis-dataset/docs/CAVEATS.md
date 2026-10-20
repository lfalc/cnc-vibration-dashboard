# Caveats and pitfalls (read before modelling)

1. **No labels.** There is no ground truth for machine state (idle, spindle on, cutting), part, tool, fault, or anomaly. Any label you use must be derived by you (thresholding, clustering, manual inspection) and described as such.
2. **Six machines only,** of three brands, with very different amounts of data (machine_01 is 38% of all rows; machine_04..06 are 3-5% each). Claims of generalisation across machines or shops are weak. Consider leave-one-machine-out evaluation and report per-machine results.
3. **Normalise per machine.** Sensor mounting, orientation, coupling and device revision differ. Absolute rms and dB levels are not comparable across machines.
4. **Autocorrelation and leakage.** Neighbouring rows (5 s apart) are nearly identical. Use blocked splits in time (e.g. first 70% / next 15% / last 15% of each machine, with a buffer between blocks), never random row splits. Beware of the same operating state appearing in both train and test.
5. **Missing values.** Empty fields are NaN. In `machine_01`, the first 26,640 rows (2.4%, about the first 1.5 days, up to 2025-01-29T07:00Z) contain only `rms` values (`x_rms`, `y_rms`, `z_rms`) and all other fields are empty; they come from an older device version. Drop or handle them. Other machines have no such block, but individual peaks can still be missing.
6. **Irregular sampling.** Rows are about 5 s apart with sub-second jitter and some long gaps (machine_06 especially: 21 gaps over 60 s, mean spacing about 22 s). Do not assume a regular grid; resample or window by real time.
7. **Partial observation.** Each row sees 2.56 s of every 5 s. Short events can be missed.
8. **Timestamps are shifted** by whole weeks (see `DATASET.md`). Do not infer dates or seasons. Time zones are unknown, so hour-of-day patterns are in UTC and offset by an unknown amount per machine.
9. **Heavy tails and states.** Machining produces strongly multi-modal data (idle vs. cutting, different spindle speeds). Distributions are skewed; consider log or robust scaling for rms and amplitudes. dB columns are already logarithmic.
10. **Dead or constant channels.** The y axis exists on every machine, but its quality may differ by sensor orientation. Check per-column variance before use.
11. **Clone data.** Rows were copied from live systems into a research copy; values are exactly as recorded, with no cleaning, outlier removal or imputation applied.
12. **Do not re-identify.** Do not try to determine the owner or site of a machine.
