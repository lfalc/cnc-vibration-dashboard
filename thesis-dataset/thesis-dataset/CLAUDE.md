# Machine vibration dataset: instructions for the AI agent

You are helping a master's student analyse a **raw, anonymised industrial machine-vibration dataset** and build their own algorithms and models on it. Read this file first, then the docs in the order below.

## What the data is
- Vibration features from **6 CNC machines** (DMG Mori, Okuma, Haas), measured by a small accelerometer unit bolted to each machine.
- One row every **5 seconds**; each row has **84 numeric features** (3 axes x 28) computed on the device from 2.56 s of 800 Hz acceleration data. No raw waveforms exist.
- About **2.9 million rows** in total, in `data/machine_01.csv.gz` ... `data/machine_06.csv.gz`.
- **No labels.** Nobody annotated machine states, parts, tools, spindle speeds or faults. Do not invent ground truth.

## Read in this order
1. `README.md` - overview and loading snippets (pandas, R, DuckDB)
2. `docs/DATASET.md` - the six machines, row counts, time spans, gaps, timestamp rules
3. `docs/DATA_DICTIONARY.md` - all 85 columns with units
4. `docs/SENSOR_AND_FEATURES.md` - how each feature is computed on the device (needed to interpret values correctly)
5. `docs/CAVEATS.md` - pitfalls (missing values, leakage, per-machine scaling). Read before modelling.
6. `docs/SUGGESTED_ANALYSES.md` - optional starting ideas
7. `MANIFEST.txt` - file sizes, row counts and sha256 checksums

## Rules
- Treat the machine files as **independent machines** (`machine_01` .. `machine_06`). Normalise per machine; do not pool raw values across machines without checking.
- **Timestamps are shifted** by a constant number of whole weeks. Weekday and hour of day (UTC) are preserved, calendar dates are not real. Do not infer dates, seasons or holidays.
- Do not try to identify the machine owners or the sites. Brand and model are the only identifying information included, on purpose.
- Use **time-blocked** train/validation/test splits. Random row splits leak, because neighbouring rows are almost identical (see `docs/CAVEATS.md`).
- Missing values are empty CSV fields (NaN). Do not fill them with zero.
- Do not claim accuracy against labels that do not exist. Report unsupervised results as such, and state clearly any weak labels you derive yourself and how.
- When unsure what a column means, check `docs/DATA_DICTIONARY.md` instead of guessing.
- Large files: load with pandas using `usecols`/`dtype=float32`, or query with DuckDB. Keep memory in mind (machine_01 is about 1.1M x 85 values).
