# Daily vibration dashboard plan

## Summary

Build a local Streamlit dashboard that combines:

1. Daily feature trends for one machine at a time.
2. A spectrogram-style time–frequency heatmap using the recorded 16 spectral bands.

The dataset contains no raw waveforms, so the heatmap will be labeled as a
“spectral band heatmap,” not a reconstructed spectrogram.

## Key changes

- Add a dashboard under `dashboard/`.
- Stream-read each `.csv.gz` file with pandas; manual extraction of the gzip
  files is unnecessary.
- Preserve the original compressed dataset unchanged.
- Write derived cache files under an ignored dashboard cache directory:
  - daily median, 10th percentile, 90th percentile, and sample count for all
    features;
  - time-binned spectral band summaries for each machine, axis, and band.
- Use fixed 15-minute bins for the spectral heatmap to keep the dashboard
  responsive while preserving operating-pattern changes.
- Build the Streamlit interface with:
  - machine selector;
  - axis selector;
  - feature selector for daily trends;
  - shifted UTC date-range selector;
  - daily median trend with 10th–90th percentile band;
  - daily observation-count/coverage view;
  - spectral band heatmap with time on the horizontal axis and bands 01–16 on
    the vertical axis.
- Use the documented band ordering and clarify that band values are relative dB
  values.
- Display an explicit note that dates are shifted and that the dataset contains
  no raw waveform, machine-state, fault, or anomaly labels.
- Use a restrained industrial-observatory visual direction: dark graphite
  interface, cool blue spectral intensity scale, amber indicators for sparse
  coverage, and accessible contrast.
- Add setup documentation and a reproducible dependency file.

## Implementation direction

Use Python, pandas, Plotly, and Streamlit for the first version. Rust is not
needed for the initial dashboard; if preprocessing becomes a bottleneck, use
Polars through Python or add a separate Rust preprocessing tool later without
changing the dashboard interface.

The preprocessing command should scan the compressed files once and generate
compact derived cache files. The app should use those cache files for normal
interaction rather than repeatedly loading millions of raw rows.

## Test plan

- Verify gzip inputs are read without modifying or duplicating the source
  dataset.
- Test preprocessing on every machine file and compare row counts and date
  ranges with the dataset documentation.
- Verify daily aggregation handles NaNs, missing days, midnight boundaries, and
  sparse sampling.
- Verify the heatmap contains exactly 16 ordered bands per selected axis.
- Confirm sparse periods and long gaps remain visible rather than being filled
  with zeros.
- Run a local Streamlit smoke test covering machine, axis, feature, date-range,
  and heatmap controls.
- Confirm the UI consistently calls the output a spectral band heatmap rather
  than a raw spectrogram.

## Assumptions

- The app remains local and uses Streamlit.
- The original `.csv.gz` files remain the source of truth.
- Preprocessing creates derived cache files for performance; cache files are not
  committed as source data.
- Daily trends use median plus 10th–90th percentile range.
- The spectral heatmap uses 15-minute median bins and the original 16 recorded
  frequency bands.
- One machine is selected at a time so machine-specific scales remain
  interpretable.
