# Data dictionary

Each CSV has **85 columns**: `time` plus 84 features. There is no machine/device column; the file name is the machine.

## Column naming
`{axis}_{feature}` with axis in `x`, `y`, `z`. 28 features per axis:

| Feature | Columns per axis | Unit | Meaning |
|---|---|---|---|
| `rms` | 1 | g | RMS of the high-pass-filtered (>10 Hz) time signal over the 2.56 s window |
| `kurtosis` | 1 | none | Fisher (excess) kurtosis of the same signal; 0 for Gaussian noise |
| `crest` | 1 | none | max abs value / rms |
| `skewness` | 1 | none | sample skewness |
| `peak1_hz` ... `peak4_hz` | 4 | Hz | frequencies of the 4 strongest spectral peaks in 10-400 Hz, strongest first |
| `peak1_amp` ... `peak4_amp` | 4 | arbitrary (linear) | FFT magnitude of those peaks; relative only, not dB, not g |
| `band_01_db` ... `band_16_db` | 16 | dB (relative) | energy in 16 log-spaced bands between 10 and 400 Hz (see `SENSOR_AND_FEATURES.md` for band edges) |

Column order in the files: for each axis x, y, z: `rms, kurtosis, crest, skewness, peak1_hz, peak1_amp, peak2_hz, peak2_amp, peak3_hz, peak3_amp, peak4_hz, peak4_amp, band_01_db ... band_16_db`.

Examples: `x_rms`, `y_kurtosis`, `z_peak2_hz`, `x_band_07_db`.

## Types and value ranges
- `time`: UTC timestamp string, shifted (see `DATASET.md`).
- All other columns: floating point, stored as float32 precision (about 7 significant digits).
- Empty field = missing (NaN). See `CAVEATS.md`.
- `*_peak*_hz` values lie between 10 and 400 Hz and fall on a 0.39 Hz grid (800 Hz / 2048 bins).
- `*_band_NN_db` are small numbers (in a sample of machine_03: roughly -24 to +30 dB, median about -8 dB) because the underlying FFT scaling is arbitrary; only differences and trends are meaningful. Silent or empty bands floor at 10*log10(1e-12) = -120 dB.
- Observed `x_rms` ranges per machine are roughly 0.008-0.4 g (machine_01 0.008-0.35, machine_02 0.009-0.29, machine_03 0.009-0.20, machine_04 0.010-0.06, machine_05 0.013-0.03, machine_06 0.009-0.41).

## Axes
x, y, z refer to the **sensor's** axes in the order the chip reports them. The sensor's orientation relative to the machine's own X/Y/Z axes is not recorded and probably differs between machines. Do not assume `x` means machine-X.
