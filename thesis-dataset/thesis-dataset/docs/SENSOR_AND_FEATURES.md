# Sensor and feature computation

Source of truth: the device design specification. This file restates the parts needed for analysis.

## Sensor
- A small device with a **3-axis MEMS accelerometer** (Bosch BMA400 class, +/-4 g range, 7.81 mg per LSB) is attached to each machine.
- Sampled at **800 Hz** output data rate, so the usable bandwidth is 0-400 Hz (Nyquist). Anything above 400 Hz is not represented.
- Hardware and firmware revisions varied between devices and over time; calibration and mounting differ between machines.

## What the device does every 5 seconds
1. Take the most recent **2048 samples (2.56 s)** per axis. Consecutive windows do not overlap; the device only sees about 51% of the time.
2. **Remove the mean** (DC) per axis.
3. **Time-domain statistics** (rms, kurtosis, crest, skewness): computed after a **2nd-order Butterworth 10 Hz high-pass** filter, on the unwindowed signal.
4. **Spectrum:** apply a **Hann window**, take a real 2048-point FFT. Bin width = 800/2048 = 0.390625 Hz. Magnitudes below 10 Hz and above 400 Hz are masked out.
5. **Peaks:** find local maxima of the masked magnitude spectrum, sort by magnitude, keep the **top 4**, with a minimum separation of 5 bins (about 2 Hz) between selected peaks. Each peak is reported as frequency (Hz) and linear magnitude.
6. **Band energies:** 16 bands with edges `10 * 1.2589^k` Hz (1/3-octave spacing), k = 0..16. For each band: sum of squared FFT magnitudes of the bins in the band, divided by the bin count, converted with `10*log10(power + 1e-12)`.

The window is a snapshot, not an average over the 5 s. Transients shorter than 2.56 s that fall between windows are never seen.

## Band edges (Hz)
| Band | Lo | Hi | Band | Lo | Hi |
|---|---|---|---|---|---|
| 01 | 10.00 | 12.59 | 09 | 63.10 | 79.43 |
| 02 | 12.59 | 15.85 | 10 | 79.43 | 100.00 |
| 03 | 15.85 | 19.95 | 11 | 100.00 | 125.89 |
| 04 | 19.95 | 25.12 | 12 | 125.89 | 158.49 |
| 05 | 25.12 | 31.62 | 13 | 158.49 | 199.53 |
| 06 | 31.62 | 39.81 | 14 | 199.53 | 251.19 |
| 07 | 39.81 | 50.12 | 15 | 251.19 | 316.23 |
| 08 | 50.12 | 63.10 | 16 | 316.23 | 398.11 |

## Interpretation notes
- `rms` is in g and comparable across machines only if mounting and coupling were similar (they were not controlled).
- `peak_amp` and `band_db` share an arbitrary FFT scale. Compare within a machine, or compare differences between bands.
- The band energies are the same spectrum as the peaks, just coarser; they are strongly correlated with the peaks and with rms.
- Peak frequencies on a 0.39 Hz grid relate to spindle speed and its harmonics, tool pass frequency and mains-related lines. The device does not record spindle speed, so any link to rpm must be inferred.
- Where fewer than four peaks are found, the remaining peak fields are missing (empty).
