# Candidate labeled dataset card

**Status:** Candidate benchmark — suitability not yet verified  
**Identifier:** `DATA-001`  
**Local source:** `archive/` (ignored and not committed)  
**Compressed copy:** `archive.zip` (ignored and not committed)

## Inventory

The archive contains 69 dataset groups across three machine/source families:

| Source family | Groups | Observed machine metadata |
|---|---:|---|
| `imi_vm20i` | 37 | Hurco, from the accompanying MTConnect XML |
| `imi_vmx30ui` | 19 | Hurco, from the accompanying MTConnect XML |
| `tmf_vf10` | 13 | Haas, from the accompanying MTConnect XML |

Observed file types include:

- acceleration or measurement CSV files;
- `label.csv` interval annotations;
- WAV sensor recordings;
- reduced MP4 recordings;
- NetCDF files;
- MTConnect XML metadata;
- `dataset_summary.xlsx`.

The local copy is approximately 38 GB extracted and 14 GB compressed. It must
remain outside Git history.

## Label observations

The label files contain columns including `start`, `end`, `label`, `direction`,
and machining parameters such as `N`, `ap`, `ae`, `F`, `Z`, `D`, `Tool`, and
`Workpiece`. The observed label values include `0`, `1`, and `2`.

The semantic meaning of those values has not yet been independently verified.
They must not be treated as cutting/non-cutting ground truth until the original
dataset documentation, publication, and annotation protocol are identified.

The `start` and `end` fields appear to define intervals within recordings, but
their exact time base and transition policy still require verification.

## Suitability gate

Before using this archive for the thesis experiments, verify:

1. the dataset's original citation, release, and license;
2. the exact meaning and provenance of labels `0`, `1`, and `2`;
3. the sampling rate and units of each signal file;
4. the relationship between CSV, WAV, NetCDF, video, and label files;
5. independent grouping units such as machine, run, session, or workpiece;
6. whether the labels represent cutting versus non-cutting rather than another
   task such as tool condition or machining quality;
7. a leakage-safe train/validation/test split at the run or session level.

Until these checks are complete, this is a promising candidate dataset, not a
validated benchmark.
