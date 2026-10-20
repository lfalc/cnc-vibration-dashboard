# Machine vibration dataset (raw 5 s features)

Anonymised raw feature data from six CNC machines, provided for thesis research. One row per 5 seconds per machine, 84 spectral and statistical vibration features (x, y, z axes).

| File | Machine | Rows | Size (gz) |
|---|---|---|---|
| `data/machine_01.csv.gz` | DMG Mori Ecomill 1100 | 1,112,884 | ~233 MB |
| `data/machine_02.csv.gz` | DMG Mori DMU 500 | 841,621 | ~181 MB |
| `data/machine_03.csv.gz` | Okuma Multus B300II | 600,899 | ~130 MB |
| `data/machine_04.csv.gz` | Haas VF2 | 133,778 | ~29 MB |
| `data/machine_05.csv.gz` | Haas VF3 | 120,346 | ~26 MB |
| `data/machine_06.csv.gz` | Haas (model not recorded) | 92,891 | ~20 MB |

Total: 2,902,419 rows, about 620 MB compressed (several GB in memory as float64).

## Contents
- `data/` - gzipped CSVs, one per machine, header row included
- `docs/` - dataset description, data dictionary, sensor/feature definitions, caveats, analysis ideas
- `CLAUDE.md` / `AGENTS.md` - instructions for AI coding agents (same content)
- `MANIFEST.txt` - checksums and counts

## Quick start

pandas:
```python
import pandas as pd
df = pd.read_csv("data/machine_01.csv.gz", parse_dates=["time"], index_col="time")
df = df.astype("float32")          # halves memory
print(df.shape, df.index.min(), df.index.max())
```

DuckDB (handles all machines without loading them into memory):
```python
import duckdb
duckdb.sql("""
  SELECT date_trunc('hour', time) AS hour, avg(x_rms) AS x_rms
  FROM read_csv('data/machine_01.csv.gz')
  GROUP BY 1 ORDER BY 1
""").df()
```

R:
```r
library(data.table)
d <- fread("data/machine_01.csv.gz")
d[, time := as.POSIXct(time, format="%Y-%m-%dT%H:%M:%OSZ", tz="UTC")]
```

## Key facts
- Cadence: one row every ~5 s while the device is online. Gaps exist.
- No labels, no machine-state information.
- Time is UTC, shifted by a constant whole number of weeks (dates are not real).
- Start with `docs/DATASET.md`, `docs/DATA_DICTIONARY.md` and `docs/CAVEATS.md`.

## Provenance and use
Data was recorded by Abnito sensor devices on production machines and anonymised for academic use. Do not redistribute the files and do not try to identify the sites or companies. Agree with Abnito how the data source may be named in the thesis and any publication before submitting.
