# Dataset description

## Machines
| ID | Machine | Rows | First row (shifted, UTC) | Last row (shifted, UTC) | Approx. days of data | Gaps > 60 s |
|---|---|---|---|---|---|---|
| machine_01 | DMG Mori Ecomill 1100 | 1,112,884 | 2025-01-27 18:00 | 2025-04-02 16:00 | 64.4 | 12 |
| machine_02 | DMG Mori DMU 500 | 841,621 | 2025-02-07 17:53 | 2025-04-02 16:00 | 48.7 | 6 |
| machine_03 | Okuma Multus B300II | 600,899 | 2025-02-26 17:36 | 2025-04-02 16:00 | 34.8 | 4 |
| machine_04 | Haas VF2 | 133,778 | 2025-02-26 22:24 | 2025-03-09 18:17 | 7.7 | 6 |
| machine_05 | Haas VF3 | 120,346 | 2025-03-06 23:02 | 2025-03-13 21:45 | 7.0 | 0 |
| machine_06 | Haas (model not recorded) | 92,891 | 2025-03-03 19:50 | 2025-03-27 20:26 | 5.4 | 21 |

"Approx. days of data" is rows x 5 s / 86400, i.e. the amount of data actually present, not the calendar span. Machine_06 spans about 24 calendar days but has only 5.4 days of rows, so it is sparse (median spacing 5 s, mean spacing about 22 s).

Machines are independent. Whether two of them are in the same shop is not disclosed.

## Rows and cadence
- Each row is one feature vector computed from a 2.56 s window of vibration data. A new window is taken every 5 s, so windows do **not** overlap and about half of the time is not observed.
- Median spacing between rows is 5 s on every machine. Timestamps have 1 ms resolution but are not aligned to a 5 s grid; they jitter by a fraction of a second. Resample or bin before treating them as a regular series.
- Rows are missing while a device was offline (power, Wi-Fi, maintenance). They are **not** zero-filled. Detect gaps from the time differences.
- Rows are sorted by time within each file.

## Timestamps
- Format `YYYY-MM-DDTHH:MM:SS.mmmZ`, UTC.
- All timestamps in every file are shifted back by the **same constant whole number of weeks**. Consequences:
  - weekday and UTC hour of day are real and consistent across machines;
  - calendar dates, months, seasons and public holidays are **not** real;
  - the relative timing between machines is preserved.
- The machines are in unspecified time zones. Shift patterns (e.g. day shift 7-15 h) appear at a machine-specific UTC offset, and the offset can change by 1 h across a daylight-saving boundary.

## What is not included
Machine names, owners, locations, serial numbers, device IDs, firmware versions, spindle speed, operating state, part programs, tool data, alarms, and any classification of the data.
