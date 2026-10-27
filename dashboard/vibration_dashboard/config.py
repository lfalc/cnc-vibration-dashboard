"""Shared dataset and dashboard configuration."""

from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = DASHBOARD_DIR / "cache"
DATA_DIR = DASHBOARD_DIR.parent / "thesis-dataset" / "thesis-dataset" / "data"

AXES = ["x", "y", "z"]
BANDS = [f"band_{i:02d}_db" for i in range(1, 17)]
BAND_CENTERS_HZ = [
    11.2, 14.1, 17.8, 22.4, 28.2, 35.5, 44.7, 56.2,
    70.8, 89.1, 112.2, 141.3, 177.8, 223.9, 281.8, 354.8,
]
MACHINE_NAMES = {
    "machine_01": "DMG Mori Ecomill 1100",
    "machine_02": "DMG Mori DMU 500",
    "machine_03": "Okuma Multus B300II",
    "machine_04": "Haas VF2",
    "machine_05": "Haas VF3",
    "machine_06": "Haas (model not recorded)",
}
FEATURES = [
    "rms", "kurtosis", "crest", "skewness",
    "peak1_hz", "peak1_amp", "peak2_hz", "peak2_amp",
    "peak3_hz", "peak3_amp", "peak4_hz", "peak4_amp",
] + BANDS
FEATURE_LABELS = {
    "rms": "RMS (g)", "kurtosis": "Kurtosis", "crest": "Crest factor",
    "skewness": "Skewness", "peak1_hz": "Peak 1 frequency (Hz)",
    "peak1_amp": "Peak 1 amplitude", "peak2_hz": "Peak 2 frequency (Hz)",
    "peak2_amp": "Peak 2 amplitude", "peak3_hz": "Peak 3 frequency (Hz)",
    "peak3_amp": "Peak 3 amplitude", "peak4_hz": "Peak 4 frequency (Hz)",
    "peak4_amp": "Peak 4 amplitude",
}

STAT_FEATURES = ["rms", "kurtosis", "crest", "skewness"]
PEAK_FEATURES = [
    "peak1_hz", "peak1_amp", "peak2_hz", "peak2_amp",
    "peak3_hz", "peak3_amp", "peak4_hz", "peak4_amp",
]
ALL_FEATURES = STAT_FEATURES + PEAK_FEATURES + BANDS
