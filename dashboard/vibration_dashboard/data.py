"""Cached loading and feature-selection helpers."""

import json
from typing import Dict, List

import pandas as pd
import streamlit as st

from vibration_dashboard.config import CACHE_DIR, FEATURES


@st.cache_data
def available_machines() -> List[str]:
    return sorted(p.stem.replace("_metadata", "") for p in CACHE_DIR.glob("*_metadata.json"))


@st.cache_data
def load_metadata(machine: str) -> Dict:
    with open(CACHE_DIR / f"{machine}_metadata.json", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_daily(machine: str) -> pd.DataFrame:
    frame = pd.read_parquet(CACHE_DIR / f"{machine}_daily.parquet")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame


@st.cache_data
def load_spectral(machine: str) -> pd.DataFrame:
    frame = pd.read_parquet(CACHE_DIR / f"{machine}_spectral_1min.parquet")
    frame["time_bin"] = pd.to_datetime(frame["time_bin"], utc=True)
    return frame


@st.cache_data
def load_features(machine: str) -> pd.DataFrame:
    frame = pd.read_parquet(CACHE_DIR / f"{machine}_features_1min.parquet")
    frame["time_bin"] = pd.to_datetime(frame["time_bin"], utc=True)
    return frame


def feature_columns(axis: str) -> List[str]:
    return [f"{axis}_{feature}" for feature in FEATURES]
