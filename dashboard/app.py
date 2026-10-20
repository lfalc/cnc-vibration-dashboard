#!/usr/bin/env python3
"""Streamlit dashboard for daily vibration features and spectral bands."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Vibration Dashboard", page_icon="📊", layout="wide")

CACHE_DIR = Path(__file__).parent / "cache"
AXES = ["x", "y", "z"]
BANDS = [f"band_{i:02d}_db" for i in range(1, 17)]
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
    frame = pd.read_parquet(CACHE_DIR / f"{machine}_spectral_15min.parquet")
    frame["time_bin"] = pd.to_datetime(frame["time_bin"], utc=True)
    return frame


def feature_columns(axis: str) -> List[str]:
    return [f"{axis}_{feature}" for feature in FEATURES]


def trend_chart(frame: pd.DataFrame, axis: str, feature: str) -> go.Figure:
    prefix = f"{axis}_{feature}"
    median = f"{prefix}_median"
    p10 = f"{prefix}_p10"
    p90 = f"{prefix}_p90"
    if median not in frame:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.date, y=frame[p90], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=frame.date, y=frame[p10], line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,123,255,.15)", name="10th–90th percentile", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=frame.date, y=frame[median], mode="lines+markers",
        line=dict(color="#007bff", width=2), marker=dict(size=4),
        name="Daily median", hovertemplate="%{x|%Y-%m-%d}<br>Median: %{y:.4g}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{axis.upper()}-axis {FEATURE_LABELS.get(feature, feature)} — daily median",
        xaxis_title="Date (shifted UTC)", yaxis_title=FEATURE_LABELS.get(feature, feature),
        hovermode="x unified", template="plotly_dark", height=400,
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def spectral_chart(frame: pd.DataFrame, axis: str) -> go.Figure:
    band_cols = [f"{axis}_{band}" for band in BANDS]
    available = [column for column in band_cols if column in frame]
    if frame.empty or not available:
        return go.Figure()
    values = frame[available].T.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmin, vmax = np.nanpercentile(finite, [1, 99]) if finite.size else (-20, 10)
    labels = [f"Band {i:02d}" for i in range(1, len(available) + 1)]
    fig = go.Figure(go.Heatmap(
        z=values, x=frame.time_bin, y=labels, zmin=vmin, zmax=vmax,
        colorscale="Blues", colorbar=dict(title="Relative dB", thickness=15, len=.8),
        connectgaps=False,
        hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>%{y}<br>%{z:.2f} dB<extra></extra>",
    ))
    fig.update_layout(
        title=f"Spectral band heatmap — {axis.upper()}-axis (15-minute median bins)",
        xaxis_title="Time (shifted UTC)", yaxis_title="Frequency band", template="plotly_dark",
        height=500, margin=dict(l=80, r=60, t=60, b=60), yaxis=dict(autorange="reversed"),
    )
    return fig


def main():
    st.markdown("""
    <style>
    .stApp { background-color: #1a1a2e; }
    .main-header { color: #e8e8e8; font-weight: 300; letter-spacing: .05em; }
    .note-text { color: #ffc107; font-size: .85rem; font-style: italic; }
    .disclaimer { color: #aaa; font-size: .8rem; margin-top: 1rem; padding: .6rem;
                  background: #0f0f1a; border-radius: 4px; border-left: 3px solid #ffc107; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">VIBRATION OBSERVATORY</h1>', unsafe_allow_html=True)
    st.markdown('<p class="note-text">Daily feature trends · Spectral band heatmap</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Controls")
        machines = available_machines()
        if not machines:
            st.error("No cached machine data found. Run preprocessing first.")
            st.stop()
        machine = st.selectbox("Machine", machines)
        metadata = load_metadata(machine)
        axis = st.selectbox("Axis", AXES)
        feature = st.selectbox("Feature", FEATURES, format_func=lambda value: FEATURE_LABELS.get(value, value))
        daily = load_daily(machine)
        minimum = daily.date.min().date()
        maximum = daily.date.max().date()
        selected = st.date_input("Date range (shifted UTC)", value=(minimum, maximum), min_value=minimum, max_value=maximum)
        if len(selected) == 2:
            start, end = selected
        else:
            start, end = minimum, maximum
        st.divider()
        st.caption(f"Rows: {metadata['row_count']:,}")
        st.caption(f"Range: {metadata['date_range']['start'][:10]} → {metadata['date_range']['end'][:10]}")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    daily_view = daily[(daily.date >= start_ts) & (daily.date < end_ts)]
    spectral = load_spectral(machine)
    spectral_view = spectral[(spectral.time_bin >= start_ts) & (spectral.time_bin < end_ts)]

    st.plotly_chart(trend_chart(daily_view, axis, feature), use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(spectral_chart(spectral_view, axis), use_container_width=True, config={"displaylogo": False})
    st.markdown("""
    <div class="disclaimer"><strong>Notes:</strong><br>
    • Dates are shifted and should not be interpreted as real calendar dates.<br>
    • The dataset has no raw waveforms, machine-state labels, fault labels, or anomaly labels.<br>
    • Band values are relative dB measurements from 16 recorded spectral bands.<br>
    • Missing values and offline gaps are preserved rather than filled with zero.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
