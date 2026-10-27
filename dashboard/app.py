#!/usr/bin/env python3
"""Streamlit entry point for the day-level vibration dashboard."""

import pandas as pd
import streamlit as st

from vibration_dashboard.charts import combined_day_chart
from vibration_dashboard.config import AXES, CACHE_DIR, FEATURE_LABELS, FEATURES, MACHINE_NAMES
from vibration_dashboard.data import (
    available_machines,
    load_daily,
    load_features,
    load_metadata,
    load_spectral,
)


st.set_page_config(page_title="Vibration Dashboard", page_icon="📊", layout="wide")


def apply_styles() -> None:
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #374151; }
    .main-header { color: #374151; font-weight: 600; letter-spacing: .01em; }
    .note-text { color: #4b5563; font-size: 1rem; }
    .disclaimer { color: #4b5563; font-size: .9rem; line-height: 1.55; margin-top: 1rem; padding: .7rem;
                  background: #f8f9fb; border-radius: 4px; border-left: 3px solid #e87919; }
    [data-testid="stSidebar"] { color: #374151; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stCaption { color: #4b5563 !important; }
    [data-testid="stCaptionContainer"] p { color: #4b5563 !important; font-size: .9rem; }
    .stSelectbox label, .stDateInput label { color: #374151 !important; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    apply_styles()
    st.markdown('<h1 class="main-header">Vibration review</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="note-text">Read-only daily feature viewer · recorded spectral bands · no classifier output</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        machines = available_machines()
        if not machines:
            st.error("No cached machine data found. Run preprocessing first.")
            st.stop()

        machine = st.selectbox("Machine", machines, format_func=lambda value: MACHINE_NAMES.get(value, value))
        metadata = load_metadata(machine)
        axis = st.selectbox("Axis", AXES)
        feature = st.selectbox("Feature", FEATURES, format_func=lambda value: FEATURE_LABELS.get(value, value))
        skip_empty = st.checkbox("Skip days with empty data", value=False)
        daily = load_daily(machine)
        minimum = daily.date.min().date()
        maximum = daily.date.max().date()
        all_days = list(pd.date_range(minimum, maximum, freq="D").date)
        count_column = f"{axis}_{feature}_count"
        populated_days = list(daily.loc[daily[count_column].fillna(0) > 0, "date"].dt.date) if count_column in daily else []
        navigation_days = populated_days if skip_empty else all_days
        if not navigation_days:
            st.warning("No non-empty days are available for this feature.")
            navigation_days = all_days

        day_key = f"review_day_value_{machine}"
        if day_key not in st.session_state:
            st.session_state[day_key] = minimum
        review_day = st.date_input(
            "Review day (shifted UTC)",
            value=st.session_state[day_key],
            min_value=minimum,
            max_value=maximum,
        )
        if skip_empty and review_day not in navigation_days:
            later_days = [day for day in navigation_days if day > review_day]
            st.session_state[day_key] = later_days[0] if later_days else navigation_days[-1]
            st.rerun()

        st.divider()
        st.caption(MACHINE_NAMES.get(machine, machine))
        st.caption(f"Rows: {metadata['row_count']:,}")
        st.caption(f"Range: {metadata['date_range']['start'][:10]} → {metadata['date_range']['end'][:10]}")

    start_ts = pd.Timestamp(review_day, tz="UTC")
    end_ts = start_ts + pd.Timedelta(days=1)
    features = load_features(machine)
    feature_view = features[(features.time_bin >= start_ts) & (features.time_bin < end_ts)]
    spectral = load_spectral(machine)
    spectral_view = spectral[(spectral.time_bin >= start_ts) & (spectral.time_bin < end_ts)]
    value_column = f"{axis}_{feature}"

    if feature_view.empty or not feature_view[value_column].notna().any():
        st.info("This feature has no recorded values on the selected day. Enable 'Skip days with empty data' to navigate only to populated days.")
    else:
        st.plotly_chart(
            combined_day_chart(feature_view, spectral_view, axis, feature, start_ts, end_ts),
            width="stretch",
            config={"displaylogo": False},
        )

    day_index = navigation_days.index(review_day) if review_day in navigation_days else 0
    previous, next_day, position = st.columns([1, 1, 2.4])
    with previous:
        if st.button("‹ Previous day", width="stretch", disabled=day_index == 0):
            st.session_state[day_key] = navigation_days[day_index - 1]
            st.rerun()
    with next_day:
        if st.button("Next day ›", width="stretch", disabled=day_index == len(navigation_days) - 1):
            st.session_state[day_key] = navigation_days[day_index + 1]
            st.rerun()
    with position:
        st.markdown(f"**{review_day}** — day {day_index + 1} of {len(navigation_days)}")

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
