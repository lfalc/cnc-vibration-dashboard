"""Plotly figures for the day-level dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vibration_dashboard.config import BAND_CENTERS_HZ, BANDS, FEATURE_LABELS


def combined_day_chart(
    feature_frame: pd.DataFrame,
    spectral_frame: pd.DataFrame,
    axis: str,
    feature: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> go.Figure:
    """Render the feature trace and spectral heatmap on one shared time axis."""
    value_column = f"{axis}_{feature}"
    band_columns = [f"{axis}_{band}" for band in BANDS]
    available_bands = [column for column in band_columns if column in spectral_frame]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.38, 0.62])

    fig.add_trace(go.Scatter(
        x=feature_frame.time_bin,
        y=feature_frame[value_column],
        mode="lines",
        line=dict(color="#e87919", width=1.8),
        connectgaps=False,
        name=FEATURE_LABELS.get(feature, feature),
        hovertemplate="%{x|%H:%M}<br>%{y:.5g}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    if available_bands and not spectral_frame.empty:
        values = spectral_frame[available_bands].T.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        vmin, vmax = np.nanpercentile(finite, [1, 99]) if finite.size else (-20, 10)
        fig.add_trace(go.Heatmap(
            z=values,
            x=spectral_frame.time_bin,
            y=BAND_CENTERS_HZ[:len(available_bands)],
            zmin=vmin,
            zmax=vmax,
            colorscale="Plasma",
            showscale=False,
            connectgaps=False,
            hovertemplate="%{x|%H:%M}<br>%{y:.1f} Hz<br>%{z:.2f} dB<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        height=780,
        margin=dict(l=70, r=24, t=18, b=58),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#4b5563", size=13),
        title_font=dict(color="#374151", size=16),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(range=[start_ts, end_ts], showgrid=False, linecolor="#cbd1da", row=2, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(
        title_text=FEATURE_LABELS.get(feature, feature), gridcolor="#e0e4ea", zeroline=False,
        tickfont=dict(color="#4b5563", size=12), title_font=dict(color="#374151"), row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Hz", type="log", gridcolor="#e0e4ea", zeroline=False,
        tickfont=dict(color="#4b5563", size=12), title_font=dict(color="#374151"), row=2, col=1,
    )
    return fig
