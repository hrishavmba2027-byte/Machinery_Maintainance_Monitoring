"""
MIOM Dashboard — dashboard/rul_chart.py
=========================================
Predicted vs Actual RUL chart and error distribution.
Uses Plotly for interactivity.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import STATUS_COLORS, CRITICAL_THRESHOLD, WARNING_THRESHOLD, RUL_CEILING


def render_rul_chart(df: pd.DataFrame) -> None:
    st.markdown("### Predicted vs Actual RUL")

    has_true = (df["true_RUL"] != -1).any()

    tab1, tab2 = st.tabs(["Scatter", "Error distribution"])

    with tab1:
        _scatter_chart(df, has_true)

    with tab2:
        _error_histogram(df, has_true)


def _scatter_chart(df: pd.DataFrame, has_true: bool) -> None:
    fig = go.Figure()

    if has_true:
        # Perfect prediction line
        max_val = int(df["true_RUL"].max()) + 10
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color="rgba(128,128,128,0.4)", dash="dash", width=1),
            name="Perfect prediction",
            hoverinfo="skip",
        ))

        # Threshold bands
        fig.add_hrect(y0=0, y1=CRITICAL_THRESHOLD,
                      fillcolor="rgba(226,75,74,0.07)", line_width=0)
        fig.add_hrect(y0=CRITICAL_THRESHOLD, y1=WARNING_THRESHOLD,
                      fillcolor="rgba(239,159,39,0.07)", line_width=0)

    # One trace per status category for colour-coded legend
    for status, color in STATUS_COLORS.items():
        mask = df["pred_status"] == status
        sub  = df[mask]
        if sub.empty:
            continue

        x_vals = sub["true_RUL"].tolist() if has_true else sub["pred_RUL"].tolist()
        hover  = (
            sub.apply(
                lambda r: (
                    f"Engine {r['engine_id']}<br>"
                    f"True RUL: {r['true_RUL']}<br>"
                    f"Pred RUL: {r['pred_RUL']}<br>"
                    f"Error: {r['error']}<br>"
                    f"Status: {r['pred_status']}"
                    + (" ⚠ MISMATCH" if r["mismatch"] else "")
                ),
                axis=1,
            ).tolist()
        )

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=sub["pred_RUL"].tolist(),
            mode="markers",
            marker=dict(
                color=color,
                size=8,
                symbol=["diamond" if m else "circle" for m in sub["mismatch"]],
                opacity=0.85,
                line=dict(
                    color=["black" if m else color for m in sub["mismatch"]],
                    width=[1.5 if m else 0 for m in sub["mismatch"]],
                ),
            ),
            name=status,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    x_label = "True RUL (cycles)" if has_true else "Engine index"
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Predicted RUL (cycles)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=50, l=50, r=20),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _error_histogram(df: pd.DataFrame, has_true: bool) -> None:
    if not has_true:
        st.info("Error distribution unavailable — no ground-truth RUL.")
        return

    errors = df["error"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        marker_color="rgba(55,138,221,0.6)",
        marker_line=dict(color="rgba(55,138,221,1)", width=0.5),
        name="Error",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(128,128,128,0.6)")
    fig.update_layout(
        xaxis_title="Prediction error (pred − true)",
        yaxis_title="Count",
        margin=dict(t=20, b=50, l=50, r=20),
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    mean_err = float(errors.mean())
    std_err  = float(errors.std())
    st.caption(
        f"Mean error: **{mean_err:+.2f}** cycles · "
        f"Std: **{std_err:.2f}** · "
        f"{'Under-predicting on average (conservative)' if mean_err < 0 else 'Over-predicting on average'}"
    )