"""
MIOM Dashboard — dashboard/fleet_overview.py
=============================================
Top-row KPI cards: status counts, RMSE, score, mismatch rate.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_fleet_overview(df: pd.DataFrame) -> None:
    """Render 5 KPI metric cards across the top of the dashboard."""

    n_total    = len(df)
    n_critical = (df["pred_status"] == "🔴 CRITICAL").sum()
    n_warning  = (df["pred_status"] == "🟡 WARNING").sum()
    n_healthy  = (df["pred_status"] == "🟢 HEALTHY").sum()
    n_mismatch = df["mismatch"].sum()

    rmse_val  = df.attrs.get("test_rmse",  None)
    score_val = df.attrs.get("test_score", None)

    # ── Status cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric("Total engines", n_total)

    with c2:
        st.metric(
            "🔴 Critical",
            n_critical,
            delta=f"{n_critical/n_total*100:.0f}% of fleet",
            delta_color="inverse",
        )

    with c3:
        st.metric(
            "🟡 Warning",
            n_warning,
            delta=f"{n_warning/n_total*100:.0f}% of fleet",
            delta_color="off",
        )

    with c4:
        st.metric(
            "🟢 Healthy",
            n_healthy,
            delta=f"{n_healthy/n_total*100:.0f}% of fleet",
            delta_color="normal",
        )

    with c5:
        if rmse_val is not None:
            st.metric("RMSE", f"{rmse_val:.2f}", help="Lower is better")
        else:
            st.metric("RMSE", "—")

    with c6:
        if n_mismatch:
            st.metric(
                "⚠️ Mismatches",
                n_mismatch,
                delta="status disagreement",
                delta_color="inverse",
                help="Engines where true status ≠ predicted status",
            )
        else:
            st.metric("Mismatches", "0 ✓")

    # ── Inline progress bars for visual breakdown ────────────────────────────
    st.markdown(
        f"""
        <style>
        .miom-bar {{ height: 8px; border-radius: 4px; margin-bottom: 4px; }}
        </style>
        <div style="display:flex; gap:4px; margin-top:4px;">
            <div class="miom-bar" style="background:#E24B4A;
                 width:{n_critical/n_total*100:.1f}%;flex-shrink:0"
                 title="Critical: {n_critical}"></div>
            <div class="miom-bar" style="background:#EF9F27;
                 width:{n_warning/n_total*100:.1f}%;flex-shrink:0"
                 title="Warning: {n_warning}"></div>
            <div class="miom-bar" style="background:#639922;
                 width:{n_healthy/n_total*100:.1f}%;flex-shrink:0"
                 title="Healthy: {n_healthy}"></div>
        </div>
        <small style="color:var(--text-color);opacity:0.5">
            Fleet composition bar — red / amber / green
        </small>
        """,
        unsafe_allow_html=True,
    )