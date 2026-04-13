"""
MIOM Dashboard — dashboard/triage_table.py
===========================================
Sortable, colour-coded fleet triage table.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import STATUS_COLORS


def render_triage_table(df: pd.DataFrame) -> None:
    st.markdown("### Fleet Triage Table")

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 3])

    with col_f1:
        status_filter = st.multiselect(
            "Filter by predicted status",
            options=["🔴 CRITICAL", "🟡 WARNING", "🟢 HEALTHY"],
            default=["🔴 CRITICAL", "🟡 WARNING", "🟢 HEALTHY"],
            label_visibility="collapsed",
        )

    with col_f2:
        mismatch_only = st.checkbox("Show mismatches only", value=False)

    with col_f3:
        max_rul = st.slider(
            "Max predicted RUL to show",
            min_value=0,
            max_value=int(df["pred_RUL"].max()) + 10,
            value=int(df["pred_RUL"].max()) + 10,
            step=5,
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = df[df["pred_status"].isin(status_filter)]
    filtered = filtered[filtered["pred_RUL"] <= max_rul]
    if mismatch_only:
        filtered = filtered[filtered["mismatch"]]

    st.caption(f"Showing **{len(filtered)}** of **{len(df)}** engines")

    # ── Build display DataFrame ───────────────────────────────────────────────
    display = filtered[
        ["engine_id", "true_RUL", "pred_RUL", "error", "true_status", "pred_status", "mismatch"]
    ].copy()

    has_true = (display["true_RUL"] != -1).any()
    if not has_true:
        display = display.drop(columns=["true_RUL", "error", "true_status"])

    display["mismatch"] = display["mismatch"].map({True: "⚠ YES", False: ""})
    display = display.rename(columns={
        "engine_id"  : "Engine",
        "true_RUL"   : "True RUL",
        "pred_RUL"   : "Pred RUL",
        "error"      : "Error",
        "true_status": "True Status",
        "pred_status": "Pred Status",
        "mismatch"   : "Mismatch",
    })

    # ── Style ─────────────────────────────────────────────────────────────────
    def _colour_status(val: str) -> str:
        color = STATUS_COLORS.get(val, "")
        if not color:
            return ""
        return f"color: {color}; font-weight: 500;"

    def _colour_error(val) -> str:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v > 10:
            return "color: #EF9F27"
        if v < -10:
            return "color: #E24B4A"
        return "color: #639922"

    def _colour_mismatch(val: str) -> str:
        return "color: #E24B4A; font-weight: 500;" if val == "⚠ YES" else ""

    styler = display.style

    if "Pred Status" in display.columns:
        styler = styler.map(_colour_status, subset=["Pred Status"])
    if "True Status" in display.columns:
        styler = styler.map(_colour_status, subset=["True Status"])
    if "Error" in display.columns:
        styler = styler.map(_colour_error, subset=["Error"])
    if "Mismatch" in display.columns:
        styler = styler.map(_colour_mismatch, subset=["Mismatch"])

    st.dataframe(
        styler,
        use_container_width=True,
        height=min(600, 38 + 35 * len(display)),
    )

    # ── Action lists ──────────────────────────────────────────────────────────
    critical_engines = (
        filtered[filtered["pred_status"] == "🔴 CRITICAL"]["engine_id"].tolist()
    )
    warning_engines = (
        filtered[filtered["pred_status"] == "🟡 WARNING"]["engine_id"].tolist()
    )

    if critical_engines:
        st.error(
            f"**Immediate action required:** Engines {critical_engines}",
            icon="🔴",
        )
    if warning_engines:
        st.warning(
            f"**Schedule maintenance:** Engines {warning_engines}",
            icon="🟡",
        )