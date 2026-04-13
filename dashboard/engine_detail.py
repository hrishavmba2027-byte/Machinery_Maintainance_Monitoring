"""
MIOM Dashboard — dashboard/engine_detail.py
=============================================
Per-engine drill-down: sensor time-series from processed CSV.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PROCESSED_DIR, FEATURE_COLS, STATUS_COLORS


def render_engine_detail(df: pd.DataFrame, subset: str) -> None:
    st.markdown("### Engine drill-down")

    # Engine selector sorted by urgency
    engine_options = df.sort_values("pred_RUL")["engine_id"].tolist()

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        selected = st.selectbox(
            "Select engine",
            options=engine_options,
            format_func=lambda eid: (
                f"Engine {eid} — "
                + df.loc[df["engine_id"] == eid, "pred_status"].values[0]
                + f" (RUL ≈ {df.loc[df['engine_id'] == eid, 'pred_RUL'].values[0]})"
            ),
        )

    if selected is None:
        return

    row = df[df["engine_id"] == selected].iloc[0]

    with col_info:
        status_color = STATUS_COLORS.get(str(row["pred_status"]), "#888")
        st.markdown(
            f"""
            <div style="padding:12px;border-left:4px solid {status_color};
                        border-radius:4px;background:rgba(128,128,128,0.05)">
                <b>Engine {row['engine_id']}</b><br>
                Predicted RUL : <b>{row['pred_RUL']} cycles</b><br>
                Status : <span style="color:{status_color};font-weight:600">
                    {row['pred_status']}</span>
                {"<br>⚠ <b>Status mismatch</b> vs ground truth" if row["mismatch"] else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Load full sensor history ───────────────────────────────────────────────
    full_path = PROCESSED_DIR / f"test_{subset}_full_scaled.csv"
    train_path = PROCESSED_DIR / f"train_{subset}_processed.csv"

    history_df = None
    for path in [full_path, train_path]:
        if path.exists():
            tmp = pd.read_csv(path)
            uid_col = tmp[tmp["unit"] == selected] if "unit" in tmp.columns else pd.DataFrame()
            if not uid_col.empty:
                history_df = uid_col.sort_values("cycle")
                break

    if history_df is None or history_df.empty:
        st.info(
            "No sensor history found for this engine in processed data.  "
            "Run the notebook pipeline first.",
            icon="ℹ️",
        )
        return

    # ── Sensor selector ───────────────────────────────────────────────────────
    available_sensors = [c for c in FEATURE_COLS if c in history_df.columns]
    selected_sensors = st.multiselect(
        "Sensors to plot",
        options=available_sensors,
        default=available_sensors[:4],
    )

    if not selected_sensors:
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = go.Figure()
    cycles = history_df["cycle"].values

    palette = [
        "#378ADD", "#1D9E75", "#BA7517", "#D85A30",
        "#7F77DD", "#D4537E", "#639922", "#E24B4A",
    ]

    for i, sensor in enumerate(selected_sensors):
        fig.add_trace(go.Scatter(
            x=cycles,
            y=history_df[sensor].values,
            mode="lines",
            name=sensor,
            line=dict(color=palette[i % len(palette)], width=1.5),
            opacity=0.85,
        ))

    fig.update_layout(
        xaxis_title="Cycle",
        yaxis_title="Scaled sensor value [−1, 1]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=50, l=50, r=20),
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Engine **{selected}** — {len(history_df)} recorded cycles · "
        f"values normalised to [−1, 1] by MinMaxScaler"
    )