"""
MIOM Dashboard — dashboard/alerts.py
======================================
Live rolling alert feed rendered in the sidebar column.
"""

from __future__ import annotations

import time
from typing import Dict, List

import streamlit as st


_SEVERITY_ICON = {
    "high"  : "🔴",
    "medium": "🟡",
    "low"   : "🟢",
}

_TYPE_LABEL = {
    "degradation" : "Status degraded",
    "improvement" : "Status improved",
    "new_critical": "New CRITICAL engine",
    "rul_spike"   : "Unexpected RUL change",
}


def render_alerts_panel(alerts: List[Dict]) -> None:
    st.markdown("### 🔔 Live alerts")

    if not alerts:
        st.info("No alerts yet.", icon="ℹ️")
        return

    for alert in alerts[:20]:
        icon      = _SEVERITY_ICON.get(alert.get("severity", "low"), "⚪")
        label     = _TYPE_LABEL.get(alert.get("type", ""), alert.get("type", ""))
        eid       = alert.get("engine_id", "?")
        old_s     = alert.get("old_status", "")
        new_s     = alert.get("new_status", "")
        old_rul   = alert.get("old_rul", "?")
        new_rul   = alert.get("new_rul", "?")
        ts        = alert.get("ts", 0.0)
        ts_str    = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""

        detail = f"RUL {old_rul} → {new_rul}" if old_rul != -1 else f"RUL {new_rul}"
        status_change = (
            f"{old_s} → {new_s}" if old_s and old_s != "N/A" else new_s
        )

        severity = alert.get("severity", "low")
        if severity == "high":
            container = st.error
        elif severity == "medium":
            container = st.warning
        else:
            container = st.success

        container(
            f"{icon} **Engine {eid}** — {label}  \n"
            f"{status_change} · {detail}  \n"
            f"⏱ {ts_str}",
            icon=None,
        )