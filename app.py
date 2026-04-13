"""
MIOM Dashboard — app.py
=======================
Main Streamlit entry point.  Continuously polls the data/live/ folder
for new sensor CSVs, reruns inference, and dynamically updates every panel.

Run:
    streamlit run app.py
"""

import time
import logging
from pathlib import Path

import streamlit as st

from config import REFRESH_INTERVAL_SEC, LOG_PATH, ACTIVE_SUBSET
from monitoring.watcher import DataWatcher
from monitoring.change_detector import ChangeDetector
from core.inference import load_pipeline, run_inference
from dashboard.fleet_overview import render_fleet_overview
from dashboard.rul_chart import render_rul_chart
from dashboard.triage_table import render_triage_table
from dashboard.engine_detail import render_engine_detail
from dashboard.alerts import render_alerts_panel

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIOM — Fleet Health Monitor",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css(path: Path) -> None:
    if path.exists():
        st.markdown(f"<style>{path.read_text()}</style>", unsafe_allow_html=True)

_load_css(Path(__file__).parent / "dashboard" / "style.css")

# ── Session-state initialisation ─────────────────────────────────────────────
if "selected_subset" not in st.session_state:
    st.session_state.selected_subset = ACTIVE_SUBSET
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None          # (model, scaler, cfg)
if "triage_df" not in st.session_state:
    st.session_state.triage_df = None
if "prev_triage_df" not in st.session_state:
    st.session_state.prev_triage_df = None
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0.0
if "watcher" not in st.session_state:
    st.session_state.watcher = DataWatcher()
if "change_detector" not in st.session_state:
    st.session_state.change_detector = ChangeDetector()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    subset_options = ["FD001", "FD002", "FD003", "FD004"]
    default_index = subset_options.index(st.session_state.selected_subset)
    selected_subset = st.selectbox(
        "Dataset subset",
        subset_options,
        index=default_index,
    )

    if selected_subset != st.session_state.selected_subset:
        st.session_state.selected_subset = selected_subset
        st.session_state.triage_df = None
        st.session_state.prev_triage_df = None
        st.session_state.alerts = []
        st.session_state.last_refresh = 0.0
        st.session_state.watcher = DataWatcher()
        st.session_state.change_detector = ChangeDetector()

    refresh_interval = st.slider(
        "Auto-refresh (seconds)", min_value=5, max_value=120,
        value=REFRESH_INTERVAL_SEC, step=5,
    )

    watch_live = st.toggle("Watch live/ folder", value=True)

    st.divider()

    if st.button("🔄 Force refresh", use_container_width=True):
        st.session_state.last_refresh = 0.0

    if st.button("🗑️ Clear alerts", use_container_width=True):
        st.session_state.alerts = []

    st.divider()
    st.caption(f"Subset: **{selected_subset}**")
    if st.session_state.triage_df is not None:
        n = len(st.session_state.triage_df)
        st.caption(f"Engines monitored: **{n}**")

# ── Load pipeline (cached by subset) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model checkpoint…")
def get_pipeline(subset: str):
    return load_pipeline(subset)

pipeline = get_pipeline(selected_subset)

# ── Check if refresh is needed ────────────────────────────────────────────────
now = time.time()
data_changed = watch_live and st.session_state.watcher.has_changed()
time_elapsed = (now - st.session_state.last_refresh) >= refresh_interval

needs_refresh = data_changed or time_elapsed or st.session_state.triage_df is None

if needs_refresh:
    log.info("Refresh triggered (data_changed=%s, time_elapsed=%s)", data_changed, time_elapsed)
    with st.spinner("Running inference on latest data…"):
        triage_df = run_inference(pipeline, selected_subset)

    if triage_df is not None:
        # Detect changes vs previous snapshot
        new_alerts = st.session_state.change_detector.detect(
            st.session_state.triage_df, triage_df
        )
        st.session_state.prev_triage_df = st.session_state.triage_df
        st.session_state.triage_df = triage_df
        st.session_state.alerts = (new_alerts + st.session_state.alerts)[:50]

    st.session_state.last_refresh = now

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_time = st.columns([5, 1])
with col_title:
    st.markdown("# 🛩️ MIOM — Fleet Health Monitor")
    st.caption(f"Subset **{selected_subset}** · refreshes every **{refresh_interval}s**")
with col_time:
    st.markdown(f"<div style='text-align:right;padding-top:20px;color:var(--text-color);opacity:0.5'>"
                f"⏱ {time.strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
if st.session_state.triage_df is None:
    st.info(
        "No data found yet.  "
        "Drop sensor CSV files into `data/live/` **or** run the notebook pipeline first "
        "so that `data/processed/` is populated.",
        icon="ℹ️",
    )
else:
    df = st.session_state.triage_df

    # ── Row 1: KPI overview ───────────────────────────────────────────────────
    render_fleet_overview(df)

    st.divider()

    # ── Row 2: RUL chart + Alerts ─────────────────────────────────────────────
    chart_col, alert_col = st.columns([3, 1])
    with chart_col:
        render_rul_chart(df)
    with alert_col:
        render_alerts_panel(st.session_state.alerts)

    st.divider()

    # ── Row 3: Triage table ───────────────────────────────────────────────────
    render_triage_table(df)

    st.divider()

    # ── Row 4: Engine drill-down ──────────────────────────────────────────────
    render_engine_detail(df, selected_subset)

# ── Auto-rerun ────────────────────────────────────────────────────────────────
time.sleep(0.5)
st.rerun()