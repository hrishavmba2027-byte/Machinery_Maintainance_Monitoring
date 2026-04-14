"""
MIOM Dashboard — config.py
==========================
Single source of truth for all constants, hyperparameters, and paths.
Edit here; all modules import from here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "Data" / "raw"
PROCESSED_DIR   = BASE_DIR / "Data" / "processed"
LIVE_DIR        = BASE_DIR / "Data" / "live"
SNAPSHOT_DIR    = BASE_DIR / "Data" / "snapshots"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"
LOG_PATH        = BASE_DIR / "logs" / "monitoring.log"

# Ensure directories exist at import time
for _d in [DATA_DIR, PROCESSED_DIR, LIVE_DIR, SNAPSHOT_DIR, CHECKPOINT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset columns ────────────────────────────────────────────────────────────
ALL_COLS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# 14 informative sensors (constant/near-constant sensors dropped)
GOOD_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

FEATURE_COLS = GOOD_SENSORS
META_COLS    = ["unit", "cycle"]

# ── Model hyperparameters ──────────────────────────────────────────────────────
SEQ_LEN         = 30          # Rolling window length
RUL_CEILING     = 125         # Piecewise-linear RUL cap
EWMA_SPAN       = 0           # 0 = no smoothing (paper-faithful)

MODEL_CONFIG = dict(
    n_features      = len(FEATURE_COLS),
    d_model         = 64,
    nhead           = 8,
    num_layers      = 3,
    dim_feedforward = 256,
    dropout         = 0.1,
    fc_dim          = 64,
    rul_ceiling     = float(RUL_CEILING),
)

# ── Urgency thresholds ─────────────────────────────────────────────────────────
CRITICAL_THRESHOLD  = 30   # ≤ 30 cycles → immediate shutdown
WARNING_THRESHOLD   = 70   # 31–70 cycles → schedule maintenance

STATUS_LABELS = {
    "CRITICAL": "🔴 CRITICAL",
    "WARNING":  "🟡 WARNING",
    "HEALTHY":  "🟢 HEALTHY",
}

STATUS_COLORS = {
    "🔴 CRITICAL": "#E24B4A",
    "🟡 WARNING":  "#EF9F27",
    "🟢 HEALTHY":  "#639922",
}

# ── Expected row counts (sanity checks) ───────────────────────────────────────
EXPECTED_TRAIN_ROWS = {
    "FD001": 17_731,
    "FD002": 48_819,
    "FD003": 21_820,
    "FD004": 57_522,
}

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# ── Dashboard / monitoring settings ───────────────────────────────────────────
ACTIVE_SUBSET           = "FD001"
REFRESH_INTERVAL_SEC    = 30      # Default auto-refresh cadence (seconds)
MAX_ALERTS_STORED       = 50      # Rolling alert buffer length