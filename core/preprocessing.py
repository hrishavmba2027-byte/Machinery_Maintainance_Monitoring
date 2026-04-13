"""
MIOM Dashboard — core/preprocessing.py
========================================
Data loading, feature selection, scaling, and RUL labelling.
Mirrors the exact notebook pipeline (cells 3–12).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import ALL_COLS, META_COLS, FEATURE_COLS, RUL_CEILING


# ── 1. Raw data loading ────────────────────────────────────────────────────────

def load_raw_subset(
    data_dir: Path,
    subset: str,
    unit_prefix: Optional[str] = None,
    unit_pad: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Loads train_{subset}.txt, test_{subset}.txt, RUL_{subset}.txt.

    If unit_prefix is provided, engine IDs are namespaced as
    {unit_prefix}_{unit:0{unit_pad}d} so multiple subsets can be concatenated.

    Returns (train_df, test_df, rul_array).
    """
    train_path = data_dir / f"train_{subset}.txt"
    test_path  = data_dir / f"test_{subset}.txt"
    rul_path   = data_dir / f"RUL_{subset}.txt"

    for p in [train_path, test_path, rul_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLS)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=ALL_COLS)
    rul_arr  = np.loadtxt(rul_path)

    if unit_prefix:
        train_df["unit"] = train_df["unit"].apply(
            lambda u: f"{unit_prefix}_{u:0{unit_pad}d}"
        )
        test_df["unit"] = test_df["unit"].apply(
            lambda u: f"{unit_prefix}_{u:0{unit_pad}d}"
        )

    return train_df, test_df, rul_arr


# ── 2. Feature selection ───────────────────────────────────────────────────────

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only META_COLS + FEATURE_COLS."""
    return df[META_COLS + FEATURE_COLS].copy()


# ── 3. Optional EWM smoothing ─────────────────────────────────────────────────

def ewm_smooth(
    df: pd.DataFrame,
    sensor_cols: List[str],
    span: int,
) -> pd.DataFrame:
    """
    Exponential weighted mean smoothing per engine.
    span=0 means no smoothing (paper-faithful).
    """
    if span == 0:
        return df.copy()
    df = df.copy()
    for col in sensor_cols:
        df[col] = (
            df.groupby("unit")[col]
              .transform(lambda x: x.ewm(span=span, adjust=False).mean())
        )
    return df


# ── 4. MinMax scaling [-1, 1] ─────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df[FEATURE_COLS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    out = df.copy()
    out[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return out


# ── 5. RUL labelling ──────────────────────────────────────────────────────────

def add_train_rul(df: pd.DataFrame, ceiling: int = RUL_CEILING) -> pd.DataFrame:
    """
    Piecewise-linear RUL label:
        RUL(t) = min(max_cycle_of_engine − t,  ceiling)
    """
    df = df.copy()
    max_cycle  = df.groupby("unit")["cycle"].transform("max")
    df["RUL"]  = (max_cycle - df["cycle"]).clip(upper=ceiling)
    return df


# ── 6. Test-set evaluation frame ──────────────────────────────────────────────

def build_test_eval(
    test_df: pd.DataFrame,
    rul_lookup: dict,
    ceiling: int = RUL_CEILING,
) -> pd.DataFrame:
    """
    One row per test engine: last recorded cycle + ground-truth RUL.

    rul_lookup : {unit_id → true_RUL_from_file}
    """
    last_cycles = (
        test_df.sort_values("cycle")
               .groupby("unit")
               .last()
               .reset_index()
    )
    last_cycles["RUL"] = (
        last_cycles["unit"]
        .map(rul_lookup)
        .clip(upper=ceiling)
        .astype(float)
    )
    return last_cycles


# ── 7. End-to-end convenience helper ─────────────────────────────────────────

def full_pipeline(
    data_dir: Path,
    subset: str,
    ewma_span: int = 0,
    rul_ceiling: int = RUL_CEILING,
):
    """
    Runs the complete preprocessing pipeline for one subset.

    Returns
    -------
    train_final   : scaled + RUL-labelled training DataFrame
    test_eval     : one row per test engine (scaled, true RUL)
    test_scaled   : full scaled test DataFrame (for window extraction)
    scaler        : fitted MinMaxScaler
    rul_lookup    : {unit_id → true RUL}
    """
    train_raw, test_raw, rul_arr = load_raw_subset(data_dir, subset)

    # Build RUL lookup
    unit_ids   = sorted(test_raw["unit"].unique())
    rul_lookup = {uid: int(r) for uid, r in zip(unit_ids, rul_arr)}

    # Feature selection
    train_fs = select_features(train_raw)
    test_fs  = select_features(test_raw)

    # Optional smoothing
    train_sm = ewm_smooth(train_fs, FEATURE_COLS, ewma_span)
    test_sm  = ewm_smooth(test_fs,  FEATURE_COLS, ewma_span)

    # Scale
    scaler      = fit_scaler(train_sm)
    train_scaled = apply_scaler(train_sm,  scaler)
    test_scaled  = apply_scaler(test_sm,   scaler)

    # Add RUL to training data
    train_final = add_train_rul(train_scaled, ceiling=rul_ceiling)

    # Build test eval frame
    test_eval = build_test_eval(test_scaled, rul_lookup, ceiling=rul_ceiling)

    return train_final, test_eval, test_scaled, scaler, rul_lookup