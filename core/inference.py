"""
MIOM Dashboard — core/inference.py
====================================
Loads a saved checkpoint and runs inference to produce the fleet triage DataFrame.

Two data sources (tried in order):
  1. data/live/   — live CSV drops (new sensor readings streamed in)
  2. data/processed/ — pre-built CSVs from the notebook pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR, DATA_DIR, LIVE_DIR, PROCESSED_DIR,
    CRITICAL_THRESHOLD, WARNING_THRESHOLD, RUL_CEILING,
    FEATURE_COLS, SEQ_LEN, MODEL_CONFIG, EWMA_SPAN,
)
from core.model import TransformerRUL
from core.dataset import CMAPSSTestDataset
from core.preprocessing import (
    full_pipeline, select_features, ewm_smooth, apply_scaler,
    build_test_eval,
)

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Urgency classification ─────────────────────────────────────────────────────

def classify_urgency(rul: float) -> str:
    if rul <= CRITICAL_THRESHOLD:
        return "🔴 CRITICAL"
    elif rul <= WARNING_THRESHOLD:
        return "🟡 WARNING"
    else:
        return "🟢 HEALTHY"


# ── Scoring ───────────────────────────────────────────────────────────────────

def cmapss_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    d = y_pred - y_true
    s = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,
        np.exp( d / 10.0) - 1,
    )
    return float(s.sum())


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ── Load pipeline (model + scaler) ────────────────────────────────────────────

def _safe_load_checkpoint(ckpt_path: Path) -> Optional[dict]:
    if not ckpt_path.exists():
        return None
    if ckpt_path.stat().st_size == 0:
        log.warning("Checkpoint file is empty: %s", ckpt_path)
        return None
    try:
        return torch.load(ckpt_path, map_location=DEVICE)
    except Exception as exc:
        log.warning("Failed to load checkpoint %s: %s", ckpt_path, exc)
        return None


def _infer_cfg_from_state_dict(state_dict: dict, base_cfg: dict) -> dict:
    cfg = base_cfg.copy()
    try:
        if "input_proj.0.weight" in state_dict:
            cfg["n_features"] = state_dict["input_proj.0.weight"].shape[1]
            cfg["d_model"] = state_dict["input_proj.0.weight"].shape[0]
        if "head.1.weight" in state_dict:
            cfg["fc_dim"] = state_dict["head.1.weight"].shape[0]
        layer_idxs = {int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.layers.")}
        if layer_idxs:
            cfg["num_layers"] = len(layer_idxs)
        if "encoder.layers.0.linear1.weight" in state_dict:
            cfg["dim_feedforward"] = state_dict["encoder.layers.0.linear1.weight"].shape[0]
    except Exception as exc:
        log.warning("Failed to infer model config from state_dict: %s", exc)
    return cfg


def load_pipeline(subset: str) -> Optional[Tuple]:
    """
    Returns (model, scaler, seq_len, model_config) loaded from checkpoint.
    Returns None if no valid checkpoint exists yet.
    """
    primary_ckpt = CHECKPOINT_DIR / f"best_transformer_{subset}.pt"
    fallback_ckpt = CHECKPOINT_DIR / f"transformer_{subset}.pt"
    scaler_path = CHECKPOINT_DIR / f"scaler_{subset}.pkl"

    ckpt = _safe_load_checkpoint(primary_ckpt)
    used_ckpt_path = primary_ckpt
    if ckpt is None:
        log.info("Primary checkpoint unavailable, trying fallback: %s", fallback_ckpt)
        ckpt = _safe_load_checkpoint(fallback_ckpt)
        used_ckpt_path = fallback_ckpt

    if ckpt is None:
        log.warning("No valid checkpoint available for %s — inference unavailable.", subset)
        return None

    cfg = ckpt.get("model_config", MODEL_CONFIG)
    cfg = _infer_cfg_from_state_dict(ckpt["model_state_dict"], cfg)
    model = TransformerRUL(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    log.info("Loaded checkpoint: %s", used_ckpt_path)

    if scaler_path.exists() and scaler_path.stat().st_size > 0:
        try:
            scaler = joblib.load(scaler_path)
            log.info("Loaded scaler: %s", scaler_path)
        except Exception as exc:
            log.warning("Failed to load scaler %s: %s", scaler_path, exc)
            scaler = None
    else:
        log.warning("No valid scaler found at %s — live input will not be scaled.", scaler_path)
        scaler = None

    seq_len = ckpt.get("seq_len", SEQ_LEN)
    return model, scaler, seq_len, cfg


# ── Live data handling ────────────────────────────────────────────────────────

def _load_live_data(subset: str, scaler) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Reads all CSVs from data/live/ that match the subset.
    CSVs must have the same columns as the processed test_eval file.

    Returns (test_eval_df, full_scaled_df) or None.
    """
    live_files = sorted(LIVE_DIR.glob(f"*{subset}*.csv"))
    if not live_files:
        return None

    frames = [pd.read_csv(f) for f in live_files]
    combined = pd.concat(frames, ignore_index=True)

    # Must contain at minimum: unit, cycle, + FEATURE_COLS
    required = {"unit", "cycle"} | set(FEATURE_COLS)
    missing  = required - set(combined.columns)
    if missing:
        log.warning("Live CSV missing columns: %s", missing)
        return None

    # Apply scaler if available
    if scaler is not None:
        combined = select_features(combined)
        combined = apply_scaler(combined, scaler)

    # Build test eval (last cycle per engine; no true RUL → set to -1)
    last_cycles = (
        combined.sort_values("cycle")
                .groupby("unit")
                .last()
                .reset_index()
    )
    last_cycles["RUL"] = -1.0     # unknown ground truth in live mode

    return last_cycles, combined


# ── Main inference entry point ────────────────────────────────────────────────

def run_inference(pipeline, subset: str) -> Optional[pd.DataFrame]:
    """
    Runs model inference and returns a triage DataFrame with columns:
        engine_id, true_RUL, pred_RUL, error, true_status, pred_status, mismatch

    Returns None if neither checkpoint nor processed data is available.
    """
    if pipeline is None:
        # Fallback: try to load processed CSVs directly (notebook already ran)
        return _load_from_processed(subset)

    model, scaler, seq_len, _ = pipeline

    # ── Try live data first ───────────────────────────────────────────────────
    live = _load_live_data(subset, scaler)
    if live is not None:
        test_eval, full_scaled = live
        has_true_rul = (test_eval["RUL"] != -1).any()
    else:
        # Fall back to processed data
        test_eval_path   = PROCESSED_DIR / f"test_{subset}_eval.csv"
        full_scaled_path = PROCESSED_DIR / f"test_{subset}_full_scaled.csv"

        if not test_eval_path.exists():
            log.warning("No processed data found for %s", subset)
            return _load_from_processed(subset)

        test_eval   = pd.read_csv(test_eval_path)
        full_scaled = pd.read_csv(full_scaled_path) if full_scaled_path.exists() else test_eval
        has_true_rul = True

    # ── Build dataset & dataloader ────────────────────────────────────────────
    ds     = CMAPSSTestDataset(test_eval, FEATURE_COLS, seq_len, full_test_df=full_scaled)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    # ── Model inference ───────────────────────────────────────────────────────
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            p = model(X_batch.to(DEVICE)).cpu().numpy()
            preds.extend(p)
            trues.extend(y_batch.numpy())

    y_pred = np.clip(np.round(np.array(preds)).astype(int), 0, RUL_CEILING)
    y_true = np.array(trues)

    unit_ids = sorted(test_eval["unit"].unique())

    triage_df = pd.DataFrame({
        "engine_id"  : unit_ids,
        "true_RUL"   : y_true.astype(int) if has_true_rul else [-1] * len(unit_ids),
        "pred_RUL"   : y_pred,
        "error"      : np.round(y_pred - y_true, 1) if has_true_rul else [None] * len(unit_ids),
    })

    triage_df["true_status"] = triage_df["true_RUL"].apply(
        lambda r: classify_urgency(r) if r != -1 else "N/A"
    )
    triage_df["pred_status"] = triage_df["pred_RUL"].apply(classify_urgency)
    triage_df["mismatch"] = (
        (triage_df["true_status"] != triage_df["pred_status"])
        & (triage_df["true_status"] != "N/A")
    )

    # Sort by predicted RUL ascending (most urgent first)
    triage_df = triage_df.sort_values("pred_RUL").reset_index(drop=True)

    # Compute metrics if ground truth available
    if has_true_rul:
        mask = triage_df["true_RUL"] != -1
        if mask.any():
            yt = triage_df.loc[mask, "true_RUL"].values
            yp = triage_df.loc[mask, "pred_RUL"].values
            triage_df.attrs["test_rmse"]  = rmse(yp, yt)
            triage_df.attrs["test_score"] = cmapss_score(yp, yt)

    log.info("Inference complete — %d engines, subset=%s", len(triage_df), subset)
    return triage_df


def _load_from_processed(subset: str) -> Optional[pd.DataFrame]:
    """
    Last resort: construct a minimal triage DataFrame from pre-saved
    test_eval CSV (notebook cell 12 output) without a model.
    Shows only ground-truth RUL; pred_RUL equals true_RUL.
    """
    path = PROCESSED_DIR / f"test_{subset}_eval.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "RUL" not in df.columns or "unit" not in df.columns:
        return None

    unit_ids = sorted(df["unit"].unique())
    rul_vals = df.sort_values("unit").set_index("unit")["RUL"]

    y_true = np.array([rul_vals[u] for u in unit_ids])

    triage_df = pd.DataFrame({
        "engine_id"  : unit_ids,
        "true_RUL"   : y_true.astype(int),
        "pred_RUL"   : y_true.astype(int),   # no model → identity
        "error"      : np.zeros(len(unit_ids)),
        "true_status": [classify_urgency(r) for r in y_true],
        "pred_status": [classify_urgency(r) for r in y_true],
        "mismatch"   : [False] * len(unit_ids),
    }).sort_values("pred_RUL").reset_index(drop=True)

    log.info("Loaded %d engines from processed CSV (no model).", len(triage_df))
    return triage_df