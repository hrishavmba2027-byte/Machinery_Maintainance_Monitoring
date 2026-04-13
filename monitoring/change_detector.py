"""
MIOM Dashboard — monitoring/change_detector.py
================================================
Compares two consecutive triage DataFrames and produces
structured alert dicts for the alerts panel.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

STATUS_PRIORITY = {
    "🔴 CRITICAL": 3,
    "🟡 WARNING":  2,
    "🟢 HEALTHY":  1,
    "N/A":         0,
}


class ChangeDetector:
    """
    Stateless differ: call .detect(old_df, new_df) to get a list
    of alert dicts describing what changed between snapshots.

    Alert dict schema:
        {
            "ts"        : float (epoch),
            "engine_id" : int or str,
            "type"      : "degradation" | "improvement" | "new_critical" | "rul_spike",
            "old_status": str,
            "new_status": str,
            "old_rul"   : int,
            "new_rul"   : int,
            "delta_rul" : int,
            "severity"  : "high" | "medium" | "low",
        }
    """

    def detect(
        self,
        old_df: Optional[pd.DataFrame],
        new_df: pd.DataFrame,
    ) -> List[Dict]:
        if old_df is None or old_df.empty:
            # First run — flag all CRITICAL engines
            return self._first_run_alerts(new_df)

        alerts: List[Dict] = []
        now = time.time()

        old_idx = old_df.set_index("engine_id")
        new_idx = new_df.set_index("engine_id")

        # Engines present in both snapshots
        common = old_idx.index.intersection(new_idx.index)

        for eid in common:
            old_row = old_idx.loc[eid]
            new_row = new_idx.loc[eid]

            old_rul    = int(old_row["pred_RUL"])
            new_rul    = int(new_row["pred_RUL"])
            old_status = str(old_row["pred_status"])
            new_status = str(new_row["pred_status"])
            delta      = new_rul - old_rul

            old_p = STATUS_PRIORITY.get(old_status, 0)
            new_p = STATUS_PRIORITY.get(new_status, 0)

            if new_p > old_p:
                # Status worsened
                alerts.append({
                    "ts"        : now,
                    "engine_id" : eid,
                    "type"      : "degradation",
                    "old_status": old_status,
                    "new_status": new_status,
                    "old_rul"   : old_rul,
                    "new_rul"   : new_rul,
                    "delta_rul" : delta,
                    "severity"  : "high" if new_status == "🔴 CRITICAL" else "medium",
                })
                log.warning(
                    "Engine %s degraded: %s → %s (RUL %d → %d)",
                    eid, old_status, new_status, old_rul, new_rul,
                )

            elif new_p < old_p:
                # Status improved
                alerts.append({
                    "ts"        : now,
                    "engine_id" : eid,
                    "type"      : "improvement",
                    "old_status": old_status,
                    "new_status": new_status,
                    "old_rul"   : old_rul,
                    "new_rul"   : new_rul,
                    "delta_rul" : delta,
                    "severity"  : "low",
                })

            elif abs(delta) >= 15 and new_status == old_status:
                # Same status band but large RUL jump (anomaly)
                alerts.append({
                    "ts"        : now,
                    "engine_id" : eid,
                    "type"      : "rul_spike",
                    "old_status": old_status,
                    "new_status": new_status,
                    "old_rul"   : old_rul,
                    "new_rul"   : new_rul,
                    "delta_rul" : delta,
                    "severity"  : "medium",
                })

        # Newly appeared engines
        new_engines = new_idx.index.difference(old_idx.index)
        for eid in new_engines:
            row = new_idx.loc[eid]
            if row["pred_status"] == "🔴 CRITICAL":
                alerts.append({
                    "ts"        : now,
                    "engine_id" : eid,
                    "type"      : "new_critical",
                    "old_status": "N/A",
                    "new_status": str(row["pred_status"]),
                    "old_rul"   : -1,
                    "new_rul"   : int(row["pred_RUL"]),
                    "delta_rul" : 0,
                    "severity"  : "high",
                })

        return alerts

    @staticmethod
    def _first_run_alerts(df: pd.DataFrame) -> List[Dict]:
        """Generate HIGH alerts for all CRITICAL engines on first load."""
        now    = time.time()
        alerts = []
        for _, row in df.iterrows():
            if row["pred_status"] == "🔴 CRITICAL":
                alerts.append({
                    "ts"        : now,
                    "engine_id" : row["engine_id"],
                    "type"      : "new_critical",
                    "old_status": "N/A",
                    "new_status": str(row["pred_status"]),
                    "old_rul"   : -1,
                    "new_rul"   : int(row["pred_RUL"]),
                    "delta_rul" : 0,
                    "severity"  : "high",
                })
        return alerts