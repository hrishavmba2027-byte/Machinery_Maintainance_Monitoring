"""
MIOM Dashboard — core/dataset.py
=================================
Dataset classes and test-set construction helpers (from notebook cells 12–13).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class CMAPSSTrainDataset(Dataset):
    """
    Rolling-window dataset for training.
    Each cycle of each engine becomes one sample.
    Engines with fewer than seq_len cycles are left-padded with zeros.
    """

    def __init__(
        self,
        df,
        feature_cols: List[str],
        seq_len: int,
        rul_col: str = "RUL",
    ):
        self.seq_len = seq_len
        self.n_feat  = len(feature_cols)

        windows, targets = [], []

        for _, group in df.groupby("unit"):
            group  = group.sort_values("cycle").reset_index(drop=True)
            feats  = group[feature_cols].values.astype(np.float32)   # (T, F)
            ruls   = group[rul_col].values.astype(np.float32)        # (T,)
            T      = len(group)

            for t in range(T):
                end   = t + 1
                start = max(0, end - seq_len)
                window = feats[start:end]           # (≤seq_len, F)

                # Left-pad with zeros if shorter than seq_len
                if len(window) < seq_len:
                    pad    = np.zeros((seq_len - len(window), self.n_feat), dtype=np.float32)
                    window = np.vstack([pad, window])

                windows.append(window)
                targets.append(ruls[t])

        self.X = torch.tensor(np.stack(windows), dtype=torch.float32)
        self.y = torch.tensor(np.array(targets), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CMAPSSTestDataset(Dataset):
    """
    Test-time dataset.
    Each engine contributes ONE window: the last seq_len cycles (or all if shorter).
    Ground-truth RUL comes from RUL_FDxxx.txt via rul_lookup.
    """

    def __init__(
        self,
        test_eval_df,
        feature_cols: List[str],
        seq_len: int,
        full_test_df=None,   # full scaled test set (for window extraction)
    ):
        self.seq_len = seq_len
        self.n_feat  = len(feature_cols)

        windows, targets = [], []
        src = full_test_df if full_test_df is not None else test_eval_df

        for _, row in test_eval_df.iterrows():
            uid  = row["unit"]
            true_rul = float(row["RUL"])

            # Get full history for this engine from scaled data
            engine_data = (
                src[src["unit"] == uid]
                .sort_values("cycle")[feature_cols]
                .values.astype(np.float32)
            )
            T = len(engine_data)

            if T >= seq_len:
                window = engine_data[-seq_len:]
            else:
                pad    = np.zeros((seq_len - T, self.n_feat), dtype=np.float32)
                window = np.vstack([pad, engine_data])

            windows.append(window)
            targets.append(true_rul)

        self.X = torch.tensor(np.stack(windows), dtype=torch.float32)
        self.y = torch.tensor(np.array(targets), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]