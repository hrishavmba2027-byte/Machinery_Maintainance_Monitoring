"""
MIOM Dashboard — core/model.py
==============================
Exact TransformerRUL architecture from the notebook.
Import this wherever you need the model class (training or inference).
"""

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerRUL(nn.Module):
    """
    Transformer-based Remaining Useful Life predictor.

    Input  : (batch, seq_len, n_features)
    Output : (batch,)  — scalar RUL prediction per sample
    """

    def __init__(
        self,
        n_features: int,
        d_model: int          = 64,
        nhead: int            = 8,
        num_layers: int       = 3,
        dim_feedforward: int  = 256,
        dropout: float        = 0.1,
        fc_dim: int           = 64,
        rul_ceiling: float    = 125.0,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.rul_ceiling = rul_ceiling

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model          = d_model,
            nhead            = nhead,
            dim_feedforward  = dim_feedforward,
            dropout          = dropout,
            activation       = "gelu",
            batch_first      = True,
            norm_first       = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)          # (B, T, d_model)
        x = x.mean(dim=1)            # Global average pool over time
        out = self.head(x).squeeze(-1)  # (B,)
        return out.clamp(0, self.rul_ceiling)