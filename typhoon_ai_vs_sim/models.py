from __future__ import annotations

import torch
from torch import nn


class MLPResidualRegressor(nn.Module):
    def __init__(self, window_size: int, input_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        flattened_dim = window_size * input_dim
        self.network = nn.Sequential(
            nn.Linear(flattened_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.reshape(inputs.shape[0], -1)
        return self.network(flattened).squeeze(-1)


class LSTMResidualRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(inputs)
        return self.head(encoded[:, -1, :]).squeeze(-1)


class TransformerResidualRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(inputs)
        projected = projected + self.position_embedding[:, : inputs.shape[1], :]
        encoded = self.encoder(projected)
        pooled = encoded[:, -1, :]
        return self.head(pooled).squeeze(-1)
