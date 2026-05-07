from __future__ import annotations

import torch
from torch import nn


class AttentionMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        output_dim = output_dim or input_dim
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(features).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(weights.unsqueeze(-1) * features, dim=0)
        return self.projection(pooled), weights
