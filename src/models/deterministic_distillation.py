from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DeterministicDistillationConfig:
    frozen_dim: int = 256
    ffpe_dim: int = 256
    mapper_hidden_dim: int = 256
    predictor_hidden_dim: int = 256
    dropout: float = 0.25


class DeterministicDistillationModel(nn.Module):
    def __init__(self, config: DeterministicDistillationConfig | None = None):
        super().__init__()
        self.config = config or DeterministicDistillationConfig()
        self.mapper = nn.Sequential(
            nn.LayerNorm(self.config.frozen_dim),
            nn.Linear(self.config.frozen_dim, self.config.mapper_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.mapper_hidden_dim, self.config.ffpe_dim),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(self.config.frozen_dim + self.config.ffpe_dim),
            nn.Linear(self.config.frozen_dim + self.config.ffpe_dim, self.config.predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.predictor_hidden_dim, 1),
        )

    def forward(self, frozen_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        ffpe_hat = self.mapper(frozen_embeddings)
        logits = self.predictor(torch.cat([frozen_embeddings, ffpe_hat], dim=-1)).squeeze(-1)
        return {
            "predicted_ffpe_embedding": ffpe_hat,
            "logits": logits,
        }
