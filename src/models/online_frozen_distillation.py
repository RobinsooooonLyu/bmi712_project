from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.models.patient_binary_model import PatientBinaryModel, PatientBinaryModelConfig


@dataclass
class OnlineFrozenDistillationConfig:
    frozen_dim: int = 256
    ffpe_dim: int = 256
    mapper_hidden_dim: int = 256
    predictor_hidden_dim: int = 256
    dropout: float = 0.25


class OnlineFrozenToFFPEDistillationModel(nn.Module):
    """Frozen WSI encoder plus a mapper into the FFPE high-risk latent space."""

    def __init__(
        self,
        frozen_model_config: PatientBinaryModelConfig | None = None,
        distill_config: OnlineFrozenDistillationConfig | None = None,
    ):
        super().__init__()
        self.frozen_model = PatientBinaryModel(frozen_model_config)
        self.config = distill_config or OnlineFrozenDistillationConfig()
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

    def forward(self, frozen_bags: list[list[torch.Tensor]]) -> dict[str, Any]:
        frozen_outputs = self.frozen_model(frozen_bags)
        frozen_embeddings = frozen_outputs["patient_embeddings"]
        predicted_ffpe_embeddings = self.mapper(frozen_embeddings)
        logits = self.predictor(
            torch.cat([frozen_embeddings, predicted_ffpe_embeddings], dim=-1)
        ).squeeze(-1)
        return {
            "logits": logits,
            "frozen_embeddings": frozen_embeddings,
            "predicted_ffpe_embeddings": predicted_ffpe_embeddings,
            "frozen_outputs": frozen_outputs,
        }
