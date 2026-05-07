from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.models.abmil import AttentionMIL
from src.models.virchow2_encoder import Virchow2Config, Virchow2Encoder


@dataclass
class PatientBinaryModelConfig:
    tile_feature_dim: int = 2560
    slide_hidden_dim: int = 512
    patient_hidden_dim: int = 256
    dropout: float = 0.25
    encoder_batch_size: int = 64
    freeze_backbone: bool = True
    unfreeze_last_block: bool = False


class PatientBinaryModel(nn.Module):
    def __init__(self, config: PatientBinaryModelConfig | None = None):
        super().__init__()
        self.config = config or PatientBinaryModelConfig()
        self.encoder = Virchow2Encoder(
            Virchow2Config(
                freeze_backbone=self.config.freeze_backbone,
                unfreeze_last_block=self.config.unfreeze_last_block,
            )
        )
        self.slide_mil = AttentionMIL(
            input_dim=self.config.tile_feature_dim,
            hidden_dim=self.config.slide_hidden_dim,
            output_dim=self.config.slide_hidden_dim,
            dropout=self.config.dropout,
        )
        self.patient_mil = AttentionMIL(
            input_dim=self.config.slide_hidden_dim,
            hidden_dim=self.config.patient_hidden_dim,
            output_dim=self.config.patient_hidden_dim,
            dropout=self.config.dropout,
        )
        self.logit_head = nn.Linear(self.config.patient_hidden_dim, 1)

    def encode_slide(self, slide_tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if slide_tiles.shape[0] <= self.config.encoder_batch_size:
            tile_features = self.encoder(slide_tiles)
        else:
            tile_features = torch.cat(
                [
                    self.encoder(chunk)
                    for chunk in torch.split(slide_tiles, self.config.encoder_batch_size, dim=0)
                ],
                dim=0,
            )
        return self.slide_mil(tile_features)

    def forward_patient(self, slide_tile_bags: list[torch.Tensor]) -> dict[str, Any]:
        slide_embeddings = []
        slide_attentions = []
        for slide_tiles in slide_tile_bags:
            embedding, tile_attention = self.encode_slide(slide_tiles)
            slide_embeddings.append(embedding)
            slide_attentions.append(tile_attention)

        slide_embeddings_tensor = torch.stack(slide_embeddings, dim=0)
        patient_embedding, slide_attention = self.patient_mil(slide_embeddings_tensor)
        logit = self.logit_head(patient_embedding).squeeze(-1)
        return {
            "logit": logit,
            "patient_embedding": patient_embedding,
            "slide_embeddings": slide_embeddings_tensor,
            "slide_attention": slide_attention,
            "tile_attentions": slide_attentions,
        }

    def forward(self, bags: list[list[torch.Tensor]]) -> dict[str, Any]:
        outputs = [self.forward_patient(patient_bag) for patient_bag in bags]
        return {
            "logits": torch.stack([item["logit"] for item in outputs], dim=0),
            "patient_embeddings": torch.stack([item["patient_embedding"] for item in outputs], dim=0),
            "per_patient": outputs,
        }
