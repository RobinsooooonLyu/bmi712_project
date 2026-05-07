from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm


class DINOProjectionHead(nn.Module):
    """Projection head used only when distill_loss='dino_ce'.

    Mirrors the standard DINO head: 3-layer MLP with GELU, L2 normalization on
    the bottleneck, then a weight-normalized linear layer producing
    `num_prototypes` logits. Applied on top of the 256-d patient embedding.
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
        num_prototypes: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        last = nn.Linear(bottleneck_dim, num_prototypes, bias=False)
        self.last_layer = weight_norm(last)
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class ContrastiveProjectionHead(nn.Module):
    """SimCLR-style 2-layer MLP projection head for contrastive alignment.

    Maps the 256-d patient embedding to a `proj_dim`-d unit vector. The
    L2-normalized output is what the InfoNCE loss operates on, so cosine
    similarity reduces to a dot product.
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1, p=2)
