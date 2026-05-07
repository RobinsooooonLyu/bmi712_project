"""Minimal LoRA implementation for the Virchow2 ViT backbone.

LoRA (Low-Rank Adaptation, Hu et al. 2021) freezes the original weight matrix
W of a Linear layer and adds a small trainable update ``B @ A`` of rank ``r``::

    y = W x + (alpha / r) * B (A x)

where ``A`` is ``(r, in_features)`` and ``B`` is ``(out_features, r)``.
``A`` is randomly initialised; ``B`` is zero-initialised so the layer behaves
identically to the frozen base at step 0. Trainable parameter count is
``r * (in_features + out_features)`` per adapted layer — orders of magnitude
fewer than fine-tuning the full Linear.

In this codebase the backbone (Virchow2 / a timm ViT) is normally fully frozen
in the baseline (`freeze_backbone=True`). This module lets us instead unfreeze
the backbone *cheaply* by adapting only a few attention projections in the
last few transformer blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class LoRAConfig:
    # Rank of the low-rank update. Smaller = fewer trainable params, more
    # conservative. 4 is a typical lower bound that still helps.
    rank: int = 4
    # LoRA scale factor; effective scale applied to the low-rank update is
    # ``alpha / rank``. With rank=4, alpha=8 gives a 2x scaling — a common
    # conservative default.
    alpha: float = 8.0
    # Dropout on the LoRA input branch. 0.0 keeps things simple/conservative.
    dropout: float = 0.0
    # How many of the *last* transformer blocks to adapt. Earlier blocks
    # encode generic low-level features; later blocks specialise. Adapting
    # only the last 1-2 blocks is the most compute-efficient option.
    last_n_blocks: int = 2
    # Which Linear submodules inside each adapted block to wrap with LoRA.
    # Paths are relative to a single timm ViT block. ``attn.qkv`` is the
    # fused q/k/v projection — adapting it is the highest-leverage single
    # target. Add ``attn.proj`` to also adapt the attention output projection.
    target_modules: list[str] = field(default_factory=lambda: ["attn.qkv"])


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with an additive low-rank update.

    The original Linear (``self.base``) keeps its pretrained weights and is
    held frozen (``requires_grad=False``). Only ``lora_A`` and ``lora_B`` are
    trained. ``lora_B`` is zero-initialised so that immediately after wrapping
    the forward pass is numerically identical to the base layer.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        # Standard LoRA init: A ~ Kaiming, B = 0  -> initial update is zero.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


def _resolve_submodule(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    """Walk a dotted path like ``attn.qkv`` and return ``(parent, attr_name)``."""
    parts = dotted.split(".")
    parent = root
    for piece in parts[:-1]:
        parent = getattr(parent, piece)
    return parent, parts[-1]


def apply_lora_to_virchow2(encoder: nn.Module, lora_config: LoRAConfig) -> dict[str, int]:
    """Inject LoRA into the last N transformer blocks of a Virchow2Encoder.

    Side effects:
      - All backbone parameters are first set to ``requires_grad=False``.
      - For each block in the last ``last_n_blocks`` and each path in
        ``target_modules``, the corresponding ``nn.Linear`` is replaced with a
        ``LoRALinear`` wrapping it. Only the new ``lora_A`` / ``lora_B``
        weights remain trainable inside the backbone.
      - Modules outside the encoder (the ABMIL heads, the logit head) are
        untouched and stay trainable as in the baseline.

    Returns a small summary dict with counts that the train script logs.
    """
    # Reach the underlying timm model. Virchow2Encoder stores it at ``.model``.
    timm_model = encoder.model
    blocks = timm_model.blocks

    # Hard freeze every backbone parameter first; LoRA replacements will
    # reintroduce trainable params only where we ask.
    for param in encoder.parameters():
        param.requires_grad = False

    n_blocks = len(blocks)
    last_n = min(lora_config.last_n_blocks, n_blocks)
    target_block_indices = list(range(n_blocks - last_n, n_blocks))

    n_layers_wrapped = 0
    n_layers_skipped = 0
    for idx in target_block_indices:
        block = blocks[idx]
        for path in lora_config.target_modules:
            parent, attr = _resolve_submodule(block, path)
            base = getattr(parent, attr)
            if not isinstance(base, nn.Linear):
                # Skip silently if a target path doesn't resolve to a Linear
                # (e.g. SwiGLU MLP layers have non-Linear submodule names).
                n_layers_skipped += 1
                continue
            wrapped = LoRALinear(
                base,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
            )
            setattr(parent, attr, wrapped)
            n_layers_wrapped += 1

    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in encoder.parameters())
    return {
        "lora_blocks_adapted": last_n,
        "lora_layers_wrapped": n_layers_wrapped,
        "lora_layers_skipped": n_layers_skipped,
        "encoder_trainable_params": n_trainable,
        "encoder_total_params": n_total,
        "target_block_indices": target_block_indices,
    }
