from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


try:
    import timm
    from timm.data import create_transform, resolve_data_config
    from timm.layers import SwiGLUPacked
except ImportError:  # pragma: no cover
    timm = None
    create_transform = None
    resolve_data_config = None
    SwiGLUPacked = None


VIRCHOW2_HF_ID = "hf-hub:paige-ai/Virchow2"


@dataclass
class Virchow2Config:
    model_name: str = VIRCHOW2_HF_ID
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_last_block: bool = False


def build_virchow2_preprocess() -> Callable:
    if timm is None or create_transform is None or resolve_data_config is None or SwiGLUPacked is None:  # pragma: no cover
        raise RuntimeError("timm is required for Virchow2 preprocessing.")
    model = timm.create_model(
        VIRCHOW2_HF_ID,
        pretrained=False,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    config = resolve_data_config(model.pretrained_cfg, model=model)
    return create_transform(**config, is_training=False)


class Virchow2Encoder(nn.Module):
    def __init__(self, config: Virchow2Config | None = None):
        super().__init__()
        if timm is None:  # pragma: no cover
            raise RuntimeError("timm is required for Virchow2 loading.")
        self.config = config or Virchow2Config()
        self.model = timm.create_model(
            self.config.model_name,
            pretrained=self.config.pretrained,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.output_dim = 2560
        self._set_trainable_params()

    def _set_trainable_params(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = not self.config.freeze_backbone

        if self.config.freeze_backbone and self.config.unfreeze_last_block:
            for block in self.model.blocks[-1:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.model, "norm"):
                for param in self.model.norm.parameters():
                    param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        class_token = output[:, 0]
        patch_tokens = output[:, 5:]
        return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
