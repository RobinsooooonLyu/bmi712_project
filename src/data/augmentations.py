from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

try:
    from skimage.color import hed2rgb, rgb2hed
except ImportError:  # pragma: no cover
    hed2rgb = None
    rgb2hed = None

from torchvision import transforms


class RandomHEDJitter:
    """Apply small HED-space perturbations for stain/intensity robustness."""

    def __init__(self, sigma: float = 0.05, bias: float = 0.02, p: float = 0.5):
        self.sigma = sigma
        self.bias = bias
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if rgb2hed is None or hed2rgb is None or random.random() > self.p:
            return image

        arr = np.asarray(image).astype(np.float32) / 255.0
        hed = rgb2hed(arr)
        hed += np.random.normal(0.0, self.sigma, size=hed.shape)
        hed += np.random.uniform(-self.bias, self.bias, size=(1, 1, 3))
        rgb = hed2rgb(hed)
        rgb = np.clip(rgb, 0.0, 1.0)
        return Image.fromarray((rgb * 255).astype(np.uint8))


class RandomGaussianBlur:
    """Frozen-specific blur augmentation."""

    def __init__(self, radius_min: float = 0.2, radius_max: float = 1.6, p: float = 0.35):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        radius = random.uniform(self.radius_min, self.radius_max)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


@dataclass
class ModalityAugmentationConfig:
    brightness: float = 0.18
    contrast: float = 0.18
    saturation: float = 0.12
    hue: float = 0.04
    hed_sigma: float = 0.05
    hed_bias: float = 0.02
    hed_p: float = 0.5
    blur_p: float = 0.35
    blur_radius_min: float = 0.2
    blur_radius_max: float = 1.6


def build_train_augmentor(
    modality: str,
    config: ModalityAugmentationConfig | None = None,
) -> Callable[[Image.Image], Image.Image]:
    config = config or ModalityAugmentationConfig()
    transforms_list: list[Callable[[Image.Image], Image.Image]] = [
        transforms.ColorJitter(
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue,
        ),
        RandomHEDJitter(
            sigma=config.hed_sigma,
            bias=config.hed_bias,
            p=config.hed_p,
        ),
    ]
    if modality == "frozen":
        transforms_list.append(
            RandomGaussianBlur(
                radius_min=config.blur_radius_min,
                radius_max=config.blur_radius_max,
                p=config.blur_p,
            )
        )
    return transforms.Compose(transforms_list)


def build_eval_augmentor() -> Callable[[Image.Image], Image.Image]:
    return transforms.Compose([])
