#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:  # pragma: no cover
    openslide = None


OPENSLIDE_MPP_X = "openslide.mpp-x"
APERIO_MPP = "aperio.MPP"
OPENSLIDE_OBJECTIVE_POWER = "openslide.objective-power"
APERIO_APP_MAG = "aperio.AppMag"


def objective_power_to_mpp(power: float) -> float | None:
    if power <= 0:
        return None
    mapping = {
        40.0: 0.25,
        20.0: 0.50,
        10.0: 1.00,
    }
    rounded = round(power, 1)
    if rounded in mapping:
        return mapping[rounded]
    return None


def get_slide_mpp(slide: "openslide.OpenSlide") -> float:
    raw = slide.properties.get(OPENSLIDE_MPP_X)
    if raw:
        return float(raw)
    raw = slide.properties.get(APERIO_MPP)
    if raw:
        return float(raw)

    for key in (OPENSLIDE_OBJECTIVE_POWER, APERIO_APP_MAG):
        raw = slide.properties.get(key)
        if not raw:
            continue
        try:
            power = float(raw)
        except ValueError:
            continue
        mpp = objective_power_to_mpp(power)
        if mpp is not None:
            return mpp
    raise RuntimeError("Unable to resolve slide microns-per-pixel from OpenSlide properties.")


def choose_level(slide: "openslide.OpenSlide", target_mpp: float) -> tuple[int, float]:
    base_mpp = get_slide_mpp(slide)
    candidates = []
    for level, downsample in enumerate(slide.level_downsamples):
        level_mpp = base_mpp * float(downsample)
        candidates.append((abs(level_mpp - target_mpp), level, level_mpp))
    _, level, level_mpp = min(candidates, key=lambda item: item[0])
    return level, level_mpp


def clamp_region(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def build_tissue_mask(region: Image.Image, threshold: int, min_saturation: int) -> np.ndarray:
    arr = np.asarray(region.convert("RGB"))
    rgb_mean = arr.mean(axis=2)
    rgb_max = arr.max(axis=2)
    rgb_min = arr.min(axis=2)
    saturation = rgb_max - rgb_min
    return (rgb_mean < threshold) & (saturation > min_saturation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument(
        "--mask-mpp",
        type=float,
        default=8.0,
        help="Approximate mpp to use for low-resolution tissue-mask generation.",
    )
    parser.add_argument("--mask-threshold", type=int, default=220)
    parser.add_argument("--min-saturation", type=int, default=12)
    parser.add_argument("--min-tissue-coverage", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if openslide is None:  # pragma: no cover
        raise RuntimeError("openslide-python is required for tile coordinate extraction.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_paths = sorted(input_dir.glob("*.svs"))
    if args.limit > 0:
        slide_paths = slide_paths[: args.limit]

    completed = 0
    skipped = 0
    failed = 0
    for slide_path in slide_paths:
        out_path = output_dir / f"{slide_path.name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"SKIP\t{slide_path.name}\t{out_path}")
            skipped += 1
            continue

        slide = openslide.OpenSlide(str(slide_path))
        try:
            level, level_mpp = choose_level(slide, args.target_mpp)
            mask_level, mask_level_mpp = choose_level(slide, args.mask_mpp)
            level_dims = slide.level_dimensions[level]
            mask_level_dims = slide.level_dimensions[mask_level]
            target_downsample = float(slide.level_downsamples[level])
            mask_downsample = float(slide.level_downsamples[mask_level])
            tile_size_level = max(1, int(round(args.tile_size * args.target_mpp / level_mpp)))

            thumbnail = slide.read_region((0, 0), mask_level, mask_level_dims).convert("RGB")
            mask = build_tissue_mask(
                thumbnail,
                threshold=args.mask_threshold,
                min_saturation=args.min_saturation,
            )

            coords = []
            x_steps = math.ceil(level_dims[0] / tile_size_level)
            y_steps = math.ceil(level_dims[1] / tile_size_level)
            for yi in range(y_steps):
                for xi in range(x_steps):
                    x0 = xi * tile_size_level
                    y0 = yi * tile_size_level
                    x1 = min(level_dims[0], x0 + tile_size_level)
                    y1 = min(level_dims[1], y0 + tile_size_level)

                    base_x0 = x0 * target_downsample
                    base_y0 = y0 * target_downsample
                    base_x1 = x1 * target_downsample
                    base_y1 = y1 * target_downsample

                    mask_x0 = int(math.floor(base_x0 / mask_downsample))
                    mask_y0 = int(math.floor(base_y0 / mask_downsample))
                    mask_x1 = int(math.ceil(base_x1 / mask_downsample))
                    mask_y1 = int(math.ceil(base_y1 / mask_downsample))
                    mask_x0, mask_y0, mask_x1, mask_y1 = clamp_region(
                        mask_x0,
                        mask_y0,
                        mask_x1,
                        mask_y1,
                        mask_level_dims[0],
                        mask_level_dims[1],
                    )
                    patch_mask = mask[mask_y0:mask_y1, mask_x0:mask_x1]
                    if patch_mask.size == 0:
                        continue
                    tissue_fraction = float(patch_mask.mean())
                    if tissue_fraction < args.min_tissue_coverage:
                        continue
                    coords.append(
                        {
                            "x": int(round(base_x0)),
                            "y": int(round(base_y0)),
                            "tile_size": args.tile_size,
                            "tissue_fraction": tissue_fraction,
                        }
                    )

            payload = {
                "slide_filename": slide_path.name,
                "target_mpp": args.target_mpp,
                "selected_level": level,
                "selected_level_mpp": level_mpp,
                "mask_mpp": args.mask_mpp,
                "mask_level": mask_level,
                "mask_level_mpp": mask_level_mpp,
                "tile_size": args.tile_size,
                "tile_size_level": tile_size_level,
                "num_tiles": len(coords),
                "coordinates": coords,
            }
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            print(f"{slide_path.name}\t{len(coords)} tiles\t{out_path}")
            completed += 1
        except Exception as exc:
            print(f"ERROR\t{slide_path.name}\t{type(exc).__name__}: {exc}")
            failed += 1
        finally:
            slide.close()

    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    main()
