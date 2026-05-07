#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch

from src.data.augmentations import build_eval_augmentor
from src.data.binary_wsi_dataset import read_tsv
from src.models.patient_binary_model import PatientBinaryModel, PatientBinaryModelConfig
from src.models.virchow2_encoder import build_virchow2_preprocess

try:
    import openslide
except ImportError:  # pragma: no cover
    openslide = None


@dataclass
class SelectedCase:
    case_id: str
    label: int
    fold: int
    frozen_probability: float | None = None
    ffpe_probability: float | None = None
    selection_role: str = ""


@dataclass
class SlideAttentionResult:
    case_id: str
    modality: str
    fold: int
    label: int
    probability: float
    logit: float
    slide_name: str
    slide_path: Path
    coord_path: Path
    read_level: int
    read_size: int
    coords: list[dict[str, int]]
    attention: torch.Tensor
    attention_norm: list[float]
    stride: int
    grid_origin_x: int
    grid_origin_y: int
    window_anchor: tuple[int, int]
    window_indices: list[int]
    window_bbox_level0: tuple[int, int, int, int]


def read_predictions(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    run_dir = Path(run_dir)
    rows: dict[str, dict[str, Any]] = {}
    for fold_dir in sorted(path for path in run_dir.glob("fold*") if path.is_dir()):
        config_path = fold_dir / "config.json"
        predictions_path = fold_dir / "test_predictions.tsv"
        if not config_path.exists() or not predictions_path.exists():
            continue
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        test_folds = config.get("test_folds", [])
        fold = int(test_folds[0]) if test_folds else int(fold_dir.name.removeprefix("fold"))
        with predictions_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows[row["case_id"]] = {
                    "case_id": row["case_id"],
                    "fold": fold,
                    "label": int(row["label"]),
                    "probability": float(row["probability"]),
                    "logit": float(row["logit"]),
                }
    return rows


def read_split_labels(split_path: str | Path) -> dict[str, dict[str, int]]:
    rows = {}
    for row in read_tsv(split_path):
        rows[row["case_id"]] = {
            "fold": int(row["fold"]),
            "label": int(row["label"]),
        }
    return rows


def auto_select_cases(
    frozen_run_dir: str | Path,
    ffpe_run_dir: str | Path,
    split_path: str | Path,
    positive_case_id: str | None,
    negative_case_id: str | None,
) -> list[SelectedCase]:
    split_rows = read_split_labels(split_path)
    frozen = read_predictions(frozen_run_dir)
    ffpe = read_predictions(ffpe_run_dir)

    def from_case_id(case_id: str, role: str) -> SelectedCase:
        if case_id not in split_rows:
            raise KeyError(f"case_id={case_id} not found in split file {split_path}")
        return SelectedCase(
            case_id=case_id,
            label=split_rows[case_id]["label"],
            fold=split_rows[case_id]["fold"],
            frozen_probability=frozen.get(case_id, {}).get("probability"),
            ffpe_probability=ffpe.get(case_id, {}).get("probability"),
            selection_role=role,
        )

    selected: list[SelectedCase] = []
    if positive_case_id:
        selected.append(from_case_id(positive_case_id, "high_risk"))
    if negative_case_id:
        selected.append(from_case_id(negative_case_id, "low_risk"))
    if len(selected) == 2:
        return selected

    candidates = []
    for case_id in sorted(set(frozen).intersection(ffpe)):
        fz = frozen[case_id]
        fp = ffpe[case_id]
        if int(fz["label"]) != int(fp["label"]):
            continue
        if int(fz["fold"]) != int(fp["fold"]):
            continue
        label = int(fz["label"])
        fz_prob = float(fz["probability"])
        fp_prob = float(fp["probability"])
        if label == 1 and fz_prob >= 0.5 and fp_prob >= 0.5:
            score = min(fz_prob, fp_prob)
            candidates.append(("high_risk", score, case_id, label, int(fz["fold"]), fz_prob, fp_prob))
        if label == 0 and fz_prob < 0.5 and fp_prob < 0.5:
            score = min(1.0 - fz_prob, 1.0 - fp_prob)
            candidates.append(("low_risk", score, case_id, label, int(fz["fold"]), fz_prob, fp_prob))

    if not positive_case_id:
        high = [item for item in candidates if item[0] == "high_risk"]
        if not high:
            raise RuntimeError("No correctly predicted high-risk candidate found in both runs.")
        role, _, case_id, label, fold, fz_prob, fp_prob = max(high, key=lambda item: item[1])
        selected.append(
            SelectedCase(case_id, label, fold, fz_prob, fp_prob, role)
        )
    if not negative_case_id:
        low = [item for item in candidates if item[0] == "low_risk"]
        if not low:
            raise RuntimeError("No correctly predicted low-risk candidate found in both runs.")
        role, _, case_id, label, fold, fz_prob, fp_prob = max(low, key=lambda item: item[1])
        selected.append(
            SelectedCase(case_id, label, fold, fz_prob, fp_prob, role)
        )
    return selected


def load_fold_config(run_dir: str | Path, fold: int) -> dict[str, Any]:
    config_path = Path(run_dir) / f"fold{fold}" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing fold config: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_model(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> PatientBinaryModel:
    model = PatientBinaryModel(
        PatientBinaryModelConfig(
            dropout=float(config.get("dropout", 0.25)),
            encoder_batch_size=int(config.get("encoder_batch_size", 64)),
            freeze_backbone=bool(config.get("freeze_backbone", True)),
            unfreeze_last_block=bool(config.get("unfreeze_last_block", False)),
        )
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def select_slide_for_case(config: dict[str, Any], case_id: str) -> tuple[str, Path, Path, dict[str, Any]]:
    manifest = {row["case_id"]: row for row in read_tsv(config["patient_manifest_path"])}
    if case_id not in manifest:
        raise KeyError(f"case_id={case_id} not found in {config['patient_manifest_path']}")
    modality = config["modality"]
    slide_key = "ffpe_filenames" if modality == "ffpe" else "frozen_filenames"
    slide_names = [name for name in manifest[case_id][slide_key].split("|") if name]
    if not slide_names:
        raise RuntimeError(f"No {modality} slides listed for case_id={case_id}")
    wsi_root = Path(config["wsi_root"])
    coords_root = Path(config["coords_root"])
    candidates = []
    for slide_name in slide_names:
        coord_path = coords_root / f"{slide_name}.json"
        if not coord_path.exists():
            continue
        with coord_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        candidates.append((int(payload.get("num_tiles", len(payload.get("coordinates", [])))), slide_name, wsi_root / slide_name, coord_path, payload))
    if not candidates:
        raise RuntimeError(f"No coordinate files found for case_id={case_id} modality={modality}")
    _, slide_name, slide_path, coord_path, payload = max(candidates, key=lambda item: item[0])
    return slide_name, slide_path, coord_path, payload


def select_attention_coords(
    payload: dict[str, Any],
    max_attention_tiles: int,
    seed: int,
) -> list[dict[str, int]]:
    coords = list(payload["coordinates"])
    if max_attention_tiles <= 0 or len(coords) <= max_attention_tiles:
        return coords
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(coords), generator=generator)[:max_attention_tiles].tolist()
    return [coords[index] for index in sorted(indices)]


def positive_median_diff(values: list[int]) -> int:
    unique = sorted(set(values))
    diffs = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    if not diffs:
        return 224
    return max(1, int(round(statistics.median(diffs))))


def infer_stride(coords: list[dict[str, int]]) -> int:
    x_stride = positive_median_diff([int(coord["x"]) for coord in coords])
    y_stride = positive_median_diff([int(coord["y"]) for coord in coords])
    return max(1, int(round(statistics.median([x_stride, y_stride]))))


def normalize_attention(attention: torch.Tensor) -> list[float]:
    values = attention.detach().cpu().float()
    if values.numel() == 0:
        return []
    sorted_values = torch.sort(values).values
    lo = sorted_values[min(int(0.05 * (values.numel() - 1)), values.numel() - 1)].item()
    hi = sorted_values[min(int(0.95 * (values.numel() - 1)), values.numel() - 1)].item()
    denom = max(hi - lo, 1e-12)
    return [max(0.0, min(1.0, (float(item) - lo) / denom)) for item in values.tolist()]


def choose_attention_window(
    coords: list[dict[str, int]],
    attention_norm: list[float],
    rows: int,
    cols: int,
    min_coverage: float,
    edge_margin_fraction: float,
    min_mean_tissue_fraction: float,
) -> tuple[tuple[int, int], list[int], int, int, int]:
    stride = infer_stride(coords)
    origin_x = min(int(coord["x"]) for coord in coords)
    origin_y = min(int(coord["y"]) for coord in coords)
    grid_to_idx: dict[tuple[int, int], int] = {}
    for idx, coord in enumerate(coords):
        gx = int(round((int(coord["x"]) - origin_x) / stride))
        gy = int(round((int(coord["y"]) - origin_y) / stride))
        grid_to_idx[(gx, gy)] = idx

    grid_x = [item[0] for item in grid_to_idx]
    grid_y = [item[1] for item in grid_to_idx]
    min_gx, max_gx = min(grid_x), max(grid_x)
    min_gy, max_gy = min(grid_y), max(grid_y)
    grid_width = max(max_gx - min_gx + 1, 1)
    grid_height = max(max_gy - min_gy + 1, 1)
    min_tiles = max(1, int(math.ceil(rows * cols * min_coverage)))
    best: tuple[float, tuple[int, int], list[int]] | None = None
    for gx, gy in grid_to_idx:
        anchor = (gx - cols // 2, gy - rows // 2)
        indices = []
        for yy in range(anchor[1], anchor[1] + rows):
            for xx in range(anchor[0], anchor[0] + cols):
                idx = grid_to_idx.get((xx, yy))
                if idx is not None:
                    indices.append(idx)
        if len(indices) < min_tiles:
            continue
        values = [attention_norm[idx] for idx in indices]
        tissue_values = [float(coords[idx].get("tissue_fraction", 1.0)) for idx in indices]
        mean_tissue = statistics.mean(tissue_values)
        if mean_tissue < min_mean_tissue_fraction:
            continue
        coverage = len(indices) / float(rows * cols)
        hot_frac = sum(value >= 0.70 for value in values) / len(values)
        saturated_frac = sum(value >= 0.95 for value in values) / len(values)
        center_x = anchor[0] + cols / 2.0
        center_y = anchor[1] + rows / 2.0
        margin_x = min(center_x - min_gx, max_gx - center_x) / grid_width
        margin_y = min(center_y - min_gy, max_gy - center_y) / grid_height
        edge_margin = max(0.0, min(margin_x, margin_y))
        edge_penalty = max(0.0, edge_margin_fraction - edge_margin) / max(edge_margin_fraction, 1e-6)
        score = (
            statistics.mean(values)
            + 0.25 * coverage
            + 0.20 * hot_frac
            + 0.20 * mean_tissue
            + 0.20 * edge_margin
            - 0.35 * max(0.0, saturated_frac - 0.35)
            - 0.45 * edge_penalty
        )
        if best is None or score > best[0]:
            best = (score, anchor, indices)

    if best is None:
        central_candidates = []
        for idx, coord in enumerate(coords):
            gx = int(round((int(coord["x"]) - origin_x) / stride))
            gy = int(round((int(coord["y"]) - origin_y) / stride))
            margin_x = min(gx - min_gx, max_gx - gx) / grid_width
            margin_y = min(gy - min_gy, max_gy - gy) / grid_height
            if min(margin_x, margin_y) >= edge_margin_fraction:
                central_candidates.append(idx)
        search_indices = central_candidates or list(range(len(attention_norm)))
        top_idx = max(search_indices, key=lambda idx: attention_norm[idx])
        coord = coords[top_idx]
        gx = int(round((int(coord["x"]) - origin_x) / stride))
        gy = int(round((int(coord["y"]) - origin_y) / stride))
        anchor = (gx - cols // 2, gy - rows // 2)
        indices = [top_idx]
    else:
        _, anchor, indices = best
    return anchor, indices, stride, origin_x, origin_y


def read_tile(
    slide: Any,
    x: int,
    y: int,
    read_level: int,
    read_size: int,
    output_tile_size: int | None = None,
) -> Image.Image:
    tile = slide.read_region((int(x), int(y)), read_level, (read_size, read_size)).convert("RGB")
    if output_tile_size is not None and tile.size != (output_tile_size, output_tile_size):
        tile = tile.resize((output_tile_size, output_tile_size), Image.Resampling.BILINEAR)
    return tile


def run_slide_attention(
    model: PatientBinaryModel,
    slide_path: Path,
    payload: dict[str, Any],
    coords: list[dict[str, int]],
    device: torch.device,
    encoder_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if openslide is None:  # pragma: no cover
        raise RuntimeError("openslide-python is required for attention visualization.")
    preprocess = build_virchow2_preprocess()
    augmentor = build_eval_augmentor()
    read_level = int(payload.get("selected_level", 0))
    read_size = int(payload.get("tile_size_level", payload.get("tile_size", 224)))
    feature_chunks = []
    slide = openslide.OpenSlide(str(slide_path))
    try:
        for start in range(0, len(coords), encoder_batch_size):
            chunk = coords[start : start + encoder_batch_size]
            tensors = []
            for coord in chunk:
                tile = read_tile(slide, int(coord["x"]), int(coord["y"]), read_level, read_size)
                tile = augmentor(tile)
                tensors.append(preprocess(tile))
            batch = torch.stack(tensors, dim=0).to(device)
            with torch.no_grad():
                features = model.encoder(batch).detach().cpu()
            feature_chunks.append(features)
    finally:
        slide.close()

    features = torch.cat(feature_chunks, dim=0).to(device)
    with torch.no_grad():
        slide_embedding, tile_attention = model.slide_mil(features)
        patient_embedding, _ = model.patient_mil(slide_embedding.unsqueeze(0))
        logit = model.logit_head(patient_embedding).squeeze(-1)
        probability = torch.sigmoid(logit)
    return tile_attention.detach().cpu(), logit.detach().cpu(), probability.detach().cpu()


def color_overlay(tile: Image.Image, strength: float, alpha_scale: float = 0.55) -> Image.Image:
    if strength <= 0.0:
        return tile
    overlay = Image.new("RGBA", tile.size, (255, 40, 20, int(255 * alpha_scale * strength)))
    base = tile.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_thumbnail_overlay(
    slide_path: Path,
    payload: dict[str, Any],
    coords: list[dict[str, int]],
    attention_norm: list[float],
    bbox_level0: tuple[int, int, int, int],
    outpath: Path,
    title: str,
    max_size: int,
    min_display_attention: float,
) -> Image.Image:
    if openslide is None:  # pragma: no cover
        raise RuntimeError("openslide-python is required for attention visualization.")
    read_level = int(payload.get("selected_level", 0))
    read_size = int(payload.get("tile_size_level", payload.get("tile_size", 224)))
    slide = openslide.OpenSlide(str(slide_path))
    try:
        width, height = slide.dimensions
        scale = min(max_size / width, max_size / height, 1.0)
        thumb = slide.get_thumbnail((int(width * scale), int(height * scale))).convert("RGBA")
        downsample = float(slide.level_downsamples[read_level])
    finally:
        slide.close()

    tile_extent = int(round(read_size * downsample))
    overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for coord, strength in zip(coords, attention_norm):
        if strength < min_display_attention:
            continue
        x0 = int(int(coord["x"]) * scale)
        y0 = int(int(coord["y"]) * scale)
        x1 = int((int(coord["x"]) + tile_extent) * scale)
        y1 = int((int(coord["y"]) + tile_extent) * scale)
        alpha = int(35 + 170 * strength)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 35, 15, alpha), outline=(255, 230, 40, min(255, alpha + 20)))

    x0, y0, x1, y1 = bbox_level0
    draw.rectangle(
        [int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)],
        outline=(0, 220, 255, 255),
        width=4,
    )
    combined = Image.alpha_composite(thumb, overlay).convert("RGB")
    combined = add_header(combined, title)
    combined.save(outpath, quality=95)
    return combined


def add_header(image: Image.Image, title: str, header_h: int = 34) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + header_h), (255, 255, 255))
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), title, fill=(0, 0, 0))
    return canvas


def draw_region_mosaic(
    slide_path: Path,
    payload: dict[str, Any],
    window_anchor: tuple[int, int],
    stride: int,
    origin_x: int,
    origin_y: int,
    rows: int,
    cols: int,
    attention_by_grid: dict[tuple[int, int], float],
    outpath: Path,
    title: str,
    tile_size: int,
    min_display_attention: float,
) -> Image.Image:
    if openslide is None:  # pragma: no cover
        raise RuntimeError("openslide-python is required for attention visualization.")
    read_level = int(payload.get("selected_level", 0))
    read_size = int(payload.get("tile_size_level", payload.get("tile_size", 224)))
    pad = 4
    header_h = 34
    width = cols * tile_size + (cols + 1) * pad
    height = header_h + rows * tile_size + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 9), title, fill=(0, 0, 0))

    slide = openslide.OpenSlide(str(slide_path))
    try:
        for row in range(rows):
            for col in range(cols):
                gx = window_anchor[0] + col
                gy = window_anchor[1] + row
                x = origin_x + gx * stride
                y = origin_y + gy * stride
                tile = read_tile(slide, x, y, read_level, read_size, output_tile_size=tile_size)
                strength = attention_by_grid.get((gx, gy), 0.0)
                if strength >= min_display_attention:
                    tile = color_overlay(tile, strength)
                px = pad + col * (tile_size + pad)
                py = header_h + pad + row * (tile_size + pad)
                canvas.paste(tile, (px, py))
                outline = (255, 80, 40) if strength >= min_display_attention else (180, 180, 180)
                draw.rectangle([px, py, px + tile_size, py + tile_size], outline=outline, width=2)
    finally:
        slide.close()

    canvas.save(outpath, quality=95)
    return canvas


def make_case_panel(
    case: SelectedCase,
    frozen_thumb: Image.Image,
    frozen_region: Image.Image,
    ffpe_thumb: Image.Image,
    ffpe_region: Image.Image,
    outpath: Path,
) -> None:
    target_h = 420

    def resize_to_h(image: Image.Image) -> Image.Image:
        scale = target_h / image.height
        return image.resize((int(image.width * scale), target_h), Image.Resampling.BILINEAR)

    panels = [resize_to_h(item) for item in [frozen_thumb, frozen_region, ffpe_thumb, ffpe_region]]
    pad = 12
    header_h = 48
    width = sum(panel.width for panel in panels) + pad * (len(panels) + 1)
    height = target_h + header_h + 2 * pad
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label = "high-risk" if case.label == 1 else "low-risk"
    draw.text(
        (pad, 12),
        f"{case.selection_role}: {case.case_id} | true={label} | fold={case.fold} | "
        f"frozen_p={case.frozen_probability} | ffpe_p={case.ffpe_probability}",
        fill=(0, 0, 0),
    )
    x = pad
    for panel in panels:
        canvas.paste(panel, (x, header_h + pad))
        x += panel.width + pad
    canvas.save(outpath, quality=95)


def render_case_modality(
    case: SelectedCase,
    modality: str,
    run_dir: str | Path,
    outdir: Path,
    device: torch.device,
    max_attention_tiles: int,
    window_rows: int,
    window_cols: int,
    min_window_coverage: float,
    edge_margin_fraction: float,
    min_window_mean_tissue_fraction: float,
    thumbnail_max_size: int,
    region_tile_size: int,
    min_display_attention: float,
    seed: int,
) -> tuple[SlideAttentionResult, Image.Image, Image.Image]:
    config = load_fold_config(run_dir, case.fold)
    checkpoint = Path(run_dir) / f"fold{case.fold}" / "model.pt"
    model = load_model(config, checkpoint, device)
    slide_name, slide_path, coord_path, payload = select_slide_for_case(config, case.case_id)
    coords = select_attention_coords(payload, max_attention_tiles=max_attention_tiles, seed=seed + case.fold)
    attention, logit, probability = run_slide_attention(
        model=model,
        slide_path=slide_path,
        payload=payload,
        coords=coords,
        device=device,
        encoder_batch_size=int(config.get("encoder_batch_size", 64)),
    )
    attention_norm = normalize_attention(attention)
    anchor, window_indices, stride, origin_x, origin_y = choose_attention_window(
        coords=coords,
        attention_norm=attention_norm,
        rows=window_rows,
        cols=window_cols,
        min_coverage=min_window_coverage,
        edge_margin_fraction=edge_margin_fraction,
        min_mean_tissue_fraction=min_window_mean_tissue_fraction,
    )
    read_level = int(payload.get("selected_level", 0))
    read_size = int(payload.get("tile_size_level", payload.get("tile_size", 224)))
    if openslide is None:  # pragma: no cover
        raise RuntimeError("openslide-python is required for attention visualization.")
    slide = openslide.OpenSlide(str(slide_path))
    try:
        downsample = float(slide.level_downsamples[read_level])
    finally:
        slide.close()
    tile_extent = int(round(read_size * downsample))
    bbox = (
        origin_x + anchor[0] * stride,
        origin_y + anchor[1] * stride,
        origin_x + (anchor[0] + window_cols - 1) * stride + tile_extent,
        origin_y + (anchor[1] + window_rows - 1) * stride + tile_extent,
    )

    grid_attention = {}
    for coord, strength in zip(coords, attention_norm):
        gx = int(round((int(coord["x"]) - origin_x) / stride))
        gy = int(round((int(coord["y"]) - origin_y) / stride))
        grid_attention[(gx, gy)] = strength

    prefix = f"{case.selection_role}_{case.case_id}_{modality}"
    probability_value = float(probability.item())
    title = (
        f"{case.case_id} | {modality} | p={probability_value:.3f} | "
        f"tiles={len(coords)} | slide={slide_name}"
    )
    thumb = draw_thumbnail_overlay(
        slide_path=slide_path,
        payload=payload,
        coords=coords,
        attention_norm=attention_norm,
        bbox_level0=bbox,
        outpath=outdir / f"{prefix}_thumbnail_attention.jpg",
        title=title,
        max_size=thumbnail_max_size,
        min_display_attention=min_display_attention,
    )
    region = draw_region_mosaic(
        slide_path=slide_path,
        payload=payload,
        window_anchor=anchor,
        stride=stride,
        origin_x=origin_x,
        origin_y=origin_y,
        rows=window_rows,
        cols=window_cols,
        attention_by_grid=grid_attention,
        outpath=outdir / f"{prefix}_{window_rows}x{window_cols}_attention_region.jpg",
        title=f"{case.case_id} | {modality} | selected {window_rows}x{window_cols} attention region",
        tile_size=region_tile_size,
        min_display_attention=min_display_attention,
    )
    result = SlideAttentionResult(
        case_id=case.case_id,
        modality=modality,
        fold=case.fold,
        label=case.label,
        probability=probability_value,
        logit=float(logit.item()),
        slide_name=slide_name,
        slide_path=slide_path,
        coord_path=coord_path,
        read_level=read_level,
        read_size=read_size,
        coords=coords,
        attention=attention,
        attention_norm=attention_norm,
        stride=stride,
        grid_origin_x=origin_x,
        grid_origin_y=origin_y,
        window_anchor=anchor,
        window_indices=window_indices,
        window_bbox_level0=bbox,
    )
    return result, thumb, region


def write_selection_summary(outdir: Path, cases: list[SelectedCase], results: list[SlideAttentionResult]) -> None:
    with (outdir / "selected_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cases": [case.__dict__ for case in cases],
                "modalities": [
                    {
                        "case_id": result.case_id,
                        "modality": result.modality,
                        "fold": result.fold,
                        "label": result.label,
                        "probability": result.probability,
                        "logit": result.logit,
                        "slide_name": result.slide_name,
                        "slide_path": str(result.slide_path),
                        "coord_path": str(result.coord_path),
                        "n_attention_tiles": len(result.coords),
                        "window_anchor": list(result.window_anchor),
                        "window_tile_count_in_attention_set": len(result.window_indices),
                        "window_bbox_level0": list(result.window_bbox_level0),
                    }
                    for result in results
                ],
            },
            handle,
            indent=2,
        )
    with (outdir / "selected_cases.tsv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "role",
            "label",
            "fold",
            "modality",
            "probability",
            "slide_name",
            "n_attention_tiles",
            "window_tile_count_in_attention_set",
            "window_bbox_level0",
        ]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        case_by_id = {case.case_id: case for case in cases}
        for result in results:
            case = case_by_id[result.case_id]
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "role": case.selection_role,
                    "label": result.label,
                    "fold": result.fold,
                    "modality": result.modality,
                    "probability": result.probability,
                    "slide_name": result.slide_name,
                    "n_attention_tiles": len(result.coords),
                    "window_tile_count_in_attention_set": len(result.window_indices),
                    "window_bbox_level0": ",".join(str(item) for item in result.window_bbox_level0),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-run-dir", default="output/tcga_luad_frozen_high_risk_baseline")
    parser.add_argument("--ffpe-run-dir", default="output/tcga_luad_ffpe_high_risk_baseline")
    parser.add_argument("--split-path", default="manifests/full_strict/tcga_luad_high_risk_grade_vpi_lvi_5fold_trainready.tsv")
    parser.add_argument("--outdir", default="output/tcga_luad_attention_region_examples")
    parser.add_argument("--positive-case-id")
    parser.add_argument("--negative-case-id")
    parser.add_argument("--max-attention-tiles", type=int, default=0, help="0 means use all tissue coordinates from selected slide.")
    parser.add_argument("--window-rows", type=int, default=8)
    parser.add_argument("--window-cols", type=int, default=8)
    parser.add_argument("--min-window-coverage", type=float, default=0.65)
    parser.add_argument("--edge-margin-fraction", type=float, default=0.08)
    parser.add_argument("--min-window-mean-tissue-fraction", type=float, default=0.35)
    parser.add_argument("--thumbnail-max-size", type=int, default=1600)
    parser.add_argument("--region-tile-size", type=int, default=128)
    parser.add_argument("--min-display-attention", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if openslide is None:  # pragma: no cover
        raise RuntimeError("openslide-python is required for attention visualization.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"Runtime device: requested={args.device} selected={device} "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"CUDA current device: {torch.cuda.current_device()} {torch.cuda.get_device_name()}", flush=True)

    cases = auto_select_cases(
        frozen_run_dir=args.frozen_run_dir,
        ffpe_run_dir=args.ffpe_run_dir,
        split_path=args.split_path,
        positive_case_id=args.positive_case_id,
        negative_case_id=args.negative_case_id,
    )
    print("Selected cases:", flush=True)
    for case in cases:
        print(
            f"  {case.selection_role}: {case.case_id} label={case.label} fold={case.fold} "
            f"frozen_p={case.frozen_probability} ffpe_p={case.ffpe_probability}",
            flush=True,
        )

    all_results: list[SlideAttentionResult] = []
    for case in cases:
        print(f"Rendering case={case.case_id} role={case.selection_role}", flush=True)
        frozen_result, frozen_thumb, frozen_region = render_case_modality(
            case=case,
            modality="frozen",
            run_dir=args.frozen_run_dir,
            outdir=outdir,
            device=device,
            max_attention_tiles=args.max_attention_tiles,
            window_rows=args.window_rows,
            window_cols=args.window_cols,
            min_window_coverage=args.min_window_coverage,
            edge_margin_fraction=args.edge_margin_fraction,
            min_window_mean_tissue_fraction=args.min_window_mean_tissue_fraction,
            thumbnail_max_size=args.thumbnail_max_size,
            region_tile_size=args.region_tile_size,
            min_display_attention=args.min_display_attention,
            seed=args.seed,
        )
        all_results.append(frozen_result)
        ffpe_result, ffpe_thumb, ffpe_region = render_case_modality(
            case=case,
            modality="ffpe",
            run_dir=args.ffpe_run_dir,
            outdir=outdir,
            device=device,
            max_attention_tiles=args.max_attention_tiles,
            window_rows=args.window_rows,
            window_cols=args.window_cols,
            min_window_coverage=args.min_window_coverage,
            edge_margin_fraction=args.edge_margin_fraction,
            min_window_mean_tissue_fraction=args.min_window_mean_tissue_fraction,
            thumbnail_max_size=args.thumbnail_max_size,
            region_tile_size=args.region_tile_size,
            min_display_attention=args.min_display_attention,
            seed=args.seed + 1000,
        )
        all_results.append(ffpe_result)
        make_case_panel(
            case=case,
            frozen_thumb=frozen_thumb,
            frozen_region=frozen_region,
            ffpe_thumb=ffpe_thumb,
            ffpe_region=ffpe_region,
            outpath=outdir / f"{case.selection_role}_{case.case_id}_paired_attention_panel.jpg",
        )
    write_selection_summary(outdir, cases, all_results)
    print(f"Saved attention visualizations to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
