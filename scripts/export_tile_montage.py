from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

try:
    import openslide
except ImportError:  # pragma: no cover
    openslide = None


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_manifest_row(manifest_path: str | Path, case_id: str) -> dict[str, str]:
    rows = {row["case_id"]: row for row in read_tsv(manifest_path)}
    if case_id not in rows:
        raise KeyError(f"Case {case_id} not found in {manifest_path}")
    return rows[case_id]


def select_slide_names(row: dict[str, str], modality: str, max_slides_per_patient: int) -> list[str]:
    key = "ffpe_filenames" if modality == "ffpe" else "frozen_filenames"
    names = [name for name in row[key].split("|") if name]
    return names[:max_slides_per_patient]


def select_coords(payload: dict[str, Any], tiles_per_slide: int, strategy: str, seed: int) -> list[dict[str, Any]]:
    coords = payload["coordinates"]
    if len(coords) <= tiles_per_slide:
        return coords
    if strategy == "first":
        return coords[:tiles_per_slide]
    rng = random.Random(seed)
    return rng.sample(coords, tiles_per_slide)


def read_tile(slide_path: Path, payload: dict[str, Any], coord: dict[str, Any], output_tile_size: int) -> Image.Image:
    if openslide is None:  # pragma: no cover
        raise RuntimeError("openslide-python is required for montage export.")
    slide = openslide.OpenSlide(str(slide_path))
    try:
        tile = slide.read_region(
            (int(coord["x"]), int(coord["y"])),
            int(payload.get("selected_level", 0)),
            (int(payload.get("tile_size_level", payload.get("tile_size", 224))),) * 2,
        ).convert("RGB")
    finally:
        slide.close()
    if tile.size != (output_tile_size, output_tile_size):
        tile = tile.resize((output_tile_size, output_tile_size), Image.Resampling.BILINEAR)
    return tile


def make_montage(
    tiles: list[Image.Image],
    labels: list[str],
    tile_size: int,
    n_cols: int,
    header: str,
) -> Image.Image:
    n_tiles = len(tiles)
    n_rows = math.ceil(n_tiles / n_cols)
    pad = 8
    label_h = 18
    header_h = 34
    width = n_cols * tile_size + (n_cols + 1) * pad
    height = header_h + n_rows * (tile_size + label_h) + (n_rows + 1) * pad
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), header, fill=(0, 0, 0))

    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        row = idx // n_cols
        col = idx % n_cols
        x = pad + col * (tile_size + pad)
        y = header_h + pad + row * (tile_size + label_h)
        canvas.paste(tile, (x, y))
        draw.rectangle([x, y, x + tile_size, y + tile_size], outline=(180, 180, 180), width=1)
        draw.text((x, y + tile_size + 2), label, fill=(0, 0, 0))
    return canvas


def export_case_montage(
    case_id: str,
    manifest_path: str | Path,
    wsi_root: str | Path,
    coords_root: str | Path,
    modality: str,
    output_dir: str | Path,
    tiles_per_slide: int,
    max_slides_per_patient: int,
    tile_size: int,
    n_cols: int,
    strategy: str,
    seed: int,
) -> Path:
    row = load_manifest_row(manifest_path, case_id)
    slide_names = select_slide_names(row, modality, max_slides_per_patient)

    all_tiles: list[Image.Image] = []
    all_labels: list[str] = []
    for slide_idx, slide_name in enumerate(slide_names):
        slide_path = Path(wsi_root) / slide_name
        coord_path = Path(coords_root) / f"{slide_name}.json"
        payload = json.loads(coord_path.read_text())
        coords = select_coords(payload, tiles_per_slide, strategy, seed + slide_idx)
        for tile_idx, coord in enumerate(coords):
            tile = read_tile(slide_path, payload, coord, tile_size)
            tissue_fraction = coord.get("tissue_fraction", 0.0)
            label = f"s{slide_idx+1} t{tile_idx+1} tf={tissue_fraction:.2f}"
            all_tiles.append(tile)
            all_labels.append(label)

    header = f"{case_id} | {modality} | slides={len(slide_names)} | tiles={len(all_tiles)}"
    montage = make_montage(all_tiles, all_labels, tile_size, n_cols, header)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / f"{case_id}_{modality}_montage.jpg"
    montage.save(outpath, quality=95)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-manifest", required=True)
    parser.add_argument("--wsi-root", required=True)
    parser.add_argument("--coords-root", required=True)
    parser.add_argument("--modality", choices=["ffpe", "frozen"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--tiles-per-slide", type=int, default=16)
    parser.add_argument("--max-slides-per-patient", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--n-cols", type=int, default=4)
    parser.add_argument("--strategy", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=20260406)
    args = parser.parse_args()

    for idx, case_id in enumerate(args.case_id):
        outpath = export_case_montage(
            case_id=case_id,
            manifest_path=args.patient_manifest,
            wsi_root=args.wsi_root,
            coords_root=args.coords_root,
            modality=args.modality,
            output_dir=args.output_dir,
            tiles_per_slide=args.tiles_per_slide,
            max_slides_per_patient=args.max_slides_per_patient,
            tile_size=args.tile_size,
            n_cols=args.n_cols,
            strategy=args.strategy,
            seed=args.seed + idx,
        )
        print(outpath)


if __name__ == "__main__":
    main()
