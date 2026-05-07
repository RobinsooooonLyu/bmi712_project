#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def effective_um(payload: dict) -> float | None:
    level_mpp = payload.get("selected_level_mpp")
    tile_size_level = payload.get("tile_size_level")
    if level_mpp is None or tile_size_level is None:
        return None
    return float(level_mpp) * int(tile_size_level)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-dir", required=True)
    parser.add_argument("--coords-dir", required=True)
    parser.add_argument("--label", default="coords")
    parser.add_argument("--target-um", type=float, default=112.0)
    parser.add_argument("--show-missing", type=int, default=10)
    parser.add_argument("--show-samples", type=int, default=5)
    args = parser.parse_args()

    wsi_dir = Path(args.wsi_dir)
    coords_dir = Path(args.coords_dir)

    slide_paths = sorted(wsi_dir.glob("*.svs"))
    coord_paths = sorted(coords_dir.glob("*.json"))
    coord_by_name = {path.name.removesuffix(".json"): path for path in coord_paths}

    missing = [path.name for path in slide_paths if path.name not in coord_by_name]
    extra = [path.name for path in coord_paths if path.name.removesuffix(".json") not in {p.name for p in slide_paths}]

    num_tiles = []
    effective_ums = []
    zero_tile = []
    bad_meta = []
    samples = []

    for slide_path in slide_paths:
        coord_path = coord_by_name.get(slide_path.name)
        if coord_path is None:
            continue
        payload = json.loads(coord_path.read_text())
        n_tiles = int(payload.get("num_tiles", 0))
        num_tiles.append(n_tiles)
        if n_tiles == 0:
            zero_tile.append(slide_path.name)
        eff = effective_um(payload)
        if eff is None:
            bad_meta.append(slide_path.name)
        else:
            effective_ums.append(eff)
        if len(samples) < args.show_samples:
            samples.append(
                {
                    "slide": slide_path.name,
                    "num_tiles": n_tiles,
                    "selected_level_mpp": payload.get("selected_level_mpp"),
                    "tile_size_level": payload.get("tile_size_level"),
                    "effective_um": eff,
                    "mask_level_mpp": payload.get("mask_level_mpp"),
                }
            )

    print(f"Label: {args.label}")
    print(f"WSI dir: {wsi_dir}")
    print(f"Coords dir: {coords_dir}")
    print(f"Slides found: {len(slide_paths)}")
    print(f"Coord JSONs found: {len(coord_paths)}")
    print(f"Missing JSONs: {len(missing)}")
    print(f"Extra JSONs: {len(extra)}")
    print(f"Zero-tile JSONs: {len(zero_tile)}")
    print(f"Bad metadata JSONs: {len(bad_meta)}")

    if num_tiles:
        print(
            "Tile count summary: "
            f"min={min(num_tiles)} "
            f"median={statistics.median(num_tiles):.1f} "
            f"mean={statistics.mean(num_tiles):.1f} "
            f"max={max(num_tiles)}"
        )
    if effective_ums:
        abs_errors = [abs(x - args.target_um) for x in effective_ums]
        print(
            "Effective FOV summary (um): "
            f"min={min(effective_ums):.2f} "
            f"median={statistics.median(effective_ums):.2f} "
            f"mean={statistics.mean(effective_ums):.2f} "
            f"max={max(effective_ums):.2f}"
        )
        print(
            "Absolute error vs target 112um: "
            f"min={min(abs_errors):.3f} "
            f"median={statistics.median(abs_errors):.3f} "
            f"mean={statistics.mean(abs_errors):.3f} "
            f"max={max(abs_errors):.3f}"
        )

    if missing:
        print("\nMissing JSON examples:")
        for name in missing[: args.show_missing]:
            print(name)

    if zero_tile:
        print("\nZero-tile examples:")
        for name in zero_tile[: args.show_missing]:
            print(name)

    if bad_meta:
        print("\nBad-metadata examples:")
        for name in bad_meta[: args.show_missing]:
            print(name)

    if samples:
        print("\nSample JSON summaries:")
        for sample in samples:
            print(sample)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
