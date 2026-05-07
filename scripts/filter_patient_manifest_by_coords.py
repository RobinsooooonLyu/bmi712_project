#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def available_slide_names(coords_dir: Path) -> set[str]:
    return {path.name.removesuffix(".json") for path in coords_dir.glob("*.json")}


def filter_slide_list(slide_list: str, available: set[str]) -> list[str]:
    names = [name for name in slide_list.split("|") if name]
    return [name for name in names if name in available]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--ffpe-coords-dir", required=True)
    parser.add_argument("--frozen-coords-dir", required=True)
    parser.add_argument(
        "--require-both-modalities",
        action="store_true",
        help="Drop patients unless at least one FFPE and one frozen slide remain.",
    )
    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    output_manifest = Path(args.output_manifest)
    ffpe_available = available_slide_names(Path(args.ffpe_coords_dir))
    frozen_available = available_slide_names(Path(args.frozen_coords_dir))

    rows = read_tsv(input_manifest)
    if not rows:
        raise RuntimeError(f"No rows found in {input_manifest}")

    kept_rows: list[dict[str, str]] = []
    dropped_cases: list[tuple[str, str]] = []
    ffpe_slide_drops = 0
    frozen_slide_drops = 0

    for row in rows:
        ffpe_before = [name for name in row.get("ffpe_filenames", "").split("|") if name]
        frozen_before = [name for name in row.get("frozen_filenames", "").split("|") if name]
        ffpe_after = filter_slide_list(row.get("ffpe_filenames", ""), ffpe_available)
        frozen_after = filter_slide_list(row.get("frozen_filenames", ""), frozen_available)

        ffpe_slide_drops += len(ffpe_before) - len(ffpe_after)
        frozen_slide_drops += len(frozen_before) - len(frozen_after)

        if args.require_both_modalities:
            if not ffpe_after and not frozen_after:
                dropped_cases.append((row["case_id"], "no_ffpe_or_frozen_after_filter"))
                continue
            if not ffpe_after:
                dropped_cases.append((row["case_id"], "no_ffpe_after_filter"))
                continue
            if not frozen_after:
                dropped_cases.append((row["case_id"], "no_frozen_after_filter"))
                continue

        new_row = dict(row)
        new_row["ffpe_filenames"] = "|".join(ffpe_after)
        new_row["frozen_filenames"] = "|".join(frozen_after)
        kept_rows.append(new_row)

    fieldnames = list(rows[0].keys())
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(output_manifest, kept_rows, fieldnames)

    print(f"Input manifest: {input_manifest}")
    print(f"Output manifest: {output_manifest}")
    print(f"Input cases: {len(rows)}")
    print(f"Output cases: {len(kept_rows)}")
    print(f"Dropped cases: {len(dropped_cases)}")
    print(f"Dropped FFPE slides: {ffpe_slide_drops}")
    print(f"Dropped frozen slides: {frozen_slide_drops}")

    if dropped_cases:
        print("\nDropped case examples:")
        for case_id, reason in dropped_cases[:20]:
            print(f"{case_id}\t{reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
