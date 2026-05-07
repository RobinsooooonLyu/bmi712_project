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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-split", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-split", required=True)
    args = parser.parse_args()

    split_rows = read_tsv(Path(args.input_split))
    manifest_rows = read_tsv(Path(args.input_manifest))
    if not split_rows:
        raise RuntimeError("No rows found in input split file.")
    if not manifest_rows:
        raise RuntimeError("No rows found in input manifest file.")

    available_cases = {row["case_id"] for row in manifest_rows}
    kept_rows = [row for row in split_rows if row["case_id"] in available_cases]
    dropped_rows = [row for row in split_rows if row["case_id"] not in available_cases]

    output_path = Path(args.output_split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(output_path, kept_rows, list(split_rows[0].keys()))

    print(f"Input split: {args.input_split}")
    print(f"Input manifest: {args.input_manifest}")
    print(f"Output split: {args.output_split}")
    print(f"Input split rows: {len(split_rows)}")
    print(f"Output split rows: {len(kept_rows)}")
    print(f"Dropped split rows: {len(dropped_rows)}")

    if dropped_rows:
        print("\nDropped cases:")
        for row in dropped_rows[:20]:
            print(row["case_id"], row.get("fold", ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
