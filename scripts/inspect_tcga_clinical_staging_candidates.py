#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


KEYWORDS = (
    "stage",
    "clinical",
    "pathologic",
    "tumor",
    "node",
    "ajcc",
    "t_stage",
    "n_stage",
)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def detect_delimiter(header_line: str) -> str:
    if "\t" in header_line:
        return "\t"
    if "," in header_line:
        return ","
    return "\t"


def inspect_table(path: Path) -> list[str]:
    with open_text(path) as handle:
        first = handle.readline()
        if not first:
            return []
        delimiter = detect_delimiter(first)
        headers = [cell.strip() for cell in first.rstrip("\n").split(delimiter)]
    matches = []
    for header in headers:
        lower = header.lower()
        if any(keyword in lower for keyword in KEYWORDS):
            matches.append(header)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="manifests/clinical_supplement_luad",
        help="Directory containing downloaded or extracted TCGA clinical supplement files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    candidate_paths = sorted(
        [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".tsv", ".csv", ".gz"}
        ]
    )

    print(f"Candidate files scanned: {len(candidate_paths)}")
    for path in candidate_paths:
        matches = inspect_table(path)
        if not matches:
            continue
        print(f"\nFILE: {path}")
        for match in matches:
            print(f"  - {match}")


if __name__ == "__main__":
    main()
