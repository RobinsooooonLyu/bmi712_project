#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path


GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data/"


def query_clinical_files(project_id: str) -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Clinical Supplement"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    fields = [
        "file_id",
        "file_name",
        "data_format",
        "data_type",
        "access",
        "file_size",
        "state",
        "cases.project.project_id",
    ]
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "1000",
    }
    url = GDC_FILES_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    return payload["data"]["hits"]


def write_listing(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["file_id", "file_name", "data_format", "file_size", "state"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_id": row["file_id"],
                    "file_name": row["file_name"],
                    "data_format": row.get("data_format", ""),
                    "file_size": row.get("file_size", ""),
                    "state": row.get("state", ""),
                }
            )


def download_file(file_id: str, out_path: Path) -> None:
    with urllib.request.urlopen(GDC_DATA_URL + file_id, timeout=120) as response:
        content = response.read()
    out_path.write_bytes(content)


def maybe_extract_archive(path: Path, extract_dir: Path) -> None:
    if tarfile.is_tarfile(path):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as archive:
            archive.extractall(extract_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="TCGA-LUAD")
    parser.add_argument(
        "--outdir",
        default="manifests/clinical_supplement_luad",
        help="Output directory for the listing and optional downloads",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the returned clinical supplement files into outdir/raw",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="If downloaded files are tar archives, extract them into outdir/extracted",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = query_clinical_files(args.project)
    listing_path = outdir / f"{args.project.lower().replace('-', '_')}_clinical_supplement_files.tsv"
    write_listing(listing_path, rows)

    print(f"Project: {args.project}")
    print(f"Clinical supplement files: {len(rows)}")
    print(f"Listing: {listing_path}")

    if not args.download:
        return

    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        out_path = raw_dir / row["file_name"]
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"Skipping existing {out_path.name}")
        else:
            print(f"Downloading {row['file_name']}")
            download_file(row["file_id"], out_path)
        if args.extract:
            maybe_extract_archive(out_path, outdir / "extracted" / out_path.stem)


if __name__ == "__main__":
    main()
