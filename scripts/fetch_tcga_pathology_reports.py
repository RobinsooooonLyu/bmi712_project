#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path


GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data/"
DEFAULT_KEYWORDS = [
    "grade",
    "tumor grade",
    "differentiated",
    "lymphovascular",
    "lymph-vascular",
    "lvi",
    "vascular invasion",
    "blood vessels",
    "pv0",
    "pv1",
    "pv2",
    "visceral pleural",
    "pleural invasion",
    "pleural infiltration",
    "vpi",
    "pl0",
    "pl1",
    "pl2",
    "pl3",
    "stas",
    "spread through air spaces",
]


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def read_case_ids(case_table: Path, limit: int | None = None) -> list[str]:
    with case_table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        case_ids = [row["case_id"] for row in reader if row.get("case_id")]
    if limit is not None:
        return case_ids[:limit]
    return case_ids


def query_pathology_report_files(project_id: str, case_ids: list[str] | None = None) -> list[dict]:
    batches = chunked(case_ids, 100) if case_ids else [None]
    merged_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for batch in batches:
        merged_rows.extend(_query_pathology_report_files_batch(project_id, batch, seen))
    merged_rows.sort(key=lambda row: (row["cases"][0]["submitter_id"], row["file_name"]))
    return merged_rows


def _query_pathology_report_files_batch(
    project_id: str,
    case_ids: list[str] | None,
    seen: set[tuple[str, str]],
) -> list[dict]:
    filters_content: list[dict] = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Pathology Report"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
    ]
    if case_ids:
        filters_content.append(
            {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}}
        )
    filters = {"op": "and", "content": filters_content}
    fields = [
        "file_id",
        "file_name",
        "data_format",
        "file_size",
        "state",
        "cases.submitter_id",
        "cases.project.project_id",
    ]
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "2000",
    }
    url = GDC_FILES_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    rows = payload["data"]["hits"]
    kept: list[dict] = []
    for row in rows:
        key = (row["cases"][0]["submitter_id"], row["file_id"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def write_listing(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["case_id", "file_id", "file_name", "data_format", "file_size", "state"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["cases"][0]["submitter_id"],
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


def maybe_extract_text(pdf_path: Path) -> str | None:
    pdftotext_bin = shutil.which("pdftotext")
    if pdftotext_bin:
        try:
            result = subprocess.run(
                [pdftotext_bin, "-layout", str(pdf_path), "-"],
                check=True,
                capture_output=True,
                text=True,
            )
            text = result.stdout.strip()
            if text:
                return text
        except Exception:
            pass

    try:
        from pypdf import PdfReader
    except Exception:
        return None

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages.append(text)
    return "\n".join(pages).strip()


def find_keyword_hits(text: str, keywords: list[str]) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    compact = re.sub(r"\s+", " ", text)
    for keyword in keywords:
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])", re.IGNORECASE)
        match = pattern.search(compact)
        if not match:
            continue
        idx = match.start()
        left = max(0, idx - 120)
        right = min(len(compact), match.end() + 120)
        snippet = compact[left:right]
        hits.append((keyword, snippet))
    return hits


def write_hits(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["case_id", "file_name", "keyword", "snippet"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="TCGA-LUAD")
    parser.add_argument("--outdir", default="manifests/pathology_reports_luad")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract-text", action="store_true")
    parser.add_argument("--case-table", help="Optional TSV with case_id column to restrict to")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[], help="Explicit TCGA case ID")
    parser.add_argument("--keyword", action="append", default=[])
    args = parser.parse_args()

    case_ids: list[str] = []
    if args.case_table:
        case_ids.extend(read_case_ids(Path(args.case_table), limit=args.limit))
    if args.case_id:
        case_ids.extend(args.case_id)
    case_ids = list(dict.fromkeys(case_ids))
    if args.limit is not None and not args.case_table:
        case_ids = case_ids[: args.limit]

    rows = query_pathology_report_files(args.project, case_ids if case_ids else None)
    if case_ids:
        wanted = set(case_ids)
        rows = [row for row in rows if row["cases"][0]["submitter_id"] in wanted]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    listing_path = outdir / f"{args.project.lower().replace('-', '_')}_pathology_report_files.tsv"
    write_listing(listing_path, rows)

    print(f"Project: {args.project}")
    print(f"Pathology report files: {len(rows)}")
    print(f"Listing: {listing_path}")

    if not args.download:
        return 0

    pdf_dir = outdir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir = outdir / "text"
    if args.extract_text:
        text_dir.mkdir(parents=True, exist_ok=True)

    keywords = args.keyword or DEFAULT_KEYWORDS
    hit_rows: list[dict] = []

    for row in rows:
        case_id = row["cases"][0]["submitter_id"]
        pdf_path = pdf_dir / row["file_name"]
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"Skipping existing {pdf_path.name}")
        else:
            print(f"Downloading {row['file_name']}")
            download_file(row["file_id"], pdf_path)

        if not args.extract_text:
            continue

        text = maybe_extract_text(pdf_path)
        if text is None:
            print("pypdf is not installed; skipping text extraction", file=sys.stderr)
            continue
        text_path = text_dir / f"{pdf_path.stem}.txt"
        text_path.write_text(text, encoding="utf-8")
        for keyword, snippet in find_keyword_hits(text, keywords):
            hit_rows.append(
                {
                    "case_id": case_id,
                    "file_name": row["file_name"],
                    "keyword": keyword,
                    "snippet": snippet,
                }
            )

    if args.extract_text and hit_rows:
        hits_path = outdir / "keyword_hits.tsv"
        write_hits(hits_path, hit_rows)
        print(f"Keyword hits: {hits_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
