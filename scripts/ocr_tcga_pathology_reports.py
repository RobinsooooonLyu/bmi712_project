#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path


def load_case_ids(path: Path, status: str) -> list[str]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8"), delimiter="\t"))
    return [row["case_id"] for row in rows if row.get("case_id") and row.get("status") == status]


def find_case_pdfs(pdf_dir: Path, case_id: str) -> list[Path]:
    return sorted(pdf_dir.glob(f"{case_id}.*.PDF"))


def run_ocr(pdf_path: Path) -> str:
    if not shutil.which("ocrmypdf"):
        raise RuntimeError("ocrmypdf not found")

    with tempfile.TemporaryDirectory(prefix="tcga_ocr_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_pdf = tmpdir_path / f"{pdf_path.stem}.ocr.pdf"
        sidecar = tmpdir_path / f"{pdf_path.stem}.sidecar.txt"
        subprocess.run(
            [
                "ocrmypdf",
                "--force-ocr",
                "--sidecar",
                str(sidecar),
                str(pdf_path),
                str(out_pdf),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return sidecar.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--case-labels", help="Optional case-level grade TSV with status column")
    parser.add_argument("--status", default="missing", help="Status to select from --case-labels")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    text_dir = Path(args.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)

    case_ids: list[str] = []
    if args.case_labels:
        case_ids.extend(load_case_ids(Path(args.case_labels), args.status))
    case_ids.extend(args.case_id)
    case_ids = list(dict.fromkeys(case_ids))
    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    print(f"Target cases: {len(case_ids)}")
    updated = 0
    skipped = 0
    failed = 0

    for case_id in case_ids:
        pdfs = find_case_pdfs(pdf_dir, case_id)
        if not pdfs:
            print(f"[missing-pdf] {case_id}")
            failed += 1
            continue

        for pdf_path in pdfs:
            text_path = text_dir / f"{pdf_path.stem}.txt"
            if text_path.exists() and not args.overwrite:
                print(f"[skip-existing] {text_path.name}")
                skipped += 1
                continue
            try:
                text = run_ocr(pdf_path)
            except Exception as exc:
                print(f"[ocr-failed] {pdf_path.name}: {exc}")
                failed += 1
                continue
            text_path.write_text(text, encoding="utf-8")
            print(f"[ocr-ok] {pdf_path.name}")
            updated += 1

    print(f"Updated text files: {updated}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
