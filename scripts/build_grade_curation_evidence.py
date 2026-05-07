#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_case_text(row: dict[str, str], text_dir: Path) -> str:
    texts: list[str] = []
    for name in filter(None, row.get("text_files", "").split("|")):
        path = text_dir / name
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return compact("\n".join(texts))


def snippet_for_pattern(text: str, patterns: list[str], radius: int = 260) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        left = max(0, match.start() - radius)
        right = min(len(text), match.end() + radius)
        return compact(text[left:right])
    return compact(text[:700])


def fallback_patterns(row: dict[str, str]) -> list[str]:
    source = row.get("grade_source", "")
    grade = row.get("tumor_grade", "")

    if "missing_pathology_report_form" in source:
        return [r"TCGA Missing Pathology Report Form"]
    if "not_adeno" in source:
        return [r"squamous", r"pleomorphic carcinoma", r"non[- ]small cell carcinoma"]
    if "no_grade" in source or "no_usable" in source or "unreliable" in source:
        return [r"adenocarcinoma", r"histologic type", r"final diagnosis", r"grade"]
    if "mixed_solid" in source:
        return [r"solid[^.]{0,120}acinar", r"adenocarcinoma[^.]{0,220}solid"]
    if "grade3of4" in source or "grade4of4" in source or "grade2to3of4" in source:
        return [r"grade\s*[234][^.\n]{0,80}\(of\s*4\)", r"grade\s*[234]"]
    if "moderate_to_poor" in source:
        return [r"moderately\s+to\s+poorly\s+differentiated", r"moderate\s+to\s+poorly\s+differentiated"]
    if "lepidic" in source or "bac" in source or "bronchiolo" in source or grade == "1":
        return [r"well[- ]differentiated", r"lepidic", r"bronchiolo[- ]?alveolar", r"minimally invasive"]
    if "moderate" in source or "acinar" in source or "papillary" in source or "tubular" in source or grade == "2":
        return [r"moderately[- ]differentiated", r"acinar", r"papillary", r"tubular"]
    if "poor" in source or "solid" in source or "micropapillary" in source or "high_grade" in source or grade == "3":
        return [r"poorly[- ]differentiated", r"solid", r"micropapillary", r"high grade", r"grade\s*(?:III|3|4)"]
    return [r"adenocarcinoma", r"histologic", r"grade", r"diagnosis"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-labels", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    case_path = Path(args.case_labels)
    text_dir = Path(args.text_dir)
    out_path = Path(args.out)

    rows = list(csv.DictReader(case_path.open("r", encoding="utf-8"), delimiter="\t"))
    out_rows: list[dict[str, str]] = []
    for row in rows:
        evidence = compact(row.get("evidence_text", ""))
        text = ""
        source = row.get("grade_source", "")
        if (source.startswith("manual_review_") and source != "manual_review") or not evidence:
            text = read_case_text(row, text_dir)
        if source.startswith("manual_review_") and text:
            candidate = snippet_for_pattern(text, fallback_patterns(row))
            if candidate:
                evidence = candidate
        elif not evidence:
            text = text or read_case_text(row, text_dir)
            evidence = snippet_for_pattern(text, fallback_patterns(row)) if text else ""
        out_rows.append(
            {
                "case_id": row["case_id"],
                "status": row["status"],
                "tumor_grade": row["tumor_grade"],
                "grade_source": row["grade_source"],
                "evidence_text": evidence,
                "text_files": row["text_files"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["case_id", "status", "tumor_grade", "grade_source", "evidence_text", "text_files"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
