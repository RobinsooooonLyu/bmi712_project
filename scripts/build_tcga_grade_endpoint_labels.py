from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


POSITIVE_VPI_PATTERNS = [
    re.compile(r"\bvisceral pleural invasion\s*[:\-]?\s*(present|identified)", re.I),
    re.compile(r"\bpleural invasion\s*[:\-]?\s*(present|identified)", re.I),
    re.compile(r"\b(tumou?r|carcinoma|neoplasm)\b.{0,60}\b(invades|involves|extends into|penetrates|infiltrates|perforates)\b.{0,60}\b(visceral\s+)?pleura\b", re.I),
    re.compile(r"\b(direct extension|extrapulmonary.{0,30}invasion)\s+of\s+tumou?r\s*[:\-]?\s*(visceral\s+)?pleura\b", re.I),
    re.compile(r"\bpleural infiltration\b", re.I),
    re.compile(r"\bPL\s*[123]\b", re.I),
]

NEGATIVE_VPI_PATTERNS = [
    re.compile(r"\bvisceral pleural invasion\s*[:\-]?\s*(not identified|absent|negative|no)\b", re.I),
    re.compile(r"\bno\s+(infiltration|invasion|involvement)\s+of\s+(the\s+)?(visceral\s+)?pleura\b", re.I),
    re.compile(r"\bdoes not (extend|involve|invade).{0,40}\b(visceral\s+)?pleura\b", re.I),
    re.compile(r"\bup to but not (into|through).{0,40}\b(visceral\s+)?pleura\b", re.I),
    re.compile(r"\b(visceral\s+)?pleura\b.{0,50}\b(tumou?r-free|tumor-free|uninvolved)\b", re.I),
]

POSITIVE_LVI_PATTERNS = [
    re.compile(r"\b(lymphovascular|lympho-vascular|angiolymphatic|lymphatic|vascular|blood vessel|blood vessels)\s+(space\s+)?invasion\s*[:\-]?\s*(present|identified|yes)\b", re.I),
    re.compile(r"\b(venous|arterial|lymphatic)\s+\([^)]+vessel\)\s+invasion\s*[:\-]?\s*(present|positive|identified|yes)\b", re.I),
    re.compile(r"\bfocal\s+angiolymphatic\s+invasion\s+identified\b", re.I),
    re.compile(r"\bp[VvLl]\s*[1-9]\b"),
]

NEGATIVE_LVI_PATTERNS = [
    re.compile(r"\b(lymphovascular|lympho-vascular|angiolymphatic|lymphatic|vascular|blood vessel|blood vessels)\s+(space\s+)?invasion\s*[:\-]?\s*(not identified|absent|negative|no)\b", re.I),
    re.compile(r"\bno\s+(unequivocal\s+)?(lymphovascular|lympho-vascular|angiolymphatic|vascular|blood vessel|blood vessels)\s+(space\s+)?invasion\b", re.I),
    re.compile(r"\bp[VL]\s*0\b", re.I),
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def load_case_texts(text_dir: Path) -> dict[str, str]:
    by_case: dict[str, list[str]] = {}
    for path in sorted(text_dir.glob("*.txt")):
        case_id = path.name[:12]
        by_case.setdefault(case_id, []).append(path.read_text(encoding="utf-8", errors="replace"))
    return {case_id: normalize_text("\n".join(texts)) for case_id, texts in by_case.items()}


def context(text: str, start: int, end: int, width: int = 180) -> str:
    left = max(start - width, 0)
    right = min(end + width, len(text))
    return text[left:right].strip()


def looks_negated(text: str, start: int, end: int) -> bool:
    window = text[max(start - 50, 0) : min(end + 80, len(text))].lower()
    negation_terms = [
        "no ",
        "not ",
        "negative",
        "absent",
        "without",
        "no evidence",
        "not identified",
        "does not",
        "not appear",
        "no extension",
        "no involvement",
        "not into",
        "not through",
    ]
    return any(term in window for term in negation_terms)


def first_match(text: str, patterns: list[re.Pattern[str]], skip_negated: bool = False) -> tuple[str, str]:
    for pattern in patterns:
        for match in pattern.finditer(text):
            if skip_negated and looks_negated(text, match.start(), match.end()):
                continue
            return "1", context(text, match.start(), match.end())
    return "0", ""


def classify_marker(
    text: str,
    positive_patterns: list[re.Pattern[str]],
    negative_patterns: list[re.Pattern[str]],
) -> tuple[str, str]:
    negative, negative_evidence = first_match(text, negative_patterns)
    positive, positive_evidence = first_match(text, positive_patterns, skip_negated=True)
    if positive == "1":
        return "positive", positive_evidence
    if negative == "1":
        return "negative", negative_evidence
    return "unknown", ""


def build_rows(
    grade_rows: list[dict[str, str]],
    texts: dict[str, str],
    evidence_rows: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    evidence_rows = evidence_rows or {}
    output = []
    for row in grade_rows:
        case_id = row["case_id"]
        grade = row.get("tumor_grade", "")
        usable_grade = row.get("status") == "parsed" and grade in {"1", "2", "3"}
        text = texts.get(case_id, "")
        vpi_status, vpi_evidence = classify_marker(text, POSITIVE_VPI_PATTERNS, NEGATIVE_VPI_PATTERNS)
        lvi_status, lvi_evidence = classify_marker(text, POSITIVE_LVI_PATTERNS, NEGATIVE_LVI_PATTERNS)

        high_grade = ""
        high_risk = ""
        risk_source = ""
        if usable_grade:
            high_grade = "1" if grade == "3" else "0"
            if grade == "3":
                high_risk = "1"
                risk_source = "grade3"
            elif vpi_status == "positive" or lvi_status == "positive":
                high_risk = "1"
                risk_source = "|".join(
                    source
                    for source, status in (("vpi_positive", vpi_status), ("lvi_positive", lvi_status))
                    if status == "positive"
                )
            else:
                high_risk = "0"
                risk_source = "grade1_2_no_positive_vpi_lvi"

        output.append(
            {
                "case_id": case_id,
                "grade_status": row.get("status", ""),
                "tumor_grade": grade,
                "high_grade": high_grade,
                "vpi_status": vpi_status,
                "lvi_status": lvi_status,
                "high_risk_grade_vpi_lvi": high_risk,
                "high_risk_source": risk_source,
                "grade_evidence": evidence_rows.get(case_id, row).get("evidence_text", ""),
                "vpi_evidence": vpi_evidence,
                "lvi_evidence": lvi_evidence,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grade-labels",
        default="manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_labels.tsv",
    )
    parser.add_argument(
        "--text-dir",
        default="manifests/pathology_reports_luad_paired_full/text",
    )
    parser.add_argument(
        "--out",
        default="manifests/pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv",
    )
    parser.add_argument(
        "--grade-evidence",
        default="manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_evidence.tsv",
    )
    args = parser.parse_args()

    grade_rows = read_tsv(Path(args.grade_labels))
    texts = load_case_texts(Path(args.text_dir))
    evidence_path = Path(args.grade_evidence)
    evidence_rows = {row["case_id"]: row for row in read_tsv(evidence_path)} if evidence_path.exists() else {}
    rows = build_rows(grade_rows, texts, evidence_rows)
    fields = [
        "case_id",
        "grade_status",
        "tumor_grade",
        "high_grade",
        "vpi_status",
        "lvi_status",
        "high_risk_grade_vpi_lvi",
        "high_risk_source",
        "grade_evidence",
        "vpi_evidence",
        "lvi_evidence",
    ]
    write_tsv(Path(args.out), rows, fields)

    usable = [row for row in rows if row["high_grade"] in {"0", "1"}]
    high_grade = sum(row["high_grade"] == "1" for row in usable)
    high_risk = [row for row in rows if row["high_risk_grade_vpi_lvi"] in {"0", "1"}]
    promoted = [
        row
        for row in high_risk
        if row["high_grade"] == "0" and row["high_risk_grade_vpi_lvi"] == "1"
    ]
    print(f"Input cases: {len(rows)}")
    print(f"Usable grade labels: {len(usable)}")
    print(f"High-grade G3 labels: {high_grade}")
    print(f"High-risk labels: {sum(row['high_risk_grade_vpi_lvi'] == '1' for row in high_risk)}")
    print(f"G1/G2 promoted by VPI/LVI: {len(promoted)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
