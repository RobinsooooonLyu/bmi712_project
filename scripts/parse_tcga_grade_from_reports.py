#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


GRADE_CODE_RE = re.compile(r"\bG\s*([123])\b", re.IGNORECASE)
WELL_RE = re.compile(r"\bwell\s+differentiated\b", re.IGNORECASE)
MOD_RE = re.compile(r"\bmoderately\s+differentiated\b", re.IGNORECASE)
POOR_RE = re.compile(r"\bpoor(?:ly)?\s+differentiated\b", re.IGNORECASE)

MANUAL_CASE_OVERRIDES: dict[str, tuple[str, str]] = {
    "TCGA-67-3773": ("2", "manual_review_resection_over_fna"),
    "TCGA-69-7978": ("3", "manual_review_intraop_and_final_patterns"),
    "TCGA-91-A4BC": ("3", "manual_review_moderate_to_poor"),
    "TCGA-95-7043": ("3", "manual_review_multifocal_prefer_high_grade"),
    "TCGA-MP-A4T6": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4TD": ("3", "manual_review_grade2to3of4"),
    "TCGA-MP-A4TF": ("3", "manual_review_grade4of4"),
    "TCGA-NJ-A55A": ("2", "manual_review_moderately_differentiated"),
    "TCGA-MP-A4SY": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4T2": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4TE": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4TI": ("3", "manual_review_grade4of4"),
    "TCGA-MP-A4TJ": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4TK": ("3", "manual_review_grade4of4"),
    "TCGA-05-4384": ("2", "manual_review_tubulopapillary_bronchioloalveolar"),
    "TCGA-05-4396": ("1", "manual_review_predominant_bronchioloalveolar"),
    "TCGA-05-4390": ("2", "manual_review_predominant_acinar_smaller_solid"),
    "TCGA-05-4405": ("2", "manual_review_acinar_papillary"),
    "TCGA-05-4402": ("2", "manual_review_predominant_acinar_smaller_solid"),
    "TCGA-05-4403": ("1", "manual_review_predominant_bronchioloalveolar_small_solid"),
    "TCGA-05-4410": ("2", "manual_review_predominant_tubular_acinar"),
    "TCGA-05-4415": ("3", "manual_review_polymorphocellular_giant_cell"),
    "TCGA-05-4420": ("2", "manual_review_predominant_papillary_small_acinar"),
    "TCGA-05-4425": ("2", "manual_review_tubulopapillary_tubular"),
    "TCGA-05-4430": ("2", "manual_review_predominant_acinar_smaller_solid"),
    "TCGA-44-6148": ("1", "manual_review_well_differentiated_mucinous"),
    "TCGA-44-7672": ("2", "manual_review_histologic_grade_moderate"),
    "TCGA-49-4487": ("2", "manual_review_moderately_differentiated_ocr"),
    "TCGA-49-4494": ("2", "manual_review_moderately_differentiated_mucinous"),
    "TCGA-49-4501": ("2", "manual_review_moderately_differentiated_with_bac"),
    "TCGA-49-4507": ("3", "manual_review_poorly_differentiated"),
    "TCGA-49-4510": ("2", "manual_review_moderately_differentiated_papillary"),
    "TCGA-55-6986": ("2", "manual_review_grade1to2_well_to_moderate"),
    "TCGA-55-7815": ("1", "manual_review_well_differentiated"),
    "TCGA-64-1676": ("3", "manual_review_predominant_solid_necrosis"),
    "TCGA-64-1677": ("3", "manual_review_acinar_grade_iii"),
    "TCGA-64-1681": ("1", "manual_review_well_differentiated"),
    "TCGA-64-5775": ("3", "manual_review_high_grade_neoplasm_primary_lung_adeno"),
    "TCGA-69-7763": ("2", "manual_review_mixed_papillary_bac_acinar"),
    "TCGA-69-7764": ("2", "manual_review_acinar_bac_components"),
    "TCGA-69-7980": ("1", "manual_review_lepidic_pattern"),
    "TCGA-69-8254": ("2", "manual_review_papillary_predominant_minor_micropapillary"),
    "TCGA-75-6212": ("3", "manual_review_micropapillary_type"),
    "TCGA-75-7031": ("1", "manual_review_bac_features"),
    "TCGA-78-7540": ("1", "manual_review_well_differentiated_predominant_bac"),
    "TCGA-86-7714": ("3", "manual_review_poorly_differentiated"),
    "TCGA-97-A4M1": ("1", "manual_review_lepidic_predominant"),
    "TCGA-J2-A4AE": ("1", "manual_review_minimally_invasive_adeno"),
    "TCGA-MP-A4SV": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4SW": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4T4": ("3", "manual_review_grade4of4"),
    "TCGA-MP-A4T8": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4T9": ("3", "manual_review_grade3of4"),
    "TCGA-MP-A4TH": ("1", "manual_review_prominent_lepidic_minimally_invasive"),
    "TCGA-MP-A5C7": ("2", "manual_review_grade2of4"),
    "TCGA-O1-A52J": ("1", "manual_review_well_differentiated"),
}

MANUAL_CASE_EXCLUDE: dict[str, str] = {
    "TCGA-05-4417": "manual_review_ambiguous_exclude",
    "TCGA-05-4418": "manual_review_ambiguous_exclude",
    "TCGA-05-4424": "manual_review_ambiguous_exclude",
    "TCGA-05-4426": "manual_review_ambiguous_exclude",
    "TCGA-35-3615": "manual_review_no_usable_text_exclude",
    "TCGA-44-2659": "manual_review_no_grade_exclude",
    "TCGA-49-4488": "manual_review_no_grade_exclude",
    "TCGA-49-4505": "manual_review_multifocal_exclude",
    "TCGA-50-5066": "manual_review_not_pure_adeno_exclude",
    "TCGA-55-5899": "manual_review_no_grade_exclude",
    "TCGA-55-6543": "manual_review_no_grade_exclude",
    "TCGA-55-6642": "manual_review_no_grade_exclude",
    "TCGA-55-7910": "manual_review_no_grade_exclude",
    "TCGA-05-5420": "manual_review_missing_pathology_report_form_exclude",
    "TCGA-05-5423": "manual_review_missing_pathology_report_form_exclude",
    "TCGA-05-5425": "manual_review_missing_pathology_report_form_exclude",
    "TCGA-05-5428": "manual_review_missing_pathology_report_form_exclude",
    "TCGA-05-5429": "manual_review_missing_pathology_report_form_exclude",
    "TCGA-35-4122": "manual_review_no_usable_text_exclude",
    "TCGA-35-5375": "manual_review_not_adeno_exclude",
    "TCGA-49-4506": "manual_review_no_tumor_diagnosis_exclude",
    "TCGA-49-4512": "manual_review_no_grade_exclude",
    "TCGA-49-4514": "manual_review_no_grade_exclude",
    "TCGA-49-6761": "manual_review_no_usable_text_exclude",
    "TCGA-49-6767": "manual_review_no_usable_text_exclude",
    "TCGA-50-5946": "manual_review_tcga_form_no_grade_exclude",
    "TCGA-55-8508": "manual_review_no_grade_exclude",
    "TCGA-64-1679": "manual_review_no_grade_exclude",
    "TCGA-69-7974": "manual_review_mixed_solid_acinar_bac_no_predominance_exclude",
    "TCGA-71-6725": "manual_review_no_reliable_grade_exclude",
    "TCGA-71-8520": "manual_review_unselected_differentiation_checklist_exclude",
    "TCGA-75-5122": "manual_review_unreliable_form_ocr_exclude",
    "TCGA-75-5125": "manual_review_unreliable_form_ocr_exclude",
    "TCGA-75-5126": "manual_review_unreliable_form_ocr_exclude",
    "TCGA-75-5146": "manual_review_unreliable_form_ocr_exclude",
    "TCGA-75-5147": "manual_review_no_grade_exclude",
    "TCGA-75-6203": "manual_review_unreliable_form_ocr_exclude",
    "TCGA-75-6205": "manual_review_unreliable_form_ocr_exclude",
    "TCGA-75-6206": "manual_review_unreliable_numeric_table_exclude",
    "TCGA-75-6207": "manual_review_unreliable_numeric_table_exclude",
    "TCGA-75-6211": "manual_review_unreliable_numeric_table_exclude",
    "TCGA-75-7025": "manual_review_unreliable_form_ocr_exclude",
    "TCGA-75-7027": "manual_review_unreliable_form_ocr_exclude",
    "TCGA-75-7030": "manual_review_unreliable_numeric_table_exclude",
    "TCGA-80-5611": "manual_review_no_grade_exclude",
    "TCGA-J2-8192": "manual_review_no_grade_exclude",
    "TCGA-J2-A4AD": "manual_review_no_grade_exclude",
}


def read_case_ids(case_table: Path) -> set[str]:
    with case_table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["case_id"] for row in reader if row.get("case_id")}


def iter_text_files(text_dir: Path):
    for path in sorted(text_dir.glob("*.txt")):
        yield path


def extract_case_id(path: Path) -> str:
    return path.stem.split(".", 1)[0]


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_snippet(text: str, pattern: re.Pattern[str], radius: int = 180) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    left = max(0, match.start() - radius)
    right = min(len(text), match.end() + radius)
    return text[left:right].strip()


def basic_parse_one(text: str, extractor: str) -> dict[str, str]:
    code_matches = GRADE_CODE_RE.findall(text)
    code_unique = sorted(set(code_matches))

    word_hits: list[tuple[str, str, re.Pattern[str]]] = []
    if WELL_RE.search(text):
        word_hits.append(("1", "well_differentiated", WELL_RE))
    if MOD_RE.search(text):
        word_hits.append(("2", "moderately_differentiated", MOD_RE))
    if POOR_RE.search(text):
        word_hits.append(("3", "poorly_differentiated", POOR_RE))

    word_unique = sorted({grade for grade, _, _ in word_hits})
    code_grade = code_unique[0] if len(code_unique) == 1 else ""
    word_grade = word_unique[0] if len(word_unique) == 1 else ""

    if code_grade and word_grade and code_grade != word_grade:
        status = "conflict"
        final_grade = ""
        source = "conflict"
        snippet = ""
    elif code_grade:
        status = "parsed"
        final_grade = code_grade
        source = "g_code"
        snippet = first_snippet(text, GRADE_CODE_RE)
    elif word_grade:
        status = "parsed"
        final_grade = word_grade
        source = "differentiation_text"
        snippet = ""
        for _, _, pattern in word_hits:
            snippet = first_snippet(text, pattern)
            if snippet:
                break
    elif len(code_unique) > 1 or len(word_unique) > 1:
        status = "ambiguous"
        final_grade = ""
        source = "ambiguous"
        snippet = ""
    else:
        status = "missing"
        final_grade = ""
        source = ""
        snippet = ""

    return {
        "status": status,
        "tumor_grade": final_grade,
        "grade_source": source,
        "evidence_extractor": extractor,
        "evidence_text": snippet,
        "code_candidates": "|".join(code_unique),
        "word_candidates": "|".join(sorted({label for _, label, _ in word_hits})),
    }


def merge_basic_results(results: list[dict[str, str]]) -> dict[str, str]:
    parsed = [r for r in results if r["status"] == "parsed" and r["tumor_grade"]]
    parsed_grades = sorted({r["tumor_grade"] for r in parsed})
    if len(parsed_grades) == 1:
        best = max(parsed, key=lambda r: (len(r["evidence_text"]), r["grade_source"]))
        return {
            **best,
            "all_grade_evidence": "",
            "range_evidence": "",
        }
    if len(parsed_grades) > 1:
        return {
            "status": "conflict",
            "tumor_grade": "",
            "grade_source": "basic_multi_grade_conflict",
            "evidence_extractor": "",
            "evidence_text": " || ".join(
                f"{r['tumor_grade']}:{r['grade_source']}:{r['evidence_extractor']}:{r['evidence_text']}"
                for r in parsed[:6]
            ),
            "all_grade_evidence": "",
            "range_evidence": "",
            "code_candidates": " || ".join(r["code_candidates"] for r in results if r["code_candidates"]),
            "word_candidates": " || ".join(r["word_candidates"] for r in results if r["word_candidates"]),
        }
    if any(r["status"] == "conflict" for r in results):
        r = next(r for r in results if r["status"] == "conflict")
        return {
            **r,
            "all_grade_evidence": "",
            "range_evidence": "",
        }
    if any(r["status"] == "ambiguous" for r in results):
        r = next(r for r in results if r["status"] == "ambiguous")
        return {
            **r,
            "all_grade_evidence": "",
            "range_evidence": "",
        }
    return {
        "status": "missing",
        "tumor_grade": "",
        "grade_source": "",
        "evidence_extractor": "",
        "evidence_text": "",
        "all_grade_evidence": "",
        "range_evidence": "",
        "code_candidates": "",
        "word_candidates": "",
    }


def extract_pypdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""

    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages.append(text)
    return compact("\n".join(pages))


def extract_pdftotext_text(pdf_path: Path) -> str:
    pdftotext_bin = shutil.which("pdftotext")
    if not pdftotext_bin:
        return ""
    try:
        result = subprocess.run(
            [pdftotext_bin, "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return compact(result.stdout)


@dataclass(frozen=True)
class Evidence:
    grade: str
    priority: int
    rule: str
    snippet: str
    extractor: str


EXPLICIT_RULES: list[tuple[str, re.Pattern[str], str, int]] = [
    (
        "explicit_grade3_phrase",
        re.compile(r"\b(?:grade\s*3|g\s*3)\b[^.]{0,120}\bpoor(?:ly)?\s+differentiated\b", re.I),
        "3",
        120,
    ),
    (
        "explicit_grade2_phrase",
        re.compile(r"\b(?:grade\s*2|g\s*2)\b[^.]{0,120}\bmoderately\s+differentiated\b", re.I),
        "2",
        120,
    ),
    (
        "explicit_grade1_phrase",
        re.compile(r"\b(?:grade\s*1|g\s*1)\b[^.]{0,120}\bwell\s+differentiated\b", re.I),
        "1",
        120,
    ),
    (
        "explicit_phrase_grade3",
        re.compile(
            r"\bpoor(?:ly)?\s+differentiated\b(?!\s*\(g?\s*2\s*[-/]\s*g?\s*3\))[^.]{0,120}\b(?:grade\s*3|g\s*3)\b",
            re.I,
        ),
        "3",
        120,
    ),
    (
        "explicit_phrase_grade2",
        re.compile(
            r"\bmoderately\s+differentiated\b(?!\s*[-/]|(?:\s+to\s+poor(?:ly)?\s+differentiated))[^.]{0,120}\b(?:grade\s*2|g\s*2)\b",
            re.I,
        ),
        "2",
        120,
    ),
    (
        "explicit_phrase_grade1",
        re.compile(r"\bwell\s+differentiated\b[^.]{0,120}\b(?:grade\s*1|g\s*1)\b", re.I),
        "1",
        120,
    ),
    (
        "predominantly_poorly_differentiated",
        re.compile(r"\bpredominantly\s+poor(?:ly)?\s+differentiated\b", re.I),
        "3",
        115,
    ),
    (
        "predominantly_moderately_differentiated",
        re.compile(r"\bpredominantly\s+moderately\s+differentiated\b", re.I),
        "2",
        115,
    ),
    (
        "predominantly_well_differentiated",
        re.compile(r"\bpredominantly\s+well\s+differentiated\b", re.I),
        "1",
        115,
    ),
    (
        "poorly_differentiated_adenocarcinoma",
        re.compile(r"\bpoor(?:ly)?\s+differentiated(?:\s+\w+){0,3}\s+adenocarcinoma\b", re.I),
        "3",
        110,
    ),
    (
        "moderately_differentiated_adenocarcinoma",
        re.compile(r"\bmoderately\s+differentiated(?:\s+\w+){0,3}\s+adenocarcinoma\b", re.I),
        "2",
        110,
    ),
    (
        "well_differentiated_adenocarcinoma",
        re.compile(r"\bwell\s+differentiated(?:\s+\w+){0,3}\s+adenocarcinoma\b", re.I),
        "1",
        110,
    ),
    (
        "histologic_grade_text_g3",
        re.compile(r"\bhistologic\s+grade\b[^.\n]{0,80}\bpoor(?:ly)?\s+differentiated\b", re.I),
        "3",
        109,
    ),
    (
        "histologic_grade_text_g2",
        re.compile(r"\bhistologic\s+grade\b[^.\n]{0,80}\bmoderately\s+differentiated\b", re.I),
        "2",
        109,
    ),
    (
        "histologic_grade_text_g1",
        re.compile(r"\bhistologic\s+grade\b[^.\n]{0,80}\bwell\s+differentiated\b", re.I),
        "1",
        109,
    ),
    (
        "histologic_grade_text_g2_range",
        re.compile(r"\bhistologic\s+grade\b[^.\n]{0,120}\bgrade\s*i+\s*[-/]\s*ii\b[^.\n]{0,80}\bwell\s+to\s+moderately\s+differentiated\b", re.I),
        "2",
        109,
    ),
    (
        "grade3_summary",
        re.compile(r"\badenocarcinoma,\s*grade\s*3\b", re.I),
        "3",
        108,
    ),
    (
        "grade2_summary",
        re.compile(r"\badenocarcinoma,\s*grade\s*2\b", re.I),
        "2",
        108,
    ),
    (
        "grade1_summary",
        re.compile(r"\badenocarcinoma,\s*grade\s*1\b", re.I),
        "1",
        108,
    ),
    (
        "frozen_poorly_differentiated_nsc",
        re.compile(r"\bpoor(?:ly)?\s+differentiated\s+non[- ]small\s+cell\s+carcinoma\b", re.I),
        "3",
        100,
    ),
    (
        "predominant_solid_pattern",
        re.compile(
            r"\b(?:very\s+|markedly\s+)?predominan\w*\s+(?:large-cell[, ]+|polymorphocellular[, ]+|partly\s+clear-cell[, ]+){0,4}solid\b"
            r"|\bsolid\s+predominan\w*\b",
            re.I,
        ),
        "3",
        98,
    ),
    (
        "predominant_micropapillary_pattern",
        re.compile(r"\b(?:very\s+|markedly\s+)?predominan\w*\s+micropapillary\b|\bmicropapillary\s+predominan\w*\b", re.I),
        "3",
        98,
    ),
    (
        "predominant_complex_gland_pattern",
        re.compile(r"\b(?:very\s+)?predominan\w*[^.]{0,80}\bcomplex\s+gland\w*\b", re.I),
        "3",
        98,
    ),
    (
        "predominant_polymorphocellular_pattern",
        re.compile(r"\b(?:very\s+)?predominan\w*[^.]{0,80}\bpolymorphocellular\b", re.I),
        "3",
        98,
    ),
    (
        "predominant_tubular_pattern",
        re.compile(r"\b(?:very\s+)?predominan\w*[^.]{0,100}\btubular\b", re.I),
        "2",
        96,
    ),
    (
        "predominant_acinar_pattern",
        re.compile(r"\b(?:very\s+)?predominan\w*[^.]{0,100}\bacinar\b", re.I),
        "2",
        96,
    ),
    (
        "predominant_papillary_pattern",
        re.compile(r"\b(?:very\s+)?predominan\w*[^.]{0,100}\bpapillary\b", re.I),
        "2",
        96,
    ),
    (
        "predominant_tubulopapillary_pattern",
        re.compile(r"\b(?:very\s+)?predominan\w*[^.]{0,100}\btubulopapillary\b", re.I),
        "2",
        96,
    ),
    (
        "predominant_bronchioloalveolar_pattern",
        re.compile(r"\b(?:very\s+)?predominan\w*[^.]{0,100}\bbronchiolo?alveolar\b", re.I),
        "1",
        97,
    ),
]

RANGE_RULES: list[tuple[str, re.Pattern[str], int]] = [
    (
        "moderate_to_poor_range",
        re.compile(r"\bmoderate(?:ly)?\s+to\s+poor(?:ly)?\s+differentiated\b", re.I),
        90,
    ),
    (
        "g2_g3_range",
        re.compile(r"\bG\s*2\s*[-/]\s*G?\s*3\b", re.I),
        90,
    ),
]

CONTEXTUAL_GRADE_RULES: list[tuple[str, re.Pattern[str], str, int]] = [
    (
        "histologic_grade_g3",
        re.compile(r"\bhistologic\s+grade\b[^.]{0,80}\b(?:grade\s*3|g\s*3)\b", re.I),
        "3",
        82,
    ),
    (
        "histologic_grade_g2",
        re.compile(r"\bhistologic\s+grade\b[^.]{0,80}\b(?:grade\s*2|g\s*2)\b", re.I),
        "2",
        82,
    ),
    (
        "histologic_grade_g1",
        re.compile(r"\bhistologic\s+grade\b[^.]{0,80}\b(?:grade\s*1|g\s*1)\b", re.I),
        "1",
        82,
    ),
    (
        "tumor_classification_g3",
        re.compile(r"\btumor\s+classification\b[^.]{0,80}\b(?:grade\s*3|g\s*3)\b", re.I),
        "3",
        78,
    ),
    (
        "tumor_classification_g2",
        re.compile(r"\btumor\s+classification\b[^.]{0,80}\b(?:grade\s*2|g\s*2)\b", re.I),
        "2",
        78,
    ),
    (
        "tumor_classification_g1",
        re.compile(r"\btumor\s+classification\b[^.]{0,80}\b(?:grade\s*1|g\s*1)\b", re.I),
        "1",
        78,
    ),
    (
        "who_classification_g1",
        re.compile(r"\bwho\s+classification\b[^.]{0,160}\bG\s*1\b", re.I),
        "1",
        78,
    ),
    (
        "who_classification_g2",
        re.compile(r"\bwho\s+classification\b[^.]{0,160}\bG\s*2\b", re.I),
        "2",
        78,
    ),
    (
        "who_classification_g3",
        re.compile(r"\bwho\s+classification\b[^.]{0,160}\bG\s*3\b", re.I),
        "3",
        78,
    ),
]


def collect_evidence(text: str, extractor: str) -> tuple[list[Evidence], list[Evidence]]:
    evidences: list[Evidence] = []
    range_hits: list[Evidence] = []

    for rule_name, pattern, grade, priority in EXPLICIT_RULES:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 220)].strip()
            evidences.append(Evidence(grade=grade, priority=priority, rule=rule_name, snippet=snippet, extractor=extractor))

    for rule_name, pattern, priority in RANGE_RULES:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 220)].strip()
            range_hits.append(Evidence(grade="", priority=priority, rule=rule_name, snippet=snippet, extractor=extractor))

    for rule_name, pattern, grade, priority in CONTEXTUAL_GRADE_RULES:
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 180) : min(len(text), match.end() + 220)].strip()
            evidences.append(Evidence(grade=grade, priority=priority, rule=rule_name, snippet=snippet, extractor=extractor))

    return evidences, range_hits


def dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str, str, str]] = set()
    kept: list[Evidence] = []
    for item in items:
        key = (item.grade, item.rule, item.extractor, item.snippet)
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def choose_grade(evidences: list[Evidence], range_hits: list[Evidence]) -> dict[str, str]:
    evidences = dedupe_evidence(evidences)
    range_hits = dedupe_evidence(range_hits)

    if evidences:
        top_priority = max(item.priority for item in evidences)
        top = [item for item in evidences if item.priority == top_priority]
        grades = sorted({item.grade for item in top})
        if len(grades) == 1:
            winner = top[0]
            if winner.rule == "frozen_poorly_differentiated_nsc":
                return {
                    "status": "ambiguous",
                    "tumor_grade": "",
                    "grade_source": "frozen_review_needed",
                    "evidence_extractor": winner.extractor,
                    "evidence_text": winner.snippet,
                    "all_grade_evidence": " || ".join(
                        f"{item.grade}:{item.rule}:{item.extractor}:{item.snippet}" for item in evidences[:12]
                    ),
                    "range_evidence": " || ".join(
                        f"{item.rule}:{item.extractor}:{item.snippet}" for item in range_hits[:8]
                    ),
                    "code_candidates": "",
                    "word_candidates": "",
                }
            return {
                "status": "parsed",
                "tumor_grade": winner.grade,
                "grade_source": winner.rule,
                "evidence_extractor": winner.extractor,
                "evidence_text": winner.snippet,
                "all_grade_evidence": " || ".join(
                    f"{item.grade}:{item.rule}:{item.extractor}:{item.snippet}" for item in evidences[:12]
                ),
                "range_evidence": " || ".join(
                    f"{item.rule}:{item.extractor}:{item.snippet}" for item in range_hits[:8]
                ),
                "code_candidates": "",
                "word_candidates": "",
            }
        return {
            "status": "conflict",
            "tumor_grade": "",
            "grade_source": "top_priority_conflict",
            "evidence_extractor": "",
            "evidence_text": " || ".join(
                f"{item.grade}:{item.rule}:{item.extractor}:{item.snippet}" for item in top[:6]
            ),
            "all_grade_evidence": " || ".join(
                f"{item.grade}:{item.rule}:{item.extractor}:{item.snippet}" for item in evidences[:12]
            ),
            "range_evidence": " || ".join(
                f"{item.rule}:{item.extractor}:{item.snippet}" for item in range_hits[:8]
            ),
            "code_candidates": "",
            "word_candidates": "",
        }

    if range_hits:
        return {
            "status": "ambiguous",
            "tumor_grade": "",
            "grade_source": "range_only",
            "evidence_extractor": "",
            "evidence_text": range_hits[0].snippet,
            "all_grade_evidence": "",
            "range_evidence": " || ".join(
                f"{item.rule}:{item.extractor}:{item.snippet}" for item in range_hits[:8]
            ),
            "code_candidates": "",
            "word_candidates": "",
        }

    return {
        "status": "missing",
        "tumor_grade": "",
        "grade_source": "",
        "evidence_extractor": "",
        "evidence_text": "",
        "all_grade_evidence": "",
        "range_evidence": "",
        "code_candidates": "",
        "word_candidates": "",
    }


def parse_grade(text_versions: list[tuple[str, str]]) -> dict[str, str]:
    basic_results = [basic_parse_one(text, extractor) for extractor, text in text_versions if text]
    basic_merged = merge_basic_results(basic_results)

    evidences: list[Evidence] = []
    range_hits: list[Evidence] = []
    for extractor, text in text_versions:
        if not text:
            continue
        found, ranges = collect_evidence(text, extractor)
        evidences.extend(found)
        range_hits.extend(ranges)
    salvage = choose_grade(evidences, range_hits)

    if basic_merged["status"] == "parsed":
        basic_merged["all_grade_evidence"] = salvage.get("all_grade_evidence", "")
        basic_merged["range_evidence"] = salvage.get("range_evidence", "")
        return basic_merged

    if salvage["status"] == "parsed":
        return salvage

    if basic_merged["status"] in {"conflict", "ambiguous"}:
        basic_merged["all_grade_evidence"] = salvage.get("all_grade_evidence", "")
        basic_merged["range_evidence"] = salvage.get("range_evidence", "")
        return basic_merged

    return salvage


def aggregate_case_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    case_rows: list[dict[str, str]] = []
    for case_id, items in sorted(by_case.items()):
        parsed_items = [item for item in items if item["status"] == "parsed" and item["tumor_grade"]]
        parsed_grades = sorted({item["tumor_grade"] for item in parsed_items})
        statuses = sorted({item["status"] for item in items})

        if len(parsed_grades) == 1:
            best = max(parsed_items, key=lambda item: (len(item["evidence_text"]), item["grade_source"]))
            status = "parsed"
            tumor_grade = parsed_grades[0]
            grade_source = best["grade_source"]
            evidence_extractor = best["evidence_extractor"]
            evidence_text = best["evidence_text"]
        elif len(parsed_grades) > 1 or "conflict" in statuses:
            best = next((item for item in items if item["status"] == "conflict"), items[0])
            status = "conflict"
            tumor_grade = ""
            grade_source = best["grade_source"]
            evidence_extractor = best["evidence_extractor"]
            evidence_text = best["evidence_text"]
        elif "ambiguous" in statuses:
            best = next((item for item in items if item["status"] == "ambiguous"), items[0])
            status = "ambiguous"
            tumor_grade = ""
            grade_source = best["grade_source"]
            evidence_extractor = best["evidence_extractor"]
            evidence_text = best["evidence_text"]
        else:
            status = "missing"
            tumor_grade = ""
            grade_source = ""
            evidence_extractor = ""
            evidence_text = ""

        if case_id in MANUAL_CASE_OVERRIDES:
            tumor_grade, grade_source = MANUAL_CASE_OVERRIDES[case_id]
            status = "parsed"
            evidence_extractor = "manual_review"
            evidence_text = next((item["evidence_text"] for item in items if item["evidence_text"]), evidence_text)
        elif case_id in MANUAL_CASE_EXCLUDE:
            status = "ambiguous"
            tumor_grade = ""
            grade_source = MANUAL_CASE_EXCLUDE[case_id]
            evidence_extractor = "manual_review"
            evidence_text = next((item["evidence_text"] for item in items if item["evidence_text"]), evidence_text)

        case_rows.append(
            {
                "case_id": case_id,
                "status": status,
                "tumor_grade": tumor_grade,
                "grade_source": grade_source,
                "evidence_extractor": evidence_extractor,
                "num_report_files": str(len(items)),
                "text_files": "|".join(item["text_file"] for item in items),
                "file_statuses": "|".join(sorted(item["status"] for item in items)),
                "evidence_text": evidence_text,
            }
        )

    return case_rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--out", required=True, help="File-level output TSV path")
    parser.add_argument("--case-out", default="", help="Optional case-level output TSV path")
    parser.add_argument("--case-table", default="")
    parser.add_argument("--pdf-dir", default="", help="Optional PDF dir for secondary pypdf extraction")
    args = parser.parse_args()

    text_dir = Path(args.text_dir)
    out_path = Path(args.out)
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None

    allowed_cases: set[str] | None = None
    if args.case_table:
        allowed_cases = read_case_ids(Path(args.case_table))

    rows: list[dict[str, str]] = []
    for path in iter_text_files(text_dir):
        case_id = extract_case_id(path)
        if allowed_cases is not None and case_id not in allowed_cases:
            continue

        text_versions: list[tuple[str, str]] = [("text_file", compact(path.read_text(encoding="utf-8", errors="ignore")))]
        if pdf_dir is not None:
            pdf_path = pdf_dir / f"{path.stem}.PDF"
            if pdf_path.exists():
                pypdf_text = extract_pypdf_text(pdf_path)
                if pypdf_text:
                    text_versions.append(("pypdf", pypdf_text))

        parsed = parse_grade(text_versions)
        rows.append(
            {
                "case_id": case_id,
                "text_file": path.name,
                **parsed,
            }
        )

    file_fieldnames = [
        "case_id",
        "text_file",
        "status",
        "tumor_grade",
        "grade_source",
        "evidence_extractor",
        "evidence_text",
        "code_candidates",
        "word_candidates",
        "all_grade_evidence",
        "range_evidence",
    ]
    write_tsv(out_path, file_fieldnames, rows)

    status_counter = Counter(row["status"] for row in rows)
    grade_counter = Counter(row["tumor_grade"] for row in rows if row["tumor_grade"])

    print(f"Text dir: {text_dir}")
    print(f"Rows parsed: {len(rows)}")
    print(f"Output: {out_path}")
    print(f"Status counts: {dict(status_counter)}")
    print(f"Grade counts: {dict(grade_counter)}")

    if args.case_out:
        case_rows = aggregate_case_rows(rows)
        case_out_path = Path(args.case_out)
        case_fieldnames = [
            "case_id",
            "status",
            "tumor_grade",
            "grade_source",
            "evidence_extractor",
            "num_report_files",
            "text_files",
            "file_statuses",
            "evidence_text",
        ]
        write_tsv(case_out_path, case_fieldnames, case_rows)
        case_status_counter = Counter(row["status"] for row in case_rows)
        case_grade_counter = Counter(row["tumor_grade"] for row in case_rows if row["tumor_grade"])
        print(f"Case output: {case_out_path}")
        print(f"Case status counts: {dict(case_status_counter)}")
        print(f"Case grade counts: {dict(case_grade_counter)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
