#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import urllib.parse
import urllib.request


GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"


def read_case_ids(case_table_path):
    case_ids = []
    with open(case_table_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            case_id = row["case_id"].strip()
            if case_id:
                case_ids.append(case_id)
    return case_ids


def chunked(values, size):
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def fetch_case_metadata(case_ids):
    results = {}
    for batch in chunked(case_ids, 100):
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-LUAD"]}},
                {"op": "in", "content": {"field": "submitter_id", "value": batch}},
            ],
        }
        fields = [
            "submitter_id",
            "demographic.gender",
            "diagnoses.age_at_diagnosis",
            "diagnoses.primary_diagnosis",
            "diagnoses.ajcc_clinical_stage",
            "diagnoses.ajcc_clinical_t",
            "diagnoses.ajcc_clinical_n",
            "diagnoses.ajcc_clinical_m",
            "diagnoses.ajcc_pathologic_stage",
            "diagnoses.ajcc_pathologic_t",
            "diagnoses.ajcc_pathologic_n",
            "diagnoses.ajcc_pathologic_m",
            "diagnoses.residual_disease",
            "diagnoses.tissue_or_organ_of_origin",
            "diagnoses.site_of_resection_or_biopsy",
        ]
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "expand": "diagnoses,demographic",
            "format": "JSON",
            "size": "1000",
        }
        url = GDC_CASES_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.load(response)
        for case in payload["data"]["hits"]:
            submitter_id = case["submitter_id"]
            diagnosis = pick_primary_diagnosis(case.get("diagnoses", []))
            demographic = case.get("demographic", {})
            clinical_stage = diagnosis.get("ajcc_clinical_stage", "")
            clinical_t = diagnosis.get("ajcc_clinical_t", "")
            pathologic_stage = diagnosis.get("ajcc_pathologic_stage", "")
            results[submitter_id] = {
                "gender": demographic.get("gender") or "",
                "age_at_diagnosis_days": diagnosis.get("age_at_diagnosis", ""),
                "primary_diagnosis": diagnosis.get("primary_diagnosis", ""),
                "ajcc_clinical_stage": clinical_stage or "",
                "ajcc_clinical_t": clinical_t or "",
                "ajcc_clinical_n": diagnosis.get("ajcc_clinical_n", "") or "",
                "ajcc_clinical_m": diagnosis.get("ajcc_clinical_m", "") or "",
                "ajcc_pathologic_stage": pathologic_stage or "",
                "ajcc_pathologic_t": diagnosis.get("ajcc_pathologic_t", "") or "",
                "ajcc_pathologic_n": diagnosis.get("ajcc_pathologic_n", "") or "",
                "ajcc_pathologic_m": diagnosis.get("ajcc_pathologic_m", "") or "",
                "residual_disease": diagnosis.get("residual_disease", "") or "",
                "tissue_or_organ_of_origin": diagnosis.get("tissue_or_organ_of_origin", "") or "",
                "site_of_resection_or_biopsy": diagnosis.get("site_of_resection_or_biopsy", "") or "",
                "pathologic_stage_group": collapse_pathologic_stage(pathologic_stage),
                "pathologic_split_group": derive_pathologic_split_group(
                    pathologic_stage,
                    diagnosis.get("ajcc_pathologic_t", ""),
                    diagnosis.get("ajcc_pathologic_n", ""),
                ),
                "clinical_stage_hybrid": derive_clinical_stage_hybrid(clinical_stage, clinical_t),
            }
    return results


def pick_primary_diagnosis(diagnoses):
    for diagnosis in diagnoses:
        if diagnosis.get("classification_of_tumor") == "primary":
            return diagnosis
    return diagnoses[0] if diagnoses else {}


def is_missing(value):
    if value in ("", None):
        return True
    text = str(value).strip()
    return text in {"#N/A", "[Not Available]", "[Unknown]", "NA", "None"}


def canonical_stage_label(raw_value):
    if is_missing(raw_value):
        return "Unknown"
    text = str(raw_value).strip().upper()
    if text.startswith("STAGE IA"):
        return "Stage IA"
    if text.startswith("STAGE IB"):
        return "Stage IB"
    if text.startswith("STAGE IC"):
        return "Stage IC"
    if text.startswith("STAGE IIA"):
        return "Stage IIA"
    if text.startswith("STAGE IIB"):
        return "Stage IIB"
    if text.startswith("STAGE IIC"):
        return "Stage IIC"
    if text.startswith("STAGE IIIA"):
        return "Stage IIIA"
    if text.startswith("STAGE IIIB"):
        return "Stage IIIB"
    if text.startswith("STAGE IIIC"):
        return "Stage IIIC"
    if text.startswith("STAGE IVA"):
        return "Stage IVA"
    if text.startswith("STAGE IVB"):
        return "Stage IVB"
    if text.startswith("STAGE IV"):
        return "Stage IV"
    if text.startswith("STAGE III"):
        return "Stage III"
    if text.startswith("STAGE II"):
        return "Stage II"
    if text.startswith("STAGE I"):
        return "Stage I"
    return "Unknown"


def collapse_pathologic_stage(raw_value):
    stage = canonical_stage_label(raw_value)
    if stage.startswith("Stage I"):
        return "Stage I"
    if stage.startswith("Stage II"):
        return "Stage II"
    if stage.startswith("Stage III") or stage.startswith("Stage IV"):
        return "Stage III+"
    return "Unknown"


def normalize_pathologic_t(raw_value):
    if is_missing(raw_value):
        return ""
    text = str(raw_value).strip()
    text = re.sub(r"\s+", "", text)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("pt"):
        suffix = text[2:]
    elif lowered.startswith("ct"):
        suffix = text[2:]
    elif lowered.startswith("t"):
        suffix = text[1:]
    else:
        return ""
    if not suffix:
        return ""
    return f"pT{suffix}"


def derive_pathologic_split_group(pathologic_stage, pathologic_t, pathologic_n):
    del pathologic_stage  # stage is not needed once T/N are available for this split rule
    path_t = normalize_pathologic_t(pathologic_t)
    path_n = str(pathologic_n or "").strip().upper().replace(" ", "")

    if path_n in {"N1", "N2", "N3", "PN1", "PN2", "PN3"}:
        return "pN1plus"

    if path_n in {"N0", "NX", "PN0", "PNX"}:
        if path_t in {"pT1", "pT1a", "pT1b", "pT1c"}:
            return "pN0_T1"
        if path_t in {"pT2", "pT2a", "pT2b"}:
            return "pN0_T2"
        if path_t in {"pT3", "pT4"}:
            return "pN0_Thigher"

    return "unknown"


def normalize_clinical_t(raw_value):
    if is_missing(raw_value):
        return ""
    text = str(raw_value).strip()
    text = re.sub(r"\s+", "", text)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("ct"):
        suffix = text[2:]
    elif lowered.startswith("pt"):
        suffix = text[2:]
    elif lowered.startswith("t"):
        suffix = text[1:]
    else:
        return ""
    if not suffix:
        return ""
    return f"cT{suffix}"


def derive_clinical_stage_hybrid(clinical_stage, clinical_t):
    stage = canonical_stage_label(clinical_stage)
    if stage in {"Stage I", "Stage IA", "Stage IB", "Stage IC"}:
        clinical_t_norm = normalize_clinical_t(clinical_t)
        return clinical_t_norm or "Stage I"
    if stage.startswith("Stage II"):
        return "Stage II"
    if stage.startswith("Stage III"):
        return "Stage III"
    if stage.startswith("Stage IV"):
        return "Stage IV"
    return "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-table",
        default="manifests/tcga_luad_paired_case_table_strict.tsv",
        help="Paired case table generated by build_tcga_luad_wsi_manifests.py",
    )
    parser.add_argument(
        "--out",
        default="manifests/tcga_luad_paired_case_data_strict.tsv",
        help="Output TSV path",
    )
    args = parser.parse_args()

    case_ids = read_case_ids(args.case_table)
    case_metadata = fetch_case_metadata(case_ids)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "case_id",
        "gender",
        "age_at_diagnosis_days",
        "primary_diagnosis",
        "ajcc_clinical_stage",
        "ajcc_clinical_t",
        "ajcc_clinical_n",
        "ajcc_clinical_m",
        "clinical_stage_hybrid",
        "ajcc_pathologic_stage",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
        "pathologic_stage_group",
        "pathologic_split_group",
        "residual_disease",
        "tissue_or_organ_of_origin",
        "site_of_resection_or_biopsy",
    ]
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for case_id in case_ids:
            row = {"case_id": case_id}
            row.update(case_metadata.get(case_id, {}))
            writer.writerow(row)

    print(f"Paired cases in input table: {len(case_ids)}")
    print(f"Output case data table: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
