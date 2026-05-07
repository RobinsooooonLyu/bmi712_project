#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict


GDC_FILES_URL = "https://api.gdc.cancer.gov/files"


def fetch_hits(project_id: str):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {
                "op": "in",
                "content": {
                    "field": "experimental_strategy",
                    "value": ["Diagnostic Slide", "Tissue Slide"],
                },
            },
            {
                "op": "in",
                "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]},
            },
        ],
    }
    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "state",
        "experimental_strategy",
        "cases.submitter_id",
        "cases.samples.sample_type",
        "cases.samples.preservation_method",
        "cases.samples.tissue_type",
        "cases.samples.portions.slides.submitter_id",
    ]
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "10000",
    }
    url = GDC_FILES_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    return payload["data"]["hits"]


def build_case_index(hits):
    by_case = defaultdict(lambda: {"Diagnostic Slide": [], "Tissue Slide": [], "tissue_pm": set()})
    for hit in hits:
        strategy = hit["experimental_strategy"]
        file_row = {
            "id": hit["file_id"],
            "filename": hit["file_name"],
            "md5": hit.get("md5sum") or "",
            "size": str(hit.get("file_size") or ""),
            "state": hit.get("state") or "",
        }
        for case in hit.get("cases", []):
            case_id = case["submitter_id"]
            by_case[case_id][strategy].append(file_row)
            if strategy == "Tissue Slide":
                for sample in case.get("samples", []):
                    pm = sample.get("preservation_method") or "None"
                    by_case[case_id]["tissue_pm"].add(pm)
    return by_case


def write_manifest(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("id\tfilename\tmd5\tsize\tstate\n")
        for row in rows:
            handle.write(
                f"{row['id']}\t{row['filename']}\t{row['md5']}\t{row['size']}\t{row['state']}\n"
            )


def write_case_table(path, case_rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "case_id\tdx_files\ttissue_files\ttissue_preservation_methods\tdx_filenames\ttissue_filenames\n"
        )
        for row in case_rows:
            handle.write(
                f"{row['case_id']}\t{row['dx_files']}\t{row['tissue_files']}\t{row['tissue_pm']}\t"
                f"{row['dx_filenames']}\t{row['tissue_filenames']}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="TCGA-LUAD")
    parser.add_argument("--outdir", default="manifests")
    parser.add_argument(
        "--strict-tissue",
        action="store_true",
        help="Exclude paired cases whose tissue-slide samples contain FFPE preservation metadata.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="If set, emit manifests for only the first N paired cases after sorting by case id.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    hits = fetch_hits(args.project)
    by_case = build_case_index(hits)

    paired_case_rows = []
    dx_manifest_rows = []
    tissue_manifest_rows = []
    paired_cases = 0

    selected_case_ids = []
    for case_id in sorted(by_case):
        record = by_case[case_id]
        if not record["Diagnostic Slide"] or not record["Tissue Slide"]:
            continue
        if args.strict_tissue and "FFPE" in record["tissue_pm"]:
            continue
        selected_case_ids.append(case_id)

    if args.limit_cases > 0:
        selected_case_ids = selected_case_ids[: args.limit_cases]

    for case_id in selected_case_ids:
        record = by_case[case_id]

        paired_cases += 1
        paired_case_rows.append(
            {
                "case_id": case_id,
                "dx_files": len(record["Diagnostic Slide"]),
                "tissue_files": len(record["Tissue Slide"]),
                "tissue_pm": ",".join(sorted(record["tissue_pm"])),
                "dx_filenames": "|".join(sorted({row["filename"] for row in record["Diagnostic Slide"]})),
                "tissue_filenames": "|".join(
                    sorted({row["filename"] for row in record["Tissue Slide"]})
                ),
            }
        )
        dx_manifest_rows.extend(record["Diagnostic Slide"])
        tissue_manifest_rows.extend(record["Tissue Slide"])

    prefix = args.project.lower().replace("-", "_")
    suffix = "_strict" if args.strict_tissue else ""
    if args.limit_cases > 0:
        suffix += f"_first{args.limit_cases}"
    case_table_path = os.path.join(args.outdir, f"{prefix}_paired_case_table{suffix}.tsv")
    dx_manifest_path = os.path.join(args.outdir, f"{prefix}_paired_dx_manifest{suffix}.txt")
    tissue_manifest_path = os.path.join(args.outdir, f"{prefix}_paired_tissue_manifest{suffix}.txt")

    write_case_table(case_table_path, paired_case_rows)
    write_manifest(dx_manifest_path, dx_manifest_rows)
    write_manifest(tissue_manifest_path, tissue_manifest_rows)

    print(f"Project: {args.project}")
    print(f"Paired cases: {paired_cases}")
    print(f"Diagnostic slide files: {len(dx_manifest_rows)}")
    print(f"Tissue slide files: {len(tissue_manifest_rows)}")
    print(f"Case table: {case_table_path}")
    print(f"DX manifest: {dx_manifest_path}")
    print(f"Tissue manifest: {tissue_manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
