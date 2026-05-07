from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_strata(rows: list[dict[str, str]], label_column: str) -> dict[str, list[dict[str, str]]]:
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        stage_group = row.get("pathologic_stage_group", "") or "unknown_stage"
        strata[f"{row[label_column]}|{stage_group}"].append(row)
    return strata


def assign_folds(rows: list[dict[str, str]], label_column: str, n_folds: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    output: list[dict[str, str]] = []
    fold_counts = [0 for _ in range(n_folds)]
    for _, stratum_rows in sorted(build_strata(rows, label_column).items(), key=lambda item: item[0]):
        rng.shuffle(stratum_rows)
        start_fold = min(range(n_folds), key=lambda fold: fold_counts[fold])
        for idx, row in enumerate(stratum_rows):
            fold = (start_fold + idx) % n_folds
            fold_counts[fold] += 1
            payload = dict(row)
            payload["fold"] = str(fold)
            output.append(payload)
    return sorted(output, key=lambda row: (int(row["fold"]), row["case_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-manifest", required=True)
    parser.add_argument("--label-table", required=True)
    parser.add_argument("--label-column", required=True, choices=["high_grade", "high_risk_grade_vpi_lvi"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260406)
    args = parser.parse_args()

    manifest_rows = {row["case_id"]: row for row in read_tsv(args.patient_manifest)}
    label_rows = {row["case_id"]: row for row in read_tsv(args.label_table)}

    rows = []
    for case_id, manifest in manifest_rows.items():
        label = label_rows.get(case_id)
        if not label or label.get(args.label_column) not in {"0", "1"}:
            continue
        rows.append(
            {
                "case_id": case_id,
                "label": label[args.label_column],
                args.label_column: label[args.label_column],
                "tumor_grade": label.get("tumor_grade", ""),
                "vpi_status": label.get("vpi_status", ""),
                "lvi_status": label.get("lvi_status", ""),
                "pathologic_stage_group": manifest.get("pathologic_stage_group", ""),
                "clinical_stage_hybrid": manifest.get("clinical_stage_hybrid", ""),
            }
        )

    if not rows:
        raise RuntimeError("No usable rows after intersecting patient manifest with label table.")

    output = assign_folds(rows, args.label_column, args.n_folds, args.seed)
    write_tsv(
        args.out,
        output,
        [
            "case_id",
            "fold",
            "label",
            args.label_column,
            "tumor_grade",
            "vpi_status",
            "lvi_status",
            "pathologic_stage_group",
            "clinical_stage_hybrid",
        ],
    )
    positives = sum(row["label"] == "1" for row in output)
    print(f"Label column: {args.label_column}")
    print(f"Input rows: {len(rows)}")
    print(f"Positive rows: {positives}")
    print(f"Negative rows: {len(rows) - positives}")
    print(f"Output split: {args.out}")


if __name__ == "__main__":
    main()
