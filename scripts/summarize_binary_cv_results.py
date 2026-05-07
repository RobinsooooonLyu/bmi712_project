#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def auc_roc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)

    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = avg_rank
        start = end

    rank_sum_pos = sum(ranks[idx] for idx, label in enumerate(labels) if label == 1)
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_accuracy(labels: list[int], probs: list[float], threshold: float = 0.5) -> float:
    if not labels:
        return float("nan")
    correct = 0
    for label, prob in zip(labels, probs):
        pred = 1 if prob >= threshold else 0
        correct += int(pred == label)
    return correct / len(labels)


def collect_run(run_dir: Path, label: str) -> dict:
    fold_dirs = sorted(path for path in run_dir.glob("fold*") if path.is_dir())
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found in {run_dir}")

    fold_rows: list[dict] = []
    merged_predictions: list[dict] = []
    modalities = set()

    for fold_dir in fold_dirs:
        metrics_path = fold_dir / "metrics.json"
        config_path = fold_dir / "config.json"
        predictions_path = fold_dir / "test_predictions.tsv"
        if not metrics_path.exists():
            raise RuntimeError(f"Missing metrics file: {metrics_path}")
        if not config_path.exists():
            raise RuntimeError(f"Missing config file: {config_path}")
        if not predictions_path.exists():
            raise RuntimeError(f"Missing predictions file: {predictions_path}")

        metrics = read_json(metrics_path)
        config = read_json(config_path)
        predictions = read_predictions(predictions_path)

        test_folds = config.get("test_folds", [])
        fold_idx = int(test_folds[0]) if test_folds else int(fold_dir.name.removeprefix("fold"))
        modality = str(config.get("modality", "unknown"))
        modalities.add(modality)

        fold_row = {
            "run_label": label,
            "modality": modality,
            "fold": fold_idx,
            "best_epoch": metrics.get("best_epoch"),
            "best_val_auc": float(metrics["best_val_auc"]),
            "test_auc": float(metrics["test_auc"]),
            "test_accuracy": float(metrics["test_accuracy"]),
            "test_loss": float(metrics["test_loss"]),
            "n_test": len(predictions),
        }
        fold_rows.append(fold_row)

        for row in predictions:
            merged_predictions.append(
                {
                    "run_label": label,
                    "modality": modality,
                    "fold": str(fold_idx),
                    "case_id": row["case_id"],
                    "logit": row["logit"],
                    "probability": row["probability"],
                    "label": row["label"],
                }
            )

    fold_rows.sort(key=lambda row: row["fold"])
    merged_predictions.sort(key=lambda row: (int(row["fold"]), row["case_id"]))

    labels = [int(row["label"]) for row in merged_predictions]
    probs = [float(row["probability"]) for row in merged_predictions]

    pooled_auc = auc_roc(labels, probs)
    pooled_acc = binary_accuracy(labels, probs)
    mean_test_auc = statistics.mean(row["test_auc"] for row in fold_rows)
    std_test_auc = statistics.stdev(row["test_auc"] for row in fold_rows) if len(fold_rows) > 1 else 0.0
    mean_val_auc = statistics.mean(row["best_val_auc"] for row in fold_rows)
    std_val_auc = statistics.stdev(row["best_val_auc"] for row in fold_rows) if len(fold_rows) > 1 else 0.0
    mean_test_acc = statistics.mean(row["test_accuracy"] for row in fold_rows)
    std_test_acc = statistics.stdev(row["test_accuracy"] for row in fold_rows) if len(fold_rows) > 1 else 0.0

    summary = {
        "run_label": label,
        "modality": next(iter(modalities)) if len(modalities) == 1 else "mixed",
        "run_dir": str(run_dir),
        "n_folds": len(fold_rows),
        "n_test_total": len(merged_predictions),
        "mean_best_val_auc": mean_val_auc,
        "std_best_val_auc": std_val_auc,
        "mean_test_auc": mean_test_auc,
        "std_test_auc": std_test_auc,
        "mean_test_accuracy": mean_test_acc,
        "std_test_accuracy": std_test_acc,
        "pooled_test_auc": pooled_auc,
        "pooled_test_accuracy": pooled_acc,
        "fold_metrics": fold_rows,
    }
    return {
        "summary": summary,
        "fold_metrics": fold_rows,
        "predictions": merged_predictions,
    }


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path).expanduser()
    path = Path(value).expanduser()
    return path.name, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run definition as label=/path/to/run_dir or just /path/to/run_dir",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory for combined summaries",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    comparison_rows: list[dict] = []

    for run_arg in args.run:
        label, run_dir = parse_run_arg(run_arg)
        payload = collect_run(run_dir, label)
        summary = payload["summary"]
        fold_rows = payload["fold_metrics"]
        predictions = payload["predictions"]

        with (outdir / f"{label}_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        write_tsv(
            outdir / f"{label}_fold_metrics.tsv",
            fold_rows,
            ["run_label", "modality", "fold", "best_epoch", "best_val_auc", "test_auc", "test_accuracy", "test_loss", "n_test"],
        )
        write_tsv(
            outdir / f"{label}_test_predictions.tsv",
            predictions,
            ["run_label", "modality", "fold", "case_id", "logit", "probability", "label"],
        )

        all_summaries.append(summary)
        comparison_rows.append(
            {
                "run_label": summary["run_label"],
                "modality": summary["modality"],
                "n_folds": summary["n_folds"],
                "n_test_total": summary["n_test_total"],
                "mean_best_val_auc": summary["mean_best_val_auc"],
                "std_best_val_auc": summary["std_best_val_auc"],
                "mean_test_auc": summary["mean_test_auc"],
                "std_test_auc": summary["std_test_auc"],
                "mean_test_accuracy": summary["mean_test_accuracy"],
                "std_test_accuracy": summary["std_test_accuracy"],
                "pooled_test_auc": summary["pooled_test_auc"],
                "pooled_test_accuracy": summary["pooled_test_accuracy"],
            }
        )

    with (outdir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2)
    write_tsv(
        outdir / "comparison_summary.tsv",
        comparison_rows,
        [
            "run_label",
            "modality",
            "n_folds",
            "n_test_total",
            "mean_best_val_auc",
            "std_best_val_auc",
            "mean_test_auc",
            "std_test_auc",
            "mean_test_accuracy",
            "std_test_accuracy",
            "pooled_test_auc",
            "pooled_test_accuracy",
        ],
    )

    for row in comparison_rows:
        print(
            f"{row['run_label']}: "
            f"mean_test_auc={float(row['mean_test_auc']):.4f} "
            f"std_test_auc={float(row['std_test_auc']):.4f} "
            f"pooled_test_auc={float(row['pooled_test_auc']):.4f} "
            f"mean_test_accuracy={float(row['mean_test_accuracy']):.4f} "
            f"pooled_test_accuracy={float(row['pooled_test_accuracy']):.4f}"
        )
    print(f"Wrote summaries to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
