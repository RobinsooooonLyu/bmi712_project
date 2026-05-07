from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from src.train.train_binary_baseline import (
    BinaryBaselineConfig,
    build_dataloader,
    load_config,
    move_bags_to_device,
    set_seed,
)
from src.models.patient_binary_model import PatientBinaryModel, PatientBinaryModelConfig


def load_run_config(run_dir: Path) -> BinaryBaselineConfig:
    config_json = run_dir / "config.json"
    if config_json.exists():
        with open(config_json, encoding="utf-8") as handle:
            return BinaryBaselineConfig(**json.load(handle))
    yaml_candidates = list(run_dir.glob("*.yaml"))
    if yaml_candidates:
        return load_config(yaml_candidates[0])
    raise FileNotFoundError(f"Could not find config.json in {run_dir}")


def export_split(
    model: PatientBinaryModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split_name: str,
    outpath: Path,
    run_dir: Path,
    config: BinaryBaselineConfig,
) -> None:
    model.eval()
    case_ids: list[str] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    patient_embeddings: list[torch.Tensor] = []
    pathologic_stage_group: list[str] = []
    clinical_stage_hybrid: list[str] = []
    view_indices: list[int] = []
    base_case_ids: list[str] = []
    row_ids: list[str] = []

    with torch.no_grad():
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            outputs = model(move_bags_to_device(batch["bags"], device))
            batch_logits = outputs["logits"].detach().cpu()
            batch_probabilities = torch.sigmoid(batch_logits)
            batch_embeddings = outputs["patient_embeddings"].detach().cpu()

            case_ids.extend(batch["case_ids"])
            labels.append(batch["labels"].detach().cpu())
            logits.append(batch_logits)
            probabilities.append(batch_probabilities)
            patient_embeddings.append(batch_embeddings)
            pathologic_stage_group.extend(batch["pathologic_stage_group"])
            clinical_stage_hybrid.extend(batch["clinical_stage_hybrid"])
            view_indices.extend([0] * len(batch["case_ids"]))
            base_case_ids.extend(batch["case_ids"])
            row_ids.extend(batch["case_ids"])

            print(
                f"[export:{split_name}] batch={batch_idx}/{total_batches} "
                f"cases={len(case_ids)}",
                flush=True,
            )

    payload = {
        "split_name": split_name,
        "source_run_dir": str(run_dir),
        "config": asdict(config),
        "case_ids": case_ids,
        "labels": torch.cat(labels, dim=0),
        "logits": torch.cat(logits, dim=0),
        "probabilities": torch.cat(probabilities, dim=0),
        "patient_embeddings": torch.cat(patient_embeddings, dim=0),
        "pathologic_stage_group": pathologic_stage_group,
        "clinical_stage_hybrid": clinical_stage_hybrid,
        "view_indices": view_indices,
        "base_case_ids": base_case_ids,
        "row_ids": row_ids,
    }
    torch.save(payload, outpath)

    tsv_path = outpath.with_suffix(".tsv")
    with open(tsv_path, "w", encoding="utf-8") as handle:
        handle.write("row_id\tcase_id\tview_index\tlabel\tprobability\tlogit\n")
        for row_id, case_id, view_index, label, probability, logit in zip(
            payload["row_ids"],
            payload["base_case_ids"],
            payload["view_indices"],
            payload["labels"].tolist(),
            payload["probabilities"].tolist(),
            payload["logits"].tolist(),
        ):
            handle.write(f"{row_id}\t{case_id}\t{view_index}\t{int(label)}\t{probability}\t{logit}\n")

    print(f"Saved {split_name} latent export to: {outpath}", flush=True)


def export_train_multiview_split(
    model: PatientBinaryModel,
    config: BinaryBaselineConfig,
    device: torch.device,
    outpath: Path,
    run_dir: Path,
    train_views: int,
) -> None:
    case_ids: list[str] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    patient_embeddings: list[torch.Tensor] = []
    pathologic_stage_group: list[str] = []
    clinical_stage_hybrid: list[str] = []
    view_indices: list[int] = []
    base_case_ids: list[str] = []
    row_ids: list[str] = []

    model.eval()
    for view_idx in range(train_views):
        view_config = replace(config, seed=config.seed + view_idx)
        loader = build_dataloader(view_config, view_config.train_folds, is_training=True)
        total_batches = len(loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                outputs = model(move_bags_to_device(batch["bags"], device))
                batch_logits = outputs["logits"].detach().cpu()
                batch_probabilities = torch.sigmoid(batch_logits)
                batch_embeddings = outputs["patient_embeddings"].detach().cpu()

                labels.append(batch["labels"].detach().cpu())
                logits.append(batch_logits)
                probabilities.append(batch_probabilities)
                patient_embeddings.append(batch_embeddings)
                pathologic_stage_group.extend(batch["pathologic_stage_group"])
                clinical_stage_hybrid.extend(batch["clinical_stage_hybrid"])

                for case_id in batch["case_ids"]:
                    case_ids.append(case_id)
                    base_case_ids.append(case_id)
                    view_indices.append(view_idx)
                    row_ids.append(f"{case_id}__view{view_idx}")

                if batch_idx == 1 or batch_idx == total_batches or batch_idx % view_config.log_interval_batches == 0:
                    print(
                        f"[export:train:view{view_idx}] batch={batch_idx}/{total_batches} "
                        f"rows={len(row_ids)}",
                        flush=True,
                    )

    payload = {
        "split_name": "train",
        "source_run_dir": str(run_dir),
        "config": asdict(config),
        "case_ids": case_ids,
        "labels": torch.cat(labels, dim=0),
        "logits": torch.cat(logits, dim=0),
        "probabilities": torch.cat(probabilities, dim=0),
        "patient_embeddings": torch.cat(patient_embeddings, dim=0),
        "pathologic_stage_group": pathologic_stage_group,
        "clinical_stage_hybrid": clinical_stage_hybrid,
        "view_indices": view_indices,
        "base_case_ids": base_case_ids,
        "row_ids": row_ids,
    }
    torch.save(payload, outpath)

    tsv_path = outpath.with_suffix(".tsv")
    with open(tsv_path, "w", encoding="utf-8") as handle:
        handle.write("row_id\tcase_id\tview_index\tlabel\tprobability\tlogit\n")
        for row_id, case_id, view_index, label, probability, logit in zip(
            payload["row_ids"],
            payload["base_case_ids"],
            payload["view_indices"],
            payload["labels"].tolist(),
            payload["probabilities"].tolist(),
            payload["logits"].tolist(),
        ):
            handle.write(f"{row_id}\t{case_id}\t{view_index}\t{int(label)}\t{probability}\t{logit}\n")

    print(f"Saved train multiview latent export to: {outpath}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--train-views", type=int, default=1)
    parser.add_argument("--augment-train", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(run_dir)
    if args.batch_size_override is not None:
        config.batch_size = args.batch_size_override

    set_seed(config.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        "Latent export runtime device: "
        f"requested={args.device} selected={device} "
        f"cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}",
        flush=True,
    )

    model = PatientBinaryModel(
        PatientBinaryModelConfig(
            dropout=config.dropout,
            encoder_batch_size=config.encoder_batch_size,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_block=config.unfreeze_last_block,
        )
    ).to(device)
    state_dict = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)

    split_map: dict[str, list[int]] = {
        "train": config.train_folds,
        "val": config.val_folds,
        "test": config.test_folds,
    }
    for split_name in args.splits:
        if split_name not in split_map:
            raise ValueError(f"Unsupported split: {split_name}")
        if split_name == "train" and args.augment_train and args.train_views > 1:
            print(
                f"Exporting split=train with augmentation views={args.train_views} "
                f"folds={split_map[split_name]}",
                flush=True,
            )
            export_train_multiview_split(
                model=model,
                config=config,
                device=device,
                outpath=outdir / "train.pt",
                run_dir=run_dir,
                train_views=args.train_views,
            )
            continue
        loader = build_dataloader(config, split_map[split_name], is_training=False)
        print(
            f"Exporting split={split_name} folds={split_map[split_name]} "
            f"n={len(loader.dataset)}",
            flush=True,
        )
        export_split(
            model=model,
            loader=loader,
            device=device,
            split_name=split_name,
            outpath=outdir / f"{split_name}.pt",
            run_dir=run_dir,
            config=config,
        )


if __name__ == "__main__":
    main()
