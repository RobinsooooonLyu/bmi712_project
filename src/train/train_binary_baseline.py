from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from src.data.augmentations import ModalityAugmentationConfig, build_eval_augmentor, build_train_augmentor
from src.data.binary_wsi_dataset import TCGALUADBinaryPatientBagDataset, collate_binary_patient_bags
from src.models.patient_binary_model import PatientBinaryModel, PatientBinaryModelConfig
from src.models.virchow2_encoder import build_virchow2_preprocess


@dataclass
class BinaryBaselineConfig:
    patient_manifest_path: str
    split_path: str
    wsi_root: str
    coords_root: str
    output_dir: str
    modality: str
    train_folds: list[int]
    val_folds: list[int]
    test_folds: list[int]
    batch_size: int = 1
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    dropout: float = 0.25
    early_stopping_patience: int = 5
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    grad_clip_norm: float = 1.0
    seed: int = 20260403
    tiles_per_slide: int = 1024
    max_slides_per_patient: int = 1
    slide_selection: str = "largest"
    encoder_batch_size: int = 64
    log_interval_batches: int = 10
    device: str = "cuda"
    freeze_backbone: bool = True
    unfreeze_last_block: bool = False
    save_predictions: bool = True
    save_latent_exports: bool = False
    latent_export_splits: list[str] | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> BinaryBaselineConfig:
    with open(path, encoding="utf-8") as handle:
        return BinaryBaselineConfig(**yaml.safe_load(handle))


def move_bags_to_device(bags: list[list[torch.Tensor]], device: torch.device) -> list[list[torch.Tensor]]:
    return [[slide.to(device, non_blocking=True) for slide in patient_bag] for patient_bag in bags]


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores_list = scores.detach().cpu().tolist()
    labels_list = [int(item) for item in labels.detach().cpu().tolist()]
    positives = [score for score, label in zip(scores_list, labels_list) if label == 1]
    negatives = [score for score, label in zip(scores_list, labels_list) if label == 0]
    if not positives or not negatives:
        return 0.0
    concordant = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                concordant += 1.0
            elif pos == neg:
                concordant += 0.5
    return concordant / total


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds.cpu() == labels.cpu()).float().mean().item()


def build_dataloader(config: BinaryBaselineConfig, folds: list[int], is_training: bool) -> DataLoader:
    preprocess = build_virchow2_preprocess()
    augmentor = build_train_augmentor(config.modality, ModalityAugmentationConfig()) if is_training else build_eval_augmentor()
    dataset = TCGALUADBinaryPatientBagDataset(
        patient_manifest_path=config.patient_manifest_path,
        split_path=config.split_path,
        wsi_root=config.wsi_root,
        coords_root=config.coords_root,
        modality=config.modality,
        folds=folds,
        is_training=is_training,
        augmentor=augmentor,
        preprocess_transform=preprocess,
        tiles_per_slide=config.tiles_per_slide,
        max_slides_per_patient=config.max_slides_per_patient,
        slide_selection=config.slide_selection,
        seed=config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_training,
        num_workers=config.num_workers,
        collate_fn=collate_binary_patient_bags,
    )


def infer_pos_weight(loader: DataLoader, device: torch.device) -> torch.Tensor:
    labels = [sample.label for sample in loader.dataset.samples]
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    weight = negatives / max(positives, 1)
    return torch.tensor(weight, dtype=torch.float32, device=device)


def summarize_labels(loader: DataLoader) -> str:
    labels = [sample.label for sample in loader.dataset.samples]
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    return f"n={len(labels)} pos={positives} neg={negatives}"


def evaluate(
    model: PatientBinaryModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    split_name: str,
    log_interval_batches: int,
) -> dict[str, Any]:
    model.eval()
    all_logits = []
    all_labels = []
    all_case_ids = []
    all_patient_embeddings = []
    all_pathologic_stage_group = []
    all_clinical_stage_hybrid = []
    losses = []
    with torch.no_grad():
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            outputs = model(move_bags_to_device(batch["bags"], device))
            labels = batch["labels"].to(device)
            loss = criterion(outputs["logits"], labels)
            losses.append(loss.item())
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(batch["labels"].cpu())
            all_patient_embeddings.append(outputs["patient_embeddings"].cpu())
            all_case_ids.extend(batch["case_ids"])
            all_pathologic_stage_group.extend(batch["pathologic_stage_group"])
            all_clinical_stage_hybrid.extend(batch["clinical_stage_hybrid"])
            if batch_idx == 1 or batch_idx % log_interval_batches == 0 or batch_idx == total_batches:
                print(
                    f"[eval:{split_name}] batch={batch_idx}/{total_batches} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "auc": binary_auc(logits, labels),
        "accuracy": binary_accuracy(logits, labels),
        "logits": logits,
        "labels": labels,
        "patient_embeddings": torch.cat(all_patient_embeddings, dim=0),
        "case_ids": all_case_ids,
        "pathologic_stage_group": all_pathologic_stage_group,
        "clinical_stage_hybrid": all_clinical_stage_hybrid,
    }


def write_latent_export(
    output_dir: Path,
    split_name: str,
    config: BinaryBaselineConfig,
    metrics: dict[str, Any],
) -> None:
    export_dir = output_dir / "latent_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    outpath = export_dir / f"{split_name}.pt"
    payload = {
        "split_name": split_name,
        "config": asdict(config),
        "case_ids": metrics["case_ids"],
        "row_ids": metrics["case_ids"],
        "base_case_ids": metrics["case_ids"],
        "view_indices": [0] * len(metrics["case_ids"]),
        "labels": metrics["labels"],
        "logits": metrics["logits"],
        "probabilities": torch.sigmoid(metrics["logits"]),
        "patient_embeddings": metrics["patient_embeddings"],
        "pathologic_stage_group": metrics["pathologic_stage_group"],
        "clinical_stage_hybrid": metrics["clinical_stage_hybrid"],
    }
    torch.save(payload, outpath)

    tsv_path = export_dir / f"{split_name}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as handle:
        handle.write("row_id\tcase_id\tview_index\tlabel\tprobability\tlogit\n")
        for case_id, label, probability, logit in zip(
            payload["case_ids"],
            payload["labels"].tolist(),
            payload["probabilities"].tolist(),
            payload["logits"].tolist(),
        ):
            handle.write(f"{case_id}\t{case_id}\t0\t{int(label)}\t{probability}\t{logit}\n")
    print(f"Saved latent export split={split_name} to: {outpath}", flush=True)


def train_one_epoch(
    model: PatientBinaryModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    epoch: int,
    log_interval_batches: int,
) -> float:
    model.train()
    losses = []
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(move_bags_to_device(batch["bags"], device))
        labels = batch["labels"].to(device)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        losses.append(loss.detach().item())
        if batch_idx == 1 or batch_idx % log_interval_batches == 0 or batch_idx == total_batches:
            print(
                f"[train] epoch={epoch} batch={batch_idx}/{total_batches} "
                f"loss={loss.item():.4f}",
                flush=True,
            )
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(
        "Runtime device: "
        f"requested={config.device} "
        f"selected={device} "
        f"cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"CUDA current device: {torch.cuda.current_device()} {torch.cuda.get_device_name()}", flush=True)
    train_loader = build_dataloader(config, config.train_folds, is_training=True)
    val_loader = build_dataloader(config, config.val_folds, is_training=False)
    test_loader = build_dataloader(config, config.test_folds, is_training=False)
    print(
        "Run config: "
        f"modality={config.modality} "
        f"train_folds={config.train_folds} "
        f"val_folds={config.val_folds} "
        f"test_folds={config.test_folds} "
        f"tiles_per_slide={config.tiles_per_slide} "
        f"max_slides_per_patient={config.max_slides_per_patient} "
        f"slide_selection={config.slide_selection} "
        f"encoder_batch_size={config.encoder_batch_size} "
        f"epochs={config.epochs} "
        f"early_stopping_patience={config.early_stopping_patience} "
        f"scheduler_patience={config.scheduler_patience} "
        f"output_dir={config.output_dir}",
        flush=True,
    )
    print(
        f"Dataset sizes: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )
    print(
        "Label counts: "
        f"train=({summarize_labels(train_loader)}) "
        f"val=({summarize_labels(val_loader)}) "
        f"test=({summarize_labels(test_loader)})",
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
    pos_weight = infer_pos_weight(train_loader, device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    best_state = None
    best_val_auc = float("-inf")
    best_epoch = -1
    history = []
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.grad_clip_norm,
            epoch,
            config.log_interval_batches,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, "val", config.log_interval_batches)
        scheduler.step(val_metrics["auc"])
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"lr={current_lr:.2e}",
            flush=True,
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"New best checkpoint: epoch={epoch} val_auc={best_val_auc:.4f}", flush=True)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} after {epochs_without_improvement} epochs without improvement.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device, "test", config.log_interval_batches)

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_val_auc": best_val_auc,
                "best_epoch": best_epoch,
                "test_loss": test_metrics["loss"],
                "test_auc": test_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "pos_weight": float(pos_weight.item()),
            },
            handle,
            indent=2,
        )
    torch.save(model.state_dict(), output_dir / "model.pt")

    if config.save_predictions:
        with open(output_dir / "test_predictions.tsv", "w", encoding="utf-8") as handle:
            handle.write("case_id\tlogit\tprobability\tlabel\n")
            probabilities = torch.sigmoid(test_metrics["logits"])
            for case_id, logit, probability, label in zip(
                test_metrics["case_ids"],
                test_metrics["logits"].tolist(),
                probabilities.tolist(),
                test_metrics["labels"].tolist(),
            ):
                handle.write(f"{case_id}\t{logit}\t{probability}\t{int(label)}\n")

    if config.save_latent_exports:
        export_splits = config.latent_export_splits or ["train", "val", "test"]
        split_to_loader = {
            "train": build_dataloader(config, config.train_folds, is_training=False),
            "val": val_loader,
            "test": test_loader,
        }
        split_to_metrics = {
            "val": val_metrics,
            "test": test_metrics,
        }
        for split_name in export_splits:
            if split_name not in split_to_loader:
                raise ValueError(f"Unsupported latent export split: {split_name}")
            if split_name == "train":
                split_metrics = evaluate(
                    model,
                    split_to_loader[split_name],
                    criterion,
                    device,
                    "train_export",
                    config.log_interval_batches,
                )
            else:
                split_metrics = split_to_metrics[split_name]
            write_latent_export(output_dir, split_name, config, split_metrics)

    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best val AUC: {best_val_auc:.4f}", flush=True)
    print(f"Test loss: {test_metrics['loss']:.4f}", flush=True)
    print(f"Test AUC: {test_metrics['auc']:.4f}", flush=True)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}", flush=True)
    print(f"Saved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
