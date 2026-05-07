from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.data.augmentations import ModalityAugmentationConfig, build_eval_augmentor, build_train_augmentor
from src.data.binary_wsi_dataset import TCGALUADBinaryPatientBagDataset
from src.data.online_frozen_distill_dataset import (
    OnlineFrozenDistillDataset,
    collate_online_frozen_distill,
)
from src.models.online_frozen_distillation import (
    OnlineFrozenDistillationConfig,
    OnlineFrozenToFFPEDistillationModel,
)
from src.models.patient_binary_model import PatientBinaryModelConfig
from src.models.virchow2_encoder import build_virchow2_preprocess
from src.train.train_binary_baseline import binary_accuracy, binary_auc, move_bags_to_device


@dataclass
class OnlineFrozenDistillTrainConfig:
    patient_manifest_path: str
    split_path: str
    frozen_wsi_root: str
    frozen_coords_root: str
    ffpe_latent_dir: str
    output_dir: str
    train_folds: list[int]
    val_folds: list[int]
    test_folds: list[int]
    modality: str = "frozen_online_distill"
    frozen_checkpoint: str | None = None
    batch_size: int = 1
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    dropout: float = 0.25
    mapper_hidden_dim: int = 256
    predictor_hidden_dim: int = 256
    early_stopping_patience: int = 4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    grad_clip_norm: float = 1.0
    seed: int = 20260412
    frozen_tiles_per_slide: int = 1024
    max_slides_per_patient: int = 1
    slide_selection: str = "largest"
    encoder_batch_size: int = 64
    freeze_backbone: bool = True
    unfreeze_last_block: bool = False
    lambda_bce: float = 1.0
    lambda_mse: float = 0.25
    lambda_cos: float = 0.5
    log_interval_batches: int = 10
    device: str = "cuda"
    save_predictions: bool = True
    max_train_cases: int | None = None
    max_val_cases: int | None = None
    max_test_cases: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> OnlineFrozenDistillTrainConfig:
    with open(path, encoding="utf-8") as handle:
        return OnlineFrozenDistillTrainConfig(**yaml.safe_load(handle))


def build_dataloader(
    config: OnlineFrozenDistillTrainConfig,
    folds: list[int],
    split_name: str,
    is_training: bool,
    max_cases: int | None,
) -> DataLoader:
    preprocess = build_virchow2_preprocess()
    augmentor = (
        build_train_augmentor("frozen", ModalityAugmentationConfig())
        if is_training
        else build_eval_augmentor()
    )
    frozen_dataset = TCGALUADBinaryPatientBagDataset(
        patient_manifest_path=config.patient_manifest_path,
        split_path=config.split_path,
        wsi_root=config.frozen_wsi_root,
        coords_root=config.frozen_coords_root,
        modality="frozen",
        folds=folds,
        is_training=is_training,
        augmentor=augmentor,
        preprocess_transform=preprocess,
        tiles_per_slide=config.frozen_tiles_per_slide,
        max_slides_per_patient=config.max_slides_per_patient,
        slide_selection=config.slide_selection,
        seed=config.seed,
    )
    dataset = OnlineFrozenDistillDataset(
        frozen_dataset=frozen_dataset,
        ffpe_latent_path=Path(config.ffpe_latent_dir) / f"{split_name}.pt",
        max_cases=max_cases,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_training,
        num_workers=config.num_workers,
        collate_fn=collate_online_frozen_distill,
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


def load_frozen_checkpoint(model: OnlineFrozenToFFPEDistillationModel, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.frozen_model.load_state_dict(state, strict=False)
    print(
        f"Loaded frozen checkpoint: {path} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def loss_terms(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ffpe_embeddings: torch.Tensor,
    bce_criterion: torch.nn.Module,
    config: OnlineFrozenDistillTrainConfig,
) -> dict[str, torch.Tensor]:
    predicted = outputs["predicted_ffpe_embeddings"]
    mse = F.mse_loss(predicted, ffpe_embeddings)
    cosine = 1.0 - F.cosine_similarity(predicted, ffpe_embeddings, dim=-1).mean()
    bce = bce_criterion(outputs["logits"], labels)
    total = config.lambda_bce * bce + config.lambda_mse * mse + config.lambda_cos * cosine
    return {
        "total": total,
        "bce": bce,
        "mse": mse,
        "cosine": cosine,
    }


def evaluate(
    model: OnlineFrozenToFFPEDistillationModel,
    loader: DataLoader,
    bce_criterion: torch.nn.Module,
    config: OnlineFrozenDistillTrainConfig,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    all_case_ids: list[str] = []
    all_logits = []
    all_labels = []
    losses: list[float] = []
    bce_values: list[float] = []
    mse_values: list[float] = []
    cosine_values: list[float] = []
    per_case_mse = []
    per_case_cosine = []
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            outputs = model(move_bags_to_device(batch["bags"], device))
            labels = batch["labels"].to(device)
            ffpe_embeddings = batch["ffpe_embeddings"].to(device)
            batch_losses = loss_terms(outputs, labels, ffpe_embeddings, bce_criterion, config)

            predicted = outputs["predicted_ffpe_embeddings"]
            per_case_mse.append(F.mse_loss(predicted, ffpe_embeddings, reduction="none").mean(dim=-1).cpu())
            per_case_cosine.append((1.0 - F.cosine_similarity(predicted, ffpe_embeddings, dim=-1)).cpu())
            all_case_ids.extend(batch["case_ids"])
            all_logits.append(outputs["logits"].detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())
            losses.append(batch_losses["total"].item())
            bce_values.append(batch_losses["bce"].item())
            mse_values.append(batch_losses["mse"].item())
            cosine_values.append(batch_losses["cosine"].item())

            if (
                batch_idx == 1
                or batch_idx % config.log_interval_batches == 0
                or batch_idx == total_batches
            ):
                print(
                    f"[eval:{split_name}] batch={batch_idx}/{total_batches} "
                    f"loss={batch_losses['total'].item():.4f} "
                    f"bce={batch_losses['bce'].item():.4f} "
                    f"mse={batch_losses['mse'].item():.4f} "
                    f"cos={batch_losses['cosine'].item():.4f}",
                    flush=True,
                )

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "bce": sum(bce_values) / max(len(bce_values), 1),
        "mse": sum(mse_values) / max(len(mse_values), 1),
        "cosine": sum(cosine_values) / max(len(cosine_values), 1),
        "auc": binary_auc(logits, labels),
        "accuracy": binary_accuracy(logits, labels),
        "logits": logits,
        "labels": labels,
        "case_ids": all_case_ids,
        "per_case_mse": torch.cat(per_case_mse, dim=0),
        "per_case_cosine": torch.cat(per_case_cosine, dim=0),
    }


def train_one_epoch(
    model: OnlineFrozenToFFPEDistillationModel,
    loader: DataLoader,
    bce_criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: OnlineFrozenDistillTrainConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_values: list[float] = []
    bce_values: list[float] = []
    mse_values: list[float] = []
    cosine_values: list[float] = []
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(move_bags_to_device(batch["bags"], device))
        labels = batch["labels"].to(device)
        ffpe_embeddings = batch["ffpe_embeddings"].to(device)
        batch_losses = loss_terms(outputs, labels, ffpe_embeddings, bce_criterion, config)
        batch_losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        loss_values.append(batch_losses["total"].detach().item())
        bce_values.append(batch_losses["bce"].detach().item())
        mse_values.append(batch_losses["mse"].detach().item())
        cosine_values.append(batch_losses["cosine"].detach().item())

        if (
            batch_idx == 1
            or batch_idx % config.log_interval_batches == 0
            or batch_idx == total_batches
        ):
            print(
                f"[train] epoch={epoch} batch={batch_idx}/{total_batches} "
                f"loss={batch_losses['total'].item():.4f} "
                f"bce={batch_losses['bce'].item():.4f} "
                f"mse={batch_losses['mse'].item():.4f} "
                f"cos={batch_losses['cosine'].item():.4f}",
                flush=True,
            )

    n = max(len(loss_values), 1)
    return {
        "loss": sum(loss_values) / n,
        "bce": sum(bce_values) / n,
        "mse": sum(mse_values) / n,
        "cosine": sum(cosine_values) / n,
    }


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

    train_loader = build_dataloader(
        config,
        config.train_folds,
        "train",
        is_training=True,
        max_cases=config.max_train_cases,
    )
    val_loader = build_dataloader(
        config,
        config.val_folds,
        "val",
        is_training=False,
        max_cases=config.max_val_cases,
    )
    test_loader = build_dataloader(
        config,
        config.test_folds,
        "test",
        is_training=False,
        max_cases=config.max_test_cases,
    )

    print(
        "Run config: "
        f"modality={config.modality} "
        f"train_folds={config.train_folds} "
        f"val_folds={config.val_folds} "
        f"test_folds={config.test_folds} "
        f"frozen_tiles_per_slide={config.frozen_tiles_per_slide} "
        f"max_slides_per_patient={config.max_slides_per_patient} "
        f"slide_selection={config.slide_selection} "
        f"encoder_batch_size={config.encoder_batch_size} "
        f"epochs={config.epochs} "
        f"early_stopping_patience={config.early_stopping_patience} "
        f"scheduler_patience={config.scheduler_patience} "
        f"lambda_bce={config.lambda_bce} "
        f"lambda_mse={config.lambda_mse} "
        f"lambda_cos={config.lambda_cos} "
        f"ffpe_latent_dir={config.ffpe_latent_dir} "
        f"frozen_checkpoint={config.frozen_checkpoint} "
        f"output_dir={config.output_dir}",
        flush=True,
    )
    print(
        "Smoke limits: "
        f"max_train_cases={config.max_train_cases} "
        f"max_val_cases={config.max_val_cases} "
        f"max_test_cases={config.max_test_cases}",
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

    sample = train_loader.dataset[0]
    ffpe_dim = int(sample["ffpe_embedding"].shape[-1])
    model = OnlineFrozenToFFPEDistillationModel(
        frozen_model_config=PatientBinaryModelConfig(
            patient_hidden_dim=ffpe_dim,
            dropout=config.dropout,
            encoder_batch_size=config.encoder_batch_size,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_block=config.unfreeze_last_block,
        ),
        distill_config=OnlineFrozenDistillationConfig(
            frozen_dim=ffpe_dim,
            ffpe_dim=ffpe_dim,
            mapper_hidden_dim=config.mapper_hidden_dim,
            predictor_hidden_dim=config.predictor_hidden_dim,
            dropout=config.dropout,
        ),
    ).to(device)
    if config.frozen_checkpoint:
        load_frozen_checkpoint(model, config.frozen_checkpoint)

    pos_weight = infer_pos_weight(train_loader, device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        train_metrics = train_one_epoch(model, train_loader, bce_criterion, optimizer, config, device, epoch)
        val_metrics = evaluate(model, val_loader, bce_criterion, config, device, "val")
        scheduler.step(val_metrics["auc"])
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_bce": train_metrics["bce"],
                "train_mse": train_metrics["mse"],
                "train_cosine": train_metrics["cosine"],
                "val_loss": val_metrics["loss"],
                "val_bce": val_metrics["bce"],
                "val_mse": val_metrics["mse"],
                "val_cosine": val_metrics["cosine"],
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_bce={train_metrics['bce']:.4f} "
            f"train_mse={train_metrics['mse']:.4f} "
            f"train_cos={train_metrics['cosine']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_bce={val_metrics['bce']:.4f} "
            f"val_mse={val_metrics['mse']:.4f} "
            f"val_cos={val_metrics['cosine']:.4f} "
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
            print(
                f"Early stopping at epoch {epoch} after "
                f"{epochs_without_improvement} epochs without improvement.",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, bce_criterion, config, device, "test")

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
                "test_bce": test_metrics["bce"],
                "test_mse": test_metrics["mse"],
                "test_cosine": test_metrics["cosine"],
                "test_auc": test_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "pos_weight": float(pos_weight.item()),
            },
            handle,
            indent=2,
        )
    torch.save(model.state_dict(), output_dir / "model.pt")

    if config.save_predictions:
        probabilities = torch.sigmoid(test_metrics["logits"])
        with open(output_dir / "test_predictions.tsv", "w", encoding="utf-8") as handle:
            handle.write("case_id\tlogit\tprobability\tlabel\tffpe_latent_mse\tffpe_latent_cosine_loss\n")
            for case_id, logit, probability, label, mse, cosine in zip(
                test_metrics["case_ids"],
                test_metrics["logits"].tolist(),
                probabilities.tolist(),
                test_metrics["labels"].tolist(),
                test_metrics["per_case_mse"].tolist(),
                test_metrics["per_case_cosine"].tolist(),
            ):
                handle.write(f"{case_id}\t{logit}\t{probability}\t{int(label)}\t{mse}\t{cosine}\n")

    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best val AUC: {best_val_auc:.4f}", flush=True)
    print(f"Test loss: {test_metrics['loss']:.4f}", flush=True)
    print(f"Test BCE: {test_metrics['bce']:.4f}", flush=True)
    print(f"Test MSE: {test_metrics['mse']:.4f}", flush=True)
    print(f"Test cosine loss: {test_metrics['cosine']:.4f}", flush=True)
    print(f"Test AUC: {test_metrics['auc']:.4f}", flush=True)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}", flush=True)
    print(f"Saved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
