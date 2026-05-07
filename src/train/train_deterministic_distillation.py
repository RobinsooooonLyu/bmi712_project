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

from src.data.paired_latent_dataset import PairedLatentDataset, collate_paired_latents
from src.models.deterministic_distillation import (
    DeterministicDistillationConfig,
    DeterministicDistillationModel,
)


@dataclass
class DistillationTrainConfig:
    frozen_latent_dir: str
    ffpe_latent_dir: str
    output_dir: str
    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    dropout: float = 0.25
    mapper_hidden_dim: int = 256
    predictor_hidden_dim: int = 256
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    grad_clip_norm: float = 1.0
    seed: int = 20260412
    lambda_recon_mse: float = 1.0
    lambda_recon_cos: float = 1.0
    lambda_bce: float = 1.0
    log_interval_batches: int = 1
    device: str = "cuda"
    save_predictions: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> DistillationTrainConfig:
    with open(path, encoding="utf-8") as handle:
        return DistillationTrainConfig(**yaml.safe_load(handle))


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


def build_dataloader(config: DistillationTrainConfig, split_name: str, is_training: bool) -> DataLoader:
    dataset = PairedLatentDataset(
        frozen_path=Path(config.frozen_latent_dir) / f"{split_name}.pt",
        ffpe_path=Path(config.ffpe_latent_dir) / f"{split_name}.pt",
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_training,
        num_workers=config.num_workers,
        collate_fn=collate_paired_latents,
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


def loss_terms(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ffpe_embeddings: torch.Tensor,
    bce_criterion: torch.nn.Module,
    config: DistillationTrainConfig,
) -> dict[str, torch.Tensor]:
    recon_mse = F.mse_loss(outputs["predicted_ffpe_embedding"], ffpe_embeddings)
    recon_cos = 1.0 - F.cosine_similarity(outputs["predicted_ffpe_embedding"], ffpe_embeddings, dim=-1).mean()
    bce = bce_criterion(outputs["logits"], labels)
    total = (
        config.lambda_recon_mse * recon_mse
        + config.lambda_recon_cos * recon_cos
        + config.lambda_bce * bce
    )
    return {
        "total": total,
        "recon_mse": recon_mse,
        "recon_cos": recon_cos,
        "bce": bce,
    }


def evaluate(
    model: DeterministicDistillationModel,
    loader: DataLoader,
    bce_criterion: torch.nn.Module,
    config: DistillationTrainConfig,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    all_case_ids: list[str] = []
    all_logits = []
    all_labels = []
    losses: list[float] = []
    recon_mse_values: list[float] = []
    recon_cos_values: list[float] = []
    bce_values: list[float] = []
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            frozen_embeddings = batch["frozen_embeddings"].to(device)
            ffpe_embeddings = batch["ffpe_embeddings"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(frozen_embeddings)
            batch_losses = loss_terms(outputs, labels, ffpe_embeddings, bce_criterion, config)

            all_case_ids.extend(batch["case_ids"])
            all_logits.append(outputs["logits"].detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())
            losses.append(batch_losses["total"].item())
            recon_mse_values.append(batch_losses["recon_mse"].item())
            recon_cos_values.append(batch_losses["recon_cos"].item())
            bce_values.append(batch_losses["bce"].item())

            if batch_idx == 1 or batch_idx % config.log_interval_batches == 0 or batch_idx == total_batches:
                print(
                    f"[eval:{split_name}] batch={batch_idx}/{total_batches} "
                    f"loss={batch_losses['total'].item():.4f} "
                    f"recon_mse={batch_losses['recon_mse'].item():.4f} "
                    f"recon_cos={batch_losses['recon_cos'].item():.4f} "
                    f"bce={batch_losses['bce'].item():.4f}",
                    flush=True,
                )

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "recon_mse": sum(recon_mse_values) / max(len(recon_mse_values), 1),
        "recon_cos": sum(recon_cos_values) / max(len(recon_cos_values), 1),
        "bce": sum(bce_values) / max(len(bce_values), 1),
        "auc": binary_auc(logits, labels),
        "accuracy": binary_accuracy(logits, labels),
        "logits": logits,
        "labels": labels,
        "case_ids": all_case_ids,
    }


def train_one_epoch(
    model: DeterministicDistillationModel,
    loader: DataLoader,
    bce_criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DistillationTrainConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_values: list[float] = []
    recon_mse_values: list[float] = []
    recon_cos_values: list[float] = []
    bce_values: list[float] = []
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        frozen_embeddings = batch["frozen_embeddings"].to(device)
        ffpe_embeddings = batch["ffpe_embeddings"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(frozen_embeddings)
        batch_losses = loss_terms(outputs, labels, ffpe_embeddings, bce_criterion, config)
        batch_losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        loss_values.append(batch_losses["total"].item())
        recon_mse_values.append(batch_losses["recon_mse"].item())
        recon_cos_values.append(batch_losses["recon_cos"].item())
        bce_values.append(batch_losses["bce"].item())

        if batch_idx == 1 or batch_idx % config.log_interval_batches == 0 or batch_idx == total_batches:
            print(
                f"[train] epoch={epoch} batch={batch_idx}/{total_batches} "
                f"loss={batch_losses['total'].item():.4f} "
                f"recon_mse={batch_losses['recon_mse'].item():.4f} "
                f"recon_cos={batch_losses['recon_cos'].item():.4f} "
                f"bce={batch_losses['bce'].item():.4f}",
                flush=True,
            )

    return {
        "loss": sum(loss_values) / max(len(loss_values), 1),
        "recon_mse": sum(recon_mse_values) / max(len(recon_mse_values), 1),
        "recon_cos": sum(recon_cos_values) / max(len(recon_cos_values), 1),
        "bce": sum(bce_values) / max(len(bce_values), 1),
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

    train_loader = build_dataloader(config, "train", is_training=True)
    val_loader = build_dataloader(config, "val", is_training=False)
    test_loader = build_dataloader(config, "test", is_training=False)
    print(
        "Run config: "
        f"frozen_latent_dir={config.frozen_latent_dir} "
        f"ffpe_latent_dir={config.ffpe_latent_dir} "
        f"epochs={config.epochs} "
        f"mapper_hidden_dim={config.mapper_hidden_dim} "
        f"predictor_hidden_dim={config.predictor_hidden_dim} "
        f"lambda_recon_mse={config.lambda_recon_mse} "
        f"lambda_recon_cos={config.lambda_recon_cos} "
        f"lambda_bce={config.lambda_bce} "
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

    sample = train_loader.dataset[0]
    model = DeterministicDistillationModel(
        DeterministicDistillationConfig(
            frozen_dim=int(sample["frozen_embedding"].shape[-1]),
            ffpe_dim=int(sample["ffpe_embedding"].shape[-1]),
            mapper_hidden_dim=config.mapper_hidden_dim,
            predictor_hidden_dim=config.predictor_hidden_dim,
            dropout=config.dropout,
        )
    ).to(device)
    pos_weight = infer_pos_weight(train_loader, device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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
                "train_recon_mse": train_metrics["recon_mse"],
                "train_recon_cos": train_metrics["recon_cos"],
                "train_bce": train_metrics["bce"],
                "val_loss": val_metrics["loss"],
                "val_recon_mse": val_metrics["recon_mse"],
                "val_recon_cos": val_metrics["recon_cos"],
                "val_bce": val_metrics["bce"],
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_recon_mse={train_metrics['recon_mse']:.4f} "
            f"train_recon_cos={train_metrics['recon_cos']:.4f} "
            f"train_bce={train_metrics['bce']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_recon_mse={val_metrics['recon_mse']:.4f} "
            f"val_recon_cos={val_metrics['recon_cos']:.4f} "
            f"val_bce={val_metrics['bce']:.4f} "
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
                "test_recon_mse": test_metrics["recon_mse"],
                "test_recon_cos": test_metrics["recon_cos"],
                "test_bce": test_metrics["bce"],
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
            handle.write("case_id\tlogit\tprobability\tlabel\n")
            for case_id, logit, probability, label in zip(
                test_metrics["case_ids"],
                test_metrics["logits"].tolist(),
                probabilities.tolist(),
                test_metrics["labels"].tolist(),
            ):
                handle.write(f"{case_id}\t{logit}\t{probability}\t{int(label)}\n")

    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best val AUC: {best_val_auc:.4f}", flush=True)
    print(f"Test loss: {test_metrics['loss']:.4f}", flush=True)
    print(f"Test AUC: {test_metrics['auc']:.4f}", flush=True)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}", flush=True)
    print(f"Saved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
