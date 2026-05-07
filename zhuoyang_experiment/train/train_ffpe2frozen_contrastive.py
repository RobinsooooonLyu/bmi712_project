"""Contrastive FFPE <-> frozen patient-embedding alignment.

Sibling experiment to ``train_ffpe2frozen_distill``. Instead of pushing the
FFPE teacher output into a frozen-section student via KL/MSE/DINO, both
modalities are encoded in parallel and their patient embeddings are pulled
together with a symmetric InfoNCE loss (paired patient = positive,
in-batch other patients = negatives). Each modality also keeps its own
BCE term against the binary label so neither tower collapses to a trivial
shortcut.

Backbone notes:
  - Virchow2 backbone is frozen for both towers to keep wall-clock close
    to the existing distillation experiment. Only the ABMIL slide/patient
    heads, the binary logit heads, and a small projection head are
    trainable.
  - Evaluation uses the frozen tower only, so val / test metrics are
    directly comparable to the frozen high-risk baseline and the
    ffpe2frozen distillation run.

Re-uses ``PairedFFPEFrozenPatientDataset`` and the baseline helpers
(``move_bags_to_device``, ``binary_auc``, ``binary_accuracy``,
``set_seed``) verbatim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.data.augmentations import (
    ModalityAugmentationConfig,
    build_eval_augmentor,
    build_train_augmentor,
)
from src.models.patient_binary_model import PatientBinaryModel, PatientBinaryModelConfig
from src.models.virchow2_encoder import build_virchow2_preprocess
from src.train.train_binary_baseline import (
    binary_accuracy,
    binary_auc,
    move_bags_to_device,
    set_seed,
)

from zhuoyang_experiment.data.paired_wsi_dataset import (
    PairedFFPEFrozenPatientDataset,
    collate_paired_patient_bags,
)
from zhuoyang_experiment.models.distill_heads import ContrastiveProjectionHead


@dataclass
class ContrastiveConfig:
    modality: str = "frozen_contrastive"

    # data
    patient_manifest_path: str = ""
    split_path: str = ""
    ffpe_wsi_root: str = ""
    ffpe_coords_root: str = ""
    frozen_wsi_root: str = ""
    frozen_coords_root: str = ""
    output_dir: str = ""

    # folds
    train_folds: list[int] = field(default_factory=list)
    val_folds: list[int] = field(default_factory=list)
    test_folds: list[int] = field(default_factory=list)

    # tile/slide sizes — halved relative to the distill experiment so the
    # larger contrastive batch keeps total per-epoch tile work comparable.
    ffpe_tiles_per_slide: int = 512
    frozen_tiles_per_slide: int = 512
    max_slides_per_patient: int = 1
    slide_selection: str = "largest"

    # model (mirrors baseline PatientBinaryModelConfig)
    dropout: float = 0.25
    encoder_batch_size: int = 64
    freeze_backbone: bool = True
    unfreeze_last_block: bool = False

    # contrastive
    proj_hidden_dim: int = 256
    proj_dim: int = 128
    contrastive_temperature: float = 0.1
    contrastive_weight: float = 1.0
    bce_frozen_weight: float = 1.0
    bce_ffpe_weight: float = 1.0
    share_projection_head: bool = False

    # optimization
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    early_stopping_patience: int = 4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    grad_clip_norm: float = 1.0
    seed: int = 20260403
    log_interval_batches: int = 10
    device: str = "cuda"
    save_predictions: bool = True


def load_config(path: str | Path) -> ContrastiveConfig:
    with open(path, encoding="utf-8") as handle:
        return ContrastiveConfig(**yaml.safe_load(handle))


def build_dataloader(config: ContrastiveConfig, folds: list[int], is_training: bool) -> DataLoader:
    preprocess = build_virchow2_preprocess()
    if is_training:
        ffpe_aug = build_train_augmentor("ffpe", ModalityAugmentationConfig())
        frozen_aug = build_train_augmentor("frozen", ModalityAugmentationConfig())
    else:
        ffpe_aug = build_eval_augmentor()
        frozen_aug = build_eval_augmentor()

    dataset = PairedFFPEFrozenPatientDataset(
        patient_manifest_path=config.patient_manifest_path,
        split_path=config.split_path,
        ffpe_wsi_root=config.ffpe_wsi_root,
        ffpe_coords_root=config.ffpe_coords_root,
        frozen_wsi_root=config.frozen_wsi_root,
        frozen_coords_root=config.frozen_coords_root,
        folds=folds,
        is_training=is_training,
        ffpe_augmentor=ffpe_aug,
        frozen_augmentor=frozen_aug,
        preprocess_transform=preprocess,
        ffpe_tiles_per_slide=config.ffpe_tiles_per_slide,
        frozen_tiles_per_slide=config.frozen_tiles_per_slide,
        max_slides_per_patient=config.max_slides_per_patient,
        slide_selection=config.slide_selection,
        seed=config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_training,
        num_workers=config.num_workers,
        collate_fn=collate_paired_patient_bags,
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


def build_patient_model(config: ContrastiveConfig, device: torch.device) -> PatientBinaryModel:
    return PatientBinaryModel(
        PatientBinaryModelConfig(
            dropout=config.dropout,
            encoder_batch_size=config.encoder_batch_size,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_block=config.unfreeze_last_block,
        )
    ).to(device)


def info_nce_symmetric(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE on L2-normalized embeddings.

    Positive pair = (z_a[i], z_b[i]); negatives = z_b[j!=i] (and vice versa).
    Falls back to 0 when the batch has no negatives (size 1) since the loss
    is undefined there.
    """
    if z_a.size(0) < 2:
        return z_a.new_zeros(())
    logits_ab = z_a @ z_b.t() / temperature
    logits_ba = z_b @ z_a.t() / temperature
    targets = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets))


def train_one_epoch(
    config: ContrastiveConfig,
    ffpe_model: PatientBinaryModel,
    frozen_model: PatientBinaryModel,
    ffpe_proj: ContrastiveProjectionHead,
    frozen_proj: ContrastiveProjectionHead,
    loader: DataLoader,
    bce_criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    ffpe_model.train()
    frozen_model.train()
    ffpe_proj.train()
    frozen_proj.train()

    total_losses: list[float] = []
    bce_frozen_losses: list[float] = []
    bce_ffpe_losses: list[float] = []
    contrastive_losses: list[float] = []
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        ffpe_bags = move_bags_to_device(batch["ffpe_bags"], device)
        frozen_bags = move_bags_to_device(batch["frozen_bags"], device)
        labels = batch["labels"].to(device)

        ffpe_out = ffpe_model(ffpe_bags)
        frozen_out = frozen_model(frozen_bags)

        z_ffpe = ffpe_proj(ffpe_out["patient_embeddings"])
        z_frozen = frozen_proj(frozen_out["patient_embeddings"])

        loss_bce_frozen = bce_criterion(frozen_out["logits"], labels)
        loss_bce_ffpe = bce_criterion(ffpe_out["logits"], labels)
        loss_contrastive = info_nce_symmetric(
            z_ffpe, z_frozen, config.contrastive_temperature
        )
        loss = (
            config.bce_frozen_weight * loss_bce_frozen
            + config.bce_ffpe_weight * loss_bce_ffpe
            + config.contrastive_weight * loss_contrastive
        )
        loss.backward()
        trainable = [p for p in ffpe_model.parameters() if p.requires_grad]
        trainable += [p for p in frozen_model.parameters() if p.requires_grad]
        trainable += list(ffpe_proj.parameters())
        if frozen_proj is not ffpe_proj:
            trainable += list(frozen_proj.parameters())
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.grad_clip_norm)
        optimizer.step()

        total_losses.append(loss.detach().item())
        bce_frozen_losses.append(loss_bce_frozen.detach().item())
        bce_ffpe_losses.append(loss_bce_ffpe.detach().item())
        contrastive_losses.append(loss_contrastive.detach().item())

        if (
            batch_idx == 1
            or batch_idx % config.log_interval_batches == 0
            or batch_idx == total_batches
        ):
            print(
                f"[train] epoch={epoch} batch={batch_idx}/{total_batches} "
                f"loss={loss.item():.4f} "
                f"bce_frozen={loss_bce_frozen.item():.4f} "
                f"bce_ffpe={loss_bce_ffpe.item():.4f} "
                f"contrastive={loss_contrastive.item():.4f}",
                flush=True,
            )

    n = max(len(total_losses), 1)
    return {
        "train_loss": sum(total_losses) / n,
        "train_loss_bce_frozen": sum(bce_frozen_losses) / n,
        "train_loss_bce_ffpe": sum(bce_ffpe_losses) / n,
        "train_loss_contrastive": sum(contrastive_losses) / n,
    }


def evaluate(
    frozen_model: PatientBinaryModel,
    loader: DataLoader,
    bce_criterion: torch.nn.Module,
    device: torch.device,
    split_name: str,
    log_interval_batches: int,
) -> dict[str, Any]:
    """Evaluate the frozen-section tower only (deployment path)."""
    frozen_model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_case_ids: list[str] = []
    losses: list[float] = []
    with torch.no_grad():
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            frozen_bags = move_bags_to_device(batch["frozen_bags"], device)
            outputs = frozen_model(frozen_bags)
            labels = batch["labels"].to(device)
            loss = bce_criterion(outputs["logits"], labels)
            losses.append(loss.item())
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(batch["labels"].cpu())
            all_case_ids.extend(batch["case_ids"])
            if (
                batch_idx == 1
                or batch_idx % log_interval_batches == 0
                or batch_idx == total_batches
            ):
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
        "case_ids": all_case_ids,
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
        print(
            f"CUDA current device: {torch.cuda.current_device()} "
            f"{torch.cuda.get_device_name()}",
            flush=True,
        )

    train_loader = build_dataloader(config, config.train_folds, is_training=True)
    val_loader = build_dataloader(config, config.val_folds, is_training=False)
    test_loader = build_dataloader(config, config.test_folds, is_training=False)

    print(
        "Run config: "
        f"contrastive_temperature={config.contrastive_temperature} "
        f"contrastive_weight={config.contrastive_weight} "
        f"bce_frozen_weight={config.bce_frozen_weight} "
        f"bce_ffpe_weight={config.bce_ffpe_weight} "
        f"share_projection_head={config.share_projection_head} "
        f"proj_dim={config.proj_dim} "
        f"train_folds={config.train_folds} val_folds={config.val_folds} "
        f"test_folds={config.test_folds} "
        f"ffpe_tiles_per_slide={config.ffpe_tiles_per_slide} "
        f"frozen_tiles_per_slide={config.frozen_tiles_per_slide} "
        f"batch_size={config.batch_size} epochs={config.epochs} "
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

    ffpe_model = build_patient_model(config, device)
    frozen_model = build_patient_model(config, device)

    ffpe_proj = ContrastiveProjectionHead(
        in_dim=ffpe_model.config.patient_hidden_dim,
        hidden_dim=config.proj_hidden_dim,
        proj_dim=config.proj_dim,
    ).to(device)
    if config.share_projection_head:
        frozen_proj = ffpe_proj
    else:
        frozen_proj = ContrastiveProjectionHead(
            in_dim=frozen_model.config.patient_hidden_dim,
            hidden_dim=config.proj_hidden_dim,
            proj_dim=config.proj_dim,
        ).to(device)

    pos_weight = infer_pos_weight(train_loader, device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainable_params = [p for p in ffpe_model.parameters() if p.requires_grad]
    trainable_params += [p for p in frozen_model.parameters() if p.requires_grad]
    trainable_params += list(ffpe_proj.parameters())
    if frozen_proj is not ffpe_proj:
        trainable_params += list(frozen_proj.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
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
    best_ffpe_state = None
    best_ffpe_proj_state = None
    best_frozen_proj_state = None
    best_val_auc = float("-inf")
    best_epoch = -1
    history = []
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(
            config,
            ffpe_model,
            frozen_model,
            ffpe_proj,
            frozen_proj,
            train_loader,
            bce_criterion,
            optimizer,
            device,
            epoch,
        )
        val_metrics = evaluate(
            frozen_model,
            val_loader,
            bce_criterion,
            device,
            "val",
            config.log_interval_batches,
        )
        scheduler.step(val_metrics["auc"])
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "train_loss_bce_frozen": train_metrics["train_loss_bce_frozen"],
                "train_loss_bce_ffpe": train_metrics["train_loss_bce_ffpe"],
                "train_loss_contrastive": train_metrics["train_loss_contrastive"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"bce_frozen={train_metrics['train_loss_bce_frozen']:.4f} "
            f"bce_ffpe={train_metrics['train_loss_bce_ffpe']:.4f} "
            f"contrastive={train_metrics['train_loss_contrastive']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"lr={current_lr:.2e}",
            flush=True,
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in frozen_model.state_dict().items()}
            best_ffpe_state = {k: v.cpu() for k, v in ffpe_model.state_dict().items()}
            best_ffpe_proj_state = {k: v.cpu() for k, v in ffpe_proj.state_dict().items()}
            if frozen_proj is not ffpe_proj:
                best_frozen_proj_state = {
                    k: v.cpu() for k, v in frozen_proj.state_dict().items()
                }
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
        frozen_model.load_state_dict(best_state)
    test_metrics = evaluate(
        frozen_model, test_loader, bce_criterion, device, "test", config.log_interval_batches
    )

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

    model_payload: dict[str, Any] = {
        "student": frozen_model.state_dict(),
        "ffpe": best_ffpe_state if best_ffpe_state is not None else ffpe_model.state_dict(),
        "ffpe_proj": (
            best_ffpe_proj_state
            if best_ffpe_proj_state is not None
            else ffpe_proj.state_dict()
        ),
    }
    if frozen_proj is not ffpe_proj:
        model_payload["frozen_proj"] = (
            best_frozen_proj_state
            if best_frozen_proj_state is not None
            else frozen_proj.state_dict()
        )
    torch.save(model_payload, output_dir / "model.pt")

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

    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best val AUC: {best_val_auc:.4f}", flush=True)
    print(f"Test loss: {test_metrics['loss']:.4f}", flush=True)
    print(f"Test AUC: {test_metrics['auc']:.4f}", flush=True)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}", flush=True)
    print(f"Saved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
