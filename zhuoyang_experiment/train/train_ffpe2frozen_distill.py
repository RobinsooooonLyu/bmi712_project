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
from zhuoyang_experiment.models.distill_heads import DINOProjectionHead


@dataclass
class DistillConfig:
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

    # tile/slide sizes
    ffpe_tiles_per_slide: int = 1024
    frozen_tiles_per_slide: int = 1024
    max_slides_per_patient: int = 1
    slide_selection: str = "largest"

    # model (mirrors baseline PatientBinaryModelConfig)
    dropout: float = 0.25
    encoder_batch_size: int = 64
    freeze_backbone: bool = True
    unfreeze_last_block: bool = False

    # distillation
    teacher_checkpoint: str | None = None
    distill_loss: str = "kl"  # "kl" | "mse_embed" | "dino_ce"
    num_prototypes: int = 256
    tau_t: float = 0.04
    tau_s: float = 0.1
    center_momentum: float = 0.9
    distill_weight: float = 1.0
    bce_weight: float = 1.0

    # optimization (names mirror baseline)
    batch_size: int = 1
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    early_stopping_patience: int = 4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    grad_clip_norm: float = 1.0
    seed: int = 20260403
    log_interval_batches: int = 10
    device: str = "cuda"
    save_predictions: bool = True


def load_config(path: str | Path) -> DistillConfig:
    with open(path, encoding="utf-8") as handle:
        return DistillConfig(**yaml.safe_load(handle))


def build_dataloader(config: DistillConfig, folds: list[int], is_training: bool) -> DataLoader:
    preprocess = build_virchow2_preprocess()
    if is_training:
        ffpe_aug = build_eval_augmentor()  # stable teacher targets
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


def build_patient_model(config: DistillConfig, device: torch.device) -> PatientBinaryModel:
    return PatientBinaryModel(
        PatientBinaryModelConfig(
            dropout=config.dropout,
            encoder_batch_size=config.encoder_batch_size,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_block=config.unfreeze_last_block,
        )
    ).to(device)


def load_teacher_checkpoint(teacher: PatientBinaryModel, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    print(
        f"Loaded teacher checkpoint: {path} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def compute_distill_loss(
    config: DistillConfig,
    t_out: dict[str, Any],
    s_out: dict[str, Any],
    t_proto: torch.Tensor | None,
    s_proto: torch.Tensor | None,
    center: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns (loss, new_center_batch_mean or None)."""
    if config.distill_loss == "kl":
        # Soft-target BCE against teacher sigmoid probability.
        t_p = torch.sigmoid(t_out["logits"].detach() / config.tau_t)
        loss = F.binary_cross_entropy_with_logits(s_out["logits"] / config.tau_s, t_p)
        return loss, None

    if config.distill_loss == "mse_embed":
        loss = F.mse_loss(
            s_out["patient_embeddings"],
            t_out["patient_embeddings"].detach(),
        )
        return loss, None

    if config.distill_loss == "dino_ce":
        assert t_proto is not None and s_proto is not None and center is not None
        t_dist = F.softmax((t_proto.detach() - center) / config.tau_t, dim=-1)
        s_logp = F.log_softmax(s_proto / config.tau_s, dim=-1)
        loss = -(t_dist * s_logp).sum(dim=-1).mean()
        return loss, t_proto.detach().mean(dim=0)

    raise ValueError(f"Unknown distill_loss: {config.distill_loss}")


def forward_paired(
    teacher: PatientBinaryModel,
    student: PatientBinaryModel,
    teacher_proj: DINOProjectionHead | None,
    student_proj: DINOProjectionHead | None,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    ffpe_bags = move_bags_to_device(batch["ffpe_bags"], device)
    frozen_bags = move_bags_to_device(batch["frozen_bags"], device)
    with torch.no_grad():
        t_out = teacher(ffpe_bags)
        t_proto = teacher_proj(t_out["patient_embeddings"]) if teacher_proj is not None else None
    s_out = student(frozen_bags)
    s_proto = student_proj(s_out["patient_embeddings"]) if student_proj is not None else None
    return t_out, s_out, t_proto, s_proto


def train_one_epoch(
    config: DistillConfig,
    teacher: PatientBinaryModel,
    student: PatientBinaryModel,
    teacher_proj: DINOProjectionHead | None,
    student_proj: DINOProjectionHead | None,
    loader: DataLoader,
    bce_criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    center: torch.Tensor | None,
    epoch: int,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    if student_proj is not None:
        student_proj.train()
    if teacher_proj is not None:
        teacher_proj.eval()

    total_losses = []
    bce_losses = []
    distill_losses = []
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        t_out, s_out, t_proto, s_proto = forward_paired(
            teacher, student, teacher_proj, student_proj, batch, device
        )
        labels = batch["labels"].to(device)
        loss_bce = bce_criterion(s_out["logits"], labels)
        loss_distill, batch_center = compute_distill_loss(
            config, t_out, s_out, t_proto, s_proto, center
        )
        loss = config.bce_weight * loss_bce + config.distill_weight * loss_distill
        loss.backward()
        trainable = [p for p in student.parameters() if p.requires_grad]
        if student_proj is not None:
            trainable += list(student_proj.parameters())
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.grad_clip_norm)
        optimizer.step()

        if center is not None and batch_center is not None:
            center.mul_(config.center_momentum).add_(
                batch_center, alpha=(1.0 - config.center_momentum)
            )

        total_losses.append(loss.detach().item())
        bce_losses.append(loss_bce.detach().item())
        distill_losses.append(loss_distill.detach().item())

        if (
            batch_idx == 1
            or batch_idx % config.log_interval_batches == 0
            or batch_idx == total_batches
        ):
            print(
                f"[train] epoch={epoch} batch={batch_idx}/{total_batches} "
                f"loss={loss.item():.4f} "
                f"bce={loss_bce.item():.4f} "
                f"distill={loss_distill.item():.4f}",
                flush=True,
            )

    n = max(len(total_losses), 1)
    return {
        "train_loss": sum(total_losses) / n,
        "train_loss_bce": sum(bce_losses) / n,
        "train_loss_distill": sum(distill_losses) / n,
    }


def evaluate(
    student: PatientBinaryModel,
    loader: DataLoader,
    bce_criterion: torch.nn.Module,
    device: torch.device,
    split_name: str,
    log_interval_batches: int,
) -> dict[str, Any]:
    """Evaluate student on frozen bags only (pure frozen inference path)."""
    student.eval()
    all_logits = []
    all_labels = []
    all_case_ids = []
    losses = []
    with torch.no_grad():
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            frozen_bags = move_bags_to_device(batch["frozen_bags"], device)
            outputs = student(frozen_bags)
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
        f"distill_loss={config.distill_loss} "
        f"teacher_checkpoint={config.teacher_checkpoint} "
        f"bce_weight={config.bce_weight} distill_weight={config.distill_weight} "
        f"tau_t={config.tau_t} tau_s={config.tau_s} "
        f"train_folds={config.train_folds} val_folds={config.val_folds} "
        f"test_folds={config.test_folds} "
        f"ffpe_tiles_per_slide={config.ffpe_tiles_per_slide} "
        f"frozen_tiles_per_slide={config.frozen_tiles_per_slide} "
        f"epochs={config.epochs} output_dir={config.output_dir}",
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

    teacher = build_patient_model(config, device)
    student = build_patient_model(config, device)
    if config.teacher_checkpoint:
        load_teacher_checkpoint(teacher, config.teacher_checkpoint)
    freeze_module(teacher)

    teacher_proj: DINOProjectionHead | None = None
    student_proj: DINOProjectionHead | None = None
    center: torch.Tensor | None = None
    if config.distill_loss == "dino_ce":
        proj_in_dim = teacher.config.patient_hidden_dim
        teacher_proj = DINOProjectionHead(
            in_dim=proj_in_dim, num_prototypes=config.num_prototypes
        ).to(device)
        student_proj = DINOProjectionHead(
            in_dim=proj_in_dim, num_prototypes=config.num_prototypes
        ).to(device)
        freeze_module(teacher_proj)
        center = torch.zeros(config.num_prototypes, device=device)

    pos_weight = infer_pos_weight(train_loader, device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if student_proj is not None:
        trainable_params += list(student_proj.parameters())
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
    best_proj_state = None
    best_center = None
    best_val_auc = float("-inf")
    best_epoch = -1
    history = []
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(
            config,
            teacher,
            student,
            teacher_proj,
            student_proj,
            train_loader,
            bce_criterion,
            optimizer,
            device,
            center,
            epoch,
        )
        val_metrics = evaluate(
            student,
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
                "train_loss_bce": train_metrics["train_loss_bce"],
                "train_loss_distill": train_metrics["train_loss_distill"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"bce={train_metrics['train_loss_bce']:.4f} "
            f"distill={train_metrics['train_loss_distill']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"lr={current_lr:.2e}",
            flush=True,
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in student.state_dict().items()}
            if student_proj is not None:
                best_proj_state = {k: v.cpu() for k, v in student_proj.state_dict().items()}
            if center is not None:
                best_center = center.detach().cpu().clone()
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
        student.load_state_dict(best_state)
    if best_proj_state is not None and student_proj is not None:
        student_proj.load_state_dict(best_proj_state)
    test_metrics = evaluate(
        student, test_loader, bce_criterion, device, "test", config.log_interval_batches
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

    model_payload: dict[str, Any] = {"student": student.state_dict()}
    if student_proj is not None:
        model_payload["student_proj"] = student_proj.state_dict()
    if best_center is not None:
        model_payload["center"] = best_center
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
