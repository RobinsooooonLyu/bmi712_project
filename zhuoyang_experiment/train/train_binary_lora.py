"""Train the patient-level binary classifier with a LoRA-adapted Virchow2 backbone.

This mirrors ``src.train.train_binary_baseline`` exactly, with one change:
after constructing the ``PatientBinaryModel`` we inject LoRA adapters into
the last few transformer blocks of the Virchow2 encoder and only those LoRA
weights (plus the ABMIL heads and the logit head, which are trainable in the
baseline already) are optimised. The backbone itself stays frozen.

Why LoRA here:
  - Full-backbone fine-tuning of a ViT-H is expensive in compute and memory.
  - The baseline freezes the backbone entirely; that's cheap but leaves the
    pretrained features unchanged.
  - LoRA adds a tiny number of trainable params (~tens of thousands at rank 4
    on 2 blocks) so we can adapt the encoder slightly to FFPE / frozen TCGA
    LUAD without paying full fine-tune cost.

All baseline outputs (config.json, history.json, metrics.json, model.pt,
test_predictions.tsv, optional latent_exports/) are produced verbatim, with
the same filenames, so downstream scripts (e.g. summarize_binary_cv_results)
work unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml

from src.models.patient_binary_model import PatientBinaryModel, PatientBinaryModelConfig
from src.train.train_binary_baseline import (
    BinaryBaselineConfig,
    build_dataloader,
    evaluate,
    infer_pos_weight,
    set_seed,
    summarize_labels,
    train_one_epoch,
    write_latent_export,
)

from zhuoyang_experiment.models.lora import LoRAConfig, apply_lora_to_virchow2


@dataclass
class BinaryLoRAConfig(BinaryBaselineConfig):
    """Same fields as the baseline config plus LoRA hyperparameters.

    Defaults are deliberately conservative so a 5-fold run finishes in a
    reasonable amount of compute on a single GPU. Override in YAML if
    you want to push harder (more blocks, higher rank, also adapt
    ``attn.proj``, etc.).
    """

    # LoRA config (see zhuoyang_experiment.models.lora.LoRAConfig).
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_last_n_blocks: int = 2
    lora_target_modules: list[str] = field(default_factory=lambda: ["attn.qkv"])

    # Per-parameter-group learning rates. Both default to ``None``, in which
    # case the baseline ``learning_rate`` field is used for both groups (so
    # behavior matches the baseline out of the box). Set them in YAML to
    # decouple — e.g. a smaller LR on the LoRA-adapted backbone weights and
    # a larger LR on the freshly-initialised ABMIL/logit heads.
    lora_learning_rate: float | None = None
    head_learning_rate: float | None = None


def load_config(path: str | Path) -> BinaryLoRAConfig:
    with open(path, encoding="utf-8") as handle:
        return BinaryLoRAConfig(**yaml.safe_load(handle))


def build_model(config: BinaryLoRAConfig, device: torch.device) -> tuple[PatientBinaryModel, dict]:
    # Build the same PatientBinaryModel the baseline uses, with the backbone
    # fully frozen. We then replace selected Linear layers inside the last few
    # ViT blocks with LoRALinear wrappers, which makes a small number of
    # backbone weights trainable via low-rank updates.
    model = PatientBinaryModel(
        PatientBinaryModelConfig(
            dropout=config.dropout,
            encoder_batch_size=config.encoder_batch_size,
            freeze_backbone=True,
            unfreeze_last_block=False,
        )
    )
    lora_summary = apply_lora_to_virchow2(
        model.encoder,
        LoRAConfig(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            last_n_blocks=config.lora_last_n_blocks,
            target_modules=list(config.lora_target_modules),
        ),
    )
    model = model.to(device)
    return model, lora_summary


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
        f"lora_rank={config.lora_rank} "
        f"lora_alpha={config.lora_alpha} "
        f"lora_dropout={config.lora_dropout} "
        f"lora_last_n_blocks={config.lora_last_n_blocks} "
        f"lora_target_modules={config.lora_target_modules} "
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

    model, lora_summary = build_model(config, device)
    n_trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        "LoRA summary: "
        f"blocks_adapted={lora_summary['lora_blocks_adapted']} "
        f"layers_wrapped={lora_summary['lora_layers_wrapped']} "
        f"layers_skipped={lora_summary['lora_layers_skipped']} "
        f"target_block_indices={lora_summary['target_block_indices']} "
        f"encoder_trainable={lora_summary['encoder_trainable_params']} "
        f"encoder_total={lora_summary['encoder_total_params']} "
        f"model_trainable={n_trainable_total} model_total={n_total}",
        flush=True,
    )

    pos_weight = infer_pos_weight(train_loader, device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Two parameter groups so the coworker can tune them independently:
    #   - "lora": the trainable params inside the encoder (LoRA A/B weights).
    #   - "heads": the ABMIL stacks + logit head (everything outside encoder).
    # Both groups default to ``config.learning_rate`` so behaviour matches
    # the baseline single-LR setup unless ``lora_learning_rate`` /
    # ``head_learning_rate`` are set explicitly in the YAML.
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    lora_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) in encoder_param_ids
    ]
    head_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in encoder_param_ids
    ]
    lora_lr = config.lora_learning_rate if config.lora_learning_rate is not None else config.learning_rate
    head_lr = config.head_learning_rate if config.head_learning_rate is not None else config.learning_rate
    print(
        f"Optimizer param groups: "
        f"lora_params={sum(p.numel() for p in lora_params)} (lr={lora_lr}) "
        f"head_params={sum(p.numel() for p in head_params)} (lr={head_lr})",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_lr, "name": "lora"},
            {"params": head_params, "lr": head_lr, "name": "heads"},
        ],
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
        # Both groups are scaled by ReduceLROnPlateau in lock-step, but log
        # each so the per-group split is visible in history.json.
        lr_by_group = {g.get("name", f"group{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)}
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": current_lr,
                "learning_rate_by_group": lr_by_group,
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
                "lora_layers_wrapped": lora_summary["lora_layers_wrapped"],
                "encoder_trainable_params": lora_summary["encoder_trainable_params"],
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
        split_to_metrics = {"val": val_metrics, "test": test_metrics}
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
