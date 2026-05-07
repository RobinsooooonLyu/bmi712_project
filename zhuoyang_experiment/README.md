# FFPE → Frozen Cross-Modal Distillation

Teacher-student experiment that pushes FFPE-level discriminative structure into
a frozen-section classifier. A teacher network processes the FFPE bag for a
patient, a student network processes the paired frozen bag, and the student is
trained with a supervised BCE term plus a distillation term that matches the
teacher's patient-level output. Both networks use the frozen Virchow2 backbone
(identical to the baselines); only the ABMIL heads and the binary logit head
are trained. This is the `PatientBinaryModel` from `src/models/` reused
verbatim, so results are directly comparable to the frozen high-risk baseline.

## Why

FFPE slides carry cleaner morphology than frozen sections and the FFPE
high-risk baseline outperforms the frozen baseline on the same task. Frozen
sections, however, are the intra-operative modality: FFPE is not available at
decision time. Distilling FFPE-level structure into a frozen-only inference
path aims to raise frozen AUC without requiring FFPE at test time.

## Prerequisites

- A trained FFPE high-risk baseline checkpoint to use as the teacher — produced
  by running the existing baseline at
  [configs/tcga_luad_ffpe_high_risk_baseline.yaml](../configs/tcga_luad_ffpe_high_risk_baseline.yaml).
  The path goes into `teacher_checkpoint` in the distillation config.
- The same patient manifest and 5-fold split used by the baselines (already
  referenced in [zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_distill.yaml](configs/tcga_luad_ffpe2frozen_distill.yaml)).
- Virchow2 cache populated — see [scripts/check_virchow2_cache.py](../scripts/check_virchow2_cache.py).

## How to run (single fold, locally)

From the repo root:

```bash
python -m zhuoyang_experiment.train.train_ffpe2frozen_distill \
    --config zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_distill.yaml
```

Outputs land under the `output_dir` specified in the config.

## How to run 5-fold on Slurm

Mirrors [scripts/slurm/train_binary_5fold_array.sbatch](../scripts/slurm/train_binary_5fold_array.sbatch)
and uses the same fold-config generator ([scripts/write_baseline_fold_configs.py](../scripts/write_baseline_fold_configs.py)):

```bash
sbatch \
  --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_distill.yaml,JOB_LABEL=ffpe2frozen_distill \
  zhuoyang_experiment/slurm/train_ffpe2frozen_distill.sbatch
```

Update `teacher_checkpoint` in the config to a fold-appropriate FFPE baseline
checkpoint before launching (the generator preserves the field across folds).

## Distillation loss options

Set via `distill_loss` in the config:

- `kl` (default) — soft-target BCE on the patient logit. No extra parameters
  vs. the baseline; the most fair comparator.
- `mse_embed` — MSE between teacher and student 256-d patient embeddings. Also
  zero extra parameters.
- `dino_ce` — centered + sharpened cross-entropy over a 256-prototype
  projection head (see [models/distill_heads.py](models/distill_heads.py)).
  Adds a small projection head; treat as an ablation.

Total loss is `bce_weight * BCE(student, label) + distill_weight * distill`.

## Outputs

Same filenames as the baseline run:

- `config.json` — full config snapshot
- `history.json` — per-epoch `train_loss`, `train_loss_bce`,
  `train_loss_distill`, `val_loss`, `val_auc`, `val_accuracy`, `learning_rate`
- `metrics.json` — `best_val_auc`, `best_epoch`, `test_loss`, `test_auc`,
  `test_accuracy`, `pos_weight`
- `model.pt` — dict containing `student` (state dict) and, for `dino_ce`,
  `student_proj` and `center`
- `test_predictions.tsv` — `case_id`, `logit`, `probability`, `label`

## Comparing to the baseline

Direct comparator is the per-fold frozen high-risk baseline under
`output/tcga_luad_frozen_high_risk_baseline/fold{k}`. Compare
`metrics.json → test_auc` fold-by-fold.

---

# LoRA backbone fine-tuning (single-modality)

Same task as the FFPE / frozen high-risk baselines, but instead of a fully
frozen Virchow2 backbone the last few transformer blocks are adapted with
[LoRA](https://arxiv.org/abs/2106.09685). Trainable backbone parameter count
is in the tens of thousands at the default settings, so this stays cheap on
a single GPU and is comparable in cost to the baseline.

## What changes vs. the baseline

- Backbone weights stay frozen; LoRA injects a low-rank trainable update
  ``B @ A`` on top of selected ``nn.Linear`` layers.
- By default we adapt only the **last 2** ViT blocks' fused ``attn.qkv``
  projection at **rank 4 / alpha 8** with no LoRA dropout.
- The ABMIL heads and the logit head are trained the same way as in the
  baseline.
- Output filenames (`config.json`, `history.json`, `metrics.json`,
  `model.pt`, `test_predictions.tsv`) match the baseline so existing
  reporting scripts keep working. `metrics.json` adds two extra fields:
  `lora_layers_wrapped` and `encoder_trainable_params`.

The implementation lives in [models/lora.py](models/lora.py); the train
script is [train/train_binary_lora.py](train/train_binary_lora.py) and
re-uses helpers from [src/train/train_binary_baseline.py](../src/train/train_binary_baseline.py)
verbatim — only the model construction differs.

## How to run (single fold, locally)

From the repo root:

```bash
# FFPE
python -m zhuoyang_experiment.train.train_binary_lora \
    --config zhuoyang_experiment/configs/tcga_luad_ffpe_high_risk_lora.yaml

# Frozen
python -m zhuoyang_experiment.train.train_binary_lora \
    --config zhuoyang_experiment/configs/tcga_luad_frozen_high_risk_lora.yaml
```

## How to run 5-fold on Slurm

Re-uses [scripts/write_baseline_fold_configs.py](../scripts/write_baseline_fold_configs.py)
to generate per-fold configs:

```bash
# FFPE
sbatch \
  --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_ffpe_high_risk_lora.yaml,JOB_LABEL=ffpe_lora \
  zhuoyang_experiment/slurm/train_binary_lora.sbatch

# Frozen
sbatch \
  --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_frozen_high_risk_lora.yaml,JOB_LABEL=frozen_lora \
  zhuoyang_experiment/slurm/train_binary_lora.sbatch
```

## LoRA hyperparameters

All set in the YAML config:

- `lora_rank` (default `4`) — rank of the low-rank update. Smaller =
  fewer trainable params and more conservative.
- `lora_alpha` (default `8.0`) — LoRA scaling. The effective scale on the
  update is ``alpha / rank``.
- `lora_dropout` (default `0.0`) — dropout on the LoRA input branch.
- `lora_last_n_blocks` (default `2`) — how many of the *last* ViT blocks
  to adapt. The ViT-H backbone has 32 blocks, so 2 is ~6% of depth.
- `lora_target_modules` (default `["attn.qkv"]`) — list of dotted module
  paths inside one transformer block to wrap. Add ``"attn.proj"`` to also
  adapt the attention output projection. Paths that don't resolve to an
  ``nn.Linear`` (e.g. SwiGLU MLP layers) are silently skipped.
- `lora_learning_rate` / `head_learning_rate` — per-parameter-group LRs.
  The optimizer is built with two groups: one for LoRA weights inside the
  encoder, one for the ABMIL stacks + logit head. Both default to the
  top-level ``learning_rate`` so a fresh run matches the baseline; tune
  them independently when iterating (a common pattern is a smaller LR on
  the LoRA group and a larger LR on the heads). The LR scheduler scales
  both groups in lock-step. Per-group LRs are recorded each epoch in
  ``history.json`` under ``learning_rate_by_group``.

## Comparing to the baseline

Direct comparators are the per-fold high-risk baselines:
- FFPE: `output/tcga_luad_ffpe_high_risk_baseline/fold{k}`
- Frozen: `output/tcga_luad_frozen_high_risk_baseline/fold{k}`

Compare `metrics.json → test_auc` fold-by-fold.

---

# FFPE ↔ Frozen contrastive alignment

Sibling to the FFPE → frozen distillation experiment above. Instead of
matching the student's latent / logit directly to a fixed FFPE teacher,
both modalities are encoded in parallel and their patient-level
embeddings are pulled together with a symmetric InfoNCE loss
(paired patient = positive, in-batch other patients = negatives). Each
modality also keeps its own BCE term against the binary label so neither
tower collapses to a trivial shortcut.

## Why

Direct latent matching forces the frozen-section embedding to *be* the
FFPE embedding, which is a strong constraint and depends on a strong,
already-trained FFPE teacher. Contrastive alignment is weaker — it only
requires paired patients to be closer than unpaired ones — and trains
both towers jointly from scratch. No teacher checkpoint is needed.

## What's the same

- Re-uses [`PairedFFPEFrozenPatientDataset`](data/paired_wsi_dataset.py)
  and `PatientBinaryModel` verbatim.
- Backbone is frozen Virchow2 (same as the distill experiment) so a 5-fold
  run stays in the same compute envelope.
- Output filenames (`config.json`, `history.json`, `metrics.json`,
  `model.pt`, `test_predictions.tsv`) match the distill / baseline runs,
  with extra per-epoch fields (`train_loss_bce_frozen`,
  `train_loss_bce_ffpe`, `train_loss_contrastive`).
- Evaluation uses the frozen tower only — directly comparable to the
  frozen high-risk baseline.

## What's different

- Two `PatientBinaryModel` towers (FFPE + frozen) trained jointly.
- A small [`ContrastiveProjectionHead`](models/distill_heads.py) on top
  of each 256-d patient embedding produces L2-normalized projections.
- Loss = `bce_frozen_weight * BCE(frozen) + bce_ffpe_weight * BCE(ffpe) +
  contrastive_weight * InfoNCE(z_ffpe, z_frozen)`.
- Defaults: `tiles_per_slide=512` (each modality) and `batch_size=4`. The
  larger batch is what gives InfoNCE its negatives; the halved tile count
  keeps total per-epoch tile work close to the distill experiment.

## How to run (single fold, locally)

```bash
python -m zhuoyang_experiment.train.train_ffpe2frozen_contrastive \
    --config zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_contrastive.yaml
```

## How to run 5-fold on Slurm

```bash
sbatch \
  --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_contrastive.yaml,JOB_LABEL=ffpe2frozen_contrastive \
  zhuoyang_experiment/slurm/train_ffpe2frozen_contrastive.sbatch
```

## 1024-tile TCGA pilot variant

For a run that matches the current TCGA pilot tile budget more closely, use
the dedicated config below:

```bash
sbatch \
  --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_contrastive_1024.yaml,JOB_LABEL=ffpe2frozen_contrastive_1024 \
  zhuoyang_experiment/slurm/train_ffpe2frozen_contrastive.sbatch
```

This variant keeps `ffpe_tiles_per_slide=1024` and
`frozen_tiles_per_slide=1024`. Because symmetric InfoNCE needs in-batch
negatives, `batch_size` cannot stay at the baseline value of `1`. The 1024-tile
config uses `batch_size=2`, which is the smallest contrastive batch that still
provides a valid in-batch negative for each anchor while staying closer to the
memory profile of the prior MIL runs.

## Outputs and evaluation

Each fold writes:

- `config.json`
- `history.json`
- `metrics.json`
- `model.pt`
- `test_predictions.tsv`

`metrics.json` reports the held-out frozen-tower deployment metrics:

- `best_val_auc`
- `best_epoch`
- `test_loss`
- `test_auc`
- `test_accuracy`

The saved `model.pt` bundle contains:

- `student`: best frozen tower weights
- `ffpe`: best FFPE tower weights
- `ffpe_proj`: best FFPE projection head
- `frozen_proj`: best frozen projection head when projection heads are not shared

Validation and test selection are done on the frozen tower only, so the
primary comparison remains directly against the frozen 1024 baseline.

To summarize after all 5 folds finish:

```bash
python scripts/summarize_binary_cv_results.py \
  --run frozen_1024=output/tcga_luad_frozen_high_risk_baseline \
  --run ffpe_1024=output/tcga_luad_ffpe_high_risk_baseline \
  --run ffpe2frozen_contrastive_1024=output/tcga_luad_ffpe2frozen_contrastive_1024 \
  --outdir output/tcga_luad_ffpe2frozen_contrastive_1024_summary
```

## Key knobs

- `contrastive_temperature` (default `0.1`) — InfoNCE temperature on the
  cosine similarity logits.
- `contrastive_weight` / `bce_frozen_weight` / `bce_ffpe_weight` — relative
  weights of the three loss terms.
- `proj_dim` (default `128`), `proj_hidden_dim` (default `256`) —
  projection head dimensions.
- `share_projection_head` (default `false`) — when `true`, FFPE and frozen
  share a single projection head.
- `batch_size` (default `4`) — must be ≥ 2 for InfoNCE to have negatives.

## Comparing

Direct comparators are the per-fold frozen high-risk baseline and the
ffpe2frozen distillation run under
`output/tcga_luad_ffpe2frozen_distill/fold{k}`. Compare
`metrics.json → test_auc` fold-by-fold.
