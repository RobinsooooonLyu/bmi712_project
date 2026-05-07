# TCGA LUAD High-Risk Pathology Handoff

Last updated: 2026-04-12

This document captures the current implementation state for the TCGA LUAD FFPE/frozen pathology baseline. It is the operational handoff for the immediate training runs.

Endpoint guardrail:

- Do not use DFI, PFI, or survival-style pilot objectives in this branch.
- Keep TCGA, MGH, BWH, and distillation planning centered on the curated high-risk pathology label unless the PI explicitly changes the endpoint.

Current post-baseline plan:

- run FFPE-only `2048`-tile sensitivity in the background
- export fold-specific FFPE teacher latents and frozen patient embeddings
- run deterministic high-risk distillation
- run cVAE latent-generation on the same fold structure
- compare frozen baseline vs deterministic distillation vs cVAE using held-out test AUC/accuracy

## Current Goal

The current primary endpoint is **pathology-derived high risk**.

Use TCGA LUAD paired FFPE/frozen WSIs to test whether Virchow2 + MIL can predict a directly observed high-risk pathology label from:

- FFPE diagnostic slides
- frozen tissue slides

The immediate purpose is to establish a robust FFPE/frozen pathology baseline before moving to high-risk-task distillation.

## Current 1024-Tile Baseline Result

5-fold CV summary from the finished high-risk baseline:

- FFPE
  - mean test AUC: `0.8088`
  - std test AUC: `0.0374`
  - pooled out-of-fold test AUC: `0.7831`
  - mean test accuracy: `0.7235`
  - pooled out-of-fold test accuracy: `0.7236`
- frozen
  - mean test AUC: `0.7751`
  - std test AUC: `0.0822`
  - pooled out-of-fold test AUC: `0.7501`
  - mean test accuracy: `0.6987`
  - pooled out-of-fold test accuracy: `0.6985`

Interpretation:

- FFPE shows a modest but real advantage over frozen on the curated high-risk endpoint.
- The gap is sufficient to justify both deterministic distillation and cVAE.
- Because FFPE is more heavily subsampled at `1024` tiles, the current FFPE advantage may still be conservative.

## Endpoint

Primary endpoint: `high_risk_grade_vpi_lvi`

Positive label:

- `G3`, or
- `G1/G2` with reviewed positive visceral pleural invasion, lymphovascular invasion, or vascular invasion evidence

Negative label:

- `G1/G2` without reviewed VPI/LVI/vascular invasion risk evidence

Current curated counts:

- usable labeled cases: `403`
- high-risk positive: `199`
- high-risk negative: `204`
- G1/G2 promoted by reviewed VPI/LVI evidence: `23`

Grade-only endpoint still exists:

- label column: `high_grade`
- positive: `G3`
- negative: `G1/G2`
- current counts: `176` positive, `227` negative

For current training, use `high_risk_grade_vpi_lvi` as the primary endpoint.

## Label Artifacts

Main files:

- `manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_labels.tsv`
- `manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_evidence.tsv`
- `manifests/pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv`
- `manifests/pathology_reports_luad_paired_full/tcga_luad_high_risk_promotion_review.tsv`

Important curation notes:

- OCR was used for low-quality pathology report PDFs.
- Missing cases were manually reviewed or explicitly excluded; there are no true unreviewed missing grade cases in the current curated table.
- `TCGA-55-8203` is kept as high-risk based on “highly suspicious” lymphovascular invasion wording, per manual decision.
- `TCGA-95-7039` is kept as high-risk by LVI; its VPI snippet is less clean but LVI says `Lymphatic invasion Present`.
- Multifocal/discordant cases are excluded unless the relevant tumor/slide could be reasonably selected.

## Current Split Design

Use 5-fold patient-level CV.

For fold `k`:

- `test = fold k`
- `val = fold (k - 1) mod 5`
- `train = remaining 3 folds`

Validation is not included in training in the current run. It is used for:

- early stopping
- LR scheduler
- best-checkpoint selection

The held-out test fold is evaluated only after validation checkpoint selection.

Future final-model note:

- after hyperparameters are fixed, it may be appropriate to train on `train + val` and evaluate on the held-out test fold, because the cohort is small.

## Current Model

Trainer:

- `src/train/train_binary_baseline.py`

Model:

- frozen Virchow2 backbone
- ABMIL-style slide/patient aggregation
- binary classification head
- loss: `BCEWithLogitsLoss`
- fold-specific `pos_weight` inferred from training labels
- metrics: AUC and accuracy

Current configs:

- FFPE high-risk: `configs/tcga_luad_ffpe_high_risk_baseline.yaml`
- frozen high-risk: `configs/tcga_luad_frozen_high_risk_baseline.yaml`
- FFPE high-risk `2048`-tile sensitivity: `configs/tcga_luad_ffpe_high_risk_2048_sensitivity.yaml`
- FFPE high-grade secondary endpoint: `configs/tcga_luad_ffpe_high_grade_baseline.yaml`
- frozen high-grade secondary endpoint: `configs/tcga_luad_frozen_high_grade_baseline.yaml`

Current schedule:

- `epochs: 10`
- `early_stopping_patience: 4`
- `scheduler_patience: 3`
- `scheduler_factor: 0.5`
- `learning_rate: 1e-4`
- `weight_decay: 1e-4`
- `dropout: 0.25`
- `grad_clip_norm: 1.0`
- Slurm wall time: `4-00:00:00`

## Planned Post-Baseline Branches

Current next branch after the finished 1024-tile baselines:

1. Deterministic distillation branch
   - frozen patient embedding `x_f`
   - FFPE teacher task latent `h_p`
   - mapper predicts one FFPE-like latent `x_p_hat`
   - prediction head uses `[x_f, x_p_hat]`

2. cVAE latent-generation branch
   - encoder uses `[x_f, h_p]`
   - outputs `mu, logvar`
   - decoder uses `[x_f, z]`
   - predicts sampled FFPE-like latent `x_p_hat_k`
   - primary prediction head uses `[x_f, x_p_hat_k]`
   - direct FFPE-teacher-head scoring is optional sensitivity analysis only

Inference rule for cVAE:

- sample `K` latent draws
- compute `K` high-risk probabilities
- use mean probability as the final case-level prediction
- optionally retain variance across samples as uncertainty

Evaluation rule:

- use the same held-out fold structure as the baselines
- compare:
  - frozen baseline
  - deterministic distillation
  - cVAE
- primary metrics:
  - test AUC
  - test accuracy

Implemented files for the deterministic branch:

- latent export: `scripts/export_binary_patient_latents.py`
- paired latent dataset: `src/data/paired_latent_dataset.py`
- mapper/predictor model: `src/models/deterministic_distillation.py`
- trainer: `src/train/train_deterministic_distillation.py`
- fold config writer: `scripts/write_distill_fold_configs.py`
- latent export Slurm array: `scripts/slurm/export_binary_latents_5fold_array.sbatch`
- distillation Slurm array: `scripts/slurm/train_deterministic_distill_5fold_array.sbatch`
- base config: `configs/tcga_luad_deterministic_distillation_high_risk.yaml`

## Tile Sampling

Current baseline tile policy:

- `tiles_per_slide: 1024`
- `max_slides_per_patient: 1`
- `slide_selection: largest`
- if a slide has fewer than 1024 retained tissue tiles, use all available tiles
- training samples a random 1024-tile subset each epoch
- validation/test use deterministic seeded 1024-tile sampling

Empirical retained tissue tile counts:

- FFPE:
  - `n_slides = 502`
  - `min = 122`
  - `q1 ≈ 5722`
  - `median ≈ 14289`
  - `q3 ≈ 21825`
  - `max = 53932`
- frozen:
  - `n_slides = 683`
  - `min = 173`
  - `q1 ≈ 1566`
  - `median ≈ 2564`
  - `q3 ≈ 4103`
  - `max = 27696`

Interpretation:

- `1024` is a practical baseline compromise.
- FFPE remains heavily subsampled even at 1024 tiles.
- Frozen is covered more substantially by 1024 tiles.
- If FFPE performance is weak or unstable, a `2048`-tile FFPE sensitivity run is the next reasonable check.

Current sensitivity artifact:

- `configs/tcga_luad_ffpe_high_risk_2048_sensitivity.yaml`
- this config also writes deterministic teacher latent exports directly into each fold output under `latent_exports/`

## Tissue Mask And Tiling

Coordinate extraction script:

- `scripts/extract_wsi_tile_coords.py`

Tissue mask is broad tissue detection, not tumor segmentation.

Mask details:

- low-resolution mask level chosen near `mask_mpp = 8.0`
- tile target resolution: `target_mpp = 0.5`
- nominal tile size: `224`
- intended physical field of view: `224 * 0.5 = 112 um`
- tissue pixel if:
  - RGB channel mean `< 220`
  - RGB saturation proxy `(max - min) > 12`
- retained tile if low-res mask tissue fraction `>= 0.5`

The dataloader reads from the stored slide pyramid level using:

- `selected_level`
- `tile_size_level`

This preserves 20x-equivalent physical field of view across FFPE and frozen slides before Virchow2 preprocessing.

## Exact Augmentation

Training augmentation is applied before Virchow2 preprocessing.

For both FFPE and frozen:

- `ColorJitter`
  - `brightness = 0.18`
  - `contrast = 0.18`
  - `saturation = 0.12`
  - `hue = 0.04`
- `RandomHEDJitter`
  - probability `p = 0.5`
  - RGB -> HED
  - add per-pixel Gaussian HED noise with `sigma = 0.05`
  - add per-channel HED bias sampled from `[-0.02, 0.02]`
  - HED -> RGB
  - clamp RGB to `[0, 1]`

Frozen training only:

- `RandomGaussianBlur`
  - probability `p = 0.35`
  - radius sampled uniformly from `0.2` to `1.6`

Validation and test:

- no augmentation
- empty eval transform before Virchow2 preprocessing

## Immediate Commands

FFPE `2048`-tile sensitivity:

```bash
PYTHON_BIN=/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 \
BASE_CONFIG_PATH=/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal/configs/tcga_luad_ffpe_high_risk_2048_sensitivity.yaml \
JOB_LABEL=tcga_luad_ffpe_high_risk_2048 \
sbatch --partition=dgx-a100 scripts/slurm/train_binary_5fold_array.sbatch
```

FFPE `2048` latent export behavior:

- no separate FFPE export job is needed for this sensitivity run
- each completed fold writes:
  - `output/tcga_luad_ffpe_high_risk_2048_sensitivity/foldK/latent_exports/train.pt`
  - `output/tcga_luad_ffpe_high_risk_2048_sensitivity/foldK/latent_exports/val.pt`
  - `output/tcga_luad_ffpe_high_risk_2048_sensitivity/foldK/latent_exports/test.pt`
- these exports are deterministic and suitable for teacher-side distillation

Export deterministic-teacher latents:

```bash
PYTHON_BIN=/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 \
BASE_RUN_ROOT=output/tcga_luad_ffpe_high_risk_baseline \
LATENT_OUTPUT_ROOT=output/tcga_luad_ffpe_high_risk_latents \
JOB_LABEL=tcga_luad_ffpe_high_risk_latents \
sbatch --partition=dgx-a100 scripts/slurm/export_binary_latents_5fold_array.sbatch
```

```bash
PYTHON_BIN=/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 \
BASE_RUN_ROOT=output/tcga_luad_frozen_high_risk_baseline \
LATENT_OUTPUT_ROOT=output/tcga_luad_frozen_high_risk_latents \
TRAIN_VIEWS=4 \
AUGMENT_TRAIN=1 \
JOB_LABEL=tcga_luad_frozen_high_risk_latents \
sbatch scripts/slurm/export_binary_latents_5fold_array.sbatch
```

Export rule:

- FFPE teacher export stays deterministic for `train`, `val`, and `test`
- frozen export uses deterministic `val`/`test`
- frozen `train` export may use multiple augmented views per case via `TRAIN_VIEWS` and `AUGMENT_TRAIN=1`
- this gives many frozen student views matched to one stable FFPE teacher target

Train deterministic distillation:

```bash
PYTHON_BIN=/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 \
BASE_CONFIG_PATH=/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal/configs/tcga_luad_deterministic_distillation_high_risk.yaml \
JOB_LABEL=tcga_luad_distill_high_risk \
sbatch scripts/slurm/train_deterministic_distill_5fold_array.sbatch
```

## Current Slurm Setup

Launcher:

- `scripts/slurm/train_binary_5fold_array.sbatch`

Key settings:

- partition default in script: `rtx8000`
- array: `0-4%2`
- one GPU per task
- `cpus-per-task = 3`
- `mem = 96G`
- wall time: `4 days`
- `HF_HUB_OFFLINE = 1`
- repo-local Hugging Face cache under `manifests/cache/huggingface`
- `PYTHONUNBUFFERED = 1`
- `PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True`

Current recommended allocation:

- FFPE: submit to `dgx-a100` because FFPE slides are much larger and slower
- frozen: `rtx8000` is acceptable because frozen slides are smaller

A100 OpenSlide note:

- A100 initially failed with `libopenslide.so.0` missing.
- This was fixed by installing:

```bash
/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 -m pip install openslide-bin
```

Verification command:

```bash
sbatch --partition=dgx-a100 --account=qtim --gres=gpu:1 --cpus-per-task=1 --mem=4G --time=00:10:00 --wrap='/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 -c "import openslide; print(openslide.__file__); print(openslide.__version__)"'
```

Expected output includes:

```text
/autofs/space/crater_001/tools/pyenv/versions/default/lib/python3.11/site-packages/openslide/__init__.py
1.4.2
```

## Commands To Run Current Primary Baselines

Pull latest code:

```bash
cd /autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal
git checkout codex/tcga-pilot
git pull --ff-only
```

Build high-risk split:

```bash
python3 scripts/make_tcga_binary_label_split.py \
  --patient-manifest manifests/full_strict/tcga_luad_patient_manifest_trainready.tsv \
  --label-table manifests/pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv \
  --label-column high_risk_grade_vpi_lvi \
  --out manifests/full_strict/tcga_luad_high_risk_grade_vpi_lvi_5fold_trainready.tsv \
  --n-folds 5 \
  --seed 20260406
```

Submit FFPE on A100:

```bash
PYTHON_BIN=/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 \
BASE_CONFIG_PATH=/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal/configs/tcga_luad_ffpe_high_risk_baseline.yaml \
JOB_LABEL=tcga_luad_ffpe_high_risk \
sbatch --partition=dgx-a100 scripts/slurm/train_binary_5fold_array.sbatch
```

Submit frozen on RTX8000:

```bash
PYTHON_BIN=/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 \
BASE_CONFIG_PATH=/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal/configs/tcga_luad_frozen_high_risk_baseline.yaml \
JOB_LABEL=tcga_luad_frozen_high_risk \
sbatch scripts/slurm/train_binary_5fold_array.sbatch
```

Monitor:

```bash
squeue -u $USER
showrunusage | grep zzy1
tail -n 120 logs/slurm/train_bin_cv_JOBID_TASKID.out
```

The trainer log prints:

- runtime CUDA device
- modality and fold assignment
- tile settings
- train/val/test label counts
- per-batch progress
- per-epoch validation loss/AUC/accuracy
- new best checkpoint messages
- final best epoch, best validation AUC, test loss, test AUC, test accuracy
- output directory

## Current Running Jobs At Time Of This Handoff

At the time this document was written, the user had:

- frozen high-risk array on `rtx8000`: `8047124`
- FFPE high-risk array on `dgx-a100`: `8047214`

These job IDs are informational only; check `squeue -u $USER` for current state.

## Endpoint Guardrail

Keep modeling, distillation, and future MGH/BWH extension centered on the curated high-risk pathology label unless the PI explicitly changes the endpoint.
