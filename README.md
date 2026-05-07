# BMI 712 Final Project — Lung Frozen-Section / FFPE Pathology

Predicting curated pathologic high risk in TCGA LUAD using paired FFPE and frozen whole-slide images (WSIs), Virchow2 patch features, and multiple-instance learning (MIL).

---

## Project overview

Intra-operative frozen sections are available at surgery, but FFPE diagnostic slides carry cleaner morphology. We ask whether a Virchow2 + ABMIL model can predict a curated composite high-risk pathology label from each modality, and whether cross-modal training improves frozen-section inference.

**Primary endpoint:** `high_risk_grade_vpi_lvi` — G3, or G1/G2 with reviewed visceral-pleural invasion, lymphovascular invasion, or vascular invasion.

**Experiments:**

| Experiment | Script / config |
|---|---|
| FFPE baseline | `configs/tcga_luad_ffpe_high_risk_baseline.yaml` |
| Frozen baseline | `configs/tcga_luad_frozen_high_risk_baseline.yaml` |
| FFPE→Frozen KL distillation | `zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_distill.yaml` |
| FFPE↔Frozen contrastive alignment | `zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_contrastive_1024.yaml` |
| LoRA backbone fine-tuning | `zhuoyang_experiment/configs/tcga_luad_ffpe_high_risk_lora.yaml` / `tcga_luad_frozen_high_risk_lora.yaml` |

---

## Repository layout

```
configs/                    training configs (baselines + ablations)
docs/                       design docs and handoff notes
manifests/                  cohort manifests and label tables (generated; not in git)
requirements/training.txt   Python dependencies
scripts/                    data prep, label curation, Slurm launchers, result summarization
  slurm/                    Slurm batch scripts
src/                        core model and training code
  data/                     WSI tile datasets
  models/                   Virchow2 encoder, ABMIL, patient aggregation
  train/                    baseline and distillation trainers
zhuoyang_experiment/        extended experiments (distillation, contrastive, LoRA)
  configs/                  experiment-specific configs
  data/                     paired dataset for cross-modal training
  models/                   LoRA wrapper, distillation/projection heads
  slurm/                    Slurm batch scripts
  train/                    experiment trainers
```

---

## Data access

All data must be obtained externally; no patient data are stored in this repository.

### TCGA LUAD WSIs

1. Register at the [GDC Data Portal](https://portal.gdc.cancer.gov/).
2. Download the GDC Transfer Tool client.
3. Run the manifest-generation and download helpers (see [Data preparation](#data-preparation) below).

Canonical server paths expected by the configs:

```
<wsi_root>/tcga_luad/ffpe/       # FFPE diagnostic WSIs
<wsi_root>/tcga_luad/frozen/     # Frozen tissue-slide WSIs
<coords_root>/tcga_luad/ffpe_20x/    # extracted tile coordinates
<coords_root>/tcga_luad/frozen_20x/  # extracted tile coordinates
```

Set `wsi_root` and `coords_root` in each config YAML to match your server layout.

### Virchow2 backbone

The Virchow2 ViT-H model is hosted on Hugging Face (`paige-ai/Virchow2`). A valid Hugging Face token with access to the model is required. Set the `HUGGINGFACE_TOKEN` environment variable or log in via `huggingface-cli login` before running any training script.

---

## Environment setup

Python ≥ 3.10 and CUDA ≥ 11.8 are recommended.

```bash
pip install -r requirements/training.txt
```

Core dependencies: `torch`, `torchvision`, `timm`, `PyYAML`, `openslide-python`, `Pillow`, `huggingface_hub`, `safetensors`.

Verify the environment and Virchow2 model cache:

```bash
python scripts/check_training_env.py
python scripts/check_virchow2_cache.py
```

---

## Data preparation

Run the following steps in order. All paths below use the canonical `manifests/full_strict/` output directory.

### 1. Build TCGA cohort manifests

```bash
# Paired slide manifest (FFPE + frozen)
python scripts/build_tcga_luad_wsi_manifests.py \
    --strict-tissue \
    --outdir manifests/full_strict

# Case-level metadata
python scripts/build_tcga_luad_case_data.py \
    --case-table manifests/full_strict/tcga_luad_paired_case_table_strict.tsv \
    --out manifests/full_strict/tcga_luad_paired_case_data_strict.tsv

# Patient-level slide manifest
python scripts/build_tcga_luad_patient_manifest.py \
    --case-table manifests/full_strict/tcga_luad_paired_case_table_strict.tsv \
    --case-data manifests/full_strict/tcga_luad_paired_case_data_strict.tsv \
    --out manifests/full_strict/tcga_luad_patient_manifest_strict.tsv
```

### 2. Download WSIs

```bash
bash scripts/download_tcga_wsi.sh
# or on Slurm:
sbatch scripts/slurm/download_tcga_wsi_minimal.sbatch
```

### 3. Extract tile coordinates (20×)

```bash
# FFPE
INPUT_DIR=<wsi_root>/tcga_luad/ffpe \
OUTPUT_DIR=<coords_root>/tcga_luad/ffpe_20x \
JOB_LABEL=tcga_luad_ffpe_coords \
sbatch scripts/slurm/extract_wsi_tile_coords_basic.sbatch

# Frozen
INPUT_DIR=<wsi_root>/tcga_luad/frozen \
OUTPUT_DIR=<coords_root>/tcga_luad/frozen_20x \
JOB_LABEL=tcga_luad_frozen_coords \
OVERWRITE=1 \
sbatch scripts/slurm/extract_wsi_tile_coords_basic.sbatch
```

### 4. Curate pathology-report labels

```bash
python scripts/fetch_tcga_pathology_reports.py
python scripts/ocr_tcga_pathology_reports.py
python scripts/parse_tcga_grade_from_reports.py
python scripts/build_tcga_grade_endpoint_labels.py
# Outputs: manifests/pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv
```

### 5. Build the 5-fold CV split

```bash
python scripts/make_tcga_binary_label_split.py \
    --patient-manifest manifests/full_strict/tcga_luad_patient_manifest_trainready.tsv \
    --label-table manifests/pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv \
    --label-column high_risk_grade_vpi_lvi \
    --out manifests/full_strict/tcga_luad_high_risk_grade_vpi_lvi_5fold_trainready.tsv \
    --n-folds 5 \
    --seed 20260406
```

---

## Running the main experiments

### FFPE and frozen baselines (5-fold CV)

Generate per-fold configs and submit:

```bash
python scripts/write_baseline_fold_configs.py \
    --base-config configs/tcga_luad_ffpe_high_risk_baseline.yaml \
    --outdir configs/ffpe_folds

python scripts/write_baseline_fold_configs.py \
    --base-config configs/tcga_luad_frozen_high_risk_baseline.yaml \
    --outdir configs/frozen_folds

# On Slurm:
sbatch --export=ALL,BASE_CONFIG_PATH=configs/tcga_luad_ffpe_high_risk_baseline.yaml,JOB_LABEL=ffpe_baseline \
    scripts/slurm/train_binary_5fold_array.sbatch

sbatch --export=ALL,BASE_CONFIG_PATH=configs/tcga_luad_frozen_high_risk_baseline.yaml,JOB_LABEL=frozen_baseline \
    scripts/slurm/train_binary_5fold_array.sbatch
```

Single-fold local run (fold 0):

```bash
python -m src.train.train_binary_baseline \
    --config configs/tcga_luad_ffpe_high_risk_baseline.yaml
```

### FFPE→Frozen KL distillation (5-fold CV)

Requires a trained FFPE baseline checkpoint. Set `teacher_checkpoint` in the config to the fold-appropriate path.

```bash
sbatch --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_distill.yaml,JOB_LABEL=ffpe2frozen_distill \
    zhuoyang_experiment/slurm/train_ffpe2frozen_distill.sbatch

# Single fold locally:
python -m zhuoyang_experiment.train.train_ffpe2frozen_distill \
    --config zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_distill.yaml
```

### FFPE↔Frozen contrastive alignment (5-fold CV)

No teacher checkpoint required; both towers train jointly.

```bash
sbatch --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_contrastive_1024.yaml,JOB_LABEL=ffpe2frozen_contrastive_1024 \
    zhuoyang_experiment/slurm/train_ffpe2frozen_contrastive.sbatch

# Single fold locally:
python -m zhuoyang_experiment.train.train_ffpe2frozen_contrastive \
    --config zhuoyang_experiment/configs/tcga_luad_ffpe2frozen_contrastive_1024.yaml
```

### LoRA backbone fine-tuning (5-fold CV)

```bash
sbatch --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_ffpe_high_risk_lora.yaml,JOB_LABEL=ffpe_lora \
    zhuoyang_experiment/slurm/train_binary_lora.sbatch

sbatch --export=ALL,BASE_CONFIG_PATH=zhuoyang_experiment/configs/tcga_luad_frozen_high_risk_lora.yaml,JOB_LABEL=frozen_lora \
    zhuoyang_experiment/slurm/train_binary_lora.sbatch
```

---

## Summarizing results

After all folds complete, summarize test AUC/accuracy across folds:

```bash
python scripts/summarize_binary_cv_results.py \
    --run ffpe_baseline=output/tcga_luad_ffpe_high_risk_baseline \
    --run frozen_baseline=output/tcga_luad_frozen_high_risk_baseline \
    --run ffpe2frozen_distill=output/tcga_luad_ffpe2frozen_distill \
    --run ffpe2frozen_contrastive_1024=output/tcga_luad_ffpe2frozen_contrastive_1024 \
    --outdir output/summary
```

Each fold writes `metrics.json` with `test_auc` and `test_accuracy` as the primary held-out metrics.

---

## Baseline results (TCGA LUAD, 1024 tiles/slide, 5-fold CV)

| Modality | Mean test AUC | Std | Pooled OOF AUC |
|---|---|---|---|
| FFPE | 0.8088 | 0.0374 | 0.7831 |
| Frozen | 0.7751 | 0.0822 | 0.7501 |

FFPE outperforms frozen on the curated high-risk endpoint. Distillation and contrastive experiments aim to close this gap at frozen-only inference time.

---

## Cohort summary

- **450** paired LUAD patients (FFPE + frozen WSIs)
- **403** usable curated high-risk labels
- **199** high-risk positive / **204** high-risk negative
- **23** G1/G2 cases promoted by reviewed VPI/LVI/vascular invasion evidence
- 5-fold patient-level, label-balanced CV split (seed `20260406`)

---

## Key configuration parameters

| Parameter | Default | Notes |
|---|---|---|
| `tiles_per_slide` | 1024 | tiles sampled per slide per epoch |
| `max_slides_per_patient` | 1 | largest slide selected per patient |
| `epochs` | 10 | max training epochs |
| `early_stopping_patience` | 4 | epochs without val-AUC improvement |
| `batch_size` | 1 | one patient per step (MIL) |
| `learning_rate` | 1e-4 | Adam |
| `freeze_backbone` | true | Virchow2 weights frozen; only ABMIL + head trained |
| `seed` | 20260403 | training seed |

---

## Further documentation

- [docs/tcga_luad_high_risk_handoff.md](docs/tcga_luad_high_risk_handoff.md) — current implementation state and augmentation details
- [docs/tcga_luad_pilot_design.md](docs/tcga_luad_pilot_design.md) — pilot design rationale
- [docs/repo_workflow.md](docs/repo_workflow.md) — local / GitHub / MLSC server sync pattern
- [zhuoyang_experiment/README.md](zhuoyang_experiment/README.md) — distillation, contrastive, and LoRA experiment details
- [Frozen_FFPE_CT_Project_Plan.md](Frozen_FFPE_CT_Project_Plan.md) — full project plan including CT multimodal extension
