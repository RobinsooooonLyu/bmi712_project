# Lung Frozen-FFPE Multimodal

This repository is for the lung adenocarcinoma project centered on:

- FFPE-to-frozen pathology distillation
- curated pathologic high-risk prediction
- TCGA LUAD proof-of-concept development
- MGH development and BWH external validation
- later multimodal fusion with longitudinal CT

## Current Locked State

The active study endpoint is **curated pathologic high risk**. Keep the current TCGA, MGH, and BWH study plan centered on this label.

Locked decisions:

- Current primary TCGA endpoint: `high_risk_grade_vpi_lvi`.
- Positive high-risk label means `G3` or reviewed VPI/LVI-positive `G1/G2`.
- Current primary baseline is FFPE vs frozen 5-fold CV on this curated endpoint.
- Distillation is **patient-level**, not tile-level or slide-level.
- The main pilot sequence is:
  1. FFPE-only high-risk baseline
  2. frozen-only high-risk baseline
  3. frozen-to-FFPE latent distillation using the high-risk task
  4. optional exploratory latent-generation branch using the same high-risk task
- The first pilot should use a **shared Virchow2 backbone** for FFPE and frozen.
- Future work may swap in a better frozen-specific backbone, but distillation should still occur in a **shared task-aware latent space**, not raw backbone feature space.
- DFI/PFI/survival-style TCGA pilot workflows are not active in this branch and should not be reintroduced without an explicit study decision.

## TCGA LUAD Pilot Cohort

Current working cohort definitions:

- **450** paired LUAD patients with both FFPE diagnostic WSI and frozen tissue-slide WSI
- **403** curated usable pathologic high-risk labels from pathology reports
- **199** high-risk positives
- **204** high-risk negatives
- **23** `G1/G2` cases promoted to high-risk by reviewed VPI/LVI/vascular invasion evidence

The current 5-fold split is patient-level and label-balanced for `high_risk_grade_vpi_lvi`.

## Current Project Phases

1. TCGA LUAD pilot:
   determine whether FFPE and frozen WSIs carry the curated high-risk pathology signal, and whether FFPE-guided distillation improves frozen high-risk prediction.
2. TCGA + MGH pathology model:
   use TCGA and MGH for paired pathology representation learning and high-risk-task distillation.
3. BWH external validation:
   keep BWH untouched until final evaluation.
4. CT multimodal extension:
   train a CT branch and later fuse it with the pathology branch for the same curated high-risk task.

## Outcome

Current primary outcome:

- composite pathologic high-risk label

For the MGH/BWH extension, the first-choice supervision target should remain the curated high-risk pathology label. Distillation may use the current high-risk label to shape the FFPE teacher latent and frozen student latent.

Secondary label kept in the repo:

- `high_grade` (`G3` vs `G1/G2`) as a sensitivity/ablation endpoint

## Repository Workflow

Code should follow this rule:

- local checkout is the primary development environment
- GitHub is the source of truth for code
- MLSC server pulls code from GitHub
- data stay on MLSC and are not stored in git

See `docs/repo_workflow.md` for the full development and sync pattern.

## Key Design Docs

- `Frozen_FFPE_CT_Project_Plan.md`
- `docs/tcga_luad_high_risk_handoff.md`
- `docs/tcga_luad_pilot_design.md`
- `docs/tcga_luad_cvae_branch.md`

The current implementation handoff is `docs/tcga_luad_high_risk_handoff.md`.

## Current Data Utilities

- `scripts/build_tcga_luad_wsi_manifests.py`
- `scripts/download_tcga_wsi.sh`
- `scripts/build_tcga_luad_case_data.py`
- `scripts/build_tcga_luad_patient_manifest.py`
- `scripts/make_tcga_binary_label_split.py`
- `scripts/fetch_tcga_pathology_reports.py`
- `scripts/ocr_tcga_pathology_reports.py`
- `scripts/parse_tcga_grade_from_reports.py`
- `scripts/build_tcga_grade_endpoint_labels.py`
- `scripts/qc_tile_coords.py`
- `scripts/summarize_download_progress.py`
- `scripts/slurm/download_tcga_wsi_minimal.sbatch`
- `scripts/slurm/extract_wsi_tile_coords_basic.sbatch`

Removed from the active branch:

- DFI/PFI probe scripts
- survival/Cox trainers
- DFI/PFI baseline configs and Slurm launchers

## TCGA Metadata And Label Flow

1. Build paired slide manifests:
   `python3 scripts/build_tcga_luad_wsi_manifests.py --strict-tissue --outdir manifests/full_strict`
2. Build paired case metadata:
   `python3 scripts/build_tcga_luad_case_data.py --case-table manifests/full_strict/tcga_luad_paired_case_table_strict.tsv --out manifests/full_strict/tcga_luad_paired_case_data_strict.tsv`
3. Build patient-level slide manifest:
   `python3 scripts/build_tcga_luad_patient_manifest.py --case-table manifests/full_strict/tcga_luad_paired_case_table_strict.tsv --case-data manifests/full_strict/tcga_luad_paired_case_data_strict.tsv --out manifests/full_strict/tcga_luad_patient_manifest_strict.tsv`
4. Use the curated pathology-report labels:
   `manifests/pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv`
5. Build the active high-risk split:
   `python3 scripts/make_tcga_binary_label_split.py --patient-manifest manifests/full_strict/tcga_luad_patient_manifest_trainready.tsv --label-table manifests/pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv --label-column high_risk_grade_vpi_lvi --out manifests/full_strict/tcga_luad_high_risk_grade_vpi_lvi_5fold_trainready.tsv --n-folds 5 --seed 20260406`

## TCGA Preprocessing Flow

1. Download paired LUAD FFPE and frozen WSIs into the canonical server paths.
2. Extract 20x-equivalent tile coordinates:
   - FFPE:
     `INPUT_DIR=/.../data/wsi/tcga_luad/ffpe OUTPUT_DIR=/.../data/coords/tcga_luad/ffpe_20x JOB_LABEL=tcga_luad_ffpe_coords sbatch scripts/slurm/extract_wsi_tile_coords_basic.sbatch`
   - frozen:
     `INPUT_DIR=/.../data/wsi/tcga_luad/frozen OUTPUT_DIR=/.../data/coords/tcga_luad/frozen_20x JOB_LABEL=tcga_luad_frozen_coords OVERWRITE=1 sbatch scripts/slurm/extract_wsi_tile_coords_basic.sbatch`

The extractor:

- builds the tissue mask at a low-resolution mask level to avoid OOM on FFPE slides
- stores `selected_level`, `selected_level_mpp`, and `tile_size_level`
- supports slides with missing `openslide.mpp-x` by falling back to `aperio.MPP` or objective-power metadata
- logs per-slide errors without killing the full job

The dataloader uses the extracted `selected_level` and `tile_size_level`, so FFPE and frozen tiles match the same physical field of view at approximately **20x / 0.5 mpp**.

## Tile-Count Summary

These numbers matter for every downstream sampling decision.

- **FFPE**:
  - `n_slides = 502`
  - `min = 122`
  - `q1 ≈ 5722`
  - `median ≈ 14289`
  - `q3 ≈ 21825`
  - `max = 53932`
- **frozen**:
  - `n_slides = 683`
  - `min = 173`
  - `q1 ≈ 1566`
  - `median ≈ 2564`
  - `q3 ≈ 4103`
  - `max = 27696`

Current interpretation:

- FFPE slides are much larger than frozen slides.
- Small tile budgets such as `64` or `256` are too shallow for FFPE.
- Current baseline uses `1024` tiles per slide from the largest slide per patient.
- Slides with fewer than the requested number of tiles contribute all available tiles.

## Current Training

Primary configs:

- `configs/tcga_luad_ffpe_high_risk_baseline.yaml`
- `configs/tcga_luad_frozen_high_risk_baseline.yaml`

Current design:

- 5-fold patient-level CV
- FFPE and frozen trained separately first
- frozen Virchow2 backbone
- online tile loading with train-time augmentation
- patch -> slide ABMIL -> patient aggregation -> binary high-risk head
- validation fold is used for early stopping and model selection
- test fold is held out until final fold evaluation

Current schedule:

- `epochs: 10`
- `early_stopping_patience: 4`
- `scheduler_patience: 3`
- `tiles_per_slide: 1024`
- `max_slides_per_patient: 1`
- `slide_selection: largest`
- Slurm wall time: `4 days`

See `docs/tcga_luad_high_risk_handoff.md` for exact augmentation, tissue-mask, Slurm, and monitoring details.
