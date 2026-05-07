# TCGA LUAD Pilot Design

This document records the current TCGA LUAD pathology-only pilot design. The active endpoint is curated pathologic high risk.

## Scope

The pilot asks:

**Can FFPE and frozen WSIs predict curated pathologic high risk, and can FFPE-guided distillation improve frozen performance?**

This pilot is not:

- pixel translation
- virtual-FFPE image generation
- direct slide matching
- modeling outside the curated high-risk pathology label without an explicit PI decision

It is a patient-level pathology distillation study using curated pathology-report labels.

## Cohort

Current paired LUAD pathology cohort:

- `450` paired LUAD patients with at least one FFPE diagnostic WSI and at least one frozen tissue-slide WSI
- `403` usable curated high-risk labels
- `199` high-risk positives
- `204` high-risk negatives

Primary label:

- `high_risk_grade_vpi_lvi`

Positive:

- `G3`, or
- reviewed VPI/LVI/vascular invasion positive `G1/G2`

Negative:

- `G1/G2` without reviewed VPI/LVI/vascular invasion evidence

## Split Strategy

Use 5-fold patient-level cross-validation.

For fold `k`:

- `test = fold k`
- `val = fold (k - 1) mod 5`
- `train = remaining 3 folds`

Validation remains separate in the current run and is used for:

- early stopping
- scheduler
- best-checkpoint selection

Later, after hyperparameters are fixed, a train+val refit may be useful because the cohort is small.

## Representation Level

All modeling and distillation should occur at the patient level.

The pairing assumption is:

- frozen and FFPE are matched at tumor biology level
- not at tile level
- not at slide level

Pipeline:

1. patches -> slide embedding
2. slides -> patient embedding
3. high-risk prediction/distillation on patient embedding

## Preprocessing

Use 20x-equivalent / `0.5 mpp`.

The coordinate extractor chooses the pyramid level closest to `0.5 mpp` and stores:

- `selected_level`
- `selected_level_mpp`
- `tile_size_level`

The dataloader reads from that stored level, so FFPE and frozen tiles match physical field of view before Virchow2 preprocessing.

## Tissue Mask

Tissue masking is broad tissue detection, not tumor segmentation.

Mask details:

- low-resolution mask level near `8.0 mpp`
- tissue pixel if RGB mean `< 220` and saturation proxy `(max - min) > 12`
- tile retained if mask tissue fraction `>= 0.5`

## Tile Counts

Current coordinate summaries:

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

Current baseline:

- `1024` tiles per slide
- largest slide per patient
- one slide per patient
- random tile sampling during training
- deterministic seeded sampling for validation/test

## Backbone And Augmentation

Use Virchow2 for the current pilot.

Training augmentation:

- FFPE and frozen:
  - color jitter
  - HED jitter
- frozen only:
  - Gaussian blur

Validation and test:

- no augmentation before Virchow2 preprocessing

Exact parameters are documented in `docs/tcga_luad_high_risk_handoff.md`.

## Model Arms

1. FFPE-only high-risk baseline
2. frozen-only high-risk baseline
3. deterministic frozen-to-FFPE high-risk latent distillation
4. optional high-risk latent-generation branch

The target for distillation should be the FFPE teacher’s high-risk task latent, not a generic backbone embedding.

## Evaluation

Primary metrics:

- AUC
- accuracy
- fold-level predictions for post-hoc inspection

All splits must be patient-level. Slides from the same patient cannot cross split boundaries.

## Current Training Configs

- `configs/tcga_luad_ffpe_high_risk_baseline.yaml`
- `configs/tcga_luad_frozen_high_risk_baseline.yaml`

Current schedule:

- `epochs: 10`
- `early_stopping_patience: 4`
- `scheduler_patience: 3`
- `tiles_per_slide: 1024`
- `max_slides_per_patient: 1`
- `slide_selection: largest`

Use `docs/tcga_luad_high_risk_handoff.md` as the operational handoff.
