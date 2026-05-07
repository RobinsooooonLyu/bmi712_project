# Source

This directory contains project code for:

- cohort construction
- WSI patch extraction
- MIL models
- FFPE-to-frozen patient-level distillation
- high-risk binary evaluation
- CT multimodal fusion

Useful TCGA cohort setup utilities currently live in `scripts/`, including:

- WSI manifest generation
- WSI download via the GDC API
- patient-level slide manifest generation
- curated pathology-report label generation
- high-risk 5-fold split generation

Current pathology pilot code modules:

- raw WSI tile coordinate extraction at 20x
- Virchow2 online tile encoding with official preprocessing
- train-time color/stain augmentation for both FFPE and frozen
- train-time blur augmentation for frozen only
- slide-level ABMIL
- patient-level aggregation across slides
- binary high-risk training and AUC evaluation
