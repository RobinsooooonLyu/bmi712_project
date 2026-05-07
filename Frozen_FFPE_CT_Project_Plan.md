# Frozen-FFPE Pathology Distillation and CT Multimodal Project Plan

This is the current project-level handoff for the lung adenocarcinoma frozen/FFPE pathology project.

## Central Goal

The project asks whether FFPE-guided learning can improve frozen-section pathology modeling for a **curated pathologic high-risk label**.

The project is not:

- pixel-level FFPE synthesis
- stain transfer
- direct tile/slide matching
- modeling outside the curated high-risk pathology label without an explicit PI decision

The model should learn patient-level morphology and high-risk pathology structure from paired FFPE and frozen slides.

## Primary Endpoint

Use the curated pathologic high-risk label as the primary endpoint across TCGA, MGH, and BWH.

Current TCGA definition:

- positive if `G3`
- positive if `G1/G2` with reviewed positive VPI, LVI, or vascular invasion evidence
- negative if `G1/G2` without reviewed VPI/LVI/vascular invasion evidence

Current TCGA counts:

- usable labeled cases: `403`
- high-risk positives: `199`
- high-risk negatives: `204`

For MGH/BWH, the same concept should be implemented using institutional pathology review. The exact high-risk components can be adjusted by the pathology team, but the supervision target should remain morphology-proximal and clinically interpretable.

## Strategic Decisions

- The pathology task is a **teacher-student distillation problem**, not pixel synthesis.
- Distillation is **patient-level**, not slide-level or tile-level.
- TCGA is the current proof-of-concept cohort.
- MGH is the future development cohort.
- BWH remains the untouched external validation cohort.
- A pathology foundation model should be used, with documentation of any known cohort exposure limitations.
- When TCGA and MGH are combined, MGH should have more influence through weighted sampling or loss weighting rather than an explicit domain indicator by default.
- The curated high-risk pathology label is the active supervision target for the current study plan.

## Data Assets

### TCGA Proof-of-Concept Cohort

Current TCGA-LUAD paired cohort:

- `450` paired LUAD patients with at least one FFPE diagnostic WSI and at least one frozen tissue-slide WSI
- `403` curated usable high-risk labels from pathology reports
- `199` high-risk positives
- `204` high-risk negatives

TCGA is used to establish:

- FFPE high-risk baseline performance
- frozen high-risk baseline performance
- whether FFPE-guided distillation can improve frozen high-risk prediction

### MGH Institutional Cohort

MGH is the planned development cohort for the downstream model.

Current expectation:

- approximately `400` patients
- small lung adenocarcinoma surgical cohort
- paired frozen and FFPE pathology
- institutional pathology review available
- CT data available

Use MGH to:

- refine the high-risk label definition with pathology input
- tune model settings after TCGA proof of concept
- train the pathology model before external validation
- train the CT branch before multimodal fusion

### BWH Institutional Cohort

BWH is the external validation cohort and must remain untouched during development.

Current expectation:

- approximately `300` patients
- paired frozen and FFPE pathology
- institutional pathology review available
- CT data available

BWH is used only for final locked evaluation.

## Phase 1: TCGA High-Risk Pathology Pilot

The immediate goal is:

**Can FFPE and frozen WSIs predict curated pathologic high risk, and can FFPE-guided distillation improve frozen performance?**

Current baseline arms:

1. FFPE-only high-risk model
2. frozen-only high-risk model
3. distilled frozen high-risk model after the baselines are understood

Current baseline model:

- Virchow2 frozen backbone
- 20x-equivalent / 0.5 mpp tiles
- 1024 tiles from the largest slide per patient
- ABMIL slide/patient aggregation
- binary high-risk head
- 5-fold patient-level CV

Current implementation handoff:

- `docs/tcga_luad_high_risk_handoff.md`

## Phase 2: Distillation

The FFPE teacher should be trained on the high-risk task.

The frozen student should learn:

- frozen patient embedding
- FFPE-like high-risk task latent
- high-risk binary prediction

Recommended deterministic structure:

- FFPE teacher produces a patient-level task latent
- frozen branch produces a frozen patient embedding
- mapper predicts an FFPE-like high-risk latent from frozen
- final head predicts high-risk label from frozen and predicted FFPE-like latent

The target latent should be the FFPE teacher’s **high-risk task latent**, not a generic backbone embedding.

Two post-baseline branches should be run:

1. deterministic distillation branch
   - frozen patient embedding -> one FFPE-like high-risk latent
   - final high-risk head predicts from `[x_f, x_p_hat]`
2. cVAE latent-generation branch
   - frozen patient embedding -> distribution of plausible FFPE-like high-risk latents
   - final prediction is the mean high-risk probability across multiple latent samples
   - primary scorer is a dedicated head on `[x_f, x_p_hat]`, not the raw FFPE teacher head

The deterministic branch should be implemented first because it is the cleaner lower-variance comparison.

The cVAE branch should be judged against:

- frozen baseline
- deterministic distillation branch

not in isolation.

## Phase 3: TCGA + MGH Development

After the TCGA proof of concept:

- use TCGA + MGH for paired FFPE/frozen representation learning and distillation
- weight MGH more heavily if combined with TCGA
- tune outcome heads on MGH
- do not use BWH for tuning

The MGH high-risk label should be curated using institutional pathology review and should be treated as the first-choice supervision target.

## Phase 4: CT Multimodal Extension

Build the pathology model first and treat it as a pretrained pathology branch.

Then:

1. train a CT branch on MGH
2. initialize a multimodal model with pathology and CT branches
3. train fusion on MGH only
4. evaluate the locked multimodal model on BWH only

The multimodal model should predict the same curated high-risk pathology endpoint unless the PI explicitly defines a new endpoint.

## Leakage Guardrails

- BWH cannot be used for model selection, threshold selection, hyperparameter tuning, or architecture iteration.
- All splits must be patient-level.
- Slides from the same patient cannot be split across train/validation/test.
- If multiple slides exist per modality, aggregation occurs within patient after the split is defined.
- If a foundation model has known exposure to an evaluation cohort, that limitation must be documented.

## Immediate Deliverables

1. Complete the TCGA FFPE and frozen high-risk 5-fold baselines.
2. Aggregate fold-level AUC, accuracy, and predictions.
3. Inspect FFPE vs frozen gap.
4. If signal is strong, export fold-specific FFPE teacher latents and frozen patient embeddings.
5. Train the deterministic high-risk distillation branch.
6. Train the cVAE high-risk latent-generation branch on the same fold structure.
7. Compare frozen baseline vs deterministic distillation vs cVAE.
8. Prepare MGH and BWH high-risk label curation tables.
9. Keep BWH locked until final evaluation.

## Short Project Summary

The project should start with a TCGA LUAD proof of concept showing whether FFPE and frozen pathology can predict curated pathologic high risk and whether FFPE-guided distillation improves frozen performance. The downstream translational model should use TCGA plus MGH for development, keep BWH as a locked external test cohort, and later add CT fusion while preserving the curated high-risk pathology endpoint as the primary target.
