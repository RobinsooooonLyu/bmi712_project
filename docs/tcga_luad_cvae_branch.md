# TCGA LUAD Latent-Generation Branch

This document describes the cVAE-style latent-generation branch for the high-risk pathology study. It should be implemented only after the FFPE and frozen baselines are complete and interpreted.

## Purpose

Frozen and FFPE slides are paired at the patient/tumor level, not as matching physical sections.

Therefore, the frozen-to-FFPE mapping can be modeled as:

- uncertain
- one-to-many
- patient-level
- task-aware

The branch should generate plausible FFPE-like **high-risk task latents** from frozen patient embeddings.

## Inputs And Targets

Use:

- `x_f`: frozen patient embedding
- `h_p`: FFPE teacher high-risk task latent

Do not use:

- raw pixels
- tile tokens
- direct slide-level matching
- supervision outside the curated high-risk pathology label

## Relationship To Deterministic Distillation

Deterministic branch:

- predicts one FFPE-like high-risk latent
- predicts high-risk label from `[x_f, x_p_hat]`

Latent-generation branch:

- predicts a distribution of plausible FFPE-like high-risk latents
- samples `x_p_hat_sample`
- predicts high-risk label from `[x_f, x_p_hat_sample]`

Both branches should use the same high-risk binary prediction interface and should be evaluated on the same 5-fold high-risk splits.

Recommended order:

1. complete FFPE and frozen baselines
2. export fold-specific FFPE/frozen patient latents from the finished baseline runs
3. implement deterministic distillation branch
4. implement cVAE branch on the same exported teacher/student latents
5. compare frozen baseline vs deterministic distillation vs cVAE

## Teacher Target

The FFPE teacher should be trained on the curated high-risk pathology label.

Recommended sequence:

1. train FFPE high-risk teacher
2. define `h_p` as the teacher’s penultimate patient embedding
3. use `h_p` as the reconstruction target for the frozen-to-FFPE latent generator

This keeps the generated latent task-aware rather than purely modality-aligned.

Practical definition:

- `x_f`: frozen patient embedding from the frozen branch
- `h_p`: FFPE teacher penultimate patient embedding
- `t_p`: optional FFPE teacher logit or probability

## Data Usage

Use all paired cases with usable high-risk labels for:

- FFPE teacher training
- frozen baseline training
- deterministic distillation
- latent-generation branch training

If future unlabeled paired cases are used, they may contribute only to unsupervised paired-latent reconstruction and should not change the supervised endpoint definition.

## Fold Rule

Keep the current 5-fold high-risk split exactly.

For each fold:

1. train the FFPE teacher on the train folds
2. choose the FFPE checkpoint using the validation fold
3. export FFPE teacher latents and logits for train/val/test for that fold
4. train the deterministic mapper and cVAE using train only
5. choose model/checkpoint using validation only
6. evaluate once on the held-out test fold

Do not export teacher targets using a model that has seen the held-out test fold.

## Model Sketch

Posterior / encoder:

- input: `[x_f, h_p]`
- output: `mu, logvar`

Latent:

- sample `z = mu + eps * exp(0.5 * logvar)`
- `eps ~ N(0, I)`

Decoder:

- input: `[x_f, z]`
- output: `x_p_hat`

Prediction head:

- input: `[x_f, x_p_hat]`
- output: high-risk logit

Primary scoring rule:

- use a dedicated high-risk prediction head trained on `[x_f, x_p_hat]`
- do not treat the FFPE teacher head as the default scorer for generated latents
- use direct FFPE-teacher-head scoring only as an optional sensitivity analysis

Recommended implementation split:

- FFPE teacher is a separate trained module
- cVAE learns latent generation
- high-risk head is trained on `[x_f, x_p_hat]`
- this head can be:
  - jointly trained with the cVAE, or
  - initialized from the deterministic branch and then fine-tuned

Start simple:

- train cVAE and high-risk head jointly
- no end-to-end tile optimization in the first version

## Losses

On labeled paired cases:

- reconstruction:
  - `L_recon = MSE(x_p_hat, h_p)`
- cosine similarity:
  - `L_cos`
- KL:
  - `L_kl`
- binary task loss:
  - `L_bce`
- optional teacher-logit distillation:
  - `L_teacher`

Example:

`L_total = lambda_recon * L_recon + lambda_cos * L_cos + beta * L_kl + lambda_task * L_bce + lambda_teacher * L_teacher`

Recommended starting weights:

- `lambda_recon = 1.0`
- `lambda_cos = 1.0`
- `beta = 0.01`
- `lambda_task = 1.0`
- `lambda_teacher = 0.0` initially

The first version should be weak-KL and task-aware, not aggressively variational.

## Recommended Training Sequence

1. train FFPE high-risk teacher
2. export patient embeddings and teacher latents
3. train deterministic frozen-to-FFPE mapper baseline
4. pretrain latent generator on paired FFPE/frozen patient embeddings
5. train or attach high-risk binary head
6. evaluate on the same high-risk 5-fold splits used by the deterministic branch

Do not start with end-to-end tile-to-generator training.

## Inference

At test time only frozen is required.

For each case:

1. compute frozen patient embedding `x_f`
2. sample `z_1 ... z_K`
3. generate `x_p_hat_1 ... x_p_hat_K`
4. compute high-risk probability for each sample
5. summarize:
   - mean probability
   - optional uncertainty / variance across samples

Suggested starting point:

- `K = 5`

How prediction should work:

- each sample `x_p_hat_k` produces a logit `s_k`
  - from the dedicated `[x_f, x_p_hat_k]` high-risk head
- convert to probability `p_k = sigmoid(s_k)`
- final case-level prediction is the mean probability:
  - `p_mean = mean_k p_k`
- optional uncertainty summary:
  - `var_p = var_k p_k`
  - or entropy of the sampled probabilities

Do not report the best sample. Use the mean prediction across samples.

Optional sensitivity analysis:

- feed `x_p_hat_k` through the FFPE teacher head
- compare that score distribution against the dedicated cVAE head
- keep this as analysis only, not the primary reported branch

## Evaluation

Primary comparison:

1. frozen baseline
2. deterministic distillation branch
3. cVAE branch

Primary metrics:

- test AUC using `p_mean`
- test accuracy using a fixed threshold rule consistent with the frozen baseline

Secondary metrics:

- calibration
- variance / uncertainty across samples
- latent reconstruction quality

Recommended reconstruction summaries:

- `MSE(x_p_hat, h_p)`
- cosine similarity between `x_p_hat` and `h_p`

Important interpretation rule:

- the cVAE is not judged by how realistic a single sampled latent looks
- it is judged by whether the final mean risk prediction improves held-out test performance over the frozen baseline and deterministic mapper

## Implementation Plan

Expected new pieces:

- `scripts/export_binary_patient_latents.py`
- `src/models/patient_cvae.py`
- `src/train/train_cvae_distill.py`
- `configs/tcga_luad_frozen_cvae_high_risk.yaml`
- `scripts/slurm/train_cvae_5fold_array.sbatch`

Expected exported fold artifacts:

- frozen patient embeddings
- FFPE teacher patient latents
- FFPE teacher logits or probabilities
- case IDs and fold assignments

The cVAE branch should operate on exported patient-level tensors first. Do not make the first version depend on online tile loading.

## Future Compatibility With Different Backbones

The latent-generation branch is compatible with:

- a shared FFPE/frozen backbone
- or different modality-specific backbones

If the backbones differ:

- do not distill raw backbone features
- project each modality into a shared high-risk task latent space first
- run the latent generator in that shared patient-level latent space
