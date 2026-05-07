# TCGA LUAD 2048-Tile Sensitivity Summary

## Purpose

This note records the FFPE 2048-tile sensitivity analysis for the TCGA LUAD high-risk pathology endpoint.

The analysis was run after the initial 1024-tile FFPE and frozen baselines to test whether increasing FFPE tiles per slide materially improved high-risk prediction performance.

## Endpoint

Primary endpoint:

- `high_risk_grade_vpi_lvi`

Positive class:

- G3 high-grade tumor, or
- G1/G2 tumor promoted to high-risk by pathologic risk factors such as visceral pleural invasion or lymphovascular invasion

DFI and PFI are not used for this study.

## Compared Runs

The comparison included:

- `frozen_1024`: frozen baseline, 1024 tiles per slide
- `ffpe_1024`: FFPE baseline, 1024 tiles per slide
- `ffpe_2048`: FFPE sensitivity run, 2048 tiles per slide

All runs used the same 5-fold high-risk split and held-out test-fold evaluation design.

## Summary Results

| Run | Modality | Folds | Test N | Mean Test AUC | SD Test AUC | Pooled Test AUC | Mean Test Accuracy | Pooled Test Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `frozen_1024` | frozen | 5 | 398 | 0.7751 | 0.0822 | 0.7501 | 0.6987 | 0.6985 |
| `ffpe_1024` | FFPE | 5 | 398 | 0.8088 | 0.0374 | 0.7831 | 0.7235 | 0.7236 |
| `ffpe_2048` | FFPE | 5 | 398 | 0.8071 | 0.0479 | 0.7909 | 0.7286 | 0.7286 |

Raw summary table reported on MLSC:

```text
run_label       modality        n_folds n_test_total    mean_best_val_auc       std_best_val_auc        mean_test_auc   std_test_auc    mean_test_accuracy  std_test_accuracy        pooled_test_auc pooled_test_accuracy
frozen_1024     frozen  5       398     0.8042051282051282      0.06688687762039516     0.7751442307692308      0.08216503689357105     0.6987025260925293  0.10916445736254318      0.7500757575757576      0.6984924623115578
ffpe_1024       ffpe    5       398     0.8340897435897436      0.05639352648367796     0.8087820512820513      0.03743979902339528     0.7235126495361328  0.04507182810325633      0.7830555555555555      0.7236180904522613
ffpe_2048       ffpe    5       398     0.8294134615384615      0.057205688579039966    0.8070801282051282      0.04789514834984487     0.7286075949668884  0.036412590604656535      0.7909343434343434      0.7286432160804021
```

## Interpretation

The 2048-tile FFPE sensitivity run did not materially improve mean fold-level test AUC compared with the 1024-tile FFPE baseline:

- FFPE 1024 mean test AUC: `0.8088`
- FFPE 2048 mean test AUC: `0.8071`

The 2048-tile run showed a small pooled-test improvement:

- FFPE 1024 pooled test AUC: `0.7831`
- FFPE 2048 pooled test AUC: `0.7909`

Accuracy was also slightly higher for 2048:

- FFPE 1024 pooled test accuracy: `0.7236`
- FFPE 2048 pooled test accuracy: `0.7286`

Given the small magnitude of improvement and higher compute cost, the working decision is:

- Use `1024` tiles for FFPE and frozen baselines.
- Use `1024` tiles for online frozen-to-FFPE representation distillation.
- Keep 2048 as a sensitivity analysis, not the primary training setting.

## Recreate Summary On MLSC

From the TCGA pilot repo root:

```bash
cd /autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal

/autofs/space/crater_001/tools/pyenv/versions/default/bin/python3 scripts/summarize_binary_cv_results.py \
  --run frozen_1024=output/tcga_luad_frozen_high_risk_baseline \
  --run ffpe_1024=output/tcga_luad_ffpe_high_risk_baseline \
  --run ffpe_2048=output/tcga_luad_ffpe_high_risk_2048_sensitivity \
  --outdir output/tcga_luad_high_risk_1024_2048_summary

cat output/tcga_luad_high_risk_1024_2048_summary/comparison_summary.tsv
```

## Relevant Files

- Config: `configs/tcga_luad_ffpe_high_risk_2048_sensitivity.yaml`
- Summary output directory on MLSC: `output/tcga_luad_high_risk_1024_2048_summary`
- Current primary distillation config: `configs/tcga_luad_online_frozen_to_ffpe_distill_high_risk.yaml`
