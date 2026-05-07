# Manifests

Store small text manifests and cohort TSVs here, not the downloaded WSI payloads.

Expected examples:

- `full_strict/tcga_luad_paired_case_table_strict.tsv`
- `full_strict/tcga_luad_paired_dx_manifest_strict.txt`
- `full_strict/tcga_luad_paired_tissue_manifest_strict.txt`
- `full_strict/tcga_luad_paired_case_data_strict.tsv`
- `full_strict/tcga_luad_patient_manifest_strict.tsv`
- `full_strict/tcga_luad_patient_manifest_trainready.tsv`
- `full_strict/tcga_luad_high_risk_grade_vpi_lvi_5fold_trainready.tsv`
- `pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv`

Current expectation:

- the active patient manifest contains slide lists and clinicopathologic metadata only
- it does not carry DFI/PFI outcome columns in the active branch
- the active training label comes from `tcga_luad_grade_endpoint_labels.tsv`

The actual downloaded slide files should live on MLSC server storage, not inside the repository.

Useful helpers:

- `python3 scripts/build_tcga_luad_wsi_manifests.py`
- `python3 scripts/build_tcga_luad_case_data.py`
- `python3 scripts/build_tcga_luad_patient_manifest.py`
- `python3 scripts/make_tcga_binary_label_split.py`
- `python3 scripts/fetch_tcga_pathology_reports.py`
- `python3 scripts/ocr_tcga_pathology_reports.py`
- `python3 scripts/parse_tcga_grade_from_reports.py`
- `python3 scripts/build_tcga_grade_endpoint_labels.py`

Important:

- staged/smoke manifests are only for verification
- they should still download into the canonical TCGA LUAD `ffpe` and `frozen` directories
- do not create separate permanent pilot data directories
- the active split table for the current pathology pilot is `tcga_luad_high_risk_grade_vpi_lvi_5fold_trainready.tsv`
