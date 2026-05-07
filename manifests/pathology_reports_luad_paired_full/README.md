# TCGA LUAD Paired Pathology Reports

This directory stores local artifacts for tumor-grade extraction from public TCGA LUAD pathology reports for the paired LUAD pilot cohort.

Contents tracked in git:
- `text/`: extracted UTF-8 text for each downloaded pathology report PDF
- `tcga_luad_pathology_report_files.tsv`: GDC report-file listing for the paired LUAD case table
- `keyword_hits.tsv`: keyword hit summary from the fetch script
- `tcga_luad_grade_labels.tsv`: file-level parsed tumor-grade labels
- `tcga_luad_grade_case_labels.tsv`: case-level aggregated tumor-grade labels
- `tcga_luad_grade_case_evidence.tsv`: case-level final decision table with determining evidence snippets
- `tcga_luad_grade_endpoint_labels.tsv`: derived binary labels for high-grade and high-risk pathology endpoints
- `tcga_luad_high_risk_promotion_review.tsv`: reviewed G1/G2 cases promoted to high-risk by VPI/LVI evidence

Contents not tracked in git:
- `pdf/`: raw downloaded PDFs are kept local only because the full corpus is large

Current coverage after OCR-assisted extraction and manual review:
- report files fetched: `451`
- unique paired LUAD cases with at least one report: `450`
- case-level usable parsed grade labels: `403 / 450`
- case-level status counts:
  - `parsed`: `403`
  - `ambiguous`: `47`
  - `missing`: `0`
  - `conflict`: `0`
- parsed case-level grade counts:
  - grade `1`: `60`
  - grade `2`: `167`
  - grade `3`: `176`
- derived high-grade endpoint:
  - usable cases: `403`
  - high-grade label: `G3` versus `G1/G2`
  - high-grade positive: `176`
  - high-grade negative: `227`
- high-risk pathology endpoint:
  - high-risk positive if `G3` or if a `G1/G2` report has explicit positive visceral pleural invasion or lymphovascular/vascular invasion evidence
  - high-risk positive: `199`
  - high-risk negative: `204`
  - G1/G2 cases promoted by conservative VPI/LVI parsing and review: `23`
  - promotion evidence and review decisions are in `tcga_luad_high_risk_promotion_review.tsv`

Decision rules:
- `G 1`, `G 2`, `G 3`
- `well differentiated`, `moderately differentiated`, `poorly differentiated`
- `grade 3 (of 4)` and `grade 4 (of 4)` are mapped to `G3`
- `grade 2 (of 4)` is mapped to `G2`
- predominant bronchioloalveolar/lepidic/minimally invasive pattern is mapped to `G1`
- predominant acinar, papillary, tubular, or tubulopapillary pattern is mapped to `G2`
- predominant solid, micropapillary, complex glandular, polymorphocellular, or high-grade wording is mapped to `G3`
- mixed solid/acinar/bronchioloalveolar without predominance or explicit grade is excluded rather than forced to `G3`
- unreliable numeric table OCR without a clear selected grade is excluded
- missing-pathology-report forms and non-diagnostic lymph-node-only pages are excluded
- multifocal reports with discordant tumor grades are excluded unless a tumor/slide target was clearly selected during review

Rebuild steps:
```bash
python3 scripts/fetch_tcga_pathology_reports.py \
  --project TCGA-LUAD \
  --case-table manifests/local_full_strict/tcga_luad_paired_case_table_strict.tsv \
  --download \
  --extract-text \
  --outdir manifests/pathology_reports_luad_paired_full

python3 scripts/ocr_tcga_pathology_reports.py \
  --pdf-dir manifests/pathology_reports_luad_paired_full/pdf \
  --text-dir manifests/pathology_reports_luad_paired_full/text \
  --case-labels manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_labels.tsv \
  --status missing \
  --overwrite

python3 scripts/parse_tcga_grade_from_reports.py \
  --text-dir manifests/pathology_reports_luad_paired_full/text \
  --pdf-dir manifests/pathology_reports_luad_paired_full/pdf \
  --case-table manifests/local_full_strict/tcga_luad_paired_case_table_strict.tsv \
  --out manifests/pathology_reports_luad_paired_full/tcga_luad_grade_labels.tsv \
  --case-out manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_labels.tsv

python3 scripts/build_grade_curation_evidence.py \
  --case-labels manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_labels.tsv \
  --text-dir manifests/pathology_reports_luad_paired_full/text \
  --out manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_evidence.tsv

python3 scripts/build_tcga_grade_endpoint_labels.py \
  --grade-labels manifests/pathology_reports_luad_paired_full/tcga_luad_grade_case_labels.tsv \
  --text-dir manifests/pathology_reports_luad_paired_full/text \
  --out manifests/pathology_reports_luad_paired_full/tcga_luad_grade_endpoint_labels.tsv
```

Notes:
- The raw PDF cache remains untracked under `pdf/`.
- The extracted text files include OCR output for previously missing or low-quality PDFs.
- The `ambiguous` status in the final case table includes manually reviewed exclusions, not only unresolved parser ambiguity.
- The primary modeling endpoint for the immediate FFPE/frozen baseline is now `high_risk_grade_vpi_lvi`. The stricter `high_grade` endpoint remains available as a secondary/sensitivity endpoint.
- Two cases (`TCGA-MP-A4T4`, `TCGA-MP-A4T9`) were corrected from OCR-derived `G1` to manual `G3` after the VPI/LVI review surfaced explicit `grade 4 (of 4)` and `grade 3 (of 4)` final diagnosis wording.
- A high-grade audit also corrected morphology-derived overcalls where reports described acinar/papillary/bronchioloalveolar-predominant tumors with only minor solid or micropapillary components. These were moved out of `high_grade`.
