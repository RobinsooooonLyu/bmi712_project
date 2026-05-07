#!/usr/bin/env python3
import argparse
import csv
import os
import sys


def read_tsv(path):
    with open(path, encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--case-table',
        default='manifests/full_strict/tcga_luad_paired_case_table_strict.tsv',
        help='Paired slide case table from build_tcga_luad_wsi_manifests.py',
    )
    parser.add_argument(
        '--case-data',
        default='manifests/full_strict/tcga_luad_paired_case_data_strict.tsv',
        help='Paired case metadata table from build_tcga_luad_case_data.py',
    )
    parser.add_argument(
        '--out',
        default='manifests/full_strict/tcga_luad_patient_manifest_strict.tsv',
        help='Output patient-level manifest TSV',
    )
    args = parser.parse_args()

    case_rows = read_tsv(args.case_table)
    data_rows = {row['case_id']: row for row in read_tsv(args.case_data)}

    fieldnames = [
        'case_id',
        'ffpe_slide_count',
        'frozen_slide_count',
        'ffpe_filenames',
        'frozen_filenames',
        'ajcc_pathologic_stage',
        'pathologic_stage_group',
        'pathologic_split_group',
        'ajcc_pathologic_t',
        'ajcc_clinical_stage',
        'ajcc_clinical_t',
        'clinical_stage_hybrid',
        'gender',
        'age_at_diagnosis_days',
        'primary_diagnosis',
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows:
            case_id = row['case_id']
            meta = data_rows.get(case_id, {})
            writer.writerow(
                {
                    'case_id': case_id,
                    'ffpe_slide_count': row['dx_files'],
                    'frozen_slide_count': row['tissue_files'],
                    'ffpe_filenames': row['dx_filenames'],
                    'frozen_filenames': row['tissue_filenames'],
                    'ajcc_pathologic_stage': meta.get('ajcc_pathologic_stage', ''),
                    'pathologic_stage_group': meta.get('pathologic_stage_group', ''),
                    'pathologic_split_group': meta.get('pathologic_split_group', ''),
                    'ajcc_pathologic_t': meta.get('ajcc_pathologic_t', ''),
                    'ajcc_clinical_stage': meta.get('ajcc_clinical_stage', ''),
                    'ajcc_clinical_t': meta.get('ajcc_clinical_t', ''),
                    'clinical_stage_hybrid': meta.get('clinical_stage_hybrid', ''),
                    'gender': meta.get('gender', ''),
                    'age_at_diagnosis_days': meta.get('age_at_diagnosis_days', ''),
                    'primary_diagnosis': meta.get('primary_diagnosis', ''),
                }
            )

    print(f'Input paired cases: {len(case_rows)}')
    print(f'Patient manifest: {args.out}')


if __name__ == '__main__':
    sys.exit(main())
