#!/bin/bash
#SBATCH --account=qtim
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tcga_wsi_dl
#SBATCH -o logs/slurm/tcga_wsi_%j.out
#SBATCH -e logs/slurm/tcga_wsi_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${SUBMIT_DIR}"
PROJECT_BASE="$(cd "${REPO_ROOT}/../.." && pwd)"

MANIFEST_PATH="${MANIFEST_PATH:?set MANIFEST_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
JOB_LABEL="${JOB_LABEL:-tcga_wsi}"
PROGRESS_LOG="${PROGRESS_LOG:-${PROJECT_BASE}/logs/progress/${JOB_LABEL}.tsv}"

mkdir -p "${PROJECT_BASE}/logs/slurm"
mkdir -p "${PROJECT_BASE}/logs/progress"

echo "===== TCGA WSI download job starting on $(hostname) at $(date) ====="
echo "Repo root    : ${REPO_ROOT}"
echo "Manifest     : ${MANIFEST_PATH}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Progress log : ${PROGRESS_LOG}"
echo "Job label    : ${JOB_LABEL}"

cd "${REPO_ROOT}"

bash scripts/download_tcga_wsi.sh "${MANIFEST_PATH}" "${OUTPUT_DIR}" "${PROGRESS_LOG}"

echo "===== TCGA WSI download job finished at $(date) ====="
