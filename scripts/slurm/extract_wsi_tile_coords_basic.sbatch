#!/bin/bash
#SBATCH --account=qtim
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tile_coords
#SBATCH -o logs/slurm/tile_coords_%j.out
#SBATCH -e logs/slurm/tile_coords_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${SUBMIT_DIR}"
PROJECT_BASE="$(cd "${REPO_ROOT}/../.." && pwd)"

INPUT_DIR="${INPUT_DIR:?set INPUT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
JOB_LABEL="${JOB_LABEL:-tile_coords}"
TARGET_MPP="${TARGET_MPP:-0.5}"
MASK_MPP="${MASK_MPP:-8.0}"
TILE_SIZE="${TILE_SIZE:-224}"
MASK_THRESHOLD="${MASK_THRESHOLD:-220}"
MIN_SATURATION="${MIN_SATURATION:-12}"
MIN_TISSUE_COVERAGE="${MIN_TISSUE_COVERAGE:-0.5}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${PROJECT_BASE}/logs/slurm"
mkdir -p "${OUTPUT_DIR}"

echo "===== Tile coordinate extraction starting on $(hostname) at $(date) ====="
echo "Repo root          : ${REPO_ROOT}"
echo "Input dir          : ${INPUT_DIR}"
echo "Output dir         : ${OUTPUT_DIR}"
echo "Job label          : ${JOB_LABEL}"
echo "Target MPP         : ${TARGET_MPP}"
echo "Mask MPP           : ${MASK_MPP}"
echo "Tile size          : ${TILE_SIZE}"
echo "Mask threshold     : ${MASK_THRESHOLD}"
echo "Min saturation     : ${MIN_SATURATION}"
echo "Min tissue coverage: ${MIN_TISSUE_COVERAGE}"
echo "Limit              : ${LIMIT}"
echo "Overwrite          : ${OVERWRITE}"

cd "${REPO_ROOT}"

CMD=(
  python3 scripts/extract_wsi_tile_coords.py
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --target-mpp "${TARGET_MPP}"
  --mask-mpp "${MASK_MPP}"
  --tile-size "${TILE_SIZE}"
  --mask-threshold "${MASK_THRESHOLD}"
  --min-saturation "${MIN_SATURATION}"
  --min-tissue-coverage "${MIN_TISSUE_COVERAGE}"
)

if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

"${CMD[@]}"

echo "===== Tile coordinate extraction finished at $(date) ====="
