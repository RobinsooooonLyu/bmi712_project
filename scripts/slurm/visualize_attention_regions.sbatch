#!/bin/bash
#SBATCH --account=qtim
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --job-name=attn_regions
#SBATCH -o logs/slurm/attn_regions_%j.out
#SBATCH -e logs/slurm/attn_regions_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${SUBMIT_DIR}"
CACHE_BASE="${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-}"
OUTDIR="${OUTDIR:-output/tcga_luad_attention_region_examples}"
MAX_ATTENTION_TILES="${MAX_ATTENTION_TILES:-0}"
MIN_DISPLAY_ATTENTION="${MIN_DISPLAY_ATTENTION:-0.10}"
EDGE_MARGIN_FRACTION="${EDGE_MARGIN_FRACTION:-0.08}"
MIN_WINDOW_MEAN_TISSUE_FRACTION="${MIN_WINDOW_MEAN_TISSUE_FRACTION:-0.35}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${CACHE_BASE}/manifests/cache/huggingface"
mkdir -p "${CACHE_BASE}/manifests/cache/torch"
mkdir -p "${CACHE_BASE}/manifests/cache/xdg"

export HF_HOME="${HF_HOME:-${CACHE_BASE}/manifests/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_BASE}/manifests/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE}/manifests/cache/xdg}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${VENV_PATH}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

cd "${REPO_ROOT}"

echo "===== Attention region visualization starting on $(hostname) at $(date) ====="
echo "Repo root           : ${REPO_ROOT}"
echo "Python bin          : ${PYTHON_BIN}"
echo "Output dir          : ${OUTDIR}"
echo "Max attention tiles : ${MAX_ATTENTION_TILES}"
echo "Min display attn    : ${MIN_DISPLAY_ATTENTION}"
echo "Edge margin frac    : ${EDGE_MARGIN_FRACTION}"
echo "Min window tissue   : ${MIN_WINDOW_MEAN_TISSUE_FRACTION}"
echo "HF_HOME             : ${HF_HOME}"
echo "HF_HUB_CACHE        : ${HF_HUB_CACHE}"
echo "HF_HUB_OFFLINE      : ${HF_HUB_OFFLINE}"

"${PYTHON_BIN}" -u scripts/visualize_attention_regions.py \
  --frozen-run-dir output/tcga_luad_frozen_high_risk_baseline \
  --ffpe-run-dir output/tcga_luad_ffpe_high_risk_baseline \
  --split-path manifests/full_strict/tcga_luad_high_risk_grade_vpi_lvi_5fold_trainready.tsv \
  --outdir "${OUTDIR}" \
  --max-attention-tiles "${MAX_ATTENTION_TILES}" \
  --window-rows 8 \
  --window-cols 8 \
  --min-window-coverage 0.65 \
  --edge-margin-fraction "${EDGE_MARGIN_FRACTION}" \
  --min-window-mean-tissue-fraction "${MIN_WINDOW_MEAN_TISSUE_FRACTION}" \
  --thumbnail-max-size 1600 \
  --region-tile-size 128 \
  --min-display-attention "${MIN_DISPLAY_ATTENTION}" \
  --device cuda

echo "===== Attention region visualization finished at $(date) ====="
