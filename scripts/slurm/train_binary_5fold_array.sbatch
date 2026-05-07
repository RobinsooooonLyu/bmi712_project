#!/bin/bash
#SBATCH --account=qtim
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --gpus=1
#SBATCH --array=0-4%2
#SBATCH --job-name=train_bin_cv
#SBATCH -o logs/slurm/train_bin_cv_%A_%a.out
#SBATCH -e logs/slurm/train_bin_cv_%A_%a.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${SUBMIT_DIR}"
CACHE_BASE="${REPO_ROOT}"

BASE_CONFIG_PATH="${BASE_CONFIG_PATH:?set BASE_CONFIG_PATH}"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${REPO_ROOT}/configs/generated}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-}"
JOB_LABEL="${JOB_LABEL:-binary_cv}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${CACHE_BASE}/manifests/cache/huggingface"
mkdir -p "${CACHE_BASE}/manifests/cache/torch"
mkdir -p "${CACHE_BASE}/manifests/cache/xdg"
mkdir -p "${GENERATED_CONFIG_DIR}"

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

"${PYTHON_BIN}" scripts/write_baseline_fold_configs.py \
  --base-config "${BASE_CONFIG_PATH}" \
  --outdir "${GENERATED_CONFIG_DIR}"

CONFIG_PATH="${GENERATED_CONFIG_DIR}/$(basename "${BASE_CONFIG_PATH%.yaml}")_fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "===== Binary CV job starting on $(hostname) at $(date) ====="
echo "Repo root         : ${REPO_ROOT}"
echo "Base config       : ${BASE_CONFIG_PATH}"
echo "Generated cfg dir : ${GENERATED_CONFIG_DIR}"
echo "Array task id     : ${SLURM_ARRAY_TASK_ID}"
echo "Config path       : ${CONFIG_PATH}"
echo "Python bin        : ${PYTHON_BIN}"
echo "Job label         : ${JOB_LABEL}"
echo "HF_HOME           : ${HF_HOME}"
echo "HF_HUB_CACHE      : ${HF_HUB_CACHE}"
echo "HF_HUB_OFFLINE    : ${HF_HUB_OFFLINE}"
echo "TORCH_HOME        : ${TORCH_HOME}"
echo "XDG_CACHE_HOME    : ${XDG_CACHE_HOME}"

"${PYTHON_BIN}" -u -m src.train.train_binary_baseline --config "${CONFIG_PATH}"

echo "===== Binary CV job finished at $(date) ====="
