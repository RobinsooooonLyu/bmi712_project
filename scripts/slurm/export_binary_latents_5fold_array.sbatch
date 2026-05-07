#!/bin/bash
#SBATCH --account=qtim
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --array=0-4%2
#SBATCH --job-name=export_lat
#SBATCH -o logs/slurm/export_lat_%A_%a.out
#SBATCH -e logs/slurm/export_lat_%A_%a.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${SUBMIT_DIR}"
CACHE_BASE="${REPO_ROOT}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:?set BASE_RUN_ROOT}"
LATENT_OUTPUT_ROOT="${LATENT_OUTPUT_ROOT:?set LATENT_OUTPUT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-}"
JOB_LABEL="${JOB_LABEL:-export_latents}"
TRAIN_VIEWS="${TRAIN_VIEWS:-1}"
AUGMENT_TRAIN="${AUGMENT_TRAIN:-0}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${CACHE_BASE}/manifests/cache/huggingface"
mkdir -p "${CACHE_BASE}/manifests/cache/torch"
mkdir -p "${CACHE_BASE}/manifests/cache/xdg"
mkdir -p "${LATENT_OUTPUT_ROOT}"

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

RUN_DIR="${BASE_RUN_ROOT}/fold${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${LATENT_OUTPUT_ROOT}/fold${SLURM_ARRAY_TASK_ID}"

echo "===== Latent export job starting on $(hostname) at $(date) ====="
echo "Repo root         : ${REPO_ROOT}"
echo "Run dir           : ${RUN_DIR}"
echo "Output dir        : ${OUT_DIR}"
echo "Array task id     : ${SLURM_ARRAY_TASK_ID}"
echo "Python bin        : ${PYTHON_BIN}"
echo "Job label         : ${JOB_LABEL}"
echo "Train views       : ${TRAIN_VIEWS}"
echo "Augment train     : ${AUGMENT_TRAIN}"
echo "HF_HOME           : ${HF_HOME}"
echo "HF_HUB_CACHE      : ${HF_HUB_CACHE}"
echo "HF_HUB_OFFLINE    : ${HF_HUB_OFFLINE}"
echo "TORCH_HOME        : ${TORCH_HOME}"
echo "XDG_CACHE_HOME    : ${XDG_CACHE_HOME}"

EXPORT_ARGS=(
  --run-dir "${RUN_DIR}"
  --outdir "${OUT_DIR}"
)

if [[ "${TRAIN_VIEWS}" != "1" ]]; then
  EXPORT_ARGS+=(--train-views "${TRAIN_VIEWS}")
fi

if [[ "${AUGMENT_TRAIN}" == "1" ]]; then
  EXPORT_ARGS+=(--augment-train)
fi

"${PYTHON_BIN}" -u scripts/export_binary_patient_latents.py "${EXPORT_ARGS[@]}"

echo "===== Latent export job finished at $(date) ====="
