#!/bin/bash
#SBATCH --account=qtim
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --array=0-4%2
#SBATCH --job-name=train_dist
#SBATCH -o logs/slurm/train_dist_%A_%a.out
#SBATCH -e logs/slurm/train_dist_%A_%a.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${SUBMIT_DIR}"

BASE_CONFIG_PATH="${BASE_CONFIG_PATH:?set BASE_CONFIG_PATH}"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${REPO_ROOT}/configs/generated}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-}"
JOB_LABEL="${JOB_LABEL:-deterministic_distill}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${GENERATED_CONFIG_DIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ -n "${VENV_PATH}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/write_distill_fold_configs.py \
  --base-config "${BASE_CONFIG_PATH}" \
  --outdir "${GENERATED_CONFIG_DIR}"

CONFIG_PATH="${GENERATED_CONFIG_DIR}/$(basename "${BASE_CONFIG_PATH%.yaml}")_fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "===== Deterministic distillation job starting on $(hostname) at $(date) ====="
echo "Repo root         : ${REPO_ROOT}"
echo "Base config       : ${BASE_CONFIG_PATH}"
echo "Generated cfg dir : ${GENERATED_CONFIG_DIR}"
echo "Array task id     : ${SLURM_ARRAY_TASK_ID}"
echo "Config path       : ${CONFIG_PATH}"
echo "Python bin        : ${PYTHON_BIN}"
echo "Job label         : ${JOB_LABEL}"

"${PYTHON_BIN}" -u -m src.train.train_deterministic_distillation --config "${CONFIG_PATH}"

echo "===== Deterministic distillation job finished at $(date) ====="
