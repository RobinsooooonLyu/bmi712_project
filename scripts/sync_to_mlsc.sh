#!/usr/bin/env bash
set -euo pipefail

# Optional local-to-MLSC rsync helper.
# Prefer server_git_sync.sh for normal code updates when MLSC can pull from GitHub.

SSH_HOST_ALIAS="${SSH_HOST_ALIAS:-mlsc}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/"
REMOTE_DIR="${REMOTE_BASE_DIR}/lung-frozen-ffpe-multimodal/"

RSYNC_FLAGS=(
  -av
  --delete
)

LOCAL_OVERRIDE="${SCRIPT_DIR}/sync_to_mlsc.local.sh"
if [[ -f "${LOCAL_OVERRIDE}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_OVERRIDE}"
fi

echo "Local source:  ${SOURCE_DIR}"
echo "Remote target: ${SSH_HOST_ALIAS}:${REMOTE_DIR}"

ssh "${SSH_HOST_ALIAS}" "mkdir -p '${REMOTE_DIR}'"
rsync "${RSYNC_FLAGS[@]}" "${SOURCE_DIR}" "${SSH_HOST_ALIAS}:${REMOTE_DIR}"

echo "Sync complete."
