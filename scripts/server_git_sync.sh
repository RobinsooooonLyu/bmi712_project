#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:yangzhizhou111/lung-frozen-ffpe-multimodal.git}"
SERVER_REPO_DIR="${SERVER_REPO_DIR:-/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

mkdir -p "$(dirname "${SERVER_REPO_DIR}")"

if [[ ! -d "${SERVER_REPO_DIR}" ]]; then
  echo "Cloning repository into ${SERVER_REPO_DIR}"
  git clone "${REPO_URL}" "${SERVER_REPO_DIR}"
fi

if [[ ! -d "${SERVER_REPO_DIR}/.git" ]]; then
  echo "ERROR: ${SERVER_REPO_DIR} exists but is not a Git repository." >&2
  exit 1
fi

cd "${SERVER_REPO_DIR}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
TARGET_BRANCH="${TARGET_BRANCH:-${CURRENT_BRANCH}}"

if [[ "${CURRENT_BRANCH}" == "HEAD" ]]; then
  echo "ERROR: repository is in detached HEAD state. Check out a branch first." >&2
  exit 1
fi

TRACKED_DIRTY_STATUS="$(git status --porcelain --untracked-files=no)"
if [[ -n "${TRACKED_DIRTY_STATUS}" && "${ALLOW_DIRTY}" != "1" ]]; then
  echo "ERROR: repository has tracked local changes. Commit, stash, or rerun with ALLOW_DIRTY=1." >&2
  git status --short --untracked-files=no
  exit 1
fi

echo "Repository:    ${SERVER_REPO_DIR}"
echo "Remote:        ${REMOTE_NAME}"
echo "Target branch: ${TARGET_BRANCH}"

git fetch "${REMOTE_NAME}" "${TARGET_BRANCH}"

UNTRACKED_FILES="$(git ls-files --others --exclude-standard)"
if [[ -n "${UNTRACKED_FILES}" ]]; then
  REMOTE_TRACKED_PATHS="$(git ls-tree -r --name-only "${REMOTE_NAME}/${TARGET_BRANCH}")"
  COLLISIONS="$(LC_ALL=C comm -12 \
    <(printf '%s\n' "${UNTRACKED_FILES}" | LC_ALL=C sort -u) \
    <(printf '%s\n' "${REMOTE_TRACKED_PATHS}" | LC_ALL=C sort -u))"

  if [[ -n "${COLLISIONS}" ]]; then
    echo "ERROR: untracked server files would be overwritten by tracked files from GitHub." >&2
    printf '%s\n' "${COLLISIONS}" >&2
    exit 1
  fi

  echo "Untracked local files detected; leaving them untouched."
fi

if [[ "${ALLOW_DIRTY}" == "1" ]]; then
  git pull --rebase --autostash "${REMOTE_NAME}" "${TARGET_BRANCH}"
else
  git pull --ff-only "${REMOTE_NAME}" "${TARGET_BRANCH}"
fi

echo "Git sync complete."
