#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/local_git_push.sh \"commit message\"" >&2
  exit 1
fi

COMMIT_MESSAGE="$*"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -d .git ]]; then
  echo "ERROR: ${REPO_ROOT} is not a Git repository." >&2
  exit 1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${CURRENT_BRANCH}" == "HEAD" ]]; then
  echo "ERROR: repository is in detached HEAD state. Check out a branch first." >&2
  exit 1
fi

echo "Repository: ${REPO_ROOT}"
echo "Branch:     ${CURRENT_BRANCH}"
echo "Message:    ${COMMIT_MESSAGE}"

git add -A

if git diff --cached --quiet; then
  echo "No new file changes to commit."
else
  git commit -m "${COMMIT_MESSAGE}"
fi

git pull --rebase origin "${CURRENT_BRANCH}"
git push origin "${CURRENT_BRANCH}"

echo "Push complete."
