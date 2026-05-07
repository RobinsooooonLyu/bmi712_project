# Repo and Server Workflow

This project should use the same local/GitHub/server pattern that has already worked in other repos.

## Source of truth

GitHub is the source of truth for code.

- local machine: editing, commits, branching, code review
- GitHub: canonical code state
- MLSC: execution environment for data-heavy work

Do not treat the server checkout as the master copy of the codebase.

## Recommended Repo Layout

Current local working repository:

`/Users/yangzhizhou/Documents/GitHub/lung-frozen-ffpe-multimodal-tcga-pilot`

Current MLSC execution checkout:

`/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal`

Recommended active worktrees:

- `main`: stable branch
- `codex/tcga-pilot`: TCGA LUAD proof-of-concept work
- `codex/mgh-pathology`: pathology-only institutional model
- `codex/ct-multimodal`: CT and fusion work

The main idea is to isolate major phases in separate worktrees rather than constantly switching one checkout between unrelated branches.

## Development loop

1. Edit locally.
2. Commit locally.
3. Push to GitHub.
4. On MLSC, run the server sync script to pull the latest code.
5. Run data jobs on MLSC.
6. Keep large data and outputs on MLSC, not in git.

## Server sync rule

Prefer `server_git_sync.sh` for normal updates.

Only use rsync when Git cannot represent the operation you need. For example:

- copying a non-git local config file to the server
- pushing a generated manifest or one-off helper file before it is committed

## Data rule

Whole-slide images and CT data should not live in the Git repository.

The repo should only contain:

- code
- docs
- manifests
- metadata tables that are safe to version
- small config templates

The actual WSI and CT payloads should live in server-side data directories configured outside git.

## Glacier Layout For This Project

Current base path:

`/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal`

Recommended subdirectories:

- repo:
  `/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/repo/lung-frozen-ffpe-multimodal`
- TCGA LUAD FFPE WSI:
  `/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/data/wsi/tcga_luad/ffpe`
- TCGA LUAD frozen WSI:
  `/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/data/wsi/tcga_luad/frozen`

There should be only one canonical TCGA LUAD WSI dataset on disk. The pilot is a manifest-level subset used to download and inspect a few matched cases first. After verification, the full manifest should continue downloading into the same `ffpe` and `frozen` directories rather than into separate pilot directories.
