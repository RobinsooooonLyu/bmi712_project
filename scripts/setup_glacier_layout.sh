#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal}"

mkdir -p "${BASE_DIR}/repo"
mkdir -p "${BASE_DIR}/data/wsi/tcga_luad/ffpe"
mkdir -p "${BASE_DIR}/data/wsi/tcga_luad/frozen"
mkdir -p "${BASE_DIR}/logs"

echo "Created layout under ${BASE_DIR}"
echo "Repo path:         ${BASE_DIR}/repo/lung-frozen-ffpe-multimodal"
echo "Full FFPE WSI:     ${BASE_DIR}/data/wsi/tcga_luad/ffpe"
echo "Full frozen WSI:   ${BASE_DIR}/data/wsi/tcga_luad/frozen"
