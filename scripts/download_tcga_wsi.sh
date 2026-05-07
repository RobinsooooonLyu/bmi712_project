#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/download_tcga_wsi.sh <manifest_path> <output_dir> [progress_log]" >&2
  exit 1
fi

MANIFEST_PATH="$1"
OUTPUT_DIR="$2"
PROGRESS_LOG="${3:-${PROGRESS_LOG:-}}"
FAIL_FAST="${FAIL_FAST:-0}"

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${PROGRESS_LOG}" ]]; then
  mkdir -p "$(dirname "${PROGRESS_LOG}")"
  if [[ ! -f "${PROGRESS_LOG}" ]]; then
    printf 'timestamp\tfile_id\tfilename\tstatus\texpected_size\tactual_size\toutput_dir\n' > "${PROGRESS_LOG}"
  fi
fi

log_progress() {
  if [[ -z "${PROGRESS_LOG}" ]]; then
    return
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$1" "$2" "$3" "$4" "$5" "$6" >> "${PROGRESS_LOG}"
}

echo "Manifest:   ${MANIFEST_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
if [[ -n "${PROGRESS_LOG}" ]]; then
  echo "Progress:   ${PROGRESS_LOG}"
fi
echo "Tip: keep FFPE and frozen downloads in separate directories."
echo "Example FFPE dir:   /autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/data/wsi/tcga_luad/ffpe"
echo "Example frozen dir: /autofs/space/crater_001/datasets/private/nsclc_multimodal/frozen_ffpe_multimodal/data/wsi/tcga_luad/frozen"
echo "Use a small manifest first to verify a few matched cases, then download the full cohort into the same canonical directories."
echo "Downloader: GDC API via curl"

completed_count=0
skipped_count=0
failed_count=0

while IFS=$'\t' read -r id filename md5 size state; do
  if [[ -z "${id}" || -z "${filename}" ]]; then
    continue
  fi
  target="${OUTPUT_DIR}/${filename}"
  current_size=0

  if [[ -f "${target}" && -n "${size}" ]]; then
    current_size="$(stat -c%s "${target}")"
    if [[ "${current_size}" == "${size}" ]]; then
      echo "Skipping ${filename} (already complete)"
      log_progress "${id}" "${filename}" "skipped_complete" "${size}" "${current_size}" "${OUTPUT_DIR}"
      skipped_count=$((skipped_count + 1))
      continue
    fi
    echo "Resuming ${filename} from ${current_size} bytes"
    log_progress "${id}" "${filename}" "resume_start" "${size}" "${current_size}" "${OUTPUT_DIR}"
  else
    echo "Downloading ${filename}"
    log_progress "${id}" "${filename}" "download_start" "${size}" "${current_size}" "${OUTPUT_DIR}"
  fi

  success=0
  for attempt in 1 2 3 4 5; do
    if curl -fL -C - --retry 3 --retry-delay 5 \
      -o "${target}" \
      "https://api.gdc.cancer.gov/data/${id}"; then
      success=1
      break
    fi
    echo "Attempt ${attempt} failed for ${filename}; sleeping before retry"
    sleep 10
  done

  if [[ "${success}" != "1" ]]; then
    echo "ERROR: failed to download ${filename}" >&2
    actual_size=0
    if [[ -f "${target}" ]]; then
      actual_size="$(stat -c%s "${target}")"
    fi
    log_progress "${id}" "${filename}" "failed" "${size}" "${actual_size}" "${OUTPUT_DIR}"
    failed_count=$((failed_count + 1))
    if [[ "${FAIL_FAST}" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  actual_size="$(stat -c%s "${target}")"
  if [[ -n "${size}" && "${actual_size}" != "${size}" ]]; then
    echo "ERROR: size mismatch for ${filename} (${actual_size} != ${size})" >&2
    log_progress "${id}" "${filename}" "size_mismatch" "${size}" "${actual_size}" "${OUTPUT_DIR}"
    failed_count=$((failed_count + 1))
    if [[ "${FAIL_FAST}" == "1" ]]; then
      exit 1
    fi
    continue
  fi
  log_progress "${id}" "${filename}" "completed" "${size}" "${actual_size}" "${OUTPUT_DIR}"
  completed_count=$((completed_count + 1))
done < <(tail -n +2 "${MANIFEST_PATH}")

echo "Completed: ${completed_count}"
echo "Skipped:   ${skipped_count}"
echo "Failed:    ${failed_count}"
echo "Download complete."
