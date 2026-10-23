#!/usr/bin/env bash
set -euo pipefail

PARENT_DIR="/nfs/masi/taylogc1/HBCD_diff_morph"
OUT_ROOT="/nfs/masi/taylogc1/HBCD_diff_morph/derivatives_autotrack_2"
T2W_SCRIPT="/nfs/masi/taylogc1/HBCD_diff_morph/T2w_diff_morph_4.py"
TRACT2TRACT_SCRIPT="/nfs/masi/taylogc1/HBCD_diff_morph/tract2tract_diff_morph_4.py"
T2W_ROOT_NAME="preproc_T2w_autotrack"
TRACT_ROOT_NAME="DSI_tracts_autotrack"
YOUNGER_SESSION="ses-V02"
OLDER_SESSION="ses-V03"

TRACT_NAMES=(
  "CST"
  "FX"
  "IFOF"
  "ILF"
  "OR"
  "UF"
)

HEMIS=(
  "L"
  "R"
)

SUBJECTS=(
"sub-9858936625"
)

for SUBJECT in "${SUBJECTS[@]}"; do
  echo "Running subject: ${SUBJECT}"

  YOUNGER_T2W_FILE="${SUBJECT}_${YOUNGER_SESSION}_T2w_RAS_synthstrip-brain_fillholesmask_N4_rescale0-1.nii.gz"
  OLDER_T2W_FILE="${SUBJECT}_${OLDER_SESSION}_T2w_RAS_synthstrip-brain_fillholesmask_N4_rescale0-1.nii.gz"
  OLDER_T2W_MASK_FILE="${SUBJECT}_${OLDER_SESSION}_T2w_RAS_synthstrip-mask-bin_fillholes.nii.gz"
  SUBJECT_OUT_ROOT="${OUT_ROOT}"

  echo "Running T2w image-based morphometry pipeline..."
  python "${T2W_SCRIPT}" \
    --parent-dir "${PARENT_DIR}" \
    --T2w-root-name "${T2W_ROOT_NAME}" \
    --tract-root-name "${TRACT_ROOT_NAME}" \
    --subject-id "${SUBJECT}" \
    --younger-session "${YOUNGER_SESSION}" \
    --older-session "${OLDER_SESSION}" \
    --younger-T2w-file "${YOUNGER_T2W_FILE}" \
    --older-T2w-file "${OLDER_T2W_FILE}" \
    --older-T2w-mask-file "${OLDER_T2W_MASK_FILE}" \
    --tract-names "${TRACT_NAMES[@]}" \
    --hemis "${HEMIS[@]}" \
    --out-dir "${SUBJECT_OUT_ROOT}/T2w_diff_morph" \
    --syn-level-iters 100 50 25 \
    --n-nodes 100 \
    --jac-step-mm 0.5 \
    --endpoint-start-distance-mm 20 \
    --endpoint-end-distance-mm 20 \
    --min-centerline-coverage-frac 0.50

  echo "Running tract-to-tract morphometry pipeline..."
  python "${TRACT2TRACT_SCRIPT}" \
    --parent-dir "${PARENT_DIR}" \
    --subject-id "${SUBJECT}" \
    --younger-session "${YOUNGER_SESSION}" \
    --older-session "${OLDER_SESSION}" \
    --tract-root-name "${TRACT_ROOT_NAME}" \
    --tract-names "${TRACT_NAMES[@]}" \
    --hemis "${HEMIS[@]}" \
    --out-dir "${SUBJECT_OUT_ROOT}/tract2tract_diff_morph" \
    --run-mask-syn \
    --run-bundlewarp \
    --mask-syn-level-iters 100 50 25 \
    --n-nodes 100 \
    --jac-step-mm 0.5 \
    --bundlewarp-alpha 0.5 \
    --bundlewarp-beta 20 \
    --bundlewarp-max-iter 15 \
    --bundlewarp-slr-min-length-mm 10 \
    --bundlewarp-slr-max-length-mm 250 \
    --endpoint-start-distance-mm 20 \
    --endpoint-end-distance-mm 20 \
    --min-centerline-coverage-frac 0.50

  echo "Finished subject: ${SUBJECT}"
done

echo "All subjects complete."