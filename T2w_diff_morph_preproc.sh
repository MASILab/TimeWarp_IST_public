#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Process single subject/session: bash T2w_diff_morph_preproc.sh sub-XXX ses-V0X
#   Process all sessions for one subject: bash T2w_diff_morph_preproc.sh sub-XXX
#   Batch process all subjects/sessions in project dir: bash T2w_diff_morph_preproc.sh --batch

PROJECT_DIR="/nfs/masi/taylogc1/HBCD_diff_morph"
RAW_ROOT="${PROJECT_DIR}/raw_T2w_autotrack"
OUT_ROOT="${PROJECT_DIR}/preproc_T2w_autotrack"

# Dockerized FreeSurfer setup
FREESURFER_DOCKER_IMAGE="freesurfer/freesurfer:8.2.0"

# Update this path if your FreeSurfer license is somewhere else.
FS_LICENSE_HOST="/home-local/taylogc1/.licenses/freesurfer/license.txt"

run_freesurfer() {
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "${PROJECT_DIR}:${PROJECT_DIR}" \
    -v "${FS_LICENSE_HOST}:/license.txt:ro" \
    -e FS_LICENSE="/license.txt" \
    "${FREESURFER_DOCKER_IMAGE}" \
    "$@"
}

run_one() {
  local SUB="$1"
  local SES="$2"

  local RAW_DIR="${RAW_ROOT}/${SUB}/${SES}"
  local OUT_DIR="${OUT_ROOT}/${SUB}/${SES}"
  local QC_DIR="${OUT_DIR}/qc"

  mkdir -p "${OUT_DIR}" "${QC_DIR}"

  # Find input T2w

  shopt -s nullglob
  local INPUTS=("${RAW_DIR}"/*_T2w.nii.gz)
  shopt -u nullglob

  if [[ "${#INPUTS[@]}" -eq 0 ]]; then
    echo "ERROR: No *_T2w.nii.gz file found in ${RAW_DIR}"
    return 1
  fi

  if [[ "${#INPUTS[@]}" -gt 1 ]]; then
    echo "ERROR: More than one *_T2w.nii.gz file found in ${RAW_DIR}"
    printf '%s\n' "${INPUTS[@]}"
    return 1
  fi

  local IN_IMG="${INPUTS[0]}"
  local IN_BASE
  IN_BASE="$(basename "${IN_IMG}" .nii.gz)"

  # Find JSON sidecar in raw T2w directory

  local JSON_SIDECAR="${RAW_DIR}/${IN_BASE}.json"

  if [[ ! -f "${JSON_SIDECAR}" ]]; then
    shopt -s nullglob
    local JSON_INPUTS=("${RAW_DIR}"/*_T2w.json)
    shopt -u nullglob

    if [[ "${#JSON_INPUTS[@]}" -eq 1 ]]; then
      JSON_SIDECAR="${JSON_INPUTS[0]}"
    else
      JSON_SIDECAR=""
    fi
  fi

  echo "Welcome to the HBCD_diff_morph preprocessing pipeline!"
  echo "Subject: ${SUB}"
  echo "Session: ${SES}"
  echo "Input image: ${IN_IMG}"
  echo "Output directory: ${OUT_DIR}"
  echo "Base name: ${IN_BASE}"

  if [[ -n "${JSON_SIDECAR}" ]]; then
    echo "JSON sidecar: ${JSON_SIDECAR}"
  else
    echo "JSON sidecar: not found"
  fi

  # Set the output filenames

  local INFO_TXT="${OUT_DIR}/${IN_BASE}_mri-info.txt"
  local JSON_QC_TXT="${OUT_DIR}/${IN_BASE}_json-sidecar-info.txt"

  local RAS_IMG="${OUT_DIR}/${IN_BASE}_RAS.nii.gz"

  local BRAIN_IMG="${OUT_DIR}/${IN_BASE}_RAS_synthstrip-brain.nii.gz"
  local MASK_IMG="${OUT_DIR}/${IN_BASE}_RAS_synthstrip-mask.nii.gz"
  local MASK_BIN_IMG="${OUT_DIR}/${IN_BASE}_RAS_synthstrip-mask-bin.nii.gz"

  local MASK_CLEAN_IMG="${OUT_DIR}/${IN_BASE}_RAS_synthstrip-mask-bin_fillholes.nii.gz"
  local BRAIN_CLEAN_IMG="${OUT_DIR}/${IN_BASE}_RAS_synthstrip-brain_fillholesmask.nii.gz"

  local N4_IMG="${OUT_DIR}/${IN_BASE}_RAS_synthstrip-brain_fillholesmask_N4.nii.gz"

  local NORM_IMG="${OUT_DIR}/${IN_BASE}_RAS_synthstrip-brain_fillholesmask_N4_rescale0-1.nii.gz"

  local QC_PDF="${OUT_DIR}/${IN_BASE}_preproc-QC.pdf"

  # 1. Inspect raw image

  echo "Step 1: Inspecting raw image"
  run_freesurfer mri_info "${IN_IMG}" > "${INFO_TXT}"

  # Save selected JSON sidecar metadata for QC reporting

  if [[ -n "${JSON_SIDECAR}" ]]; then
    python - <<PYCODE
import json

json_file = "${JSON_SIDECAR}"
out_file = "${JSON_QC_TXT}"

fields = [
    "Modality",
    "Manufacturer",
    "ManufacturersModelName",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "FlipAngle",
    "PhaseEncodingDirection",
    "EffectiveEchoSpacing",
    "TotalReadoutTime",
    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "PulseSequenceType",
    "ImageType",
    "ProtocolName",
    "SeriesDescription",
    "BodyPartExamined",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ReconMatrixPE",
    "AcquisitionMatrixPE",
    "PixelBandwidth",
]

try:
    with open(json_file, "r") as f:
        meta = json.load(f)

    lines = []
    lines.append(f"JSON sidecar: {json_file}")
    lines.append("")
    lines.append("Selected metadata:")
    lines.append("")

    for key in fields:
        if key in meta:
            lines.append(f"{key}: {meta[key]}")

    if len(lines) <= 4:
        lines.append("No selected metadata fields found.")

    with open(out_file, "w") as f:
        f.write("\\n".join(lines) + "\\n")

except Exception as e:
    with open(out_file, "w") as f:
        f.write(f"Could not read JSON sidecar: {json_file}\\n")
        f.write(f"Error: {e}\\n")
PYCODE
  else
    {
      echo "JSON sidecar: not found"
      echo ""
      echo "No JSON metadata was added to the QC report."
    } > "${JSON_QC_TXT}"
  fi

  # 2. Reorient to RAS

  echo "Step 2: Checking orientation and reorienting to RAS if needed"

  ORIENT="$(python - <<PYCODE
import nibabel as nib
from nibabel.orientations import aff2axcodes

img = nib.load("${IN_IMG}")
print("".join(aff2axcodes(img.affine)))
PYCODE
)"

  echo "Detected orientation: ${ORIENT}"

  if [[ "${ORIENT}" == "RAS" ]]; then
    echo "Input image is already RAS; copying without reorientation"
    cp "${IN_IMG}" "${RAS_IMG}"
  else
    echo "Input image is ${ORIENT}; reorienting to RAS"
    run_freesurfer mri_convert \
      --out_orientation RAS \
      "${IN_IMG}" \
      "${RAS_IMG}"
  fi

  if [[ ! -f "${RAS_IMG}" ]]; then
    echo "ERROR: RAS image was not created: ${RAS_IMG}"
    return 1
  fi

  # 3. Skull-strip with SynthStrip

  echo "Step 3: Skull-stripping with SynthStrip"
  run_freesurfer mri_synthstrip \
    -i "${RAS_IMG}" \
    -o "${BRAIN_IMG}" \
    -m "${MASK_IMG}" 

  # 4. Binarize SynthStrip mask

  echo "Step 4: Binarizing SynthStrip mask"
  run_freesurfer mri_binarize \
    --i "${MASK_IMG}" \
    --min 0.5 \
    --o "${MASK_BIN_IMG}"

  # 4b. Fill holes in SynthStrip mask

  echo "Step 4b: Filling holes in SynthStrip mask"

  python - <<PYCODE
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes

in_mask = "${MASK_BIN_IMG}"
out_mask = "${MASK_CLEAN_IMG}"

img = nib.load(in_mask)
data = img.get_fdata() > 0

filled = np.zeros_like(data, dtype=bool)

# Fill holes slice-wise in all three anatomical planes.
# The union is intentionally conservative for small internal mask holes.
for z in range(data.shape[2]):
    filled[:, :, z] |= binary_fill_holes(data[:, :, z])

for y in range(data.shape[1]):
    filled[:, y, :] |= binary_fill_holes(data[:, y, :])

for x in range(data.shape[0]):
    filled[x, :, :] |= binary_fill_holes(data[x, :, :])

# Preserve original mask and add only filled internal regions.
filled |= data

out = filled.astype(np.uint8)

nib.save(
    nib.Nifti1Image(out, img.affine, img.header),
    out_mask,
)
PYCODE

  if [[ ! -f "${MASK_CLEAN_IMG}" ]]; then
    echo "ERROR: Cleaned mask was not created: ${MASK_CLEAN_IMG}"
    return 1
  fi
  
  # 4c. Apply cleaned SynthStrip mask

  echo "Step 4c: Applying cleaned SynthStrip mask"
  run_freesurfer mri_mask \
    "${RAS_IMG}" \
    "${MASK_CLEAN_IMG}" \
    "${BRAIN_CLEAN_IMG}"

  if [[ ! -f "${BRAIN_CLEAN_IMG}" ]]; then
    echo "ERROR: Cleaned brain image was not created: ${BRAIN_CLEAN_IMG}"
    return 1
  fi

  # 5. N4 bias-field correction
  # Runs on cleaned skull-stripped image, using cleaned SynthStrip mask

  echo "Step 5: N4 bias-field correction"
  N4BiasFieldCorrection \
    -d 3 \
    -i "${BRAIN_CLEAN_IMG}" \
    -x "${MASK_CLEAN_IMG}" \
    -o "${N4_IMG}"

  if [[ ! -f "${N4_IMG}" ]]; then
    echo "ERROR: N4-corrected image was not created: ${N4_IMG}"
    return 1
  fi

  # 6. Intensity normalization

  echo "Step 6: Intensity rescaling (0-1)"
  ImageMath 3 \
    "${NORM_IMG}" \
    RescaleImage \
    "${N4_IMG}" \
    0 1

  if [[ ! -f "${NORM_IMG}" ]]; then
    echo "ERROR: Normalized image was not created: ${NORM_IMG}"
    return 1
  fi

  # 7. QC PDF

  echo "Step 7: Creating QC PDF"

  python - <<PYCODE
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

images = [
    ("Raw input", "${IN_IMG}"),
    ("RAS checked/reoriented", "${RAS_IMG}"),
    ("SynthStrip brain", "${BRAIN_IMG}"),
    ("SynthStrip mask", "${MASK_BIN_IMG}"),
    ("Hole-filled SynthStrip mask", "${MASK_CLEAN_IMG}"),
    ("Hole-filled SynthStrip brain", "${BRAIN_CLEAN_IMG}"),
    ("N4 corrected brain", "${N4_IMG}"),
    ("N4 + rescale 0-1 brain", "${NORM_IMG}"),
]

qc_pdf = "${QC_PDF}"
info_txt = "${INFO_TXT}"
json_qc_txt = "${JSON_QC_TXT}"

def load_img(path):
    img = nib.load(path)
    data = img.get_fdata()
    return img, data

def robust_limits(data):
    finite = np.isfinite(data)
    vals = data[finite]
    vals = vals[vals != 0]
    if vals.size < 10:
        vals = data[finite]
    if vals.size < 10:
        return 0, 1
    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    if lo == hi:
        hi = lo + 1
    return lo, hi

def get_slices(data):
    shape = data.shape[:3]
    cx, cy, cz = [s // 2 for s in shape]
    sagittal = np.rot90(data[cx, :, :])
    coronal = np.rot90(data[:, cy, :])
    axial = np.rot90(data[:, :, cz])
    return sagittal, coronal, axial

with PdfPages(qc_pdf) as pdf:
    # Cover page
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("T2w preprocessing QC", fontsize=18, y=0.96)

    lines = [
        "Subject: ${SUB}",
        "Session: ${SES}",
        "Input image:",
        "${IN_IMG}",
        "",
        "JSON sidecar:",
        "${JSON_SIDECAR}",
        "",
        "Detected orientation:",
        "${ORIENT}",
        "",
        "Output directory:",
        "${OUT_DIR}",
        "",
        "Preprocessing steps:",
        "1. Docker FreeSurfer: mri_info",
        "2. Orientation check with nibabel; copy if RAS, otherwise Docker FreeSurfer mri_convert --out_orientation RAS",
        "3. Docker FreeSurfer: mri_synthstrip",
        "4. Docker FreeSurfer: mri_binarize",
        "4b. Python: slice-wise binary_fill_holes in x/y/z planes",
        "4c. Docker FreeSurfer: mri_mask with hole-filled mask",
        "5. N4BiasFieldCorrection",
        "6. ImageMath RescaleImage 0 1",
        "",
        "No cropping, padding, voxel-size changes, or template normalization are performed.",
    ]

    y = 0.92
    for line in lines:
        fig.text(0.08, y, line, fontsize=8.5, family="monospace", va="top")
        y -= 0.026

    pdf.savefig(fig)
    plt.close(fig)

    # JSON sidecar metadata page
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("JSON sidecar metadata", fontsize=16, y=0.96)

    try:
        with open(json_qc_txt, "r") as f:
            txt = f.read()
    except Exception as e:
        txt = f"Could not read JSON QC text: {e}"

    txt = "\\n".join(txt.splitlines()[:90])
    fig.text(0.05, 0.93, txt, fontsize=7.5, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)

    # Geometry page
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("mri_info", fontsize=16, y=0.96)

    try:
        with open(info_txt, "r") as f:
            txt = f.read()
    except Exception as e:
        txt = f"Could not read mri_info text: {e}"

    # Keep page readable
    txt = "\\n".join(txt.splitlines()[:70])
    fig.text(0.05, 0.93, txt, fontsize=7, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)

    # One page per image
    for title, path in images:
        img, data = load_img(path)

        is_binary_mask = (
            "mask" in title.lower()
            and "brain" not in title.lower()
        )

        if is_binary_mask:
            vmin, vmax = 0, 1
            interp = "nearest"
        else:
            vmin, vmax = robust_limits(data)
            interp = "none"

        slices = get_slices(data)

        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        fig.suptitle(title + "\\n" + os.path.basename(path), fontsize=12)

        for ax, slc, plane in zip(axes, slices, ["Sagittal", "Coronal", "Axial"]):
            ax.imshow(slc, cmap="gray", vmin=vmin, vmax=vmax, interpolation=interp)
            ax.set_title(plane)
            ax.axis("off")

        fig.text(
            0.02, 0.02,
            f"Shape: {data.shape[:3]} | Voxel sizes: {img.header.get_zooms()[:3]}",
            fontsize=8,
            family="monospace",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.90])
        pdf.savefig(fig)
        plt.close(fig)

    # Mask overlay page
    try:
        ras_img, ras = load_img("${RAS_IMG}")
        mask_img, mask = load_img("${MASK_CLEAN_IMG}")
        vmin, vmax = robust_limits(ras)

        ras_slices = get_slices(ras)
        mask_slices = get_slices(mask)

        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        fig.suptitle("Hole-filled SynthStrip mask overlay on RAS image", fontsize=12)

        for ax, base_slc, mask_slc, plane in zip(
            axes,
            ras_slices,
            mask_slices,
            ["Sagittal", "Coronal", "Axial"],
        ):
            ax.imshow(base_slc, cmap="gray", vmin=vmin, vmax=vmax)
            ax.contour(mask_slc > 0, levels=[0.5], linewidths=0.8)
            ax.set_title(plane)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        pdf.savefig(fig)
        plt.close(fig)

    except Exception as e:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, f"Could not generate mask overlay: {e}", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

print(f"Wrote QC PDF: {qc_pdf}")
PYCODE

  if [[ ! -f "${QC_PDF}" ]]; then
    echo "ERROR: QC PDF was not created: ${QC_PDF}"
    return 1
  fi

  echo "Finished preprocessing for ${SUB} ${SES}!"
  echo "Final preprocessed image for DIPY registration lives here:"
  echo "${NORM_IMG}"
  echo "Make sure to check the QC report here:"
  echo "${QC_PDF}"
}

check_commands() {
  for cmd in docker N4BiasFieldCorrection ImageMath python; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "ERROR: Required command not found: ${cmd}"
      exit 1
    fi
  done

  python - <<PYCODE
try:
    import nibabel
    import scipy
except Exception as e:
    raise SystemExit(f"ERROR: Python packages nibabel and scipy are required but could not be imported: {e}")
PYCODE

  if [[ ! -f "${FS_LICENSE_HOST}" ]]; then
    echo "ERROR: FreeSurfer license not found: ${FS_LICENSE_HOST}"
    echo "Set FS_LICENSE_HOST near the top of this script."
    exit 1
  fi
}

run_subject_all_sessions() {
  local SUB="$1"
  local SUB_DIR="${RAW_ROOT}/${SUB}"

  if [[ ! -d "${SUB_DIR}" ]]; then
    echo "ERROR: Subject directory not found: ${SUB_DIR}"
    exit 1
  fi

  shopt -s nullglob
  local SESSIONS=("${SUB_DIR}"/ses-*)
  shopt -u nullglob

  if [[ "${#SESSIONS[@]}" -eq 0 ]]; then
    echo "ERROR: No session directories found for ${SUB} in ${SUB_DIR}"
    exit 1
  fi

  local failures=0

  for SES_DIR in "${SESSIONS[@]}"; do
    local SES
    SES="$(basename "${SES_DIR}")"

    if ! run_one "${SUB}" "${SES}"; then
      echo "FAILED: ${SUB} ${SES}"
      failures=$((failures + 1))
    fi
  done

  if [[ "${failures}" -gt 0 ]]; then
    echo "Completed ${SUB} with ${failures} failed session(s)."
    exit 1
  fi
}

run_batch_all_subjects_sessions() {
  if [[ ! -d "${RAW_ROOT}" ]]; then
    echo "ERROR: Raw root directory not found: ${RAW_ROOT}"
    exit 1
  fi

  shopt -s nullglob
  local SUBJECTS=("${RAW_ROOT}"/sub-*)
  shopt -u nullglob

  if [[ "${#SUBJECTS[@]}" -eq 0 ]]; then
    echo "ERROR: No subject directories found in ${RAW_ROOT}"
    exit 1
  fi

  local failures=0

  for SUB_DIR in "${SUBJECTS[@]}"; do
    local SUB
    SUB="$(basename "${SUB_DIR}")"

    shopt -s nullglob
    local SESSIONS=("${SUB_DIR}"/ses-*)
    shopt -u nullglob

    if [[ "${#SESSIONS[@]}" -eq 0 ]]; then
      echo "WARNING: No session directories found for ${SUB}; skipping."
      continue
    fi

    for SES_DIR in "${SESSIONS[@]}"; do
      local SES
      SES="$(basename "${SES_DIR}")"

      if ! run_one "${SUB}" "${SES}"; then
        echo "FAILED: ${SUB} ${SES}"
        failures=$((failures + 1))
      fi
    done
  done

  if [[ "${failures}" -gt 0 ]]; then
    echo "Batch completed with ${failures} failed subject/session(s)."
    exit 1
  fi

  echo "Batch completed successfully!"
}

# Main

check_commands

if [[ "$#" -eq 1 && "$1" == "--batch" ]]; then
  run_batch_all_subjects_sessions

elif [[ "$#" -eq 1 ]]; then
  SUB="$1"
  run_subject_all_sessions "${SUB}"

elif [[ "$#" -eq 2 ]]; then
  SUB="$1"
  SES="$2"
  run_one "${SUB}" "${SES}"

else
  echo "Usage:"
  echo "  bash $0 sub-XXX ses-V0X"
  echo "  bash $0 sub-XXX"
  echo "  bash $0 --batch"
  exit 1
fi