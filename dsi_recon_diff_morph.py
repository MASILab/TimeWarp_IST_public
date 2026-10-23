#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


RECON_MODE = "gqi"
GQI_METHOD_CODE = 4
GQI_PARAM0 = 1.25
GQI_TEMPLATE = None

BASE_DIR = Path("/nfs/masi/taylogc1")
PROJECT_DIR = BASE_DIR / "HBCD_diff_morph"
RECON_ROOT = PROJECT_DIR / "dsi_recon"
BIN_DIR = BASE_DIR / "bin"

DSI_STUDIO_BIN = str(BIN_DIR / "dsi_studio_singularity.sh")
SUBJECTS_JSON_OUT = RECON_ROOT / "subjects.json"

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf8") as fh:
        return json.load(fh)


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(data, fh, indent=2)



def subject_session_stem(subject: str, session: str) -> str:
    return f"{subject}_{session}"


def session_output_dir(subject: str, session: str) -> Path:
    return RECON_ROOT / subject / session



def infer_mask_from_dwi(dwi_path: Path) -> Path:
    return dwi_path.with_name("mask.nii.gz")


def run_cmd(cmd: List[str], log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with open(log_path, "a", encoding="utf8") as log:
        log.write(" ".join(str(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(str(x) for x in cmd)}")


def load_manifest(manifest_json: Path) -> List[Dict[str, Any]]:
    data = load_json(manifest_json)
    if not isinstance(data, list) or not data:
        raise ValueError("Manifest JSON must be a non-empty list.")

    required = {"subject", "session", "dwi", "bvec", "bval"}
    out: List[Dict[str, Any]] = []

    for i, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {i} is not an object.")
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Manifest row {i} missing required keys: {sorted(missing)}")

        subject = str(row["subject"]).strip()
        session = str(row["session"]).strip()

        dwi = Path(str(row["dwi"])).expanduser()
        bvec = Path(str(row["bvec"])).expanduser()
        bval = Path(str(row["bval"])).expanduser()

        for pth, label in [(dwi, "dwi"), (bvec, "bvec"), (bval, "bval")]:
            if not pth.exists():
                raise FileNotFoundError(f"Missing {label} file for {subject} {session}: {pth}")

        out.append({
            "subject": subject,
            "session": session,
            "dwi": dwi,
            "bvec": bvec,
            "bval": bval,
        })

    return out


def build_src_cmd(
    dsi_studio_bin: str,
    dwi: Path,
    bval: Path,
    bvec: Path,
    out_sz: Path,
) -> List[str]:
    return [
        dsi_studio_bin,
        "--action=src",
        f"--source={dwi}",
        f"--bval={bval}",
        f"--bvec={bvec}",
        f"--output={out_sz}",
        "--overwrite=1",
    ]


def build_rec_cmd(
    dsi_studio_bin: str,
    sz_path: Path,
    out_fz: Path,
) -> List[str]:
    return [
        dsi_studio_bin,
        "--action=rec",
        f"--source={sz_path}",
        f"--method={GQI_METHOD_CODE}",
        f"--param0={GQI_PARAM0}",
        f"--output={out_fz}",
        "--record_odf=1",
        "--overwrite=1",
    ]


def reconstruct_subject_session(
    row: Dict[str, Any],
    dsi_studio_bin: str,
    copy_mask: bool = True,
) -> Dict[str, Any]:
    subject = row["subject"]
    session = row["session"]
    dwi = Path(row["dwi"])
    bvec = Path(row["bvec"])
    bval = Path(row["bval"])

    for p in [dwi, bvec, bval]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    out_dir = session_output_dir(subject, session)
    ensure_dir(out_dir)

    stem = subject_session_stem(subject, session)
    sz_path = out_dir / f"{stem}.sz"
    fz_path = out_dir / f"{stem}.gqi.fz"

    mask_original = infer_mask_from_dwi(dwi)
    mask_copied = out_dir / "mask.nii.gz"
    log_file = out_dir / "recon.log"
    meta_file = out_dir / "recon_meta.json"

    if copy_mask:
        if not mask_original.exists():
            raise FileNotFoundError(f"Expected mask not found next to dwi: {mask_original}")
        shutil.copy2(mask_original, mask_copied)

    print("DEBUG manifest row:")
    print("  subject:", subject)
    print("  session:", session)
    print("  dwi:", dwi)
    print("  bval:", bval)
    print("  bvec:", bvec)

    if not sz_path.exists():
        run_cmd(
            build_src_cmd(
                dsi_studio_bin=dsi_studio_bin,
                dwi=dwi,
                bval=bval,
                bvec=bvec,
                out_sz=sz_path,
            ),
            log_path=log_file,
        )

    if not fz_path.exists():
        run_cmd(
            build_rec_cmd(
                dsi_studio_bin=dsi_studio_bin,
                sz_path=sz_path,
                out_fz=fz_path,
            ),
            log_path=log_file,
        )

    meta = {
        "subject": subject,
        "session": session,
        "subject_session_id": f"{subject}_{session}",
        "dwi": str(dwi),
        "bval": str(bval),
        "bvec": str(bvec),
        "mask_original": str(mask_original),
        "mask_copied": str(mask_copied),
        "mask_was_copied": bool(copy_mask),
        "sz": str(sz_path),
        "fz": str(fz_path),
        "recon_mode": RECON_MODE,
        "param0": GQI_PARAM0,
        "template": GQI_TEMPLATE,
    }
    save_json(meta, meta_file)
    return meta



def build_subjects_json(recon_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in recon_rows:
        mask_path = row["mask_copied"] if row.get("mask_was_copied", False) else row["mask_original"]
        out.append({
            "subject": row["subject"],
            "session": row["session"],
            "fib": row["fz"],
            "mask": mask_path,
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DSI Studio GQI reconstruction from a manifest and write per-session recon files under /nfs/masi/taylogc1/HBCD_diff_morph."
    )
    p.add_argument("manifest_json", type=Path, help="Input manifest JSON with subject/session/dwi/bvec/bval")
    p.add_argument("--dsi_studio_bin", default=DSI_STUDIO_BIN)
    p.add_argument("--copy_mask", action="store_true", help="Copy mask.nii.gz from the dwi directory into each session folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = load_manifest(args.manifest_json)

    recon_rows: List[Dict[str, Any]] = []
    for row in manifest_rows:
        recon_rows.append(
            reconstruct_subject_session(
                row=row,
                dsi_studio_bin=args.dsi_studio_bin,
                copy_mask=args.copy_mask,
            )
        )

    save_json(build_subjects_json(recon_rows), SUBJECTS_JSON_OUT)

    print(f"Wrote subjects.json: {SUBJECTS_JSON_OUT}")
    print(f"Per-session recon files, meta, and logs were written under {RECON_ROOT}/sub-*/ses-*/")


if __name__ == "__main__":
    main()