#!/usr/bin/env python3
"""Post-hoc de-meaning (HBCD_time_warp v5 math) + dual-set Figures 2-5 for the
manual 6-subject HBCD_diff_morph run. No pipelines are re-run: domain means are
recomputed from saved warps/masks, then subtracted from nodewise log-J columns."""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.spatial import cKDTree
from dipy.align import read_mapping

DEFAULT_T2W_GLOB = "**/*_logJ_node_summary.csv"
DEFAULT_TRACT_GLOB = "**/*_mask_syn_node_profile.csv"
DEFAULT_T2W_TRK_GLOB = "**/*_filtered_warped_to_*.trk"
DEFAULT_TRACT_TRK_GLOB = "**/*_mask_syn_filtered_warped_to_*.trk"
DEFAULT_BUNDLEWARP_SUMMARY_GLOB = "**/bundlewarp_registration_summary.csv"
DEFAULT_METRIC_CANDIDATES = [
    "mean_logJ",
    "mean_logj",
    "mean_log_j",
    "mean_mask_syn_logJ",
    "mean_mask_syn_logj",
    "logJ",
    "logj",
]
N_BINS = 10
BIN_LABELS = [f"{i * 10}-{(i + 1) * 10}%" for i in range(N_BINS)]

t2w_label = "Anatomical"
tract_label = "Tractography"
hemi_order = ["L", "R"]
heatmap_test = "ttest"
heatmap_use_fdr = True
heatmap_cmap = "coolwarm"
method_disagreement_test = "ttest"
method_use_fdr = True
method_disagreement_cmap = "viridis"
pipeline_effect_test = "ttest"
pipeline_effect_use_fdr = True
pipeline_effect_cmap = "coolwarm"
endpoint_core_pipeline_test = "ttest"
endpoint_core_pipeline_use_fdr = True
method_disagreement_max_points_per_bin = 25000
FDR_SIGNIFICANT_TEXT_COLOR = "black"
NON_FDR_TEXT_COLOR = "black"
bundlewarp_distribution_ncols = 3

T2W_IMAGE_DEMEANED_COL = "mean_t2w_image_demeaned_logJ"
TRACT_MASK_DEMEANED_COL = "mean_tract_mask_demeaned_logJ"
NON_DEMEANED_SET = "non_demeaned"
IMAGE_DEMEANED_SET = "image_demeaned"
SHARED_SET = "shared"
DEFAULT_YOUNGER_SESSION = "ses-V02"
DEFAULT_OLDER_SESSION = "ses-V03"
YOUNGER_T2W_FILE_TPL = "{subject}_{session}_T2w_RAS_synthstrip-brain_fillholesmask_N4_rescale0-1.nii.gz"
T2W_REG_SUBDIR = "registration"
T2W_MAPPING_FILE = "younger_to_older_syn_mapping.nii.gz"
T2W_PREALIGN_FILE = "prealign.txt"
T2W_WARPED_VERIFY_FILE = "younger_T2w_warped_to_older_T2w.nii.gz"
T2W_NODEWISE_GLOB = "morphometry/nodewise_logJ_summaries/*_logJ_node_summary.csv"
T2W_DEMEAN_META_FILE = "morphometry/t2w_image_demeaning.json"
TRACT_MASKS_SESSION_DIR_TPL = "generated_tract_masks/{session}"
MASK_SYN_DIR = "mask_syn"
MASK_MAPPING_FILE = "younger_to_older_mask_syn_mapping.nii.gz"
MASK_PREALIGN_FILE = "mask_prealign.txt"
MASK_WARPED_VERIFY_FILE = "younger_tract_mask_warped_to_older_mask.nii.gz"
MASK_NODE_PROFILE_SUFFIX = "_mask_syn_node_profile.csv"
TRACT_MASK_SUFFIX = "_mask.nii.gz"
TRACT_DEMEAN_META_FILE = "tract_mask_demeaning.csv"
WARP_VERIFY_MIN_CORR = 0.98


def _load_prealign_or_none(prealign_path):
    if not Path(prealign_path).is_file() or Path(prealign_path).stat().st_size == 0:
        return None
    mat = np.loadtxt(prealign_path)
    if mat.size != 16:
        return None
    return np.linalg.inv(mat)


def _load_session_mapping(mapping_path, younger_img_path, older_img_path, prealign_path):
    return read_mapping(
        str(mapping_path),
        nib.load(str(younger_img_path)),
        nib.load(str(older_img_path)),
        prealign=_load_prealign_or_none(prealign_path),
    )


def _warp_correlation(mapping, moving_path, saved_warped_path):
    moving = np.squeeze(np.asarray(nib.load(str(moving_path)).dataobj, dtype=np.float32))
    saved = np.squeeze(np.asarray(nib.load(str(saved_warped_path)).dataobj, dtype=np.float64))
    repro = np.asarray(mapping.transform(moving), dtype=np.float64)
    if repro.shape != saved.shape:
        return float("nan")
    a, b = repro.ravel(), saved.ravel()
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _phi_younger_to_older(mapping, points_rasmm):
    return mapping.transform_points_inverse(points_rasmm)


def _jacobian_matrix_at_points(mapping, points_rasmm, step_mm=0.5):
    points = np.asarray(points_rasmm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3, 3), dtype=np.float64)
    n = points.shape[0]
    J = np.full((n, 3, 3), np.nan, dtype=np.float64)
    basis = np.eye(3, dtype=np.float64)
    for axis in range(3):
        delta = step_mm * basis[axis]
        f_plus = _phi_younger_to_older(mapping, points + delta)
        f_minus = _phi_younger_to_older(mapping, points - delta)
        J[:, :, axis] = (f_plus - f_minus) / (2.0 * step_mm)
    return J


def _logdet_from_jacobian_matrices(J):
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 3 or J.shape[1:] != (3, 3):
        return np.full((0,), np.nan, dtype=np.float64)
    detJ = np.linalg.det(J)
    logJ = np.full(detJ.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(detJ) & (detJ > 0)
    logJ[valid] = np.log(detJ[valid])
    return logJ


def _jacobian_logdet_at_points(mapping, points_rasmm, step_mm=0.5):
    return _logdet_from_jacobian_matrices(_jacobian_matrix_at_points(mapping, points_rasmm, step_mm=step_mm))


def _compute_domain_logJ_summary(mapping, mask_data, affine, step_mm, chunk_size=250000):
    mask = np.squeeze(np.asarray(mask_data)) > 0
    if mask.ndim != 3:
        return {"n_voxels": 0, "n_finite_logJ_voxels": 0, "mean_logJ": np.nan}
    ijk = np.argwhere(mask)
    chunks = []
    for start in range(0, ijk.shape[0], chunk_size):
        chunk = ijk[start:start + chunk_size]
        points_rasmm = nib.affines.apply_affine(affine, chunk)
        logJ = _jacobian_logdet_at_points(mapping, points_rasmm, step_mm=step_mm)
        logJ = logJ[np.isfinite(logJ)]
        if logJ.size:
            chunks.append(logJ.astype(np.float64, copy=False))
    vals = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float64)
    return {
        "n_voxels": int(ijk.shape[0]),
        "n_finite_logJ_voxels": int(vals.size),
        "mean_logJ": _safe_nanmean(vals),
        "median_logJ": _safe_nanmedian(vals),
        "std_logJ": _safe_nanstd(vals),
    }


def _mean_minus_scalar(values, scalar):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(scalar):
        return arr - float(scalar)
    return np.full(arr.shape, np.nan)

def _finite_1d(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]

def _safe_nanmean(values):
    vals = _finite_1d(values)
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))

def _safe_nanmedian(values):
    vals = _finite_1d(values)
    if vals.size == 0:
        return np.nan
    return float(np.median(vals))

def _safe_nanstd(values):
    vals = _finite_1d(values)
    if vals.size == 0:
        return np.nan
    return float(np.std(vals))

def set_minimal_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "white",
        "axes.linewidth": 0.0,
        "axes.grid": False,
        "xtick.color": "0.25",
        "ytick.color": "0.25",
        "text.color": "0.15",
        "axes.labelcolor": "0.20",
        "axes.titlecolor": "0.15",
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 9,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })

def remove_axes_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

def minimal_colorbar(cbar):
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=7, colors="0.30")
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

def add_heatmap_cell_grid_colorbar(fig, ax, im, label, height_fraction=1.0):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    grid_bbox = im.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    cbar_height = grid_bbox.height * height_fraction
    cbar_y0 = grid_bbox.y0 + 0.5 * (grid_bbox.height - cbar_height)
    cbar_width = max(0.010, min(0.014, grid_bbox.width * 0.018))
    cbar_x0 = grid_bbox.x1 + 0.008
    cax = fig.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=8, color="0.25")
    minimal_colorbar(cbar)
    return cbar

def align_heatmap_side_label_to_cell_grid(ax, x=-0.075):
    ax.yaxis.set_label_coords(x, 0.5)

def normalize_subject_id(subject_id):
    if str(subject_id).startswith("sub-"):
        return str(subject_id)
    return f"sub-{subject_id}"

def strip_known_suffixes(path, suffixes):
    name = Path(path).name
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem

def strip_tract_filename_to_key(path):
    name = Path(path).name
    if name.endswith(".trk"):
        name = name[:-4]
    patterns = [
        r"_mask_syn_filtered_warped_to_[A-Za-z0-9\-]+$",
        r"_mask_syn_unfiltered_warped_to_[A-Za-z0-9\-]+$",
        r"_mask_syn_warped_to_[A-Za-z0-9\-]+$",
        r"_filtered_similarity_warped_to_[A-Za-z0-9\-]+$",
        r"_filtered_warped_to_[A-Za-z0-9\-]+$",
        r"_unfiltered_warped_to_[A-Za-z0-9\-]+$",
        r"_warped_to_[A-Za-z0-9\-]+$",
        r"_bundlewarp_nonlinear_moved_filtered$",
        r"_bundlewarp_linear_moved_filtered$",
        r"_bundlewarp_nonlinear_moved_unfiltered$",
        r"_bundlewarp_linear_moved_unfiltered$",
        r"_bundlewarp_nonlinear_moved$",
        r"_bundlewarp_linear_moved$",
    ]
    for pattern in patterns:
        name = re.sub(pattern, "", name)
    return name

def parse_tract_key(tract_key):
    tokens = tract_key.split("_")
    hemi_tokens = {"L", "R", "all"}
    for i, token in enumerate(tokens):
        if token in hemi_tokens and i > 0:
            family = "_".join(tokens[:i])
            hemi = token
            label = f"{family}_{hemi}"
            return family, hemi, label
    if len(tokens) >= 2 and tokens[-1] in hemi_tokens:
        family = "_".join(tokens[:-1])
        hemi = tokens[-1]
        label = f"{family}_{hemi}"
        return family, hemi, label
    return tract_key, "all", tract_key

def find_subject_dirs(root, subjects=None):
    root = Path(root)
    if subjects:
        dirs = []
        for subject in subjects:
            subject = normalize_subject_id(subject)
            d = root / subject
            if d.exists():
                dirs.append(d)
            else:
                print(f"WARNING: subject directory not found: {d}")
        return dirs
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")])

def choose_metric_column(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested metric column {requested} not found. Columns: {list(df.columns)}")
        return requested
    for col in DEFAULT_METRIC_CANDIDATES:
        if col in df.columns:
            return col
    numeric_cols = [c for c in df.columns if c != "node" and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(
        "Could not infer nodewise metric column. "
        f"Expected one of {DEFAULT_METRIC_CANDIDATES}; found columns {list(df.columns)}"
    )

def read_nodewise_csv(csv_path, subject, pipeline, metric_col=None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    col = choose_metric_column(df, metric_col)
    if "node" not in df.columns:
        df = df.copy()
        df["node"] = np.arange(len(df))
    tract_key = strip_known_suffixes(
        csv_path.name,
        suffixes=[
            "_logJ_node_summary.csv",
            "_mask_syn_node_profile.csv",
            "_node_summary.csv",
            ".csv",
        ],
    )
    family, hemi, tract_label = parse_tract_key(tract_key)
    out = pd.DataFrame({
        "pipeline": pipeline,
        "subject": subject,
        "tract_key": tract_key,
        "family": family,
        "hemi": hemi,
        "tract_label": tract_label,
        "node": pd.to_numeric(df["node"], errors="coerce").astype("Int64"),
        "logJ": pd.to_numeric(df[col], errors="coerce"),
        "source_csv": str(csv_path),
    })
    out = out.dropna(subset=["node", "logJ"])
    out["node"] = out["node"].astype(int)
    return out

def load_pipeline_data(root, pipeline, glob_pattern, metric_col=None, subjects=None):
    rows = []
    for subject_dir in find_subject_dirs(root, subjects=subjects):
        subject = subject_dir.name
        files = sorted(subject_dir.glob(glob_pattern))
        if not files:
            continue
        for f in files:
            try:
                rows.append(read_nodewise_csv(f, subject, pipeline, metric_col=metric_col))
            except Exception as exc:
                print(f"WARNING: skipping {f}: {exc}")
    if not rows:
        raise RuntimeError(f"No nodewise data loaded for pipeline {pipeline} from {root}")
    return pd.concat(rows, ignore_index=True)

def load_pipeline_trk_map(root, pipeline, glob_pattern, subjects=None):
    rows = []
    for subject_dir in find_subject_dirs(root, subjects=subjects):
        subject = subject_dir.name
        files = sorted(subject_dir.glob(glob_pattern))
        if not files:
            continue
        for f in files:
            tract_key = strip_tract_filename_to_key(f)
            family, hemi, tract_label = parse_tract_key(tract_key)
            rows.append({
                "pipeline": pipeline,
                "subject": subject,
                "tract_key": tract_key,
                "family": family,
                "hemi": hemi,
                "tract_label": tract_label,
                "trk_path": str(f),
            })
    if not rows:
        return pd.DataFrame(columns=["pipeline", "subject", "tract_key", "family", "hemi", "tract_label", "trk_path"])
    df = pd.DataFrame(rows)
    duplicates = df.duplicated(subset=["pipeline", "subject", "tract_label"], keep=False)
    if duplicates.any():
        dup = df.loc[duplicates, ["pipeline", "subject", "tract_label", "trk_path"]]
        print("WARNING: duplicate .trk matches found; using the first per subject/tract_label:")
        print(dup.to_string(index=False))
    return df.drop_duplicates(subset=["pipeline", "subject", "tract_label"], keep="first")

def ordered_unique(values, requested_order=None):
    vals = list(dict.fromkeys([v for v in values if pd.notna(v)]))
    if requested_order:
        requested = [x for x in requested_order if x in vals]
        rest = sorted([x for x in vals if x not in requested])
        return requested + rest
    return sorted(vals)

def get_symmetric_norm(values, vlim=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if vlim is None:
        if values.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.nanmax(np.abs(values)))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
    else:
        vmax = float(abs(vlim))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), vmax

def _arc_length_fraction(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if points.shape[0] == 1:
        return np.zeros((1,), dtype=float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = dist[-1]
    if not np.isfinite(total) or total <= 0:
        return np.zeros((points.shape[0],), dtype=float)
    return dist / total

def _interpolate_native_at_fraction(arr_native, fractions, target_fraction):
    target_fraction = float(np.clip(target_fraction, 0.0, 1.0))
    out = np.empty((3,), dtype=np.float32)
    for dim in range(3):
        out[dim] = np.interp(target_fraction, fractions, arr_native[:, dim])
    return out

def _streamline_segment_by_fraction(arr_native, arr_rasmm, start_fraction, end_fraction):
    if arr_native.shape[0] < 2:
        return None
    fractions = _arc_length_fraction(arr_rasmm)
    if fractions.shape[0] != arr_native.shape[0] or fractions[-1] <= fractions[0]:
        return None
    start_fraction = float(np.clip(start_fraction, 0.0, 1.0))
    end_fraction = float(np.clip(end_fraction, 0.0, 1.0))
    if end_fraction <= start_fraction:
        return None
    inner = np.where((fractions > start_fraction) & (fractions < end_fraction))[0]
    pts = [_interpolate_native_at_fraction(arr_native, fractions, start_fraction)]
    for idx in inner:
        pts.append(arr_native[idx].astype(np.float32))
    pts.append(_interpolate_native_at_fraction(arr_native, fractions, end_fraction))
    seg = np.asarray(pts, dtype=np.float32)
    if seg.shape[0] < 2:
        return None
    if np.nanmax(np.linalg.norm(np.diff(seg, axis=0), axis=1)) <= 0:
        return None
    return seg

def bin_subject_tract_nodes(df, n_bins=N_BINS):
    rows = []
    for keys, g in df.groupby(["pipeline", "subject", "tract_label", "family", "hemi"], dropna=False):
        pipeline, subject, tract_label, family, hemi = keys
        g = g.sort_values("node")
        vals = g["logJ"].to_numpy(dtype=float)
        nodes = g["node"].to_numpy(dtype=int)
        idx_bins = np.array_split(np.arange(len(vals)), n_bins)
        for b, idx in enumerate(idx_bins):
            if len(idx) == 0:
                continue
            rows.append({
                "pipeline": pipeline,
                "subject": subject,
                "tract_label": tract_label,
                "family": family,
                "hemi": hemi,
                "bin": b + 1,
                "bin_label": BIN_LABELS[b] if b < len(BIN_LABELS) else f"bin{b+1}",
                "node_start": int(nodes[idx[0]]),
                "node_end": int(nodes[idx[-1]]),
                "mean_logJ": float(np.nanmean(vals[idx])),
            })
    return pd.DataFrame(rows)

def bin_nodewise_values(df, n_bins=N_BINS):
    rows = []
    for keys, g in df.groupby(["pipeline", "subject", "tract_label", "family", "hemi"], dropna=False):
        pipeline, subject, tract_label, family, hemi = keys
        g = g.sort_values("node").reset_index(drop=True)
        vals = g["logJ"].to_numpy(dtype=float)
        nodes = g["node"].to_numpy(dtype=int)
        idx_bins = np.array_split(np.arange(len(vals)), n_bins)
        for b, idx in enumerate(idx_bins):
            if len(idx) == 0:
                continue
            for node_within_bin, row_idx in enumerate(idx):
                rows.append({
                    "pipeline": pipeline,
                    "subject": subject,
                    "tract_label": tract_label,
                    "family": family,
                    "hemi": hemi,
                    "bin": b + 1,
                    "bin_label": BIN_LABELS[b] if b < len(BIN_LABELS) else f"bin{b+1}",
                    "node": int(nodes[row_idx]),
                    "node_within_bin": int(node_within_bin),
                    "logJ": float(vals[row_idx]),
                })
    return pd.DataFrame(rows)

def make_tract_column_order(labels, family_order=None, hemi_order=None):
    hemi_order = hemi_order or ["L", "R", "all"]
    parsed = []
    for label in labels:
        family, hemi, _ = parse_tract_key(label)
        parsed.append((label, family, hemi))
    families = ordered_unique([x[1] for x in parsed], family_order)
    out = []
    for fam in families:
        fam_labels = [x for x in parsed if x[1] == fam]
        for hemi in hemi_order:
            for label, _, h in sorted(fam_labels):
                if h == hemi and label not in out:
                    out.append(label)
        for label, _, h in sorted(fam_labels):
            if label not in out:
                out.append(label)
    return out

def get_group_binned_heatmap_values(binned):
    if binned.empty:
        return np.asarray([], dtype=float)
    group = (
        binned
        .groupby(["pipeline", "tract_label", "family", "hemi", "bin", "bin_label"], dropna=False)["mean_logJ"]
        .mean()
        .reset_index()
    )
    return group["mean_logJ"].to_numpy(dtype=float)

def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(pvals.shape, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not np.any(finite):
        return qvals
    p = pvals[finite]
    order = np.argsort(p)
    ranked = p[order]
    m = float(len(ranked))
    q_ranked = ranked * m / (np.arange(len(ranked)) + 1.0)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    qvals[finite] = q
    return qvals

def p_to_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.05:
        return "*"
    return ""


def _test_p_and_q_columns(test):
    if test == "wilcoxon":
        return "wilcoxon_p", "wilcoxon_q_fdr"
    return "ttest_p", "ttest_q_fdr"


def significance_symbol_from_row(inference_row, test="ttest"):
    if inference_row is None or len(inference_row) == 0:
        return ""
    p_col, q_col = _test_p_and_q_columns(test)
    raw_p = float(inference_row[p_col].iloc[0]) if p_col in inference_row.columns else np.nan
    fdr_q = float(inference_row[q_col].iloc[0]) if q_col in inference_row.columns else np.nan
    if np.isfinite(fdr_q) and fdr_q < 0.05:
        return "**"
    if np.isfinite(raw_p) and raw_p < 0.05:
        return "*"
    return ""

def heatmap_symbol_header_text():
    return "× = non-significant; * = nominal p < .05; ** = qFDR < .05"

def profile_symbol_header_text():
    return "* = nominal p < .05; ** = qFDR < .05"

def _draw_nonsignificant_cell_cross(ax, row_index, col_index, startoff=0.25, endoff=0.75):
    left = col_index - 0.5
    top = row_index - 0.5
    x_min = left + startoff
    x_max = left + endoff
    y_min = top + startoff
    y_max = top + endoff
    ax.plot([x_min, x_max], [y_min, y_max], color="black", linewidth=1.0, alpha=0.5)
    ax.plot([x_min, x_max], [y_max, y_min], color="black", linewidth=1.0, alpha=0.5)

def _cell_label_from_raw_and_fdr(inference_row, test="ttest"):
    if inference_row is None or len(inference_row) == 0:
        return "", True
    symbol = significance_symbol_from_row(inference_row, test=test)
    if symbol:
        return symbol, False
    return "", True

def _compute_heatmap_inference_one_pipeline(binned_one_pipeline, pipeline_name, test="ttest"):
    rows = []
    group_cols = ["tract_label", "family", "hemi", "bin", "bin_label"]
    for keys, g in binned_one_pipeline.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(g["mean_logJ"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        median = float(np.median(vals)) if n else np.nan
        t_stat = np.nan
        p_ttest = np.nan
        w_stat = np.nan
        p_wilcoxon = np.nan
        if n >= 2:
            try:
                res = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
                t_stat = float(res.statistic)
                p_ttest = float(res.pvalue)
            except Exception:
                pass
        if n >= 3 and not np.allclose(vals, 0.0):
            try:
                res = stats.wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
                w_stat = float(res.statistic)
                p_wilcoxon = float(res.pvalue)
            except Exception:
                pass
        row = {"pipeline": pipeline_name}
        row.update(dict(zip(group_cols, keys)))
        row.update({
            "n_subjects": n,
            "mean_logJ": mean,
            "median_logJ": median,
            "std_logJ": sd,
            "ttest_stat": t_stat,
            "ttest_p": p_ttest,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_wilcoxon,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["ttest_q_fdr"] = benjamini_hochberg(out["ttest_p"].to_numpy(dtype=float))
    out["wilcoxon_q_fdr"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy(dtype=float))
    out["fdr_scope"] = str(pipeline_name)
    out["n_tests_in_ttest_fdr_scope"] = int(np.sum(np.isfinite(out["ttest_p"].to_numpy(dtype=float))))
    out["n_tests_in_wilcoxon_fdr_scope"] = int(np.sum(np.isfinite(out["wilcoxon_p"].to_numpy(dtype=float))))
    return out

def compute_heatmap_inference(binned, test="ttest"):
    frames = []
    for pipeline_name, sub in binned.groupby("pipeline", dropna=False):
        frames.append(_compute_heatmap_inference_one_pipeline(sub.copy(), pipeline_name, test=test))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _heatmap_p_column(test, use_fdr=True):
    if test == "wilcoxon":
        return "wilcoxon_q_fdr" if use_fdr else "wilcoxon_p"
    return "ttest_q_fdr" if use_fdr else "ttest_p"

def plot_group_heatmap(binned, pipeline_name, out_file, cmap_name="coolwarm", vlim=None, column_order=None, family_order=None, hemi_order=None, title=None, inference=None, heatmap_test="ttest", use_fdr=True):
    sub = binned[binned["pipeline"] == pipeline_name].copy()
    if sub.empty:
        raise RuntimeError(f"No binned data for pipeline {pipeline_name}")
    group = sub.groupby(["tract_label", "family", "hemi", "bin", "bin_label"], dropna=False)["mean_logJ"].mean().reset_index()
    inf = pd.DataFrame()
    if inference is not None and not inference.empty:
        inf = inference[inference["pipeline"] == pipeline_name].copy()
    cols = column_order or make_tract_column_order(group["tract_label"].unique(), family_order=family_order, hemi_order=hemi_order)
    rows = list(range(1, len(BIN_LABELS) + 1))
    mat = np.full((len(rows), len(cols)), np.nan, dtype=float)
    cell_text = [["" for _ in cols] for _ in rows]
    cell_draw_cross = [[False for _ in cols] for _ in rows]
    for i, b in enumerate(rows):
        for j, label in enumerate(cols):
            val = group[(group["tract_label"] == label) & (group["bin"] == b)]["mean_logJ"]
            if len(val):
                mat[i, j] = float(val.iloc[0])
                star = ""
                draw_cross = False
                if not inf.empty:
                    row = inf[(inf["tract_label"] == label) & (inf["bin"] == b)]
                    star, draw_cross = _cell_label_from_raw_and_fdr(row, test=heatmap_test)
                if draw_cross:
                    cell_draw_cross[i][j] = True
                    cell_text[i][j] = ""
                else:
                    cell_text[i][j] = f"{mat[i, j]:.3f}{star}"
    norm, _ = get_symmetric_norm(mat.ravel(), vlim=vlim)
    fig_w = max(8, 0.55 * len(cols) + 2.5)
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap=cmap_name, norm=norm, aspect="auto", interpolation="nearest")
    remove_axes_box(ax)
    ax.set_title(title or f"Group mean log det $J$ by tract decile: {pipeline_name} warps", fontsize=11, pad=18, loc="left")
    ax.text(0.0, 1.02, heatmap_symbol_header_text(), transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.25", clip_on=False)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(BIN_LABELS, fontsize=8)
    ax.set_ylabel("tract decile", fontsize=9)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([])
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)

    for j, label in enumerate(cols):
        hemi = parse_tract_key(label)[1]
        ax.text(j, -0.62, hemi, ha="center", va="center", fontsize=8, clip_on=False)

    start = 0
    while start < len(cols):
        fam = parse_tract_key(cols[start])[0]
        end = start + 1
        while end < len(cols) and parse_tract_key(cols[end])[0] == fam:
            end += 1
        center = 0.5 * (start + end - 1)
        ax.text(center, -1.25, fam, ha="center", va="center", fontsize=8, fontweight="bold", clip_on=False)
        if end - 1 > start:
            ax.plot([start - 0.45, end - 1 + 0.45], [-0.92, -0.92], color="0.60", linewidth=0.5, clip_on=False)
        start = end

    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                if cell_draw_cross[i][j]:
                    _draw_nonsignificant_cell_cross(ax, i, j)
                elif cell_text[i][j]:
                    ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=6, color="black", fontweight="normal")
    align_heatmap_side_label_to_cell_grid(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    cbar = add_heatmap_cell_grid_colorbar(fig, ax, im, "mean binned log det $J$")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file

def _trk_bin_point_clouds(in_trk, n_bins=N_BINS, max_points_per_bin=25000):
    trk_file = nib.streamlines.load(str(in_trk))
    tractogram = trk_file.tractogram
    affine = tractogram.affine_to_rasmm
    if affine is None:
        affine = np.eye(4)
    bins = [[] for _ in range(n_bins)]
    for sl in tractogram.streamlines:
        arr_native = np.asarray(sl, dtype=np.float32)
        if arr_native.ndim != 2 or arr_native.shape[1] != 3 or arr_native.shape[0] < 2:
            continue
        arr_rasmm = nib.affines.apply_affine(affine, arr_native)
        for b in range(n_bins):
            seg_native = _streamline_segment_by_fraction(
                arr_native,
                arr_rasmm,
                b / n_bins,
                (b + 1) / n_bins,
            )
            if seg_native is None:
                continue
            seg_rasmm = nib.affines.apply_affine(affine, seg_native)
            bins[b].append(seg_rasmm)
    out = []
    rng = np.random.default_rng(12345)
    for chunks in bins:
        if not chunks:
            out.append(np.empty((0, 3), dtype=np.float32))
            continue
        pts = np.vstack(chunks).astype(np.float32)
        if pts.shape[0] > max_points_per_bin:
            idx = rng.choice(pts.shape[0], size=max_points_per_bin, replace=False)
            pts = pts[np.sort(idx)]
        out.append(pts)
    return out

def _symmetric_nn_distance(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    tree_b = cKDTree(B)
    d_ab, _ = tree_b.query(A, k=1)
    tree_a = cKDTree(A)
    d_ba, _ = tree_a.query(B, k=1)
    return float(0.5 * (np.nanmean(d_ab) + np.nanmean(d_ba)))

def _centroid_distance(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    return float(np.linalg.norm(np.nanmean(A, axis=0) - np.nanmean(B, axis=0)))

def compute_method_disagreement_by_bin(t2w_trks, tract_trks, n_bins=N_BINS, max_points_per_bin=25000):
    if t2w_trks.empty or tract_trks.empty:
        return pd.DataFrame()
    left = t2w_trks.rename(columns={"trk_path": "t2w_trk_path"})
    right = tract_trks.rename(columns={"trk_path": "tract_trk_path"})
    merged = left.merge(
        right[["subject", "tract_label", "tract_trk_path"]],
        on=["subject", "tract_label"],
        how="inner",
    )
    rows = []
    cache = {}
    for row in merged.to_dict("records"):
        key_t2w = row["t2w_trk_path"]
        key_tract = row["tract_trk_path"]
        try:
            if key_t2w not in cache:
                cache[key_t2w] = _trk_bin_point_clouds(key_t2w, n_bins=n_bins, max_points_per_bin=max_points_per_bin)
            if key_tract not in cache:
                cache[key_tract] = _trk_bin_point_clouds(key_tract, n_bins=n_bins, max_points_per_bin=max_points_per_bin)
            t2w_bins = cache[key_t2w]
            tract_bins = cache[key_tract]
        except Exception as exc:
            print(f"WARNING: failed loading tract pair for method disagreement: {key_t2w} / {key_tract}: {exc}")
            continue
        for b in range(n_bins):
            A = t2w_bins[b]
            B = tract_bins[b]
            rows.append({
                "subject": row["subject"],
                "tract_label": row["tract_label"],
                "family": row["family"],
                "hemi": row["hemi"],
                "bin": b + 1,
                "bin_label": BIN_LABELS[b] if b < len(BIN_LABELS) else f"bin{b+1}",
                "t2w_trk_path": key_t2w,
                "tract_trk_path": key_tract,
                "n_points_t2w": int(A.shape[0]),
                "n_points_tractography": int(B.shape[0]),
                "centroid_distance_mm": _centroid_distance(A, B),
                "symmetric_nn_distance_mm": _symmetric_nn_distance(A, B),
            })
    return pd.DataFrame(rows)

def compute_method_disagreement_inference(disagreement, metric="symmetric_nn_distance_mm", test="ttest"):
    rows = []
    if disagreement.empty or metric not in disagreement.columns:
        return pd.DataFrame()
    group_cols = ["tract_label", "family", "hemi", "bin", "bin_label"]
    for keys, g in disagreement.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        mean = float(np.mean(vals)) if n else np.nan
        median = float(np.median(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        t_stat = np.nan
        p_ttest = np.nan
        w_stat = np.nan
        p_wilcoxon = np.nan
        if n >= 2:
            try:
                res = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
                t_stat = float(res.statistic)
                p_ttest = float(res.pvalue)
            except Exception:
                pass
        if n >= 3 and not np.allclose(vals, 0.0):
            try:
                res = stats.wilcoxon(vals, zero_method="wilcox", alternative="greater")
                w_stat = float(res.statistic)
                p_wilcoxon = float(res.pvalue)
            except Exception:
                pass
        row = dict(zip(group_cols, keys))
        row.update({
            "metric": metric,
            "n_subjects": n,
            "mean": mean,
            "median": median,
            "std": sd,
            "ttest_stat": t_stat,
            "ttest_p": p_ttest,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_wilcoxon,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_q_fdr"] = benjamini_hochberg(out["ttest_p"].to_numpy(dtype=float))
    out["wilcoxon_q_fdr"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy(dtype=float))
    return out

def compute_endpoint_core_disagreement_tests(disagreement, metric="symmetric_nn_distance_mm", test="ttest", endpoint_bins=(1, 2, 9, 10), core_bins=(5, 6)):
    rows = []
    if disagreement.empty or metric not in disagreement.columns:
        return pd.DataFrame()
    for keys, g in disagreement.groupby(["tract_label", "family", "hemi"], dropna=False):
        tract_label, family, hemi = keys
        subj_rows = []
        for subject, sg in g.groupby("subject"):
            endpoint_vals = pd.to_numeric(sg.loc[sg["bin"].isin(endpoint_bins), metric], errors="coerce").to_numpy(dtype=float)
            core_vals = pd.to_numeric(sg.loc[sg["bin"].isin(core_bins), metric], errors="coerce").to_numpy(dtype=float)
            endpoint_vals = endpoint_vals[np.isfinite(endpoint_vals)]
            core_vals = core_vals[np.isfinite(core_vals)]
            if endpoint_vals.size == 0 or core_vals.size == 0:
                continue
            subj_rows.append({
                "subject": subject,
                "endpoint_mean": float(np.mean(endpoint_vals)),
                "core_mean": float(np.mean(core_vals)),
                "endpoint_minus_core": float(np.mean(endpoint_vals) - np.mean(core_vals)),
            })
        comp = pd.DataFrame(subj_rows)
        vals = comp["endpoint_minus_core"].to_numpy(dtype=float) if not comp.empty else np.asarray([], dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        t_stat = np.nan
        p_ttest = np.nan
        w_stat = np.nan
        p_wilcoxon = np.nan
        if n >= 2:
            try:
                res = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
                t_stat = float(res.statistic)
                p_ttest = float(res.pvalue)
            except Exception:
                pass
        if n >= 3 and not np.allclose(vals, 0.0):
            try:
                res = stats.wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
                w_stat = float(res.statistic)
                p_wilcoxon = float(res.pvalue)
            except Exception:
                pass
        rows.append({
            "tract_label": tract_label,
            "family": family,
            "hemi": hemi,
            "metric": metric,
            "endpoint_bins": ",".join(str(x) for x in endpoint_bins),
            "core_bins": ",".join(str(x) for x in core_bins),
            "n_subjects": n,
            "mean_endpoint_minus_core": float(np.mean(vals)) if n else np.nan,
            "median_endpoint_minus_core": float(np.median(vals)) if n else np.nan,
            "std_endpoint_minus_core": float(np.std(vals, ddof=1)) if n > 1 else np.nan,
            "ttest_stat": t_stat,
            "ttest_p": p_ttest,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_wilcoxon,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_q_fdr"] = benjamini_hochberg(out["ttest_p"].to_numpy(dtype=float))
    out["wilcoxon_q_fdr"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy(dtype=float))
    return out

def _method_test_p_column(test, use_fdr=True):
    if test == "wilcoxon":
        return "wilcoxon_q_fdr" if use_fdr else "wilcoxon_p"
    return "ttest_q_fdr" if use_fdr else "ttest_p"

def plot_method_disagreement_heatmap(disagreement, out_file, metric="symmetric_nn_distance_mm", column_order=None, family_order=None, hemi_order=None, cmap_name="viridis", inference=None, test="ttest", use_fdr=True, title=None):
    group = disagreement.groupby(["tract_label", "family", "hemi", "bin", "bin_label"], dropna=False)[metric].mean().reset_index()
    cols = column_order or make_tract_column_order(group["tract_label"].unique(), family_order=family_order, hemi_order=hemi_order)
    rows = list(range(1, len(BIN_LABELS) + 1))
    mat = np.full((len(rows), len(cols)), np.nan, dtype=float)
    txt = [["" for _ in cols] for _ in rows]
    txt_draw_cross = [[False for _ in cols] for _ in rows]
    for i, b in enumerate(rows):
        for j, label in enumerate(cols):
            vals = group[(group["tract_label"] == label) & (group["bin"] == b)][metric]
            if len(vals):
                mat[i, j] = float(vals.iloc[0])
                star = ""
                draw_cross = False
                if inference is not None and not inference.empty:
                    row = inference[(inference["tract_label"] == label) & (inference["bin"] == b)]
                    star, draw_cross = _cell_label_from_raw_and_fdr(row, test=test)
                if draw_cross:
                    txt_draw_cross[i][j] = True
                    txt[i][j] = ""
                else:
                    txt[i][j] = f"{mat[i, j]:.2f}{star}"
    finite = mat[np.isfinite(mat)]
    vmax = float(np.nanpercentile(finite, 95)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig_w = max(8, 0.55 * len(cols) + 2.5)
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap=cmap_name, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    remove_axes_box(ax)
    ax.set_title(title or "Method disagreement by tract decile: Anatomical vs Tractography warps", fontsize=11, pad=18, loc="left")
    ax.text(0.0, 1.02, heatmap_symbol_header_text(), transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.25", clip_on=False)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(BIN_LABELS, fontsize=8)
    ax.set_ylabel("tract decile", fontsize=9)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([])
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    for j, label in enumerate(cols):
        hemi = parse_tract_key(label)[1]
        ax.text(j, -0.62, hemi, ha="center", va="center", fontsize=8, clip_on=False)
    start = 0
    while start < len(cols):
        fam = parse_tract_key(cols[start])[0]
        end = start + 1
        while end < len(cols) and parse_tract_key(cols[end])[0] == fam:
            end += 1
        center = 0.5 * (start + end - 1)
        ax.text(center, -1.25, fam, ha="center", va="center", fontsize=8, fontweight="bold", clip_on=False)
        if end - 1 > start:
            ax.plot([start - 0.45, end - 1 + 0.45], [-0.92, -0.92], color="0.60", linewidth=0.5, clip_on=False)
        start = end
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                if txt_draw_cross[i][j]:
                    _draw_nonsignificant_cell_cross(ax, i, j)
                elif txt[i][j]:
                    ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=6, color="black", fontweight="normal")
    align_heatmap_side_label_to_cell_grid(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    cbar = add_heatmap_cell_grid_colorbar(fig, ax, im, "mean symmetric NN distance (mm)")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file

def plot_method_disagreement_profile_grid(disagreement, out_file, metric="symmetric_nn_distance_mm", family_order=None, hemi_order=None, endpoint_core_tests=None, test="ttest", use_fdr=True):
    families = ordered_unique(disagreement["family"].unique(), family_order)
    hemis = ordered_unique(disagreement["hemi"].unique(), hemi_order or ["L", "R", "all"])
    n_rows = max(1, len(families))
    n_cols = max(1, len(hemis))
    fig_w = max(6, 3.2 * n_cols)
    fig_h = max(4, 2.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    x = np.arange(1, N_BINS + 1)
    p_col = _method_test_p_column(test, use_fdr=use_fdr)
    for r, family in enumerate(families):
        for c, hemi in enumerate(hemis):
            ax = axes[r, c]
            remove_axes_box(ax)
            sub = disagreement[(disagreement["family"] == family) & (disagreement["hemi"] == hemi)]
            if sub.empty:
                ax.axis("off")
                continue
            grp = sub.groupby("bin")[metric]
            means = grp.mean().reindex(x).to_numpy(dtype=float)
            medians = grp.median().reindex(x).to_numpy(dtype=float)
            counts = grp.count().reindex(x).to_numpy(dtype=float)
            stds = grp.std().reindex(x).to_numpy(dtype=float)
            sem = stds / np.sqrt(np.maximum(counts, 1))
            ax.plot(x, means, marker="o", linewidth=1.25, markersize=3, color="0.18", label="mean")
            ax.plot(x, medians, marker="s", linewidth=1.0, markersize=2.5, linestyle="--", color="0.48", label="median")
            ax.fill_between(x, means - sem, means + sem, color="0.70", alpha=0.28, linewidth=0)
            ax.axvspan(0.5, 2.5, color="0.94", zorder=0)
            ax.axvspan(8.5, 10.5, color="0.94", zorder=0)
            ax.axvspan(4.5, 6.5, color="0.985", zorder=0)
            ax.grid(True, axis="y", color="0.92", linewidth=0.5)
            ax.set_axisbelow(True)
            star = ""
            diff_txt = ""
            if endpoint_core_tests is not None and not endpoint_core_tests.empty:
                row = endpoint_core_tests[(endpoint_core_tests["family"] == family) & (endpoint_core_tests["hemi"] == hemi)]
                if len(row):
                    star = significance_symbol_from_row(row, test=test)
                    diff = row["mean_endpoint_minus_core"].iloc[0]
                    if np.isfinite(diff):
                        diff_txt = f" endpoint-core={diff:.2f} mm{star}"
            ax.set_title(f"{family}_{hemi}{diff_txt}", fontsize=8, pad=6, loc="left")
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in x], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            if r == n_rows - 1:
                ax.set_xlabel("decile bin", fontsize=8)
            if c == 0:
                ax.set_ylabel("symmetric NN distance (mm)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=8, ncol=len(handles), bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Method-disagreement profiles: Anatomical vs Tractography warps", fontsize=12, x=0.01, ha="left", y=0.985)
    fig.text(0.01, 0.962, profile_symbol_header_text(), ha="left", va="center", fontsize=7, color="0.25")
    fig.tight_layout(rect=[0, 0.065, 1, 0.955])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file

def compute_pipeline_effect_difference_tests(binned, pipeline_a="Anatomical", pipeline_b="Tractography", test="ttest"):
    if binned.empty:
        return pd.DataFrame()
    rows = []
    grouped = binned.groupby(["family", "hemi", "tract_label", "bin", "bin_label"], dropna=False)
    for keys, g in grouped:
        family, hemi, tract_label, bin_idx, bin_label = keys
        a = g[g["pipeline"] == pipeline_a][["subject", "mean_logJ"]].rename(columns={"mean_logJ": "a"})
        b = g[g["pipeline"] == pipeline_b][["subject", "mean_logJ"]].rename(columns={"mean_logJ": "b"})
        merged = pd.merge(a, b, on="subject", how="inner")
        avals = pd.to_numeric(merged["a"], errors="coerce").to_numpy(dtype=float)
        bvals = pd.to_numeric(merged["b"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(avals) & np.isfinite(bvals)
        avals = avals[valid]
        bvals = bvals[valid]
        diffs = avals - bvals
        if diffs.size >= 2 and np.nanstd(diffs) > 0:
            try:
                t_stat, t_p = stats.ttest_rel(avals, bvals, nan_policy="omit")
            except Exception:
                t_stat, t_p = np.nan, np.nan
        else:
            t_stat, t_p = np.nan, np.nan
        if diffs.size >= 1 and np.any(np.abs(diffs) > 0):
            try:
                w_stat, w_p = stats.wilcoxon(diffs)
            except Exception:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan
        rows.append({
            "family": family,
            "hemi": hemi,
            "tract_label": tract_label,
            "bin": int(bin_idx),
            "bin_label": bin_label,
            "n_subjects": int(diffs.size),
            f"mean_{pipeline_a}": _safe_nanmean(avals),
            f"mean_{pipeline_b}": _safe_nanmean(bvals),
            f"mean_{pipeline_a}_minus_{pipeline_b}": _safe_nanmean(diffs),
            "ttest_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "ttest_p": float(t_p) if np.isfinite(t_p) else np.nan,
            "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
            "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_q_fdr"] = benjamini_hochberg(out["ttest_p"].to_numpy(dtype=float))
    out["wilcoxon_q_fdr"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy(dtype=float))
    return out


def plot_pipeline_effect_difference_heatmap(binned, tests_df, out_file, pipeline_a="Anatomical", pipeline_b="Tractography", column_order=None, family_order=None, hemi_order=None, cmap_name="coolwarm", title=None, test="ttest", use_fdr=True):
    if binned.empty:
        return None
    a = (
        binned[binned["pipeline"] == pipeline_a]
        .groupby(["tract_label", "family", "hemi", "bin", "bin_label"], dropna=False)["mean_logJ"]
        .mean()
        .reset_index()
        .rename(columns={"mean_logJ": "a_mean"})
    )
    b = (
        binned[binned["pipeline"] == pipeline_b]
        .groupby(["tract_label", "family", "hemi", "bin", "bin_label"], dropna=False)["mean_logJ"]
        .mean()
        .reset_index()
        .rename(columns={"mean_logJ": "b_mean"})
    )
    group = pd.merge(a, b, on=["tract_label", "family", "hemi", "bin", "bin_label"], how="inner")
    if group.empty:
        return None
    group["mean_diff"] = group["a_mean"] - group["b_mean"]
    cols = column_order or make_tract_column_order(group["tract_label"].unique(), family_order=family_order, hemi_order=hemi_order)
    rows = list(range(1, len(BIN_LABELS) + 1))
    mat = np.full((len(rows), len(cols)), np.nan, dtype=float)
    txt = [["" for _ in cols] for _ in rows]
    txt_draw_cross = [[False for _ in cols] for _ in rows]
    for i, bin_idx in enumerate(rows):
        for j, label in enumerate(cols):
            vals = group[(group["tract_label"] == label) & (group["bin"] == bin_idx)]["mean_diff"]
            if len(vals):
                mat[i, j] = float(vals.iloc[0])
                star = ""
                draw_cross = False
                if tests_df is not None and not tests_df.empty:
                    row = tests_df[(tests_df["tract_label"] == label) & (tests_df["bin"] == bin_idx)]
                    star, draw_cross = _cell_label_from_raw_and_fdr(row, test=test)
                if draw_cross:
                    txt_draw_cross[i][j] = True
                    txt[i][j] = ""
                else:
                    txt[i][j] = f"{mat[i, j]:.3f}{star}"
    norm, _ = get_symmetric_norm(mat.ravel(), vlim=None)
    fig_w = max(8, 0.55 * len(cols) + 2.5)
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap=cmap_name, norm=norm, aspect="auto", interpolation="nearest")
    remove_axes_box(ax)
    ax.set_title(title or f"Method effect difference by tract decile: {pipeline_a} minus {pipeline_b} warps", fontsize=11, pad=18, loc="left")
    ax.text(0.0, 1.02, heatmap_symbol_header_text(), transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.25", clip_on=False)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(BIN_LABELS, fontsize=8)
    ax.set_ylabel("tract decile", fontsize=9)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([])
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    for j, label in enumerate(cols):
        hemi = parse_tract_key(label)[1]
        ax.text(j, -0.62, hemi, ha="center", va="center", fontsize=8, clip_on=False)
    start = 0
    while start < len(cols):
        fam = parse_tract_key(cols[start])[0]
        end = start + 1
        while end < len(cols) and parse_tract_key(cols[end])[0] == fam:
            end += 1
        center = 0.5 * (start + end - 1)
        ax.text(center, -1.25, fam, ha="center", va="center", fontsize=8, fontweight="bold", clip_on=False)
        if end - 1 > start:
            ax.plot([start - 0.45, end - 1 + 0.45], [-0.92, -0.92], color="0.60", linewidth=0.5, clip_on=False)
        start = end
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                if txt_draw_cross[i][j]:
                    _draw_nonsignificant_cell_cross(ax, i, j)
                elif txt[i][j]:
                    ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=6, color="black", fontweight="normal")
    align_heatmap_side_label_to_cell_grid(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    cbar = add_heatmap_cell_grid_colorbar(fig, ax, im, "mean binned log det $J$")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file


def compute_endpoint_core_pipeline_effect_subject_contrasts(binned, pipeline_a="Anatomical", pipeline_b="Tractography", endpoint_bins=(1, 2, 9, 10), core_bins=(5, 6)):
    if binned.empty:
        return pd.DataFrame()
    rows = []
    grouped = binned.groupby(["family", "hemi", "tract_label", "subject"], dropna=False)
    for keys, g in grouped:
        family, hemi, tract_label, subject = keys
        row = {
            "family": family,
            "hemi": hemi,
            "tract_label": tract_label,
            "subject": subject,
        }
        ok = True
        for pipe, prefix in [(pipeline_a, "a"), (pipeline_b, "b")]:
            gp = g[g["pipeline"] == pipe]
            endpoint_vals = pd.to_numeric(gp.loc[gp["bin"].isin(endpoint_bins), "mean_logJ"], errors="coerce").to_numpy(dtype=float)
            core_vals = pd.to_numeric(gp.loc[gp["bin"].isin(core_bins), "mean_logJ"], errors="coerce").to_numpy(dtype=float)
            endpoint_vals = endpoint_vals[np.isfinite(endpoint_vals)]
            core_vals = core_vals[np.isfinite(core_vals)]
            if endpoint_vals.size == 0 or core_vals.size == 0:
                ok = False
                break
            endpoint_mean = float(np.mean(endpoint_vals))
            core_mean = float(np.mean(core_vals))
            row[f"{prefix}_pipeline"] = pipe
            row[f"{prefix}_endpoint_mean"] = endpoint_mean
            row[f"{prefix}_core_mean"] = core_mean
            row[f"{prefix}_endpoint_minus_core"] = endpoint_mean - core_mean
        if not ok:
            continue
        row[f"{pipeline_a}_endpoint_minus_core"] = row["a_endpoint_minus_core"]
        row[f"{pipeline_b}_endpoint_minus_core"] = row["b_endpoint_minus_core"]
        row[f"{pipeline_a}_minus_{pipeline_b}_endpoint_core_difference"] = row["a_endpoint_minus_core"] - row["b_endpoint_minus_core"]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_endpoint_core_pipeline_effect_tests(subject_contrasts, pipeline_a="Anatomical", pipeline_b="Tractography", test="ttest"):
    if subject_contrasts.empty:
        return pd.DataFrame()
    diff_col = f"{pipeline_a}_minus_{pipeline_b}_endpoint_core_difference"
    rows = []
    for keys, g in subject_contrasts.groupby(["family", "hemi", "tract_label"], dropna=False):
        family, hemi, tract_label = keys
        diffs = pd.to_numeric(g[diff_col], errors="coerce").to_numpy(dtype=float)
        diffs = diffs[np.isfinite(diffs)]
        a_vals = pd.to_numeric(g[f"{pipeline_a}_endpoint_minus_core"], errors="coerce").to_numpy(dtype=float)
        a_vals = a_vals[np.isfinite(a_vals)]
        b_vals = pd.to_numeric(g[f"{pipeline_b}_endpoint_minus_core"], errors="coerce").to_numpy(dtype=float)
        b_vals = b_vals[np.isfinite(b_vals)]
        if diffs.size >= 2 and np.nanstd(diffs) > 0:
            try:
                t_stat, t_p = stats.ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
            except Exception:
                t_stat, t_p = np.nan, np.nan
        else:
            t_stat, t_p = np.nan, np.nan
        if diffs.size >= 1 and np.any(np.abs(diffs) > 0):
            try:
                w_stat, w_p = stats.wilcoxon(diffs)
            except Exception:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan
        rows.append({
            "family": family,
            "hemi": hemi,
            "tract_label": tract_label,
            "n_subjects": int(diffs.size),
            f"mean_{pipeline_a}_endpoint_minus_core": _safe_nanmean(a_vals),
            f"mean_{pipeline_b}_endpoint_minus_core": _safe_nanmean(b_vals),
            f"mean_{pipeline_a}_minus_{pipeline_b}_endpoint_core_difference": _safe_nanmean(diffs),
            "ttest_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "ttest_p": float(t_p) if np.isfinite(t_p) else np.nan,
            "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
            "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_q_fdr"] = benjamini_hochberg(out["ttest_p"].to_numpy(dtype=float))
    out["wilcoxon_q_fdr"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy(dtype=float))
    return out


def plot_endpoint_core_pipeline_effect_contrasts(subject_contrasts, tests_df, out_file, pipeline_a="Anatomical", pipeline_b="Tractography", family_order=None, hemi_order=None, use_fdr=True, test="ttest"):
    if subject_contrasts.empty:
        return None
    labels = make_tract_column_order(sorted(subject_contrasts["tract_label"].unique()), family_order=family_order, hemi_order=hemi_order)
    fig_w = max(10, 0.72 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 3.9))
    xbase = np.arange(len(labels), dtype=float)
    rng = np.random.default_rng(12345)
    diff_col = f"{pipeline_a}_minus_{pipeline_b}_endpoint_core_difference"
    for i, label in enumerate(labels):
        vals = pd.to_numeric(subject_contrasts.loc[subject_contrasts["tract_label"] == label, diff_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        jitter = rng.normal(0, 0.045, size=vals.size)
        ax.scatter(np.full(vals.size, xbase[i]) + jitter, vals, s=13, alpha=0.55, linewidths=0, color="0.25")
        mean_val = np.mean(vals)
        ax.plot([xbase[i] - 0.22, xbase[i] + 0.22], [mean_val, mean_val], color="tab:blue", linewidth=1.5)
    ax.axhline(0, color="0.75", linewidth=0.6)
    ax.grid(True, axis="y", color="0.92", linewidth=0.5)
    ax.set_axisbelow(True)
    remove_axes_box(ax)
    ax.set_xticks(xbase)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(top=False, labeltop=False)
    p_col = f"{test}_q_fdr" if use_fdr else f"{test}_p"
    tract_star_map = {}
    if tests_df is not None and not tests_df.empty and p_col in tests_df.columns:
        for label in labels:
            row = tests_df[tests_df["tract_label"] == label]
            if len(row) == 0:
                continue
            stars = significance_symbol_from_row(row, test=test)
            if stars:
                tract_star_map[label] = stars
    for j, label in enumerate(labels):
        hemi = parse_tract_key(label)[1]
        stars = tract_star_map.get(label, "")
        hemi_label = f"{hemi}{stars}"
        ax.text(j, 1.018, hemi_label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, clip_on=False)
    start = 0
    while start < len(labels):
        fam = parse_tract_key(labels[start])[0]
        end = start + 1
        while end < len(labels) and parse_tract_key(labels[end])[0] == fam:
            end += 1
        center = 0.5 * (start + end - 1)
        ax.text(center, 1.125, fam, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, fontweight="bold", clip_on=False)
        if end - 1 > start:
            ax.plot([start - 0.45, end - 1 + 0.45], [1.095, 1.095], transform=ax.get_xaxis_transform(), color="0.60", linewidth=0.5, clip_on=False)
        start = end
    ax.set_ylabel(f"({pipeline_a} end-core) - ({pipeline_b} end-core)\nmean log det $J$", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.32, top=0.73)
    title_x = ax.get_position().x0
    fig.text(title_x, 0.992, f"Endpoint-core log det $J$ contrast by tract: {pipeline_a} minus {pipeline_b} warps", ha="left", va="top", fontsize=12, color="0.15")
    fig.text(title_x, 0.925, profile_symbol_header_text(), ha="left", va="top", fontsize=7, color="0.25")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250)
    plt.close(fig)
    return out_file


def load_bundlewarp_summaries(root, glob_pattern=DEFAULT_BUNDLEWARP_SUMMARY_GLOB, subjects=None):
    rows = []
    for subject_dir in find_subject_dirs(root, subjects=subjects):
        subject = subject_dir.name
        files = sorted(subject_dir.glob(glob_pattern))
        if not files:
            continue
        for f in files:
            try:
                df = pd.read_csv(f)
                if "subject" not in df.columns:
                    df.insert(0, "subject", subject)
                else:
                    df["subject"] = df["subject"].fillna(subject)
                if "tract" in df.columns:
                    tract_keys = df["tract"].astype(str)
                elif "tract_label" in df.columns:
                    tract_keys = df["tract_label"].astype(str)
                else:
                    print(f"WARNING: BundleWarp summary lacks tract column, skipping {f}")
                    continue
                parsed = tract_keys.apply(parse_tract_key)
                df["family"] = [x[0] for x in parsed]
                df["hemi"] = [x[1] for x in parsed]
                df["tract_label"] = [x[2] for x in parsed]
                df["source_csv"] = str(f)
                rows.append(df)
            except Exception as exc:
                print(f"WARNING: failed reading BundleWarp summary {f}: {exc}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


BUNDLEWARP_METRIC_GROUPS = {
    "higher_is_better": [
        "mean_centroid_improvement",
        "percent_centroid_improvement",
        "mean_node_residual_improvement",
        "percent_nodes_improved",
        "fraction_younger_streamlines_matched",
        "fraction_older_streamlines_matched",
        "buan_improvement_threshold_5mm",
        "buan_improvement_threshold_7mm",
        "buan_improvement_threshold_10mm",
    ],
    "lower_is_better": [
        "mean_node_residual_post",
        "mean_centroid_post_distance",
        "mean_spread_residual_to_older",
        "match_reuse_rate",
        "mean_tangent_angle_change_deg",
        "p95_tangent_angle_change_deg",
        "p95_abs_log_tangent_stretch",
        "endpoint_displacement_start",
        "endpoint_displacement_end",
        "first10_mean_displacement",
        "last10_mean_displacement",
    ],
    "deformation_magnitude_or_direction": [
        "mean_bundlewarp_displacement",
        "core_mean_displacement",
        "mean_perpendicular_displacement",
        "mean_parallel_displacement",
        "median_abs_parallel_displacement",
        "mean_log_tangent_stretch",
        "mean_spread_change",
    ],
}

def bundlewarp_metric_group(metric):
    for group, metrics in BUNDLEWARP_METRIC_GROUPS.items():
        if metric in metrics:
            return group
    return "other"

def bundlewarp_metric_group_label(group):
    labels = {
        "higher_is_better": "Higher = better",
        "lower_is_better": "Lower = better",
        "deformation_magnitude_or_direction": "Magnitude/direction",
        "other": "Other",
    }
    return labels.get(group, group.replace("_", " "))

def default_bundlewarp_metrics():
    out = []
    for group in ["higher_is_better", "lower_is_better", "deformation_magnitude_or_direction"]:
        out.extend(BUNDLEWARP_METRIC_GROUPS[group])
    return out

def choose_bundlewarp_metrics(df):
    if df is None or df.empty:
        return []
    metrics = default_bundlewarp_metrics()
    return [m for m in metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]

def summarize_bundlewarp_population(df, metrics):
    rows = []
    if df.empty or not metrics:
        return pd.DataFrame()
    for keys, g in df.groupby(["tract_label", "family", "hemi"], dropna=False):
        tract_label, family, hemi = keys
        row = {"tract_label": tract_label, "family": family, "hemi": hemi, "n_subjects": int(g["subject"].nunique())}
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{metric}_median"] = float(np.median(vals)) if vals.size else np.nan
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
            row[f"{metric}_n"] = int(vals.size)
        rows.append(row)
    return pd.DataFrame(rows)

def metric_display_name(metric):
    replacements = {
        "mean_bundlewarp_displacement": "BW displacement",
        "core_mean_displacement": "Core displacement",
        "mean_perpendicular_displacement": "Perpendicular displacement",
        "mean_parallel_displacement": "Parallel displacement",
        "median_abs_parallel_displacement": "Abs. parallel displacement",
        "mean_log_tangent_stretch": "Log tangent stretch",
        "p95_abs_log_tangent_stretch": "P95 abs. log stretch",
        "mean_tangent_angle_change_deg": "Tangent angle change",
        "p95_tangent_angle_change_deg": "P95 tangent angle change",
        "mean_centroid_improvement": "Centroid improvement",
        "percent_centroid_improvement": "% centroid improvement",
        "mean_centroid_post_distance": "Post centroid distance",
        "mean_node_residual_improvement": "Residual improvement",
        "percent_nodes_improved": "% nodes improved",
        "mean_node_residual_post": "Post residual",
        "mean_spread_change": "Spread change",
        "mean_spread_residual_to_older": "Spread residual",
        "fraction_younger_streamlines_matched": "V02 match fraction",
        "fraction_older_streamlines_matched": "V03 match fraction",
        "match_reuse_rate": "Match reuse rate",
        "endpoint_displacement_start": "Start endpoint displacement",
        "endpoint_displacement_end": "End endpoint displacement",
        "first10_mean_displacement": "First decile displacement",
        "last10_mean_displacement": "Last decile displacement",
        "buan_improvement_threshold_5mm": "BUAN improvement 5 mm",
        "buan_improvement_threshold_7mm": "BUAN improvement 7 mm",
        "buan_improvement_threshold_10mm": "BUAN improvement 10 mm",
    }
    return replacements.get(metric, metric.replace("_", " "))

def plot_bundlewarp_metric_distributions(df, metrics, out_file, family_order=None, hemi_order=None, n_cols=4):
    if df.empty or not metrics:
        return None
    labels = make_tract_column_order(sorted(df["tract_label"].dropna().unique()), family_order=family_order, hemi_order=hemi_order)
    n_metrics = len(metrics)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig_w = max(12, 0.32 * len(labels) * n_cols + 2.0)
    fig_h = max(2.8 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    rng = np.random.default_rng(12345)
    for i, metric in enumerate(metrics):
        ax = axes[i // n_cols, i % n_cols]
        remove_axes_box(ax)
        for x, label in enumerate(labels):
            vals = pd.to_numeric(df.loc[df["tract_label"] == label, metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            jitter = rng.normal(0, 0.045, size=vals.size)
            ax.scatter(np.full(vals.size, x) + jitter, vals, s=13, alpha=0.55, linewidths=0, color="0.25")
            med = np.median(vals)
            ax.plot([x - 0.22, x + 0.22], [med, med], color="tab:blue", linewidth=1.5)
        ax.axhline(0, color="0.75", linewidth=0.6)
        ax.grid(True, axis="y", color="0.92", linewidth=0.5)
        group_label = bundlewarp_metric_group_label(bundlewarp_metric_group(metric))
        ax.set_title(f"{group_label}: {metric_display_name(metric)}", fontsize=9, pad=6, loc="left")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for j in range(n_metrics, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")
    fig.suptitle("Population-level BundleWarp metrics by tract", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file


def demean_t2w_subject(subject, t2w_root, preproc_t2w_root, demeaned_t2w_root, jac_step_mm, force=False):
    t2w_root = Path(t2w_root)
    preproc_t2w_root = Path(preproc_t2w_root)
    out_subject = Path(demeaned_t2w_root) / subject
    meta_path = out_subject / T2W_DEMEAN_META_FILE
    if meta_path.is_file() and not force:
        print(f"[demean/T2w] {subject}: cached, use --force-demean to recompute")
        return True

    reg_dir = t2w_root / subject / T2W_REG_SUBDIR
    mapping_path = reg_dir / T2W_MAPPING_FILE
    younger_img_path = preproc_t2w_root / subject / DEFAULT_YOUNGER_SESSION / YOUNGER_T2W_FILE_TPL.format(
        subject=subject, session=DEFAULT_YOUNGER_SESSION
    )
    older_img_path = preproc_t2w_root / subject / DEFAULT_OLDER_SESSION / YOUNGER_T2W_FILE_TPL.format(
        subject=subject, session=DEFAULT_OLDER_SESSION
    )
    for p in (mapping_path, younger_img_path, older_img_path):
        if not p.is_file():
            print(f"[demean/T2w] {subject}: missing input {p}; skipped")
            return False

    mapping = _load_session_mapping(mapping_path, younger_img_path, older_img_path, reg_dir / T2W_PREALIGN_FILE)
    verify_path = reg_dir / T2W_WARPED_VERIFY_FILE
    corr = _warp_correlation(mapping, younger_img_path, verify_path) if verify_path.is_file() else float("nan")
    if not (corr >= WARP_VERIFY_MIN_CORR):
        raise RuntimeError(f"[demean/T2w] {subject}: mapping re-warp check failed (corr={corr}); refusing to demean")

    younger_img = nib.load(str(younger_img_path))
    younger_data = np.squeeze(np.asarray(younger_img.dataobj))
    t2w_image_domain = np.isfinite(younger_data) & (younger_data > 0)
    summary = _compute_domain_logJ_summary(mapping, t2w_image_domain, younger_img.affine, step_mm=jac_step_mm)
    scalar = summary["mean_logJ"]
    if not np.isfinite(scalar):
        print(f"[demean/T2w] {subject}: no finite domain logJ; skipped")
        return False

    written = []
    for src in sorted((t2w_root / subject).glob(T2W_NODEWISE_GLOB)):
        df = pd.read_csv(src)
        base_col = choose_metric_column(df, requested=None)
        df[T2W_IMAGE_DEMEANED_COL] = _mean_minus_scalar(df[base_col], scalar)
        dst = out_subject / src.relative_to(t2w_root / subject)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
        written.append(str(dst))

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "subject": subject,
        "demean_domain": "younger_skullstripped_preprocessed_T2w_nonzero_voxels",
        "demean_domain_image": str(younger_img_path),
        "mapping": str(mapping_path),
        "jac_step_mm": float(jac_step_mm),
        "rewarp_verify_corr": float(corr),
        "t2w_image_logJ_demeaning": summary,
        "demeaned_column": T2W_IMAGE_DEMEANED_COL,
        "demeaned_csvs": written,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[demean/T2w] {subject}: mean logJ {scalar:+.4f} over {summary['n_finite_logJ_voxels']} voxels; {len(written)} CSVs")
    return True


def demean_tract_subject(subject, tract_root, demeaned_tract_root, jac_step_mm, force=False):
    tract_root = Path(tract_root)
    sub_dir = tract_root / subject
    out_subject = Path(demeaned_tract_root) / subject
    mask_syn_dir = sub_dir / MASK_SYN_DIR
    meta_path = out_subject / MASK_SYN_DIR / TRACT_DEMEAN_META_FILE
    if meta_path.is_file() and not force:
        print(f"[demean/tract] {subject}: cached, use --force-demean to recompute")
        return True
    if not mask_syn_dir.is_dir():
        print(f"[demean/tract] {subject}: no mask_syn directory; skipped")
        return False

    rows = []
    for branch in sorted(p for p in mask_syn_dir.iterdir() if p.is_dir()):
        key = branch.name
        mapping_path = branch / MASK_MAPPING_FILE
        profile_csv = branch / f"{key}{MASK_NODE_PROFILE_SUFFIX}"
        if not mapping_path.is_file() or not profile_csv.is_file():
            print(f"[demean/tract] {subject} {key}: mapping or node profile missing; skipped")
            continue
        family = parse_tract_key(key)[0]
        younger_mask_dir = sub_dir / TRACT_MASKS_SESSION_DIR_TPL.format(session=DEFAULT_YOUNGER_SESSION)
        older_mask_dir = sub_dir / TRACT_MASKS_SESSION_DIR_TPL.format(session=DEFAULT_OLDER_SESSION)
        younger_mask = younger_mask_dir / f"{family}_tracts" / f"{key}{TRACT_MASK_SUFFIX}"
        older_key = key.replace(DEFAULT_YOUNGER_SESSION, DEFAULT_OLDER_SESSION)
        older_mask = older_mask_dir / f"{family}_tracts" / f"{older_key}{TRACT_MASK_SUFFIX}"
        if not younger_mask.is_file():
            hits = sorted(younger_mask_dir.glob(f"**/{key}{TRACT_MASK_SUFFIX}"))
            younger_mask = hits[0] if hits else younger_mask
        if not older_mask.is_file():
            hits = sorted(older_mask_dir.glob(f"**/{older_key}{TRACT_MASK_SUFFIX}"))
            older_mask = hits[0] if hits else older_mask
        if not younger_mask.is_file() or not older_mask.is_file():
            print(f"[demean/tract] {subject} {key}: tract masks missing; skipped")
            continue

        mapping = _load_session_mapping(mapping_path, younger_mask, older_mask, branch / MASK_PREALIGN_FILE)
        verify_path = branch / MASK_WARPED_VERIFY_FILE
        corr = _warp_correlation(mapping, younger_mask, verify_path) if verify_path.is_file() else float("nan")
        if not (corr >= WARP_VERIFY_MIN_CORR):
            raise RuntimeError(f"[demean/tract] {subject} {key}: mask re-warp check failed (corr={corr}); refusing to demean")

        mask_img = nib.load(str(younger_mask))
        mask_data = np.squeeze(np.asarray(mask_img.dataobj)) > 0
        summary = _compute_domain_logJ_summary(mapping, mask_data, mask_img.affine, step_mm=jac_step_mm)
        scalar = summary["mean_logJ"]
        if not np.isfinite(scalar):
            print(f"[demean/tract] {subject} {key}: no finite domain logJ; skipped")
            continue

        df = pd.read_csv(profile_csv)
        base_col = choose_metric_column(df, requested=None)
        df[TRACT_MASK_DEMEANED_COL] = _mean_minus_scalar(df[base_col], scalar)
        dst = out_subject / profile_csv.relative_to(sub_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
        rows.append({
            "subject": subject,
            "tract": key,
            "tract_mask_demean_domain": "younger_moving_tract_mask",
            "tract_mask_demean_mask": str(younger_mask),
            "mapping": str(mapping_path),
            "rewarp_verify_corr": float(corr),
            "mask_n_voxels": summary["n_voxels"],
            "mask_n_finite_logJ_voxels": summary["n_finite_logJ_voxels"],
            "tract_mask_mean_logJ": scalar,
            "demeaned_column": TRACT_MASK_DEMEANED_COL,
            "demeaned_csv": str(dst),
        })
        print(f"[demean/tract] {subject} {key}: mean logJ {scalar:+.4f} over {summary['n_finite_logJ_voxels']} voxels")

    if not rows:
        print(f"[demean/tract] {subject}: nothing demeaned")
        return False
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return True


def run_logj_figure_set(set_name, t2w_root, tract_root, out_root, t2w_metric_col=None, tract_metric_col=None, subjects=None, fig_format="png"):
    set_dir = Path(out_root) / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    t2w = load_pipeline_data(t2w_root, t2w_label, DEFAULT_T2W_GLOB, metric_col=t2w_metric_col, subjects=subjects)
    tract = load_pipeline_data(tract_root, tract_label, DEFAULT_TRACT_GLOB, metric_col=tract_metric_col, subjects=subjects)
    all_data = pd.concat([t2w, tract], ignore_index=True)
    binned = bin_subject_tract_nodes(all_data, n_bins=N_BINS)

    heatmap_inference = compute_heatmap_inference(binned, test=heatmap_test)
    heatmap_inference.to_csv(set_dir / "figure2_heatmap_logJ_one_sample_tests.csv", index=False)

    all_labels = sorted(binned["tract_label"].unique())
    column_order = make_tract_column_order(all_labels, family_order=None, hemi_order=hemi_order)
    _, heatmap_vmax = get_symmetric_norm(get_group_binned_heatmap_values(binned), vlim=None)

    plot_group_heatmap(binned, t2w_label, set_dir / f"figure2_group_heatmap_{t2w_label}.{fig_format}", cmap_name=heatmap_cmap, vlim=heatmap_vmax, column_order=column_order, hemi_order=hemi_order, title=f"Group mean log det $J$ by tract decile: {t2w_label} warps", inference=heatmap_inference, heatmap_test=heatmap_test, use_fdr=heatmap_use_fdr)
    plot_group_heatmap(binned, tract_label, set_dir / f"figure2_group_heatmap_{tract_label}.{fig_format}", cmap_name=heatmap_cmap, vlim=heatmap_vmax, column_order=column_order, hemi_order=hemi_order, title=f"Group mean log det $J$ by tract decile: {tract_label} warps", inference=heatmap_inference, heatmap_test=heatmap_test, use_fdr=heatmap_use_fdr)
    heatmap_color_metadata = {
        "metric_set": set_name,
        "t2w_metric_column": t2w_metric_col or "mean_logJ",
        "tract_metric_column": tract_metric_col or "mean_logJ",
        "cmap": heatmap_cmap,
        "vmin": -float(heatmap_vmax),
        "vcenter": 0.0,
        "vmax": float(heatmap_vmax),
        "scale_basis": "subject-level decile-binned mean log$J$ values across both pipelines",
        "heatmap_test": heatmap_test,
        "heatmap_significance_annotation": "pipeline-specific FDR q-values",
    }
    with open(set_dir / "figure2_heatmap_color_metadata.json", "w") as f:
        json.dump(heatmap_color_metadata, f, indent=2)

    figure4_cell_tests = compute_pipeline_effect_difference_tests(
        binned,
        pipeline_a=t2w_label,
        pipeline_b=tract_label,
        test=pipeline_effect_test,
    )
    figure4_cell_tests.to_csv(set_dir / "figure4_effect_difference_cell_tests.csv", index=False)
    plot_pipeline_effect_difference_heatmap(
        binned,
        figure4_cell_tests,
        set_dir / f"figure4_effect_difference_heatmap.{fig_format}",
        pipeline_a=t2w_label,
        pipeline_b=tract_label,
        column_order=column_order,
        family_order=None,
        hemi_order=hemi_order,
        cmap_name=pipeline_effect_cmap,
        title="Method effect difference by tract decile: Anatomical minus Tractography",
        test=pipeline_effect_test,
        use_fdr=pipeline_effect_use_fdr,
    )
    figure4_endpoint_core_subjects = compute_endpoint_core_pipeline_effect_subject_contrasts(
        binned,
        pipeline_a=t2w_label,
        pipeline_b=tract_label,
    )
    figure4_endpoint_core_tests = compute_endpoint_core_pipeline_effect_tests(
        figure4_endpoint_core_subjects,
        pipeline_a=t2w_label,
        pipeline_b=tract_label,
        test=endpoint_core_pipeline_test,
    )
    figure4_endpoint_core_tests.to_csv(set_dir / "figure4_effect_difference_endpoint_vs_core_tests.csv", index=False)
    plot_endpoint_core_pipeline_effect_contrasts(
        figure4_endpoint_core_subjects,
        figure4_endpoint_core_tests,
        set_dir / f"figure4_effect_difference_profiles.{fig_format}",
        pipeline_a=t2w_label,
        pipeline_b=tract_label,
        family_order=None,
        hemi_order=hemi_order,
        use_fdr=endpoint_core_pipeline_use_fdr,
        test=endpoint_core_pipeline_test,
    )
    return column_order


def parse_args():
    p = argparse.ArgumentParser(description="De-mean saved registration/mask-SyN warp profiles post-hoc, then create dual-set (non-demeaned vs image-demeaned) Figures 2-4 and shared Figures 3/5.")
    p.add_argument("--t2w-output-root", required=True, help="Directory containing subject folders from the Anatomical registration pipeline.")
    p.add_argument("--tract-output-root", required=True, help="Directory containing subject folders from the Tractography pipeline.")
    p.add_argument("--out-dir", required=True, help="Directory where outputs will be written.")
    p.add_argument("--subjects", nargs="+", default=None, help="Optional subject IDs to include.")
    p.add_argument("--preproc-t2w-root", default=None, help="Root of preprocessed T2w images (e.g. preproc_T2w). Required for de-meaning.")
    p.add_argument("--demeaned-out-root", default="derivatives_2_de_meaned", help="Root for augmented per-subject CSVs (new directory; originals untouched).")
    p.add_argument("--jac-step-mm", type=float, default=0.5, help="Central-difference Jacobian step, matching the pipeline default.")
    p.add_argument("--skip-demean", action="store_true", help="Skip the de-meaning phase; plot the non-demeaned set only.")
    p.add_argument("--force-demean", action="store_true", help="Recompute de-meaning even if cached outputs exist.")
    p.add_argument("--skip-trk-figures", action="store_true", help="Skip Figure 3 (.trk point-cloud method disagreement).")
    return p.parse_args()


def main():
    args = parse_args()
    set_minimal_plot_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t2w_root = Path(args.t2w_output_root)
    tract_root = Path(args.tract_output_root)
    demeaned_root = Path(args.demeaned_out_root)
    d_t2w_root = demeaned_root / t2w_root.name
    d_tract_root = demeaned_root / tract_root.name

    if not args.skip_demean:
        if not args.preproc_t2w_root:
            raise SystemExit("--preproc-t2w-root is required unless --skip-demean is set")
        preproc_t2w_root = Path(args.preproc_t2w_root)
        demeaned_root.mkdir(parents=True, exist_ok=True)
        for subject_dir in find_subject_dirs(t2w_root, subjects=args.subjects):
            subject = subject_dir.name
            demean_t2w_subject(subject, t2w_root, preproc_t2w_root, d_t2w_root, args.jac_step_mm, force=args.force_demean)
            demean_tract_subject(subject, tract_root, d_tract_root, args.jac_step_mm, force=args.force_demean)

    metric_sets = [(NON_DEMEANED_SET, t2w_root, tract_root, None, None)]
    if not args.skip_demean:
        metric_sets.append((IMAGE_DEMEANED_SET, d_t2w_root, d_tract_root, T2W_IMAGE_DEMEANED_COL, TRACT_MASK_DEMEANED_COL))

    fig_format = "png"
    for set_name, tw_root, tr_root, tw_col, tr_col in metric_sets:
        print(f"\n=== figure set: {set_name} ===")
        run_logj_figure_set(set_name, tw_root, tr_root, out_dir, t2w_metric_col=tw_col, tract_metric_col=tr_col, subjects=args.subjects, fig_format=fig_format)

    shared_dir = out_dir / SHARED_SET
    shared_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_trk_figures:
        print("Skipping Figure 3 (--skip-trk-figures).")
    else:
        method_disagreement_metric = "symmetric_nn_distance_mm"
        t2w_trks = load_pipeline_trk_map(t2w_root, t2w_label, DEFAULT_T2W_TRK_GLOB, subjects=args.subjects)
        tract_trks = load_pipeline_trk_map(tract_root, tract_label, DEFAULT_TRACT_TRK_GLOB, subjects=args.subjects)
        shared_labels = sorted(set(t2w_trks["tract_label"]) & set(tract_trks["tract_label"]))
        column_order = make_tract_column_order(shared_labels, family_order=None, hemi_order=hemi_order)

        method_disagreement = compute_method_disagreement_by_bin(
            t2w_trks,
            tract_trks,
            n_bins=N_BINS,
            max_points_per_bin=method_disagreement_max_points_per_bin,
        )
        method_disagreement_tests = compute_method_disagreement_inference(
            method_disagreement,
            metric=method_disagreement_metric,
            test=method_disagreement_test,
        )
        method_disagreement_tests.to_csv(shared_dir / "figure3_method_disagreement_cell_tests.csv", index=False)
        endpoint_core_tests = compute_endpoint_core_disagreement_tests(
            method_disagreement,
            metric=method_disagreement_metric,
            test=method_disagreement_test,
        )
        endpoint_core_tests.to_csv(shared_dir / "figure3_method_disagreement_endpoint_vs_core_tests.csv", index=False)
        plot_method_disagreement_heatmap(
            method_disagreement,
            shared_dir / f"figure3_method_disagreement_heatmap.{fig_format}",
            metric=method_disagreement_metric,
            column_order=column_order,
            family_order=None,
            hemi_order=hemi_order,
            cmap_name=method_disagreement_cmap,
            inference=method_disagreement_tests,
            test=method_disagreement_test,
            use_fdr=method_use_fdr,
            title="Method disagreement by tract decile: Anatomical vs Tractography warps",
        )
        plot_method_disagreement_profile_grid(
            method_disagreement,
            shared_dir / f"figure3_method_disagreement_profiles.{fig_format}",
            metric=method_disagreement_metric,
            family_order=None,
            hemi_order=hemi_order,
            endpoint_core_tests=endpoint_core_tests,
            test=method_disagreement_test,
            use_fdr=method_use_fdr,
        )

    bw = load_bundlewarp_summaries(tract_root, glob_pattern=DEFAULT_BUNDLEWARP_SUMMARY_GLOB, subjects=args.subjects)
    if not bw.empty:
        bw_metrics = choose_bundlewarp_metrics(bw)
        if bw_metrics:
            bw_summary = summarize_bundlewarp_population(bw, bw_metrics)
            bw_summary.to_csv(shared_dir / "figure5_bundlewarp_population_metric_summary.csv", index=False)
            plot_bundlewarp_metric_distributions(
                bw,
                bw_metrics,
                shared_dir / f"figure5_bundlewarp_metric_distributions.{fig_format}",
                family_order=None,
                hemi_order=hemi_order,
                n_cols=bundlewarp_distribution_ncols,
            )
    else:
        print("BundleWarp summaries not found; skipping Figure 5.")

    layout_metadata = {
        "metric_sets": [
            {
                "name": set_name,
                "t2w_root": str(tw_root),
                "tract_root": str(tr_root),
                "t2w_metric_column": tw_col or "mean_logJ",
                "tract_metric_column": tr_col or "mean_logJ",
            }
            for set_name, tw_root, tr_root, tw_col, tr_col in metric_sets
        ],
        "shared_dir": SHARED_SET,
        "shared_figure_notes": "Figure 3 (NN-distance disagreement) and Figure 5 (BundleWarp scalars) are invariant to log-J de-meaning",
        "demeaning": {
            "t2w_domain": "younger_skullstripped_preprocessed_T2w_nonzero_voxels",
            "tract_domain": "younger_moving_tract_mask",
            "jac_step_mm": float(args.jac_step_mm),
            "t2w_demeaned_column": T2W_IMAGE_DEMEANED_COL,
            "tract_demeaned_column": TRACT_MASK_DEMEANED_COL,
            "demeaned_root": str(demeaned_root) if not args.skip_demean else None,
        },
    }
    with open(out_dir / "plot_output_layout_metadata.json", "w") as f:
        json.dump(layout_metadata, f, indent=2)

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()

