#!/usr/bin/env python

import argparse
import csv
import traceback
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from scipy.ndimage import binary_dilation, gaussian_filter, zoom
from dipy.align import write_mapping
from dipy.align.imaffine import transform_centers_of_mass, AffineRegistration, MutualInformationMetric
from dipy.align.transforms import (TranslationTransform3D,RigidTransform3D,AffineTransform3D,)
from dipy.align.metrics import CCMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.io.image import load_nifti, save_nifti
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.utils import density_map
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric

def normalize_subject_id(subject_id):
    if subject_id.startswith("sub-"):
        return subject_id
    return f"sub-{subject_id}"

def get_session_matched_tract_key(tract_key, source_session, target_session):
    source_token = f"_{source_session}_T2w"
    target_token = f"_{target_session}_T2w"
    if source_token not in tract_key:
        raise ValueError(f"Could not identify session token {source_token} in tract key: {tract_key}")
    return tract_key.replace(source_token, target_token, 1)

def get_T2w_path(parent_dir, subject_id, session, T2w_file, T2w_root_name="preproc_T2w"):
    subject = normalize_subject_id(subject_id)
    return Path(parent_dir) / T2w_root_name / subject / session / T2w_file

def get_tract_paths(parent_dir, subject_id, session, tract_names, tract_filetype=".trk", tract_root_name="DSI_tracts", hemis=None):
    subject = normalize_subject_id(subject_id)
    base = Path(parent_dir) / tract_root_name / subject / session
    out = {}

    if hemis is None:
        hemis = ["L", "R"]

    for tract in tract_names:
        tract_dir = base / f"{tract}_tracts"
        for hemi in hemis:
            key = f"{tract}_{hemi}_{subject}_{session}_T2w"
            candidates = [
                tract_dir / f"{tract}_{hemi}_{subject}_{session}_T2w{tract_filetype}",
                tract_dir / f"{tract}_{hemi}_{subject}_{session}_T2w",
            ]
            found = None
            for candidate in candidates:
                if candidate.exists():
                    found = candidate
                    break
            if found is None:
                print(
                    f"[SKIP] Missing tract file for {tract}_{hemi} {subject} {session}. "
                    "This tract/hemi/session will be skipped if no matched pair exists."
                )
                continue
            out[key] = found
    return out

def _resolve_roi_path(path_value, parent_dir):
    path = Path(str(path_value).strip())
    if path.is_absolute():
        return path
    return Path(parent_dir) / path


def _roi_centroid_rasmm(roi_path):
    roi_path = Path(roi_path)
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing orientation ROI: {roi_path}")
    img = nib.load(str(roi_path))
    data = np.asarray(img.get_fdata()) > 0
    vox = np.argwhere(data)
    if vox.shape[0] == 0:
        raise ValueError(f"Orientation ROI is empty: {roi_path}")
    centroid_vox = vox.astype(np.float64).mean(axis=0)
    return nib.affines.apply_affine(img.affine, centroid_vox)

def _roi_points_rasmm(roi_path):
    roi_path = Path(roi_path)
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing orientation ROI: {roi_path}")
    img = nib.load(str(roi_path))
    data = np.asarray(img.get_fdata()) > 0
    vox = np.argwhere(data)
    if vox.shape[0] == 0:
        raise ValueError(f"Orientation ROI is empty: {roi_path}")
    return nib.affines.apply_affine(img.affine, vox.astype(np.float64))

def _distance_to_roi_mask(point_rasmm, roi_points_rasmm):
    point_rasmm = np.asarray(point_rasmm, dtype=np.float64)
    tree = cKDTree(np.asarray(roi_points_rasmm, dtype=np.float64))
    dist, _ = tree.query(point_rasmm)
    return float(dist)


def load_orientation_roi_table(parent_dir, subject, sessions, tract_names, hemis, tract_root_name="DSI_tracts", required_entries=None):
    subject = normalize_subject_id(subject)
    out = {}
    missing = []

    if required_entries is None:
        entries = [(tract, hemi, session) for session in sessions for tract in tract_names for hemi in hemis]
    else:
        entries = sorted(set(required_entries))

    for tract, hemi, session in entries:
        base = Path(parent_dir) / tract_root_name / subject / session
        roi_dir = base / f"{tract}_regions"
        start_roi = roi_dir / f"{tract}_{hemi}_start.nii.gz"
        end_roi = roi_dir / f"{tract}_{hemi}_end.nii.gz"
        if not start_roi.exists():
            missing.append(str(start_roi))
        if not end_roi.exists():
            missing.append(str(end_roi))
        if start_roi.exists() and end_roi.exists():
            out[(tract, hemi, session)] = {
                "tract": tract,
                "hemi": hemi,
                "session": session,
                "start_roi": str(start_roi),
                "end_roi": str(end_roi),
                "start_centroid_rasmm": _roi_centroid_rasmm(start_roi),
                "end_centroid_rasmm": _roi_centroid_rasmm(end_roi),
                "start_points_rasmm": _roi_points_rasmm(start_roi),
                "end_points_rasmm": _roi_points_rasmm(end_roi),
            }
    if missing:
        raise FileNotFoundError(
            "Missing required orientation ROI file(s) for tract file(s) that are present. "
            "Missing tract files are skipped, but a present tract file must have matching start/end ROI masks. Missing:\n"
            + "\n".join(missing)
        )
    if not out:
        raise ValueError(
            f"No usable orientation ROIs found for subject {subject}. "
            "Check --tract-root-name, --tract-names, --hemis, session names, and available tract files."
        )
    return out


def parse_tract_family_hemi(tract_key, tract_names, hemis):
    for tract in sorted(tract_names, key=len, reverse=True):
        for hemi in hemis:
            prefix = f"{tract}_{hemi}"
            if tract_key == prefix or tract_key.startswith(prefix + "_"):
                return tract, hemi
    raise ValueError(
        f"Could not parse tract family/hemi from tract key {tract_key}. "
        "Check --tract-names and --hemis."
    )


def get_orientation_landmarks(roi_table, tract, hemi, session):
    key = (tract, hemi, session)
    fallback_key = (tract, hemi, "any")
    if key in roi_table:
        return roi_table[key]
    if fallback_key in roi_table:
        return roi_table[fallback_key]
    raise KeyError(
        f"Missing required orientation ROIs for tract={tract}, hemi={hemi}, session={session}. "
        "Check that <tract>_regions/<tract>_<hemi>_start.nii.gz and <tract>_<hemi>_end.nii.gz exist for each session."
    )


def orient_streamlines_by_landmark_rois(streamlines, start_point, end_point):
    start_point = np.asarray(start_point, dtype=np.float64)
    end_point = np.asarray(end_point, dtype=np.float64)
    oriented = []
    n_flipped = 0
    for sl in streamlines:
        arr = np.asarray(sl, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            continue
        forward = np.linalg.norm(arr[0] - start_point) + np.linalg.norm(arr[-1] - end_point)
        reverse = np.linalg.norm(arr[-1] - start_point) + np.linalg.norm(arr[0] - end_point)
        if reverse < forward:
            oriented.append(arr[::-1].copy())
            n_flipped += 1
        else:
            oriented.append(arr.copy())
    return oriented, n_flipped



def filter_streamlines_by_endpoint_masks(streamlines, landmarks, start_distance_mm, end_distance_mm):
    kept = []
    dropped = []
    for idx, sl in enumerate(streamlines):
        arr = np.asarray(sl, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            dropped.append({
                "streamline_index": int(idx),
                "drop_reason": "invalid_streamline",
                "start_distance_mm": np.nan,
                "end_distance_mm": np.nan,
                "centerline_node_coverage_fraction": np.nan,
            })
            continue
        start_dist = _distance_to_roi_mask(arr[0], landmarks["start_points_rasmm"])
        end_dist = _distance_to_roi_mask(arr[-1], landmarks["end_points_rasmm"])
        if start_dist <= start_distance_mm and end_dist <= end_distance_mm:
            kept.append(arr)
        else:
            dropped.append({
                "streamline_index": int(idx),
                "drop_reason": "endpoint_roi_distance",
                "start_distance_mm": float(start_dist),
                "end_distance_mm": float(end_dist),
                "centerline_node_coverage_fraction": np.nan,
            })
    return kept, dropped

def centerline_node_range_fraction(points_rasmm, centerline, n_nodes):
    if len(points_rasmm) == 0:
        return 0.0
    node_idx = assign_points_to_centerline(points_rasmm, centerline)
    if node_idx.size == 0:
        return 0.0
    return float((int(np.max(node_idx)) - int(np.min(node_idx)) + 1) / max(n_nodes, 1))

def filter_streamlines_by_centerline_coverage(streamlines, centerline, n_nodes, min_coverage_frac, transform_func=None):
    kept = []
    dropped = []
    coverage_values = []
    for idx, sl in enumerate(streamlines):
        arr = np.asarray(sl, dtype=np.float64)
        pts = transform_func(arr) if transform_func is not None else arr
        coverage = centerline_node_range_fraction(pts, centerline, n_nodes)
        coverage_values.append(coverage)
        if coverage >= min_coverage_frac:
            kept.append(arr)
        else:
            dropped.append({
                "streamline_index": int(idx),
                "drop_reason": "centerline_node_coverage",
                "start_distance_mm": np.nan,
                "end_distance_mm": np.nan,
                "centerline_node_coverage_fraction": float(coverage),
            })
    return kept, dropped, coverage_values

def streamline_filter_summary(original_count, endpoint_dropped, coverage_dropped, coverage_values):
    n_endpoint = len(endpoint_dropped)
    n_coverage = len(coverage_dropped)
    n_kept = int(original_count - n_endpoint - n_coverage)
    return {
        "n_streamlines_before_filtering": int(original_count),
        "n_streamlines_analyzed_after_filtering": int(n_kept),
        "n_streamlines_dropped_endpoint_filter": int(n_endpoint),
        "n_streamlines_dropped_centerline_coverage_filter": int(n_coverage),
        "percent_streamlines_dropped_total": float(100.0 * (n_endpoint + n_coverage) / max(original_count, 1)),
        "mean_centerline_node_coverage_fraction_after_endpoint_filter": float(np.nanmean(coverage_values)) if len(coverage_values) else np.nan,
        "median_centerline_node_coverage_fraction_after_endpoint_filter": float(np.nanmedian(coverage_values)) if len(coverage_values) else np.nan,
    }

def write_streamline_filter_qc(out_csv, dropped_rows):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["streamline_index", "drop_reason", "start_distance_mm", "end_distance_mm", "centerline_node_coverage_fraction"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in dropped_rows:
            writer.writerow(row)

def node_contribution_counts(profiles):
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2:
        return np.zeros(0, dtype=int)
    return np.sum(np.isfinite(profiles), axis=0).astype(int)

def build_landmark_oriented_centerline(streamlines, start_point, end_point, n_nodes, qb_threshold=10.0):
    """
    Build a tract reference centerline after enforcing landmark-based orientation.

    The resulting centerline defines the node axis used for all downstream nodewise
    summaries. Individual streamline vertices are assigned to their nearest
    centerline node rather than treated as equal arc-length percentages.
    """
    oriented, n_flipped = orient_streamlines_by_landmark_rois(streamlines, start_point, end_point)
    if len(oriented) == 0:
        raise ValueError("Cannot build centerline from an empty streamline set.")
    resampled = set_number_of_points(oriented, n_nodes)
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=qb_threshold, metric=metric)
    clusters = qb.cluster(resampled)
    if len(clusters) == 0:
        raise RuntimeError("QuickBundles could not estimate a tract centerline.")
    largest_idx = int(np.argmax([len(c.indices) for c in clusters]))
    centerline = np.asarray(clusters.centroids[largest_idx], dtype=np.float64)
    centerline_oriented, centerline_flipped = orient_streamlines_by_landmark_rois(
        [centerline], start_point, end_point
    )
    return np.asarray(centerline_oriented[0], dtype=np.float64), oriented, n_flipped, int(centerline_flipped)


def assign_points_to_centerline(points_rasmm, centerline):
    points_rasmm = np.asarray(points_rasmm, dtype=np.float64)
    if points_rasmm.ndim != 2 or points_rasmm.shape[1] != 3:
        raise ValueError("points_rasmm must be shape (N, 3).")
    tree = cKDTree(np.asarray(centerline, dtype=np.float64))
    _, node_idx = tree.query(points_rasmm)
    return node_idx.astype(int)


def summarize_centerline_profiles(profiles):
    profiles = np.asarray(profiles, dtype=np.float64)
    return (
        _safe_nanmean_axis0(profiles),
        _safe_nanmedian_axis0(profiles),
        _safe_nanstd_axis0(profiles),
    )


def aggregate_values_by_centerline_node(values, node_idx, n_nodes):
    values = np.asarray(values, dtype=np.float64)
    node_idx = np.asarray(node_idx, dtype=int)
    out = np.full(n_nodes, np.nan, dtype=np.float64)
    for node in range(n_nodes):
        vals = values[node_idx == node]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[node] = float(np.mean(vals))
    return out


def save_centerline_as_trk(centerline, reference, out_trk):
    out_trk = Path(out_trk)
    out_trk.parent.mkdir(parents=True, exist_ok=True)
    sft = StatefulTractogram([np.asarray(centerline, dtype=np.float64)], reference, Space.RASMM)
    save_tractogram(sft, str(out_trk), bbox_valid_check=False)
    return out_trk



def save_streamlines_as_trk(streamlines, reference, out_trk):
    out_trk = Path(out_trk)
    out_trk.parent.mkdir(parents=True, exist_ok=True)
    sft = StatefulTractogram(streamlines, reference, Space.RASMM)
    save_tractogram(sft, str(out_trk), bbox_valid_check=False)
    return out_trk

def estimate_prealign(static, static_affine, moving, moving_affine, affine_model="rigid"):
    c_of_mass = transform_centers_of_mass(
        static,
        static_affine,
        moving,
        moving_affine,
    )

    metric = MutualInformationMetric(nbins=32)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=[10000, 1000, 100],
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1],
    )

    translation = affreg.optimize(
        static,
        moving,
        TranslationTransform3D(),
        None,
        static_affine,
        moving_affine,
        starting_affine=c_of_mass.affine,
    )

    rigid = affreg.optimize(
        static,
        moving,
        RigidTransform3D(),
        None,
        static_affine,
        moving_affine,
        starting_affine=translation.affine,
    )

    if affine_model == "rigid":
        return rigid.affine

    affine = affreg.optimize(
        static,
        moving,
        AffineTransform3D(),
        None,
        static_affine,
        moving_affine,
        starting_affine=rigid.affine,
    )
    return affine.affine

def run_syn(static, moving, static_affine, moving_affine, prealign, level_iters):
    metric = CCMetric(3)
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters=level_iters)

    mapping = sdr.optimize(
        static,
        moving,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        prealign=prealign,
    )
    return mapping



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


def _safe_nanpercentile(values, percentile):
    vals = _finite_1d(values)
    if vals.size == 0:
        return np.nan
    return float(np.percentile(vals, percentile))


def _safe_axis0(values, func):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.asarray(func([arr.item()]), dtype=np.float64)
    if arr.ndim == 1:
        return np.asarray([func(arr)], dtype=np.float64)
    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for col in range(arr.shape[1]):
        out[col] = func(arr[:, col])
    return out


def _safe_nanmean_axis0(values):
    return _safe_axis0(values, _safe_nanmean)


def _safe_nanmedian_axis0(values):
    return _safe_axis0(values, _safe_nanmedian)


def _safe_nanstd_axis0(values):
    return _safe_axis0(values, _safe_nanstd)


def _safe_nanpercentile_axis0(values, percentile):
    return _safe_axis0(values, lambda x: _safe_nanpercentile(x, percentile))


def _profiles_to_2d(profiles, n_nodes):
    arr = np.asarray(profiles, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, n_nodes), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] == n_nodes:
            return arr.reshape((1, n_nodes))
        raise ValueError(f'Expected profile length {n_nodes}, got {arr.shape[0]}.')
    if arr.ndim == 2:
        if arr.shape[1] != n_nodes:
            raise ValueError(f'Expected profiles with {n_nodes} nodes, got {arr.shape[1]}.')
        return arr
    raise ValueError(f'Profiles must be 1D or 2D, got shape {arr.shape}.')


def _safe_fraction_true(values):
    arr = np.asarray(values)
    if arr.size == 0:
        return np.nan
    return float(100.0 * np.mean(arr))


def load_checked_tractogram(trk_path, ref_img, label, bbox_valid_check=False):
    trk_path = Path(trk_path)
    if not trk_path.exists():
        raise FileNotFoundError(f"Missing tract file for {label}: {trk_path}")
    if trk_path.name.endswith(".trk.gz") or trk_path.suffix != ".trk":
        raise ValueError(f"Tract file for {label} must be an uncompressed .trk file: {trk_path}")
    sft = load_tractogram(
        str(trk_path),
        ref_img,
        bbox_valid_check=bbox_valid_check,
    )
    if sft is False:
        raise RuntimeError(f"Could not load tract file for {label} with the provided T2w reference: {trk_path}")
    sft.to_rasmm()
    if len(sft.streamlines) == 0:
        raise ValueError(f"Tract file for {label} contains no streamlines: {trk_path}")
    return sft

def _sample_streamline_points(streamlines, max_points=200000):
    pts_list = []
    total = 0
    for sl in streamlines:
        arr = np.asarray(sl)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            continue
        pts_list.append(arr)
        total += arr.shape[0]
        if total >= max_points:
            break
    if not pts_list:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.vstack(pts_list)
    if pts.shape[0] > max_points:
        step = int(np.ceil(pts.shape[0] / max_points))
        pts = pts[::step]
    return pts

def _fraction_points_inside_T2w_grid(points_rasmm, ref_img):
    if points_rasmm.shape[0] == 0:
        return 0.0
    vox = nib.affines.apply_affine(np.linalg.inv(ref_img.affine), points_rasmm)
    shape = np.asarray(ref_img.shape[:3], dtype=float)
    inside = np.all((vox >= 0) & (vox < shape), axis=1)
    return float(np.mean(inside))

def repair_tract_header_to_T2w_reference(trk_path, ref_img, out_trk, label, min_inside_fraction=0.50):
    trk_path = Path(trk_path)
    out_trk = Path(out_trk)
    if not trk_path.exists():
        raise FileNotFoundError(f"Missing tract file for {label}: {trk_path}")
    if trk_path.name.endswith(".trk.gz") or trk_path.suffix != ".trk":
        raise ValueError(f"Tract file for {label} must be an uncompressed .trk file: {trk_path}")
    sft = load_tractogram(
        str(trk_path),
        "same",
        bbox_valid_check=False,
    )
    if sft is False:
        raise RuntimeError(f"Could not load tract file for {label} using reference='same': {trk_path}")
    sft.to_rasmm()
    streamlines = list(sft.streamlines)
    if len(streamlines) == 0:
        raise ValueError(f"Tract file for {label} contains no streamlines: {trk_path}")
    sampled_points = _sample_streamline_points(streamlines)
    fraction_inside = _fraction_points_inside_T2w_grid(sampled_points, ref_img)
    if fraction_inside < min_inside_fraction:
        raise ValueError(
            f"Only {100.0 * fraction_inside:.2f}% of sampled streamline points for {label} fall inside the T2w image grid. "
            "The tract does not appear to be in the expected T2w physical space, so header repair is not safe."
        )
    repaired_sft = StatefulTractogram(
        streamlines,
        ref_img,
        Space.RASMM,
    )
    out_trk.parent.mkdir(parents=True, exist_ok=True)
    save_tractogram(
        repaired_sft,
        str(out_trk),
        bbox_valid_check=False,
    )
    return {
        "tract": label,
        "original_trk_path": str(trk_path),
        "repaired_trk_path": str(out_trk),
        "n_streamlines": int(len(streamlines)),
        "fraction_points_inside_T2w_grid": fraction_inside,
        "streamlines": streamlines,
    }

def warp_tractogram_with_mapping(younger_trk, younger_img, older_img, mapping, out_trk):
    sft = load_checked_tractogram(
        younger_trk,
        younger_img,
        label=Path(younger_trk).name,
        bbox_valid_check=False,
    )
    warped_streamlines = mapping.transform_points_inverse(sft.streamlines)
    warped_sft = StatefulTractogram(
        warped_streamlines,
        older_img,
        Space.RASMM,
    )
    save_tractogram(
        warped_sft,
        str(out_trk),
        bbox_valid_check=False,
    )

def load_streamlines_rasmm(trk_path, ref_img):
    sft = load_checked_tractogram(
        trk_path,
        ref_img,
        label=Path(trk_path).name,
        bbox_valid_check=False,
    )
    return list(sft.streamlines)

def phi_younger_to_older(mapping, points_rasmm):
    return mapping.transform_points_inverse(points_rasmm)

def jacobian_logdet_at_points(mapping, points_rasmm, step_mm=0.5):
    points = np.asarray(points_rasmm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_rasmm must be shape (N, 3).")

    n = points.shape[0]
    J = np.zeros((n, 3, 3), dtype=np.float64)
    basis = np.eye(3)

    for axis in range(3):
        delta = step_mm * basis[axis]
        f_plus = phi_younger_to_older(mapping, points + delta)
        f_minus = phi_younger_to_older(mapping, points - delta)
        J[:, :, axis] = (f_plus - f_minus) / (2.0 * step_mm)
    detJ = np.linalg.det(J)
    logJ = np.full(detJ.shape, np.nan, dtype=np.float64)
    valid = detJ > 0
    logJ[valid] = np.log(detJ[valid])
    return logJ, detJ


def compute_bundle_logJ_profiles(mapping, younger_streamlines, n_nodes, step_mm):
    younger_resampled = set_number_of_points(younger_streamlines, n_nodes)

    profiles = []
    for sl in younger_resampled:
        logJ, _ = jacobian_logdet_at_points(
            mapping,
            sl,
            step_mm=step_mm,
        )
        profiles.append(logJ)

    profiles = _profiles_to_2d(profiles, n_nodes)
    mean_profile = _safe_nanmean_axis0(profiles)
    median_profile = _safe_nanmedian_axis0(profiles)
    std_profile = _safe_nanstd_axis0(profiles)

    summary = {
        "n_streamlines": int(profiles.shape[0]),
        "n_nodes": int(n_nodes),
        "mean_logJ": _safe_nanmean(profiles),
        "median_logJ": _safe_nanmedian(profiles),
        "std_logJ": _safe_nanstd(profiles),
        "percent_nodes_expanding": _safe_fraction_true(profiles > 0),
        "percent_nodes_contracting": _safe_fraction_true(profiles < 0),
        "percent_invalid_or_nonpositive_detJ": _safe_fraction_true(~np.isfinite(profiles)),
    }
    return profiles, mean_profile, median_profile, std_profile, summary

def compute_centerline_logJ_profiles(mapping, younger_streamlines, centerline, n_nodes, step_mm):
    """
    Compute centerline-node log-Jacobian profiles from already oriented and
    filtered younger streamlines. Values are sampled at original younger-space
    vertices, then assigned to nearest older-session centerline nodes after
    the vertices are mapped into older-session space.
    """
    profiles = []
    for sl in younger_streamlines:
        arr = np.asarray(sl, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            continue
        logJ, _ = jacobian_logdet_at_points(mapping, arr, step_mm=step_mm)
        warped_pts = mapping.transform_points_inverse(arr)
        node_idx = assign_points_to_centerline(warped_pts, centerline)
        profiles.append(aggregate_values_by_centerline_node(logJ, node_idx, n_nodes))

    profiles = _profiles_to_2d(profiles, n_nodes)
    mean_profile, median_profile, std_profile = summarize_centerline_profiles(profiles)
    summary = {
        "n_streamlines": int(profiles.shape[0]),
        "n_nodes": int(n_nodes),
        "node_assignment": "nearest_older_session_centerline_node",
        "mean_logJ": _safe_nanmean(profiles),
        "median_logJ": _safe_nanmedian(profiles),
        "std_logJ": _safe_nanstd(profiles),
        "percent_nodes_expanding": _safe_fraction_true(profiles > 0),
        "percent_nodes_contracting": _safe_fraction_true(profiles < 0),
        "percent_invalid_or_unassigned_nodes": _safe_fraction_true(~np.isfinite(profiles)),
    }
    return profiles, mean_profile, median_profile, std_profile, summary

def save_profiles_csv(out_csv, profiles):
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim == 1:
        if profiles.size == 0:
            n_nodes = 0
            profiles = np.empty((0, 0), dtype=np.float64)
        else:
            n_nodes = profiles.shape[0]
            profiles = profiles.reshape((1, n_nodes))
    else:
        n_nodes = profiles.shape[1]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["streamline_index"] + [f"node_{i:03d}" for i in range(n_nodes)]
        writer.writerow(header)
        for i, row in enumerate(profiles):
            writer.writerow([i] + list(row))

def save_node_summary_csv(out_csv, mean_profile, median_profile, std_profile, n_streamlines_contributing=None, global_filter_summary=None):
    if n_streamlines_contributing is None:
        n_streamlines_contributing = np.full(len(mean_profile), np.nan)
    global_filter_summary = global_filter_summary or {}
    global_keys = [
        "n_streamlines_before_filtering",
        "n_streamlines_analyzed_after_filtering",
        "n_streamlines_dropped_endpoint_filter",
        "n_streamlines_dropped_centerline_coverage_filter",
        "percent_streamlines_dropped_total",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "mean_logJ", "median_logJ", "std_logJ", "n_streamlines_contributing"] + global_keys)
        for i, vals in enumerate(zip(mean_profile, median_profile, std_profile, n_streamlines_contributing)):
            writer.writerow([i, vals[0], vals[1], vals[2], vals[3]] + [global_filter_summary.get(k, np.nan) for k in global_keys])

def centroid_distance_mm(bundle_a, bundle_b, qb_threshold=10.0):
    if len(bundle_a) == 0 or len(bundle_b) == 0:
        return np.nan

    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=qb_threshold, metric=metric)
    clusters_a = qb.cluster(bundle_a)
    clusters_b = qb.cluster(bundle_b)

    if len(clusters_a.centroids) == 0 or len(clusters_b.centroids) == 0:
        return np.nan

    centroid_a = clusters_a.centroids[0]
    centroid_b = clusters_b.centroids[0]

    n = min(len(centroid_a), len(centroid_b))
    centroid_a = set_number_of_points([centroid_a], n)[0]
    centroid_b = set_number_of_points([centroid_b], n)[0]

    return float(np.mean(np.linalg.norm(centroid_a - centroid_b, axis=1)))

def tract_density_dice(bundle_a, bundle_b, ref_img):
    shape = ref_img.shape[:3]
    affine = ref_img.affine
    dm_a = density_map(bundle_a, affine, shape)
    dm_b = density_map(bundle_b, affine, shape)
    bin_a = dm_a > 0
    bin_b = dm_b > 0
    denom = bin_a.sum() + bin_b.sum()

    if denom == 0:
        return np.nan
    dice = 2.0 * np.logical_and(bin_a, bin_b).sum() / denom
    return float(dice)

def compute_similarity_metrics(warped_younger_streamlines, older_streamlines, ref_img, n_nodes):
    warped_resampled = set_number_of_points(warped_younger_streamlines, n_nodes)
    older_resampled = set_number_of_points(older_streamlines, n_nodes)
    rng = np.random.default_rng(12345)
    buan_scores = {}
    for threshold in [3, 5, 7, 10]:
        score = bundle_shape_similarity(
            warped_resampled,
            older_resampled,
            rng,
            clust_thr=[0],
            threshold=threshold,
        )
        buan_scores[f"buan_threshold_{threshold}mm"] = float(score)

    metrics = {
        **buan_scores,
        "centroid_distance_mm": centroid_distance_mm(
            warped_resampled,
            older_resampled,
            qb_threshold=10.0,
        ),
        "tract_density_dice": tract_density_dice(
            warped_resampled,
            older_resampled,
            ref_img,
        ),
        "n_streamlines_warped_younger": int(len(warped_resampled)),
        "n_streamlines_older": int(len(older_resampled)),
    }
    return metrics

def save_dicts_csv(out_csv, rows):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = [
        "tract",
        "status",
        "younger_trk",
        "older_trk",
        "unfiltered_warped_younger_trk",
        "filtered_warped_younger_trk",
        "filtered_older_trk",
        "profile_csv",
        "node_summary_csv",
    ]
    all_keys = {key for row in rows for key in row.keys()}
    keys = [key for key in preferred if key in all_keys]
    keys += sorted(all_keys - set(keys))
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def _safe_percentile_window(data, low=1, high=99):
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0.0, 1.0
    vals = data[finite]
    vmin, vmax = np.percentile(vals, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)

def _normalize_for_display(data):
    vmin, vmax = _safe_percentile_window(data)
    out = (data.astype(np.float32) - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out

def _get_slice(data, axis, index):
    if axis == 0:
        sl = data[index, :, :]
    elif axis == 1:
        sl = data[:, index, :]
    elif axis == 2:
        sl = data[:, :, index]
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return np.rot90(sl)

def _make_rgb_overlay(base, overlay, alpha=0.45):
    base_norm = _normalize_for_display(base)
    overlay_norm = _normalize_for_display(overlay)
    rgb = np.zeros(base_norm.shape + (3,), dtype=np.float32)
    rgb[..., 0] = base_norm
    rgb[..., 1] = base_norm
    rgb[..., 2] = base_norm
    rgb[..., 0] = np.clip((1.0 - alpha) * rgb[..., 0] + alpha * overlay_norm, 0.0, 1.0)
    rgb[..., 1] = np.clip((1.0 - alpha) * rgb[..., 1], 0.0, 1.0)
    rgb[..., 2] = np.clip((1.0 - alpha) * rgb[..., 2], 0.0, 1.0)
    return rgb

def _save_qc_page(pdf, fig):
    pdf.savefig(fig, dpi=80)
    plt.close(fig)

def _max_density_slice_index(density, axis, fallback_shape):
    if density.size == 0 or np.nanmax(density) <= 0:
        return fallback_shape[axis] // 2
    sum_axes = tuple(i for i in range(3) if i != axis)
    profile = np.sum(density, axis=sum_axes)
    if profile.size == 0 or np.nanmax(profile) <= 0:
        return fallback_shape[axis] // 2
    return int(np.argmax(profile))

def _get_slab(data, axis, index, half_width=2, mode="mean"):
    lo = max(0, int(index) - half_width)
    hi = min(data.shape[axis], int(index) + half_width + 1)
    if axis == 0:
        slab = data[lo:hi, :, :]
    elif axis == 1:
        slab = data[:, lo:hi, :]
    elif axis == 2:
        slab = data[:, :, lo:hi]
    else:
        raise ValueError("axis must be 0, 1, or 2")
    if mode == "max":
        sl = np.nanmax(slab, axis=axis)
    else:
        sl = np.nanmean(slab, axis=axis)
    return np.rot90(sl)

def _plot_registration_overlay_page(pdf, older_data, warped_younger_data, title):
    axes_info = [
        (0, older_data.shape[0] // 2, "sagittal"),
        (1, older_data.shape[1] // 2, "coronal"),
        (2, older_data.shape[2] // 2, "axial"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5))
    fig.suptitle(title, fontsize=11)

    for col, (axis, idx, label) in enumerate(axes_info):
        older_sl = _get_slice(older_data, axis, idx)
        warped_sl = _get_slice(warped_younger_data, axis, idx)
        overlay = _make_rgb_overlay(older_sl, warped_sl, alpha=0.45)

        axes[0, col].imshow(overlay)
        axes[0, col].set_title(f"{label}: warped overlay")
        axes[0, col].axis("off")

        diff = np.abs(_normalize_for_display(older_sl) - _normalize_for_display(warped_sl))
        axes[1, col].imshow(diff, cmap="gray")
        axes[1, col].set_title(f"{label}: abs. diff.")
        axes[1, col].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _save_qc_page(pdf, fig)

def _tract_density_for_display(streamlines, ref_img):
    if len(streamlines) == 0:
        return np.zeros(ref_img.shape[:3], dtype=np.float32)
    dm = density_map(streamlines, ref_img.affine, ref_img.shape[:3])
    return dm.astype(np.float32)

def _plot_native_tract_T2w_qc_page(pdf, tract_key, session_label, T2w_data, ref_img, streamlines, fraction_inside):
    density = _tract_density_for_display(streamlines, ref_img)
    axes_info = [
        (0, _max_density_slice_index(density, 0, T2w_data.shape), "sagittal"),
        (1, _max_density_slice_index(density, 1, T2w_data.shape), "coronal"),
        (2, _max_density_slice_index(density, 2, T2w_data.shape), "axial"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
    fig.suptitle(f"Native T2w-space tract QC: {tract_key} ({session_label}); {100.0 * fraction_inside:.2f}% sampled points inside T2w grid", fontsize=10)

    for ax, (axis, idx, label) in zip(axes, axes_info):
        T2w_sl = _get_slab(T2w_data, axis, idx, half_width=2, mode="mean")
        density_sl = _get_slab(density, axis, idx, half_width=2, mode="max")
        ax.imshow(_normalize_for_display(T2w_sl), cmap="gray", interpolation="nearest")
        if np.nanmax(density_sl) > 0:
            density_masked = np.ma.masked_where(density_sl <= 0, density_sl)
            ax.imshow(density_masked, cmap="autumn", alpha=0.90, interpolation="nearest")
            ax.contour(density_sl > 0, levels=[0.5], colors="cyan", linewidths=0.5)
        ax.set_title(f"{label}, tract-centered slab index {idx}", fontsize=8)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    _save_qc_page(pdf, fig)

def _plot_summary_page(pdf, metadata, morphometry_summaries, similarity_summaries):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [
        "T2w registration, tract warping, morphometry QC report",
        "",
        f"Subject: {metadata['subject']}",
        f"Younger session: {metadata['younger_session']}",
        f"Older session: {metadata['older_session']}",
        f"Younger T2w: {metadata['younger_T2w']}",
        f"Older T2w: {metadata['older_T2w']}",
        f"Linear model: {metadata['linear_model']}",
        f"SyN level iterations: {metadata['syn_level_iters']}",
        f"Number of tract labels processed: {len(metadata['outputs']['warped_tracts'])}",
        f"Number of morphometry summaries: {len(morphometry_summaries)}",
        f"Number of similarity summaries: {len(similarity_summaries)}",
        "",
        "Interpretation notes:",
        "- Registration overlays: older T2w is gray; warped younger T2w overlay is red.",
        "- Difference images show absolute normalized intensity differences.",
        "- Warped tract QC pages use the older-session T2w mask as a glass-brain shell.",
        "- Low tract similarity and/or visual overlap -> perform manual inspection before interpreting log-Jacobians.",
    ]
    y = 0.97
    for i, line in enumerate(lines):
        fontsize = 13 if i == 0 else 9
        weight = "bold" if i == 0 else "normal"
        ax.text(0.03, y, line, fontsize=fontsize, fontweight=weight, va="top", wrap=True)
        y -= 0.035 if i == 0 else 0.026
    _save_qc_page(pdf, fig)

def _plot_tract_glass_brain_qc_page(pdf, tract_key, older_mask_img, warped_younger_streamlines, older_streamlines, sim_metrics, shell=None, zoom_factors=None, target_vox=0.75):
    if shell is None or zoom_factors is None:
        shell, zoom_factors = compute_glass_brain(older_mask_img, target_vox=target_vox)
    warped_vox = _streamlines_to_vox(warped_younger_streamlines, older_mask_img)
    older_vox = _streamlines_to_vox(older_streamlines, older_mask_img)
    view_labels = ["Sagittal", "Coronal", "Axial"]
    proj_axes = [0, 1, 2]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.8), facecolor="white")
    fig.suptitle(f"Warped younger vs older native tract QC: {tract_key}", fontsize=10)
    for ax, proj_axis, label in zip(axes, proj_axes, view_labels):
        gb_proj = project_glass_brain(shell, proj_axis)
        gb_max = gb_proj.max()
        if gb_max > 0:
            gb_proj = gb_proj / gb_max
        ax.imshow(gb_proj, cmap="gray_r", vmin=0, vmax=1, alpha=0.45, aspect="equal")
        older_segs = _streamlines_to_glass_segments(older_vox, zoom_factors, proj_axis, shell.shape)
        if len(older_segs) > 0:
            lc_old = LineCollection(older_segs, colors="tab:blue", linewidths=0.30, alpha=0.70, rasterized=True)
            ax.add_collection(lc_old)
        warped_segs = _streamlines_to_glass_segments(warped_vox, zoom_factors, proj_axis, shell.shape)
        if len(warped_segs) > 0:
            lc_warped = LineCollection(warped_segs, colors="tab:orange", linewidths=0.30, alpha=0.75, rasterized=True)
            ax.add_collection(lc_warped)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        ax.text(0.02, 0.98, "warped younger", transform=ax.transAxes, va="top", fontsize=7, color="tab:orange")
        ax.text(0.02, 0.90, "older native", transform=ax.transAxes, va="top", fontsize=7, color="tab:blue")
    metric_lines = []
    for key in sorted(sim_metrics.keys()):
        if key in ["tract", "warped_younger_trk", "older_trk"]:
            continue
        value = sim_metrics[key]
        if isinstance(value, float):
            metric_lines.append(f"{key}: {value:.4g}")
        else:
            metric_lines.append(f"{key}: {value}")
    fig.text(0.02, 0.02, " | ".join(metric_lines[:6]), fontsize=6)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    _save_qc_page(pdf, fig)

def _flip_roi_for_qc_display(roi_data):
    return np.flip(np.asarray(roi_data), axis=1)

def _plot_orientation_roi_qc_page(pdf, tract_key, session_label, T2w_data, start_roi_path, end_roi_path):
    start_img = nib.load(str(start_roi_path))
    end_img = nib.load(str(end_roi_path))
    start_data = _flip_roi_for_qc_display(np.asarray(start_img.get_fdata()) > 0)
    end_data = _flip_roi_for_qc_display(np.asarray(end_img.get_fdata()) > 0)
    rows = [("start ROI", start_data), ("end ROI", end_data)]
    axes_info = [(0, "sagittal"), (1, "coronal"), (2, "axial")]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
    fig.suptitle(f"Orientation ROI QC: {tract_key} ({session_label})", fontsize=10)
    for r, (roi_label, roi_data) in enumerate(rows):
        for c, (axis, view_label) in enumerate(axes_info):
            if np.any(roi_data):
                idx = _max_density_slice_index(roi_data.astype(np.float32), axis, T2w_data.shape)
            else:
                idx = T2w_data.shape[axis] // 2
            t2_sl = _get_slab(T2w_data, axis, idx, half_width=1, mode="mean")
            roi_sl = _get_slab(roi_data.astype(np.float32), axis, idx, half_width=1, mode="max")
            ax = axes[r, c]
            ax.imshow(_normalize_for_display(t2_sl), cmap="gray", interpolation="nearest")
            if np.nanmax(roi_sl) > 0:
                roi_masked = np.ma.masked_where(roi_sl <= 0, roi_sl)
                ax.imshow(roi_masked, cmap="autumn" if r == 0 else "winter", alpha=0.70, interpolation="nearest")
                ax.contour(roi_sl > 0, levels=[0.5], colors="yellow" if r == 0 else "cyan", linewidths=0.7)
            ax.set_title(f"{roi_label}: {view_label}, slice {idx}", fontsize=8)
            ax.axis("off")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save_qc_page(pdf, fig)

def write_qc_pdf(qc_pdf_path, metadata, younger_data, older_data, younger_img, older_img, older_mask_img, warped_younger_data, native_tract_qc_records, orientation_roi_qc_records, tract_qc_records, morphometry_summaries, similarity_summaries):
    qc_pdf_path = Path(qc_pdf_path)
    qc_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(qc_pdf_path)) as pdf:
        _plot_summary_page(pdf, metadata, morphometry_summaries, similarity_summaries)
        for record in native_tract_qc_records:
            if record["session_label"] == metadata["younger_session"]:
                T2w_data = younger_data
                ref_img = younger_img
            else:
                T2w_data = older_data
                ref_img = older_img
            _plot_native_tract_T2w_qc_page(
                pdf,
                record["tract"],
                record["session_label"],
                T2w_data,
                ref_img,
                record["streamlines"],
                record["fraction_points_inside_T2w_grid"],
            )
        for record in orientation_roi_qc_records:
            T2w_data = younger_data if record["session_label"] == metadata["younger_session"] else older_data
            _plot_orientation_roi_qc_page(
                pdf,
                record["tract"],
                record["session_label"],
                T2w_data,
                record["start_roi"],
                record["end_roi"],
            )
        _plot_registration_overlay_page(
            pdf,
            older_data,
            warped_younger_data,
            "Registration QC: younger T2w warped to older T2w",
        )
        glass_shell, glass_zoom_factors = compute_glass_brain(older_mask_img, target_vox=0.75)
        for record in tract_qc_records:
            _plot_tract_glass_brain_qc_page(
                pdf,
                record["tract"],
                older_mask_img,
                record["warped_younger_streamlines"],
                record["older_streamlines"],
                record["similarity_metrics"],
                shell=glass_shell,
                zoom_factors=glass_zoom_factors,
                target_vox=0.75,
            )

def _compute_glass_brain_shell(binary_data, vox_sizes, target_vox=0.75, stdev_mm=2.0, dilate_passes=5):
    zoom_factors = vox_sizes / target_vox
    data_up = zoom(binary_data.astype(float), zoom_factors, order=1)
    stdev_vox = stdev_mm / target_vox
    data_smooth = gaussian_filter(data_up, sigma=stdev_vox)
    data_thres = (data_smooth >= 0.5).astype(np.uint8)
    data_dilated = binary_dilation(data_thres, iterations=dilate_passes).astype(np.uint8)
    shell = (data_dilated - data_thres).astype(np.uint8)
    return shell, zoom_factors

def compute_glass_brain(mask_img, target_vox=0.75, stdev_mm=2.0, dilate_passes=5):
    affine = mask_img.affine
    vox_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    binary_data = mask_img.get_fdata() > 0
    shell, zoom_factors = _compute_glass_brain_shell(binary_data, vox_sizes, target_vox, stdev_mm, dilate_passes)
    return shell, zoom_factors

def _orient_projection(proj, axis):
    return proj.T[::-1, :]

def project_glass_brain(shell, axis):
    proj = shell.astype(float).sum(axis=axis)
    return _orient_projection(proj, axis)

def _streamlines_to_vox(streamlines_rasmm, ref_img):
    inv_affine = np.linalg.inv(ref_img.affine)
    vox_sls = []
    for sl in streamlines_rasmm:
        arr = np.asarray(sl)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            continue
        vox_sls.append(nib.affines.apply_affine(inv_affine, arr))
    return vox_sls

def _streamlines_to_glass_segments(streamlines_vox, zoom_factors, axis, shell_shape, max_streamlines=700, n_points=36):
    if len(streamlines_vox) == 0:
        return np.empty((0, 2, 2))
    step = max(1, int(np.ceil(len(streamlines_vox) / max_streamlines)))
    selected = list(streamlines_vox)[::step][:max_streamlines]
    keep = [a for a in range(3) if a != axis]
    n_rows = shell_shape[keep[1]]
    seg_list = []
    for sl in selected:
        arr = np.asarray(sl)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            continue
        if arr.shape[0] > n_points:
            idx = np.linspace(0, arr.shape[0] - 1, n_points).astype(int)
            arr = arr[idx]
        pts_up = arr * zoom_factors
        col = pts_up[:, keep[0]]
        row = (n_rows - 1) - pts_up[:, keep[1]]
        pts_2d = np.column_stack([col, row])
        seg_list.append(np.stack([pts_2d[:-1], pts_2d[1:]], axis=1))
    if not seg_list:
        return np.empty((0, 2, 2))
    return np.concatenate(seg_list)

def render_two_tracts_glass_brain(mask_img, warped_younger_streamlines, older_streamlines, out_png, title, labels=("warped younger", "older native"), tract_colors=("tab:orange", "tab:blue"), target_vox=0.75, glass_alpha=0.45, tract_alpha=0.70, tract_lw=0.35, precomputed_shell=None, precomputed_zoom=None):
    if precomputed_shell is not None and precomputed_zoom is not None:
        shell = precomputed_shell
        zoom_factors = precomputed_zoom
    else:
        shell, zoom_factors = compute_glass_brain(mask_img, target_vox=target_vox)
    warped_vox = _streamlines_to_vox(warped_younger_streamlines, mask_img)
    older_vox = _streamlines_to_vox(older_streamlines, mask_img)
    view_labels = ["Sagittal", "Coronal", "Axial"]
    proj_axes = [0, 1, 2]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    fig.suptitle(title, color="black", fontsize=12, fontweight="bold")
    for ax, proj_axis, vlabel in zip(axs, proj_axes, view_labels):
        gb_proj = project_glass_brain(shell, proj_axis)
        gb_max = gb_proj.max()
        if gb_max > 0:
            gb_proj = gb_proj / gb_max
        ax.imshow(gb_proj, cmap="gray_r", vmin=0, vmax=1, alpha=glass_alpha, aspect="equal")
        older_segs = _streamlines_to_glass_segments(older_vox, zoom_factors, proj_axis, shell.shape)
        if len(older_segs) > 0:
            lc_old = LineCollection(older_segs, colors=tract_colors[1], alpha=tract_alpha, linewidths=tract_lw, rasterized=True)
            ax.add_collection(lc_old)
        warped_segs = _streamlines_to_glass_segments(warped_vox, zoom_factors, proj_axis, shell.shape)
        if len(warped_segs) > 0:
            lc_warped = LineCollection(warped_segs, colors=tract_colors[0], alpha=tract_alpha, linewidths=tract_lw, rasterized=True)
            ax.add_collection(lc_warped)
        ax.set_title(vlabel, color="black", fontsize=9)
        ax.axis("off")
        ax.text(0.02, 0.98, labels[0], transform=ax.transAxes, va="top", fontsize=8, color=tract_colors[0])
        ax.text(0.02, 0.90, labels[1], transform=ax.transAxes, va="top", fontsize=8, color=tract_colors[1])
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def write_glass_brain_pair_qc(glass_brain_dir, older_mask_img, tract_qc_records, target_vox=0.75):
    glass_brain_dir = Path(glass_brain_dir)
    glass_brain_dir.mkdir(parents=True, exist_ok=True)
    shell, zoom_factors = compute_glass_brain(older_mask_img, target_vox=target_vox)
    outputs = {}
    for record in tract_qc_records:
        tract_key = record["tract"]
        out_png = glass_brain_dir / f"{tract_key}_warped_younger_vs_older_native_glassbrain.png"
        render_two_tracts_glass_brain(
            mask_img=older_mask_img,
            warped_younger_streamlines=record["warped_younger_streamlines"],
            older_streamlines=record["older_streamlines"],
            out_png=out_png,
            title=f"{tract_key}: warped younger vs older native, V03 T2w mask glass brain",
            precomputed_shell=shell,
            precomputed_zoom=zoom_factors,
            target_vox=target_vox,
        )
        outputs[tract_key] = str(out_png)
    return outputs

def check_exists(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--T2w-root-name", default="preproc_T2w")
    parser.add_argument("--tract-root-name", default="DSI_tracts")
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--younger-session", default="ses-V02")
    parser.add_argument("--older-session", default="ses-V03")
    parser.add_argument("--younger-T2w-file", required=True)
    parser.add_argument("--older-T2w-file", required=True)
    parser.add_argument("--older-T2w-mask-file", required=True)
    parser.add_argument("--tract-names", nargs="+", required=True)
    parser.add_argument(
        "--hemis",
        nargs="+",
        default=["L", "R"],
        choices=["L", "R"],
        help="Tract labels to search for.",
    )
    parser.add_argument("--tract-filetype", default=".trk")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--linear-model",
        choices=["rigid", "affine"],
        default="rigid",
        help="Use rigid for same-subject longitudinal morphometry unless affine is needed.",
    )
    parser.add_argument(
        "--syn-level-iters",
        nargs="+",
        type=int,
        default=[100, 50, 25],
        help="SyN multiresolution iterations.",
    )
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--jac-step-mm", type=float, default=0.5)
    parser.add_argument("--endpoint-start-distance-mm", type=float, default=20.0, help="Maximum distance from streamline start endpoint to start ROI mask for nodewise analysis.")
    parser.add_argument("--endpoint-end-distance-mm", type=float, default=20.0, help="Maximum distance from streamline end endpoint to end ROI mask for nodewise analysis.")
    parser.add_argument("--min-centerline-coverage-frac", type=float, default=0.50, help="Minimum fraction of the older-session centerline node range a streamline must cover for nodewise analysis.")

    args = parser.parse_args()

    subject = normalize_subject_id(args.subject_id)
    parent_dir = Path(args.parent_dir)
    out_dir = Path(args.out_dir) / subject
    registration_dir = out_dir / "registration"
    morphometry_dir = out_dir / "morphometry"
    profiles_dir = morphometry_dir / "logJ_profiles"
    node_summary_dir = morphometry_dir / "nodewise_logJ_summaries"
    centerline_dir = morphometry_dir / "older_session_centerlines"
    filter_qc_dir = morphometry_dir / "streamline_filter_qc"

    registration_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    node_summary_dir.mkdir(parents=True, exist_ok=True)
    centerline_dir.mkdir(parents=True, exist_ok=True)
    filter_qc_dir.mkdir(parents=True, exist_ok=True)

    younger_T2w = get_T2w_path(
        parent_dir,
        subject,
        args.younger_session,
        args.younger_T2w_file,
        T2w_root_name=args.T2w_root_name,
    )

    older_T2w = get_T2w_path(
        parent_dir,
        subject,
        args.older_session,
        args.older_T2w_file,
        T2w_root_name=args.T2w_root_name,
    )
    older_T2w_mask = get_T2w_path(
        parent_dir,
        subject,
        args.older_session,
        args.older_T2w_mask_file,
        T2w_root_name=args.T2w_root_name,
    )

    check_exists(younger_T2w, "younger T2w image")
    check_exists(older_T2w, "older T2w image")
    check_exists(older_T2w_mask, "older T2w mask image")

    younger_trks_by_name = get_tract_paths(
        parent_dir=parent_dir,
        subject_id=subject,
        session=args.younger_session,
        tract_names=args.tract_names,
        tract_filetype=args.tract_filetype,
        tract_root_name=args.tract_root_name,
        hemis=args.hemis,
    )

    older_trks_by_name = get_tract_paths(
        parent_dir=parent_dir,
        subject_id=subject,
        session=args.older_session,
        tract_names=args.tract_names,
        tract_filetype=args.tract_filetype,
        tract_root_name=args.tract_root_name,
        hemis=args.hemis,
    )

    paired_younger_trks_by_name = {}
    skipped_unpaired_tracts = []
    for tract_key, trk_path in sorted(younger_trks_by_name.items()):
        older_tract_key = get_session_matched_tract_key(
            tract_key,
            args.younger_session,
            args.older_session,
        )
        if older_tract_key not in older_trks_by_name:
            print(
                f"[SKIP] Found younger tract {tract_key}, but missing matched older-session tract "
                f"{older_tract_key}. Skipping this tract/hemi."
            )
            skipped_unpaired_tracts.append({
                "tract": tract_key,
                "status": "skipped_missing_matched_older_trk",
                "missing_older_tract_key": older_tract_key,
                "younger_trk": str(trk_path),
            })
            continue
        paired_younger_trks_by_name[tract_key] = trk_path

    for older_tract_key, trk_path in sorted(older_trks_by_name.items()):
        younger_tract_key = get_session_matched_tract_key(
            older_tract_key,
            args.older_session,
            args.younger_session,
        )
        if younger_tract_key not in younger_trks_by_name:
            print(
                f"[SKIP] Found older tract {older_tract_key}, but missing matched younger-session tract "
                f"{younger_tract_key}. Skipping this tract/hemi."
            )
            skipped_unpaired_tracts.append({
                "tract": older_tract_key,
                "status": "skipped_missing_matched_younger_trk",
                "missing_younger_tract_key": younger_tract_key,
                "older_trk": str(trk_path),
            })

    younger_trks_by_name = paired_younger_trks_by_name
    older_needed_keys = {
        get_session_matched_tract_key(k, args.younger_session, args.older_session)
        for k in younger_trks_by_name
    }
    older_trks_by_name = {k: v for k, v in older_trks_by_name.items() if k in older_needed_keys}

    if not younger_trks_by_name:
        raise RuntimeError(
            "No matched younger/older tract pairs were found after skipping missing .trk files. "
            "Check --tract-root-name, --tract-names, --hemis, and session labels."
        )

    required_roi_entries = []
    for tract_key in younger_trks_by_name:
        tract_family, hemi = parse_tract_family_hemi(tract_key, args.tract_names, args.hemis)
        required_roi_entries.append((tract_family, hemi, args.younger_session))
        required_roi_entries.append((tract_family, hemi, args.older_session))

    orientation_roi_table = load_orientation_roi_table(
        parent_dir=parent_dir,
        subject=subject,
        sessions=[args.younger_session, args.older_session],
        tract_names=args.tract_names,
        hemis=args.hemis,
        tract_root_name=args.tract_root_name,
        required_entries=required_roi_entries,
    )

    warped_tract_root = (
        out_dir
        / f"warped_{args.tract_root_name}"
        / subject
        / f"warped_to_{args.older_session}"
    )
    filtered_warped_tract_root = (
        out_dir
        / f"warped_filtered_{args.tract_root_name}"
        / subject
        / f"warped_to_{args.older_session}"
    )
    filtered_older_tract_root = (
        out_dir
        / f"older_filtered_{args.tract_root_name}"
        / subject
        / args.older_session
    )
    repaired_tract_root = (
        out_dir
        / f"T2w_header_repaired_{args.tract_root_name}"
        / subject
    )
    glass_brain_qc_dir = out_dir / "glass_brain_tract_QC"

    # Load preprocessed images
    younger_data, younger_affine = load_nifti(str(younger_T2w))
    older_data, older_affine = load_nifti(str(older_T2w))

    # Make sure the images are 3D
    younger_data = np.squeeze(younger_data)
    older_data = np.squeeze(older_data)

    if younger_data.ndim != 3 or older_data.ndim != 3:
        raise ValueError("Both T2w inputs must be 3D images after squeeze().")

    # static = older image; moving = younger image
    static = older_data.astype(np.float32)
    moving = younger_data.astype(np.float32)

    older_img = nib.load(str(older_T2w))
    younger_img = nib.load(str(younger_T2w))
    older_mask_img = nib.load(str(older_T2w_mask))

    native_tract_qc_records = []
    younger_repaired_trks_by_name = {}
    older_repaired_trks_by_name = {}
    for tract_key, trk in younger_trks_by_name.items():
        print(f"Repairing and checking younger native T2w-space tract: {tract_key}")
        out_trk = repaired_tract_root / args.younger_session / trk.parent.name / trk.name
        qc_record = repair_tract_header_to_T2w_reference(trk, younger_img, out_trk, f"younger {tract_key}")
        qc_record["session_label"] = args.younger_session
        native_tract_qc_records.append(qc_record)
        younger_repaired_trks_by_name[tract_key] = out_trk
    for tract_key, trk in older_trks_by_name.items():
        print(f"Repairing and checking older native T2w-space tract: {tract_key}")
        out_trk = repaired_tract_root / args.older_session / trk.parent.name / trk.name
        qc_record = repair_tract_header_to_T2w_reference(trk, older_img, out_trk, f"older {tract_key}")
        qc_record["session_label"] = args.older_session
        native_tract_qc_records.append(qc_record)
        older_repaired_trks_by_name[tract_key] = out_trk

    prealign = estimate_prealign(
        static,
        older_affine,
        moving,
        younger_affine,
        affine_model=args.linear_model,
    )

    prealign_path = registration_dir / "prealign.txt"
    np.savetxt(prealign_path, prealign)

    mapping = run_syn(
        static,
        moving,
        older_affine,
        younger_affine,
        prealign,
        level_iters=args.syn_level_iters,
    )

    warped_moving = mapping.transform(moving)

    warped_T2w_path = registration_dir / "younger_T2w_warped_to_older_T2w.nii.gz"
    save_nifti(str(warped_T2w_path), warped_moving, older_affine)

    mapping_path = registration_dir / "younger_to_older_syn_mapping.nii.gz"
    write_mapping(mapping, str(mapping_path))

    warped_outputs = {}

    for tract_key, trk in younger_repaired_trks_by_name.items():
        tract_base = tract_key.rsplit("_", 4)[0]
        tract_out_dir = warped_tract_root / f"{tract_base}_tracts"
        tract_out_dir.mkdir(parents=True, exist_ok=True)
        out_trk = tract_out_dir / f"{tract_key}_warped_to_{args.older_session}.trk"
        print(f"Warping {tract_key}: {trk} -> {out_trk}")

        warp_tractogram_with_mapping(
            younger_trk=trk,
            younger_img=younger_img,
            older_img=older_img,
            mapping=mapping,
            out_trk=out_trk,
        )
        warped_outputs[tract_key] = str(out_trk)

    morphometry_summaries = []
    similarity_summaries = []
    tract_qc_records = []
    orientation_roi_qc_records = []

    for tract_key, younger_trk in younger_repaired_trks_by_name.items():
        try:
            print(f"Measuring morphometry and similarity for {tract_key}")

            younger_streamlines = load_streamlines_rasmm(
                younger_trk,
                younger_img,
            )
            tract_family, hemi = parse_tract_family_hemi(tract_key, args.tract_names, args.hemis)
            younger_landmarks = get_orientation_landmarks(orientation_roi_table, tract_family, hemi, args.younger_session)
            older_landmarks = get_orientation_landmarks(orientation_roi_table, tract_family, hemi, args.older_session)
            orientation_roi_qc_records.extend([
                {"tract": tract_key, "session_label": args.younger_session, "start_roi": younger_landmarks["start_roi"], "end_roi": younger_landmarks["end_roi"]},
                {"tract": tract_key, "session_label": args.older_session, "start_roi": older_landmarks["start_roi"], "end_roi": older_landmarks["end_roi"]},
            ])

            older_tract_key = get_session_matched_tract_key(
                tract_key,
                args.younger_session,
                args.older_session,
            )

            if older_tract_key not in older_repaired_trks_by_name:
                raise KeyError(
                    f"Found {tract_key} in the younger session, but could not find the expected "
                    f"older-session tract key {older_tract_key}. "
                    "Check that tract labels match across sessions apart from the session label."
                )

            older_streamlines = load_streamlines_rasmm(
                older_repaired_trks_by_name[older_tract_key],
                older_img,
            )
            older_streamlines, n_older_initial_flipped = orient_streamlines_by_landmark_rois(
                older_streamlines,
                older_landmarks["start_centroid_rasmm"],
                older_landmarks["end_centroid_rasmm"],
            )
            n_older_before_filtering = len(older_streamlines)
            older_streamlines_endpoint, older_endpoint_dropped = filter_streamlines_by_endpoint_masks(
                older_streamlines,
                older_landmarks,
                args.endpoint_start_distance_mm,
                args.endpoint_end_distance_mm,
            )
            centerline, older_streamlines_for_centerline, n_older_flipped, n_centerline_flipped = build_landmark_oriented_centerline(
                older_streamlines_endpoint,
                older_landmarks["start_centroid_rasmm"],
                older_landmarks["end_centroid_rasmm"],
                args.n_nodes,
            )
            older_streamlines, older_coverage_dropped, older_coverage_values = filter_streamlines_by_centerline_coverage(
                older_streamlines_for_centerline,
                centerline,
                args.n_nodes,
                args.min_centerline_coverage_frac,
            )
            older_filter_summary = streamline_filter_summary(
                n_older_before_filtering,
                older_endpoint_dropped,
                older_coverage_dropped,
                older_coverage_values,
            )
            centerline_path = save_centerline_as_trk(
                centerline,
                older_img,
                centerline_dir / f"{tract_key}_older_session_centerline.trk",
            )
            older_tract_base = older_tract_key.rsplit("_", 4)[0]
            filtered_older_dir = filtered_older_tract_root / f"{older_tract_base}_tracts"
            filtered_older_trk = filtered_older_dir / f"{older_tract_key}_filtered.trk"
            save_streamlines_as_trk(older_streamlines, older_img, filtered_older_trk)
            younger_streamlines, n_younger_flipped = orient_streamlines_by_landmark_rois(
                younger_streamlines,
                younger_landmarks["start_centroid_rasmm"],
                younger_landmarks["end_centroid_rasmm"],
            )
            n_younger_before_filtering = len(younger_streamlines)
            younger_streamlines_endpoint, younger_endpoint_dropped = filter_streamlines_by_endpoint_masks(
                younger_streamlines,
                younger_landmarks,
                args.endpoint_start_distance_mm,
                args.endpoint_end_distance_mm,
            )
            younger_streamlines, younger_coverage_dropped, younger_coverage_values = filter_streamlines_by_centerline_coverage(
                younger_streamlines_endpoint,
                centerline,
                args.n_nodes,
                args.min_centerline_coverage_frac,
                transform_func=mapping.transform_points_inverse,
            )
            younger_filter_summary = streamline_filter_summary(
                n_younger_before_filtering,
                younger_endpoint_dropped,
                younger_coverage_dropped,
                younger_coverage_values,
            )
            dropped_rows = younger_endpoint_dropped + younger_coverage_dropped
            filter_qc_csv = filter_qc_dir / f"{tract_key}_younger_nodewise_filter_qc.csv"
            write_streamline_filter_qc(filter_qc_csv, dropped_rows)

            tract_base = tract_key.rsplit("_", 4)[0]
            filtered_warped_streamlines_for_nodewise = [
                mapping.transform_points_inverse(np.asarray(sl))
                for sl in younger_streamlines
            ]
            filtered_warped_dir = filtered_warped_tract_root / f"{tract_base}_tracts"
            filtered_warped_trk = filtered_warped_dir / f"{tract_key}_filtered_warped_to_{args.older_session}.trk"
            save_streamlines_as_trk(filtered_warped_streamlines_for_nodewise, older_img, filtered_warped_trk)

            profiles, mean_profile, median_profile, std_profile, morph_summary = compute_centerline_logJ_profiles(
                mapping=mapping,
                younger_streamlines=younger_streamlines,
                centerline=centerline,
                n_nodes=args.n_nodes,
                step_mm=args.jac_step_mm,
            )
            profile_csv = profiles_dir / f"{tract_key}_logJ_streamline_by_node.csv"
            node_csv = node_summary_dir / f"{tract_key}_logJ_node_summary.csv"
            save_profiles_csv(profile_csv, profiles)
            save_node_summary_csv(
                node_csv,
                mean_profile,
                median_profile,
                std_profile,
                n_streamlines_contributing=node_contribution_counts(profiles),
                global_filter_summary=younger_filter_summary,
            )

            morph_summary["tract"] = tract_key
            morph_summary["node_assignment"] = "nearest_older_session_centerline_node"
            morph_summary["older_centerline_trk"] = str(centerline_path)
            morph_summary["n_younger_streamlines_flipped_for_orientation"] = int(n_younger_flipped)
            morph_summary.update({f"younger_filter_{k}": v for k, v in younger_filter_summary.items()})
            morph_summary.update({f"older_filter_{k}": v for k, v in older_filter_summary.items()})
            morph_summary["younger_streamline_filter_qc_csv"] = str(filter_qc_csv)
            morph_summary["n_older_streamlines_flipped_for_centerline_orientation"] = int(n_older_initial_flipped + n_older_flipped)
            morph_summary["n_centerline_flipped_for_orientation"] = int(n_centerline_flipped)
            morph_summary["younger_orientation_start_roi"] = younger_landmarks["start_roi"]
            morph_summary["younger_orientation_end_roi"] = younger_landmarks["end_roi"]
            morph_summary["older_orientation_start_roi"] = older_landmarks["start_roi"]
            morph_summary["older_orientation_end_roi"] = older_landmarks["end_roi"]
            morph_summary["younger_trk"] = str(younger_trk)
            morph_summary["unfiltered_warped_younger_trk"] = warped_outputs[tract_key]
            morph_summary["filtered_warped_younger_trk"] = str(filtered_warped_trk)
            morph_summary["profile_csv"] = str(profile_csv)
            morph_summary["node_summary_csv"] = str(node_csv)
            morphometry_summaries.append(morph_summary)

            warped_younger_streamlines = load_streamlines_rasmm(
                warped_outputs[tract_key],
                older_img,
            )
            warped_younger_streamlines, n_warped_flipped = orient_streamlines_by_landmark_rois(
                warped_younger_streamlines,
                older_landmarks["start_centroid_rasmm"],
                older_landmarks["end_centroid_rasmm"],
            )
            n_warped_before_filtering = len(warped_younger_streamlines)
            warped_younger_endpoint, warped_endpoint_dropped = filter_streamlines_by_endpoint_masks(
                warped_younger_streamlines,
                older_landmarks,
                args.endpoint_start_distance_mm,
                args.endpoint_end_distance_mm,
            )
            warped_younger_streamlines, warped_coverage_dropped, warped_coverage_values = filter_streamlines_by_centerline_coverage(
                warped_younger_endpoint,
                centerline,
                args.n_nodes,
                args.min_centerline_coverage_frac,
            )
            warped_filter_summary = streamline_filter_summary(
                n_warped_before_filtering,
                warped_endpoint_dropped,
                warped_coverage_dropped,
                warped_coverage_values,
            )
            filtered_similarity_warped_trk = filtered_warped_dir / f"{tract_key}_filtered_similarity_warped_to_{args.older_session}.trk"
            save_streamlines_as_trk(warped_younger_streamlines, older_img, filtered_similarity_warped_trk)
            sim_metrics = compute_similarity_metrics(
                warped_younger_streamlines=warped_younger_streamlines,
                older_streamlines=older_streamlines,
                ref_img=older_img,
                n_nodes=args.n_nodes,
            )
            sim_metrics["tract"] = tract_key
            sim_metrics["n_warped_younger_streamlines_flipped_for_orientation"] = int(n_warped_flipped)
            sim_metrics.update({f"warped_younger_filter_{k}": v for k, v in warped_filter_summary.items()})
            sim_metrics.update({f"older_filter_{k}": v for k, v in older_filter_summary.items()})
            sim_metrics["n_older_streamlines_flipped_for_orientation"] = int(n_older_initial_flipped + n_older_flipped)
            sim_metrics["older_orientation_start_roi"] = older_landmarks["start_roi"]
            sim_metrics["older_orientation_end_roi"] = older_landmarks["end_roi"]
            sim_metrics["warped_younger_trk"] = warped_outputs[tract_key]
            sim_metrics["unfiltered_warped_younger_trk"] = warped_outputs[tract_key]
            sim_metrics["filtered_warped_younger_trk"] = str(filtered_similarity_warped_trk)
            sim_metrics["older_trk"] = str(older_repaired_trks_by_name[older_tract_key])
            sim_metrics["filtered_older_trk"] = str(filtered_older_trk)
            similarity_summaries.append(sim_metrics)

            tract_qc_records.append(
                {
                    "tract": tract_key,
                    "warped_younger_streamlines": warped_younger_streamlines,
                    "older_streamlines": older_streamlines,
                    "similarity_metrics": sim_metrics,
                }
            )

        except ValueError as exc:
            if "Cannot build centerline from an empty streamline set" not in str(exc):
                raise
            print(f"[SKIP] {tract_key}: {exc}. This tract/hemi will be marked failed and the script will continue.")
            error_dir = morphometry_dir / "failed_tracts"
            error_dir.mkdir(parents=True, exist_ok=True)
            error_log = error_dir / f"{tract_key}_centerline_error.txt"
            with open(error_log, "w", encoding="utf8") as fh:
                fh.write(traceback.format_exc())
            morphometry_summaries.append({
                "tract": tract_key,
                "status": "failed",
                "failure_reason": "centerline_empty_streamline_set",
                "error_log": str(error_log),
                "younger_trk": str(younger_trk),
            })
            similarity_summaries.append({
                "tract": tract_key,
                "status": "failed",
                "failure_reason": "centerline_empty_streamline_set",
                "error_log": str(error_log),
                "younger_trk": str(younger_trk),
            })
            continue
    morph_csv = morphometry_dir / "tract_morphometry_summary.csv"
    sim_csv = morphometry_dir / "tract_similarity_summary.csv"

    save_dicts_csv(morph_csv, morphometry_summaries)
    save_dicts_csv(sim_csv, similarity_summaries)

    print(f"Writing glass-brain tract QC PNGs to: {glass_brain_qc_dir}")
    glass_brain_qc_outputs = write_glass_brain_pair_qc(
        glass_brain_dir=glass_brain_qc_dir,
        older_mask_img=older_mask_img,
        tract_qc_records=tract_qc_records,
    )

    qc_pdf_path = out_dir / "QC_sanitycheck.pdf"

    metadata = {
        "parent_dir": str(parent_dir),
        "T2w_root_name": args.T2w_root_name,
        "tract_root_name": args.tract_root_name,
        "subject": subject,
        "younger_session": args.younger_session,
        "older_session": args.older_session,
        "younger_T2w": str(younger_T2w),
        "older_T2w": str(older_T2w),
        "older_T2w_mask": str(older_T2w_mask),
        "tract_names": args.tract_names,
        "hemis": args.hemis,
        "tract_filetype": args.tract_filetype,
        "linear_model": args.linear_model,
        "syn_level_iters": args.syn_level_iters,
        "n_nodes": args.n_nodes,
        "jac_step_mm": args.jac_step_mm,
        "endpoint_start_distance_mm": args.endpoint_start_distance_mm,
        "endpoint_end_distance_mm": args.endpoint_end_distance_mm,
        "min_centerline_coverage_frac": args.min_centerline_coverage_frac,
        "inputs": {
            "younger_tracts": {k: str(v) for k, v in younger_trks_by_name.items()},
            "older_tracts": {k: str(v) for k, v in older_trks_by_name.items()},
            "younger_T2w_header_repaired_tracts": {k: str(v) for k, v in younger_repaired_trks_by_name.items()},
            "older_T2w_header_repaired_tracts": {k: str(v) for k, v in older_repaired_trks_by_name.items()},
        },
        "native_tract_T2w_space_qc": [
            {
                "tract": r["tract"],
                "session_label": r["session_label"],
                "original_trk_path": r["original_trk_path"],
                "repaired_trk_path": r["repaired_trk_path"],
                "n_streamlines": r["n_streamlines"],
                "fraction_points_inside_T2w_grid": r["fraction_points_inside_T2w_grid"],
            }
            for r in native_tract_qc_records
        ],
        "outputs": {
            "registration_dir": str(registration_dir),
            "morphometry_dir": str(morphometry_dir),
            "prealign": str(prealign_path),
            "mapping": str(mapping_path),
            "warped_younger_T2w": str(warped_T2w_path),
            "warped_tract_root": str(warped_tract_root),
            "filtered_warped_tract_root": str(filtered_warped_tract_root),
            "filtered_older_tract_root": str(filtered_older_tract_root),
            "T2w_header_repaired_tract_root": str(repaired_tract_root),
            "warped_tracts": warped_outputs,
            "morphometry_summary_csv": str(morph_csv),
            "similarity_summary_csv": str(sim_csv),
            "qc_pdf": str(qc_pdf_path),
            "glass_brain_qc_dir": str(glass_brain_qc_dir),
            "glass_brain_qc_pngs": glass_brain_qc_outputs,
        },
    }

    print(f"Writing QC PDF: {qc_pdf_path}")
    write_qc_pdf(
        qc_pdf_path=qc_pdf_path,
        metadata=metadata,
        younger_data=moving,
        older_data=static,
        younger_img=younger_img,
        older_img=older_img,
        older_mask_img=older_mask_img,
        warped_younger_data=warped_moving,
        native_tract_qc_records=native_tract_qc_records,
        orientation_roi_qc_records=orientation_roi_qc_records,
        tract_qc_records=tract_qc_records,
        morphometry_summaries=morphometry_summaries,
        similarity_summaries=similarity_summaries,
    )

    print("Done.")
    print(f"Saved registration outputs to: {registration_dir}")
    print(f"Saved warped tracts to: {warped_tract_root}")
    print(f"Saved filtered warped tracts to: {filtered_warped_tract_root}")
    print(f"Saved filtered older tracts to: {filtered_older_tract_root}")
    print(f"Saved morphometry outputs to: {morphometry_dir}")
    print(f"Saved QC PDF to: {qc_pdf_path}")
    print(f"Saved glass-brain tract QC PNGs to: {glass_brain_qc_dir}")

if __name__ == "__main__":
    main()
