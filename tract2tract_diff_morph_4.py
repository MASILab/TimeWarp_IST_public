#!/usr/bin/env python

import argparse
import csv
import traceback
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

from dipy.align import write_mapping
from dipy.align.imaffine import transform_centers_of_mass, AffineRegistration, MutualInformationMetric
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D,)
from dipy.align.metrics import CCMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
import dipy.align.streamwarp as streamwarp_module
from dipy.align.streamwarp import bundlewarp_shape_analysis, bundlewarp_vector_filed
from dipy.io.image import load_nifti, save_nifti
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import Streamlines, set_number_of_points
from dipy.tracking.utils import density_map


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


def get_tract_paths(parent_dir, subject_id, session, tract_names, tract_root_name="DSI_tracts", hemis=None):
    subject = normalize_subject_id(subject_id)
    base = Path(parent_dir) / tract_root_name / subject / session
    out = {}

    if hemis is None:
        hemis = ["L", "R"]

    for tract in tract_names:
        tract_dir = base / f"{tract}_tracts"
        for hemi in hemis:
            key = f"{tract}_{hemi}_{subject}_{session}_T2w"
            candidate = tract_dir / f"{tract}_{hemi}_{subject}_{session}_T2w.trk"
            if candidate.exists():
                out[key] = candidate
            else:
                print(f"SKIP: missing tract file for {key}: {candidate}")
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
    tree = cKDTree(np.asarray(roi_points_rasmm, dtype=np.float64))
    dist, _ = tree.query(np.asarray(point_rasmm, dtype=np.float64))
    return float(dist)


def load_orientation_roi_table(parent_dir, subject, sessions, tract_names, hemis, tract_root_name="DSI_tracts"):
    subject = normalize_subject_id(subject)
    out = {}
    for session in sessions:
        base = Path(parent_dir) / tract_root_name / subject / session
        for tract in tract_names:
            roi_dir = base / f"{tract}_regions"
            for hemi in hemis:
                start_roi = roi_dir / f"{tract}_{hemi}_start.nii.gz"
                end_roi = roi_dir / f"{tract}_{hemi}_end.nii.gz"
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
                elif start_roi.exists() != end_roi.exists():
                    missing = end_roi if start_roi.exists() else start_roi
                    print(f"NOTE: incomplete ROI pair found and will error only if matching tract exists: {missing}")
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





def validate_orientation_rois_for_pairs(parent_dir, subject, tract_root_name, pair_keys, tract_names, hemis, younger_session, older_session):
    subject = normalize_subject_id(subject)
    missing = []
    checked = []
    for tract_key in sorted(set(pair_keys)):
        tract, hemi = parse_tract_family_hemi(tract_key, tract_names, hemis)
        for session in [younger_session, older_session]:
            roi_dir = Path(parent_dir) / tract_root_name / subject / session / f"{tract}_regions"
            start_roi = roi_dir / f"{tract}_{hemi}_start.nii.gz"
            end_roi = roi_dir / f"{tract}_{hemi}_end.nii.gz"
            checked.extend([str(start_roi), str(end_roi)])
            if not start_roi.exists():
                missing.append(str(start_roi))
            if not end_roi.exists():
                missing.append(str(end_roi))
    if missing:
        raise FileNotFoundError(
            "Missing required orientation ROI file(s) for tract file(s) that are present in both sessions. "
            "Missing .trk files are skipped, but every analyzed tract/hemi must have start/end ROI masks for both sessions. Missing:\n"
            + "\n".join(missing)
        )
    if not checked:
        raise ValueError(
            "No orientation ROI files were checked because no matched tract pairs were supplied."
        )
    return True

def is_empty_centerline_error(exc):
    return isinstance(exc, ValueError) and "Cannot build centerline from an empty streamline set" in str(exc)


def append_centerline_failure_rows(mask_rows, bundle_rows, row_base, run_mask_syn, run_bundlewarp, exc, error_file):
    failure = dict(row_base)
    failure.update({
        "status": "failed",
        "failure_stage": "older_centerline_construction",
        "failure_reason": str(exc),
        "error_log": str(error_file),
    })
    if run_mask_syn:
        row = dict(failure)
        row["branch"] = "mask_syn"
        mask_rows.append(row)
    if run_bundlewarp:
        row = dict(failure)
        row["branch"] = "bundlewarp"
        bundle_rows.append(row)

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
    return Streamlines(oriented), n_flipped

def check_trk(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing tract file for {label}: {path}")
    if path.name.endswith(".trk.gz") or path.suffix != ".trk":
        raise ValueError(f"This script only supports uncompressed .trk files. Problem for {label}: {path}")




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


def load_streamlines_same_rasmm(trk_path, label=None):
    trk_path = Path(trk_path)
    check_trk(trk_path, label or trk_path.name)
    sft = load_tractogram(str(trk_path), reference="same", bbox_valid_check=False)
    if sft is False:
        raise RuntimeError(f"Could not load tract file using reference='same': {trk_path}")
    sft.to_rasmm()
    if len(sft.streamlines) == 0:
        raise ValueError(f"Tract file contains no streamlines: {trk_path}")
    return list(sft.streamlines)


def save_streamlines_as_trk(streamlines, reference, out_trk):
    out_trk = Path(out_trk)
    out_trk.parent.mkdir(parents=True, exist_ok=True)
    sft = StatefulTractogram(streamlines, reference, Space.RASMM)
    save_tractogram(sft, str(out_trk), bbox_valid_check=False)
    return out_trk


def _tractogram_reference_geometry(trk_path):
    trk_file = nib.streamlines.load(str(trk_path), lazy_load=True)
    header = trk_file.header
    if "dimensions" not in header:
        raise KeyError(f"Could not find TrackVis dimensions in header for: {trk_path}")
    shape = tuple(int(x) for x in header["dimensions"][:3])
    affine = None
    for key in ["voxel_to_rasmm", "vox_to_ras"]:
        if key in header:
            affine = np.asarray(header[key], dtype=np.float64)
            break
    if affine is None:
        raise KeyError(f"Could not find TrackVis voxel-to-RASMM affine in header for: {trk_path}")
    if len(shape) != 3 or any(x <= 0 for x in shape):
        raise ValueError(f"Invalid TrackVis dimensions for {trk_path}: {shape}")
    if affine.shape != (4, 4):
        raise ValueError(f"Invalid TrackVis affine shape for {trk_path}: {affine.shape}")
    return shape, affine


def create_binary_tract_mask_from_trk(trk_path, out_mask, label):
    streamlines = load_streamlines_same_rasmm(trk_path, label)
    shape, affine = _tractogram_reference_geometry(trk_path)
    dm = density_map(streamlines, affine, shape)
    mask = (dm > 0).astype(np.uint8)
    out_mask = Path(out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(str(out_mask), mask, affine)
    return out_mask


def generate_tract_masks_from_trks(trks_by_name, generated_mask_root, session):
    masks_by_name = {}
    for tract_key, trk_path in trks_by_name.items():
        tract_base = tract_key.rsplit("_", 4)[0]
        out_dir = Path(generated_mask_root) / session / f"{tract_base}_tracts"
        out_mask = out_dir / f"{tract_key}_mask.nii.gz"
        print(f"Creating binary tract mask for {tract_key}: {out_mask}")
        masks_by_name[tract_key] = create_binary_tract_mask_from_trk(
            trk_path,
            out_mask,
            tract_key,
        )
    return masks_by_name




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


def summarize_centerline_profiles(profiles, n_nodes=None):
    profiles = np.asarray(profiles, dtype=np.float64)
    if n_nodes is not None:
        profiles = _profiles_to_2d(profiles, n_nodes)
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


def filter_streamlines_by_endpoint_masks(streamlines, landmarks, start_distance_mm, end_distance_mm):
    kept = []
    dropped = []
    for idx, sl in enumerate(streamlines):
        arr = np.asarray(sl, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            dropped.append({"streamline_index": int(idx), "drop_reason": "invalid_streamline", "start_distance_mm": np.nan, "end_distance_mm": np.nan, "centerline_node_coverage_fraction": np.nan})
            continue
        start_dist = _distance_to_roi_mask(arr[0], landmarks["start_points_rasmm"])
        end_dist = _distance_to_roi_mask(arr[-1], landmarks["end_points_rasmm"])
        if start_dist <= start_distance_mm and end_dist <= end_distance_mm:
            kept.append(arr)
        else:
            dropped.append({"streamline_index": int(idx), "drop_reason": "endpoint_roi_distance", "start_distance_mm": float(start_dist), "end_distance_mm": float(end_dist), "centerline_node_coverage_fraction": np.nan})
    return Streamlines(kept), dropped

def centerline_node_range_fraction(points_rasmm, centerline, n_nodes):
    points_rasmm = np.asarray(points_rasmm, dtype=np.float64)
    if points_rasmm.ndim != 2 or points_rasmm.shape[1] != 3 or points_rasmm.shape[0] == 0:
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
        arr_for_coverage = arr
        if transform_func is not None:
            arr_for_coverage = np.asarray(transform_func(arr), dtype=np.float64)
        coverage = centerline_node_range_fraction(arr_for_coverage, centerline, n_nodes)
        coverage_values.append(coverage)
        if coverage >= min_coverage_frac:
            kept.append(arr)
        else:
            dropped.append({"streamline_index": int(idx), "drop_reason": "centerline_node_coverage", "start_distance_mm": np.nan, "end_distance_mm": np.nan, "centerline_node_coverage_fraction": float(coverage)})
    return Streamlines(kept), dropped, coverage_values

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

def aggregate_streamline_values_to_centerline(values_by_streamline, streamlines, centerline, n_nodes, min_coverage_frac=0.0):
    profiles = []
    for vals, sl in zip(values_by_streamline, streamlines):
        arr = np.asarray(sl, dtype=np.float64)
        vals = np.asarray(vals, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or vals.shape[0] != arr.shape[0]:
            continue
        if centerline_node_range_fraction(arr, centerline, n_nodes) < min_coverage_frac:
            continue
        node_idx = assign_points_to_centerline(arr, centerline)
        profiles.append(aggregate_values_by_centerline_node(vals, node_idx, n_nodes))
    return _profiles_to_2d(profiles, n_nodes)

def add_centerline_profile_stats_basic(node_metrics, prefix, profiles):
    profiles = _profiles_to_2d(profiles, len(node_metrics.get("node_assignment_index", []))) if "node_assignment_index" in node_metrics else np.asarray(profiles, dtype=np.float64)
    node_metrics[f"mean_{prefix}"] = _safe_nanmean_axis0(profiles)
    node_metrics[f"median_{prefix}"] = _safe_nanmedian_axis0(profiles)
    node_metrics[f"std_{prefix}"] = _safe_nanstd_axis0(profiles)

def centerline_node_centroids_and_spread(streamlines, centerline, n_nodes, min_coverage_frac=0.0):
    points_by_node = [[] for _ in range(n_nodes)]
    for sl in streamlines:
        arr = np.asarray(sl, dtype=np.float64)
        if centerline_node_range_fraction(arr, centerline, n_nodes) < min_coverage_frac:
            continue
        node_idx = assign_points_to_centerline(arr, centerline)
        for pt, node in zip(arr, node_idx):
            points_by_node[int(node)].append(pt)
    centroids = np.full((n_nodes, 3), np.nan, dtype=np.float64)
    trace = np.full(n_nodes, np.nan, dtype=np.float64)
    spread1 = np.full(n_nodes, np.nan, dtype=np.float64)
    spread2 = np.full(n_nodes, np.nan, dtype=np.float64)
    spread3 = np.full(n_nodes, np.nan, dtype=np.float64)
    area_proxy = np.full(n_nodes, np.nan, dtype=np.float64)
    n_points = np.zeros(n_nodes, dtype=int)
    for node, pts in enumerate(points_by_node):
        if not pts:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        n_points[node] = pts.shape[0]
        centroids[node] = np.nanmean(pts, axis=0)
        if pts.shape[0] < 2:
            continue
        cov = np.cov(pts, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.maximum(eig, 0.0))[::-1]
        trace[node] = float(np.sum(eig))
        spread1[node], spread2[node], spread3[node] = np.sqrt(eig)
        area_proxy[node] = float(np.sqrt(max(eig[0] * eig[1], 0.0)))
    return {"centroids": centroids, "spread_trace": trace, "spread_pc1": spread1, "spread_pc2": spread2, "spread_pc3": spread3, "spread_area_proxy": area_proxy, "n_points_per_node": n_points}

def matched_residual_centerline_profiles(static, moving_aligned, deformed, match_pairs, centerline, n_nodes, min_coverage_frac=0.0):
    static_idx, moving_idx, orientation = parse_match_pairs(match_pairs, static, moving_aligned, deformed)
    pre_profiles = []
    post_profiles = []
    kept_static = []
    kept_moving = []
    for sidx, midx in zip(static_idx, moving_idx):
        s = np.asarray(static[sidx])
        a = np.asarray(moving_aligned[midx])
        d = np.asarray(deformed[midx])
        if centerline_node_range_fraction(d, centerline, n_nodes) < min_coverage_frac:
            continue
        node_idx = assign_points_to_centerline(d, centerline)
        pre_vals = np.linalg.norm(a - s, axis=1)
        post_vals = np.linalg.norm(d - s, axis=1)
        pre_profiles.append(aggregate_values_by_centerline_node(pre_vals, node_idx, n_nodes))
        post_profiles.append(aggregate_values_by_centerline_node(post_vals, node_idx, n_nodes))
        kept_static.append(sidx)
        kept_moving.append(midx)
    pre = np.asarray(pre_profiles, dtype=np.float64)
    post = np.asarray(post_profiles, dtype=np.float64)
    if pre.size == 0:
        empty = np.empty((0, n_nodes), dtype=float)
        return empty, empty, empty, np.full(n_nodes, np.nan), orientation, 0, 0, 0.0
    improvement = pre - post
    percent_improved_by_node = 100.0 * np.nanmean(improvement > 0, axis=0)
    unique_static = len(np.unique(kept_static))
    unique_moving = len(np.unique(kept_moving))
    match_reuse_rate = 1.0 - (unique_static / max(len(kept_static), 1))
    return pre, post, improvement, percent_improved_by_node, orientation, unique_static, unique_moving, float(match_reuse_rate)

def save_centerline_as_trk(centerline, reference, out_trk):
    out_trk = Path(out_trk)
    out_trk.parent.mkdir(parents=True, exist_ok=True)
    sft = StatefulTractogram([np.asarray(centerline, dtype=np.float64)], reference, Space.RASMM)
    save_tractogram(sft, str(out_trk), bbox_valid_check=False)
    return out_trk

def estimate_prealign(static, static_affine, moving, moving_affine, affine_model="rigid"):
    if affine_model == "none":
        return None

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


def run_mask_syn(static, moving, static_affine, moving_affine, prealign, level_iters):
    metric = CCMetric(3)
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters=level_iters)
    return sdr.optimize(
        static,
        moving,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        prealign=prealign,
    )


def transform_streamlines_with_mapping(streamlines, mapping):
    out = []
    for sl in streamlines:
        out.append(mapping.transform_points_inverse(np.asarray(sl)))
    return Streamlines(out)


def phi_moving_to_static(mapping, points_rasmm):
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
        f_plus = phi_moving_to_static(mapping, points + delta)
        f_minus = phi_moving_to_static(mapping, points - delta)
        J[:, :, axis] = (f_plus - f_minus) / (2.0 * step_mm)

    detJ = np.linalg.det(J)
    logJ = np.full(detJ.shape, np.nan, dtype=np.float64)
    valid = detJ > 0
    logJ[valid] = np.log(detJ[valid])
    return logJ, detJ


def summarize_profiles(profiles):
    profiles = np.asarray(profiles, dtype=np.float64)
    return {
        "mean": _safe_nanmean_axis0(profiles),
        "median": _safe_nanmedian_axis0(profiles),
        "std": _safe_nanstd_axis0(profiles),
        "p05": _safe_nanpercentile_axis0(profiles, 5),
        "p25": _safe_nanpercentile_axis0(profiles, 25),
        "p75": _safe_nanpercentile_axis0(profiles, 75),
        "p95": _safe_nanpercentile_axis0(profiles, 95),
    }


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": _safe_nanmean(values),
        "median": _safe_nanmedian(values),
        "std": _safe_nanstd(values),
        "p05": _safe_nanpercentile(values, 5),
        "p95": _safe_nanpercentile(values, 95),
    }


def core_slice(n_nodes, frac=0.10):
    lo = int(np.floor(frac * n_nodes))
    hi = int(np.ceil((1.0 - frac) * n_nodes))
    if hi <= lo:
        return slice(0, n_nodes)
    return slice(lo, hi)


def write_node_profile_csv(out_csv, node_metrics):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(node_metrics.keys())
    n_nodes = len(node_metrics[keys[0]]) if keys else 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node"] + keys)
        for i in range(n_nodes):
            writer.writerow([i] + [node_metrics[k][i] for k in keys])


def write_summary_csv(out_csv, rows):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = [
        "tract",
        "branch",
        "younger_trk",
        "older_trk",
        "warped_trk",
        "linear_moved_trk",
        "nonlinear_moved_trk",
        "n_streamlines_younger",
        "n_streamlines_older",
        "n_matched_pairs",
    ]
    all_keys = {key for row in rows for key in row.keys()}
    keys = [key for key in preferred if key in all_keys]
    keys += sorted(all_keys - set(keys))
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def buan_scores(static_streamlines, moving_streamlines, thresholds, prefix="buan"):
    rng = np.random.default_rng(12345)
    scores = {}
    for threshold in thresholds:
        score = bundle_shape_similarity(
            static_streamlines,
            moving_streamlines,
            rng,
            clust_thr=[0],
            threshold=threshold,
        )
        scores[f"{prefix}_threshold_{threshold}mm"] = float(score)
    return scores


def mask_syn_along_tract_profiles(mapping, moving_streamlines, centerline, n_nodes, step_mm, min_centerline_coverage_frac=0.0):
    """
    Sample mask-SyN logJ/displacement at original moving streamline vertices,
    then assign each value to the nearest older-session centerline node after
    mapping that vertex into older-session space.
    """
    warped_streamlines = transform_streamlines_with_mapping(moving_streamlines, mapping)
    logj_profiles = []
    disp_profiles = []
    for moving_sl, warped_sl in zip(moving_streamlines, warped_streamlines):
        moving_arr = np.asarray(moving_sl, dtype=np.float64)
        warped_arr = np.asarray(warped_sl, dtype=np.float64)
        if moving_arr.ndim != 2 or moving_arr.shape[1] != 3 or moving_arr.shape[0] == 0:
            continue
        logJ, _ = jacobian_logdet_at_points(mapping, moving_arr, step_mm=step_mm)
        disp = np.linalg.norm(warped_arr - moving_arr, axis=1)
        if centerline_node_range_fraction(warped_arr, centerline, n_nodes) < min_centerline_coverage_frac:
            continue
        node_idx = assign_points_to_centerline(warped_arr, centerline)
        logj_profiles.append(aggregate_values_by_centerline_node(logJ, node_idx, n_nodes))
        disp_profiles.append(aggregate_values_by_centerline_node(disp, node_idx, n_nodes))

    logj_profiles = _profiles_to_2d(logj_profiles, n_nodes)
    disp_profiles = _profiles_to_2d(disp_profiles, n_nodes)
    logj = summarize_profiles(logj_profiles)
    disp = summarize_profiles(disp_profiles)
    return warped_streamlines, {
        "mean_logJ": logj["mean"],
        "median_logJ": logj["median"],
        "std_logJ": logj["std"],
        "mean_displacement_mm": disp["mean"],
        "median_displacement_mm": disp["median"],
        "std_displacement_mm": disp["std"],
        "n_values_per_centerline_node": np.sum(np.isfinite(logj_profiles), axis=0),
    }, logj_profiles, disp_profiles

def segment_lengths(streamlines):
    values = []
    for sl in streamlines:
        arr = np.asarray(sl)
        if arr.shape[0] < 2:
            values.append(np.full(arr.shape[0], np.nan))
            continue
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        node_vals = np.empty(arr.shape[0], dtype=float)
        node_vals[0] = seg[0]
        node_vals[-1] = seg[-1]
        if arr.shape[0] > 2:
            node_vals[1:-1] = 0.5 * (seg[:-1] + seg[1:])
        values.append(node_vals)
    return np.asarray(values, dtype=float)


def tangent_vectors(streamlines):
    tangents = []
    for sl in streamlines:
        arr = np.asarray(sl, dtype=np.float64)
        n = arr.shape[0]
        t = np.full((n, 3), np.nan, dtype=np.float64)
        if n < 2:
            tangents.append(t)
            continue
        if n == 2:
            vec = arr[1] - arr[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                t[:] = vec / norm
            tangents.append(t)
            continue
        vecs = np.zeros((n, 3), dtype=np.float64)
        vecs[0] = arr[1] - arr[0]
        vecs[-1] = arr[-1] - arr[-2]
        vecs[1:-1] = arr[2:] - arr[:-2]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        valid = norms[:, 0] > 0
        t[valid] = vecs[valid] / norms[valid]
        tangents.append(t)
    return np.asarray(tangents, dtype=np.float64)


def curvature_profiles(streamlines):
    profiles = []
    tangents = tangent_vectors(streamlines)
    for sl, tan in zip(streamlines, tangents):
        arr = np.asarray(sl, dtype=np.float64)
        n = arr.shape[0]
        curv = np.full(n, np.nan, dtype=np.float64)
        if n < 3:
            profiles.append(curv)
            continue
        for i in range(1, n - 1):
            dt = tan[i + 1] - tan[i - 1]
            ds = np.linalg.norm(arr[i + 1] - arr[i - 1])
            if ds > 0 and np.all(np.isfinite(dt)):
                curv[i] = np.linalg.norm(dt) / ds
        curv[0] = curv[1]
        curv[-1] = curv[-2]
        profiles.append(curv)
    return np.asarray(profiles, dtype=np.float64)


def displacement_decomposition(aligned, deformed):
    aligned = Streamlines(aligned)
    deformed = Streamlines(deformed)
    tan = tangent_vectors(aligned)
    displacement = []
    dx = []
    dy = []
    dz = []
    parallel = []
    abs_parallel = []
    perpendicular = []
    perp_ratio = []
    for a, d, t in zip(aligned, deformed, tan):
        delta = np.asarray(d) - np.asarray(a)
        total = np.linalg.norm(delta, axis=1)
        par = np.sum(delta * t, axis=1)
        par_vec = par[:, None] * t
        perp = np.linalg.norm(delta - par_vec, axis=1)
        ratio = perp / np.maximum(total, 1e-8)
        displacement.append(total)
        dx.append(delta[:, 0])
        dy.append(delta[:, 1])
        dz.append(delta[:, 2])
        parallel.append(par)
        abs_parallel.append(np.abs(par))
        perpendicular.append(perp)
        perp_ratio.append(ratio)
    return {
        "displacement": np.asarray(displacement, dtype=np.float64),
        "delta_x": np.asarray(dx, dtype=np.float64),
        "delta_y": np.asarray(dy, dtype=np.float64),
        "delta_z": np.asarray(dz, dtype=np.float64),
        "parallel_displacement": np.asarray(parallel, dtype=np.float64),
        "abs_parallel_displacement": np.asarray(abs_parallel, dtype=np.float64),
        "perpendicular_displacement": np.asarray(perpendicular, dtype=np.float64),
        "perpendicular_fraction": np.asarray(perp_ratio, dtype=np.float64),
    }


def log_tangent_stretch_profiles(aligned, deformed):
    aligned_len = segment_lengths(aligned)
    deformed_len = segment_lengths(deformed)
    stretch = deformed_len / np.maximum(aligned_len, 1e-8)
    return np.log(np.where(stretch > 0, stretch, np.nan))


def tangent_angle_change_profiles(aligned, deformed):
    tan_a = tangent_vectors(aligned)
    tan_d = tangent_vectors(deformed)
    dot = np.nansum(tan_a * tan_d, axis=2)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def node_centroids_and_spread(streamlines, n_nodes):
    arrs = [np.asarray(sl, dtype=np.float64) for sl in Streamlines(set_number_of_points(streamlines, n_nodes))]
    data = np.stack(arrs, axis=0)
    centroids = np.nanmean(data, axis=0)
    trace = np.full(n_nodes, np.nan, dtype=np.float64)
    spread1 = np.full(n_nodes, np.nan, dtype=np.float64)
    spread2 = np.full(n_nodes, np.nan, dtype=np.float64)
    spread3 = np.full(n_nodes, np.nan, dtype=np.float64)
    area_proxy = np.full(n_nodes, np.nan, dtype=np.float64)
    for node in range(n_nodes):
        pts = data[:, node, :]
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] < 2:
            continue
        cov = np.cov(pts, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.maximum(eig, 0.0))[::-1]
        trace[node] = float(np.sum(eig))
        spread1[node], spread2[node], spread3[node] = np.sqrt(eig)
        area_proxy[node] = float(np.sqrt(max(eig[0] * eig[1], 0.0)))
    return {
        "centroids": centroids,
        "spread_trace": trace,
        "spread_pc1": spread1,
        "spread_pc2": spread2,
        "spread_pc3": spread3,
        "spread_area_proxy": area_proxy,
    }


def parse_match_pairs(match_pairs, static, moving_aligned, deformed):
    pairs = np.asarray(match_pairs)
    if pairs.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), "none"
    pairs = pairs.reshape((-1, 2)).astype(int)
    n_static = len(static)
    n_moving = len(moving_aligned)

    candidates = []
    for orientation in ["static_moving", "moving_static"]:
        if orientation == "static_moving":
            static_idx = pairs[:, 0]
            moving_idx = pairs[:, 1]
        else:
            moving_idx = pairs[:, 0]
            static_idx = pairs[:, 1]
        valid = (
            (static_idx >= 0) & (static_idx < n_static) &
            (moving_idx >= 0) & (moving_idx < n_moving)
        )
        if np.count_nonzero(valid) == 0:
            continue
        static_idx = static_idx[valid]
        moving_idx = moving_idx[valid]
        residuals = []
        for sidx, midx in zip(static_idx, moving_idx):
            residuals.append(np.nanmedian(np.linalg.norm(np.asarray(deformed[midx]) - np.asarray(static[sidx]), axis=1)))
        candidates.append((float(np.nanmedian(residuals)), static_idx, moving_idx, orientation))
    if not candidates:
        raise ValueError("Could not parse BundleWarp match_pairs into valid static/moving indices.")
    candidates.sort(key=lambda x: x[0])
    _, static_idx, moving_idx, orientation = candidates[0]
    return static_idx, moving_idx, orientation


def matched_residual_profiles(static, moving_aligned, deformed, match_pairs):
    static_idx, moving_idx, orientation = parse_match_pairs(match_pairs, static, moving_aligned, deformed)
    if static_idx.size == 0:
        n_nodes = len(moving_aligned[0]) if len(moving_aligned) else 0
        empty = np.empty((0, n_nodes), dtype=float)
        return empty, empty, empty, np.full(n_nodes, np.nan), orientation, 0, 0, 0.0
    pre = []
    post = []
    for sidx, midx in zip(static_idx, moving_idx):
        s = np.asarray(static[sidx])
        a = np.asarray(moving_aligned[midx])
        d = np.asarray(deformed[midx])
        pre.append(np.linalg.norm(a - s, axis=1))
        post.append(np.linalg.norm(d - s, axis=1))
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    improvement = pre - post
    percent_improved_by_node = 100.0 * np.nanmean(improvement > 0, axis=0)
    unique_static = len(np.unique(static_idx))
    unique_moving = len(np.unique(moving_idx))
    match_reuse_rate = 1.0 - (unique_static / max(len(static_idx), 1))
    return pre, post, improvement, percent_improved_by_node, orientation, unique_static, unique_moving, float(match_reuse_rate)


def add_profile_stats(node_metrics, prefix, profiles):
    stats = summarize_profiles(profiles)
    for name, values in stats.items():
        node_metrics[f"{name}_{prefix}"] = values

def add_profile_stats_basic(node_metrics, prefix, profiles):
    profiles = np.asarray(profiles, dtype=np.float64)
    node_metrics[f"mean_{prefix}"] = _safe_nanmean_axis0(profiles)
    node_metrics[f"median_{prefix}"] = _safe_nanmedian_axis0(profiles)
    node_metrics[f"std_{prefix}"] = _safe_nanstd_axis0(profiles)


def bundlewarp_deformation_profiles(static, moving_aligned, deformed_bundle, match_pairs, centerline, n_nodes, min_centerline_coverage_frac=0.0):
    """
    Compute BundleWarp measures at BundleWarp streamline vertices and aggregate
    them to nearest older-session centerline nodes. This avoids interpreting node
    index as equal percent arc length along each streamline.
    """
    static = Streamlines(static)
    moving_aligned = Streamlines(moving_aligned)
    deformed_bundle = Streamlines(deformed_bundle)

    node_metrics = {"node_assignment_index": np.arange(n_nodes, dtype=int)}
    arrays = {}
    deformed_coverage_values = [centerline_node_range_fraction(np.asarray(sl), centerline, n_nodes) for sl in deformed_bundle]
    n_deformed_dropped_coverage = int(np.sum(np.asarray(deformed_coverage_values) < min_centerline_coverage_frac))

    decomp = displacement_decomposition(moving_aligned, deformed_bundle)
    for name, arr in decomp.items():
        prof = aggregate_streamline_values_to_centerline(arr, deformed_bundle, centerline, n_nodes, min_centerline_coverage_frac)
        arrays[f"{name}_profiles"] = prof
        add_centerline_profile_stats_basic(node_metrics, name, prof)

    log_stretch_raw = log_tangent_stretch_profiles(moving_aligned, deformed_bundle)
    log_stretch = aggregate_streamline_values_to_centerline(log_stretch_raw, deformed_bundle, centerline, n_nodes, min_centerline_coverage_frac)
    arrays["log_tangent_stretch_profiles"] = log_stretch
    add_centerline_profile_stats_basic(node_metrics, "log_tangent_stretch", log_stretch)
    abs_log_stretch = np.abs(log_stretch)
    arrays["abs_log_tangent_stretch_profiles"] = abs_log_stretch
    node_metrics["percent_stretching"] = 100.0 * np.nanmean(log_stretch > 0, axis=0)
    node_metrics["percent_compressing"] = 100.0 * np.nanmean(log_stretch < 0, axis=0)

    aligned_curv_raw = curvature_profiles(moving_aligned)
    deformed_curv_raw = curvature_profiles(deformed_bundle)
    curvature_change_raw = deformed_curv_raw - aligned_curv_raw
    aligned_curv = aggregate_streamline_values_to_centerline(aligned_curv_raw, deformed_bundle, centerline, n_nodes, min_centerline_coverage_frac)
    deformed_curv = aggregate_streamline_values_to_centerline(deformed_curv_raw, deformed_bundle, centerline, n_nodes, min_centerline_coverage_frac)
    curvature_change = aggregate_streamline_values_to_centerline(curvature_change_raw, deformed_bundle, centerline, n_nodes, min_centerline_coverage_frac)
    arrays["curvature_aligned_profiles"] = aligned_curv
    arrays["curvature_deformed_profiles"] = deformed_curv
    arrays["curvature_change_profiles"] = curvature_change
    add_centerline_profile_stats_basic(node_metrics, "curvature_aligned", aligned_curv)
    add_centerline_profile_stats_basic(node_metrics, "curvature_deformed", deformed_curv)
    add_centerline_profile_stats_basic(node_metrics, "curvature_change", curvature_change)
    add_centerline_profile_stats_basic(node_metrics, "abs_curvature_change", np.abs(curvature_change))

    angle_change_raw = tangent_angle_change_profiles(moving_aligned, deformed_bundle)
    angle_change = aggregate_streamline_values_to_centerline(angle_change_raw, deformed_bundle, centerline, n_nodes, min_centerline_coverage_frac)
    arrays["tangent_angle_change_deg_profiles"] = angle_change
    add_centerline_profile_stats_basic(node_metrics, "tangent_angle_change_deg", angle_change)
    node_metrics["percent_angle_change_gt15deg"] = 100.0 * np.nanmean(angle_change > 15.0, axis=0)
    node_metrics["percent_angle_change_gt30deg"] = 100.0 * np.nanmean(angle_change > 30.0, axis=0)

    spread_aligned = centerline_node_centroids_and_spread(moving_aligned, centerline, n_nodes, min_centerline_coverage_frac)
    spread_deformed = centerline_node_centroids_and_spread(deformed_bundle, centerline, n_nodes, min_centerline_coverage_frac)
    spread_static = centerline_node_centroids_and_spread(static, centerline, n_nodes, min_centerline_coverage_frac)
    centroid_displacement = np.linalg.norm(spread_deformed["centroids"] - spread_aligned["centroids"], axis=1)
    centroid_pre_distance = np.linalg.norm(spread_aligned["centroids"] - spread_static["centroids"], axis=1)
    centroid_post_distance = np.linalg.norm(spread_deformed["centroids"] - spread_static["centroids"], axis=1)
    centroid_improvement = centroid_pre_distance - centroid_post_distance
    centroid_percent_improvement = 100.0 * centroid_improvement / np.maximum(centroid_pre_distance, 1e-8)
    node_metrics["centroid_displacement"] = centroid_displacement
    node_metrics["centroid_pre_distance_to_older"] = centroid_pre_distance
    node_metrics["centroid_post_distance_to_older"] = centroid_post_distance
    node_metrics["centroid_distance_improvement"] = centroid_improvement
    node_metrics["centroid_percent_improvement"] = centroid_percent_improvement
    node_metrics["n_deformed_points_per_centerline_node"] = spread_deformed["n_points_per_node"]
    node_metrics["n_older_points_per_centerline_node"] = spread_static["n_points_per_node"]

    for spread_name in ["spread_trace", "spread_pc1", "spread_pc2", "spread_pc3", "spread_area_proxy"]:
        node_metrics[f"aligned_{spread_name}"] = spread_aligned[spread_name]
        node_metrics[f"deformed_{spread_name}"] = spread_deformed[spread_name]
        node_metrics[f"older_{spread_name}"] = spread_static[spread_name]
        node_metrics[f"change_{spread_name}_aligned_to_deformed"] = spread_deformed[spread_name] - spread_aligned[spread_name]
        node_metrics[f"residual_{spread_name}_deformed_to_older"] = spread_deformed[spread_name] - spread_static[spread_name]

    pre_residual, post_residual, residual_improvement, percent_improved_by_node, orientation, unique_static, unique_moving, match_reuse_rate = matched_residual_centerline_profiles(
        static,
        moving_aligned,
        deformed_bundle,
        match_pairs,
        centerline,
        n_nodes,
        min_centerline_coverage_frac,
    )
    arrays["pre_residual_profiles"] = pre_residual
    arrays["post_residual_profiles"] = post_residual
    arrays["residual_improvement_profiles"] = residual_improvement
    add_centerline_profile_stats_basic(node_metrics, "pre_residual", pre_residual)
    add_centerline_profile_stats_basic(node_metrics, "post_residual", post_residual)
    add_centerline_profile_stats_basic(node_metrics, "residual_improvement", residual_improvement)
    node_metrics["percent_matched_pairs_improved"] = percent_improved_by_node

    disp = arrays["displacement_profiles"]
    perp = arrays["perpendicular_displacement_profiles"]
    par = arrays["parallel_displacement_profiles"]
    abspar = arrays["abs_parallel_displacement_profiles"]
    summary = {
        "node_assignment": "nearest_older_session_centerline_node",
        "n_bundlewarp_streamlines_before_centerline_coverage_filter": int(len(deformed_bundle)),
        "n_bundlewarp_streamlines_dropped_centerline_coverage_filter": int(n_deformed_dropped_coverage),
        "n_bundlewarp_streamlines_analyzed_after_centerline_coverage_filter": int(len(deformed_bundle) - n_deformed_dropped_coverage),
        "mean_bundlewarp_centerline_node_coverage_fraction": float(np.nanmean(deformed_coverage_values)) if len(deformed_coverage_values) else np.nan,
        "median_bundlewarp_centerline_node_coverage_fraction": float(np.nanmedian(deformed_coverage_values)) if len(deformed_coverage_values) else np.nan,
        "mean_bundlewarp_displacement": float(np.nanmean(disp)),
        "median_bundlewarp_displacement": float(np.nanmedian(disp)),
        "p95_bundlewarp_displacement": float(np.nanpercentile(disp, 95)),
        "mean_perpendicular_displacement": float(np.nanmean(perp)),
        "median_perpendicular_displacement": float(np.nanmedian(perp)),
        "mean_parallel_displacement": float(np.nanmean(par)),
        "median_abs_parallel_displacement": float(np.nanmedian(abspar)),
        "mean_log_tangent_stretch": float(np.nanmean(log_stretch)),
        "median_log_tangent_stretch": float(np.nanmedian(log_stretch)),
        "p95_abs_log_tangent_stretch": float(np.nanpercentile(abs_log_stretch, 95)),
        "mean_tangent_angle_change_deg": float(np.nanmean(angle_change)),
        "median_tangent_angle_change_deg": float(np.nanmedian(angle_change)),
        "p95_tangent_angle_change_deg": float(np.nanpercentile(angle_change, 95)),
        "mean_centroid_pre_distance": float(np.nanmean(centroid_pre_distance)),
        "mean_centroid_post_distance": float(np.nanmean(centroid_post_distance)),
        "mean_centroid_improvement": float(np.nanmean(centroid_improvement)),
        "percent_centroid_improvement": float(100.0 * np.nanmean(centroid_improvement) / max(np.nanmean(centroid_pre_distance), 1e-8)),
        "mean_node_residual_pre": float(np.nanmean(pre_residual)),
        "mean_node_residual_post": float(np.nanmean(post_residual)),
        "mean_node_residual_improvement": float(np.nanmean(residual_improvement)),
        "percent_nodes_improved": float(100.0 * np.nanmean(residual_improvement > 0)),
        "mean_spread_aligned": float(np.nanmean(spread_aligned["spread_trace"])),
        "mean_spread_deformed": float(np.nanmean(spread_deformed["spread_trace"])),
        "mean_spread_older": float(np.nanmean(spread_static["spread_trace"])),
        "mean_spread_change": float(np.nanmean(spread_deformed["spread_trace"] - spread_aligned["spread_trace"])),
        "mean_spread_residual_to_older": float(np.nanmean(spread_deformed["spread_trace"] - spread_static["spread_trace"])),
        "endpoint_displacement_start": float(np.nanmedian(disp[:, 0])),
        "endpoint_displacement_end": float(np.nanmedian(disp[:, -1])),
        "n_matched_pairs": int(pre_residual.shape[0]),
        "match_pair_orientation_used": orientation,
        "n_unique_older_streamlines_matched": int(unique_static),
        "n_unique_younger_streamlines_matched": int(unique_moving),
        "match_reuse_rate": float(match_reuse_rate),
        "fraction_younger_streamlines_matched": float(unique_moving / max(len(moving_aligned), 1)),
        "fraction_older_streamlines_matched": float(unique_static / max(len(static), 1)),
    }
    cs = core_slice(n_nodes)
    summary.update({
        "core_mean_displacement": float(np.nanmean(disp[:, cs])),
        "core_mean_residual_improvement": float(np.nanmean(residual_improvement[:, cs])) if residual_improvement.size else np.nan,
        "first10_mean_displacement": float(np.nanmean(disp[:, :cs.start])) if cs.start > 0 else np.nan,
        "last10_mean_displacement": float(np.nanmean(disp[:, cs.stop:])) if cs.stop < n_nodes else np.nan,
    })
    overwarp = (
        (summary["p95_bundlewarp_displacement"] > 20.0) or
        (summary["p95_tangent_angle_change_deg"] > 30.0) or
        (summary["p95_abs_log_tangent_stretch"] > np.log(2.0))
    )
    summary["overwarp_flag"] = bool(overwarp)
    summary["overwarp_rule"] = "p95_displacement>20mm OR p95_angle_change>30deg OR p95_abs_log_stretch>log(2)"
    return node_metrics, arrays, summary

def save_bundlewarp_arrays(branch_dir, tract_key, arrays):
    saved = {}
    for name, arr in arrays.items():
        out = Path(branch_dir) / f"{tract_key}_bundlewarp_{name}.npy"
        np.save(out, arr)
        saved[name] = str(out)
    return saved


def ensure_pandas_for_bundlewarp():
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DIPY BundleWarp requires pandas, but pandas is not installed in this environment. "
            "Install it with: python -m pip install pandas"
        ) from exc


def run_bundlewarp_with_slr_length_gate(
    static,
    moving,
    alpha,
    beta,
    max_iter,
    slr_min_length_mm,
    slr_max_length_mm,
):
    original_slr_with_qbx = streamwarp_module.slr_with_qbx

    def slr_with_qbx_length_gate(static_bundle, moving_bundle, *args, **kwargs):
        if slr_min_length_mm is not None:
            kwargs["greater_than"] = slr_min_length_mm
        if slr_max_length_mm is not None:
            kwargs["less_than"] = slr_max_length_mm
        return original_slr_with_qbx(static_bundle, moving_bundle, *args, **kwargs)

    streamwarp_module.slr_with_qbx = slr_with_qbx_length_gate
    try:
        return streamwarp_module.bundlewarp(
            static,
            moving,
            alpha=alpha,
            beta=beta,
            max_iter=max_iter,
        )
    finally:
        streamwarp_module.slr_with_qbx = original_slr_with_qbx


def streamline_length_stats_for_summary(streamlines, prefix):
    lengths = segment_lengths(streamlines)
    finite = lengths[np.isfinite(lengths)]
    if finite.size == 0:
        return {
            f"{prefix}_n": int(len(lengths)),
            f"{prefix}_length_min_mm": np.nan,
            f"{prefix}_length_median_mm": np.nan,
            f"{prefix}_length_max_mm": np.nan,
        }
    return {
        f"{prefix}_n": int(len(lengths)),
        f"{prefix}_length_min_mm": float(np.min(finite)),
        f"{prefix}_length_median_mm": float(np.median(finite)),
        f"{prefix}_length_max_mm": float(np.max(finite)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--younger-session", default="ses-V02")
    parser.add_argument("--older-session", default="ses-V03")
    parser.add_argument("--tract-root-name", default="DSI_tracts")
    parser.add_argument("--generated-mask-dir-name", default="generated_tract_masks")
    parser.add_argument("--tract-names", nargs="+", required=True)
    parser.add_argument("--hemis", nargs="+", default=["L", "R"], choices=["L", "R"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-mask-syn", action="store_true", help="Run binary-mask SyN branch.")
    parser.add_argument("--run-bundlewarp", action="store_true", help="Run BundleWarp streamline branch.")
    parser.add_argument("--mask-linear-model", choices=["none", "rigid", "affine"], default="rigid")
    parser.add_argument("--mask-syn-level-iters", nargs="+", type=int, default=[100, 50, 25])
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--jac-step-mm", type=float, default=0.5)
    parser.add_argument("--endpoint-start-distance-mm", type=float, default=20.0, help="Maximum distance from streamline start endpoint to start ROI mask for nodewise analysis.")
    parser.add_argument("--endpoint-end-distance-mm", type=float, default=20.0, help="Maximum distance from streamline end endpoint to end ROI mask for nodewise analysis.")
    parser.add_argument("--min-centerline-coverage-frac", type=float, default=0.50, help="Minimum fraction of the older-session centerline node range a streamline must cover for nodewise analysis.")
    parser.add_argument("--bundlewarp-alpha", type=float, default=0.5)
    parser.add_argument("--bundlewarp-beta", type=float, default=20.0)
    parser.add_argument("--bundlewarp-max-iter", type=int, default=15)
    parser.add_argument("--bundlewarp-slr-min-length-mm", type=float, default=50.0, help="Minimum streamline length used by BundleWarp's internal SLR initialization. This patches DIPY slr_with_qbx greater_than during BundleWarp.")
    parser.add_argument("--bundlewarp-slr-max-length-mm", type=float, default=250.0, help="Maximum streamline length used by BundleWarp's internal SLR initialization. This patches DIPY slr_with_qbx less_than during BundleWarp.")
    parser.add_argument("--stop-on-bundlewarp-error", action="store_true", help="Stop immediately if BundleWarp fails. By default failed BundleWarp tracts are recorded and the script continues.")
    parser.add_argument("--buan-thresholds", nargs="+", type=float, default=[3, 5, 7, 10])

    args = parser.parse_args()
    if not args.run_mask_syn and not args.run_bundlewarp:
        args.run_mask_syn = True
        args.run_bundlewarp = True
    if args.run_bundlewarp:
        ensure_pandas_for_bundlewarp()

    subject = normalize_subject_id(args.subject_id)
    parent_dir = Path(args.parent_dir)
    orientation_roi_table = load_orientation_roi_table(
        parent_dir=parent_dir,
        subject=subject,
        sessions=[args.younger_session, args.older_session],
        tract_names=args.tract_names,
        hemis=args.hemis,
        tract_root_name=args.tract_root_name,
    )
    out_dir = Path(args.out_dir) / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    centerline_dir = out_dir / "older_session_centerlines"
    filter_qc_dir = out_dir / "streamline_filter_qc"
    centerline_dir.mkdir(parents=True, exist_ok=True)
    filter_qc_dir.mkdir(parents=True, exist_ok=True)

    younger_trks_by_name = get_tract_paths(parent_dir, subject, args.younger_session, args.tract_names, args.tract_root_name, args.hemis)
    older_trks_by_name = get_tract_paths(parent_dir, subject, args.older_session, args.tract_names, args.tract_root_name, args.hemis)

    paired_younger_trks_by_name = {}
    for younger_key, younger_trk in sorted(younger_trks_by_name.items()):
        older_key = get_session_matched_tract_key(
            younger_key,
            args.younger_session,
            args.older_session,
        )
        if older_key not in older_trks_by_name:
            print(
                f"SKIP: missing older-session tract file for {older_key}; "
                f"younger tract {younger_key} will not be analyzed."
            )
            continue
        paired_younger_trks_by_name[younger_key] = younger_trk

    for older_key in sorted(older_trks_by_name):
        younger_key = get_session_matched_tract_key(
            older_key,
            args.older_session,
            args.younger_session,
        )
        if younger_key not in younger_trks_by_name:
            print(
                f"SKIP: missing younger-session tract file for {younger_key}; "
                f"older tract {older_key} will not be analyzed."
            )

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

    validate_orientation_rois_for_pairs(
        parent_dir=parent_dir,
        subject=subject,
        tract_root_name=args.tract_root_name,
        pair_keys=younger_trks_by_name.keys(),
        tract_names=args.tract_names,
        hemis=args.hemis,
        younger_session=args.younger_session,
        older_session=args.older_session,
    )

    generated_mask_root = out_dir / args.generated_mask_dir_name
    if args.run_mask_syn:
        younger_masks_by_name = generate_tract_masks_from_trks(
            younger_trks_by_name,
            generated_mask_root,
            args.younger_session,
        )
        older_masks_by_name = generate_tract_masks_from_trks(
            older_trks_by_name,
            generated_mask_root,
            args.older_session,
        )

    mask_syn_summary_rows = []
    bundlewarp_summary_rows = []
    for younger_key, younger_trk in younger_trks_by_name.items():
        older_key = get_session_matched_tract_key(
            younger_key,
            args.younger_session,
            args.older_session,
        )
        if older_key not in older_trks_by_name:
            print(
                f"SKIP: missing older-session tract file for {older_key}; "
                f"younger tract {younger_key} will not be analyzed."
            )
            continue

        print(f"Processing {younger_key} -> {older_key}")
        older_trk = older_trks_by_name[older_key]
        moving_streamlines = load_streamlines_same_rasmm(younger_trk, younger_key)
        static_streamlines = load_streamlines_same_rasmm(older_trk, older_key)
        tract_family, hemi = parse_tract_family_hemi(younger_key, args.tract_names, args.hemis)
        younger_landmarks = get_orientation_landmarks(orientation_roi_table, tract_family, hemi, args.younger_session)
        older_landmarks = get_orientation_landmarks(orientation_roi_table, tract_family, hemi, args.older_session)
        static_oriented_initial, n_static_initial_flipped = orient_streamlines_by_landmark_rois(
            static_streamlines,
            older_landmarks["start_centroid_rasmm"],
            older_landmarks["end_centroid_rasmm"],
        )
        n_static_before_filtering = len(static_oriented_initial)
        static_endpoint, static_endpoint_dropped = filter_streamlines_by_endpoint_masks(
            static_oriented_initial,
            older_landmarks,
            args.endpoint_start_distance_mm,
            args.endpoint_end_distance_mm,
        )
        pre_centerline_row_base = {
            "tract": younger_key,
            "younger_trk": str(younger_trk),
            "older_trk": str(older_trk),
            "n_streamlines_younger": int(len(moving_streamlines)),
            "n_streamlines_older": int(len(static_streamlines)),
            "n_older_streamlines_before_filtering": int(n_static_before_filtering),
            "n_older_streamlines_after_endpoint_filtering": int(len(static_endpoint)),
            "n_older_streamlines_dropped_endpoint_filter": int(len(static_endpoint_dropped)),
            "younger_orientation_start_roi": younger_landmarks["start_roi"],
            "younger_orientation_end_roi": younger_landmarks["end_roi"],
            "older_orientation_start_roi": older_landmarks["start_roi"],
            "older_orientation_end_roi": older_landmarks["end_roi"],
        }
        try:
            centerline, static_for_centerline, n_static_flipped, n_centerline_flipped = build_landmark_oriented_centerline(
                static_endpoint,
                older_landmarks["start_centroid_rasmm"],
                older_landmarks["end_centroid_rasmm"],
                args.n_nodes,
            )
        except Exception as exc:
            if not is_empty_centerline_error(exc):
                raise
            error_file = filter_qc_dir / f"{younger_key}_centerline_error.txt"
            error_file.write_text(traceback.format_exc())
            static_filter_qc_csv = filter_qc_dir / f"{younger_key}_older_filter_qc.csv"
            write_streamline_filter_qc(static_filter_qc_csv, static_endpoint_dropped)
            failed_base = dict(pre_centerline_row_base)
            failed_base.update({
                "older_filter_qc_csv": str(static_filter_qc_csv),
                "n_younger_streamlines_flipped_for_orientation": np.nan,
                "n_older_streamlines_flipped_for_centerline_orientation": int(n_static_initial_flipped),
                "n_centerline_flipped_for_orientation": np.nan,
                "node_assignment": "nearest_older_session_centerline_node",
                "older_centerline_trk": "",
            })
            append_centerline_failure_rows(
                mask_syn_summary_rows,
                bundlewarp_summary_rows,
                failed_base,
                args.run_mask_syn,
                args.run_bundlewarp,
                exc,
                error_file,
            )
            print(
                f"WARNING: Cannot build older-session centerline for {younger_key}; "
                f"{len(static_endpoint)} older streamlines remain after endpoint filtering. "
                f"Skipping this tract/hemi and continuing. See {error_file}"
            )
            continue
        static_oriented, static_coverage_dropped, static_coverage_values = filter_streamlines_by_centerline_coverage(
            static_for_centerline,
            centerline,
            args.n_nodes,
            args.min_centerline_coverage_frac,
        )
        static_filter_summary = streamline_filter_summary(
            n_static_before_filtering,
            static_endpoint_dropped,
            static_coverage_dropped,
            static_coverage_values,
        )
        static_filter_qc_csv = filter_qc_dir / f"{younger_key}_older_filter_qc.csv"
        write_streamline_filter_qc(static_filter_qc_csv, static_endpoint_dropped + static_coverage_dropped)
        centerline_path = save_streamlines_as_trk(
            [centerline],
            older_trk,
            centerline_dir / f"{younger_key}_older_session_centerline.trk",
        )
        moving_oriented, n_moving_flipped = orient_streamlines_by_landmark_rois(
            moving_streamlines,
            younger_landmarks["start_centroid_rasmm"],
            younger_landmarks["end_centroid_rasmm"],
        )
        n_moving_before_filtering = len(moving_oriented)
        moving_endpoint, moving_endpoint_dropped = filter_streamlines_by_endpoint_masks(
            moving_oriented,
            younger_landmarks,
            args.endpoint_start_distance_mm,
            args.endpoint_end_distance_mm,
        )
        moving_oriented = Streamlines(moving_endpoint)
        moving_endpoint_filter_summary = streamline_filter_summary(
            n_moving_before_filtering,
            moving_endpoint_dropped,
            [],
            [],
        )
        moving_filter_qc_csv = filter_qc_dir / f"{younger_key}_younger_endpoint_filter_qc.csv"
        write_streamline_filter_qc(moving_filter_qc_csv, moving_endpoint_dropped)
        moving = Streamlines(set_number_of_points(moving_oriented, args.n_nodes))
        static = Streamlines(set_number_of_points(static_oriented, args.n_nodes))

        row_base = {
            "tract": younger_key,
            "younger_trk": str(younger_trk),
            "older_trk": str(older_trk),
            "n_streamlines_younger": int(len(moving_streamlines)),
            "n_streamlines_older": int(len(static_streamlines)),
            "n_younger_streamlines_flipped_for_orientation": int(n_moving_flipped),
            "n_older_streamlines_flipped_for_centerline_orientation": int(n_static_initial_flipped + n_static_flipped),
            "n_centerline_flipped_for_orientation": int(n_centerline_flipped),
            "younger_endpoint_filter_qc_csv": str(moving_filter_qc_csv),
            "older_filter_qc_csv": str(static_filter_qc_csv),
            "node_assignment": "nearest_older_session_centerline_node",
            "older_centerline_trk": str(centerline_path),
            "younger_orientation_start_roi": younger_landmarks["start_roi"],
            "younger_orientation_end_roi": younger_landmarks["end_roi"],
            "older_orientation_start_roi": older_landmarks["start_roi"],
            "older_orientation_end_roi": older_landmarks["end_roi"],
        }
        row_base.update({f"younger_filter_{k}": v for k, v in moving_endpoint_filter_summary.items()})
        row_base.update({f"older_filter_{k}": v for k, v in static_filter_summary.items()})

        # Default filtered analysis input. Branches may apply additional centerline coverage after warping.
        moving_endpoint_filtered = Streamlines(moving_oriented)

        if args.run_mask_syn:
            if younger_key not in younger_masks_by_name:
                raise KeyError(f"Could not find younger mask for {younger_key}.")
            if older_key not in older_masks_by_name:
                raise KeyError(f"Could not find older mask for {older_key}.")

            branch_dir = out_dir / "mask_syn" / younger_key
            branch_dir.mkdir(parents=True, exist_ok=True)
            moving_mask, moving_affine = load_nifti(str(younger_masks_by_name[younger_key]))
            static_mask, static_affine = load_nifti(str(older_masks_by_name[older_key]))
            moving_mask = (np.squeeze(moving_mask) > 0).astype(np.float32)
            static_mask = (np.squeeze(static_mask) > 0).astype(np.float32)

            prealign = estimate_prealign(static_mask, static_affine, moving_mask, moving_affine, args.mask_linear_model)
            if prealign is not None:
                np.savetxt(branch_dir / "mask_prealign.txt", prealign)

            mapping = run_mask_syn(static_mask, moving_mask, static_affine, moving_affine, prealign, args.mask_syn_level_iters)
            warped_mask = mapping.transform(moving_mask)
            save_nifti(str(branch_dir / "younger_tract_mask_warped_to_older_mask.nii.gz"), warped_mask, static_affine)
            write_mapping(mapping, str(branch_dir / "younger_to_older_mask_syn_mapping.nii.gz"))

            mask_syn_unfiltered_streamlines = transform_streamlines_with_mapping(moving_streamlines, mapping)
            mask_syn_unfiltered_trk = save_streamlines_as_trk(
                mask_syn_unfiltered_streamlines,
                older_trk,
                branch_dir / f"{younger_key}_mask_syn_unfiltered_warped_to_{args.older_session}.trk",
            )
            mask_syn_filtered_input, mask_syn_coverage_dropped, mask_syn_coverage_values = filter_streamlines_by_centerline_coverage(
                moving_endpoint_filtered,
                centerline,
                args.n_nodes,
                args.min_centerline_coverage_frac,
                transform_func=mapping.transform_points_inverse,
            )
            mask_syn_filtered_streamlines = transform_streamlines_with_mapping(mask_syn_filtered_input, mapping)
            mask_syn_filtered_trk = save_streamlines_as_trk(
                mask_syn_filtered_streamlines,
                older_trk,
                branch_dir / f"{younger_key}_mask_syn_filtered_warped_to_{args.older_session}.trk",
            )
            warped_streamlines, node_profile, logj_profiles, disp_profiles = mask_syn_along_tract_profiles(
                mapping,
                moving_endpoint_filtered,
                centerline,
                args.n_nodes,
                args.jac_step_mm,
                args.min_centerline_coverage_frac,
            )
            warped_trk = mask_syn_filtered_trk
            write_node_profile_csv(branch_dir / f"{younger_key}_mask_syn_node_profile.csv", node_profile)
            np.save(branch_dir / f"{younger_key}_mask_syn_logJ_profiles.npy", logj_profiles)
            np.save(branch_dir / f"{younger_key}_mask_syn_displacement_profiles.npy", disp_profiles)

            scores = buan_scores(static, Streamlines(set_number_of_points(warped_streamlines, args.n_nodes)), args.buan_thresholds, prefix="buan_mask_syn")
            row = dict(row_base)
            row.update({
                "branch": "mask_syn",
                "warped_trk": str(warped_trk),
                "warped_trk_filtered": str(mask_syn_filtered_trk),
                "warped_trk_unfiltered": str(mask_syn_unfiltered_trk),
                "mean_logJ": float(np.nanmean(logj_profiles)),
                "median_logJ": float(np.nanmedian(logj_profiles)),
                "mean_displacement_mm": float(np.nanmean(disp_profiles)),
                "median_displacement_mm": float(np.nanmedian(disp_profiles)),
                "n_mask_syn_streamlines_before_centerline_coverage_filter": int(len(moving_oriented)),
                "n_mask_syn_streamlines_analyzed_after_centerline_coverage_filter": int(logj_profiles.shape[0]),
                "n_mask_syn_streamlines_dropped_centerline_coverage_filter": int(len(moving_oriented) - logj_profiles.shape[0]),
            })
            row.update(scores)
            mask_syn_summary_rows.append(row)

        if args.run_bundlewarp:
            branch_dir = out_dir / "bundlewarp" / younger_key
            branch_dir.mkdir(parents=True, exist_ok=True)
            try:
                deformed_bundle, moving_aligned, distances, match_pairs, warp_map = run_bundlewarp_with_slr_length_gate(
                    static,
                    moving,
                    alpha=args.bundlewarp_alpha,
                    beta=args.bundlewarp_beta,
                    max_iter=args.bundlewarp_max_iter,
                    slr_min_length_mm=args.bundlewarp_slr_min_length_mm,
                    slr_max_length_mm=args.bundlewarp_slr_max_length_mm,
                )
                deformed_bundle = Streamlines(set_number_of_points(deformed_bundle, args.n_nodes))
                moving_aligned = Streamlines(set_number_of_points(moving_aligned, args.n_nodes))

                linear_unfiltered_trk = save_streamlines_as_trk(moving_aligned, older_trk, branch_dir / f"{younger_key}_bundlewarp_linear_moved_unfiltered.trk")
                nonlinear_unfiltered_trk = save_streamlines_as_trk(deformed_bundle, older_trk, branch_dir / f"{younger_key}_bundlewarp_nonlinear_moved_unfiltered.trk")
                moving_aligned_filtered, bw_linear_coverage_dropped, bw_linear_coverage_values = filter_streamlines_by_centerline_coverage(
                    moving_aligned,
                    centerline,
                    args.n_nodes,
                    args.min_centerline_coverage_frac,
                )
                deformed_bundle_filtered, bw_nonlinear_coverage_dropped, bw_nonlinear_coverage_values = filter_streamlines_by_centerline_coverage(
                    deformed_bundle,
                    centerline,
                    args.n_nodes,
                    args.min_centerline_coverage_frac,
                )
                linear_filtered_trk = save_streamlines_as_trk(moving_aligned_filtered, older_trk, branch_dir / f"{younger_key}_bundlewarp_linear_moved_filtered.trk")
                nonlinear_filtered_trk = save_streamlines_as_trk(deformed_bundle_filtered, older_trk, branch_dir / f"{younger_key}_bundlewarp_nonlinear_moved_filtered.trk")
                linear_trk = linear_filtered_trk
                nonlinear_trk = nonlinear_filtered_trk
                np.save(branch_dir / "bundlewarp_distances.npy", distances)
                np.save(branch_dir / "bundlewarp_match_pairs.npy", match_pairs)
                np.save(branch_dir / "bundlewarp_warp_map.npy", warp_map, allow_pickle=True)

                offsets, directions, colors = bundlewarp_vector_filed(moving_aligned, deformed_bundle)
                np.save(branch_dir / "bundlewarp_vector_offsets.npy", offsets)
                np.save(branch_dir / "bundlewarp_vector_directions.npy", directions)
                np.save(branch_dir / "bundlewarp_vector_colors.npy", colors)

                try:
                    shape_profile, centroid_profile = bundlewarp_shape_analysis(moving_aligned, deformed_bundle, no_disks=10, plotting=False)
                    np.save(branch_dir / "bundlewarp_shape_profile.npy", shape_profile)
                    np.save(branch_dir / "bundlewarp_centroid_profile.npy", centroid_profile)
                except Exception as exc:
                    print(f"WARNING: bundlewarp_shape_analysis failed for {younger_key}: {exc}")

                node_profile, arrays, bw_summary = bundlewarp_deformation_profiles(
                    static,
                    moving_aligned,
                    deformed_bundle,
                    match_pairs,
                    centerline,
                    args.n_nodes,
                    args.min_centerline_coverage_frac,
                )
                write_node_profile_csv(branch_dir / f"{younger_key}_bundlewarp_node_profile.csv", node_profile)
                save_bundlewarp_arrays(branch_dir, younger_key, arrays)

                aligned_scores = buan_scores(static, moving_aligned, args.buan_thresholds, prefix="buan_aligned")
                deformed_scores = buan_scores(static, deformed_bundle, args.buan_thresholds, prefix="buan_deformed")
                improvement_scores = {}
                for threshold in args.buan_thresholds:
                    a_key = f"buan_aligned_threshold_{threshold}mm"
                    d_key = f"buan_deformed_threshold_{threshold}mm"
                    improvement_scores[f"buan_improvement_threshold_{threshold}mm"] = deformed_scores[d_key] - aligned_scores[a_key]

                row = dict(row_base)
                row.update({
                    "branch": "bundlewarp",
                    "status": "ok",
                    "linear_moved_trk": str(linear_trk),
                    "nonlinear_moved_trk": str(nonlinear_trk),
                    "linear_moved_trk_filtered": str(linear_filtered_trk),
                    "nonlinear_moved_trk_filtered": str(nonlinear_filtered_trk),
                    "linear_moved_trk_unfiltered": str(linear_unfiltered_trk),
                    "nonlinear_moved_trk_unfiltered": str(nonlinear_unfiltered_trk),
                    "mean_bundlewarp_mdf_distance": float(np.nanmean(distances)),
                    "median_bundlewarp_mdf_distance": float(np.nanmedian(distances)),
                    "bundlewarp_slr_min_length_mm": float(args.bundlewarp_slr_min_length_mm),
                    "bundlewarp_slr_max_length_mm": float(args.bundlewarp_slr_max_length_mm),
                })
                row.update(streamline_length_stats_for_summary(moving, "bundlewarp_moving_input"))
                row.update(streamline_length_stats_for_summary(static, "bundlewarp_static_input"))
                row.update(bw_summary)
                row.update(aligned_scores)
                row.update(deformed_scores)
                row.update(improvement_scores)
                bundlewarp_summary_rows.append(row)
            except Exception as exc:
                error_file = branch_dir / "bundlewarp_error.txt"
                error_file.write_text(traceback.format_exc())
                print(f"WARNING: BundleWarp failed for {younger_key}; continuing. Error: {exc}")
                row = dict(row_base)
                row.update({
                    "branch": "bundlewarp",
                    "status": "failed",
                    "failure_reason": str(exc),
                    "bundlewarp_error_log": str(error_file),
                    "bundlewarp_slr_min_length_mm": float(args.bundlewarp_slr_min_length_mm),
                    "bundlewarp_slr_max_length_mm": float(args.bundlewarp_slr_max_length_mm),
                })
                row.update(streamline_length_stats_for_summary(moving, "bundlewarp_moving_input"))
                row.update(streamline_length_stats_for_summary(static, "bundlewarp_static_input"))
                bundlewarp_summary_rows.append(row)
                if args.stop_on_bundlewarp_error:
                    raise

    mask_syn_summary_csv = out_dir / "mask_syn_registration_summary.csv"
    bundlewarp_summary_csv = out_dir / "bundlewarp_registration_summary.csv"

    write_summary_csv(mask_syn_summary_csv, mask_syn_summary_rows)
    write_summary_csv(bundlewarp_summary_csv, bundlewarp_summary_rows)

    print("Done.")
    print(f"Saved outputs to: {out_dir}")
    if mask_syn_summary_rows:
        print(f"Saved mask-SyN summary CSV to: {mask_syn_summary_csv}")
    if bundlewarp_summary_rows:
        print(f"Saved BundleWarp summary CSV to: {bundlewarp_summary_csv}")


if __name__ == "__main__":
    main()
