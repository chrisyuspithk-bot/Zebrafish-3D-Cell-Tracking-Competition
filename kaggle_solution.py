#!/usr/bin/env python3
"""
Biohub - Cell Tracking During Development
==========================================
Complete solution for 3D+time cell tracking in zebrafish embryo microscopy.

Pipeline:
1. Multi-scale 3D cell detection (percentile-adaptive thresholding)
2. LAP-based tracking with motion prediction
3. Integrated division detection
4. Submission generation

Usage (Kaggle notebook):
    from kaggle_solution import run_pipeline
    run_pipeline(
        input_dir="/kaggle/input/biohub-cell-tracking-during-development/test",
        output_path="/kaggle/working/submission.csv",
    )
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
import zarr
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_cells_cc(
    volume: np.ndarray,
    percentile: float = 90.0,
    filter_size: int = 3,
    downsample: int = 4,
) -> np.ndarray:
    """
    Detect cells via percentile threshold + connected components.

    This is the approach from the official getting-started notebook.
    Downsamples, applies a uniform (box) filter, thresholds at a
    brightness percentile, labels connected components, and returns
    the centroid of each component.

    Simple, fast, and effective for well-separated fluorescent nuclei.

    Args:
        volume: 3D array (Z, Y, X), typically uint16
        percentile: Brightness percentile for threshold (0-100).
                    p90 keeps top 10% brightest pixels.
        filter_size: Size of uniform (box) filter kernel
        downsample: Downsampling factor in Z, Y, X (1 = no downsampling)

    Returns:
        Array of shape (N, 3) with (z, y, x) integer voxel coordinates
    """
    from scipy.ndimage import uniform_filter, label as ndlabel

    vol = volume.astype(np.float32)

    # Downsample
    if downsample > 1:
        ds = downsample
        # Crop to multiple of downsample for clean slicing
        Z, Y, X = vol.shape
        z_crop = (Z // ds) * ds
        y_crop = (Y // ds) * ds
        x_crop = (X // ds) * ds
        vol = vol[:z_crop, :y_crop, :x_crop].reshape(
            z_crop // ds, ds, y_crop // ds, ds, x_crop // ds, ds
        ).mean(axis=(1, 3, 5))

    # Box filter
    smoothed = uniform_filter(vol, size=filter_size)

    # Percentile threshold
    threshold = np.percentile(smoothed, percentile)
    binary = smoothed > threshold

    # Connected components
    labeled, n_features = ndlabel(binary)

    if n_features == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Compute centroids
    centroids = np.zeros((n_features, 3), dtype=np.float32)
    for comp_id in range(1, n_features + 1):
        coords = np.argwhere(labeled == comp_id)
        centroids[comp_id - 1] = coords.mean(axis=0) * downsample

    return np.round(centroids).astype(np.int32)


def detect_cells_3d(
    volume: np.ndarray,
    sigma: float = 3.0,
    threshold_rel: float = 0.02,
    min_distance: int = 5,
    downsample: int = 1,
) -> np.ndarray:
    """
    Detect cells via Gaussian smoothing + local peak finding.

    Better for dense populations where cells overlap in brightness.

    Args:
        volume: 3D array (Z, Y, X), typically uint16
        sigma: Gaussian sigma for smoothing
        threshold_rel: Threshold as fraction of (peak - background) intensity
        min_distance: Minimum separation between detections (voxels)
        downsample: Downsampling factor (1 = no downsampling)

    Returns:
        Array of shape (N, 3) with (z, y, x) integer voxel coordinates
    """
    vol = volume.astype(np.float32)

    if downsample > 1:
        ds = downsample
        Z, Y, X = vol.shape
        z_crop = (Z // ds) * ds
        y_crop = (Y // ds) * ds
        x_crop = (X // ds) * ds
        vol = vol[:z_crop, :y_crop, :x_crop].reshape(
            z_crop // ds, ds, y_crop // ds, ds, x_crop // ds, ds
        ).mean(axis=(1, 3, 5))

    smoothed = gaussian_filter(vol, sigma=sigma)
    bg = np.percentile(smoothed, 50)
    peak = np.percentile(smoothed, 99.9)
    abs_thresh = bg + threshold_rel * (peak - bg)

    if abs_thresh <= bg or peak <= bg:
        return np.zeros((0, 3), dtype=np.int32)

    # Scale min_distance for downsampled space
    md = max(2, min_distance // downsample) if downsample > 1 else min_distance

    coords = peak_local_max(
        smoothed, min_distance=md,
        threshold_abs=abs_thresh, exclude_border=2,
    )
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Scale coordinates back to original resolution
    result = coords.astype(np.float32) * downsample
    return np.round(result).astype(np.int32)


def detect_cells_multiscale(
    volume: np.ndarray,
    sigmas: tuple = (2.5, 3.5, 5.0),
    threshold_rel: float = 0.02,
    min_distance: int = 5,
    downsample: int = 1,
) -> np.ndarray:
    """Multi-scale DoG detection: combines results from multiple smoothing scales."""
    all_coords = []
    for sigma in sigmas:
        coords = detect_cells_3d(
            volume, sigma=sigma, threshold_rel=threshold_rel,
            min_distance=min_distance, downsample=downsample,
        )
        if len(coords) > 0:
            all_coords.append(coords)

    if not all_coords:
        return np.zeros((0, 3), dtype=np.int32)

    merged = np.vstack(all_coords)
    if len(merged) > 1:
        merged = _deduplicate(merged, min_distance)
    return merged.astype(np.int32)


def detect_cells_cellpose(
    volume_3d: np.ndarray,
    model=None,
    diameter: float = 25.0,
    gpu: bool = False,
    min_size: int = 30,
) -> np.ndarray:
    """
    Detect cells using Cellpose instance segmentation on MIP projection.

    Parameters
    ----------
    volume_3d : np.ndarray, shape (Z, Y, X), float32
        3D frame.
    model : CellposeModel or None
        Pre-loaded Cellpose model. Loads 'nuclei' model if None.
    diameter : float
        Expected cell diameter in pixels (~25 for zebrafish nuclei).
    gpu : bool
        Use GPU for inference.
    min_size : int
        Minimum mask area to keep (filters noise).

    Returns
    -------
    centroids : np.ndarray, shape (N, 3), float32
        (z, y, x) centroids.
    """
    if model is None:
        from cellpose import models
        model = models.CellposeModel(gpu=gpu, model_type='nuclei')

    mip = volume_3d.max(axis=0)
    if mip.max() <= 1:
        return np.empty((0, 3), dtype=np.float32)

    # Normalize to 0-1 for Cellpose
    mip_norm = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
    masks, _, _ = model.eval(mip_norm, diameter=diameter, channels=[0, 0])

    centroids = []
    for label in np.unique(masks):
        if label == 0:
            continue
        region = masks == label
        if region.sum() < min_size:
            continue
        ys, xs = np.where(region)
        z_profile = volume_3d[:, region].mean(axis=1)
        z_center = float(np.argmax(z_profile))
        centroids.append((z_center, ys.mean(), xs.mean()))

    return np.array(centroids, dtype=np.float32) if centroids else np.empty((0, 3), dtype=np.float32)


def _deduplicate(coords: np.ndarray, min_dist: float) -> np.ndarray:
    keep = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if not keep[i]:
            continue
        dist = np.sqrt(np.sum((coords[i+1:] - coords[i]) ** 2, axis=1))
        keep[i+1:][dist < min_dist] = False
    return coords[keep]


# ──────────────────────────────────────────────────────────────────────────────
# Tracking
# ──────────────────────────────────────────────────────────────────────────────

def track_cells(
    detections_by_frame: list,
    max_move: float = 15.0,
    division_threshold: float = 20.0,
    frame_buffer: int = 2,
    motion_weight: float = 0.5,
    voxel_scale: tuple = (1.625, 0.40625, 0.40625),
) -> tuple:
    """
    Track cells frame-to-frame with LAP + motion prediction + division detection.

    Args:
        detections_by_frame: List of (N_i, 3) arrays per frame
        max_move: Max linking distance (microns if voxel_scale provided, else voxels)
        division_threshold: Max distance for division candidates (microns/voxels)
        frame_buffer: Frames before terminating a lost track
        motion_weight: Weight of motion-compensated vs raw distance [0-1]
        voxel_scale: Physical voxel size in (z, y, x) µm/voxel. Used to compute
                     physically-meaningful distances. Set to (1,1,1) to use voxels.

    Returns:
        (nodes, edges) where nodes are [{id, t, z, y, x}] and edges [{source, target}]
    """
    T = len(detections_by_frame)
    sz, sy, sx = voxel_scale

    def _phys_dist(a, b):
        """Physical Euclidean distance between two (z, y, x) voxel coordinates."""
        return np.sqrt(
            ((a[0] - b[0]) * sz) ** 2 +
            ((a[1] - b[1]) * sy) ** 2 +
            ((a[2] - b[2]) * sx) ** 2
        )

    def _phys_cdist(A, B):
        """Physical Euclidean distance matrix between two sets of (z, y, x) voxel coords."""
        dz = (A[:, 0:1] - B[:, 0].T) * sz
        dy = (A[:, 1:2] - B[:, 1].T) * sy
        dx = (A[:, 2:3] - B[:, 2].T) * sx
        return np.sqrt(dz**2 + dy**2 + dx**2)

    # Build nodes
    node_id = 1
    nodes = []
    frame_nodes = []
    for t in range(T):
        coords = detections_by_frame[t]
        fn = []
        for i in range(len(coords)):
            n = {"id": node_id, "t": t,
                 "z": int(coords[i, 0]), "y": int(coords[i, 1]), "x": int(coords[i, 2])}
            nodes.append(n)
            fn.append(n)
            node_id += 1
        frame_nodes.append(fn)

    edges = []

    # Active tracks: track_id -> {node, last_seen, history}
    active = {}
    next_tid = 0

    def _new_track(n, t_):
        nonlocal next_tid
        tid = next_tid
        next_tid += 1
        pos = np.array([n["z"], n["y"], n["x"]], dtype=np.float64)
        active[tid] = {"node": n, "last_seen": t_, "history": [(t_, pos)]}
        return tid

    def _predict(info):
        h = info["history"]
        if len(h) >= 2:
            t1, p1 = h[-2]
            t2, p2 = h[-1]
            dt = t2 - t1
            if dt > 0:
                return p2 + (p2 - p1) / dt
        return np.array([info["node"]["z"], info["node"]["y"], info["node"]["x"]], dtype=np.float64)

    for t in range(T):
        cur_nodes = frame_nodes[t]

        if t == 0:
            for n in cur_nodes:
                _new_track(n, t)
            continue

        if not active:
            for n in cur_nodes:
                _new_track(n, t)
            continue

        active_items = list(active.items())
        n_active = len(active_items)
        n_cur = len(cur_nodes)

        if n_cur == 0:
            stale = [tid for tid, info in active.items()
                     if t - info["last_seen"] >= frame_buffer]
            for tid in stale:
                del active[tid]
            continue

        # Build cost matrix with physical distances
        active_last = np.array([[info["node"]["z"], info["node"]["y"], info["node"]["x"]]
                                for _, info in active_items], dtype=np.float64)
        active_pred = np.array([_predict(info) for _, info in active_items], dtype=np.float64)
        cur_pos = np.array([[n["z"], n["y"], n["x"]] for n in cur_nodes], dtype=np.float64)

        raw_d = _phys_cdist(active_last, cur_pos)
        mot_d = _phys_cdist(active_pred, cur_pos)
        cost = (1 - motion_weight) * raw_d + motion_weight * mot_d
        cost[cost >= max_move] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_active = set()
        matched_cur = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e9:
                continue
            tid, info = active_items[r]
            cn = cur_nodes[c]
            edges.append({"source": info["node"]["id"], "target": cn["id"]})
            info["node"] = cn
            info["last_seen"] = t
            info["history"].append((t, np.array([cn["z"], cn["y"], cn["x"]], dtype=np.float64)))
            if len(info["history"]) > 5:
                info["history"] = info["history"][-5:]
            matched_active.add(r)
            matched_cur.add(c)

        # Division detection (conservative: only flag when parent was just seen
        # and two daughters appear close together near the predicted position)
        for i, (tid, info) in enumerate(active_items):
            if i in matched_active:
                continue
            if t - info["last_seen"] > 1:
                continue
            # Parent must have been tracked in the previous frame (not a new track)
            if len(info["history"]) < 2:
                continue

            pred_pos = _predict(info)
            candidates = []
            for j, cn in enumerate(cur_nodes):
                if j in matched_cur:
                    continue
                d = _phys_dist(np.array([cn["z"], cn["y"], cn["x"]], dtype=np.float64), pred_pos)
                if d < division_threshold:
                    candidates.append((d, j))

            if len(candidates) >= 2:
                candidates.sort()
                # Extra check: the two daughters should be close to each other
                c0 = np.array([cur_nodes[candidates[0][1]]["z"],
                               cur_nodes[candidates[0][1]]["y"],
                               cur_nodes[candidates[0][1]]["x"]], dtype=np.float64)
                c1 = np.array([cur_nodes[candidates[1][1]]["z"],
                               cur_nodes[candidates[1][1]]["y"],
                               cur_nodes[candidates[1][1]]["x"]], dtype=np.float64)
                daughter_dist = _phys_dist(c0, c1)
                if daughter_dist > division_threshold * 1.5:
                    continue  # Daughters too far apart, unlikely to be a real division

                for _, j in candidates[:2]:
                    cn = cur_nodes[j]
                    edges.append({"source": info["node"]["id"], "target": cn["id"]})
                    if not any(cn["id"] == a_info["node"]["id"] for _, a_info in active.items()):
                        _new_track(cn, t)
                    matched_cur.add(j)
                del active[tid]
            elif len(candidates) == 0 and t - info["last_seen"] >= frame_buffer:
                del active[tid]

        # Cull stale tracks
        stale = [tid for tid, info in active.items()
                 if t - info["last_seen"] >= frame_buffer]
        for tid in stale:
            if tid in active:
                del active[tid]

        # New tracks for unmatched detections
        for j in range(len(cur_nodes)):
            if j not in matched_cur:
                _new_track(cur_nodes[j], t)

    return nodes, edges


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_volume(zarr_path: str) -> np.ndarray:
    """Load a Zarr volume (v2 or v3). Returns (T, Z, Y, X) uint16 array."""
    try:
        store = zarr.storage.LocalStore(zarr_path, read_only=True)
        arr = zarr.open(store, path="0")
    except Exception:
        try:
            arr = zarr.open(zarr_path, path="0")
        except Exception:
            arr = zarr.open(zarr_path)
    return arr[:]


def load_geff_metadata(geff_path: str) -> dict:
    """
    Read metadata from a .geff ground-truth directory.

    Returns dict with keys like 'estimated_number_of_nodes'.
    The estimated_number_of_nodes is the T_true used in the
    adjusted edge Jaccard node-count penalty.
    """
    import json
    zarr_json = os.path.join(geff_path, "zarr.json")
    if os.path.exists(zarr_json):
        with open(zarr_json) as f:
            return json.load(f)
    # Try nodes/zarr.json or root
    for sub in ["nodes", ""]:
        p = os.path.join(geff_path, sub, "zarr.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(
    zarr_path: str,
    geff_path: str,
    sigma: float = 3.0,
    min_distance: int = 5,
    target_ratio: float = 1.0,
    use_multiscale: bool = False,
) -> float:
    """
    Find threshold_rel that produces approximately the right number of detections.

    Uses the estimated_number_of_nodes from the .geff metadata as the target.
    Searches threshold_rel values to match (within target_ratio).

    Args:
        zarr_path: Path to .zarr image volume
        geff_path: Path to .geff ground truth
        sigma: Gaussian sigma for detection
        min_distance: Min cell separation
        target_ratio: Ratio of predicted/expected nodes to target (1.0 = exact match)
        use_multiscale: Whether to use multi-scale detection

    Returns:
        Best threshold_rel value
    """
    meta = load_geff_metadata(geff_path)
    expected_nodes = meta.get("estimated_number_of_nodes", None)
    if expected_nodes is None:
        print("  WARNING: estimated_number_of_nodes not found in .geff, using default")
        return 0.02

    volume = load_volume(zarr_path)
    T = volume.shape[0]
    expected_per_frame = expected_nodes / T

    detect_fn = detect_cells_multiscale if use_multiscale else detect_cells_3d
    det_kwargs = {"min_distance": min_distance}
    if not use_multiscale:
        det_kwargs["sigma"] = sigma

    # Binary search for threshold_rel
    best_thresh = 0.02
    best_diff = float("inf")

    for thresh in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        total = 0
        for t in range(min(T, 10)):  # Sample first 10 frames
            frame = volume[t].astype(np.float32)
            spots = detect_fn(frame, threshold_rel=thresh, **det_kwargs)
            total += len(spots)
        avg_per_frame = total / min(T, 10)
        diff = abs(avg_per_frame - expected_per_frame * target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh

    print(f"  Calibrated: threshold_rel={best_thresh:.4f} "
          f"(expected {expected_per_frame:.0f} cells/frame)")
    return best_thresh


def process_dataset(
    zarr_path: str,
    detection_params: dict,
    tracking_params: dict,
    detection_method: str = "cc",
) -> dict:
    """
    Process a single .zarr dataset through the full pipeline.

    detection_method: "cc" (connected components), "dog" (DoG peaks), "multiscale", or "cellpose"
    """
    dataset_name = os.path.basename(zarr_path).replace(".zarr", "")

    volume = load_volume(zarr_path)
    T = volume.shape[0]

    # Select detection function and params
    if detection_method == "cc":
        detect_fn = detect_cells_cc
        det_params = {
            "percentile": detection_params.get("percentile", 90),
            "filter_size": detection_params.get("filter_size", 3),
            "downsample": detection_params.get("downsample", 4),
        }
    elif detection_method == "dog":
        detect_fn = detect_cells_3d
        det_params = {
            "sigma": detection_params.get("sigma", 3.0),
            "threshold_rel": detection_params.get("threshold_rel", 0.02),
            "min_distance": detection_params.get("min_distance", 5),
            "downsample": detection_params.get("downsample", 1),
        }
    elif detection_method == "multiscale":
        detect_fn = detect_cells_multiscale
        det_params = {
            "threshold_rel": detection_params.get("threshold_rel", 0.02),
            "min_distance": detection_params.get("min_distance", 5),
            "downsample": detection_params.get("downsample", 1),
        }
    elif detection_method == "cellpose":
        from cellpose import models
        gpu = detection_params.get("gpu", False)
        cp_model = models.CellposeModel(gpu=gpu, model_type='nuclei')
        detect_fn = detect_cells_cellpose
        det_params = {
            "model": cp_model,
            "diameter": detection_params.get("diameter", 25.0),
            "gpu": gpu,
            "min_size": detection_params.get("min_size", 30),
        }
    else:
        raise ValueError(f"Unknown detection_method: {detection_method}")

    detections_by_frame = []
    for t in range(T):
        frame = volume[t].astype(np.float32)
        spots = detect_fn(frame, **det_params)
        detections_by_frame.append(spots)

    nodes, edges = track_cells(detections_by_frame, **tracking_params)

    return {"dataset": dataset_name, "nodes": nodes, "edges": edges}


def build_submission(results: list, output_path: str):
    """Write results to submission.csv."""
    rows = []
    row_id = 0

    for result in results:
        dataset = result["dataset"]
        for node in result["nodes"]:
            rows.append({
                "id": row_id, "dataset": dataset, "row_type": "node",
                "node_id": node["id"], "t": node["t"], "z": node["z"],
                "y": node["y"], "x": node["x"],
                "source_id": -1, "target_id": -1,
            })
            row_id += 1

        for edge in result["edges"]:
            rows.append({
                "id": row_id, "dataset": dataset, "row_type": "edge",
                "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": edge["source"], "target_id": edge["target"],
            })
            row_id += 1

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
            "source_id", "target_id"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Submission: {output_path} ({len(rows)} rows)")


def run_pipeline(
    input_dir: str = "/kaggle/input/biohub-cell-tracking-during-development/test",
    output_path: str = "submission.csv",
    detection_method: str = "cc",
    sigma: float = 3.0,
    threshold_rel: float = 0.02,
    percentile: float = 90.0,
    downsample: int = 4,
    min_distance: int = 5,
    diameter: float = 25.0,
    min_size: int = 30,
    gpu: bool = False,
    max_move: float = 15.0,
    division_threshold: float = 15.0,
    motion_weight: float = 0.0,
    frame_buffer: int = 1,
    train_dir: str = None,
    target_ratio: float = 1.0,
) -> str:
    """
    Run the full cell tracking pipeline.

    If train_dir is provided, uses .geff files to auto-calibrate
    threshold_rel per dataset to match the estimated number of nodes.

    Args:
        input_dir: Directory containing .zarr test datasets
        output_path: Path for output submission.csv
        detection_method: "cc", "dog", "multiscale", or "cellpose"
        sigma: Gaussian sigma for DoG detection (voxels)
        threshold_rel: Detection threshold for DoG detection
        percentile: Brightness percentile for cc detection (p90 = top 10%)
        downsample: Downsampling factor (4 = 64x fewer voxels, recommended for cc)
        min_distance: Minimum distance between cells (voxels)
        diameter: Expected cell diameter for Cellpose (pixels)
        min_size: Minimum mask area for Cellpose detection
        gpu: Use GPU for Cellpose inference
        max_move: Maximum cell movement between frames (µm)
        division_threshold: Max distance for division detection (µm)
        motion_weight: Weight of motion-compensated distance [0-1]
        frame_buffer: Frames before terminating lost tracks
        train_dir: Optional path to training data with .zarr/.geff pairs for calibration
        target_ratio: Ratio of predicted/expected nodes (1.0 = exact, >1 = over-predict)

    Returns:
        Path to the generated submission file
    """
    input_path = Path(input_dir)
    zarr_files = sorted(input_path.glob("*.zarr"))

    if not zarr_files:
        # Try test/ subdirectory (Kaggle layout)
        zarr_files = sorted((input_path / "test").glob("*.zarr"))
    if not zarr_files:
        zarr_files = sorted(Path(".").glob("*.zarr"))

    if not zarr_files:
        print(f"ERROR: No .zarr files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(zarr_files)} datasets")
    if detection_method == "cellpose":
        print(f"Detection: method=cellpose, diameter={diameter}, min_size={min_size}, gpu={gpu}")
    else:
        print(f"Detection: method={detection_method}, downsample={downsample}, "
              f"percentile={percentile}, sigma={sigma}, thresh_rel={threshold_rel}")
    print(f"Tracking: max_move={max_move}, div_thresh={division_threshold}, "
          f"motion_weight={motion_weight}, frame_buffer={frame_buffer}")

    detection_params = {
        "sigma": sigma,
        "threshold_rel": threshold_rel,
        "percentile": percentile,
        "downsample": downsample,
        "min_distance": min_distance,
        "diameter": diameter,
        "min_size": min_size,
        "gpu": gpu,
    }
    tracking_params = {
        "max_move": max_move,
        "division_threshold": division_threshold,
        "motion_weight": motion_weight,
        "frame_buffer": frame_buffer,
    }

    # Build calibration map if train_dir provided (only for DoG methods)
    calibration = {}
    if train_dir and detection_method in ("dog", "multiscale"):
        train_path = Path(train_dir)
        for geff_path in sorted(train_path.glob("*.geff")):
            dataset_name = geff_path.name.replace(".geff", "")
            zarr_path = train_path / f"{dataset_name}.zarr"
            if zarr_path.exists():
                use_ms = (detection_method == "multiscale")
                calibration[dataset_name] = calibrate_threshold(
                    str(zarr_path), str(geff_path),
                    sigma=sigma, min_distance=min_distance,
                    target_ratio=target_ratio, use_multiscale=use_ms,
                )

    results = []
    t_start = time.time()

    for zarr_path in tqdm(zarr_files, desc="Processing"):
        dataset_name = os.path.basename(str(zarr_path)).replace(".zarr", "")

        det_params = detection_params.copy()
        if dataset_name in calibration:
            det_params["threshold_rel"] = calibration[dataset_name]
        elif calibration:
            det_params["threshold_rel"] = np.mean(list(calibration.values()))

        result = process_dataset(str(zarr_path), det_params,
                                 tracking_params, detection_method)
        results.append(result)
        print(f"  {result['dataset']}: {len(result['nodes'])} nodes, "
              f"{len(result['edges'])} edges")

    elapsed = time.time() - t_start
    total_nodes = sum(len(r["nodes"]) for r in results)
    total_edges = sum(len(r["edges"]) for r in results)
    print(f"\nDone: {len(results)} datasets, {total_nodes} total nodes, "
          f"{total_edges} total edges in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    build_submission(results, output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Main (for local testing)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cell Tracking Pipeline")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--method", type=str, default="cc",
                        choices=["cc", "dog", "multiscale", "cellpose"],
                        help="Detection method: cc, dog, multiscale, cellpose")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--threshold-rel", type=float, default=0.02)
    parser.add_argument("--percentile", type=float, default=90.0,
                        help="Percentile for cc threshold (p90 = top 10%%)")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Downsampling factor (4 recommended for cc)")
    parser.add_argument("--min-distance", type=int, default=5)
    parser.add_argument("--diameter", type=float, default=25.0,
                        help="Cell diameter in pixels for Cellpose")
    parser.add_argument("--min-size", type=int, default=30,
                        help="Minimum mask area for Cellpose")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for Cellpose inference")
    parser.add_argument("--max-move", type=float, default=15.0)
    parser.add_argument("--division-threshold", type=float, default=15.0)
    parser.add_argument("--motion-weight", type=float, default=0.0)
    parser.add_argument("--frame-buffer", type=int, default=1)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--target-ratio", type=float, default=1.0)
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_path=args.output,
        detection_method=args.method,
        sigma=args.sigma,
        threshold_rel=args.threshold_rel,
        percentile=args.percentile,
        downsample=args.downsample,
        min_distance=args.min_distance,
        diameter=args.diameter,
        min_size=args.min_size,
        gpu=args.gpu,
        max_move=args.max_move,
        division_threshold=args.division_threshold,
        motion_weight=args.motion_weight,
        frame_buffer=args.frame_buffer,
        train_dir=args.train_dir,
        target_ratio=args.target_ratio,
    )
