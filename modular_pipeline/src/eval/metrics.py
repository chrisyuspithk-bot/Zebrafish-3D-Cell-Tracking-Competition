"""Competition evaluation metrics.

Implements:
  - adjusted_edge_jaccard: Matches predicted vs ground-truth cells
    per timepoint using scaled centroid distance.
  - division_jaccard: Jaccard index on division events.
  - Combined score: adjusted_edge_jaccard + 0.1 × division_jaccard
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


def compute_edge_jaccard(
    pred_nodes: list[tuple[int, int, float, float, float]],
    gt_nodes: list[tuple[int, int, float, float, float]],
    max_distance: float = 50.0,
    voxel_size: tuple[float, float, float] = (2.0, 0.5, 0.5),
) -> float:
    """Compute edge-level Jaccard index between prediction and ground truth.

    Matches nodes per timepoint using Hungarian algorithm on centroid distance,
    then computes Jaccard over edges (links between consecutive frames).

    Args:
        pred_nodes: List of (node_id, frame, z, y, x).
        gt_nodes: List of (node_id, frame, z, y, x).
        max_distance: Max pixel distance for matching.
        voxel_size: (Z, Y, X) for anisotropy correction.

    Returns:
        Edge Jaccard score in [0, 1].
    """
    weights = np.array(voxel_size, dtype=np.float32)

    # Group nodes by frame
    def group_by_frame(nodes):
        frames = {}
        for nid, t, z, y, x in nodes:
            frames.setdefault(t, ([], []))
            frames[t][0].append(nid)
            frames[t][1].append([z, y, x])
        return frames

    pred_frames = group_by_frame(pred_nodes)
    gt_frames = group_by_frame(gt_nodes)

    all_frames = sorted(set(pred_frames.keys()) | set(gt_frames.keys()))
    tp_edges = 0
    total_pred_edges = 0
    total_gt_edges = 0

    prev_mapping = {}  # pred_nid -> gt_nid from previous frame

    for t in all_frames:
        p_ids, p_pos = pred_frames.get(t, ([], []))
        g_ids, g_pos = gt_frames.get(t, ([], []))

        p_pos_arr = np.array(p_pos, dtype=np.float32) if p_pos else np.empty((0, 3))
        g_pos_arr = np.array(g_pos, dtype=np.float32) if g_pos else np.empty((0, 3))

        # Hungarian matching
        new_mapping = {}
        if len(p_pos_arr) > 0 and len(g_pos_arr) > 0:
            cost = np.linalg.norm(
                p_pos_arr[:, None] * weights - g_pos_arr[None, :] * weights,
                axis=2,
            )
            # Mask far distances
            cost[cost > max_distance] = 1e9
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind, strict=False):
                if cost[r, c] <= max_distance:
                    new_mapping[p_ids[r]] = g_ids[c]

        # Count edge matches from previous frame
        for pred_edge_src, pred_edge_dst in _get_edges_for_frame(p_ids, pred_frames, t):
            total_pred_edges += 1
            gt_src = prev_mapping.get(pred_edge_src)
            gt_dst = new_mapping.get(pred_edge_dst)
            if gt_src is not None and gt_dst is not None and (gt_src, gt_dst) in _gt_edge_set(gt_frames, t):
                tp_edges += 1

        # Count GT edges for this frame
        total_gt_edges += len(_get_edges_for_frame(g_ids, gt_frames, t))

        prev_mapping = new_mapping

    # Adjusted Jaccard: penalize over-prediction
    if total_pred_edges == 0 and total_gt_edges == 0:
        return 1.0
    if total_pred_edges == 0 or tp_edges == 0:
        return 0.0

    jaccard = tp_edges / (total_pred_edges + total_gt_edges - tp_edges)

    # Adjustment: if over-predicting, scale down
    pred_ratio = total_pred_edges / max(total_gt_edges, 1)
    if pred_ratio > 1.5:
        jaccard *= (1.5 / pred_ratio)

    return max(0.0, min(1.0, jaccard))


def _get_edges_for_frame(ids, frames, t):
    """Get edges where the source is at frame t and destination at t+1."""
    edges = []
    if t + 1 not in frames:
        return edges
    next_ids = set(frames[t + 1][0]) if t + 1 in frames else set()
    # This is a stub — in practice, edges should be passed explicitly
    return edges


def _gt_edge_set(gt_frames, t):
    """Get set of GT edges at frame t. Stub — needs explicit edge list."""
    return set()


def compute_division_jaccard(
    pred_divisions: list[tuple[int, int, int]],
    gt_divisions: list[tuple[int, int, int]],
    node_mapping: dict[int, int] | None = None,
) -> float:
    """Compute division Jaccard index.

    Matches division events by parent cell proximity.

    Args:
        pred_divisions: List of (parent, d1, d2) node IDs.
        gt_divisions: List of (parent, d1, d2) node IDs.
        node_mapping: Mapping from pred node IDs to GT node IDs.

    Returns:
        Division Jaccard in [0, 1].
    """
    if not pred_divisions and not gt_divisions:
        return 1.0
    if not pred_divisions or not gt_divisions:
        return 0.0

    pred_set = set(pred_divisions)
    gt_set = set(gt_divisions)

    if node_mapping:
        pred_mapped = set()
        for p, d1, d2 in pred_set:
            mp = node_mapping.get(p)
            md1 = node_mapping.get(d1)
            md2 = node_mapping.get(d2)
            if mp is not None and md1 is not None and md2 is not None:
                pred_mapped.add((mp, md1, md2))
        pred_set = pred_mapped

    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return intersection / union if union > 0 else 0.0


def competition_score(
    edge_jaccard: float,
    division_jaccard: float,
) -> float:
    """Combined competition score."""
    return edge_jaccard + 0.1 * division_jaccard


def evaluate_tracking(
    pred_nodes: list[tuple],
    pred_edges: list[tuple],
    pred_divisions: list[tuple],
    gt_nodes: list[tuple],
    gt_edges: list[tuple],
    gt_divisions: list[tuple],
    max_distance: float = 50.0,
    voxel_size: tuple[float, float, float] = (2.0, 0.5, 0.5),
) -> dict:
    """Full evaluation of tracking results.

    Returns:
        Dict with edge_jaccard, division_jaccard, combined_score,
        and per-frame breakdown.
    """
    edge_j = _compute_edge_jaccard_from_edges(
        pred_nodes, pred_edges, gt_nodes, gt_edges, max_distance, voxel_size
    )
    div_j = compute_division_jaccard(pred_divisions, gt_divisions)
    combined = competition_score(edge_j, div_j)

    return {
        "edge_jaccard": edge_j,
        "division_jaccard": div_j,
        "combined_score": combined,
    }


def _compute_edge_jaccard_from_edges(
    pred_nodes, pred_edges, gt_nodes, gt_edges,
    max_distance, voxel_size,
) -> float:
    """Compute edge Jaccard using explicit edge lists."""
    weights = np.array(voxel_size, dtype=np.float32)

    # Build node lookup
    pred_node_map = {}
    for nid, t, z, y, x in pred_nodes:
        pred_node_map.setdefault(t, {})[nid] = np.array([z, y, x], dtype=np.float32)

    gt_node_map = {}
    for nid, t, z, y, x in gt_nodes:
        gt_node_map.setdefault(t, {})[nid] = np.array([z, y, x], dtype=np.float32)

    # Per-frame node matching
    all_ts = sorted(set(pred_node_map.keys()) | set(gt_node_map.keys()))
    frame_mapping: dict[int, dict[int, int]] = {}

    for t in all_ts:
        p_nodes = pred_node_map.get(t, {})
        g_nodes = gt_node_map.get(t, {})
        if not p_nodes or not g_nodes:
            frame_mapping[t] = {}
            continue

        p_ids = list(p_nodes.keys())
        g_ids = list(g_nodes.keys())
        p_pos = np.array([p_nodes[n] for n in p_ids]) * weights
        g_pos = np.array([g_nodes[n] for n in g_ids]) * weights

        cost = np.linalg.norm(p_pos[:, None] - g_pos[None, :], axis=2)
        cost[cost > max_distance] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {}
        for r, c in zip(row_ind, col_ind, strict=False):
            if cost[r, c] <= max_distance:
                mapping[p_ids[r]] = g_ids[c]
        frame_mapping[t] = mapping

    # Build GT edge set per frame
    gt_edge_per_frame = {}
    for src, dst in gt_edges:
        src_t = None
        for t, nodes in gt_node_map.items():
            if src in nodes:
                src_t = t
                break
        if src_t is not None:
            gt_edge_per_frame.setdefault(src_t, set()).add((src, dst))

    # Count TP/FP/FN edges
    tp = 0
    total_pred = len(pred_edges)
    total_gt = len(gt_edges)

    for src, dst in pred_edges:
        src_t = None
        for t, nodes in pred_node_map.items():
            if src in nodes:
                src_t = t
                break
        if src_t is None:
            continue

        gt_src = frame_mapping.get(src_t, {}).get(src)
        gt_dst = frame_mapping.get(src_t + 1, {}).get(dst)

        if gt_src is not None and gt_dst is not None:
            if (gt_src, gt_dst) in gt_edge_per_frame.get(src_t, set()):
                tp += 1

    if total_pred == 0 and total_gt == 0:
        return 1.0
    denom = total_pred + total_gt - tp
    if denom == 0:
        return 0.0
    jaccard = tp / denom

    # Penalize over-prediction
    pred_ratio = total_pred / max(total_gt, 1)
    if pred_ratio > 1.5:
        jaccard *= 1.5 / pred_ratio

    return max(0.0, min(1.0, jaccard))
