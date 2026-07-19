"""Tracking graph construction from per-frame detections.

Builds a spatiotemporal graph linking detections across frames,
encoding possible cell movements, divisions, and gaps.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_tracking_graph(
    detections: list[np.ndarray],
    max_linking_distance: float = 50.0,
    max_gap_frames: int = 4,
    voxel_size: tuple[float, float, float] = (2.0, 0.5, 0.5),
) -> dict:
    """Build a spatiotemporal graph for tracking.

    Graph encoding:
      - Nodes: (t, idx_within_frame)
      - Edges:
        * move edges: t -> t+1 link (weighted by distance + appearance)
        * division edges: t -> t+1 (one parent -> two daughters)
        * gap edges: t -> t+k for k in [2, max_gap_frames]
        * enter/exit: source->node and node->sink (birth/death cost)

    Args:
        detections: List of shape (N_t, 3) centroids per frame.
        max_linking_distance: Max pixel distance for candidate edges.
        max_gap_frames: Max temporal gap to consider for gap closing.
        voxel_size: (Z, Y, X) for anisotropy correction.

    Returns:
        Dict with 'nodes', 'move_edges', 'div_candidates', 'gap_edges',
        and metadata for ILP solver.
    """
    T = len(detections)
    weights = np.array(voxel_size, dtype=np.float32)

    # Flatten nodes: assign unique IDs
    node_to_frame = []
    node_offset = []
    offset = 0
    for t in range(T):
        n = len(detections[t])
        for j in range(n):
            node_to_frame.append((t, j, detections[t][j]))
        node_offset.append(offset)
        offset += n

    num_nodes = len(node_to_frame)
    node_offsets_arr = np.array(node_offset + [num_nodes], dtype=np.int32)

    # Compute positions in anisotropy-corrected space
    positions = np.array([d for _, _, d in node_to_frame], dtype=np.float32) * weights

    # Build move edges (t -> t+1)
    move_edges = []
    for t in range(T - 1):
        src_start = node_offsets_arr[t]
        src_end = node_offsets_arr[t + 1]
        dst_start = node_offsets_arr[t + 1]
        dst_end = node_offsets_arr[t + 2]

        if src_start == src_end or dst_start == dst_end:
            continue

        src_pos = positions[src_start:src_end]
        dst_pos = positions[dst_start:dst_end]

        tree = cKDTree(dst_pos)
        neighbors = tree.query_ball_point(src_pos, max_linking_distance)

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                dist = np.linalg.norm(src_pos[i] - dst_pos[j])
                move_edges.append((src_start + i, dst_start + j, dist, "move"))

    # Build gap edges (t -> t+k for k in [2, max_gap_frames])
    gap_edges = []
    for t in range(T - 2):
        for k in range(2, min(max_gap_frames + 1, T - t)):
            t_target = t + k
            src_start = node_offsets_arr[t]
            src_end = node_offsets_arr[t + 1]
            dst_start = node_offsets_arr[t_target]
            dst_end = node_offsets_arr[t_target + 1]

            if src_start == src_end or dst_start == dst_end:
                continue

            src_pos = positions[src_start:src_end]
            dst_pos = positions[dst_start:dst_end]

            tree = cKDTree(dst_pos)
            neighbors = tree.query_ball_point(src_pos, max_linking_distance * k)

            for i, nbrs in enumerate(neighbors):
                for j in nbrs:
                    dist = np.linalg.norm(src_pos[i] - dst_pos[j])
                    gap_edges.append((src_start + i, dst_start + j, dist, k))

    # Division candidates: pairs of nearby detections in frame t+1
    div_candidates = []
    for t in range(T - 1):
        dst_start = node_offsets_arr[t + 1]
        dst_end = node_offsets_arr[t + 2]
        n = dst_end - dst_start
        if n < 2:
            continue

        dst_pos = positions[dst_start:dst_end]
        tree = cKDTree(dst_pos)
        pairs = tree.query_pairs(30.0)

        src_start = node_offsets_arr[t]
        src_end = node_offsets_arr[t + 1]
        src_pos = positions[src_start:src_end]

        for d1, d2 in pairs:
            midpoint = (dst_pos[d1] + dst_pos[d2]) / 2
            tree_src = cKDTree(src_pos)
            nearby_src = tree_src.query_ball_point(midpoint, max_linking_distance)
            for parent_idx in nearby_src:
                div_candidates.append((
                    src_start + parent_idx,
                    dst_start + d1,
                    dst_start + d2,
                    np.linalg.norm(midpoint - src_pos[parent_idx]),
                ))

    return {
        "nodes": [(nid, t, j, pos) for nid, (t, j, pos) in enumerate(node_to_frame)],
        "num_nodes": num_nodes,
        "num_frames": T,
        "node_offsets": node_offsets_arr,
        "move_edges": move_edges,
        "gap_edges": gap_edges,
        "div_candidates": div_candidates,
        "positions": positions,
        "voxel_size": voxel_size,
    }
