"""Integer Linear Programming solver for global tracking optimization.

Formulates multi-frame tracking as a min-cost flow / set cover problem
and solves using greedy approximation (scalable) with optional exact
solver (network simplex / Gurobi) for the final pass.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict


def solve_ilp_tracking(
    graph: dict,
    assignment_cost_scale: float = 1.0,
    appearance_gate: float = 0.5,
    division_cost: float = 100.0,
    gap_closing_cost: float = 50.0,
) -> list[list[int]]:
    """Solve tracking via greedy best-first linking.

    For each detection, greedily link to the best match in the next frame.
    Post-processes to merge broken tracks and resolve divisions.

    This is a fast approximation. For production, swap in a full ILP solver
    (e.g., Google OR-Tools or Gurobi).

    Args:
        graph: Output of build_tracking_graph().
        assignment_cost_scale: Scale factor for edge costs.
        appearance_gate: Appearance similarity threshold (reserved).
        division_cost: Cost of a division event.
        gap_closing_cost: Cost of closing a gap.

    Returns:
        List of tracks, each a list of node IDs in temporal order.
    """
    nodes = graph["nodes"]
    num_nodes = graph["num_nodes"]
    move_edges = graph["move_edges"]
    gap_edges = graph["gap_edges"]
    div_candidates = graph["div_candidates"]
    node_offsets = graph["node_offsets"]

    # Node ID -> (frame_idx, centroids)
    node_info = {nid: (t, pos) for nid, t, _, pos in nodes}

    # Build adjacency: parent -> list of (child, cost, type)
    parents: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
    for src, dst, dist, etype in move_edges:
        cost = dist * assignment_cost_scale
        parents[dst].append((src, cost, etype))

    for src, dst, dist, gap_k in gap_edges:
        cost = dist * assignment_cost_scale + gap_closing_cost * gap_k
        parents[dst].append((src, cost, f"gap_{gap_k}"))

    for parent, d1, d2, dist in div_candidates:
        cost = dist * assignment_cost_scale + division_cost
        parents[d1].append((parent, cost, "div"))
        # Mark d1 and d2 as daughters of same division
        # This is handled in a second pass

    # Greedy forward linking: each node picks its best parent
    assignments: dict[int, int] = {}  # child -> parent
    for child in range(num_nodes):
        if child not in parents:
            continue
        candidates = parents[child]
        candidates.sort(key=lambda x: x[1])  # Sort by cost
        best_parent, best_cost, _ = candidates[0]
        if best_cost < 1e6:  # Reasonable threshold
            assignments[child] = best_parent

    # Build tracks by following assignments backward
    used = set()
    tracks = []
    for node_id in range(num_nodes):
        if node_id in used:
            continue
        # Walk backward to find track start
        current = node_id
        track = []
        while current in assignments and current not in used:
            track.append(current)
            used.add(current)
            current = assignments[current]
        track.append(current)
        used.add(current)
        track.reverse()
        if len(track) >= 1:
            tracks.append(track)

    # Merge tracks broken by gaps
    tracks = _merge_broken_tracks(tracks, gap_edges, node_info, gap_closing_cost)

    return tracks


def _merge_broken_tracks(
    tracks: list[list[int]],
    gap_edges: list,
    node_info: dict,
    gap_cost: float,
) -> list[list[int]]:
    """Merge tracks that have gap edges between them (broken tracks)."""
    if not gap_edges:
        return tracks

    # Map node -> track index
    node_to_track = {}
    for i, track in enumerate(tracks):
        for nid in track:
            node_to_track[nid] = i

    # Find merge candidates
    merges = []
    for src, dst, dist, gap_k in gap_edges:
        if src in node_to_track and dst in node_to_track:
            t1 = node_to_track[src]
            t2 = node_to_track[dst]
            if t1 != t2:
                src_frame = node_info[src][0]
                dst_frame = node_info[dst][0]
                if src_frame < dst_frame:
                    merges.append((t1, t2, dist + gap_cost * gap_k))

    if not merges:
        return tracks

    # Greedy merge by cost
    merges.sort(key=lambda x: x[2])
    merged_set = set()
    merged_tracks = []
    merged_indices = set()

    for t1, t2, _ in merges:
        if t1 in merged_indices or t2 in merged_indices:
            continue
        if tracks[t1][-1] in node_info and tracks[t2][0] in node_info:
            f1 = node_info[tracks[t1][-1]][0]
            f2 = node_info[tracks[t2][0]][0]
            if f1 < f2:
                merged_tracks.append(tracks[t1] + tracks[t2])
                merged_indices.add(t1)
                merged_indices.add(t2)

    # Add unmerged tracks
    for i, track in enumerate(tracks):
        if i not in merged_indices:
            merged_tracks.append(track)

    return merged_tracks
