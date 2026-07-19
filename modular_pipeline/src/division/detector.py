"""Division (mitosis) detection module.

Identifies cell division events by detecting parent cells that split
into two daughter cells in subsequent frames. Uses spatial proximity
and geometric constraints.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def detect_divisions(
    detections: list[np.ndarray],
    tracks: list[list[int]],
    node_info: dict[int, tuple[int, np.ndarray]],
    min_cell_distance: float = 3.0,
    max_cell_distance: float = 30.0,
    temporal_window: int = 5,
    voxel_size: tuple[float, float, float] = (2.0, 0.5, 0.5),
) -> list[tuple[int, int, int]]:
    """Detect cell division events from tracks and detections.

    A division occurs when a parent track ends at time t, and two daughter
    tracks begin at time t+1 with centroids close to the parent's last position.

    Also detects divisions within a single track: when a track splits into
    two tracks at the same timepoint.

    Args:
        detections: Per-frame detection arrays.
        tracks: List of tracks, each a list of node IDs.
        node_info: Map from node ID to (frame_idx, centroid).
        min_cell_distance: Min centroid distance between daughter cells.
        max_cell_distance: Max centroid distance between daughter cells.
        temporal_window: Frames to search for division events.
        voxel_size: (Z, Y, X) for anisotropy correction.

    Returns:
        List of (parent_node_id, daughter1_node_id, daughter2_node_id).
    """
    weights = np.array(voxel_size, dtype=np.float32)

    # Build maps
    # For each frame, map node_id -> centroid
    frame_nodes: dict[int, list[tuple[int, np.ndarray]]] = {}
    for nid, (t, pos) in node_info.items():
        frame_nodes.setdefault(t, []).append((nid, pos))

    # Track start/end frames
    track_start: dict[int, int] = {}  # track_idx -> first frame
    track_end: dict[int, int] = {}    # track_idx -> last frame
    track_last_node: dict[int, int] = {}
    track_first_node: dict[int, int] = {}

    for i, track in enumerate(tracks):
        if not track:
            continue
        first_t = node_info[track[0]][0]
        last_t = node_info[track[-1]][0]
        track_start[i] = first_t
        track_end[i] = last_t
        track_first_node[i] = track[0]
        track_last_node[i] = track[-1]

    divisions = []

    # Strategy 1: Track ends and new tracks begin at t+1
    for p_idx, p_end in track_end.items():
        p_t = p_end
        p_node = track_last_node[p_idx]
        p_pos = node_info[p_node][1] * weights

        # Find tracks starting at t+1 or t+2
        candidates = []
        for c_idx, c_start in track_start.items():
            if c_idx == p_idx:
                continue
            if 1 <= c_start - p_t <= temporal_window:
                c_node = track_first_node[c_idx]
                c_pos = node_info[c_node][1] * weights
                dist = np.linalg.norm(p_pos - c_pos)
                if dist <= max_cell_distance:
                    candidates.append((c_idx, c_node, c_pos, dist))

        # Need exactly 2 daughters starting near the parent end
        if len(candidates) >= 2:
            # Sort by distance
            candidates.sort(key=lambda x: x[3])
            d1_idx, d1_node, d1_pos, _ = candidates[0]
            d2_idx, d2_node, d2_pos, _ = candidates[1]

            # Check daughter separation
            daughter_dist = np.linalg.norm(d1_pos - d2_pos)
            if min_cell_distance <= daughter_dist <= max_cell_distance:
                divisions.append((p_node, d1_node, d2_node))

    # Strategy 2: Single track branching at a timepoint
    # (Detect when two tracks share a parent at the same time)
    for t in range(len(detections) - 1):
        parents = {nid for nid, _ in frame_nodes.get(t, [])}
        children = {nid for nid, _ in frame_nodes.get(t + 1, [])}

        # Find parent tracks ending at t and children tracks starting at t+1
        end_tracks = {i for i, e in track_end.items() if e == t}
        start_tracks = {i for i, s in track_start.items() if s == t + 1}

        if len(end_tracks) < 1 or len(start_tracks) < 2:
            continue

        # For each parent, find the two nearest children
        for p_idx in end_tracks:
            p_node = track_last_node[p_idx]
            p_pos = node_info[p_node][1] * weights

            nearby = []
            for c_idx in start_tracks:
                if c_idx == p_idx:
                    continue
                c_node = track_first_node[c_idx]
                c_pos = node_info[c_node][1] * weights
                dist = np.linalg.norm(p_pos - c_pos)
                if dist <= max_cell_distance:
                    nearby.append((c_node, c_pos, dist))

            if len(nearby) >= 2:
                nearby.sort(key=lambda x: x[2])
                d1_node, d1_pos, _ = nearby[0]
                d2_node, d2_pos, _ = nearby[1]
                daughter_dist = np.linalg.norm(d1_pos - d2_pos)
                if min_cell_distance <= daughter_dist <= max_cell_distance:
                    div = (p_node, d1_node, d2_node)
                    if div not in divisions:
                        divisions.append(div)

    return divisions
