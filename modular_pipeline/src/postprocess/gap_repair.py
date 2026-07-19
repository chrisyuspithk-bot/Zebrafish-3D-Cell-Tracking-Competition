"""Gap repair: bridge broken tracks by interpolating missing detections.

When tracking fails for a few frames (e.g., due to occlusion or dim signal),
this module reconstructs the missing links using centroid interpolation
and local re-detection.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def repair_gaps(
    tracks: list[list[int]],
    node_info: dict[int, tuple[int, np.ndarray]],
    max_gap: int = 4,
    max_displacement: float = 30.0,
    min_track_length: int = 3,
    voxel_size: tuple[float, float, float] = (2.0, 0.5, 0.5),
) -> list[list[int]]:
    """Repair broken tracks by interpolating across short gaps.

    If a track has a temporal gap (missing frames) within it, we can
    interpolate the centroid positions and add placeholder nodes.

    This also connects separate tracks that are likely the same cell
    with a brief detection gap.

    Args:
        tracks: List of tracks (each a list of node IDs).
        node_info: Map from node ID to (frame_idx, centroid).
        max_gap: Max frames to bridge.
        max_displacement: Max allowed displacement per gap frame.
        min_track_length: Min frames for a valid track.
        voxel_size: (Z, Y, X) for anisotropy correction.

    Returns:
        Repaired tracks list with interpolated nodes (negative IDs).
    """
    weights = np.array(voxel_size, dtype=np.float32)
    repaired = []
    next_virtual_id = -1

    for track in tracks:
        if len(track) < 2:
            if len(track) >= min_track_length:
                repaired.append(track)
            continue

        # Get frame sequence
        frames = [node_info[nid][0] for nid in track]
        positions = np.array([node_info[nid][1] for nid in track], dtype=np.float32)

        # Detect gaps
        new_track = [track[0]]
        for i in range(1, len(track)):
            gap = frames[i] - frames[i - 1]

            if gap == 1:
                new_track.append(track[i])
            elif 1 < gap <= max_gap:
                # Check if displacement is reasonable
                disp = np.linalg.norm((positions[i] - positions[i - 1]) * weights)
                if disp <= max_displacement * gap:
                    # Interpolate intermediate positions
                    prev_pos = positions[i - 1]
                    curr_pos = positions[i]
                    prev_frame = frames[i - 1]
                    curr_frame = frames[i]

                    for f in range(prev_frame + 1, curr_frame):
                        alpha = (f - prev_frame) / gap
                        interp_pos = prev_pos + alpha * (curr_pos - prev_pos)
                        # Virtual node with negative ID
                        new_track.append(next_virtual_id)
                        node_info[next_virtual_id] = (f, interp_pos)
                        next_virtual_id -= 1

                    new_track.append(track[i])
                else:
                    # Gap too large, start new track fragment
                    if len(new_track) >= min_track_length:
                        repaired.append(new_track)
                    new_track = [track[i]]
            else:
                # Large gap, split
                if len(new_track) >= min_track_length:
                    repaired.append(new_track)
                new_track = [track[i]]

        if len(new_track) >= min_track_length:
            repaired.append(new_track)

    # Connect separate tracks with small gaps
    repaired = _connect_tracks(repaired, node_info, max_gap, max_displacement, weights)

    return repaired


def _connect_tracks(
    tracks: list[list[int]],
    node_info: dict[int, tuple[int, np.ndarray]],
    max_gap: int,
    max_displacement: float,
    weights: np.ndarray,
) -> list[list[int]]:
    """Connect separate tracks that are likely the same cell."""
    if len(tracks) <= 1:
        return tracks

    # Build track endpoints
    endpoints = []
    for i, track in enumerate(tracks):
        first_frame = node_info[track[0]][0]
        last_frame = node_info[track[-1]][0]
        first_pos = node_info[track[0]][1] * weights
        last_pos = node_info[track[-1]][1] * weights
        endpoints.append((first_frame, last_frame, first_pos, last_pos, i))

    # Find connectable pairs: track A ends before track B starts
    connections = []
    for i in range(len(endpoints)):
        for j in range(len(endpoints)):
            if i == j:
                continue
            _, a_last, _, a_pos, _ = endpoints[i]
            b_first, _, b_pos, _, _ = endpoints[j]
            gap = b_first - a_last
            if 1 < gap <= max_gap:
                disp = np.linalg.norm(a_pos - b_pos)
                if disp <= max_displacement * gap:
                    connections.append((i, j, gap, disp))

    if not connections:
        return tracks

    # Greedy merge
    connections.sort(key=lambda x: x[2])  # Sort by gap size
    merged = set()
    result = []

    for i, j, gap, _ in connections:
        if i in merged or j in merged:
            continue
        merged.add(i)
        merged.add(j)
        result.append(tracks[i] + tracks[j])

    for i in range(len(tracks)):
        if i not in merged:
            result.append(tracks[i])

    return result
