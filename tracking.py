"""
Cell tracking via Linear Assignment Problem (LAP) with motion prediction.

Links cell detections frame-to-frame using:
1. Velocity-based motion prediction (2-frame Kalman-style extrapolation)
2. Hungarian algorithm for optimal assignment
3. Birth/death handling for cells entering/leaving the field
4. Integrated division detection (1-to-2 associations)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def track_cells_lap(
    detections_by_frame: list,
    max_move: float = 15.0,
    division_threshold: float = 20.0,
    frame_buffer: int = 2,
    motion_weight: float = 0.5,
) -> tuple:
    """
    Track cells across frames using LAP with motion prediction.

    For each frame transition, builds a cost matrix between active tracks
    and new detections. Cost combines spatial distance with motion-compensated
    distance (where motion is estimated from the last two positions).

    Args:
        detections_by_frame: List of (N_i, 3) arrays per frame [z, y, x]
        max_move: Maximum distance for linking (voxels)
        division_threshold: Max distance for division candidate search
        frame_buffer: Frames a track persists without detection before termination
        motion_weight: Weight of motion-compensated distance vs raw distance (0-1)

    Returns:
        (nodes, edges) where nodes is list of {id, t, z, y, x} dicts
        and edges is list of {source, target} dicts
    """
    T = len(detections_by_frame)

    # Build global node list
    node_id = 1
    nodes = []
    frame_nodes = []  # frame_nodes[t] = list of node dicts

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

    # Active tracks: track_id -> {node, last_seen_t, positions_history}
    active = {}
    next_track_id = 0

    def _new_track(n, t):
        nonlocal next_track_id
        tid = next_track_id
        next_track_id += 1
        active[tid] = {"node": n, "last_seen": t, "history": [(t, np.array([n["z"], n["y"], n["x"]], dtype=np.float64))]}
        return tid

    def _predict_position(info):
        """Predict position at next frame using velocity from last two positions."""
        hist = info["history"]
        if len(hist) >= 2:
            t1, p1 = hist[-2]
            t2, p2 = hist[-1]
            dt = t2 - t1
            if dt > 0:
                velocity = (p2 - p1) / dt
                return p2 + velocity
        return np.array([info["node"]["z"], info["node"]["y"], info["node"]["x"]], dtype=np.float64)

    for t in range(T):
        cur_nodes = frame_nodes[t]
        cur_coords = np.array([[n["z"], n["y"], n["x"]] for n in cur_nodes], dtype=np.float64) if cur_nodes else np.zeros((0, 3))

        if t == 0:
            for n in cur_nodes:
                _new_track(n, t)
            continue

        if not active:
            for n in cur_nodes:
                _new_track(n, t)
            continue

        active_items = list(active.items())
        if not active_items:
            for n in cur_nodes:
                _new_track(n, t)
            continue

        # Build cost matrix using vectorized cdist
        n_active = len(active_items)
        n_cur = len(cur_nodes)

        active_last = np.array([[info["node"]["z"], info["node"]["y"], info["node"]["x"]]
                                for _, info in active_items], dtype=np.float64)
        active_pred = np.array([_predict_position(info) for _, info in active_items], dtype=np.float64)
        cur_pos_all = np.array([[n["z"], n["y"], n["x"]] for n in cur_nodes], dtype=np.float64)

        raw_dist = cdist(active_last, cur_pos_all, metric="euclidean")
        motion_dist = cdist(active_pred, cur_pos_all, metric="euclidean")
        cost = (1 - motion_weight) * raw_dist + motion_weight * motion_dist
        cost[cost >= max_move] = 1e9

        # Solve assignment
        if n_active > 0 and n_cur > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

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
            # Keep history bounded
            if len(info["history"]) > 5:
                info["history"] = info["history"][-5:]

            matched_active.add(r)
            matched_cur.add(c)

        # --- Division detection ---
        # For unmatched active tracks, check if they have two close detections
        for i, (tid, info) in enumerate(active_items):
            if i in matched_active:
                continue
            if t - info["last_seen"] > 1:
                continue  # already stale

            pred_pos = _predict_position(info)
            # Find close unmatched detections
            candidates = []
            for j, cn in enumerate(cur_nodes):
                if j in matched_cur:
                    continue
                cur_pos = np.array([cn["z"], cn["y"], cn["x"]], dtype=np.float64)
                dist = np.linalg.norm(cur_pos - pred_pos)
                if dist < division_threshold:
                    candidates.append((dist, j))

            if len(candidates) >= 2:
                # Division: link parent to two daughters
                candidates.sort()
                for _, j in candidates[:2]:
                    cn = cur_nodes[j]
                    edges.append({"source": info["node"]["id"], "target": cn["id"]})
                    if cn["id"] not in active:
                        _new_track(cn, t)
                    matched_cur.add(j)
                # Remove parent from active tracks
                del active[tid]
            elif len(candidates) == 1 and t - info["last_seen"] >= frame_buffer:
                del active[tid]

        # Cull stale tracks
        stale = [tid for tid, info in active.items() if t - info["last_seen"] >= frame_buffer and tid in active]
        for tid in stale:
            del active[tid]

        # Add unmatched current detections as new tracks
        for j in range(len(cur_nodes)):
            if j not in matched_cur:
                _new_track(cur_nodes[j], t)

    return nodes, edges
