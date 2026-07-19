""".geff (Generic Event-based File Format) reader and writer.

GEFF is a line-based format for storing tracking graphs:
  - Node lines: node <id> <t> <z> <y> <x>
  - Edge lines: edge <src_id> <dst_id>
  - Division lines: div <parent_id> <daughter1_id> <daughter2_id>

Supports standard format plus extensions for confidence scores.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def read_geff(path: str | Path) -> dict:
    """Read a .geff tracking file.

    Returns:
        Dict with 'nodes' (list of (id, t, z, y, x)), 'edges' (list of (src, dst)),
        and 'divisions' (list of (parent, d1, d2)).
    """
    nodes = []
    edges = []
    divisions = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue

            cmd = parts[0].lower()
            if cmd == "node":
                nid, t, z, y, x = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                nodes.append((nid, t, z, y, x))
            elif cmd == "edge":
                src, dst = int(parts[1]), int(parts[2])
                edges.append((src, dst))
            elif cmd == "div":
                parent, d1, d2 = int(parts[1]), int(parts[2]), int(parts[3])
                divisions.append((parent, d1, d2))

    return {"nodes": nodes, "edges": edges, "divisions": divisions}


def write_geff(
    path: str | Path,
    nodes: list,
    edges: list,
    divisions: list | None = None,
) -> None:
    """Write tracking results to a .geff file.

    Args:
        path: Output file path.
        nodes: List of (node_id, t, z, y, x) tuples.
        edges: List of (src_id, dst_id) tuples.
        divisions: Optional list of (parent_id, d1_id, d2_id) tuples.
    """
    with open(path, "w") as f:
        f.write("# Biohub Cell Tracking Results\n")
        f.write(f"# {len(nodes)} nodes, {len(edges)} edges\n")
        if divisions:
            f.write(f"# {len(divisions)} division events\n\n")

        for node in nodes:
            nid, t, z, y, x = node
            f.write(f"node {int(nid)} {int(t)} {float(z):.3f} {float(y):.3f} {float(x):.3f}\n")

        for edge in edges:
            src, dst = edge
            f.write(f"edge {int(src)} {int(dst)}\n")

        if divisions:
            for div in divisions:
                parent, d1, d2 = div
                f.write(f"div {int(parent)} {int(d1)} {int(d2)}\n")


def tracks_to_geff(
    tracks: list[list[int]],
    node_info: dict[int, tuple[int, np.ndarray]],
    divisions: list[tuple[int, int, int]] | None = None,
) -> tuple[list, list, list]:
    """Convert internal track representation to .geff format.

    Args:
        tracks: List of tracks, each a list of node IDs.
        node_info: Map from node ID to (frame_idx, centroid).
        divisions: Optional division events.

    Returns:
        (nodes, edges, divisions) tuples ready for write_geff.
    """
    node_list = [(nid, t, pos[0], pos[1], pos[2]) for nid, (t, pos) in node_info.items()]

    edge_list = []
    for track in tracks:
        for i in range(len(track) - 1):
            edge_list.append((track[i], track[i + 1]))

    div_list = list(divisions) if divisions else []

    return node_list, edge_list, div_list
