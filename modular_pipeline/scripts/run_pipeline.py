#!/usr/bin/env python3
"""End-to-end cell tracking pipeline for the Biohub Kaggle competition.

Usage:
    python scripts/run_pipeline.py --data-dir /path/to/zarr/data --output submission.geff
    python scripts/run_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from modular_pipeline.src.config import load_config
from modular_pipeline.src.data.loader import ZarrVolume, find_zarr_volumes, find_geff_files
from modular_pipeline.src.detection.hybrid import HybridDetector
from modular_pipeline.src.tracking.graph_builder import build_tracking_graph
from modular_pipeline.src.tracking.ilp_solver import solve_ilp_tracking
from modular_pipeline.src.division.detector import detect_divisions
from modular_pipeline.src.postprocess.gap_repair import repair_gaps
from modular_pipeline.src.eval.geff_io import write_geff, tracks_to_geff

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    data_dir: str | Path,
    output_path: str | Path,
    config: dict | None = None,
    limit_frames: int | None = None,
) -> dict:
    """Run the complete cell tracking pipeline.

    Args:
        data_dir: Directory containing .zarr volume(s).
        output_path: Path for the .geff output file.
        config: Configuration dict (uses defaults if None).
        limit_frames: Only process first N frames (for debugging).

    Returns:
        Dict with pipeline stats.
    """
    if config is None:
        config = load_config()

    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    voxel_size = tuple(config["data"]["voxel_size"])

    # 1. Find and load data
    logger.info("Step 1/6: Loading data...")
    zarr_paths = find_zarr_volumes(data_dir)
    if not zarr_paths:
        msg = f"No .zarr volumes found in {data_dir}"
        raise FileNotFoundError(msg)

    logger.info("Found %d Zarr volume(s): %s", len(zarr_paths), zarr_paths)
    volume = ZarrVolume(zarr_paths[0])
    T = min(volume.T, limit_frames) if limit_frames else volume.T
    logger.info("Volume shape: %s, processing %d frames", volume.shape, T)

    # 2. Detection
    logger.info("Step 2/6: Detecting cells...")
    detector = HybridDetector(
        iou_threshold=config["detection"]["fusion"]["iou_threshold"],
        min_confidence=config["detection"]["fusion"]["min_confidence"],
        voxel_size=voxel_size,
        dog_config=config["detection"]["dog"],
        unet_config=config["detection"]["unet"],
    )

    detections: list[np.ndarray] = []
    t0 = time.perf_counter()
    for t in range(T):
        frame = volume.read_timepoint(t)
        dets = detector.detect(frame)
        detections.append(dets)
        if (t + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("  Frame %d/%d: %d cells (%.1fs)", t + 1, T, len(dets), elapsed)
    detect_time = time.perf_counter() - t0
    total_cells = sum(len(d) for d in detections)
    logger.info("Detection complete: %d cells in %.1fs (avg %.0f/frame)",
                total_cells, detect_time, len(detections))

    # 3. Build tracking graph
    logger.info("Step 3/6: Building tracking graph...")
    t0 = time.perf_counter()
    graph = build_tracking_graph(
        detections,
        max_linking_distance=config["tracking"]["graph"]["max_linking_distance"],
        max_gap_frames=config["tracking"]["graph"]["max_gap_frames"],
        voxel_size=voxel_size,
    )
    graph_time = time.perf_counter() - t0
    logger.info("Graph built: %d nodes, %d move edges, %d gap edges, %d div candidates (%.1fs)",
                graph["num_nodes"], len(graph["move_edges"]),
                len(graph["gap_edges"]), len(graph["div_candidates"]), graph_time)

    # 4. Solve tracking (ILP + optional GNN)
    logger.info("Step 4/6: Solving tracking optimization...")
    t0 = time.perf_counter()

    if config["tracking"]["ilp"]["enabled"]:
        tracks = solve_ilp_tracking(
            graph,
            assignment_cost_scale=config["tracking"]["ilp"]["assignment_cost_scale"],
            appearance_gate=config["tracking"]["ilp"]["appearance_gate"],
            division_cost=config["tracking"]["ilp"]["division_cost"],
            gap_closing_cost=config["tracking"]["ilp"]["gap_closing_cost"],
        )
    else:
        # Use simple greedy linking
        tracks = _greedy_linking(graph)

    track_time = time.perf_counter() - t0
    logger.info("Tracking solved: %d tracks (%.1fs)", len(tracks), track_time)

    # 5. Division detection
    logger.info("Step 5/6: Detecting divisions...")
    t0 = time.perf_counter()
    node_info = {nid: (t, pos) for nid, t, _, pos in graph["nodes"]}

    if config["division"]["enabled"]:
        divisions = detect_divisions(
            detections, tracks, node_info,
            min_cell_distance=config["division"]["min_cell_distance"],
            max_cell_distance=config["division"]["max_cell_distance"],
            temporal_window=config["division"]["temporal_window"],
            voxel_size=voxel_size,
        )
    else:
        divisions = []
    div_time = time.perf_counter() - t0
    logger.info("Detected %d division events (%.1fs)", len(divisions), div_time)

    # 6. Postprocess: gap repair
    logger.info("Step 6/6: Postprocessing tracks...")
    t0 = time.perf_counter()
    if config["postprocess"]["gap_repair"]["enabled"]:
        tracks = repair_gaps(
            tracks, node_info,
            max_gap=config["postprocess"]["gap_repair"]["max_gap"],
            max_displacement=config["postprocess"]["gap_repair"]["max_displacement"],
            min_track_length=config["postprocess"]["gap_repair"]["min_track_length"],
            voxel_size=voxel_size,
        )
    post_time = time.perf_counter() - t0
    logger.info("Postprocessing complete: %d tracks (%.1fs)", len(tracks), post_time)

    # Write output
    geo_nodes, geo_edges, geo_divs = tracks_to_geff(tracks, node_info, divisions)
    write_geff(output_path, geo_nodes, geo_edges, geo_divs)
    logger.info("Output written to %s", output_path)

    stats = {
        "frames": T,
        "detections": total_cells,
        "tracks": len(tracks),
        "divisions": len(divisions),
        "nodes": len(geo_nodes),
        "edges": len(geo_edges),
        "detect_time_s": detect_time,
        "graph_time_s": graph_time,
        "track_time_s": track_time,
        "div_time_s": div_time,
        "post_time_s": post_time,
        "total_time_s": detect_time + graph_time + track_time + div_time + post_time,
    }

    logger.info("Pipeline complete: %s", stats)
    return stats


def _greedy_linking(graph: dict) -> list[list[int]]:
    """Simple greedy nearest-neighbor linking (fallback)."""
    move_edges = graph["move_edges"]
    node_offsets = graph["node_offsets"]

    assignments = {}
    for src, dst, dist, _ in sorted(move_edges, key=lambda x: x[2]):
        if dst not in assignments and src not in assignments.values():
            assignments[dst] = src

    # Build tracks
    all_nodes = set(range(graph["num_nodes"]))
    used = set()
    tracks = []

    for nid in sorted(all_nodes):
        if nid in used:
            continue
        current = nid
        track = []
        while current in assignments and current not in used:
            track.append(current)
            used.add(current)
            current = assignments[current]
        track.append(current)
        used.add(current)
        track.reverse()
        if track:
            tracks.append(track)

    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Biohub Cell Tracking Pipeline")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/biohub-cell-tracking-during-development/train",
                        help="Directory with .zarr volumes")
    parser.add_argument("--output", type=str, default="submission.geff",
                        help="Output .geff file path")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--limit-frames", type=int, default=None,
                        help="Process only first N frames (debug)")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    run_pipeline(args.data_dir, args.output, config, args.limit_frames)


if __name__ == "__main__":
    main()
