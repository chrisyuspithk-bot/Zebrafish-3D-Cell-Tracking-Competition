# Biohub Cell Tracking Competition

## Project Overview
Kaggle competition: detect and track zebrafish cells through 3D+time microscopy data.
- Input: Zarr v3 volumes (T, Z, Y, X) uint16 — typically (100, 64, 256, 256)
- Output: submission.csv with nodes (detections) and edges (tracks)
- Metric: Combined Edge Jaccard + Division Jaccard (details in metrics.md)
- Deadline: September 29, 2026

## Metric Details (from metrics.md)
- **Sparse GT**: Only a subset of cells are annotated. Unmatched predicted nodes are NOT FPs.
- **Adjusted Edge Jaccard**: `max(0, jaccard * (1 - 0.1 * (T_pred - T_true) / T_true))`
  - T_true = `estimated_number_of_nodes` from .geff metadata
  - Over-predicting nodes by 10% costs ~1% on Jaccard → better to slightly over-detect
- **Edge FP**: Occurs when endpoint matches GT node connected to a DIFFERENT node
- **Division window**: Forks ±1 frame from GT split can match. Division weight = 0.1
- **Micro-averaging**: TP/FP/FN summed across all samples, then Jaccard computed
- **Final score**: `adjusted_edge_jaccard + 0.1 * division_jaccard`

## Key Files
- `kaggle_solution.py` — Self-contained solution (notebook-ready for Kaggle)
- `detection.py` — Modular detection module (reference)
- `tracking.py` — Modular tracking module (reference)

## Pipeline
1. **Detection**: Background subtraction → Gaussian smoothing → percentile-adaptive threshold → peak_local_max
2. **Tracking**: LAP frame-to-frame linking with:
   - Velocity-based motion prediction
   - Physical distance scaling (z=1.625, y=x=0.40625 µm/voxel)
   - Conservative division detection (parent must be tracked, daughters must be close together)
   - Birth/death handling with frame_buffer
3. **Calibration** (optional): Uses .geff `estimated_number_of_nodes` to auto-tune `threshold_rel`

## Key Parameters (need tuning for real data)
- `sigma=3.0` — Gaussian sigma for detection smoothing
- `threshold_rel=0.02` — Detection sensitivity (fraction of p99.9-p50 range). Use `train_dir` for auto-calibration
- `min_distance=5` — Minimum cell separation (voxels)
- `max_move=15.0` — Max linking distance (µm). Node matching in metric uses 7 µm
- `division_threshold=20.0` — Max distance for division candidates (µm)
- `motion_weight=0.5` — Motion prediction vs raw distance weight
- `frame_buffer=2` — Frames before terminating lost tracks
- `target_ratio=1.0` — Calibration target ratio. Use >1 to over-predict (recommended since unmatched nodes aren't FP)

## Physical Scale
Voxel dimensions: z=1.625, y=0.40625, x=0.40625 µm/voxel
Evaluation node matching: max 7.0 µm (physical distance)

## Competition Constraints
- CPU ≤ 12 hours
- GPU ≤ 12 hours
- No internet during runtime
- Public pre-trained models allowed
- Submission must be named submission.csv
