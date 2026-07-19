# Biohub — Cell Tracking During Development

[Kaggle Competition](https://kaggle.com/competitions/biohub-cell-tracking-during-development) · Deadline: September 29, 2026

Detect and track zebrafish cells through 3D space and time in fluorescence microscopy volumes. The goal is to build robust algorithms that handle dense cell populations, imaging noise, and cell divisions.

## Solution Overview

A classical computer-vision pipeline for 3D+time cell tracking:

1. **Detection** — Background subtraction → multi-scale Gaussian smoothing → percentile-adaptive threshold → local peak detection
2. **Tracking** — Linear Assignment Problem (Hungarian algorithm) with velocity-based motion prediction and physical distance scaling
3. **Division Detection** — Conservative parent→2-daughters linking integrated into the tracking loop
4. **Calibration** — Optional auto-tuning of detection sensitivity using `.geff` ground-truth metadata

## Quick Start

```python
from kaggle_solution import run_pipeline

# Basic usage
run_pipeline(
    input_dir="/kaggle/input/biohub-cell-tracking-during-development/test",
    output_path="/kaggle/working/submission.csv",
)

# With auto-calibration from training data
run_pipeline(
    input_dir="/kaggle/input/biohub-cell-tracking-during-development/test",
    output_path="/kaggle/working/submission.csv",
    train_dir="/kaggle/input/biohub-cell-tracking-during-development/train",
    target_ratio=1.05,   # slight over-predict (unmatched nodes aren't FP)
    use_multiscale=True,
)
```

## Dependencies

```
numpy scipy scikit-image zarr lap tqdm numcodecs
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `sigma` | 3.0 | Gaussian smoothing sigma (voxels) |
| `threshold_rel` | 0.02 | Detection sensitivity (auto-calibrated if `train_dir` set) |
| `min_distance` | 5 | Minimum cell separation (voxels) |
| `max_move` | 15.0 | Maximum linking distance (µm) |
| `division_threshold` | 20.0 | Maximum distance for division candidates (µm) |
| `motion_weight` | 0.5 | Weight of motion-compensated vs raw distance |
| `frame_buffer` | 2 | Frames before terminating lost tracks |
| `target_ratio` | 1.0 | Calibration target (>1 = over-predict, recommended) |

## Metric

Submissions are evaluated using a combined score:

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```

- **Adjusted Edge Jaccard**: Measures how well cells are linked across time, with a penalty for over-predicting total node count. Ground truth is sparse — unmatched predicted nodes are not false positives.
- **Division Jaccard**: Measures how well cell mitosis events are identified using a ±1 frame window.

Node matching uses physical distance (max 7.0 µm, voxel scale: z=1.625, y=x=0.40625 µm/voxel).

See [metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md) for full details.

## Data Format

- **Input**: Zarr volumes (T, Z, Y, X) uint16 — typically (100, 64, 256, 256)
- **Ground Truth**: `.geff` directories with sparse node/edge annotations
- **Submission**: CSV with node rows (detections) and edge rows (temporal links)

## Files

| File | Description |
|---|---|
| `kaggle_solution.py` | Self-contained pipeline (main deliverable for Kaggle) |
| `detection.py` | Modular detection module |
| `tracking.py` | Modular tracking module |
| `AGENTS.md` | Detailed project notes and metric analysis |

## Competition Constraints

- CPU ≤ 12 hours / GPU ≤ 12 hours
- No internet during runtime
- Public pre-trained models allowed
- Submission must be named `submission.csv`
