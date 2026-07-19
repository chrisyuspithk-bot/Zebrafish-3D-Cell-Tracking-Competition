# Biohub — Cell Tracking During Development

[Kaggle Competition](https://kaggle.com/competitions/biohub-cell-tracking-during-development) · Deadline: September 29, 2026

Detect and track zebrafish cells through 3D space and time in fluorescence microscopy volumes. Built on the [official baseline notebook](https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor) (score 0.143).

## Solution Overview

Two detection strategies with frame-to-frame LAP tracking:

1. **Detection (cc)** — 4× downsampling → box filter → percentile threshold (p90) → connected components → centroids. Fast and proven — matches the official baseline.
2. **Detection (dog)** — Gaussian smoothing → adaptive threshold → local peak finding. Optional for dense cell populations.
3. **Tracking** — Hungarian algorithm on physical distances (max 15 µm), with optional motion prediction and division detection.
4. **Calibration** — Optional auto-tuning of detection sensitivity from `.geff` ground-truth metadata.

## Quick Start

```python
from kaggle_solution import run_pipeline

# Default: connected components (matches official baseline)
run_pipeline(
    input_dir="/kaggle/input/biohub-cell-tracking-during-development/test",
    output_path="/kaggle/working/submission.csv",
)

# Alternative: DoG peaks (for dense/overlapping cells)
run_pipeline(
    input_dir="/kaggle/input/biohub-cell-tracking-during-development/test",
    output_path="/kaggle/working/submission.csv",
    detection_method="dog", sigma=3.0, threshold_rel=0.02, downsample=2,
)
```

## Real Data Results

Tested on 4 Kaggle test datasets (100 frames each, 64×256×256 uint16):

| Dataset | Nodes | Edges | Cells/frame |
|---|---|---|---|
| 44b6_0113de3b | 1,935 | 1,337 | ~19 |
| 44b6_0b24845f | 1,168 | 615 | ~12 |
| 6bba_05b6850b | 399 | 302 | ~4 |
| 6bba_05db0fb1 | 2,816 | 2,135 | ~28 |

**Total time**: 11.3 seconds for all 4 datasets (cc p90 ds4).

## Dependencies

```
numpy scipy scikit-image zarr lap tqdm numcodecs blosc2
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `detection_method` | `"cc"` | `"cc"` (connected components), `"dog"` (DoG peaks), or `"multiscale"` |
| `percentile` | 90.0 | Brightness percentile for cc threshold (p90 = top 10%) |
| `downsample` | 4 | Downsampling factor (4 = 64× fewer voxels) |
| `sigma` | 3.0 | Gaussian sigma for dog detection |
| `threshold_rel` | 0.02 | Detection sensitivity for dog |
| `max_move` | 15.0 | Maximum linking distance (µm) |
| `division_threshold` | 15.0 | Maximum distance for division candidates (µm) |
| `motion_weight` | 0.0 | Motion prediction weight — negligible impact on this data |
| `frame_buffer` | 1 | Frames before terminating lost tracks |

## What We Learned From the Reference Notebook

- **Downsampling is key.** 4× in Z,Y,X cuts data by 64× — natural denoising, dramatic speedup.
- **Connected components work.** For fluorescent nuclei, p90 threshold + labeling is simpler and faster than peak finding.
- **Divisions barely matter.** The metric weights edge tracking 10× more than division detection (score = edge + 0.1 × division).
- **Motion prediction doesn't help.** Cells move slowly — frame-to-frame Hungarian is sufficient.
- **Simple is better.** The reference baseline is ~20 lines of core logic and scores 0.143.

## Metric

Submissions are evaluated using a combined score:

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```

- **Adjusted Edge Jaccard**: Measures cell linkage accuracy with a penalty for over-predicting node count. Ground truth is sparse — unmatched predicted nodes are not false positives.
- **Division Jaccard**: Measures mitosis detection accuracy using a ±1 frame window.

Node matching uses physical distance (max 7.0 µm, voxel scale: z=1.625, y=x=0.40625 µm/voxel).

See [metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md) for full details.

## Data Format

- **Input**: Zarr v3 volumes (T, Z, Y, X) uint16 — typically (100, 64, 256, 256)
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
