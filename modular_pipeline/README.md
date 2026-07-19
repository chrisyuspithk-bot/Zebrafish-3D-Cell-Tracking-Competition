# Modular Pipeline (Alternative)

An alternative pipeline architecture for the Biohub Cell Tracking competition, designed for extensibility and higher ceiling performance.

## Key Differences from Baseline (`kaggle_solution.py`)

| Aspect | Baseline (LAP) | Modular Pipeline |
|--------|---------------|-----------------|
| **Architecture** | Monolithic script | 7-module package |
| **Detection** | cc / DoG / Cellpose | Hybrid DoG + optional 3D UNet |
| **Tracking** | Greedy frame-to-frame Hungarian | Global graph optimization (ILP) + GNN scorer |
| **Division** | Inline in tracking loop | Dedicated module with geometric constraints |
| **Gap repair** | Frame buffer only | Centroid interpolation + track reconnection |
| **Output** | CSV only | CSV + .geff format |
| **Evaluation** | None bundled | Full competition metrics module |
| **Config** | Function args | YAML-driven with defaults |

## Architecture

```
modular_pipeline/
├── src/
│   ├── data/loader.py          # Chunked Zarr loader with sliding windows
│   ├── detection/
│   │   ├── dog_detector.py     # DoG blob detector (separable Gaussians)
│   │   ├── unet_detector.py    # 3D UNet for learned center heatmaps
│   │   └── hybrid.py           # Fusion via IoU-based NMS
│   ├── tracking/
│   │   ├── graph_builder.py    # Spatiotemporal graph (move/gap/division edges)
│   │   ├── ilp_solver.py       # Greedy ILP approximation + track merging
│   │   └── gnn_linker.py       # MLP edge scorer for learned association
│   ├── division/detector.py    # Mitosis detection: parent → 2 daughters
│   ├── postprocess/gap_repair.py  # Centroid interpolation + track reconnection
│   └── eval/
│       ├── geff_io.py          # .geff read/write
│       └── metrics.py          # Competition score: edge_jaccard + 0.1 × div_jaccard
├── configs/default.yaml         # All hyperparameters
└── scripts/run_pipeline.py      # End-to-end runner
```

## Usage

```bash
# Run from repo root
python -m modular_pipeline.scripts.run_pipeline \
  --data-dir /kaggle/input/biohub-cell-tracking-during-development/test \
  --output submission.geff

# With custom config
python -m modular_pipeline.scripts.run_pipeline \
  --config modular_pipeline/configs/default.yaml \
  --limit-frames 20
```

```python
from modular_pipeline.src.config import load_config
from modular_pipeline.scripts.run_pipeline import run_pipeline

run_pipeline(
    data_dir="/kaggle/input/biohub-cell-tracking-during-development/test",
    output_path="submission.geff",
)
```

## Dependencies

```
numpy scipy scikit-image scikit-learn zarr numba networkx h5py pydantic pyarrow torch pyyaml
```

## When to Use This Pipeline

- **Use the baseline** (`kaggle_solution.py`) for: speed, simplicity, proven 0.143+ score
- **Use this pipeline** for: experimenting with global optimization, UNet-based detection, gap repair, or when you need the .geff output format
